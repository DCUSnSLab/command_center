#!/usr/bin/env python3
"""
Sensor Processor for SMPPI
Processes /scan and /odom data for MPPI optimization
"""

import numpy as np
import torch
from typing import Optional, List, Tuple
from transforms3d.euler import quat2euler

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from smppi.msg import ProcessedObstacles, MPPIState

# TF2 imports
import rclpy
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from .geometry import GeometryUtils


class SensorProcessor:
    """
    Process sensor data for SMPPI controller
    """
    
    def __init__(self, params: dict):
        """Initialize sensor processor"""
        # Laser parameters
        self.laser_min_range = params.get('laser_min_range', 0.1)
        self.laser_max_range = params.get('laser_max_range', 5.0)
        self.angle_filter_range = params.get('angle_filter_range', np.pi)  # Filter range in radians
        
        # Processing parameters
        self.downsample_factor = params.get('downsample_factor', 1)  # Skip every N points
        self.max_obstacles = params.get('max_obstacles', 1000)
        
        # Vehicle footprint parameters
        self.footprint = params.get('footprint', [0.0, 0.0])
        self.footprint_padding = params.get('footprint_padding', 0.0)
        self.use_footprint_filtering = params.get('use_footprint_filtering', False)
        
        # Convert footprint to polygon if provided
        if self.footprint is not None and self.use_footprint_filtering:
            self.base_footprint_polygon = GeometryUtils.footprint_to_polygon(self.footprint)
            print(f"[SensorProcessor] Using footprint filtering with {len(self.footprint)//2} vertices, padding={self.footprint_padding}")
        else:
            self.base_footprint_polygon = None
            print(f"[SensorProcessor] Footprint filtering disabled")
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        
        # TF buffer (will be set by parent node)
        self.tf_buffer: Optional[tf2_ros.Buffer] = None
        self.target_frame = 'odom'  # Target frame for obstacle coordinates
        
        print(f"[SensorProcessor] laser_range=({self.laser_min_range}, {self.laser_max_range})")
    
    def set_tf_buffer(self, tf_buffer: tf2_ros.Buffer):
        """Set TF buffer for coordinate transformations"""
        self.tf_buffer = tf_buffer
    
    def process_laser_scan(self, scan: LaserScan) -> ProcessedObstacles:
        """
        Process LaserScan message to extract obstacle points
        
        Args:
            scan: LaserScan message from /scan topic
            
        Returns:
            obstacles: ProcessedObstacles message
        """
        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
        
        # Filter valid ranges
        valid_mask = self.create_valid_mask(ranges, angles, scan)
        
        if not np.any(valid_mask):
            return self.create_empty_obstacles()
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        # Downsample if needed
        if self.downsample_factor > 1:
            indices = np.arange(0, len(valid_ranges), self.downsample_factor)
            valid_ranges = valid_ranges[indices]
            valid_angles = valid_angles[indices]
        
        # Convert to Cartesian coordinates in laser frame
        x_coords_laser = valid_ranges * np.cos(valid_angles)
        y_coords_laser = valid_ranges * np.sin(valid_angles)
        
        # Transform to odom frame
        x_coords_odom, y_coords_odom = self._transform_points_to_odom(
            x_coords_laser, y_coords_laser, scan.header)
        
        # Apply footprint filtering if enabled
        if self.use_footprint_filtering and self.base_footprint_polygon is not None:
            x_coords_filtered, y_coords_filtered = self._filter_obstacles_by_footprint(
                x_coords_odom, y_coords_odom, scan.header)
            x_coords = x_coords_filtered
            y_coords = y_coords_filtered
        else:
            # Use all odom coordinates
            x_coords = x_coords_odom
            y_coords = y_coords_odom
        
        # Limit number of obstacles
        if len(x_coords) > self.max_obstacles:
            indices = np.linspace(0, len(x_coords)-1, self.max_obstacles, dtype=int)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
        
        # Create obstacle points (now in odom frame)
        obstacle_points = []
        for x, y in zip(x_coords, y_coords):
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            obstacle_points.append(point)
        
        # Create ProcessedObstacles message with updated header
        obstacles = ProcessedObstacles()
        obstacles.header = scan.header
        # Update frame_id to odom since coordinates are now in odom frame
        obstacles.header.frame_id = 'odom'
        obstacles.obstacle_points = obstacle_points
        obstacles.distances = valid_ranges.tolist()
        obstacles.costs = [0.0] * len(obstacle_points)  # Initialize with zero costs
        
        return obstacles
    
    def create_valid_mask(self, ranges: np.ndarray, angles: np.ndarray, 
                         scan: LaserScan) -> np.ndarray:
        """
        Create mask for valid laser scan points
        
        Args:
            ranges: Array of range measurements
            angles: Array of angles
            scan: LaserScan message
            
        Returns:
            valid_mask: Boolean mask for valid points
        """
        # Basic range filtering
        range_valid = (ranges >= self.laser_min_range) & (ranges <= self.laser_max_range)
        range_valid = range_valid & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        range_valid = range_valid & np.isfinite(ranges)
        
        # Angle filtering (only forward-facing if needed)
        angle_valid = np.abs(angles) <= self.angle_filter_range
        
        return range_valid & angle_valid
    
    def create_empty_obstacles(self) -> ProcessedObstacles:
        """Create empty ProcessedObstacles message"""
        obstacles = ProcessedObstacles()
        obstacles.obstacle_points = []
        obstacles.distances = []
        obstacles.costs = []
        return obstacles
    
    def process_odometry(self, odom: Odometry) -> MPPIState:
        """
        Process Odometry message to extract robot state
        
        Args:
            odom: Odometry message from /odom topic
            
        Returns:
            state: MPPIState message
        """
        # Extract position
        position = odom.pose.pose.position
        x = position.x
        y = position.y
        z = position.z
        
        # Extract orientation (convert quaternion to yaw)
        orientation = odom.pose.pose.orientation
        yaw = self.quaternion_to_yaw(orientation)
        
        # Extract velocities
        linear_vel = odom.twist.twist.linear
        angular_vel = odom.twist.twist.angular
        
        v_x = linear_vel.x
        v_y = linear_vel.y
        w_z = angular_vel.z
        
        # Create MPPIState message
        state = MPPIState()
        state.header = odom.header
        state.pose = odom.pose.pose
        state.velocity = odom.twist.twist
        state.state_vector = [float(x), float(y), float(yaw)]
        
        return state
    
    def quaternion_to_yaw(self, quat) -> float:
        """
        Convert quaternion to yaw angle
        
        Args:
            quat: Quaternion message
            
        Returns:
            yaw: Yaw angle in radians
        """
        # Method 1: Direct calculation (faster)
        yaw = np.arctan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        )
        
        return yaw
    
    def state_to_tensor(self, state: MPPIState) -> torch.Tensor:
        """
        Convert MPPIState to tensor for optimization
        
        Args:
            state: MPPIState message
            
        Returns:
            state_tensor: [5] tensor (x, y, yaw, v_x, w_z)
        """
        x, y, yaw = state.state_vector
        v_x = state.velocity.linear.x
        w_z = state.velocity.angular.z
        
        return torch.tensor([x, y, yaw, v_x, w_z],
                           device=self.device, dtype=self.dtype)
    
    def obstacles_to_tensor(self, obstacles: ProcessedObstacles) -> Optional[torch.Tensor]:
        """
        Convert ProcessedObstacles to tensor
        
        Args:
            obstacles: ProcessedObstacles message
            
        Returns:
            obstacle_tensor: [N, 2] tensor of obstacle points or None
        """
        if len(obstacles.obstacle_points) == 0:
            return None
        
        points = []
        for point in obstacles.obstacle_points:
            points.append([point.x, point.y])
        
        return torch.tensor(points, device=self.device, dtype=self.dtype)
    
    def filter_obstacles_by_distance(self, obstacles: ProcessedObstacles, 
                                   robot_pos: Tuple[float, float], 
                                   max_distance: float) -> ProcessedObstacles:
        """
        Filter obstacles by distance from robot
        
        Args:
            obstacles: ProcessedObstacles message
            robot_pos: (x, y) robot position
            max_distance: Maximum distance to keep obstacles
            
        Returns:
            filtered_obstacles: Filtered ProcessedObstacles message
        """
        if len(obstacles.obstacle_points) == 0:
            return obstacles
        
        robot_x, robot_y = robot_pos
        filtered_points = []
        
        for point in obstacles.obstacle_points:
            distance = np.sqrt((point.x - robot_x)**2 + (point.y - robot_y)**2)
            if distance <= max_distance:
                filtered_points.append(point)
        
        filtered_obstacles = ProcessedObstacles()
        filtered_obstacles.header = obstacles.header
        filtered_obstacles.obstacle_points = filtered_points
        filtered_obstacles.distances = []  # Could be computed but not needed for filtering
        filtered_obstacles.costs = []  # Could be computed but not needed for filtering
        
        return filtered_obstacles
    
    def _transform_points_to_odom(self, x_coords_laser: np.ndarray, y_coords_laser: np.ndarray, 
                                 scan_header) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform laser points to odom frame
        
        Args:
            x_coords_laser: X coordinates in laser frame
            y_coords_laser: Y coordinates in laser frame
            scan_header: LaserScan header with frame_id and timestamp
            
        Returns:
            x_coords_odom, y_coords_odom: Coordinates in odom frame
        """
        if self.tf_buffer is None:
            print("[SensorProcessor] Warning: TF buffer not set, returning laser frame coordinates")
            return x_coords_laser, y_coords_laser
        
        try:
            # Get transform from laser frame to odom frame
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,  # target frame (odom)
                scan_header.frame_id,  # source frame (laser)
                scan_header.stamp,  # time
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Transform each point
            x_coords_odom = []
            y_coords_odom = []
            
            for x_laser, y_laser in zip(x_coords_laser, y_coords_laser):
                # Create PointStamped in laser frame
                point_laser = PointStamped()
                point_laser.header = scan_header
                point_laser.point.x = float(x_laser)
                point_laser.point.y = float(y_laser)
                point_laser.point.z = 0.0
                
                # Transform to odom frame
                try:
                    point_odom = tf2_geometry_msgs.do_transform_point(point_laser, transform)
                    x_coords_odom.append(point_odom.point.x)
                    y_coords_odom.append(point_odom.point.y)
                except Exception as e:
                    print(f"[SensorProcessor] Point transform error: {e}")
                    # Fallback to laser coordinates
                    x_coords_odom.append(x_laser)
                    y_coords_odom.append(y_laser)
            
            return np.array(x_coords_odom), np.array(y_coords_odom)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            print(f"[SensorProcessor] TF transform error: {e}")
            # Fallback to laser frame coordinates
            return x_coords_laser, y_coords_laser
    
    def _filter_obstacles_by_footprint(self, x_coords_odom: np.ndarray, y_coords_odom: np.ndarray, 
                                     scan_header) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter obstacles that are too close to robot footprint
        
        Args:
            x_coords_odom: X coordinates in odom frame
            y_coords_odom: Y coordinates in odom frame
            scan_header: LaserScan header for robot pose lookup
            
        Returns:
            x_coords_filtered, y_coords_filtered: Filtered coordinates
        """
        if self.tf_buffer is None or self.base_footprint_polygon is None:
            return x_coords_odom, y_coords_odom
        
        try:
            # Get robot pose in odom frame
            robot_transform = self.tf_buffer.lookup_transform(
                self.target_frame,  # target frame (odom)
                'base_link',        # robot base frame
                scan_header.stamp,  # time
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Extract robot pose
            robot_x = robot_transform.transform.translation.x
            robot_y = robot_transform.transform.translation.y
            robot_orientation = robot_transform.transform.rotation
            robot_yaw = self.quaternion_to_yaw(robot_orientation)
            robot_pose = (robot_x, robot_y, robot_yaw)
            
            # Create robot footprint at current pose
            robot_footprint = GeometryUtils.create_robot_footprint_at_pose(
                self.footprint, robot_pose, self.footprint_padding)
            
            # Filter obstacles
            filtered_x = []
            filtered_y = []
            
            for x, y in zip(x_coords_odom, y_coords_odom):
                obstacle_point = np.array([x, y])
                
                # Calculate distance to robot footprint
                distance = GeometryUtils.point_to_polygon_distance(obstacle_point, robot_footprint)
                
                # Keep obstacles that are outside the footprint (positive distance)
                # and not too close (additional safety margin could be added here)
                if distance > 0.05:  # Small safety margin to avoid numerical issues
                    filtered_x.append(x)
                    filtered_y.append(y)
            
            return np.array(filtered_x), np.array(filtered_y)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            print(f"[SensorProcessor] Robot pose lookup error for footprint filtering: {e}")
            # Fallback to no filtering
            return x_coords_odom, y_coords_odom