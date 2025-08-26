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
from geometry_msgs.msg import Point, PoseStamped
from bae_mppi.msg import ProcessedObstacles, MPPIState


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
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        
        print(f"[SensorProcessor] laser_range=({self.laser_min_range}, {self.laser_max_range})")
    
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
        
        # Convert to Cartesian coordinates
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)
        
        # Limit number of obstacles
        if len(x_coords) > self.max_obstacles:
            indices = np.linspace(0, len(x_coords)-1, self.max_obstacles, dtype=int)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
        
        # Create obstacle points
        obstacle_points = []
        for x, y in zip(x_coords, y_coords):
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            obstacle_points.append(point)
        
        # Create ProcessedObstacles message
        obstacles = ProcessedObstacles()
        obstacles.header = scan.header
        obstacles.obstacle_points = obstacle_points
        obstacles.ranges = valid_ranges.tolist()
        obstacles.angles = valid_angles.tolist()
        obstacles.min_distance = float(np.min(valid_ranges)) if len(valid_ranges) > 0 else float('inf')
        
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
        obstacles.ranges = []
        obstacles.angles = []
        obstacles.min_distance = float('inf')
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
        filtered_obstacles.ranges = []  # Could be computed but not needed for filtering
        filtered_obstacles.angles = []  # Could be computed but not needed for filtering
        filtered_obstacles.min_distance = float(np.min([np.sqrt((p.x - robot_x)**2 + (p.y - robot_y)**2) 
                                                        for p in filtered_points])) if filtered_points else float('inf')
        
        return filtered_obstacles