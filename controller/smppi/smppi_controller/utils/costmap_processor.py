#!/usr/bin/env python3
"""
Costmap Processor for SMPPI
Processes /costmap (OccupancyGrid) data for MPPI optimization
Replaces scan-based obstacle detection with costmap-based approach
"""

import numpy as np
import torch
from typing import Optional, Tuple
from transforms3d.euler import quat2euler

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point
from smppi.msg import ProcessedObstacles, MPPIState


class CostmapProcessor:
    """
    Process costmap data for SMPPI controller
    Replaces scan-based processing with grid-based approach
    """

    def __init__(self, params: dict):
        """Initialize costmap processor"""

        # Costmap processing parameters
        self.occupied_threshold = params.get('occupied_threshold', 50)  # Cost >= 50 is occupied
        self.max_obstacles = params.get('max_obstacles', 1000)
        self.downsample_factor = params.get('downsample_factor', 1)  # Sample every N cells

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32

        # Cache for latest costmap
        self.latest_costmap: Optional[OccupancyGrid] = None
        self.costmap_data: Optional[np.ndarray] = None  # 2D grid (height, width)
        self.costmap_resolution: float = 0.1
        self.costmap_origin: Tuple[float, float] = (0.0, 0.0)
        self.costmap_width: int = 0
        self.costmap_height: int = 0

        print(f"[CostmapProcessor] Initialized with occupied_threshold={self.occupied_threshold}")

    def process_costmap(self, costmap: OccupancyGrid) -> ProcessedObstacles:
        """
        Process OccupancyGrid message to extract obstacle points

        Args:
            costmap: OccupancyGrid message from /costmap topic

        Returns:
            obstacles: ProcessedObstacles message with occupied cell centers
        """
        # Cache costmap for later grid-based queries
        self.latest_costmap = costmap
        self.costmap_resolution = costmap.info.resolution
        self.costmap_origin = (costmap.info.origin.position.x, costmap.info.origin.position.y)
        self.costmap_width = costmap.info.width
        self.costmap_height = costmap.info.height

        # Reshape data to 2D grid (row-major)
        costmap_data = np.array(costmap.data, dtype=np.int8).reshape(
            (self.costmap_height, self.costmap_width))
        self.costmap_data = costmap_data

        # Find occupied cells (cost >= threshold)
        occupied_mask = costmap_data >= self.occupied_threshold

        # Get grid indices of occupied cells
        occupied_indices = np.argwhere(occupied_mask)  # Shape: (N, 2) - [row, col]

        if len(occupied_indices) == 0:
            return self.create_empty_obstacles()

        # Downsample if needed
        if self.downsample_factor > 1:
            indices = np.arange(0, len(occupied_indices), self.downsample_factor)
            occupied_indices = occupied_indices[indices]

        # Limit number of obstacles
        if len(occupied_indices) > self.max_obstacles:
            # Random sampling to preserve spatial distribution
            indices = np.random.choice(len(occupied_indices), self.max_obstacles, replace=False)
            occupied_indices = occupied_indices[indices]

        # Convert grid indices to world coordinates (cell centers)
        # occupied_indices is [row, col], need to convert to [x, y]
        obstacle_points = []
        for row, col in occupied_indices:
            # Grid to world: x = origin_x + (col + 0.5) * resolution
            #                y = origin_y + (row + 0.5) * resolution
            x = self.costmap_origin[0] + (col + 0.5) * self.costmap_resolution
            y = self.costmap_origin[1] + (row + 0.5) * self.costmap_resolution

            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            obstacle_points.append(point)

        # Create ProcessedObstacles message
        obstacles = ProcessedObstacles()
        obstacles.header = costmap.header
        obstacles.obstacle_points = obstacle_points
        obstacles.distances = [0.0] * len(obstacle_points)  # Not used in costmap mode
        obstacles.costs = [0.0] * len(obstacle_points)  # Will be computed by critic

        return obstacles

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
        (Same as SensorProcessor for consistency)

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

    def get_costmap_info(self) -> dict:
        """
        Get costmap metadata for grid-based collision checking

        Returns:
            dict with costmap metadata (resolution, origin, width, height, data)
        """
        if self.costmap_data is None:
            return None

        return {
            'resolution': self.costmap_resolution,
            'origin_x': self.costmap_origin[0],
            'origin_y': self.costmap_origin[1],
            'width': self.costmap_width,
            'height': self.costmap_height,
            'data': self.costmap_data
        }

    def world_to_grid(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert world coordinates to grid indices (vectorized)

        Args:
            x: X coordinates in world frame (numpy array)
            y: Y coordinates in world frame (numpy array)

        Returns:
            gx, gy: Grid indices (col, row)
        """
        gx = ((x - self.costmap_origin[0]) / self.costmap_resolution).astype(np.int32)
        gy = ((y - self.costmap_origin[1]) / self.costmap_resolution).astype(np.int32)
        return gx, gy

    def get_cost_at_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get costmap costs at world coordinates (vectorized)

        Args:
            x: X coordinates in world frame (numpy array)
            y: Y coordinates in world frame (numpy array)

        Returns:
            costs: Costmap costs at each point (255 for out-of-bounds)
        """
        if self.costmap_data is None:
            return np.full_like(x, 255, dtype=np.uint8)

        # Convert to grid coordinates
        gx, gy = self.world_to_grid(x, y)

        # Check bounds
        valid_mask = (gx >= 0) & (gx < self.costmap_width) & (gy >= 0) & (gy < self.costmap_height)

        # Initialize with out-of-bounds cost
        costs = np.full(len(x), 255, dtype=np.uint8)

        # Get costs for valid points
        costs[valid_mask] = self.costmap_data[gy[valid_mask], gx[valid_mask]]

        return costs
