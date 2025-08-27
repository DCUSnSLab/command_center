#!/usr/bin/env python3
"""
Obstacle Avoidance Critic for SMPPI
Processes /scan data for collision avoidance
"""

import torch
import numpy as np
from typing import Optional, Any

from .base_critic import BaseCritic


class ObstacleCritic(BaseCritic):
    """
    Obstacle avoidance critic using laser scan data
    """
    
    def __init__(self, params: dict):
        """Initialize obstacle critic"""
        super().__init__("ObstacleCritic", params)
        
        # Obstacle parameters
        self.safety_radius = params.get('safety_radius', 0.5)
        self.max_range = params.get('max_range', 5.0)
        self.collision_cost = params.get('collision_cost', 1000.0)
        self.repulsion_factor = params.get('repulsion_factor', 2.0)
        
        # Vehicle parameters
        self.vehicle_radius = params.get('vehicle_radius', 0.3)
        
        print(f"[ObstacleCritic] safety_radius={self.safety_radius}, vehicle_radius={self.vehicle_radius}")
    
    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                    robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                    obstacles: Optional[Any]) -> torch.Tensor:
        """
        Compute obstacle avoidance cost
        
        Args:
            trajectories: [K, T+1, 3] trajectory states
            controls: [K, T, 2] control sequences  
            robot_state: [5] current robot state
            goal_state: [3] goal state or None
            obstacles: Processed obstacle data from /scan
            
        Returns:
            costs: [K] obstacle costs for each trajectory
        """
        if not self.enabled or obstacles is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)
        
        batch_size = trajectories.shape[0]
        time_steps = trajectories.shape[1]
        
        # Convert obstacles to tensor
        obstacle_points = self.process_obstacles(obstacles)
        
        if obstacle_points is None or obstacle_points.shape[0] == 0:
            return torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        
        # Compute costs for each trajectory
        costs = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        
        for k in range(batch_size):
            trajectory = trajectories[k, :, :2]  # [T+1, 2] positions only
            
            # Compute minimum distances to obstacles for each waypoint
            for t in range(time_steps):
                position = trajectory[t:t+1, :]  # [1, 2]
                
                # Distance to all obstacles
                distances = torch.sqrt(torch.sum(
                    (obstacle_points - position) ** 2, dim=1
                ))  # [n_obstacles]
                
                min_distance = torch.min(distances)
                
                # Collision check
                if min_distance < self.vehicle_radius:
                    costs[k] += self.collision_cost
                
                # Repulsion cost
                elif min_distance < self.safety_radius:
                    repulsion_cost = self.repulsion_factor * (
                        1.0 / min_distance - 1.0 / self.safety_radius
                    ) ** 2
                    costs[k] += repulsion_cost
        
        return self.apply_weight(costs)
    
    def process_obstacles(self, obstacles) -> Optional[torch.Tensor]:
        """
        Process obstacle data from sensor
        
        Args:
            obstacles: Obstacle data (could be LaserScan or processed points)
            
        Returns:
            obstacle_points: [N, 2] obstacle points or None
        """
        if hasattr(obstacles, 'obstacle_points') and obstacles.obstacle_points:
            # Processed obstacles with points
            points = []
            for point in obstacles.obstacle_points:
                points.append([point.x, point.y])
            
            if points:
                return torch.tensor(points, device=self.device, dtype=self.dtype)
        
        elif hasattr(obstacles, 'ranges') and hasattr(obstacles, 'angle_min'):
            # Raw LaserScan data
            return self.process_laser_scan(obstacles)
        
        return None
    
    def process_laser_scan(self, scan) -> Optional[torch.Tensor]:
        """
        Convert LaserScan to obstacle points
        
        Args:
            scan: LaserScan message
            
        Returns:
            obstacle_points: [N, 2] obstacle points
        """
        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
        
        # Filter valid ranges
        valid_mask = (ranges > scan.range_min) & (ranges < scan.range_max) & (ranges < self.max_range)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return None
        
        # Convert to Cartesian coordinates
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        points = np.column_stack([x, y])
        
        return torch.tensor(points, device=self.device, dtype=self.dtype)
    
    def update_parameters(self, params: dict):
        """Update obstacle critic parameters"""
        if 'safety_radius' in params:
            self.safety_radius = params['safety_radius']
        if 'collision_cost' in params:
            self.collision_cost = params['collision_cost']
        if 'repulsion_factor' in params:
            self.repulsion_factor = params['repulsion_factor']
            
        print(f"[ObstacleCritic] Parameters updated")