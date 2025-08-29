#!/usr/bin/env python3
"""
Obstacle Avoidance Critic for SMPPI
Processes /scan data for collision avoidance
"""

import torch
import numpy as np
from typing import Optional, Any

from .base_critic import BaseCritic
from ..utils.geometry import GeometryUtils, TorchGeometryUtils


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
        
        # Footprint parameters
        self.footprint = params.get('footprint', None)
        self.footprint_padding = params.get('footprint_padding', 0.0)
        self.use_polygon_collision = params.get('use_polygon_collision', False)
        
        # Convert footprint to polygon if provided
        if self.footprint is not None:
            self.base_footprint_polygon = GeometryUtils.footprint_to_polygon(self.footprint)
            print(f"[ObstacleCritic] Using polygon footprint: {len(self.footprint)//2} vertices, padding={self.footprint_padding}")
        else:
            self.base_footprint_polygon = None
            print(f"[ObstacleCritic] Using circular approximation: vehicle_radius={self.vehicle_radius}")
        
        print(f"[ObstacleCritic] safety_radius={self.safety_radius}, use_polygon={self.use_polygon_collision}")
    
    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                    robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                    obstacles: Optional[Any]) -> torch.Tensor:
        if not self.enabled or obstacles is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        # 장애물 좌표 텐서 변환
        obstacle_points = self.process_obstacles(obstacles)
        if obstacle_points is None or obstacle_points.shape[0] == 0:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        # Choose collision detection method
        if self.use_polygon_collision and self.base_footprint_polygon is not None:
            return self.compute_cost_polygon(trajectories, obstacle_points)
        else:
            return self.compute_cost_circular(trajectories, obstacle_points)
    
    def compute_cost_circular(self, trajectories: torch.Tensor, obstacle_points: torch.Tensor) -> torch.Tensor:
        """Original circular collision detection"""
        batch_size, time_steps = trajectories.shape[:2]

        # trajectories: [K,T+1,2], obstacles: [N,2]
        traj_xy = trajectories[:, :, :2]  # [K,T+1,2]
        obs_xy = obstacle_points.unsqueeze(0).unsqueeze(0)  # [1,1,N,2]
        traj_xy_exp = traj_xy.unsqueeze(2)  # [K,T+1,1,2]

        # 모든 trajectory × timestep × obstacle 거리 계산
        dists = torch.norm(traj_xy_exp - obs_xy, dim=-1)  # [K,T+1,N]
        min_dists, _ = torch.min(dists, dim=2)  # [K,T+1]

        # 충돌 여부 및 repulsion cost 벡터화
        collision_mask = min_dists < self.vehicle_radius
        repulsion_mask = (min_dists >= self.vehicle_radius) & (min_dists < self.safety_radius)

        costs = torch.zeros_like(min_dists)
        costs[collision_mask] = self.collision_cost
        costs[repulsion_mask] = self.repulsion_factor * (
            1.0 / min_dists[repulsion_mask] - 1.0 / self.safety_radius
        ) ** 2

        # Trajectory별 총합 비용 [K]
        total_costs = costs.sum(dim=1)
        return self.apply_weight(total_costs)
    
    def compute_cost_polygon(self, trajectories: torch.Tensor, obstacle_points: torch.Tensor) -> torch.Tensor:
        """Polygon-based collision detection"""
        batch_size, time_steps = trajectories.shape[:2]
        
        # Convert to numpy for geometry operations (CPU-based for now)
        traj_np = trajectories.detach().cpu().numpy()  # [K, T+1, 3]
        obs_np = obstacle_points.detach().cpu().numpy()  # [N, 2]
        
        # Initialize costs
        total_costs = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        
        # Process each trajectory
        for k in range(batch_size):
            trajectory = traj_np[k]  # [T+1, 3]
            trajectory_cost = 0.0
            
            # Check each point in trajectory
            for t in range(time_steps + 1):
                pose = trajectory[t]  # [x, y, theta]
                robot_pose = (pose[0], pose[1], pose[2])
                
                # Create robot footprint at this pose
                robot_footprint = GeometryUtils.create_robot_footprint_at_pose(
                    self.footprint, robot_pose, self.footprint_padding
                )
                
                # Check collision with each obstacle
                point_cost = 0.0
                for obs_point in obs_np:
                    # Distance from obstacle to robot footprint
                    distance = GeometryUtils.point_to_polygon_distance(obs_point, robot_footprint)
                    
                    # Apply cost based on distance
                    if distance < 0:  # Inside footprint (collision)
                        point_cost += self.collision_cost
                    elif distance < self.safety_radius:  # In repulsion zone
                        repulsion_cost = self.repulsion_factor * (
                            1.0 / max(distance, 1e-6) - 1.0 / self.safety_radius
                        ) ** 2
                        point_cost += repulsion_cost
                
                trajectory_cost += point_cost
            
            total_costs[k] = trajectory_cost
        
        return self.apply_weight(total_costs)
    
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