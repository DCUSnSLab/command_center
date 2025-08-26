#!/usr/bin/env python3
"""
Goal Tracking Critic for SMPPI
Tracks /goal_pose for navigation
"""

import torch
from typing import Optional, Any

from .base_critic import BaseCritic


class GoalCritic(BaseCritic):
    """
    Goal tracking critic for navigation
    """
    
    def __init__(self, params: dict):
        """Initialize goal critic"""
        super().__init__("GoalCritic", params)
        
        # Goal tracking parameters
        self.xy_goal_tolerance = params.get('xy_goal_tolerance', 0.25)
        self.yaw_goal_tolerance = params.get('yaw_goal_tolerance', 0.25)
        self.distance_scale = params.get('distance_scale', 1.0)
        self.angle_scale = params.get('angle_scale', 1.0)
        
        print(f"[GoalCritic] xy_tolerance={self.xy_goal_tolerance}, yaw_tolerance={self.yaw_goal_tolerance}")
    
    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                    robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                    obstacles: Optional[Any]) -> torch.Tensor:
        """
        Compute goal tracking cost
        
        Args:
            trajectories: [K, T+1, 3] trajectory states (x, y, theta)
            controls: [K, T, 2] control sequences
            robot_state: [5] current robot state
            goal_state: [3] goal state (x, y, theta) or None
            obstacles: Not used for goal critic
            
        Returns:
            costs: [K] goal costs for each trajectory
        """
        if not self.enabled or goal_state is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)
        
        batch_size = trajectories.shape[0]
        
        # Use final trajectory points for goal cost
        final_positions = trajectories[:, -1, :]  # [K, 3]
        
        # Goal position and orientation
        goal_pos = goal_state[:2]  # [2]
        goal_yaw = goal_state[2]   # scalar
        
        # Distance cost
        position_errors = final_positions[:, :2] - goal_pos  # [K, 2]
        distance_costs = torch.sum(position_errors ** 2, dim=1)  # [K]
        
        # Angle cost
        angle_errors = self.normalize_angle(final_positions[:, 2] - goal_yaw)  # [K]
        angle_costs = angle_errors ** 2  # [K]
        
        # Combined cost
        total_costs = (self.distance_scale * distance_costs + 
                      self.angle_scale * angle_costs)
        
        return self.apply_weight(total_costs)
    
    def is_goal_reached(self, current_pose: torch.Tensor, goal_state: torch.Tensor) -> bool:
        """
        Check if goal is reached
        
        Args:
            current_pose: [3] current pose (x, y, theta)
            goal_state: [3] goal state (x, y, theta)
            
        Returns:
            reached: True if goal is reached
        """
        if goal_state is None:
            return False
        
        # Distance check
        distance = torch.sqrt(torch.sum((current_pose[:2] - goal_state[:2]) ** 2))
        
        # Angle check
        angle_diff = abs(self.normalize_angle(current_pose[2] - goal_state[2]))
        
        return (distance < self.xy_goal_tolerance and 
                angle_diff < self.yaw_goal_tolerance)
    
    def compute_progress_cost(self, trajectories: torch.Tensor, 
                             goal_state: torch.Tensor) -> torch.Tensor:
        """
        Compute progress-based cost (reward getting closer to goal)
        
        Args:
            trajectories: [K, T+1, 3] trajectory states
            goal_state: [3] goal state
            
        Returns:
            costs: [K] progress costs (negative for progress)
        """
        if goal_state is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)
        
        # Initial distances
        initial_distances = torch.sqrt(torch.sum(
            (trajectories[:, 0, :2] - goal_state[:2]) ** 2, dim=1
        ))
        
        # Final distances  
        final_distances = torch.sqrt(torch.sum(
            (trajectories[:, -1, :2] - goal_state[:2]) ** 2, dim=1
        ))
        
        # Progress (negative cost for getting closer)
        progress = initial_distances - final_distances
        
        return -progress  # Negative for reward
    
    def update_parameters(self, params: dict):
        """Update goal critic parameters"""
        if 'xy_goal_tolerance' in params:
            self.xy_goal_tolerance = params['xy_goal_tolerance']
        if 'yaw_goal_tolerance' in params:  
            self.yaw_goal_tolerance = params['yaw_goal_tolerance']
        if 'distance_scale' in params:
            self.distance_scale = params['distance_scale']
        if 'angle_scale' in params:
            self.angle_scale = params['angle_scale']
            
        print(f"[GoalCritic] Parameters updated")