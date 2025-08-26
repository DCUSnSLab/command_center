#!/usr/bin/env python3
"""
Control Effort Critic for SMPPI
Penalizes excessive control inputs and control changes
"""

import torch
from typing import Optional, Any

from .base_critic import BaseCritic


class ControlCritic(BaseCritic):
    """
    Control effort and smoothness critic
    """
    
    def __init__(self, params: dict):
        """Initialize control critic"""
        super().__init__("ControlCritic", params)
        
        # Control effort parameters
        self.linear_cost_weight = params.get('linear_cost_weight', 1.0)
        self.angular_cost_weight = params.get('angular_cost_weight', 1.0)
        
        # Control change parameters (SMPPI smoothness)
        self.linear_change_weight = params.get('linear_change_weight', 1.0)
        self.angular_change_weight = params.get('angular_change_weight', 1.0)
        
        # Reference velocities (preferred speeds)
        self.preferred_linear_velocity = params.get('preferred_linear_velocity', 1.0)
        self.preferred_angular_velocity = params.get('preferred_angular_velocity', 0.0)
        
        print(f"[ControlCritic] linear_weight={self.linear_cost_weight}, angular_weight={self.angular_cost_weight}")
    
    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                    robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                    obstacles: Optional[Any]) -> torch.Tensor:
        """
        Compute control effort cost
        
        Args:
            trajectories: [K, T+1, 3] trajectory states
            controls: [K, T, 2] control sequences (v, w)
            robot_state: [5] current robot state
            goal_state: [3] goal state or None
            obstacles: Not used for control critic
            
        Returns:
            costs: [K] control costs for each trajectory
        """
        if not self.enabled:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)
        
        # Control effort cost
        effort_cost = self.compute_effort_cost(controls)
        
        # Control change cost (smoothness)
        change_cost = self.compute_change_cost(controls)
        
        # Combined cost
        total_costs = effort_cost + change_cost
        
        return self.apply_weight(total_costs)
    
    def compute_effort_cost(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Compute control effort cost
        
        Args:
            controls: [K, T, 2] control sequences
            
        Returns:
            costs: [K] effort costs
        """
        linear_velocities = controls[:, :, 0]   # [K, T]
        angular_velocities = controls[:, :, 1]  # [K, T]
        
        # Quadratic cost on control magnitudes
        linear_costs = self.linear_cost_weight * torch.sum(linear_velocities ** 2, dim=1)
        angular_costs = self.angular_cost_weight * torch.sum(angular_velocities ** 2, dim=1)
        
        return linear_costs + angular_costs
    
    def compute_change_cost(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Compute control change cost (SMPPI smoothness)
        
        Args:
            controls: [K, T, 2] control sequences
            
        Returns:
            costs: [K] change costs
        """
        if controls.shape[1] < 2:  # Need at least 2 time steps
            return torch.zeros(controls.shape[0], device=self.device, dtype=self.dtype)
        
        # Control differences
        control_diff = controls[:, 1:, :] - controls[:, :-1, :]  # [K, T-1, 2]
        
        linear_changes = control_diff[:, :, 0]   # [K, T-1]
        angular_changes = control_diff[:, :, 1]  # [K, T-1]
        
        # Quadratic cost on control changes
        linear_change_costs = self.linear_change_weight * torch.sum(linear_changes ** 2, dim=1)
        angular_change_costs = self.angular_change_weight * torch.sum(angular_changes ** 2, dim=1)
        
        return linear_change_costs + angular_change_costs
    
    def compute_preferred_velocity_cost(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Compute cost for deviating from preferred velocities
        
        Args:
            controls: [K, T, 2] control sequences
            
        Returns:
            costs: [K] preference costs
        """
        linear_deviations = controls[:, :, 0] - self.preferred_linear_velocity
        angular_deviations = controls[:, :, 1] - self.preferred_angular_velocity
        
        linear_costs = torch.sum(linear_deviations ** 2, dim=1)
        angular_costs = torch.sum(angular_deviations ** 2, dim=1)
        
        return linear_costs + angular_costs
    
    def compute_acceleration_cost(self, controls: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute acceleration-based cost
        
        Args:
            controls: [K, T, 2] control sequences
            dt: Time step size
            
        Returns:
            costs: [K] acceleration costs
        """
        if controls.shape[1] < 2:
            return torch.zeros(controls.shape[0], device=self.device, dtype=self.dtype)
        
        # Approximate acceleration
        velocity_diff = controls[:, 1:, :] - controls[:, :-1, :]  # [K, T-1, 2]
        accelerations = velocity_diff / dt  # [K, T-1, 2]
        
        # Quadratic cost on accelerations
        accel_costs = torch.sum(accelerations ** 2, dim=(1, 2))  # [K]
        
        return accel_costs
    
    def update_parameters(self, params: dict):
        """Update control critic parameters"""
        if 'linear_cost_weight' in params:
            self.linear_cost_weight = params['linear_cost_weight']
        if 'angular_cost_weight' in params:
            self.angular_cost_weight = params['angular_cost_weight']
        if 'linear_change_weight' in params:
            self.linear_change_weight = params['linear_change_weight']
        if 'angular_change_weight' in params:
            self.angular_change_weight = params['angular_change_weight']
            
        print(f"[ControlCritic] Parameters updated")