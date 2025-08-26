#!/usr/bin/env python3
"""
Ackermann Motion Model for SMPPI
Based on BAE MPPI's dynamics but simplified
"""

import torch
import math
from typing import Tuple


class AckermannModel:
    """
    Ackermann vehicle motion model
    """
    
    def __init__(self, params: dict):
        """Initialize Ackermann model"""
        self.wheelbase = params.get('wheelbase', 1.0)
        self.max_steering_angle = params.get('max_steering_angle', math.pi / 4)
        self.min_turning_radius = params.get('min_turning_radius', 0.5)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        
        print(f"[AckermannModel] wheelbase={self.wheelbase}, max_steering={self.max_steering_angle}")
    
    def forward(self, states: torch.Tensor, controls: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Forward integrate Ackermann dynamics
        
        Args:
            states: [K, 3] current states (x, y, theta)
            controls: [K, 2] control inputs (v, w)
            dt: Time step
            
        Returns:
            next_states: [K, 3] next states
        """
        batch_size = states.shape[0]
        
        # Current state
        x = states[:, 0]      # [K]
        y = states[:, 1]      # [K]  
        theta = states[:, 2]  # [K]
        
        # Control inputs
        v = controls[:, 0]    # [K] linear velocity
        w = controls[:, 1]    # [K] angular velocity
        
        # Convert angular velocity to steering angle
        # w = v * tan(delta) / L  =>  delta = atan(w * L / v)
        steering_angles = self.angular_to_steering(v, w)
        
        # Ackermann kinematics
        x_next = x + v * torch.cos(theta) * dt
        y_next = y + v * torch.sin(theta) * dt
        theta_next = theta + w * dt
        
        # Normalize angles
        theta_next = self.normalize_angle(theta_next)
        
        # Stack results
        next_states = torch.stack([x_next, y_next, theta_next], dim=1)
        
        return next_states
    
    def angular_to_steering(self, velocity: torch.Tensor, 
                           angular_velocity: torch.Tensor) -> torch.Tensor:
        """
        Convert angular velocity to steering angle
        
        Args:
            velocity: [K] linear velocity
            angular_velocity: [K] angular velocity
            
        Returns:
            steering_angles: [K] steering angles
        """
        # Avoid division by zero
        safe_velocity = torch.where(
            torch.abs(velocity) < 1e-6,
            torch.sign(velocity) * 1e-6,
            velocity
        )
        
        # Compute steering angle: delta = atan(w * L / v)
        steering_angles = torch.atan(angular_velocity * self.wheelbase / safe_velocity)
        
        # Clamp to maximum steering angle
        steering_angles = torch.clamp(
            steering_angles, 
            -self.max_steering_angle, 
            self.max_steering_angle
        )
        
        return steering_angles
    
    def steering_to_angular(self, velocity: torch.Tensor, 
                           steering_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert steering angle to angular velocity
        
        Args:
            velocity: [K] linear velocity
            steering_angle: [K] steering angles
            
        Returns:
            angular_velocity: [K] angular velocities
        """
        # w = v * tan(delta) / L
        angular_velocity = velocity * torch.tan(steering_angle) / self.wheelbase
        
        return angular_velocity
    
    def get_turning_radius(self, velocity: torch.Tensor, 
                          angular_velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute turning radius
        
        Args:
            velocity: [K] linear velocity
            angular_velocity: [K] angular velocity
            
        Returns:
            radius: [K] turning radii
        """
        # Avoid division by zero
        safe_angular = torch.where(
            torch.abs(angular_velocity) < 1e-6,
            torch.sign(angular_velocity) * 1e-6,
            angular_velocity
        )
        
        radius = torch.abs(velocity / safe_angular)
        
        return radius
    
    def validate_controls(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Validate and constrain control inputs based on vehicle limits
        
        Args:
            controls: [K, 2] control inputs (v, w)
            
        Returns:
            valid_controls: [K, 2] validated controls
        """
        v = controls[:, 0]
        w = controls[:, 1]
        
        # Convert to steering angle and validate
        steering_angles = self.angular_to_steering(v, w)
        
        # Convert back to angular velocity (this applies steering limits)
        w_validated = self.steering_to_angular(v, steering_angles)
        
        # Check turning radius constraint
        turning_radii = self.get_turning_radius(v, w_validated)
        
        # Apply minimum turning radius constraint
        invalid_mask = turning_radii < self.min_turning_radius
        if torch.any(invalid_mask):
            # Reduce angular velocity for trajectories violating min radius
            w_validated[invalid_mask] = v[invalid_mask] / self.min_turning_radius * torch.sign(w_validated[invalid_mask])
        
        return torch.stack([v, w_validated], dim=1)
    
    def normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-pi, pi]"""
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    
    def rollout_batch(self, initial_states: torch.Tensor, 
                     controls: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Batch rollout for multiple trajectories (optimization)
        
        Args:
            initial_states: [K, 3] initial states
            controls: [K, T, 2] control sequences
            dt: Time step
            
        Returns:
            trajectories: [K, T+1, 3] full trajectories
        """
        batch_size, time_steps, _ = controls.shape
        trajectories = torch.zeros(
            batch_size, time_steps + 1, 3, 
            device=self.device, dtype=self.dtype
        )
        
        # Set initial states
        trajectories[:, 0, :] = initial_states
        
        # Forward integrate
        for t in range(time_steps):
            current_states = trajectories[:, t, :]
            current_controls = controls[:, t, :]
            
            # Validate controls
            valid_controls = self.validate_controls(current_controls)
            
            # Forward step
            next_states = self.forward(current_states, valid_controls, dt)
            trajectories[:, t + 1, :] = next_states
        
        return trajectories