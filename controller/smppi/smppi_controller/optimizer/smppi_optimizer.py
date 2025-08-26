#!/usr/bin/env python3
"""
SMPPI Optimizer - Nav2 structure with SMPPI enhancements
Combines Nav2's stable MPPI implementation with SMPPI's smoothing features
"""

import torch
import numpy as np
from typing import Optional, Tuple
import time

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path


class SMPPIOptimizer:
    """
    Nav2-based MPPI with SMPPI enhancements
    """
    
    def __init__(self, params: dict):
        """Initialize SMPPI Optimizer"""
        # Nav2-style parameters
        self.K = params.get('batch_size', 1000)  # Number of samples
        self.T = params.get('time_steps', 30)    # Time horizon
        self.dt = params.get('model_dt', 0.1)    # Time step
        self.temperature = params.get('temperature', 1.0)
        self.iteration_count = params.get('iteration_count', 1)  # Nav2: single iteration
        
        # SMPPI enhancements
        self.lambda_action = params.get('lambda_action', 0.1)
        self.smoothing_factor = params.get('smoothing_factor', 0.8)
        
        # Control constraints
        self.v_min = params.get('v_min', 0.0)
        self.v_max = params.get('v_max', 2.0)
        self.w_min = params.get('w_min', -1.0)
        self.w_max = params.get('w_max', 1.0)
        
        # State variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        
        # Control sequence (Nav2 style)
        self.control_sequence = torch.zeros(self.T, 2, device=self.device, dtype=self.dtype)
        
        # Sampling noise
        self.noise_std = torch.tensor([0.2, 0.2], device=self.device, dtype=self.dtype)  # [v, w]
        
        # State storage
        self.robot_state = None
        self.goal_state = None
        self.obstacles = None
        self.path = None
        
        # Critics (to be set externally)
        self.critics = []
        
        # Motion model (to be set externally) 
        self.motion_model = None
        
        print(f"[SMPPI] Initialized with K={self.K}, T={self.T}, device={self.device}")
    
    def set_motion_model(self, motion_model):
        """Set motion model (Nav2 pattern)"""
        self.motion_model = motion_model
    
    def add_critic(self, critic):
        """Add critic function (Nav2 pattern)"""
        self.critics.append(critic)
    
    def prepare(self, robot_pose: PoseStamped, robot_velocity: Twist, 
               path: Optional[Path] = None, goal: Optional[PoseStamped] = None):
        """
        Prepare state for optimization (Nav2 pattern)
        """
        # Extract robot state
        x = robot_pose.pose.position.x
        y = robot_pose.pose.position.y
        
        # Convert quaternion to yaw
        quat = robot_pose.pose.orientation
        yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                        1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z))
        
        self.robot_state = torch.tensor([x, y, yaw, robot_velocity.linear.x, robot_velocity.angular.z],
                                       device=self.device, dtype=self.dtype)
        
        # Set goal
        if goal:
            goal_x = goal.pose.position.x
            goal_y = goal.pose.position.y
            goal_quat = goal.pose.orientation
            goal_yaw = np.arctan2(2.0 * (goal_quat.w * goal_quat.z + goal_quat.x * goal_quat.y),
                                 1.0 - 2.0 * (goal_quat.y * goal_quat.y + goal_quat.z * goal_quat.z))
            self.goal_state = torch.tensor([goal_x, goal_y, goal_yaw],
                                          device=self.device, dtype=self.dtype)
        
        # Store path
        self.path = path
    
    def set_obstacles(self, obstacles):
        """Set obstacle data"""
        self.obstacles = obstacles
    
    def optimize(self) -> torch.Tensor:
        """
        Main optimization loop (Nav2 pattern with SMPPI enhancements)
        """
        for iteration in range(self.iteration_count):
            # 1. Generate noised trajectories (Nav2 style)
            trajectories, controls = self.generate_noised_trajectories()
            
            # 2. Compute trajectory costs (Nav2 critic system)
            trajectory_costs = self.evaluate_trajectories(trajectories, controls)
            
            # 3. SMPPI enhancement: Add action sequence cost
            action_costs = self.compute_action_sequence_cost(controls)
            
            # 4. Combine costs 
            total_costs = trajectory_costs + self.lambda_action * action_costs
            
            # 5. Update control sequence (Nav2 softmax)
            self.update_control_sequence(controls, total_costs)
        
        return self.control_sequence
    
    def generate_noised_trajectories(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate noised trajectories (Nav2 pattern)
        """
        # Sample noise
        noise = torch.randn(self.K, self.T, 2, device=self.device, dtype=self.dtype)
        noise = noise * self.noise_std
        
        # Add noise to control sequence
        controls = self.control_sequence.unsqueeze(0).repeat(self.K, 1, 1) + noise
        
        # Apply control constraints (Nav2 style)
        controls[:, :, 0] = torch.clamp(controls[:, :, 0], self.v_min, self.v_max)
        controls[:, :, 1] = torch.clamp(controls[:, :, 1], self.w_min, self.w_max)
        
        # Forward simulate trajectories
        trajectories = self.simulate_trajectories(controls)
        
        return trajectories, controls
    
    def simulate_trajectories(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Forward simulate trajectories using motion model
        """
        if self.motion_model is None:
            raise ValueError("Motion model not set")
        
        batch_size = controls.shape[0]
        trajectories = torch.zeros(batch_size, self.T + 1, 3, device=self.device, dtype=self.dtype)
        
        # Set initial state
        trajectories[:, 0, :] = self.robot_state[:3].unsqueeze(0).repeat(batch_size, 1)
        
        # Forward simulate
        for t in range(self.T):
            current_state = trajectories[:, t, :]
            control = controls[:, t, :]
            next_state = self.motion_model.forward(current_state, control, self.dt)
            trajectories[:, t + 1, :] = next_state
        
        return trajectories
    
    def evaluate_trajectories(self, trajectories: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """
        Evaluate trajectories using critic functions (Nav2 pattern)
        """
        total_costs = torch.zeros(self.K, device=self.device, dtype=self.dtype)
        
        for critic in self.critics:
            costs = critic.compute_cost(trajectories, controls, 
                                      self.robot_state, self.goal_state, self.obstacles)
            total_costs += costs
        
        return total_costs
    
    def compute_action_sequence_cost(self, controls: torch.Tensor) -> torch.Tensor:
        """
        SMPPI: Compute action sequence smoothness cost
        """
        # Compute control differences
        control_diff = controls[:, 1:, :] - controls[:, :-1, :]
        
        # L2 norm of differences
        smoothness_cost = torch.sum(control_diff ** 2, dim=(1, 2))
        
        return smoothness_cost
    
    def update_control_sequence(self, controls: torch.Tensor, costs: torch.Tensor):
        """
        Update control sequence using softmax weighting (Nav2 style)
        """
        # Normalize costs
        costs_normalized = costs - torch.min(costs)
        
        # Softmax weights (Nav2 style)
        weights = torch.softmax(-costs_normalized / self.temperature, dim=0)
        
        # Weighted average (Nav2 style)
        self.control_sequence = torch.sum(controls * weights.unsqueeze(1).unsqueeze(2), dim=0)
        
        # SMPPI: Apply trajectory smoothing
        self.smooth_control_sequence()
    
    def smooth_control_sequence(self):
        """
        SMPPI: Apply smoothing to control sequence
        """
        if self.smoothing_factor > 0:
            # Simple exponential smoothing
            smoothed = torch.zeros_like(self.control_sequence)
            smoothed[0] = self.control_sequence[0]
            
            for t in range(1, self.T):
                smoothed[t] = (self.smoothing_factor * smoothed[t-1] + 
                              (1 - self.smoothing_factor) * self.control_sequence[t])
            
            self.control_sequence = smoothed
    
    def shift_control_sequence(self):
        """
        Shift control sequence for next iteration (Nav2 style)
        """
        # Simple shift (Nav2 pattern)
        self.control_sequence = torch.roll(self.control_sequence, -1, dims=0)
        self.control_sequence[-1] = self.control_sequence[-2]  # Repeat last control
    
    def get_control_command(self) -> Twist:
        """
        Get current control command
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = float(self.control_sequence[0, 0])
        cmd_vel.angular.z = float(self.control_sequence[0, 1])
        
        return cmd_vel
    
    def getOptimizedTrajectory(self) -> Optional[torch.Tensor]:
        """
        Get optimized trajectory for visualization
        
        Returns:
            trajectory: [T+1, 3] optimized trajectory or None
        """
        if self.robot_state is None:
            return None
        
        # Use current control sequence to simulate trajectory
        controls = self.control_sequence.unsqueeze(0)  # [1, T, 2]
        initial_state = self.robot_state[:3].unsqueeze(0)  # [1, 3]
        
        trajectory = self.simulate_trajectories(controls)  # [1, T+1, 3]
        
        return trajectory[0]  # [T+1, 3]
    
    def reset(self):
        """
        Reset optimizer state
        """
        self.control_sequence.zero_()
        print("[SMPPI] Optimizer reset")