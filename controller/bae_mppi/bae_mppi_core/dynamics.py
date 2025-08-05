"""
Dynamics models for MPPI controller
"""
import torch
import numpy as np


class TwistDynamics:
    """Twist-based dynamics model compatible with standard ROS cmd_vel interface"""
    
    def __init__(self, dt=0.1, device='cpu'):
        """
        Initialize twist dynamics
        
        Args:
            dt (float): Time step size
            device (str): PyTorch device ('cpu' or 'cuda')
        """
        self.dt = dt
        self.device = device
    
    def __call__(self, state, action):
        """
        Predict next state using twist model (nav2_mppi style)
        
        Args:
            state (torch.Tensor): Current state [x, y, theta] (K x 3)
            action (torch.Tensor): Control action [vx, wz] (K x 2)
                                  vx: linear velocity, wz: angular velocity
            
        Returns:
            torch.Tensor: Next state [x, y, theta] (K x 3)
        """
        # Extract state components
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        
        # Extract control components (Twist interface)
        vx = action[:, 0]  # linear velocity
        wz = action[:, 1]  # angular velocity
        
        # Simple velocity-based integration (like nav2_mppi)
        next_x = x + vx * torch.cos(theta) * self.dt
        next_y = y + vx * torch.sin(theta) * self.dt
        next_theta = theta + wz * self.dt
        
        # Normalize angle to [-pi, pi]
        next_theta = torch.atan2(torch.sin(next_theta), torch.cos(next_theta))
        
        # Stack and return next state
        next_state = torch.stack([next_x, next_y, next_theta], dim=1)
        return next_state.to(self.device)


class DifferentialDriveDynamics:
    """Differential drive robot dynamics model"""
    
    def __init__(self, dt=0.1, device='cpu'):
        """
        Initialize differential drive dynamics
        
        Args:
            dt (float): Time step size
            device (str): PyTorch device ('cpu' or 'cuda')
        """
        self.dt = dt
        self.device = device
    
    def __call__(self, state, action):
        """
        Predict next state given current state and action
        
        Args:
            state (torch.Tensor): Current state [x, y, theta] (K x 3)
            action (torch.Tensor): Control action [v, w] (K x 2)
            
        Returns:
            torch.Tensor: Next state [x, y, theta] (K x 3)
        """
        # Extract state components
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        
        # Extract control components
        v = action[:, 0]  # linear velocity
        w = action[:, 1]  # angular velocity
        
        # Forward dynamics using Euler integration
        next_x = x + v * torch.cos(theta) * self.dt
        next_y = y + v * torch.sin(theta) * self.dt
        next_theta = theta + w * self.dt
        
        # Normalize angle to [-pi, pi]
        next_theta = torch.atan2(torch.sin(next_theta), torch.cos(next_theta))
        
        # Stack and return next state
        next_state = torch.stack([next_x, next_y, next_theta], dim=1)
        return next_state.to(self.device)


class AckermannDynamics:
    """HUNTER Robot Tricycle Model (rear-wheel referenced Ackermann steering)"""
    
    def __init__(self, wheelbase=0.65, dt=0.1, device='cpu'):
        """
        Initialize HUNTER tricycle dynamics (rear-wheel reference)
        
        Args:
            wheelbase (float): Distance between front and rear axles (m) - HUNTER: 0.65m
            dt (float): Time step size
            device (str): PyTorch device ('cpu' or 'cuda')
        """
        self.wheelbase = wheelbase
        self.dt = dt
        self.device = device
    
    def __call__(self, state, action):
        """
        Predict next state using HUNTER tricycle model (rear-wheel reference)
        
        Args:
            state (torch.Tensor): Current state [x, y, theta] (K x 3)
                                 Position is at rear axle center
            action (torch.Tensor): Control action [v_rear, delta] (K x 2)
                                  v_rear: rear wheel velocity, delta: front steering angle
            
        Returns:
            torch.Tensor: Next state [x, y, theta] (K x 3)
        """
        # Extract state components (rear axle position)
        x_rear = state[:, 0]
        y_rear = state[:, 1]
        theta = state[:, 2]
        
        # Extract control components
        v_rear = action[:, 0]    # rear wheel velocity (traction)
        delta = action[:, 1]     # front wheel steering angle
        
        # HUNTER Tricycle Model (rear-wheel reference)
        # The robot rotates around the rear axle center
        next_x_rear = x_rear + v_rear * torch.cos(theta) * self.dt
        next_y_rear = y_rear + v_rear * torch.sin(theta) * self.dt
        next_theta = theta + (v_rear / self.wheelbase) * torch.tan(delta) * self.dt
        
        # Normalize angle to [-pi, pi]
        next_theta = torch.atan2(torch.sin(next_theta), torch.cos(next_theta))
        
        # Stack and return next state
        next_state = torch.stack([next_x_rear, next_y_rear, next_theta], dim=1)
        return next_state.to(self.device)