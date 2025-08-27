#!/usr/bin/env python3
"""
Trajectory Sampler for SMPPI
Nav2-style noise generation with SMPPI enhancements
"""

import torch
import numpy as np
from typing import Tuple


class TrajectorySampler:
    """
    Generate trajectory samples for SMPPI optimization
    """
    
    def __init__(self, params: dict):
        """Initialize trajectory sampler"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        
        # Sampling parameters
        self.batch_size = params.get('batch_size', 1000)
        self.time_steps = params.get('time_steps', 30)
        
        # Noise parameters
        self.noise_std = torch.tensor(
            params.get('noise_std', [0.2, 0.2]), 
            device=self.device, dtype=self.dtype
        )
        
        # Control bounds
        self.control_bounds = {
            'v_min': params.get('v_min', 0.0),
            'v_max': params.get('v_max', 2.0),
            'w_min': params.get('w_min', -1.0),
            'w_max': params.get('w_max', 1.0)
        }
        
        print(f"[TrajectorySampler] Initialized with batch_size={self.batch_size}")
    
    def sample_controls(self, nominal_sequence: torch.Tensor) -> torch.Tensor:
        """
        Sample control sequences around nominal sequence
        
        Args:
            nominal_sequence: [T, 2] nominal control sequence
            
        Returns:
            controls: [K, T, 2] sampled control sequences
        """
        # Generate Gaussian noise
        noise = torch.randn(
            self.batch_size, self.time_steps, 2,
            device=self.device, dtype=self.dtype
        )
        
        # Scale by noise standard deviation
        noise = noise * self.noise_std
        
        # Add to nominal sequence
        controls = nominal_sequence.unsqueeze(0).repeat(self.batch_size, 1, 1) + noise
        
        # Apply control bounds
        controls = self.apply_control_bounds(controls)
        
        return controls
    
    def apply_control_bounds(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Apply control bounds to sampled trajectories
        
        Args:
            controls: [K, T, 2] control sequences
            
        Returns:
            bounded_controls: [K, T, 2] bounded control sequences
        """
        # Clamp linear velocity
        controls[:, :, 0] = torch.clamp(
            controls[:, :, 0], 
            self.control_bounds['v_min'], 
            self.control_bounds['v_max']
        )
        
        # Clamp angular velocity
        controls[:, :, 1] = torch.clamp(
            controls[:, :, 1],
            self.control_bounds['w_min'],
            self.control_bounds['w_max']
        )
        
        return controls
    
    def sample_correlated_noise(self, nominal_sequence: torch.Tensor, 
                               correlation_factor: float = 0.8) -> torch.Tensor:
        """
        Sample temporally correlated noise (SMPPI enhancement)
        
        Args:
            nominal_sequence: [T, 2] nominal sequence
            correlation_factor: temporal correlation factor
            
        Returns:
            controls: [K, T, 2] correlated control samples
        """
        controls = torch.zeros(
            self.batch_size, self.time_steps, 2,
            device=self.device, dtype=self.dtype
        )
        
        # Set initial control
        controls[:, 0, :] = nominal_sequence[0].unsqueeze(0).repeat(self.batch_size, 1)
        
        # Generate correlated noise
        for t in range(1, self.time_steps):
            # White noise
            white_noise = torch.randn(
                self.batch_size, 2, 
                device=self.device, dtype=self.dtype
            ) * self.noise_std
            
            # Correlated update
            controls[:, t, :] = (
                correlation_factor * controls[:, t-1, :] + 
                (1 - correlation_factor) * nominal_sequence[t] + 
                white_noise
            )
        
        # Apply bounds
        controls = self.apply_control_bounds(controls)
        
        return controls