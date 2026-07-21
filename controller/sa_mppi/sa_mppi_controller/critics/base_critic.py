#!/usr/bin/env python3
"""
Base Critic Class for SMPPI
Abstract base class following Nav2 critic pattern
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Any


class BaseCritic(ABC):
    """
    Abstract base class for SMPPI critics
    Follows Nav2 critic pattern
    """
    
    def __init__(self, name: str, params: dict):
        """
        Initialize base critic
        
        Args:
            name: Name of the critic
            params: Parameters dictionary
        """
        self.name = name
        self.weight = params.get('weight', 1.0)
        self.enabled = params.get('enabled', True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        
        print(f"[{self.name}] Initialized with weight={self.weight}")
    
    @abstractmethod
    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                    robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                    obstacles: Optional[Any]) -> torch.Tensor:
        """
        Compute cost for trajectories
        
        Args:
            trajectories: [K, T+1, 3] trajectory states (x, y, theta)
            controls: [K, T, 2] control sequences (v, w)
            robot_state: [5] current robot state (x, y, theta, v, w)
            goal_state: [3] goal state (x, y, theta) or None
            obstacles: Obstacle data or None
            
        Returns:
            costs: [K] cost values for each trajectory
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if critic is enabled"""
        return self.enabled
    
    def set_weight(self, weight: float):
        """Set critic weight"""
        self.weight = weight
        print(f"[{self.name}] Weight updated to {self.weight}")
    
    def set_enabled(self, enabled: bool):
        """Enable/disable critic"""
        self.enabled = enabled
        print(f"[{self.name}] {'Enabled' if enabled else 'Disabled'}")
    
    def normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-pi, pi]"""
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    
    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distance between positions"""
        return torch.sqrt(torch.sum((pos1 - pos2) ** 2, dim=-1))
    
    def apply_weight(self, costs: torch.Tensor) -> torch.Tensor:
        """Apply critic weight to costs"""
        return self.weight * costs