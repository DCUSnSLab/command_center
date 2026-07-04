#!/usr/bin/env python3
"""
Lateral Bias Critic for SMPPI

Sidewalk driving: the curb (drop to the road) is on one known side of the
route. When avoiding an obstacle the optimizer should prefer swerving AWAY
from the curb side. This critic adds an asymmetric lateral cost relative to
the chord from the robot to the current goal:

    e = signed lateral offset of each trajectory point from the chord
        (left of travel direction = +, right = -)
    cost = weight * sum(max(0, side * e - deadband)^2)

with side = -1 penalising RIGHT offsets (curb on the right, the recorded
route) and side = +1 penalising LEFT offsets. Offsets inside `deadband` are
free, so normal path tracking is unaffected; only avoidance excursions toward
the curb get taxed. The hard no-go remains the lethal costmap (curb walls +
corridor keepout) — this critic only shapes WHICH side avoidance prefers.
"""

import torch
from typing import Optional, Any

from .base_critic import BaseCritic


class LateralBiasCritic(BaseCritic):
    """Asymmetric lateral-offset cost (prefer avoiding away from the curb)."""

    SIDES = {'right': -1.0, 'left': 1.0, 'none': 0.0}

    def __init__(self, params: dict):
        super().__init__("LateralBiasCritic", params)

        side_name = str(params.get('bias_side', 'right')).lower()
        if side_name not in self.SIDES:
            print(f"[LateralBiasCritic] unknown bias_side '{side_name}', using 'right'")
            side_name = 'right'
        self.side_name = side_name
        self.side = self.SIDES[side_name]
        self.deadband = float(params.get('deadband', 0.25))

        print(f"[LateralBiasCritic] bias_side={self.side_name} "
              f"(penalise offsets toward it), deadband={self.deadband} m, "
              f"weight={self.weight}")

    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                     robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                     obstacles: Optional[Any]) -> torch.Tensor:
        """
        Args:
            trajectories: [K, T+1, 3] (x, y, theta)
            robot_state:  [5] (x, y, yaw, v, w)
            goal_state:   [3] (x, y, theta) current goal
        Returns:
            costs: [K]
        """
        K = trajectories.shape[0]
        if (not self.enabled or self.side == 0.0 or goal_state is None
                or robot_state is None):
            return torch.zeros(K, device=self.device, dtype=self.dtype)

        # travel chord: robot -> goal
        p0 = robot_state[:2]
        d = goal_state[:2] - p0
        norm = torch.linalg.norm(d)
        if norm < 1e-3:
            return torch.zeros(K, device=self.device, dtype=self.dtype)
        d = d / norm

        # signed lateral offset of every trajectory point from the chord:
        # e = cross(d, p - p0);  e > 0 -> point is LEFT of travel direction
        rel = trajectories[:, :, :2] - p0          # [K, T+1, 2]
        e = d[0] * rel[:, :, 1] - d[1] * rel[:, :, 0]   # [K, T+1]

        # offsets toward the penalised side, beyond the deadband
        toward = torch.clamp(self.side * e - self.deadband, min=0.0)
        costs = (toward ** 2).sum(dim=1)           # [K]

        return self.apply_weight(costs)

    def update_parameters(self, params: dict):
        if 'bias_side' in params:
            name = str(params['bias_side']).lower()
            if name in self.SIDES:
                self.side_name = name
                self.side = self.SIDES[name]
        if 'deadband' in params:
            self.deadband = float(params['deadband'])
        if 'weight' in params:
            self.weight = float(params['weight'])
        print(f"[LateralBiasCritic] updated: side={self.side_name}, "
              f"deadband={self.deadband}, weight={self.weight}")
