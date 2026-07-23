#!/usr/bin/env python3
"""
Obstacle Avoidance Critic for SMPPI
Costmap-based grid collision detection (optimized for performance)
"""

import torch
import numpy as np
from typing import Optional, Any

from .base_critic import BaseCritic


class ObstacleCritic(BaseCritic):
    """
    Obstacle avoidance critic using costmap grid-based collision detection

    Costmap value mapping:
        0-49:   Free space    → cost = 0
        50-79:  Inflation     → cost = scaled repulsion
        80-100: Occupied      → cost = collision_cost
    """

    def __init__(self, params: dict):
        """Initialize costmap-based obstacle critic"""
        super().__init__("ObstacleCritic", params)

        # Collision cost parameters
        self.collision_cost = params.get('collision_cost', 1000.0)
        self.repulsion_factor = params.get('repulsion_factor', 2.0)

        # Costmap threshold parameters
        self.occupied_cost_threshold = params.get('occupied_cost_threshold', 80)  # Cost >= 80 is occupied
        self.inflation_zone_start = params.get('inflation_zone_start', 50)        # Cost >= 50 is inflation zone

        # Vehicle footprint parameters
        self.footprint = params.get('footprint', [0.49, 0.3725, 0.49, -0.3725, -0.49, -0.3725, -0.49, 0.3725])
        self.footprint_padding = params.get('footprint_padding', 0.15)
        self.use_polygon_collision = params.get('use_polygon_collision', True)

        # Parse footprint into list of (x, y) tuples
        self.footprint_points = []
        for i in range(0, len(self.footprint), 2):
            if i + 1 < len(self.footprint):
                self.footprint_points.append((self.footprint[i], self.footprint[i+1]))

        # Footprint as a GPU tensor [V, 2] (built once) for the vectorized path.
        if len(self.footprint_points) > 0:
            self._footprint_t = torch.tensor(self.footprint_points,
                                             device=self.device, dtype=self.dtype)  # [V,2]
        else:
            self._footprint_t = None

        # Costmap cache (updated via set_costmap_info). We keep both the raw dict
        # and a cached GPU tensor of the grid so per-tick cost eval stays on GPU
        # (no per-cycle GPU<->CPU transfer of the trajectories).
        self.costmap_info = None
        self._costmap_t = None       # [H, W] float tensor on device
        self._cm_res = 0.1
        self._cm_ox = 0.0
        self._cm_oy = 0.0
        self._cm_w = 0
        self._cm_h = 0

        print(f"[ObstacleCritic] Costmap-based collision detection")
        print(f"[ObstacleCritic] collision_cost={self.collision_cost}, repulsion_factor={self.repulsion_factor}")
        print(f"[ObstacleCritic] thresholds: inflation>={self.inflation_zone_start}, occupied>={self.occupied_cost_threshold}")
        print(f"[ObstacleCritic] footprint: {self.footprint_points}, padding={self.footprint_padding}, use_polygon={self.use_polygon_collision}")

    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                    robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                    obstacles: Optional[Any]) -> torch.Tensor:
        """
        Compute obstacle avoidance cost using costmap grid lookup

        Args:
            trajectories: [K, T+1, 3] sampled trajectories (x, y, theta)
            controls: [K, T, 2] control sequences (unused)
            robot_state: [5] robot state (unused)
            goal_state: [3] goal state (unused)
            obstacles: Obstacle data (unused in costmap mode)

        Returns:
            costs: [K] total obstacle costs per trajectory
        """
        if not self.enabled:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        # Check if costmap is available
        if self.costmap_info is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        return self.compute_cost_from_costmap(trajectories)

    def compute_cost_from_costmap(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Vectorized costmap collision cost, computed **entirely on the GPU**.

        The trajectories stay on-device (no .cpu()/.numpy()); only the small
        costmap grid is uploaded once per tick in set_costmap_info().  Semantics
        are identical to the previous NumPy version:
            free (<inflation_start)          -> 0
            inflation [inflation, occupied)  -> repulsion * norm^2 * 100
            occupied (>=occupied)            -> collision_cost
            out-of-bounds                    -> collision_cost
            per pose: max over footprint vertices; per traj: sum over horizon.
        """
        if self._costmap_t is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        K, T1, _ = trajectories.shape
        res = self._cm_res
        ox, oy = self._cm_ox, self._cm_oy
        W, H = self._cm_w, self._cm_h

        xy = trajectories[:, :, :2]                       # [K, T1, 2]
        if self.use_polygon_collision and self._footprint_t is not None:
            # Rotate footprint into world frame for every pose (GPU broadcast).
            theta = trajectories[:, :, 2]                 # [K, T1]
            cos_t = torch.cos(theta).unsqueeze(-1)        # [K, T1, 1]
            sin_t = torch.sin(theta).unsqueeze(-1)
            fx = self._footprint_t[:, 0].view(1, 1, -1)   # [1,1,V]
            fy = self._footprint_t[:, 1].view(1, 1, -1)
            wx = xy[:, :, 0:1] + (fx * cos_t - fy * sin_t)  # [K, T1, V]
            wy = xy[:, :, 1:2] + (fx * sin_t + fy * cos_t)  # [K, T1, V]
        else:
            wx = xy[:, :, 0:1]                            # [K, T1, 1]
            wy = xy[:, :, 1:2]

        gx = torch.floor((wx - ox) / res).long()          # [K, T1, V]
        gy = torch.floor((wy - oy) / res).long()
        valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        gxc = gx.clamp(0, W - 1)
        gyc = gy.clamp(0, H - 1)
        flat = (gyc * W + gxc).reshape(-1)                # [K*T1*V]
        vals = self._costmap_t.reshape(-1)[flat].reshape(gx.shape)  # [K, T1, V] in [0,100]

        # Map costmap values -> costs (vectorized, same thresholds as before).
        infl_range = max(float(self.occupied_cost_threshold - self.inflation_zone_start), 1e-6)
        norm = (vals - self.inflation_zone_start) / infl_range
        cost = torch.zeros_like(vals)
        infl_mask = (vals >= self.inflation_zone_start) & (vals < self.occupied_cost_threshold)
        cost = torch.where(infl_mask,
                           self.repulsion_factor * norm.pow(2) * 100.0,
                           cost)
        cost = torch.where(vals >= self.occupied_cost_threshold,
                           torch.full_like(cost, self.collision_cost),
                           cost)
        # out-of-bounds -> collision cost
        cost = torch.where(valid, cost, torch.full_like(cost, self.collision_cost))

        pose_cost = cost.max(dim=-1).values               # [K, T1] (most conservative vertex)
        total_costs = pose_cost.sum(dim=1)                # [K]
        return self.apply_weight(total_costs)

    def set_costmap_info(self, costmap_info: dict):
        """
        Set costmap info and cache the grid as a GPU tensor (uploaded once per
        tick).  The grid is small (~200x200) so this transfer is negligible,
        whereas keeping the per-cycle trajectory lookup on-GPU removes the large
        GPU->CPU sync that dominated runtime.
        """
        self.costmap_info = costmap_info
        if not costmap_info:
            self._costmap_t = None
            return
        data = costmap_info['data']  # (H, W) numpy array or tensor
        if isinstance(data, torch.Tensor):
            cm = data.to(device=self.device, dtype=self.dtype)
        else:
            cm = torch.as_tensor(np.asarray(data, dtype=np.float32),
                                 device=self.device, dtype=self.dtype)
        self._costmap_t = cm
        self._cm_res = float(costmap_info['resolution'])
        self._cm_ox = float(costmap_info['origin_x'])
        self._cm_oy = float(costmap_info['origin_y'])
        self._cm_w = int(costmap_info['width'])
        self._cm_h = int(costmap_info['height'])

    def update_parameters(self, params: dict):
        """
        Update obstacle critic parameters dynamically

        Args:
            params: Dictionary with parameter updates
                - collision_cost: Cost for occupied cells
                - repulsion_factor: Scaling factor for inflation zone
                - occupied_cost_threshold: Threshold for occupied cells
                - inflation_zone_start: Threshold for inflation zone start
        """
        if 'collision_cost' in params:
            self.collision_cost = params['collision_cost']
        if 'repulsion_factor' in params:
            self.repulsion_factor = params['repulsion_factor']
        if 'occupied_cost_threshold' in params:
            self.occupied_cost_threshold = params['occupied_cost_threshold']
        if 'inflation_zone_start' in params:
            self.inflation_zone_start = params['inflation_zone_start']

        print(f"[ObstacleCritic] Parameters updated: "
              f"collision={self.collision_cost}, repulsion={self.repulsion_factor}, "
              f"thresholds=[{self.inflation_zone_start}, {self.occupied_cost_threshold}]")
