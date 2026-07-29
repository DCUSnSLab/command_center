#!/usr/bin/env python3
"""
Obstacle Avoidance Critic for SMPPI
Footprint-based costmap collision checking, fully on-device (torch)

For every trajectory pose the (padded) robot footprint is rotated/translated
into the world frame and sampled along its perimeter (plus the center point).
Each sample point does one costmap lookup:

    -1 (UNKNOWN):                       collision if unknown_is_lethal (default), else free
    >= collision_value_threshold (100): collision (actual obstacle cell)
    1..99 (inflation gradient):         continuous repulsion = repulsion_factor * (v/100)^2 * 100
    0 (FREE):                           no cost
    out of map bounds:                  collision (conservative; rolling window makes this rare)

Per pose: collision if ANY sample collides; repulsion uses the max sample value.
A trajectory containing any colliding pose gets `collision_cost` added once,
sized to dominate any accumulated repulsion so MPPI can never prefer a
colliding trajectory over one that merely grazes the inflation zone.
"""

import torch
import numpy as np
from typing import Optional, Any

from .base_critic import BaseCritic


class ObstacleCritic(BaseCritic):
    """Obstacle avoidance critic using footprint-sampled costmap lookup"""

    def __init__(self, params: dict):
        """Initialize costmap-based obstacle critic"""
        super().__init__("ObstacleCritic", params)

        # Per-trajectory penalty when any pose collides.
        # Must dominate worst-case repulsion sum (~repulsion_factor * 100 * horizon)
        self.collision_cost = params.get('collision_cost', 100000.0)

        # Scaling of the continuous repulsion term in the inflation zone
        self.repulsion_factor = params.get('repulsion_factor', 2.0)

        # Costmap value at/above which a sample point is a definite collision
        # (100 = OCCUPIED in the local_costmap package; inflation stays <= 99)
        self.collision_value_threshold = params.get('collision_value_threshold', 100)

        # UNKNOWN (-1) cells: lethal by default so a cleared/stale costmap
        # (e.g. sensor timeout) stops the robot instead of freeing all space
        self.unknown_is_lethal = params.get('unknown_is_lethal', True)

        # Vehicle footprint polygon [x1,y1,x2,y2,...] in base frame
        default_footprint = [0.49, 0.3725, 0.49, -0.3725, -0.49, -0.3725, -0.49, 0.3725]
        self.footprint = list(params.get('footprint', default_footprint))
        if len(self.footprint) < 6 or len(self.footprint) % 2 != 0:
            print(f"[ObstacleCritic] Invalid footprint {self.footprint}, using default")
            self.footprint = default_footprint
        self.footprint_padding = params.get('footprint_padding', 0.0)

        # Perimeter sample spacing; half the costmap resolution so no cell
        # crossed by the footprint boundary is skipped
        self.sample_spacing = params.get('footprint_sample_spacing', 0.05)

        # Precomputed sample points in body frame, [P, 2] on device
        self.sample_points = self._build_sample_points()

        # Costmap snapshot; the whole dict is swapped atomically by the
        # subscriber thread, readers grab one local reference per call
        self.costmap_info = None

        print(f"[ObstacleCritic] Footprint-sampled costmap collision checking")
        print(f"[ObstacleCritic] collision_cost={self.collision_cost}, "
              f"repulsion_factor={self.repulsion_factor}, "
              f"collision_threshold>={self.collision_value_threshold}, "
              f"unknown_is_lethal={self.unknown_is_lethal}")
        print(f"[ObstacleCritic] footprint vertices={len(self.footprint) // 2}, "
              f"padding={self.footprint_padding}, "
              f"sample_points={self.sample_points.shape[0]}")

    def _build_sample_points(self) -> torch.Tensor:
        """
        Build footprint sample points in the body frame: padded polygon
        vertices, perimeter samples every `sample_spacing`, and the center.

        Returns:
            sample_points: [P, 2] tensor on device
        """
        flat = self.footprint
        vertices = []
        for i in range(0, len(flat) - 1, 2):
            x, y = float(flat[i]), float(flat[i + 1])
            # Nav2-style padding: push each vertex outward along both axes
            x += np.sign(x) * self.footprint_padding
            y += np.sign(y) * self.footprint_padding
            vertices.append((x, y))

        points = [(0.0, 0.0)]  # center: catches obstacles fully inside the footprint

        if len(vertices) >= 3:
            n = len(vertices)
            for i in range(n):
                x1, y1 = vertices[i]
                x2, y2 = vertices[(i + 1) % n]
                edge_len = float(np.hypot(x2 - x1, y2 - y1))
                num_seg = max(1, int(np.ceil(edge_len / self.sample_spacing)))
                for k in range(num_seg):  # includes vertex, excludes edge end (next edge adds it)
                    t = k / num_seg
                    points.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))

        return torch.tensor(points, device=self.device, dtype=self.dtype)

    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                     robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                     obstacles: Optional[Any]) -> torch.Tensor:
        """
        Compute obstacle avoidance cost using footprint-sampled costmap lookup

        Args:
            trajectories: [K, T+1, 3] sampled trajectories (x, y, theta)
            controls: [K, T, 2] control sequences (unused)
            robot_state: [5] robot state (unused)
            goal_state: [3] goal state (unused)
            obstacles: Obstacle data (unused in costmap mode)

        Returns:
            costs: [K] total obstacle costs per trajectory
        """
        info = self.costmap_info

        if not self.enabled or info is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        width = info['width']
        height = info['height']

        # Rotate/translate footprint samples into world frame: [K, T+1, P]
        theta = trajectories[:, :, 2].unsqueeze(-1)      # [K, T+1, 1]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        fx = self.sample_points[:, 0]                    # [P]
        fy = self.sample_points[:, 1]                    # [P]
        wx = trajectories[:, :, 0].unsqueeze(-1) + fx * cos_t - fy * sin_t
        wy = trajectories[:, :, 1].unsqueeze(-1) + fx * sin_t + fy * cos_t

        # World to grid (floor, so cells left/below the origin stay negative)
        gx = torch.floor((wx - info['origin_x']) / info['resolution']).long()
        gy = torch.floor((wy - info['origin_y']) / info['resolution']).long()

        inside = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)  # [K, T+1, P]

        # Clamped index; out-of-bounds samples are overridden below.
        # SINGLE gather from the packed grid (see set_costmap_info encoding)
        g = info['packed_grid'][gy.clamp(0, height - 1), gx.clamp(0, width - 1)]

        if self.unknown_is_lethal:
            collision = ~inside | (g >= 10.0)              # collision or unknown
        else:
            collision = ~inside | ((g >= 10.0) & (g < 20.0))  # collision only

        pose_collision = collision.any(dim=2)  # [K, T+1]

        # Continuous repulsion over the inflation gradient: max sample value
        # per pose (most conservative). Sentinel cells (>= 10) contribute 0;
        # colliding poses are zeroed since the collision penalty dominates
        sample_rep = torch.where(g < 10.0, g, torch.zeros_like(g))
        pose_max = sample_rep.max(dim=2).values  # [K, T+1], already squared
        repulsion = torch.where(pose_collision,
                                torch.zeros_like(pose_max),
                                self.repulsion_factor * pose_max * 100.0)

        total_costs = repulsion.sum(dim=1)
        total_costs = total_costs + pose_collision.any(dim=1).to(total_costs.dtype) * self.collision_cost

        return self.apply_weight(total_costs)

    def set_costmap_info(self, costmap_info: dict):
        """
        Set costmap information for grid-based collision detection

        Args:
            costmap_info: Dictionary with costmap metadata
                - resolution: cell size in meters (e.g., 0.1)
                - origin_x, origin_y: map origin in world coordinates
                - width, height: grid dimensions in cells
                - data: 2D numpy array of shape (height, width), values [-1, 100]
                - costmap_tensor (optional): pre-built torch tensor of the same
                  data; passed in to share one on-device copy between consumers
        """
        info = dict(costmap_info)

        tensor = info.get('costmap_tensor')
        if tensor is None:
            tensor = torch.as_tensor(np.ascontiguousarray(info['data']))
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        info['costmap_tensor'] = tensor

        # Precompute ONE packed grid per costmap (40k cells at ~10 Hz) so the
        # per-cycle lookup over K*T*P (millions of) sample points is a SINGLE
        # random-access gather. Encoding:
        #   [0, 1):  repulsion (value/100)^2, squared so per-pose reduce is max()
        #   10.0:    collision cell (value >= collision_value_threshold)
        #   20.0:    unknown cell (value < 0)
        info['packed_grid'] = self._build_packed_grid(tensor)

        self.costmap_info = info

    def _build_packed_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        packed = (tensor.clamp(min=0.0) / 100.0).square()
        packed = torch.where(tensor >= self.collision_value_threshold,
                             torch.full_like(packed, 10.0), packed)
        packed = torch.where(tensor < 0, torch.full_like(packed, 20.0), packed)
        return packed

    def update_parameters(self, params: dict):
        """
        Update obstacle critic parameters dynamically

        Args:
            params: Dictionary with parameter updates
                - collision_cost: Per-trajectory penalty for colliding trajectories
                - repulsion_factor: Scaling factor for inflation-zone repulsion
                - collision_value_threshold: Costmap value treated as collision
                - unknown_is_lethal: Whether UNKNOWN (-1) cells are collisions
                - footprint / footprint_padding: Vehicle footprint update
        """
        if 'collision_cost' in params:
            self.collision_cost = params['collision_cost']
        if 'repulsion_factor' in params:
            self.repulsion_factor = params['repulsion_factor']
        if 'collision_value_threshold' in params:
            self.collision_value_threshold = params['collision_value_threshold']
            # packed_grid was precomputed with the old threshold
            if self.costmap_info is not None:
                info = dict(self.costmap_info)
                info['packed_grid'] = self._build_packed_grid(info['costmap_tensor'])
                self.costmap_info = info
        if 'unknown_is_lethal' in params:
            self.unknown_is_lethal = params['unknown_is_lethal']

        if 'footprint' in params or 'footprint_padding' in params:
            if 'footprint' in params:
                self.footprint = params['footprint']
            if 'footprint_padding' in params:
                self.footprint_padding = params['footprint_padding']
            self.sample_points = self._build_sample_points()

        print(f"[ObstacleCritic] Parameters updated: "
              f"collision={self.collision_cost}, repulsion={self.repulsion_factor}, "
              f"collision_threshold>={self.collision_value_threshold}, "
              f"unknown_is_lethal={self.unknown_is_lethal}, "
              f"sample_points={self.sample_points.shape[0]}")
