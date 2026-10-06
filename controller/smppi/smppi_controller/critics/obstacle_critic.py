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

Semantic keep-out (non_drivable, /costmap/keepout) is a separate channel with
the same graded shape but `keepout_cost` < `collision_cost`: outside keep-out it
still acts as a wall, and once the robot is inside it escaping is preferred but
never by touching a physical obstacle, which stays visible because /costmap is
no longer overwritten by the semantic layer.
"""

import torch
import numpy as np
from typing import Optional, Any

from .base_critic import BaseCritic
from .._verbose import vprint


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

        # Per-trajectory penalty for entering semantic keep-out. Kept below
        # collision_cost so a physical obstacle always outranks keep-out.
        self.keepout_cost = params.get('keepout_cost', 20000.0)

        # 전부 충돌인 사이클 진단용. collides.all() 은 GPU->CPU 동기화라
        # 매 사이클 부르면 optimize 가 26.6 -> 33.2 ms 로 늘어난다 (실측).
        # 20 사이클마다만 확인해 비용을 1/20 로 누른다.
        self._all_blocked_count = 0
        self._cycle = 0

        # Vehicle footprint polygon [x1,y1,x2,y2,...] in base frame
        default_footprint = [0.49, 0.3725, 0.49, -0.3725, -0.49, -0.3725, -0.49, 0.3725]
        self.footprint = list(params.get('footprint', default_footprint))
        if len(self.footprint) < 6 or len(self.footprint) % 2 != 0:
            vprint(f"[ObstacleCritic] Invalid footprint {self.footprint}, using default")
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
        # Keep-out snapshot (same swap rule); None until /costmap/keepout arrives
        self.keepout_info = None

        vprint(f"[ObstacleCritic] Footprint-sampled costmap collision checking")
        vprint(f"[ObstacleCritic] collision_cost={self.collision_cost}, "
              f"repulsion_factor={self.repulsion_factor}, "
              f"collision_threshold>={self.collision_value_threshold}, "
              f"unknown_is_lethal={self.unknown_is_lethal}")
        vprint(f"[ObstacleCritic] footprint vertices={len(self.footprint) // 2}, "
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
        # per pose (most conservative). Sentinel cells (>= 10) contribute 0.
        #
        # Colliding poses used to be zeroed here. That removed the only "move away
        # from the wall" signal at exactly the moment it was needed: once the
        # current footprint touched a lethal cell, every rollout scored the same
        # constant and the obstacle term carried no direction at all. Keep it.
        sample_rep = torch.where(g < 10.0, g, torch.zeros_like(g))
        pose_max = sample_rep.max(dim=2).values  # [K, T+1], already squared
        repulsion = self.repulsion_factor * pose_max * 100.0

        # t=0 is the pose the robot is already in. No control sequence can change
        # it, so penalising it adds the same constant to all K rollouts - measured
        # in the field as every cost pinned at 1e7, which left the steering
        # decision to the goal term alone, 34,000x smaller and quantised away in
        # float32. Judge only what the plan can still avoid.
        future_collision = pose_collision[:, 1:] if pose_collision.shape[1] > 1 \
            else pose_collision

        severity, first_hit, collides = self._graded_severity(future_collision, repulsion.dtype)
        T_f = future_collision.shape[1]

        total_costs = repulsion.sum(dim=1) + severity * self.collision_cost

        # Semantic keep-out, own grid and own (lower) cost. Indexed with the
        # keep-out grid's own origin: it is published with /costmap but the two
        # callbacks are independent, so the windows can be one cycle apart.
        kinfo = self.keepout_info
        if kinfo is not None and self.keepout_cost > 0.0:
            kx = torch.floor((wx - kinfo['origin_x']) / kinfo['resolution']).long()
            ky = torch.floor((wy - kinfo['origin_y']) / kinfo['resolution']).long()
            k_inside = (kx >= 0) & (kx < kinfo['width']) & (ky >= 0) & (ky < kinfo['height'])
            k_cell = kinfo['mask'][ky.clamp(0, kinfo['height'] - 1), kx.clamp(0, kinfo['width'] - 1)]
            k_pose = (k_inside & k_cell).any(dim=2)                      # [K, T+1]
            k_future = k_pose[:, 1:] if k_pose.shape[1] > 1 else k_pose  # t=0: same for all rollouts
            k_severity, _, _ = self._graded_severity(k_future, repulsion.dtype)
            total_costs = total_costs + k_severity * self.keepout_cost

        # 충돌 없는 궤적이 하나도 없는 상황을 드러낸다. 조용한 포화가
        # 현장 실패를 bag 재생으로만 찾을 수 있게 만든 원인이었다.
        self._cycle += 1
        if self._cycle % 20 == 0 and bool(collides.all()):
            self._all_blocked_count += 1
            print(f"[ObstacleCritic] ALL {collides.numel()} rollouts collide "
                  f"(earliest step median={first_hit.median().item():.0f}/{T_f}) "
                  f"- no collision-free plan exists this cycle", flush=True)

        return self.apply_weight(total_costs)

    @staticmethod
    def _graded_severity(future_hits: torch.Tensor, dtype):
        """Severity in [0, 1] per rollout from a [K, T] hit mask (0 if never hit).

        Grade by WHEN the collision happens instead of a binary any(). A binary
        flag makes "hits the wall next step" and "hits it 59 steps out" cost the
        same, so there is no gradient to escape along. Later is cheaper, and the
        spread is large enough to survive float32 at this magnitude.

        언제 부딪히는지(first_hit)만 보면 "깊게 관통하면서 충돌을 뒤로 미루는"
        궤적이 싸진다. 실측: 궤적점의 46.1% 가 치명 셀, base 는 29.9% 였다.
        그래서 관통 깊이(충돌 자세 개수)를 같은 무게로 함께 벌한다.
          늦게 스치기        : first_hit 큼, depth 작음  -> 싸다 (탈출 기울기 유지)
          깊게 관통          : depth 큼                  -> 비싸다 (침범 억제)
        """
        T_f = future_hits.shape[1]
        idx = torch.arange(T_f, device=future_hits.device).to(dtype)
        big = torch.full_like(idx, float(T_f))
        first_hit = torch.where(future_hits, idx.expand_as(future_hits),
                                big.expand_as(future_hits)).min(dim=1).values
        collides = future_hits.any(dim=1)
        depth = future_hits.to(dtype).sum(dim=1) / float(T_f)
        severity = torch.where(collides,
                               0.5 * (1.0 - first_hit / float(T_f)) + 0.5 * depth,
                               torch.zeros_like(first_hit))
        return severity, first_hit, collides

    def set_keepout_info(self, keepout_info: dict):
        """
        Set the semantic keep-out grid (/costmap/keepout from local_costmap).

        Args:
            keepout_info: resolution, origin_x, origin_y, width, height and
                data (2D int8 array, 100 = non_drivable, 0 = clear,
                -1 = no semantic information -> no penalty)
        """
        tensor = torch.as_tensor(np.ascontiguousarray(keepout_info['data']), device=self.device)
        info = {k: keepout_info[k] for k in ('resolution', 'origin_x', 'origin_y', 'width', 'height')}
        info['mask'] = tensor >= 100
        self.keepout_info = info

    @property
    def all_blocked(self) -> bool:
        """직전 사이클에 충돌 없는 궤적이 하나도 없었는가 (상위 노드 진단용)."""
        return self._all_blocked_count > 0

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
        if 'keepout_cost' in params:
            self.keepout_cost = params['keepout_cost']

        if 'footprint' in params or 'footprint_padding' in params:
            if 'footprint' in params:
                self.footprint = params['footprint']
            if 'footprint_padding' in params:
                self.footprint_padding = params['footprint_padding']
            self.sample_points = self._build_sample_points()

        vprint(f"[ObstacleCritic] Parameters updated: "
              f"collision={self.collision_cost}, repulsion={self.repulsion_factor}, "
              f"collision_threshold>={self.collision_value_threshold}, "
              f"unknown_is_lethal={self.unknown_is_lethal}, "
              f"sample_points={self.sample_points.shape[0]}")
