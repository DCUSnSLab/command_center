#!/usr/bin/env python3
"""
Obstacle Avoidance Critic for SMPPI
Clearance-based (metric distance field), fully on-device (torch)

Once per costmap a Euclidean distance field (metres, cell centre to cell
centre) to the nearest blocked cell is built:

    blocked = OCCUPIED (>= collision_value_threshold)
              + UNKNOWN (-1) if unknown_is_lethal (default)
              + the map border (leaving the rolling window counts as an obstacle)

The footprint is covered by `footprint_circles` discs along its long axis.
The discs contain the whole polygon (corners exact; HUNTER 980x745 with 3
discs: sides +3.4 cm, front/rear +24 cm on the centre line), so a disc test
never misses what the polygon would hit. Per pose, with bilinear lookups:

    clearance = min_i( dist(center_i) - r )     [m, < 0 = overlap]

    collision : clearance < collision_margin
                The margin only covers grid quantisation (an obstacle cell's
                centre is up to res/sqrt(2) from the point that marked it);
                it is NOT a safety margin. Graded severity * collision_cost.
    shortfall : max(0, d_req(v) - clearance),
                d_req(v) = clearance_min + clearance_time_gap * |v|
                soft cost = weight * (worst shortfall along the plan)^2

d_req grows with the rollout's own speed, so near obstacles MPPI can lower
the cost by slowing down, not only by steering away or stopping. Beyond d_req
the term is exactly zero: open space is left to the goal critic. The term is
in m^2 next to the goal critic's horizon-averaged metres, so the weight ratio
reads directly (obstacle/goal = 100: a 0.1 m worst shortfall costs as much as
1 m of carrot error, 0.2 m as much as 4 m).

Semantic keep-out (non_drivable, /costmap/keepout) is a separate channel: the
same disc test against its own distance field, graded severity *
`keepout_cost` < `collision_cost`. Outside keep-out it still acts as a wall,
and once the robot is inside it escaping is preferred but never by touching
a physical obstacle, which stays visible because /costmap is no longer
overwritten by the semantic layer.

collision_cost and keepout_cost are absolute; `weight` scales the soft term only.
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Optional, Any

from .base_critic import BaseCritic
from .._verbose import vprint


class ObstacleCritic(BaseCritic):
    """Obstacle avoidance critic using a metric clearance field"""

    def __init__(self, params: dict):
        """Initialize clearance-based obstacle critic"""
        super().__init__("ObstacleCritic", params)

        # Per-trajectory penalty when any pose collides (absolute, not scaled by
        # weight). Must dominate the soft term plus the goal term:
        # min severity (one late pose) = 1/T -> 1e6/60 = 16.7k > ~1.3k worst case
        self.collision_cost = params.get('collision_cost', 1.0e6)

        # Per-trajectory penalty for entering semantic keep-out (absolute).
        # Kept below collision_cost so a physical obstacle always outranks keep-out.
        self.keepout_cost = params.get('keepout_cost', 2.0e5)

        # Clearance below which a pose is a collision [m]. Grid quantisation only.
        self.collision_margin = params.get('collision_margin', 0.08)

        # Required clearance d_req(v) = clearance_min + clearance_time_gap * |v|
        self.clearance_min = params.get('clearance_min', 0.05)            # [m]
        self.clearance_time_gap = params.get('clearance_time_gap', 0.4)   # [s]

        # Costmap value at/above which a cell is an obstacle
        # (100 = OCCUPIED in the local_costmap package; inflation stays <= 99)
        self.collision_value_threshold = params.get('collision_value_threshold', 100)

        # UNKNOWN (-1) cells: obstacles by default so a cleared/stale costmap
        # (e.g. sensor timeout) stops the robot instead of freeing all space
        self.unknown_is_lethal = params.get('unknown_is_lethal', True)

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
        self.footprint_circles = int(params.get('footprint_circles', 3))

        # Disc centres in body frame [C, 2] on device, common radius [m]
        self.circle_centers, self.circle_radius = self._build_circles()

        # Costmap snapshot; the whole dict is swapped atomically by the
        # subscriber thread, readers grab one local reference per call
        self.costmap_info = None
        # Keep-out snapshot (same swap rule); None until /costmap/keepout arrives
        self.keepout_info = None

        vprint(f"[ObstacleCritic] Clearance field, weight={self.weight}, "
              f"collision_cost={self.collision_cost}, keepout_cost={self.keepout_cost}")
        vprint(f"[ObstacleCritic] collision_margin={self.collision_margin} m, "
              f"d_req = {self.clearance_min} + {self.clearance_time_gap}*|v| m, "
              f"discs={self.circle_centers.shape[0]} r={self.circle_radius:.3f} m, "
              f"unknown_is_lethal={self.unknown_is_lethal}")

    def _build_circles(self):
        """
        Cover the footprint's bounding box with equal discs along its long axis.

        Returns:
            centers: [C, 2] tensor on device (body frame)
            radius: disc radius [m]
        """
        xs, ys = self.footprint[0::2], self.footprint[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        n = max(1, self.footprint_circles)
        if x1 - x0 >= y1 - y0:
            half_seg, half_width = (x1 - x0) / (2 * n), (y1 - y0) / 2
            centers = [(x0 + (2 * i + 1) * half_seg, (y0 + y1) / 2) for i in range(n)]
        else:
            half_seg, half_width = (y1 - y0) / (2 * n), (x1 - x0) / 2
            centers = [((x0 + x1) / 2, y0 + (2 * i + 1) * half_seg) for i in range(n)]
        radius = float(np.hypot(half_seg, half_width))
        return torch.tensor(centers, device=self.device, dtype=self.dtype), radius

    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                     robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                     obstacles: Optional[Any]) -> torch.Tensor:
        """
        Compute obstacle cost from the clearance field

        Args:
            trajectories: [K, T+1, 3] sampled trajectories (x, y, theta)
            controls: [K, T, 2] action sequences (v, delta); controls[:, t, 0]
                is the speed that carries pose t to pose t+1
            robot_state: [5] robot state (unused)
            goal_state: [3] goal state (unused)
            obstacles: Obstacle data (unused in costmap mode)

        Returns:
            costs: [K] total obstacle costs per trajectory
        """
        info = self.costmap_info

        if not self.enabled or info is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        # Disc centres in world frame: [K, T+1, C]
        theta = trajectories[:, :, 2].unsqueeze(-1)      # [K, T+1, 1]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        cx = self.circle_centers[:, 0]                   # [C]
        cy = self.circle_centers[:, 1]                   # [C]
        wx = trajectories[:, :, 0].unsqueeze(-1) + cx * cos_t - cy * sin_t
        wy = trajectories[:, :, 1].unsqueeze(-1) + cx * sin_t + cy * cos_t

        # Outside the rolling window counts as touching an obstacle (distance 0)
        dist, inside = self._sample_field(info, wx, wy)
        dist = torch.where(inside, dist, torch.zeros_like(dist))
        clearance = dist.min(dim=2).values - self.circle_radius   # [K, T+1]

        # t=0 is the pose the robot is already in. No control sequence can change
        # it, so penalising it adds the same constant to all K rollouts - measured
        # in the field as every cost pinned at 1e7, which left the steering
        # decision to the goal term alone, 34,000x smaller and quantised away in
        # float32. Judge only what the plan can still avoid.
        future_clear = clearance[:, 1:] if clearance.shape[1] > 1 else clearance
        T_f = future_clear.shape[1]

        future_collision = future_clear < self.collision_margin
        severity, first_hit, collides = self._graded_severity(future_collision, clearance.dtype)

        # Speed-dependent required clearance: slowing down is a way to pay less
        speed = controls[:, :T_f, 0].abs()
        required = self.clearance_min + self.clearance_time_gap * speed
        shortfall = (required - future_clear).clamp(min=0.0)
        # Worst shortfall along the plan, not the horizon mean: a mean lets a
        # 0.3 m shortfall lasting 0.5 s cost 1/12 of the same pass held for 6 s,
        # so brushing past an obstacle at speed was nearly free, while creeping
        # through a long narrow passage was charged for every step spent in it.
        soft = shortfall.max(dim=1).values.square()                 # [K], m^2

        total_costs = self.weight * soft + self.collision_cost * severity

        # Semantic keep-out, own field and own (lower) cost. Indexed with the
        # keep-out grid's own origin: it is published with /costmap but the two
        # callbacks are independent, so the windows can be one cycle apart.
        # Outside the keep-out window there is no semantic information: no penalty.
        kinfo = self.keepout_info
        if kinfo is not None and self.keepout_cost > 0.0:
            kdist, k_inside = self._sample_field(kinfo, wx, wy)
            kdist = torch.where(k_inside, kdist, torch.full_like(kdist, float('inf')))
            k_clear = kdist.min(dim=2).values - self.circle_radius
            k_future = k_clear[:, 1:] if k_clear.shape[1] > 1 else k_clear   # t=0: same for all rollouts
            k_severity, _, _ = self._graded_severity(k_future < self.collision_margin, clearance.dtype)
            total_costs = total_costs + self.keepout_cost * k_severity

        # 충돌 없는 궤적이 하나도 없는 상황을 드러낸다. 조용한 포화가
        # 현장 실패를 bag 재생으로만 찾을 수 있게 만든 원인이었다.
        self._cycle += 1
        if self._cycle % 20 == 0 and bool(collides.all()):
            self._all_blocked_count += 1
            print(f"[ObstacleCritic] ALL {collides.numel()} rollouts collide "
                  f"(earliest step median={first_hit.median().item():.0f}/{T_f}) "
                  f"- no collision-free plan exists this cycle", flush=True)

        return total_costs

    @staticmethod
    def _sample_field(info: dict, wx: torch.Tensor, wy: torch.Tensor):
        """
        Bilinear read of a cell-centred field at world points.

        Returns:
            values: same shape as wx (clamped to the field edge outside)
            inside: bool mask, point lies within the outermost cell centres
        """
        field = info['distance_field']
        h, w = field.shape
        u = (wx - info['origin_x']) / info['resolution'] - 0.5
        v = (wy - info['origin_y']) / info['resolution'] - 0.5
        inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        u = u.clamp(0, w - 1)
        v = v.clamp(0, h - 1)
        i0 = u.floor().long().clamp(max=w - 2)
        j0 = v.floor().long().clamp(max=h - 2)
        fu, fv = u - i0, v - j0
        flat = field.reshape(-1)
        idx = j0 * w + i0
        top = flat[idx] * (1 - fu) + flat[idx + 1] * fu
        bottom = flat[idx + w] * (1 - fu) + flat[idx + w + 1] * fu
        return top * (1 - fv) + bottom * fv, inside

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

    def _distance_field(self, blocked: np.ndarray, resolution: float) -> torch.Tensor:
        """Metric distance (cell centre to cell centre) to the nearest blocked cell."""
        if blocked.any():
            dist = distance_transform_edt(~blocked) * resolution
        else:
            dist = np.full(blocked.shape, 1.0e3)   # nothing to keep away from
        return torch.as_tensor(dist, device=self.device, dtype=self.dtype)

    def _costmap_blocked(self, data: np.ndarray) -> np.ndarray:
        blocked = data >= self.collision_value_threshold
        if self.unknown_is_lethal:
            blocked |= data < 0
        # Leaving the window is a collision, and discs near the edge must see it
        blocked[0, :] = blocked[-1, :] = True
        blocked[:, 0] = blocked[:, -1] = True
        return blocked

    def set_keepout_info(self, keepout_info: dict):
        """
        Set the semantic keep-out grid (/costmap/keepout from local_costmap).

        Args:
            keepout_info: resolution, origin_x, origin_y, width, height and
                data (2D int8 array, 100 = non_drivable, 0 = clear,
                -1 = no semantic information -> no penalty)
        """
        info = {k: keepout_info[k] for k in ('resolution', 'origin_x', 'origin_y', 'width', 'height')}
        data = np.asarray(keepout_info['data'])
        info['distance_field'] = self._distance_field(data >= 100, info['resolution'])
        self.keepout_info = info

    @property
    def all_blocked(self) -> bool:
        """직전 사이클에 충돌 없는 궤적이 하나도 없었는가 (상위 노드 진단용)."""
        return self._all_blocked_count > 0

    def set_costmap_info(self, costmap_info: dict):
        """
        Set costmap information and rebuild the clearance field

        Args:
            costmap_info: Dictionary with costmap metadata
                - resolution: cell size in meters (e.g., 0.1)
                - origin_x, origin_y: map origin in world coordinates
                - width, height: grid dimensions in cells
                - data: 2D numpy array of shape (height, width), values [-1, 100]
        """
        info = {k: costmap_info[k] for k in ('resolution', 'origin_x', 'origin_y', 'width', 'height')}
        info['data'] = np.asarray(costmap_info['data'])
        # One field per costmap (~1 ms on CPU for 200x200) so the per-cycle
        # lookup over K*T*C points is a handful of on-device gathers
        info['distance_field'] = self._distance_field(self._costmap_blocked(info['data']),
                                                      info['resolution'])
        self.costmap_info = info

    def update_parameters(self, params: dict):
        """
        Update obstacle critic parameters dynamically

        Args:
            params: Dictionary with parameter updates
                - collision_cost / keepout_cost: per-trajectory penalties
                - collision_margin, clearance_min, clearance_time_gap: clearance [m, s]
                - collision_value_threshold / unknown_is_lethal: what blocks
                - footprint / footprint_circles: vehicle discs
        """
        for key in ('collision_cost', 'keepout_cost', 'collision_margin',
                    'clearance_min', 'clearance_time_gap'):
            if key in params:
                setattr(self, key, params[key])

        if 'collision_value_threshold' in params or 'unknown_is_lethal' in params:
            self.collision_value_threshold = params.get('collision_value_threshold',
                                                        self.collision_value_threshold)
            self.unknown_is_lethal = params.get('unknown_is_lethal', self.unknown_is_lethal)
            # the field was built with the old blocking rule
            if self.costmap_info is not None:
                self.set_costmap_info(self.costmap_info)

        if 'footprint' in params or 'footprint_circles' in params:
            if 'footprint' in params:
                self.footprint = list(params['footprint'])
            if 'footprint_circles' in params:
                self.footprint_circles = int(params['footprint_circles'])
            self.circle_centers, self.circle_radius = self._build_circles()

        vprint(f"[ObstacleCritic] Parameters updated: "
              f"collision={self.collision_cost}, keepout={self.keepout_cost}, "
              f"margin={self.collision_margin}, d_req={self.clearance_min}+{self.clearance_time_gap}*|v|, "
              f"discs={self.circle_centers.shape[0]} r={self.circle_radius:.3f}")
