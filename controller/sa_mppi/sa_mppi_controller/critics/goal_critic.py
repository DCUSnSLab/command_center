#!/usr/bin/env python3
"""
Goal tracking critic for SMPPI (carrot-lookahead)

Tracks a lookahead point ("carrot") projected along the waypoint chain
[current_goal, next_waypoints...] so the robot does not decelerate at
intermediate nodes. Active cost is the distance-to-carrot term; heading /
alignment terms were removed after causing oscillation (see git history).

Lookahead distance is STATIC (clamp(base, min, max)) — velocity-adaptive
lookahead and curve-based reduction were removed: deployments ran with
base == max, and MPPI's own carrot-tracking cost already slows the robot
in turns (a slow accurate arc tracks the carrot better than a fast wide
one).

The carrot is rate-limited (`lookahead_max_step` per cycle) so behavior
transitions move it smoothly instead of jumping; it snaps only when the
path itself changes (new path_id).

Performance: the carrot is pure scalar geometry, so it is computed in
plain Python floats (waypoints arrive as floats anyway) — no per-waypoint
GPU tensor churn. The only GPU work is the [K, T+1] distance cost; the
only sync is one .tolist() of the robot/goal positions per cycle.
"""

import math
import torch
from typing import Optional, Any, Tuple
from .base_critic import BaseCritic


class GoalCritic(BaseCritic):
    """Carrot-lookahead goal tracking critic"""

    def __init__(self, params: dict):
        super().__init__("GoalCritic", params)

        # Goal tolerances (also used by is_goal_reached)
        self.xy_goal_tolerance = params.get('xy_goal_tolerance', 0.05)
        self.yaw_goal_tolerance = params.get('yaw_goal_tolerance', 0.25)

        # Cost scales
        self.distance_scale = params.get('distance_scale', 1.0)
        self.progress_scale = params.get('progress_scale', 0.0)
        self.use_progress_reward = params.get('use_progress_reward', False)

        # Lookahead: static distance = clamp(base, min, max).
        # 'lookahead_velocity_factor' is accepted but ignored (legacy key).
        self.lookahead_base_distance = params.get('lookahead_base_distance', 2.5)
        self.lookahead_min_distance = params.get('lookahead_min_distance', 1.0)
        self.lookahead_max_distance = params.get('lookahead_max_distance', 6.0)

        # Carrot rate limiting (meters per compute cycle); <= 0 disables
        self.lookahead_max_step = params.get('lookahead_max_step', 0.5)

        # Behavior options
        self.use_multiple_waypoints = params.get('use_multiple_waypoints', True)

        # State
        self.multiple_waypoints = None
        self.previous_node_type = None
        self._carrot: Optional[Tuple[float, float]] = None
        self._carrot_path_id: Optional[str] = None

        # Visualization cache
        self.last_lookahead_point: Optional[torch.Tensor] = None
        self.last_lookahead_yaw: Optional[torch.Tensor] = None
        self.last_target_direction: Optional[torch.Tensor] = None

        print(f"[GoalCritic] lookahead={self._static_lookahead():.2f}m (static), "
              f"max_step={self.lookahead_max_step}m/cycle")

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def set_multiple_waypoints(self, waypoints_msg):
        self.multiple_waypoints = waypoints_msg

    def get_lookahead_point(self) -> Optional[torch.Tensor]:
        return self.last_lookahead_point

    def get_lookahead_yaw(self) -> Optional[torch.Tensor]:
        return self.last_lookahead_yaw

    def get_target_direction(self) -> Optional[torch.Tensor]:
        return self.last_target_direction

    def is_goal_reached(self, current_pose: torch.Tensor, goal_state: torch.Tensor) -> bool:
        if goal_state is None:
            return False
        dist = torch.norm(current_pose[:2] - goal_state[:2])
        ang = abs(self.normalize_angle(current_pose[2] - goal_state[2]))
        return (dist < self.xy_goal_tolerance) and (ang < self.yaw_goal_tolerance)

    def update_parameters(self, params: dict):
        self.xy_goal_tolerance = params.get('xy_goal_tolerance', self.xy_goal_tolerance)
        self.yaw_goal_tolerance = params.get('yaw_goal_tolerance', self.yaw_goal_tolerance)
        self.distance_scale = params.get('distance_scale', self.distance_scale)
        self.progress_scale = params.get('progress_scale', self.progress_scale)
        self.use_progress_reward = params.get('use_progress_reward', self.use_progress_reward)
        self.lookahead_base_distance = params.get('lookahead_base_distance', self.lookahead_base_distance)
        self.lookahead_min_distance = params.get('lookahead_min_distance', self.lookahead_min_distance)
        self.lookahead_max_distance = params.get('lookahead_max_distance', self.lookahead_max_distance)
        self.lookahead_max_step = params.get('lookahead_max_step', self.lookahead_max_step)
        self.use_multiple_waypoints = params.get('use_multiple_waypoints', self.use_multiple_waypoints)
        # legacy keys (lookahead_velocity_factor, curve_*, angle_scale, ...) are ignored
        print("[GoalCritic] Parameters updated")

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                     robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                     obstacles: Optional[Any]) -> torch.Tensor:
        """
        trajectories: [K, T+1, 3] (x, y, theta)
        robot_state:  [5] (x, y, yaw, v, w)
        goal_state:   [3] (x, y, yaw)
        """
        if not self.enabled or goal_state is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        trajectories = trajectories.to(self.device, self.dtype)

        K, T_plus_1, _ = trajectories.shape

        # Single sync: robot & goal positions to python floats
        (rx, ry), (gx, gy) = torch.stack(
            (robot_state[:2], goal_state[:2])).tolist()

        # Carrot from the waypoint chain (pure float math), then rate-limited
        raw_carrot = self._compute_lookahead(rx, ry, gx, gy)
        cx, cy = self._rate_limit_carrot(raw_carrot)

        # Visualization cache (cpu tensors, no GPU roundtrip)
        dx, dy = cx - rx, cy - ry
        d = math.hypot(dx, dy) + 1e-9
        self.last_lookahead_point = torch.tensor([cx, cy])
        self.last_target_direction = torch.tensor([dx / d, dy / d])
        self.last_lookahead_yaw = torch.tensor(math.atan2(dy, dx))

        # Distance-to-carrot cost, time-weighted (later poses matter more),
        # hinged so poses within tolerance cost nothing
        carrot = torch.tensor([cx, cy], device=self.device, dtype=self.dtype)
        traj_positions = trajectories[:, :, :2]                     # [K, T+1, 2]
        weights = torch.linspace(0.3, 1.0, steps=T_plus_1, device=self.device, dtype=self.dtype)
        weights = weights / weights.sum()
        distances = torch.norm(traj_positions - carrot.view(1, 1, 2), dim=2)
        hinge_d = torch.clamp(distances - self.xy_goal_tolerance, min=0.0)
        lookahead_cost = (hinge_d * weights.view(1, -1)).sum(dim=1)  # [K]

        total_cost = self.distance_scale * lookahead_cost

        # Optional progress reward: reduce cost by net approach to the goal
        if self.use_progress_reward:
            goal_pos = goal_state[:2].to(self.device, self.dtype)
            init_d = torch.norm(traj_positions[:, 0, :] - goal_pos.view(1, 2), dim=1)
            final_d = torch.norm(traj_positions[:, -1, :] - goal_pos.view(1, 2), dim=1)
            total_cost = total_cost + self.progress_scale * (final_d - init_d)

        return self.apply_weight(total_cost)

    # ------------------------------------------------------------------
    # Carrot computation (pure python floats)
    # ------------------------------------------------------------------

    def _static_lookahead(self) -> float:
        return max(self.lookahead_min_distance,
                   min(self.lookahead_max_distance, self.lookahead_base_distance))

    def _rate_limit_carrot(self, raw: Tuple[float, float]) -> Tuple[float, float]:
        """
        Limit carrot displacement per cycle so behavior transitions and goal
        switches move it smoothly. Snaps on path change.
        """
        if self.lookahead_max_step <= 0.0:
            return raw

        path_id = None
        if self.multiple_waypoints is not None:
            path_id = getattr(self.multiple_waypoints, 'path_id', None)

        if self._carrot is None or path_id != self._carrot_path_id:
            self._carrot = raw
            self._carrot_path_id = path_id
            return raw

        dx = raw[0] - self._carrot[0]
        dy = raw[1] - self._carrot[1]
        step = math.hypot(dx, dy)
        if step > self.lookahead_max_step:
            scale = self.lookahead_max_step / step
            self._carrot = (self._carrot[0] + dx * scale, self._carrot[1] + dy * scale)
        else:
            self._carrot = raw
        return self._carrot

    def _compute_lookahead(self, rx: float, ry: float,
                           gx: float, gy: float) -> Tuple[float, float]:
        if self.use_multiple_waypoints and self.multiple_waypoints is not None:
            return self._compute_waypoint_chain_lookahead(rx, ry)
        return self._compute_single_goal_lookahead(rx, ry, gx, gy)

    def _compute_single_goal_lookahead(self, rx: float, ry: float,
                                       gx: float, gy: float) -> Tuple[float, float]:
        lookahead = self._static_lookahead()
        dx, dy = gx - rx, gy - ry
        d = math.hypot(dx, dy) + 1e-9
        if d <= lookahead:
            return (gx, gy)
        return (rx + dx / d * lookahead, ry + dy / d * lookahead)

    @staticmethod
    def _get_behavior_group(node_type: int) -> int:
        """Map node types to behavior groups; lookahead never extends
        across a group boundary so behavior changes happen at the node"""
        if node_type in (2, 4):
            return 1
        if node_type in (1, 3, 5, 6, 9):
            return 2
        if node_type in (7, 8, 10):
            return 3
        return 0

    def _compute_waypoint_chain_lookahead(self, rx: float, ry: float) -> Tuple[float, float]:
        wp = self.multiple_waypoints

        if not hasattr(wp, "current_goal") or wp.current_goal is None:
            return (rx, ry)

        cgx = wp.current_goal.pose.position.x
        cgy = wp.current_goal.pose.position.y

        current_node_type = getattr(wp, "current_goal_node_type", 1)
        current_behavior_group = self._get_behavior_group(current_node_type)

        behavior_changed = (
            self.previous_node_type is not None and
            current_behavior_group != self._get_behavior_group(self.previous_node_type))
        self.previous_node_type = current_node_type

        # On a behavior change, track the boundary node itself
        # (the rate limiter walks the carrot there smoothly)
        if behavior_changed:
            return (cgx, cgy)

        total_lookahead = self._static_lookahead()
        next_wps = getattr(wp, "next_waypoints", None) or []

        dx, dy = cgx - rx, cgy - ry
        d_cur = math.hypot(dx, dy)
        if d_cur > total_lookahead:
            return (rx + dx / (d_cur + 1e-9) * total_lookahead,
                    ry + dy / (d_cur + 1e-9) * total_lookahead)

        # Close to the current goal: extend along next waypoints,
        # but never across a behavior-group boundary
        next_node_types = getattr(wp, "next_waypoints_node_types", None) or []
        if len(next_wps) == 0:
            return (cgx, cgy)
        if len(next_node_types) > 0 and \
                self._get_behavior_group(next_node_types[0]) != current_behavior_group:
            return (cgx, cgy)

        return self._extend_through_waypoints(
            (cgx, cgy), total_lookahead - d_cur, next_wps, next_node_types,
            current_behavior_group)

    def _extend_through_waypoints(self, start_pos: Tuple[float, float],
                                  remaining_distance: float,
                                  next_waypoints: list, next_node_types: list,
                                  current_behavior_group: int) -> Tuple[float, float]:
        cx, cy = start_pos
        remaining = float(remaining_distance)

        for i, waypoint in enumerate(next_waypoints):
            if i < len(next_node_types) and \
                    self._get_behavior_group(next_node_types[i]) != current_behavior_group:
                return (cx, cy)  # stop at behavior boundary

            nx = waypoint.pose.position.x
            ny = waypoint.pose.position.y
            sx, sy = nx - cx, ny - cy
            seg_len = math.hypot(sx, sy)

            if remaining <= seg_len:
                if seg_len > 1e-9:
                    return (cx + sx / seg_len * remaining, cy + sy / seg_len * remaining)
                return (cx, cy)
            remaining -= seg_len
            cx, cy = nx, ny

        return (cx, cy)
