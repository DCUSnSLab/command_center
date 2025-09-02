#!/usr/bin/env python3
import math
import torch
from typing import Optional, Any
from .base_critic import BaseCritic


class GoalCritic(BaseCritic):
    """
    Goal tracking critic for navigation (SMPPI)
    """

    def __init__(self, params: dict):
        super().__init__("GoalCritic", params)

        # Goal tracking parameters
        self.xy_goal_tolerance = params.get('xy_goal_tolerance', 0.25)
        self.yaw_goal_tolerance = params.get('yaw_goal_tolerance', 0.25)

        # Scales (actually used in total cost)
        self.distance_scale = params.get('distance_scale', 1.0)
        self.angle_scale = params.get('angle_scale', 1.0)
        self.alignment_scale = params.get('alignment_scale', 1.0)
        self.progress_scale = params.get('progress_scale', 0.0)  # default off

        # Progress reward toggle
        self.use_progress_reward = params.get('use_progress_reward', False)

        # Lookahead parameters
        self.lookahead_base_distance = params.get('lookahead_base_distance', 2.5)
        self.lookahead_velocity_factor = params.get('lookahead_velocity_factor', 1.2)
        self.lookahead_min_distance = params.get('lookahead_min_distance', 1.0)
        self.lookahead_max_distance = params.get('lookahead_max_distance', 6.0)

        # Behavior options
        self.use_multiple_waypoints = params.get('use_multiple_waypoints', True)
        self.respect_reverse_heading = params.get('respect_reverse_heading', True)
        self.yaw_blend_distance = params.get('yaw_blend_distance', 1.5)  # near-goal 헤딩 블렌딩

        # Debug
        self.debug = params.get('debug', False)
        self.debug_level = params.get('debug_level', 1)

        self.multiple_waypoints = None
        print(f"[GoalCritic] xy_tol={self.xy_goal_tolerance}, yaw_tol={self.yaw_goal_tolerance}")
        print(f"[GoalCritic] Lookahead: base={self.lookahead_base_distance}, vel_fac={self.lookahead_velocity_factor}, "
              f"range=[{self.lookahead_min_distance}-{self.lookahead_max_distance}]")
        print(f"[GoalCritic] Multi-waypoints: {self.use_multiple_waypoints}, reverse_heading: {self.respect_reverse_heading}")

    @staticmethod
    def _relu(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.)

    @staticmethod
    def _huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        # smooth L1
        absx = torch.abs(x)
        quad = torch.minimum(absx, torch.tensor(delta, device=x.device, dtype=x.dtype))
        lin = absx - quad
        return 0.5 * quad * quad + delta * lin

    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                     robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                     obstacles: Optional[Any]) -> torch.Tensor:
        """
        trajectories: [K, T+1, 3] (x, y, theta)
        controls:     [K, T, 2]   (unused here but kept for interface)
        robot_state:  [5] e.g., (x, y, yaw, v, w)
        goal_state:   [3] (x, y, yaw)
        """
        if not self.enabled or goal_state is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        # --- Device/dtype alignment
        trajectories = trajectories.to(self.device, self.dtype)
        robot_state = robot_state.to(self.device, self.dtype)
        goal_state = goal_state.to(self.device, self.dtype)

        K, T_plus_1, _ = trajectories.shape
        current_pos = robot_state[:2]                      # [2]
        v_scalar = robot_state[3]                          # signed speed
        v_abs = torch.abs(v_scalar)
        goal_pos = goal_state[:2]                          # [2]

        # --- Lookahead point (multi-waypoints aware)
        lookahead_point = self._compute_multiple_waypoints_lookahead(current_pos, v_abs, goal_pos)
        # store for viz without holding the graph
        self.last_lookahead_point = lookahead_point.detach().cpu()

        # --- Target direction / yaw (with reverse-aware option)
        lookahead_vec = lookahead_point - current_pos
        lookahead_dist = torch.norm(lookahead_vec) + 1e-9
        target_direction = lookahead_vec / lookahead_dist
        target_yaw = torch.atan2(target_direction[1], target_direction[0])

        if self.respect_reverse_heading and (v_scalar < 0):
            target_yaw = self.normalize_angle(target_yaw + math.pi)

        # Near-goal: blend target yaw towards desired final yaw to avoid flipping
        goal_vec = goal_pos - current_pos
        dist_to_goal = torch.norm(goal_vec)
        if (dist_to_goal < self.yaw_blend_distance):
            alpha = (self.yaw_blend_distance - dist_to_goal) / self.yaw_blend_distance  # 0..1
            blended = self.normalize_angle(goal_state[2])
            # slerp-like in angle space
            dyaw = self.normalize_angle(blended - target_yaw)
            target_yaw = self.normalize_angle(target_yaw + alpha * dyaw)

        # --- Trajectory slices
        traj_positions = trajectories[:, :, :2]  # [K, T+1, 2]
        traj_yaws = trajectories[:, :, 2]        # [K, T+1]

        # 1) Lookahead tracking cost (weighted)
        weights = torch.linspace(0.3, 1.0, steps=T_plus_1, device=self.device, dtype=self.dtype)
        weights = weights / weights.sum()
        distances_to_lookahead = torch.norm(traj_positions - lookahead_point.view(1, 1, 2), dim=2)  # [K, T+1]

        # hinge on tolerance -> inside tol => 0
        hinge_d = self._relu(distances_to_lookahead - self.xy_goal_tolerance)
        lookahead_cost = (hinge_d * weights.view(1, -1)).sum(dim=1)  # [K]

        # 2) Heading alignment (final yaw vs target_yaw), hinge on yaw tolerance
        final_yaws = traj_yaws[:, -1]  # [K]
        yaw_errors = torch.abs(self.normalize_angle(final_yaws - target_yaw))
        hinge_yaw = self._relu(yaw_errors - self.yaw_goal_tolerance)
        heading_cost = self._huber(hinge_yaw, delta=0.5)  # smoother than square

        # 3) Path alignment cost (step-wise cosine alignment to target direction)
        if T_plus_1 > 1:
            steps = traj_positions[:, 1:, :] - traj_positions[:, :-1, :]  # [K, T, 2]
            step_norms = torch.norm(steps, dim=2, keepdim=True).clamp_min(1e-6)
            step_dirs = steps / step_norms
            alignment = torch.sum(step_dirs * target_direction.view(1, 1, 2), dim=2)  # cos in [-1,1]
            # 1 - cos -> [0,2], use huber for smoothness
            alignment_cost = self._huber(1.0 - alignment).mean(dim=1)
        else:
            alignment_cost = torch.zeros(K, device=self.device, dtype=self.dtype)

        # 4) (옵션) Progress reward: 가까워질수록 비용을 깎음
        progress_term = torch.zeros(K, device=self.device, dtype=self.dtype)
        if self.use_progress_reward:
            init_d = torch.norm(traj_positions[:, 0, :] - goal_pos.view(1, 2), dim=1)
            final_d = torch.norm(traj_positions[:, -1, :] - goal_pos.view(1, 2), dim=1)
            progress = init_d - final_d  # >0 이면 진전
            progress_term = -progress  # 비용에 더하므로, 진전이 크면 더 작은 비용

        # --- Combine with scales
        total_cost = (
            self.distance_scale * lookahead_cost +
            self.angle_scale    * heading_cost +
            self.alignment_scale * alignment_cost +
            self.progress_scale  * progress_term
        )

        return self.apply_weight(total_cost)

    def _compute_nav2_lookahead(self, current_pos: torch.Tensor, current_vel_abs: torch.Tensor,
                                goal_pos: torch.Tensor) -> torch.Tensor:
        base_offset = self.lookahead_base_distance
        velocity_offset = self.lookahead_velocity_factor * current_vel_abs
        total_lookahead = torch.clamp(base_offset + velocity_offset,
                                      self.lookahead_min_distance,
                                      self.lookahead_max_distance)
        goal_vec = goal_pos - current_pos
        goal_distance = torch.norm(goal_vec) + 1e-9

        if (goal_distance <= total_lookahead).item():
            return goal_pos
        goal_dir = goal_vec / goal_distance
        return current_pos + goal_dir * total_lookahead

    def get_lookahead_point(self) -> Optional[torch.Tensor]:
        return getattr(self, 'last_lookahead_point', None)

    def set_multiple_waypoints(self, waypoints_msg):
        self.multiple_waypoints = waypoints_msg

    def _compute_multiple_waypoints_lookahead(self, current_pos: torch.Tensor, current_vel_abs: torch.Tensor,
                                              goal_pos: torch.Tensor) -> torch.Tensor:
        if self.use_multiple_waypoints and self.multiple_waypoints is not None:
            return self._compute_waypoint_chain_lookahead(current_pos, current_vel_abs)
        return self._compute_nav2_lookahead(current_pos, current_vel_abs, goal_pos)

    def _compute_waypoint_chain_lookahead(self, current_pos: torch.Tensor, current_vel_abs: torch.Tensor) -> torch.Tensor:
        wp = self.multiple_waypoints

        # 방어적 가드
        if not hasattr(wp, "current_goal") or wp.current_goal is None:
            return current_pos  # 정보 없으면 현재 위치 반환

        current_goal_pos = torch.tensor([
            wp.current_goal.pose.position.x,
            wp.current_goal.pose.position.y
        ], device=self.device, dtype=self.dtype)

        total_lookahead = torch.clamp(
            self.lookahead_base_distance + self.lookahead_velocity_factor * current_vel_abs,
            self.lookahead_min_distance,
            self.lookahead_max_distance
        )

        d_cur = torch.norm(current_pos - current_goal_pos)

        if (d_cur > total_lookahead).item():
            direction = (current_goal_pos - current_pos) / (d_cur + 1e-9)
            return current_pos + direction * total_lookahead

        # 가까우면 next_waypoints로 연장
        next_wps = getattr(wp, "next_waypoints", None) or []
        if len(next_wps) == 0:
            return current_goal_pos

        remaining = float((total_lookahead - d_cur).item())
        return self._extend_lookahead_through_waypoints(current_goal_pos, remaining, next_wps)

    def _extend_lookahead_through_waypoints(self, start_pos: torch.Tensor, remaining_distance: float, next_waypoints: list) -> torch.Tensor:
        current = start_pos
        remaining = float(remaining_distance)

        for waypoint in next_waypoints:
            next_pos = torch.tensor(
                [waypoint.pose.position.x, waypoint.pose.position.y],
                device=self.device, dtype=self.dtype
            )
            seg_vec = next_pos - current
            seg_len = float(torch.norm(seg_vec).item())

            if remaining <= seg_len:
                if seg_len > 1e-9:
                    direction = seg_vec / (seg_len + 1e-9)
                    return current + direction * remaining
                else:
                    return current
            else:
                remaining -= seg_len
                current = next_pos

        # 모든 세그먼트를 지나도 남으면 마지막 점
        return current

    def is_goal_reached(self, current_pose: torch.Tensor, goal_state: torch.Tensor) -> bool:
        if goal_state is None:
            return False
        dist = torch.norm(current_pose[:2] - goal_state[:2])
        ang = abs(self.normalize_angle(current_pose[2] - goal_state[2]))
        return (dist < self.xy_goal_tolerance) and (ang < self.yaw_goal_tolerance)

    def update_parameters(self, params: dict):
        # 토러런스/스케일/옵션 갱신
        self.xy_goal_tolerance = params.get('xy_goal_tolerance', self.xy_goal_tolerance)
        self.yaw_goal_tolerance = params.get('yaw_goal_tolerance', self.yaw_goal_tolerance)
        self.distance_scale = params.get('distance_scale', self.distance_scale)
        self.angle_scale = params.get('angle_scale', self.angle_scale)
        self.alignment_scale = params.get('alignment_scale', self.alignment_scale)
        self.progress_scale = params.get('progress_scale', self.progress_scale)
        self.use_progress_reward = params.get('use_progress_reward', self.use_progress_reward)
        self.respect_reverse_heading = params.get('respect_reverse_heading', self.respect_reverse_heading)
        self.yaw_blend_distance = params.get('yaw_blend_distance', self.yaw_blend_distance)
        self.lookahead_base_distance = params.get('lookahead_base_distance', self.lookahead_base_distance)
        self.lookahead_velocity_factor = params.get('lookahead_velocity_factor', self.lookahead_velocity_factor)
        self.lookahead_min_distance = params.get('lookahead_min_distance', self.lookahead_min_distance)
        self.lookahead_max_distance = params.get('lookahead_max_distance', self.lookahead_max_distance)
        self.use_multiple_waypoints = params.get('use_multiple_waypoints', self.use_multiple_waypoints)
        print("[GoalCritic] Parameters updated")
