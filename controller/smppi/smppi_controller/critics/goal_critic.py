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
from .._verbose import vprint


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
        self.respect_reverse_heading = params.get('respect_reverse_heading', False)
        # 회두 항 (compute_cost 말미 참조 — 목표가 등 뒤일 때만 활성)
        self.turnaround_enabled = params.get('turnaround_enabled', True)
        self.turnaround_on_deg = params.get('turnaround_on_deg', 100.0)
        self.turnaround_off_deg = params.get('turnaround_off_deg', 55.0)
        self.turnaround_weight = params.get('turnaround_weight', 2.0)
        self.yaw_blend_distance = params.get('yaw_blend_distance', 1.5)  # near-goal 헤딩 블렌딩

        # Debug
        self.debug = params.get('debug', False)
        self.debug_level = params.get('debug_level', 1)

        # State
        self.multiple_waypoints = None
        self.previous_node_type = None
        self._carrot: Optional[Tuple[float, float]] = None
        self._carrot_path_id: Optional[str] = None

        # Visualization cache
        self.last_lookahead_point: Optional[torch.Tensor] = None
        self.last_lookahead_yaw: Optional[torch.Tensor] = None
        self.last_target_direction: Optional[torch.Tensor] = None

        vprint(f"[GoalCritic] lookahead={self._static_lookahead():.2f}m (static), "
              f"max_step={self.lookahead_max_step}m/cycle")

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
        
        # === DIAGNOSTIC LOGGING: Track lookahead changes ===
        prev_lookahead = getattr(self, '_prev_lookahead_point', None)
        if prev_lookahead is not None:
            prev_lookahead_tensor = torch.tensor(prev_lookahead, device=self.device, dtype=self.dtype)
            lookahead_jump = float(torch.norm(lookahead_point - prev_lookahead_tensor).detach().cpu().item())
            # if lookahead_jump > 0.3:  # 더 민감하게 감지 (0.3m 이상)
            #     print(f"🚨 [LOOKAHEAD JUMP] {lookahead_jump:.3f}m")
            #     print(f"   From: {prev_lookahead}")
            #     print(f"   To:   {lookahead_point.detach().cpu().numpy()}")
        
        # Store current lookahead for next iteration
        self._prev_lookahead_point = lookahead_point.detach().cpu().numpy()
        
        # --- SIMPLIFIED: Distance-only tracking (target_yaw disabled for stability) ---
        lookahead_vec = lookahead_point - current_pos
        lookahead_dist = torch.norm(lookahead_vec) + 1e-9
        target_direction = lookahead_vec / lookahead_dist
        
        # === COMMENTED OUT: Target yaw calculation (causing oscillation) ===
        # target_yaw = torch.atan2(target_direction[1], target_direction[0])
        # 
        # # waypoint Pose 관련 lookahead 디버깅 블록  
        # # Apply reverse heading when respect_reverse_heading is True (reverse mode)
        # if self.respect_reverse_heading:
        #     target_yaw = self.normalize_angle(target_yaw + math.pi)
        #
        # # Near-goal: blend target yaw towards desired final yaw to avoid flipping
        # # 이거 파라미터 때문에 안쓰고있을수도
        # goal_vec = goal_pos - current_pos
        # dist_to_goal = torch.norm(goal_vec)
        # if (dist_to_goal < self.yaw_blend_distance):
        #     alpha = (self.yaw_blend_distance - dist_to_goal) / self.yaw_blend_distance  # 0..1
        #     blended = self.normalize_angle(goal_state[2])
        #     # slerp-like in angle space
        #     dyaw = self.normalize_angle(blended - target_yaw)
        #     target_yaw = self.normalize_angle(target_yaw + alpha * dyaw)
        
        # Dummy target_yaw for visualization (current robot yaw)
        target_yaw = robot_state[2] if robot_state is not None else torch.zeros(1, device=self.device, dtype=self.dtype)

        # store lookahead point, yaw and target direction for viz without holding the graph
        self.last_lookahead_point = lookahead_point.detach().cpu()
        self.last_lookahead_yaw = target_yaw.detach().cpu()
        self.last_target_direction = target_direction.detach().cpu()

        # --- Trajectory slices
        traj_positions = trajectories[:, :, :2]  # [K, T+1, 2]
        traj_yaws = trajectories[:, :, 2]        # [K, T+1]

        # 1) Lookahead tracking cost (weighted)
        weights = torch.linspace(0.3, 1.0, steps=T_plus_1, device=self.device, dtype=self.dtype)
        weights = weights / weights.sum()
        distances_to_lookahead = torch.norm(traj_positions - lookahead_point.view(1, 1, 2), dim=2)  # [K, T+1]

        # hinge on tolerance -> inside tol => 0
        # xy_goal 점검 가까우면 0으로 줌
        hinge_d = self._relu(distances_to_lookahead - self.xy_goal_tolerance)
        lookahead_cost = (hinge_d * weights.view(1, -1)).sum(dim=1)  # [K]

        # === COMMENTED OUT: Heading alignment cost (causing oscillation) ===
        # 2) Heading alignment (final yaw vs target_yaw), hinge on yaw tolerance
        # # self.yaw_goal_tolerance도 체크해봐야될듯
        # final_yaws = traj_yaws[:, -1]  # [K]
        # yaw_errors = torch.abs(self.normalize_angle(final_yaws - target_yaw))
        # hinge_yaw = self._relu(yaw_errors - self.yaw_goal_tolerance)
        # heading_cost = self._huber(hinge_yaw, delta=0.5)  # smoother than square
        
        # DISABLED: Set heading cost to zero for pure distance tracking
        heading_cost = torch.zeros(K, device=self.device, dtype=self.dtype)

        # === COMMENTED OUT: Path alignment cost (causing oscillation) ===
        # 3) Path alignment cost (step-wise cosine alignment to target direction)
        # # 필요한지 체크
        # if T_plus_1 > 1:
        #     steps = traj_positions[:, 1:, :] - traj_positions[:, :-1, :]  # [K, T, 2]
        #     step_norms = torch.norm(steps, dim=2, keepdim=True).clamp_min(1e-6)
        #     step_dirs = steps / step_norms
        #     alignment = torch.sum(step_dirs * target_direction.view(1, 1, 2), dim=2)  # cos in [-1,1]
        #     # 1 - cos -> [0,2], use huber for smoothness
        #     alignment_cost = self._huber(1.0 - alignment).mean(dim=1)
        # else:
        #     alignment_cost = torch.zeros(K, device=self.device, dtype=self.dtype)
        
        # DISABLED: Set alignment cost to zero for pure distance tracking
        alignment_cost = torch.zeros(K, device=self.device, dtype=self.dtype)

        # 4) (옵션) Progress reward: 가까워질수록 비용을 깎음
        # 필요한지 체크
        progress_term = torch.zeros(K, device=self.device, dtype=self.dtype)
        if self.use_progress_reward:
            print("이거씀??????????????????????????????????")
            init_d = torch.norm(traj_positions[:, 0, :] - goal_pos.view(1, 2), dim=1)
            final_d = torch.norm(traj_positions[:, -1, :] - goal_pos.view(1, 2), dim=1)
            progress = init_d - final_d  # >0 이면 진전
            progress_term = -progress  # 비용에 더하므로, 진전이 크면 더 작은 비용

        # --- Combine with scales (SIMPLIFIED: distance-only) ---
        distance_term = self.distance_scale * lookahead_cost
        # angle_term = self.angle_scale * heading_cost  # DISABLED
        # alignment_term = self.alignment_scale * alignment_cost  # DISABLED
        progress_term_scaled = self.progress_scale * progress_term
        
        # === DIAGNOSTIC LOGGING: Track distance cost changes ===
        current_distance_cost = float(distance_term.mean().detach().cpu().item())
        prev_distance_cost = getattr(self, '_prev_distance_cost', current_distance_cost)
        distance_cost_change = abs(current_distance_cost - prev_distance_cost)
        
        # if distance_cost_change > 10.0:  # Significant cost change
        #     print(f"💥 [DISTANCE COST JUMP] {distance_cost_change:.3f}")
        #     print(f"   From: {prev_distance_cost:.3f} -> To: {current_distance_cost:.3f}")
        #     print(f"   Min distance to lookahead: {float(distances_to_lookahead.min().detach().cpu().item()):.3f}m")
        
        # Store for next iteration
        self._prev_distance_cost = current_distance_cost
        
        # SIMPLIFIED: Only distance and progress terms active
        total_cost = distance_term + progress_term_scaled

        # --- 회두(turn-around) 항: 목표가 '크게 뒤'에 있을 때만 켠다 ----------
        # 거리-전용 비용의 구조적 함정: 전진 전용 속도범위에서 목표가 등 뒤면
        # 어떤 전진 호도 당장은 거리를 늘리므로 '정지'가 국소최소가 된다
        # (2026-08-02 챔버 D1 실측: Cmd v=0 ω=0 로 400 s 동결). 위에서 주석
        # 처리된 전역 헤딩 항은 정상 추종 중 진동을 일으켜 꺼진 것이므로
        # 되살리지 않는다 — 대신 게이트를 둔다:
        #   · 켜짐: 현재 로봇 기준 목표 방위각 오차 > turnaround_on (기본 100도)
        #   · 꺼짐: 오차 < turnaround_off (기본 55도, 히스테리시스로 채터링 방지)
        # 게이트는 '현재 상태'로만 판정하므로 한 사이클의 K개 샘플이 전부 같은
        # 항을 받는다(샘플 간 불연속 없음). 정상 추종 영역(목표가 전방)에서는
        # 이 항이 0이라 기존 거동이 완전히 보존된다.
        if getattr(self, 'turnaround_enabled', True):
            on_rad = math.radians(getattr(self, 'turnaround_on_deg', 100.0))
            off_rad = math.radians(getattr(self, 'turnaround_off_deg', 55.0))
            w_turn = getattr(self, 'turnaround_weight', 2.0)
            bearing = torch.atan2(lookahead_point[1] - current_pos[1],
                                  lookahead_point[0] - current_pos[0])
            err_now = torch.abs(torch.atan2(
                torch.sin(bearing - robot_state[2]),
                torch.cos(bearing - robot_state[2])))
            active = getattr(self, '_turnaround_active', False)
            if err_now > on_rad:
                active = True
            elif err_now < off_rad:
                active = False
            self._turnaround_active = active
            if active:
                final_yaws = traj_yaws[:, -1]
                yaw_err = torch.abs(torch.atan2(torch.sin(final_yaws - bearing),
                                                torch.cos(final_yaws - bearing)))
                total_cost = total_cost + w_turn * self._huber(yaw_err, delta=0.5)
        
        # === DEBUG OUTPUT (simplified for distance-only tracking) ===
        lookahead_cpu = lookahead_point.detach().cpu().numpy()
        robot_pos_cpu = current_pos.detach().cpu().numpy()
        robot_to_lookahead_dist = float(torch.norm(lookahead_point - current_pos).detach().cpu().item())
        
        # Regular debug output (reduced frequency)
        debug_counter = getattr(self, '_debug_counter', 0) + 1
        self._debug_counter = debug_counter
        
        # if debug_counter % 10 == 0:  # Every 10th call
        #     print(f"🎯 [LOOKAHEAD DEBUG] robot: {np.round(robot_pos_cpu, 3)}")
        #     print(f"   lookahead_point: {np.round(lookahead_cpu, 3)}")
        #     print(f"   calculated_distance: {robot_to_lookahead_dist:.3f}m")
        #     print(f"   current_velocity: {float(v_abs):.3f}m/s")
        #     print(f"   distance_cost: {current_distance_cost:.3f}")
        #     print(f"   min_traj_dist_to_lookahead: {float(distances_to_lookahead.min().detach().cpu().item()):.3f}m")

        return self.apply_weight(total_cost)

    def _compute_nav2_lookahead(self, current_pos: torch.Tensor, current_vel_abs: torch.Tensor,
                                goal_pos: torch.Tensor) -> torch.Tensor:
        base_offset = self.lookahead_base_distance
        velocity_offset = self.lookahead_velocity_factor * current_vel_abs
        base_lookahead = torch.clamp(base_offset + velocity_offset,
                                   self.lookahead_min_distance,
                                   self.lookahead_max_distance)

        # Debug output for lookahead calculation
        debug_counter = getattr(self, '_nav2_debug_counter', 0) + 1
        self._nav2_debug_counter = debug_counter

        # if debug_counter % 20 == 0:  # Every 20th call
        #     print(f"📏 [NAV2 LOOKAHEAD] vel: {float(current_vel_abs):.3f}m/s")
        #     print(f"   base_offset: {base_offset:.3f}m")
        #     print(f"   velocity_offset: {float(velocity_offset):.3f}m")
        #     print(f"   base_lookahead: {float(base_lookahead):.3f}m (after clamp)")
        #     print(f"   range: [{self.lookahead_min_distance:.3f}, {self.lookahead_max_distance:.3f}]m")

        # Apply curve-based lookahead adjustment for single goal case
        # (For simplicity, we treat single goal as a 2-point path)
        total_lookahead = self._adjust_lookahead_for_curves(
            base_lookahead, current_pos, goal_pos, []
        )

        if debug_counter % 20 == 0 and abs(float(total_lookahead) - float(base_lookahead)) > 0.1:
            print(f"   curve_adjusted: {float(base_lookahead):.3f} -> {float(total_lookahead):.3f}m")

        goal_vec = goal_pos - current_pos
        goal_distance = torch.norm(goal_vec) + 1e-9

        if (goal_distance <= total_lookahead).item():
            return goal_pos
        goal_dir = goal_vec / goal_distance
        return current_pos + goal_dir * total_lookahead

    def get_lookahead_point(self) -> Optional[torch.Tensor]:
        return getattr(self, 'last_lookahead_point', None)
    
    def get_lookahead_yaw(self) -> Optional[torch.Tensor]:
        return getattr(self, 'last_lookahead_yaw', None)
    
    def get_target_direction(self) -> Optional[torch.Tensor]:
        return getattr(self, 'last_target_direction', None)

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
        vprint("[GoalCritic] Parameters updated")

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
