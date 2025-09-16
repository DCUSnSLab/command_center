#!/usr/bin/env python3
import math
import torch
from typing import Optional, Any
from .base_critic import BaseCritic
import numpy as np

class GoalCritic(BaseCritic):
    """
    Goal tracking critic for navigation (SMPPI)
    """

    def __init__(self, params: dict):
        super().__init__("GoalCritic", params)

        # Goal tracking parameters
        self.xy_goal_tolerance = params.get('xy_goal_tolerance', 0.05)
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

        # Curve-based lookahead parameters
        self.curve_detection_enabled = params.get('curve_detection_enabled', True)
        self.curve_angle_threshold = params.get('curve_angle_threshold', 30.0)  # degrees
        self.curve_lookahead_reduction_factor = params.get('curve_lookahead_reduction_factor', 0.3)  # 30% of normal
        self.curve_min_lookahead = params.get('curve_min_lookahead', 0.5)  # minimum lookahead in curves
        self.curve_detection_distance = params.get('curve_detection_distance', 3.0)  # distance to look ahead for curve detection

        # Phase 1: Basic lookahead extension parameters
        self.lookahead_extension_enabled = params.get('lookahead_extension_enabled', True)
        self.lookahead_extension_distance = params.get('lookahead_extension_distance', 1.0)  # 1m extension

        # Behavior options
        self.use_multiple_waypoints = params.get('use_multiple_waypoints', True)
        self.respect_reverse_heading = params.get('respect_reverse_heading', False)
        self.yaw_blend_distance = params.get('yaw_blend_distance', 1.5)  # near-goal 헤딩 블렌딩

        # Debug
        self.debug = params.get('debug', False)
        self.debug_level = params.get('debug_level', 1)

        self.multiple_waypoints = None
        self.previous_node_type = None  # Track previous node type for behavior change detection
        print(f"[GoalCritic] xy_tol={self.xy_goal_tolerance}, yaw_tol={self.yaw_goal_tolerance}")
        print(f"[GoalCritic] Lookahead: base={self.lookahead_base_distance}, vel_fac={self.lookahead_velocity_factor}, "
              f"range=[{self.lookahead_min_distance}-{self.lookahead_max_distance}]")
        print(f"[GoalCritic] Multi-waypoints: {self.use_multiple_waypoints}, reverse_heading: {self.respect_reverse_heading}")
        print(f"[GoalCritic] Curve detection: {self.curve_detection_enabled}, angle_threshold: {self.curve_angle_threshold}°, reduction: {self.curve_lookahead_reduction_factor}x")
        print(f"[GoalCritic] Lookahead extension: {self.lookahead_extension_enabled}, distance: {self.lookahead_extension_distance}m")

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

        # Store robot_state for lookahead extension functions
        self.current_robot_state = robot_state

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
            if lookahead_jump > 0.5:  # Significant jump > 0.5m
                print(f"🚨 [LOOKAHEAD JUMP] {lookahead_jump:.3f}m")
                print(f"   From: {prev_lookahead}")
                print(f"   To:   {lookahead_point.detach().cpu().numpy()}")
        
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

        # Phase 1: Apply lookahead extension
        extended_lookahead_point = self._extend_lookahead_by_vehicle_heading(lookahead_point, robot_state)

        # store extended lookahead point, yaw and target direction for viz without holding the graph
        self.last_lookahead_point = extended_lookahead_point.detach().cpu()
        self.last_lookahead_yaw = target_yaw.detach().cpu()
        self.last_target_direction = target_direction.detach().cpu()

        # Update lookahead_point for cost calculation to use extended version
        lookahead_point = extended_lookahead_point

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
        
        if distance_cost_change > 10.0:  # Significant cost change
            print(f"💥 [DISTANCE COST JUMP] {distance_cost_change:.3f}")
            print(f"   From: {prev_distance_cost:.3f} -> To: {current_distance_cost:.3f}")
            print(f"   Min distance to lookahead: {float(distances_to_lookahead.min().detach().cpu().item()):.3f}m")
        
        # Store for next iteration
        self._prev_distance_cost = current_distance_cost
        
        # SIMPLIFIED: Only distance and progress terms active
        total_cost = distance_term + progress_term_scaled
        
        # === DEBUG OUTPUT (simplified for distance-only tracking) ===
        lookahead_cpu = lookahead_point.detach().cpu().numpy()
        robot_pos_cpu = current_pos.detach().cpu().numpy()
        robot_to_lookahead_dist = float(torch.norm(lookahead_point - current_pos).detach().cpu().item())
        
        # Regular debug output (reduced frequency)
        debug_counter = getattr(self, '_debug_counter', 0) + 1
        self._debug_counter = debug_counter
        
        # if debug_counter % 10 == 0:  # Every 10th call
        #     print(f"[DISTANCE-ONLY] robot: {np.round(robot_pos_cpu, 3)}")
        #     print(f"[DISTANCE-ONLY] lookahead: {np.round(lookahead_cpu, 3)} (dist: {robot_to_lookahead_dist:.3f}m)")
        #     print(f"[DISTANCE-ONLY] distance_cost: {current_distance_cost:.3f}")
        #     print(f"[DISTANCE-ONLY] min_traj_dist_to_lookahead: {float(distances_to_lookahead.min().detach().cpu().item()):.3f}m")

        return self.apply_weight(total_cost)

    def _compute_nav2_lookahead(self, current_pos: torch.Tensor, current_vel_abs: torch.Tensor,
                                goal_pos: torch.Tensor) -> torch.Tensor:
        base_offset = self.lookahead_base_distance
        velocity_offset = self.lookahead_velocity_factor * current_vel_abs
        base_lookahead = torch.clamp(base_offset + velocity_offset,
                                   self.lookahead_min_distance,
                                   self.lookahead_max_distance)

        # Apply curve-based lookahead adjustment for single goal case
        # (For simplicity, we treat single goal as a 2-point path)
        total_lookahead = self._adjust_lookahead_for_curves(
            base_lookahead, current_pos, goal_pos, []
        )

        goal_vec = goal_pos - current_pos
        goal_distance = torch.norm(goal_vec) + 1e-9

        if (goal_distance <= total_lookahead).item():
            lookahead_point = goal_pos
        else:
            goal_dir = goal_vec / goal_distance
            lookahead_point = current_pos + goal_dir * total_lookahead

        # Phase 1: Apply lookahead extension (assuming robot_state available in context)
        if hasattr(self, 'current_robot_state') and self.current_robot_state is not None:
            lookahead_point = self._extend_lookahead_by_vehicle_heading(lookahead_point, self.current_robot_state)

        return lookahead_point

    def get_lookahead_point(self) -> Optional[torch.Tensor]:
        return getattr(self, 'last_lookahead_point', None)
    
    def get_lookahead_yaw(self) -> Optional[torch.Tensor]:
        return getattr(self, 'last_lookahead_yaw', None)
    
    def get_target_direction(self) -> Optional[torch.Tensor]:
        return getattr(self, 'last_target_direction', None)

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

        current_node_type = getattr(wp, "current_goal_node_type", 1)
        behavior_changed = (self.previous_node_type is not None and
                           current_node_type != self.previous_node_type)
        # Update previous node type
        self.previous_node_type = current_node_type

        # If behavior changed, use goal tracking only (no lookahead extension)
        if behavior_changed:
            return current_goal_pos

        # Calculate base lookahead distance
        base_lookahead = torch.clamp(
            self.lookahead_base_distance + self.lookahead_velocity_factor * current_vel_abs,
            self.lookahead_min_distance,
            self.lookahead_max_distance
        )

        # Apply curve-based lookahead adjustment
        total_lookahead = self._adjust_lookahead_for_curves(
            base_lookahead, current_pos, current_goal_pos,
            getattr(wp, "next_waypoints", [])[:3]  # Look at next 3 waypoints for curve detection
        )

        d_cur = torch.norm(current_pos - current_goal_pos)

        if (d_cur > total_lookahead).item():
            direction = (current_goal_pos - current_pos) / (d_cur + 1e-9)
            return current_pos + direction * total_lookahead

        # 가까우면 next_waypoints로 연장 (but with curve-adjusted lookahead)
        next_wps = getattr(wp, "next_waypoints", None) or []
        next_node_types = getattr(wp, "next_waypoints_node_types", None) or []

        if len(next_wps) == 0:
            return current_goal_pos

        # Check if next waypoint has different node_type (behavior change)
        if len(next_node_types) > 0 and next_node_types[0] != current_node_type:
            # Next waypoint has different behavior - keep lookahead at current goal
            return current_goal_pos

        remaining = float((total_lookahead - d_cur).item())
        lookahead_point = self._extend_lookahead_through_waypoints(current_goal_pos, remaining, next_wps, next_node_types, current_node_type)

        # Phase 1: Apply lookahead extension (robot_state should be available in compute_cost context)
        if hasattr(self, 'current_robot_state') and self.current_robot_state is not None:
            lookahead_point = self._extend_lookahead_by_vehicle_heading(lookahead_point, self.current_robot_state)

        return lookahead_point

    def _extend_lookahead_through_waypoints(self, start_pos: torch.Tensor, remaining_distance: float, 
                                          next_waypoints: list, next_node_types: list, current_node_type: int) -> torch.Tensor:
        current = start_pos
        remaining = float(remaining_distance)

        for i, waypoint in enumerate(next_waypoints):
            # Check if this waypoint has different node_type
            # If next_node_types is shorter than next_waypoints, assume same type as current for remaining waypoints
            if i < len(next_node_types):
                waypoint_node_type = next_node_types[i]
                
                if waypoint_node_type != current_node_type:
                    # Different behavior type - stop lookahead extension here
                    # Return current position (don't extend lookahead beyond behavior boundary)
                    return current
            # If no node_type info available, assume same as current (continue extending)
            
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

    def _adjust_lookahead_for_curves(self, base_lookahead: torch.Tensor, current_pos: torch.Tensor,
                                   current_goal: torch.Tensor, next_waypoints: list) -> torch.Tensor:
        """Adjust lookahead distance based on upcoming curve detection"""
        if not self.curve_detection_enabled or len(next_waypoints) == 0:
            return base_lookahead

        # Create path segments for curve analysis
        path_points = [current_pos, current_goal]

        # Add next waypoints (up to 3 for curve detection)
        for wp in next_waypoints[:3]:  # Limit to 3 waypoints to avoid excessive computation
            next_pos = torch.tensor([
                wp.pose.position.x,
                wp.pose.position.y
            ], device=self.device, dtype=self.dtype)
            path_points.append(next_pos)

        # Detect curves in the path
        max_curve_angle = self._detect_path_curvature(path_points)

        # Adjust lookahead based on curve severity
        if max_curve_angle > self.curve_angle_threshold:
            # Sharp curve detected - reduce lookahead significantly
            curve_factor = self._calculate_curve_reduction_factor(max_curve_angle)
            adjusted_lookahead = base_lookahead * curve_factor

            # Ensure minimum lookahead distance
            adjusted_lookahead = torch.clamp(
                adjusted_lookahead,
                min=self.curve_min_lookahead,
                max=base_lookahead.item()
            )

            # Debug logging for significant adjustments
            if abs(float(adjusted_lookahead) - float(base_lookahead)) > 0.5:
                print(f"[CURVE ADJUST] angle: {max_curve_angle:.1f}°, factor: {curve_factor:.2f}, "
                      f"lookahead: {float(base_lookahead):.2f}m -> {float(adjusted_lookahead):.2f}m")

            return adjusted_lookahead

        return base_lookahead

    def _detect_path_curvature(self, path_points: list) -> float:
        """Detect maximum curvature angle in the given path segments"""
        if len(path_points) < 3:
            return 0.0

        max_angle = 0.0
        debug_info = []

        for i in range(len(path_points) - 2):
            p1 = path_points[i]
            p2 = path_points[i + 1]
            p3 = path_points[i + 2]

            # Calculate vectors
            v1 = p2 - p1  # Vector from p1 to p2
            v2 = p3 - p2  # Vector from p2 to p3

            # Calculate angle between vectors
            angle_rad = self._calculate_angle_between_vectors(v1, v2)
            angle_deg = float(torch.rad2deg(torch.tensor(angle_rad)).item())

            # Debug information
            p1_np = p1.detach().cpu().numpy() if hasattr(p1, 'detach') else p1
            p2_np = p2.detach().cpu().numpy() if hasattr(p2, 'detach') else p2
            p3_np = p3.detach().cpu().numpy() if hasattr(p3, 'detach') else p3
            v1_np = v1.detach().cpu().numpy() if hasattr(v1, 'detach') else v1
            v2_np = v2.detach().cpu().numpy() if hasattr(v2, 'detach') else v2

            debug_info.append({
                'segment': i,
                'p1': p1_np, 'p2': p2_np, 'p3': p3_np,
                'v1': v1_np, 'v2': v2_np,
                'angle': angle_deg
            })

            max_angle = max(max_angle, angle_deg)

        # Debug output for the first few calls to understand what's happening
        debug_counter = getattr(self, '_curve_debug_counter', 0) + 1
        self._curve_debug_counter = debug_counter

        if debug_counter % 50 == 0:  # Every 50th call
            print(f"[CURVE DEBUG] max_angle: {max_angle:.1f}°")
            for info in debug_info:
                print(f"  Seg{info['segment']}: P1{info['p1']} -> P2{info['p2']} -> P3{info['p3']}")
                print(f"    V1{info['v1']}, V2{info['v2']}, angle: {info['angle']:.1f}°")

        return max_angle

    def _calculate_angle_between_vectors(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """Calculate angle between two vectors in radians"""
        # Normalize vectors
        v1_norm = v1 / (torch.norm(v1) + 1e-9)
        v2_norm = v2 / (torch.norm(v2) + 1e-9)

        # Calculate dot product
        dot_product = torch.dot(v1_norm, v2_norm)

        # Clamp to avoid numerical issues
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Calculate angle between vectors
        # angle_rad = 0 means vectors are aligned (straight)
        # angle_rad = π means vectors are opposite (U-turn)
        angle_rad = torch.acos(dot_product)

        # Return the deflection angle (0 = straight, π = U-turn)
        return float(angle_rad.item())

    def _calculate_curve_reduction_factor(self, curve_angle_deg: float) -> float:
        """Calculate lookahead reduction factor based on curve angle"""
        if curve_angle_deg <= self.curve_angle_threshold:
            return 1.0  # No reduction for gentle curves

        # Progressive reduction based on curve severity
        # 30° -> 1.0x, 90° -> 0.3x (curve_lookahead_reduction_factor), 180° -> 0.1x
        angle_ratio = (curve_angle_deg - self.curve_angle_threshold) / (180.0 - self.curve_angle_threshold)
        angle_ratio = min(1.0, max(0.0, angle_ratio))  # Clamp to [0, 1]

        # Interpolate between 1.0 and curve_lookahead_reduction_factor
        min_factor = max(0.1, self.curve_lookahead_reduction_factor)  # Minimum 10% of original
        reduction_factor = 1.0 - angle_ratio * (1.0 - min_factor)

        return reduction_factor

    def _get_vehicle_heading_direction(self, robot_state: torch.Tensor) -> torch.Tensor:
        """Get current vehicle heading direction as unit vector"""
        if robot_state is None or len(robot_state) < 3:
            # Fallback: return forward direction
            return torch.tensor([1.0, 0.0], device=self.device, dtype=self.dtype)

        vehicle_yaw = robot_state[2]  # Assuming robot_state = [x, y, yaw, ...]
        heading_direction = torch.tensor([
            torch.cos(vehicle_yaw),
            torch.sin(vehicle_yaw)
        ], device=self.device, dtype=self.dtype)

        return heading_direction

    def _extend_lookahead_by_vehicle_heading(self, original_lookahead: torch.Tensor,
                                           robot_state: torch.Tensor) -> torch.Tensor:
        """Phase 1: Extend lookahead by 1m in vehicle heading direction"""
        if not self.lookahead_extension_enabled:
            return original_lookahead

        # Get current vehicle heading direction
        heading_direction = self._get_vehicle_heading_direction(robot_state)

        # Extend lookahead by specified distance in heading direction
        extension_vector = heading_direction * self.lookahead_extension_distance
        extended_lookahead = original_lookahead + extension_vector

        return extended_lookahead

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

        # Update curve-based parameters
        self.curve_detection_enabled = params.get('curve_detection_enabled', self.curve_detection_enabled)
        self.curve_angle_threshold = params.get('curve_angle_threshold', self.curve_angle_threshold)
        self.curve_lookahead_reduction_factor = params.get('curve_lookahead_reduction_factor', self.curve_lookahead_reduction_factor)
        self.curve_min_lookahead = params.get('curve_min_lookahead', self.curve_min_lookahead)
        self.curve_detection_distance = params.get('curve_detection_distance', self.curve_detection_distance)

        # Update lookahead extension parameters
        self.lookahead_extension_enabled = params.get('lookahead_extension_enabled', self.lookahead_extension_enabled)
        self.lookahead_extension_distance = params.get('lookahead_extension_distance', self.lookahead_extension_distance)
        print("[GoalCritic] Parameters updated")
