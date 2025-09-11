# === smppi_optimizer.py (DIAGNOSTIC PATCH) ===
import torch, numpy as np
import math, time
from typing import Optional, Tuple, Dict, Any

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path

def _now_sync():
    """CUDA 사용시 비동기 커널 동기화 후 perf_counter 반환"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def _entropy(p: torch.Tensor) -> float:
    return float(-(p * (p + 1e-12).log()).sum().detach().cpu().item())

def _omega_to_delta(v, omega, wheelbase, v_eps=1e-3):
    """각속도 -> 조향각"""
    if abs(v) < v_eps:
        return 0.0
    safe_v = max(1e-6, abs(v)) if v >= 0 else min(-1e-6, v)
    return math.atan((omega * wheelbase) / safe_v) # 이 식에서 알아서 부호 반대

class SMPPIOptimizer:
    def __init__(self, params: dict):
        # ----- Repro seeds (옵션) -----
        seed = int(params.get('seed', 0))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # -----------------------------

        self.K = params.get('batch_size', 1000)
        self.T = params.get('time_steps', 30)
        self.dt = params.get('model_dt', 0.1)
        self.temperature = params.get('temperature', 1.0)
        self.iteration_count = params.get('iteration_count', 1)

        self.lambda_action = params.get('lambda_action', 0.0)
        self.omega = torch.tensor(params.get('omega_diag', [1.0, 1.0]),
                                  dtype=torch.float32)

        self.v_min = params.get('v_min', 0.0)
        self.v_max = params.get('v_max', 2.0)
        self.w_min = params.get('w_min', -1.0)
        self.w_max = params.get('w_max', 1.0)
        self.wheelbase = params.get('wheelbase', 0.65)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32

        # U (derivative control)
        self.control_sequence = torch.zeros(self.T, 2, device=self.device, dtype=self.dtype)

        # Noise for U
        self.noise_std = torch.tensor(params.get('noise_std_u', [0.2, 0.2]),
                                      device=self.device, dtype=self.dtype)

        # States
        self.robot_state = None
        self.goal_state = None
        self.obstacles = None
        self.path = None

        # External
        self.critics = []
        self.motion_model = None

        # ===== DEBUG STORAGE =====
        self.debug: Dict[str, Any] = {}
        self.last_cmd_applied = torch.zeros(2, device=self.device, dtype=self.dtype)  # for pub vs plan diff

        print(f"[SMPPI] Initialized (SMPPI core) K={self.K}, T={self.T}, dt={self.dt}, dev={self.device}")

    # ---------- external hooks ----------
    def set_motion_model(self, motion_model): self.motion_model = motion_model
    def add_critic(self, critic): self.critics.append(critic)

    # ---------- prepare ----------
    def prepare(self, robot_pose: PoseStamped, robot_velocity: Twist,
                path: Optional[Path] = None, goal: Optional[PoseStamped] = None):
        x = robot_pose.pose.position.x
        y = robot_pose.pose.position.y
        quat = robot_pose.pose.orientation
        yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                         1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z))
        self.robot_state = torch.tensor([x, y, yaw,
                                         robot_velocity.linear.x,
                                         robot_velocity.angular.z],
                                        device=self.device, dtype=self.dtype)

        if goal:
            gx = goal.pose.position.x
            gy = goal.pose.position.y
            gq = goal.pose.orientation
            gyaw = np.arctan2(2.0 * (gq.w * gq.z + gq.x * gq.y),
                              1.0 - 2.0 * (gq.y * gq.y + gq.z * gq.z))
            self.goal_state = torch.tensor([gx, gy, gyaw], device=self.device, dtype=self.dtype)
        self.path = path

    def set_obstacles(self, obstacles): self.obstacles = obstacles

    # ---------- optimize (instrumented) ----------
    def optimize(self) -> torch.Tensor:
        # reset per-iteration debug
        self.debug = {}
        for it in range(self.iteration_count):
            U_samples, A_samples, eps = self._sample_U_and_build_A()
            traj = self._simulate_from_A(A_samples)  # [K,T+1,3]
            traj_costs = self._evaluate_trajectories(traj, A_samples)  # [K]
            action_costs = self._compute_action_sequence_cost(A_samples)  # 제어값 변화량에 대한 cost
            total_costs = traj_costs + self.lambda_action * action_costs

            # importance weights
            beta = torch.min(total_costs)
            weights = torch.exp(-(total_costs - beta) / max(1e-9, self.temperature))
            weights = weights / (torch.sum(weights) + 1e-12)

            # update U
            dU = torch.sum(weights[:, None, None] * eps, dim=0)  # [T,2]
            
            # DEBUG: Control sequence 변화량 추적
            old_U = self.control_sequence.clone()
            self.control_sequence = self.control_sequence + dU
            dU_norm = torch.norm(dU).item()
            max_dU_delta = torch.max(torch.abs(dU[:, 1])).item()  # 조향각 변화량
            
            # 매 5회마다 출력
            if hasattr(self, '_control_debug_counter'):
                self._control_debug_counter += 1
            else:
                self._control_debug_counter = 0
                
            # if self._control_debug_counter % 5 == 0 and dU_norm > 0.01:
            #     print(f"[CONTROL UPDATE] dU_norm: {dU_norm:.4f} | max_delta_change: {max_dU_delta:.4f}")
            #     print(f"  U[0:3] before: [{old_U[0,0]:.3f}, {old_U[0,1]:.3f}], [{old_U[1,0]:.3f}, {old_U[1,1]:.3f}], [{old_U[2,0]:.3f}, {old_U[2,1]:.3f}]")
            #     print(f"  U[0:3] after:  [{self.control_sequence[0,0]:.3f}, {self.control_sequence[0,1]:.3f}], [{self.control_sequence[1,0]:.3f}, {self.control_sequence[1,1]:.3f}], [{self.control_sequence[2,0]:.3f}, {self.control_sequence[2,1]:.3f}]")

            # ====== DEBUG METRICS ======
            with torch.no_grad():
                # a0 from odom
                a0 = self.robot_state[3:5] if self.robot_state is not None \
                    else torch.zeros(2, device=self.device, dtype=self.dtype)

                # plan preview (first action)
                A_preview = self._integrate_U_to_A(a0, self.control_sequence.unsqueeze(0))[0]  # [T,2]
                A_preview[:, 0] = torch.clamp(A_preview[:, 0], self.v_min, self.v_max)
                A_preview[:, 1] = torch.clamp(A_preview[:, 1], self.w_min, self.w_max)
                a_first = A_preview[0]

                # progress (dot with goal direction)
                prog = self._compute_progress(traj)  # [K] per sample
                clamp_v = ((A_samples[...,0] <= self.v_min+1e-6) | (A_samples[...,0] >= self.v_max-1e-6)).float().mean().item()
                clamp_w = ((A_samples[...,1] <= self.w_min+1e-6) | (A_samples[...,1] >= self.w_max-1e-6)).float().mean().item()

                self.debug.update({
                    "iter": it,
                    "a0_v": float(a0[0]), "a0_w": float(a0[1]),
                    "a_first_v": float(a_first[0]), "a_first_w": float(a_first[1]),
                    "weights_entropy": _entropy(weights),
                    "traj_cost_mean": float(traj_costs.mean().detach().cpu().item()),
                    "traj_cost_min": float(traj_costs.min().detach().cpu().item()),
                    "omega_cost_mean": float(action_costs.mean().detach().cpu().item()),
                    "total_cost_min": float(total_costs.min().detach().cpu().item()),
                    "dU_norm": float(torch.norm(dU).detach().cpu().item()),
                    "progress_mean": float(prog.mean().detach().cpu().item()),
                    "progress_max": float(prog.max().detach().cpu().item()),
                    "clamp_ratio_v": clamp_v, "clamp_ratio_w": clamp_w,
                })
        return self.control_sequence
    
    # ---------- sample & integrate ----------
    def _sample_U_and_build_A(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.robot_state is not None:
            v_odom = float(self.robot_state[3])
            w_odom = float(self.robot_state[4])      # yaw rate
            delta_odom = _omega_to_delta(v_odom, w_odom, self.wheelbase)  # <<< 변환
            a0_odom = torch.tensor([abs(v_odom), delta_odom],
                                device=self.device, dtype=self.dtype)
        else:
            a0_odom = torch.zeros(2, device=self.device, dtype=self.dtype)
        
        noise = torch.randn(self.K, self.T, 2, device=self.device, dtype=self.dtype) * self.noise_std
        U_nom = self.control_sequence.unsqueeze(0).repeat(self.K, 1, 1)
        U_samples = U_nom + noise
        eps = U_samples - U_nom
        # a0 = self.robot_state[3:5] if self.robot_state is not None \
        #      else torch.zeros(2, device=self.device, dtype=self.dtype)
        alpha = 0.3
        a0 = alpha * a0_odom + (1 - alpha) * self.last_cmd_applied
        if abs(float(self.robot_state[3])) < 0.05:  # v_odom < 5 cm/s
            a0 = self.last_cmd_applied.clone()
        
        # Debug: last_cmd_applied 출력
        # print(f"[SMPPI] last_cmd_applied: v={float(self.last_cmd_applied[0]):.3f}, δ={float(self.last_cmd_applied[1]):.3f}")
        # print(f"[SMPPI] a0_odom: v={float(abs(a0_odom[0])):.3f}, δ={float(a0_odom[1]):.3f}")
        # print(f"[SMPPI] a0_final: v={float(a0[0]):.3f}, δ={float(a0[1]):.3f}")
        # print(f"[SMPPI] w_min: {self.w_min}, w_max: {self.w_max}")
        a0[0] = torch.clamp(a0[0], self.v_min, self.v_max)
        a0[1] = torch.clamp(a0[1], self.w_min, self.w_max)     # δ 한계 보장
        A_samples = self._integrate_U_to_A(a0, U_samples)
        
        # 속도 한계 적용
        A_samples[..., 0] = torch.clamp(A_samples[..., 0], self.v_min, self.v_max)
        A_samples[..., 1] = torch.clamp(A_samples[..., 1], self.w_min, self.w_max)
        
        return U_samples, A_samples, eps

    def _integrate_U_to_A(self, a0: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        K, T, nu = U.shape
        dA = U * self.dt
        A = torch.zeros(K, T, nu, device=self.device, dtype=self.dtype)
        A[:, 0, :] = a0 + dA[:, 0, :]
        for t in range(1, T):
            A[:, t, :] = A[:, t - 1, :] + dA[:, t, :]
        return A

    # ---------- simulate ----------
    def _simulate_from_A(self, A_samples: torch.Tensor) -> torch.Tensor:
        if self.motion_model is None:
            raise ValueError("Motion model not set")
        K = A_samples.shape[0]
        x0 = self.robot_state[:3].unsqueeze(0).repeat(K, 1)
        if hasattr(self.motion_model, 'rollout_batch'):
            return self.motion_model.rollout_batch(x0, A_samples, self.dt)
        # 아래는 안씀
        traj = torch.zeros(K, self.T + 1, 3, device=self.device, dtype=self.dtype)
        traj[:, 0, :] = x0
        cur = x0
        for t in range(self.T):
            u = A_samples[:, t, :]
            if hasattr(self.motion_model, 'validate_controls'):
                u = self.motion_model.validate_controls(u)
            next_state = self.motion_model.forward(cur, u, self.dt)
            traj[:, t + 1, :] = next_state
            cur = next_state
        return traj

    # ---------- critics ----------
    def _evaluate_trajectories(self, trajectories: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(self.K, device=self.device, dtype=self.dtype)
        timings = []  # <-- critic별 시간 저장
        for i, critic in enumerate(self.critics):
            t0 = time.perf_counter()
            cost = critic.compute_cost(
                trajectories, actions,
                self.robot_state, self.goal_state, self.obstacles
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_cost = (time.perf_counter() - t0) * 1000.0
            timings.append((critic.__class__.__name__, t_cost, cost.shape))

            total += cost
        # 디버깅 정보 저장
        self.debug["critic_timings"] = timings
        return total

    def _compute_action_sequence_cost(self, A_samples: torch.Tensor) -> torch.Tensor:
        diff = A_samples[:, 1:, :] - A_samples[:, :-1, :]
        w = self.omega.to(self.device, self.dtype).view(1, 1, -1)
        return torch.sum((diff ** 2) * w, dim=(1, 2))

    # ---------- small helpers ----------
    def _compute_progress(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        목표 방향으로의 전진량(보상 아님, 진단용)
        """
        if self.goal_state is None or self.robot_state is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)
        traj_xy = trajectories[:, :, :2]               # [K,T+1,2]
        dx = traj_xy[:, 1:, :] - traj_xy[:, :-1, :]    # [K,T,2]
        g = self.goal_state[:2] - self.robot_state[:2] # [2]
        g = g / (torch.norm(g) + 1e-9)
        prog = torch.sum(torch.matmul(dx, g), dim=1)   # [K]
        return prog

    # ---------- horizon shift ----------
    def shift_control_sequence(self):
        # DEBUG: Tail value 변화 추적
        old_tail = self.control_sequence[-1].clone()
        old_second_last = self.control_sequence[-2].clone()
        
        self.control_sequence = torch.roll(self.control_sequence, -1, dims=0)
        self.control_sequence[-1] = 0.0  # derivative control의 일반적 꼬리값
        
        # Tail discontinuity 추적
        if hasattr(self, '_tail_debug_counter'):
            self._tail_debug_counter += 1
        else:
            self._tail_debug_counter = 0
            
        tail_change = torch.norm(old_second_last - self.control_sequence[-1]).item()
        if self._tail_debug_counter % 20 == 0 and tail_change > 0.1:
            print(f"[TAIL SHIFT] old_tail: [{old_tail[0]:.3f}, {old_tail[1]:.3f}] -> new_tail: [0.0, 0.0]")
            print(f"  tail_discontinuity: {tail_change:.4f} (old_second_last vs new_tail)")

    def get_control_command(self) -> Twist:
        if self.robot_state is None:
            return Twist()

        a0 = self.last_cmd_applied  # [v, δ]로 유지하면 더 안정적
        U = self.control_sequence.unsqueeze(0)
        A = self._integrate_U_to_A(a0, U)[0]   # [T,2] = [v, δ]
        A_unclamped = A[0].clone()  # Store unclamped values for debugging
        
        # DEBUG: Clipping 전후 비교
        v_before, delta_before = float(A[0,0]), float(A[0,1])
        A[:, 0] = torch.clamp(A[:, 0], self.v_min, self.v_max)
        A[:, 1] = torch.clamp(A[:, 1], self.w_min, self.w_max)
        v_next, delta_next = float(A[0,0]), float(A[0,1])
        
        # Clipping 발생 추적
        if hasattr(self, '_clipping_debug_counter'):
            self._clipping_debug_counter += 1
        else:
            self._clipping_debug_counter = 0
            
        delta_clipped = abs(delta_before - delta_next) > 1e-6
        if self._clipping_debug_counter % 10 == 0 and delta_clipped:
            print(f"[CLIPPING] delta: {delta_before:.4f} -> {delta_next:.4f} (limits: [{self.w_min:.3f}, {self.w_max:.3f}])")
            omega_before = (v_next / self.wheelbase) * math.tan(delta_before) if abs(v_next) > 1e-3 else 0.0
            omega_after = (v_next / self.wheelbase) * math.tan(delta_next) if abs(v_next) > 1e-3 else 0.0
            print(f"  omega: {omega_before:.4f} -> {omega_after:.4f}")
        

        # δ -> ω 변환 (Ackermann Model 공식 사용 - 후진/전진 자동 처리)
        if abs(v_next) < 1e-3:
            omega_next = 0.0
        else:
            omega_next = (v_next / self.wheelbase) * math.tan(delta_next)
        
        # === DIAGNOSTIC LOGGING: Track command changes ===
        prev_cmd = getattr(self, '_prev_cmd', [0.0, 0.0])
        cmd_v_change = abs(v_next - prev_cmd[0])
        cmd_w_change = abs(omega_next - prev_cmd[1]) if v_next > 0 else abs(-omega_next - prev_cmd[1])
        
        # Log significant command changes
        if cmd_w_change > 0.5:  # Angular velocity change > 0.5 rad/s
            print(f"🎮 [CMD CHANGE] v: {prev_cmd[0]:.3f}->{v_next:.3f} ({cmd_v_change:.3f}), ω: {prev_cmd[1]:.3f}->{omega_next if v_next > 0 else -omega_next:.3f} ({cmd_w_change:.3f})")
            print(f"   δ: {float(self.last_cmd_applied[1]):.4f} -> {delta_next:.4f}")
        
        # Twist publish
        cmd = Twist()
        cmd.linear.x  = v_next
        
        # 흠 왜 됨?
        if v_next > 0:
            cmd.angular.z = omega_next
            self._prev_cmd = [v_next, omega_next]
        else:
            cmd.angular.z = -omega_next
            self._prev_cmd = [v_next, -omega_next]
        
        # Goal 관련 코스트 시각화 (매 10회마다)
        if hasattr(self, '_goal_debug_counter'):
            self._goal_debug_counter += 1
        else:
            self._goal_debug_counter = 0
            
        if self._goal_debug_counter % 10 == 0:
            self._print_goal_cost_status()
        
        # 반드시 last_cmd_applied는 [v, δ]로 저장 (내부 일관성)
        self.last_cmd_applied = torch.tensor([v_next, delta_next],
                                            device=self.device, dtype=self.dtype)
        return cmd

    def getOptimizedTrajectory(self) -> Optional[torch.Tensor]:
        if self.robot_state is None:
            return None
        a0 = self.last_cmd_applied
        U = self.control_sequence.unsqueeze(0)
        A = self._integrate_U_to_A(a0, U)
        traj = self._simulate_from_A(A)
        return traj[0]

    def reset(self):
        self.control_sequence.zero_()
        # Also reset last_cmd_applied to prevent inconsistency after parameter changes
        self.last_cmd_applied = torch.zeros(2, device=self.device, dtype=self.dtype)
        print("[SMPPI] Optimizer reset (U and last_cmd_applied cleared)")

    # ---------- dynamic parameter updates ----------
    def update_velocity_limits(self, min_v: float = None, max_v: float = None, 
                              min_w: float = None, max_w: float = None):
        """Update velocity limits dynamically"""
        if min_v is not None:
            self.v_min = min_v
        if max_v is not None:
            self.v_max = max_v
        if min_w is not None:
            self.w_min = min_w
        if max_w is not None:
            self.w_max = max_w
        
        print(f"[SMPPI] Velocity limits updated: v_min={self.v_min:.2f}, v_max={self.v_max:.2f}, "
              f"w_min={self.w_min:.3f}, w_max={self.w_max:.3f}")
    
    def set_action_bounds(self, v_bounds: list = None, w_bounds: list = None):
        """Alternative method for setting action bounds"""
        if v_bounds and len(v_bounds) == 2:
            self.v_min, self.v_max = v_bounds[0], v_bounds[1]
        if w_bounds and len(w_bounds) == 2:
            self.w_min, self.w_max = w_bounds[0], w_bounds[1]
        
        print(f"[SMPPI] Action bounds updated: v=[{self.v_min:.2f}, {self.v_max:.2f}], "
              f"w=[{self.w_min:.3f}, {self.w_max:.3f}]")

    # ---------- expose debug ----------
    def get_debug(self) -> Dict[str, Any]:
        """
        최근 optimize()에서 모은 디버그 값 반환
        """
        return dict(self.debug)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        if isinstance(angle, torch.Tensor):
            return torch.atan2(torch.sin(angle), torch.cos(angle))
        else:
            return math.atan2(math.sin(angle), math.cos(angle))
    
    def _print_goal_cost_status(self):
        # """Goal 관련 코스트 상세 시각화"""
        # print("=" * 80)
        # print("[GOAL COST ANALYSIS]")
        
        # # 로봇과 골 상태
        # if hasattr(self, 'robot_state') and self.robot_state is not None:
        #     x, y, yaw, v_odom, w_odom = [float(x) for x in self.robot_state[:5]]
        #     print(f"🚗  Robot: pos=({x:.2f},{y:.2f}) | yaw={yaw:.3f} | v={v_odom:.3f} | ω={w_odom:.3f}")
        
        # if hasattr(self, 'goal_state') and self.goal_state is not None:
        #     gx, gy, gyaw = [float(x) for x in self.goal_state[:3]]
        #     print(f"🎯  Goal: pos=({gx:.2f},{gy:.2f}) | yaw={gyaw:.3f}")
            
        #     # 거리 계산
        #     if hasattr(self, 'robot_state') and self.robot_state is not None:
        #         robot_pos = self.robot_state[:2]
        #         goal_pos = self.goal_state[:2]
        #         distance = float(torch.norm(goal_pos - robot_pos))
        #         direction = goal_pos - robot_pos
        #         direction = direction / (torch.norm(direction) + 1e-9)
        #         target_yaw = float(torch.atan2(direction[1], direction[0]))
        #         yaw_error = abs(self.normalize_angle(yaw - target_yaw))
        #         print(f"📏  Distance: {distance:.3f}m | Target yaw: {target_yaw:.3f} | Yaw error: {yaw_error:.3f}")
        
        # # Goal critic 정보 (critics에서 goal critic 찾기)
        # for critic in self.critics:
        #     if hasattr(critic, '__class__') and 'Goal' in critic.__class__.__name__:
        #         print(f"⚖️   Goal Critic: weight={critic.weight:.1f}")
        #         if hasattr(critic, 'xy_goal_tolerance'):
        #             print(f"      xy_tol={critic.xy_goal_tolerance:.3f} | yaw_tol={critic.yaw_goal_tolerance:.3f}")
        #         if hasattr(critic, 'lookahead_base_distance'):
        #             print(f"      lookahead: base={critic.lookahead_base_distance:.1f} | "
        #                   f"vel_fac={critic.lookahead_velocity_factor:.1f} | "
        #                   f"range=[{critic.lookahead_min_distance:.1f}-{critic.lookahead_max_distance:.1f}]")
        #         if hasattr(critic, 'respect_reverse_heading'):
        #             print(f"      reverse_heading={critic.respect_reverse_heading} | "
        #                   f"use_multi_wp={critic.use_multiple_waypoints}")
                
        #         # Lookahead point 정보
        #         if hasattr(critic, 'last_lookahead_point') and critic.last_lookahead_point is not None:
        #             lp = critic.last_lookahead_point
        #             ly = critic.last_lookahead_yaw if hasattr(critic, 'last_lookahead_yaw') else 0.0
        #             print(f"👀  Lookahead: pos=({float(lp[0]):.2f},{float(lp[1]):.2f}) | yaw={float(ly):.3f}")
        #         break
        
        # print("=" * 80)
        pass
