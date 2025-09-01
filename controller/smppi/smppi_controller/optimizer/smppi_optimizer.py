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
    if abs(v) < v_eps:
        return 0.0
    return math.atan((omega * wheelbase) / max(1e-6, v))

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
            action_costs = self._compute_action_sequence_cost(A_samples)  # [K]
            total_costs = traj_costs + self.lambda_action * action_costs

            # importance weights
            beta = torch.min(total_costs)
            weights = torch.exp(-(total_costs - beta) / max(1e-9, self.temperature))
            weights = weights / (torch.sum(weights) + 1e-12)

            # update U
            dU = torch.sum(weights[:, None, None] * eps, dim=0)  # [T,2]
            self.control_sequence = self.control_sequence + dU

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
            a0_odom = torch.tensor([v_odom, delta_odom],
                                device=self.device, dtype=self.dtype)
        else:
            a0_odom = torch.zeros(2, device=self.device, dtype=self.dtype)
        alpha = 0.0
        noise = torch.randn(self.K, self.T, 2, device=self.device, dtype=self.dtype) * self.noise_std
        U_nom = self.control_sequence.unsqueeze(0).repeat(self.K, 1, 1)
        U_samples = U_nom + noise
        eps = U_samples - U_nom
        # a0 = self.robot_state[3:5] if self.robot_state is not None \
        #      else torch.zeros(2, device=self.device, dtype=self.dtype)
        a0 = alpha * a0_odom + (1 - alpha) * self.last_cmd_applied
        if abs(float(self.robot_state[3])) < 0.05:  # v_odom < 5 cm/s
            a0 = self.last_cmd_applied.clone()
        a0[0] = torch.clamp(a0[0], self.v_min, self.v_max)     # v >= 0 보장
        a0[1] = torch.clamp(a0[1], self.w_min, self.w_max)     # δ 한계 보장
        A_samples = self._integrate_U_to_A(a0, U_samples)
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
        self.control_sequence = torch.roll(self.control_sequence, -1, dims=0)
        self.control_sequence[-1] = 0.0  # derivative control의 일반적 꼬리값

    def get_control_command(self) -> Twist:
        if self.robot_state is None:
            return Twist()

        a0 = self.last_cmd_applied  # [v, δ]로 유지하면 더 안정적
        U = self.control_sequence.unsqueeze(0)
        A = self._integrate_U_to_A(a0, U)[0]   # [T,2] = [v, δ]
        A[:, 0] = torch.clamp(A[:, 0], self.v_min, self.v_max)
        A[:, 1] = torch.clamp(A[:, 1], self.w_min, self.w_max)
        v_next, delta_next = float(A[0,0]), float(A[0,1])

        # δ -> ω 변환 (Twist 규약 준수)
        if abs(v_next) < 1e-3:
            omega_next = 0.0
        else:
            omega_next = (v_next / self.wheelbase) * math.tan(delta_next)

        # Twist publish
        cmd = Twist()
        cmd.linear.x  = v_next
        cmd.angular.z = omega_next

        # Convert omega back to delta for verification
        if abs(v_next) < 1e-3:
            delta_recovered = 0.0
        else:
            delta_recovered = math.atan((omega_next * self.wheelbase) / v_next)
        
        # Debug log - show steering angle in radians (not rad/s)
        # print(f"[SMPPI] Control: v={v_next:.3f} m/s, δ={delta_next:.3f} rad, ω={omega_next:.3f} rad/s, δ_recovered={delta_recovered:.3f} rad")

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
        A[..., 0] = torch.clamp(A[..., 0], self.v_min, self.v_max)
        A[..., 1] = torch.clamp(A[..., 1], self.w_min, self.w_max)
        traj = self._simulate_from_A(A)
        return traj[0]

    def reset(self):
        self.control_sequence.zero_()
        print("[SMPPI] Optimizer reset (U cleared)")

    # ---------- expose debug ----------
    def get_debug(self) -> Dict[str, Any]:
        """
        최근 optimize()에서 모은 디버그 값 반환
        """
        return dict(self.debug)
