# === base_mppi_optimizer.py ===
# Standard (base) MPPI core, drop-in replacement for the SMPPI core.
#
# Unlike SMPPIOptimizer (which samples action DERIVATIVES U and integrates
# U->A with an action-smoothing penalty), this samples the control/action
# sequence DIRECTLY: A_samples = control_sequence + noise, with noise std on
# [v, delta].  This matches the base MPPI logic validated in the SA-MPPI
# experiments (mppi_core), so the tuned params (noise_sigma, lambda) apply
# directly.  The situation-aware layer (proposal adaptation + IS correction +
# braking-distance CBF speed cap) is preserved unchanged.
#
# Public interface is identical to SMPPIOptimizer so mppi_main_node,
# critics, motion model and the situation layer all work without change:
#   set_motion_model, add_critic, prepare, set_obstacles,
#   set_runtime_adaptation, get_runtime_velocity_limits, get_nominal_plan_A,
#   optimize, get_control_command, shift_control_sequence,
#   getOptimizedTrajectory, reset, update_velocity_limits, set_action_bounds,
#   get_debug ; attributes: critics, device, dtype, robot_state,
#   control_sequence, last_cmd_applied.
import time
import math
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path


def _entropy(p: torch.Tensor) -> float:
    return float(-(p * (p + 1e-12).log()).sum().detach().cpu().item())


def _omega_to_delta(v, omega, wheelbase, v_eps=1e-3):
    if abs(v) < v_eps:
        return 0.0
    safe_v = max(1e-6, abs(v)) if v >= 0 else min(-1e-6, v)
    return math.atan((omega * wheelbase) / safe_v)


class BaseMPPIOptimizer:
    def __init__(self, params: dict):
        seed = int(params.get('seed', 0))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.K = params.get('batch_size', 3000)
        self.T = params.get('time_steps', 30)
        self.dt = params.get('model_dt', 0.1)
        self.temperature = params.get('temperature', 20.0)   # base-MPPI lambda
        self.iteration_count = params.get('iteration_count', 1)

        self.v_min = params.get('v_min', 0.0)
        self.v_max = params.get('v_max', 2.0)
        # w_min/w_max are used as STEERING-ANGLE (delta) limits in this stack
        # (see mppi_main_node: w_min=-max_steering, w_max=+max_steering).
        self.w_min = params.get('w_min', -0.3665)
        self.w_max = params.get('w_max', 0.3665)
        self.wheelbase = params.get('wheelbase', 0.65)
        self.max_steering_angle = params.get('max_steering_angle', self.w_max)
        self.min_speed_for_cap = params.get('min_speed_for_cap', 0.10)
        self.max_lateral_acc = params.get('max_lateral_acc', 3.9)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32

        # Control/action plan [T,2] = [v, delta]  (DIRECT, not derivative).
        self.control_sequence = torch.zeros(self.T, 2, device=self.device, dtype=self.dtype)

        # Base MPPI control-noise std on [v, delta].  Prefer 'noise_sigma';
        # fall back to the SMPPI 'noise_std_u' if that is all that is provided.
        ns = params.get('noise_sigma', params.get('noise_std_u', [0.15, 0.2]))
        self.noise_sigma = torch.tensor(ns, device=self.device, dtype=self.dtype)

        # === SA-MPPI runtime adaptation state (proposal q_phi) ===
        # base_* = nominal proposal p + limits; runtime_* = situation-adapted.
        # Equal by default -> log_is == 0 -> reduces exactly to base MPPI.
        self.base_noise_sigma = self.noise_sigma.clone()
        self.base_temperature = float(self.temperature)
        self.base_v_max = float(self.v_max)
        self.base_v_min = float(self.v_min)
        self.runtime_noise_sigma = self.base_noise_sigma.clone()
        self.runtime_temperature = self.base_temperature
        self.runtime_speed_scale = 1.0
        self.runtime_reverse_scale = 1.0
        self.runtime_v_bias = 0.0

        # States
        self.robot_state = None
        self.goal_state = None
        self.obstacles = None
        self.path = None

        # External hooks
        self.critics = []
        self.motion_model = None

        # Debug / bookkeeping
        self.debug: Dict[str, Any] = {}
        self.last_cmd_applied = torch.zeros(2, device=self.device, dtype=self.dtype)
        self.last_weights = None
        self.last_ess = 0.0
        self._prev_cmd = [0.0, 0.0]

        print(f"[BaseMPPI] Initialized (base MPPI core) K={self.K}, T={self.T}, "
              f"dt={self.dt}, lambda={self.temperature}, sigma={ns}, dev={self.device}")

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

    # ---------- SA-MPPI runtime adaptation ----------
    def set_runtime_adaptation(self, profile=None):
        """Apply a situation-aware adaptation profile (proposal q_phi).
        Empty/None restores the nominal proposal p => plain base MPPI."""
        if not profile:
            profile = {}
        noise_scale_v = max(float(profile.get('noise_scale_v', 1.0)), 0.05)
        noise_scale_delta = max(float(profile.get('noise_scale_delta', 1.0)), 0.05)
        lambda_scale = max(float(profile.get('lambda_scale', 1.0)), 0.1)
        speed_scale = min(max(float(profile.get('speed_scale', 1.0)), 0.05), 1.5)
        reverse_scale = min(max(float(profile.get('reverse_scale', 1.0)), 0.0), 2.0)
        v_bias = float(profile.get('v_bias', 0.0))

        self.runtime_noise_sigma = self.base_noise_sigma.clone()
        self.runtime_noise_sigma[0] = self.base_noise_sigma[0] * noise_scale_v
        self.runtime_noise_sigma[1] = self.base_noise_sigma[1] * noise_scale_delta
        self.runtime_temperature = max(self.base_temperature * lambda_scale, 1e-4)
        self.runtime_speed_scale = speed_scale
        self.runtime_reverse_scale = reverse_scale
        self.runtime_v_bias = v_bias

    def get_runtime_velocity_limits(self):
        """Velocity clamp bounds after runtime speed scaling (situation speed
        cap incl. braking-distance CBF from the SA layer). Returns (v_min, v_max)."""
        v_max = max(0.05, self.base_v_max * self.runtime_speed_scale)
        v_min = self.base_v_min
        if v_min < 0.0:
            v_min = v_min * self.runtime_reverse_scale
        return v_min, v_max

    def get_nominal_plan_A(self) -> torch.Tensor:
        """The absolute [v, delta] plan [T,2] (== control_sequence in base MPPI),
        for the situation layer's curvature (internal) score."""
        return self.control_sequence.clone()

    # ---------- optimize ----------
    def optimize(self) -> torch.Tensor:
        self.debug = {}
        if self.motion_model is None:
            raise ValueError("Motion model not set")
        x0 = self.robot_state[:3].unsqueeze(0).repeat(self.K, 1)
        v_min_rt, v_max_rt = self.get_runtime_velocity_limits()

        for it in range(self.iteration_count):
            # Step 1: sample the control/action sequence directly from q_phi.
            noise = torch.randn(self.K, self.T, 2, device=self.device, dtype=self.dtype) \
                * self.runtime_noise_sigma
            nominal = self.control_sequence.clone()
            if abs(self.runtime_v_bias) > 1e-6:
                nominal[:, 0] = nominal[:, 0] + self.runtime_v_bias
            sampled = nominal.unsqueeze(0) + noise                       # [K,T,2]
            sampled[..., 0] = torch.clamp(sampled[..., 0], v_min_rt, v_max_rt)
            sampled[..., 1] = torch.clamp(sampled[..., 1], self.w_min, self.w_max)

            # Step 2: rollout
            traj = self.motion_model.rollout_batch(x0, sampled, self.dt)  # [K,T+1,3]

            # Step 3: critic costs
            total_costs = torch.zeros(self.K, device=self.device, dtype=self.dtype)
            timings = []
            for critic in self.critics:
                t0 = time.perf_counter()
                cost = critic.compute_cost(traj, sampled,
                                           self.robot_state, self.goal_state, self.obstacles)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append((critic.__class__.__name__, (time.perf_counter() - t0) * 1000.0))
                total_costs = total_costs + cost

            # Step 4: importance weights with situation-aware IS correction.
            # q_phi ~ N(nominal(incl v_bias), runtime_sigma); p ~ N(control_seq, base_sigma).
            # log p - log q_phi de-biases the exponential weighting (Biased-MPPI 2024).
            # runtime==base and v_bias==0 -> log_is==0 -> identical to base MPPI.
            sig_q = self.runtime_noise_sigma.view(1, 1, 2)
            sig_p = self.base_noise_sigma.view(1, 1, 2)
            d_q = sampled - nominal.unsqueeze(0)
            d_p = sampled - self.control_sequence.unsqueeze(0)
            log_is = (0.5 * (d_q / sig_q) ** 2).sum(dim=(1, 2)) \
                   - (0.5 * (d_p / sig_p) ** 2).sum(dim=(1, 2))          # [K] = log p - log q_phi
            exp_arg = -total_costs / max(1e-9, self.runtime_temperature) + log_is
            exp_arg = exp_arg - exp_arg.max()
            weights = torch.exp(exp_arg)
            weights = weights / (torch.sum(weights) + 1e-12)
            self.last_ess = float(1.0 / torch.clamp((weights * weights).sum(), min=1e-12))
            self.last_weights = weights

            # Step 5: update the plan = weighted average of sampled sequences.
            self.control_sequence = torch.sum(weights[:, None, None] * sampled, dim=0)
            self.control_sequence[:, 0] = torch.clamp(self.control_sequence[:, 0], v_min_rt, v_max_rt)
            self.control_sequence[:, 1] = torch.clamp(self.control_sequence[:, 1], self.w_min, self.w_max)

            with torch.no_grad():
                a_first = self.control_sequence[0]
                self.debug.update({
                    "iter": it,
                    "a_first_v": float(a_first[0]), "a_first_delta": float(a_first[1]),
                    "weights_entropy": _entropy(weights),
                    "ess": self.last_ess,
                    "traj_cost_mean": float(total_costs.mean().item()),
                    "traj_cost_min": float(total_costs.min().item()),
                    "critic_timings": timings,
                })
        return self.control_sequence

    # ---------- horizon shift ----------
    def shift_control_sequence(self):
        """Receding-horizon warm start: shift plan by one step, repeat the last
        action (base MPPI warm start, unlike SMPPI which zeros the derivative tail)."""
        last = self.control_sequence[-1].clone()
        self.control_sequence = torch.roll(self.control_sequence, -1, dims=0)
        self.control_sequence[-1] = last

    # ---------- output command ----------
    def get_control_command(self) -> Twist:
        if self.robot_state is None:
            return Twist()
        A0 = self.control_sequence[0].clone()          # [v, delta]
        v_next = float(A0[0])
        delta_next = float(A0[1])

        # runtime speed cap (incl. CBF) + static steering-angle limit
        _vmin_rt, _vmax_rt = self.get_runtime_velocity_limits()
        v_next = float(min(max(v_next, _vmin_rt), _vmax_rt))
        delta_next = float(min(max(delta_next, self.w_min), self.w_max))

        # speed-dependent steering cap: delta <= min(delta_max, atan(L*ay_max/v^2))
        L = float(self.wheelbase)
        dmax = float(self.max_steering_angle)
        ay_max = float(self.max_lateral_acc)
        v_abs = max(abs(v_next), float(self.min_speed_for_cap))
        delta_dyn_max = min(dmax, math.atan((L * ay_max) / (v_abs * v_abs)))
        delta_next = max(-delta_dyn_max, min(delta_dyn_max, delta_next))

        # delta -> omega (Ackermann)
        omega_next = 0.0 if v_abs < 1e-3 else (v_next / L) * math.tan(delta_next)

        # speed-dependent yaw cap + delta re-sync
        omega_geom = (v_abs / L) * math.tan(dmax)
        omega_fric = ay_max / v_abs
        omega_cap = min(omega_geom, omega_fric)
        omega_before = omega_next
        omega_next = max(-omega_cap, min(omega_cap, omega_next))
        if abs(omega_before - omega_next) > 1e-6 and v_abs > 1e-3:
            delta_mag = math.atan((L * abs(omega_next)) / v_abs)
            delta_next = math.copysign(delta_mag, delta_next)

        cmd = Twist()
        cmd.linear.x = v_next
        cmd.angular.z = omega_next
        self._prev_cmd = [v_next, omega_next]
        self.last_cmd_applied = torch.tensor([v_next, delta_next],
                                             device=self.device, dtype=self.dtype)
        return cmd

    def getOptimizedTrajectory(self) -> Optional[torch.Tensor]:
        if self.robot_state is None or self.motion_model is None:
            return None
        x0 = self.robot_state[:3].unsqueeze(0)
        traj = self.motion_model.rollout_batch(x0, self.control_sequence.unsqueeze(0), self.dt)
        return traj[0]

    # ---------- misc ----------
    def reset(self):
        self.control_sequence.zero_()
        self.last_cmd_applied = torch.zeros(2, device=self.device, dtype=self.dtype)
        self.runtime_noise_sigma = self.base_noise_sigma.clone()
        self.runtime_temperature = self.base_temperature
        self.runtime_speed_scale = 1.0
        self.runtime_reverse_scale = 1.0
        self.runtime_v_bias = 0.0
        print("[BaseMPPI] Optimizer reset (plan and last_cmd_applied cleared)")

    def update_velocity_limits(self, min_v=None, max_v=None, min_w=None, max_w=None):
        if min_v is not None:
            self.v_min = min_v; self.base_v_min = float(min_v)
        if max_v is not None:
            self.v_max = max_v; self.base_v_max = float(max_v)
        if min_w is not None:
            self.w_min = min_w
        if max_w is not None:
            self.w_max = max_w

    def set_action_bounds(self, v_bounds: list = None, w_bounds: list = None):
        if v_bounds and len(v_bounds) == 2:
            self.v_min, self.v_max = v_bounds[0], v_bounds[1]
            self.base_v_min, self.base_v_max = float(v_bounds[0]), float(v_bounds[1])
        if w_bounds and len(w_bounds) == 2:
            self.w_min, self.w_max = w_bounds[0], w_bounds[1]

    def get_debug(self) -> Dict[str, Any]:
        return dict(self.debug)

    def normalize_angle(self, angle):
        if isinstance(angle, torch.Tensor):
            return torch.atan2(torch.sin(angle), torch.cos(angle))
        return math.atan2(math.sin(angle), math.cos(angle))
