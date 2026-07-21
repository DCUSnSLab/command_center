"""Situation-aware context classification and heuristic adaptation for SA-MPPI."""
import math
import torch

from .situations import external_score, internal_score


def clamp01(value):
    return min(max(float(value), 0.0), 1.0)


class SituationAware:
    """Rule-based context estimator with hysteresis and adaptation smoothing."""

    NORMAL = "Normal"
    CROWDED = "Crowded"   # external axis (surrounding density)
    CURVED = "Curved"     # internal axis (rotational motion)

    def __init__(self, config, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

        self.L = max(float(config.get("L", 0.65)), 1e-3)
        self.max_delta = float(config.get("max_delta", 0.461))
        self.dt = max(float(config.get("dt", 0.1)), 1e-4)
        self.max_speed = max(float(config.get("max_speed", 1.0)), 1e-3)
        self.horizon = max(int(config.get("horizon", 20)), 1)

        self.max_decel = max(float(config.get("max_decel", 1.5)), 1e-3)
        self.sensor_error = float(config.get("sensor_error", 0.3))
        self.min_safety_range = max(float(config.get("min_safety_range", 1.0)), 0.05)
        self.max_safety_range = max(float(config.get("max_safety_range", 6.0)), self.min_safety_range)
        self.lookahead_time = max(float(config.get("lookahead_time", 3.0)), 0.0)
        self.occupied_thresh = float(config.get("occupied_cost_threshold", 80.0))
        # external-axis (density) sector threshold for reachable-density.
        self.crowded_cost_threshold = float(config.get("crowded_cost_threshold", 35.0))
        self.max_reachable_half_angle = math.radians(
            float(config.get("max_reachable_half_angle_deg", 80.0))
        )
        self.crowded_near_distance = max(float(config.get("crowded_near_distance", 2.0)), 0.2)
        self.crowded_proximity_weight = clamp01(
            float(config.get("crowded_proximity_weight", 0.65))
        )

        # Ramp thresholds for the two situation axes.
        self.crowded_entry = float(config.get("crowded_density_entry", 0.30))      # external (density)
        self.crowded_exit = float(config.get("crowded_density_exit", 0.20))
        self.curved_entry_deg = float(config.get("curved_angle_entry_deg", 45.0))  # internal (rotation)
        self.curved_exit_deg = float(config.get("curved_angle_exit_deg", 30.0))
        self.crowded_saturation = float(
            config.get("crowded_density_saturation", self.crowded_entry + 0.10)
        )
        self.curved_saturation_deg = float(
            config.get("curved_angle_saturation_deg", self.curved_entry_deg + 20.0)
        )
        self.context_activation_score = clamp01(
            float(config.get("context_activation_score", 0.15))
        )
        self.context_exit_score = clamp01(
            float(config.get("context_exit_score", self.context_activation_score * 0.6))
        )

        self.switch_dwell_ticks = max(int(config.get("switch_dwell_ticks", 3)), 1)
        self.adaptation_smoothing_alpha = min(
            max(float(config.get("adapt_smoothing_alpha", 0.70)), 0.0), 0.99
        )
        self.curved_window_steps = max(
            int(config.get("curved_window_steps", max(int(round(1.0 / max(self.dt, 1e-3))), 1))),
            1,
        )

        # Keep valid ramp intervals even with odd parameter sets.
        self.crowded_saturation = max(self.crowded_saturation, self.crowded_entry + 1e-3)
        self.curved_saturation_deg = max(self.curved_saturation_deg, self.curved_entry_deg + 1e-3)
        self.context_exit_score = min(self.context_exit_score, self.context_activation_score)

        # Two-axis adaptation profiles. The situation is described by two
        # INDEPENDENT axes (not mutually exclusive labels):
        #   - internal (rotational motion)  -> CURVED profile
        #   - external (surrounding density) -> CROWDED profile
        # NORMAL is the identity base (no adaptation). BLOCKED is dropped from
        # this refactor (handled as an out-of-scope global-planning limitation).
        # The CURVED/CROWDED config keys are reused so existing yamls still apply.
        self.profiles = {
            self.NORMAL: self._parse_profile(config, "normal", 1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
            self.CROWDED: self._parse_profile(config, "crowded", 1.0, 1.8, 0.60, 1.0, 0.0, 1.0),
            self.CURVED: self._parse_profile(config, "curved", 1.0, 1.0, 0.70, 1.0, 0.0, 1.0),
        }

        # Cached observations
        self.costmap_tensor = None
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.width = 0
        self.height = 0
        self.control_sequence = None
        self.current_speed = 0.0

        # State for stabilization
        self.current_context = self.NORMAL
        self._pending_context = self.NORMAL
        self._pending_ticks = 0
        self._smoothed_adapt = dict(self.profiles[self.NORMAL])
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def _parse_profile(
        self,
        config,
        name,
        noise_scale_v,
        noise_scale_delta,
        speed_scale,
        reverse_scale,
        v_bias,
        lambda_scale,
    ):
        prefix = f"adaptation.{name}"
        base_noise = float(config.get(f"{prefix}.noise_scale", 1.0))
        noise_v = float(config.get(f"{prefix}.noise_scale_v", noise_scale_v if noise_scale_v is not None else base_noise))
        noise_delta = float(
            config.get(f"{prefix}.noise_scale_delta", noise_scale_delta if noise_scale_delta is not None else base_noise)
        )
        return {
            "noise_scale": 0.5 * (noise_v + noise_delta),
            "noise_scale_v": noise_v,
            "noise_scale_delta": noise_delta,
            "speed_scale": float(config.get(f"{prefix}.speed_scale", speed_scale)),
            "reverse_scale": float(config.get(f"{prefix}.reverse_scale", reverse_scale)),
            "v_bias": float(config.get(f"{prefix}.v_bias", v_bias)),
            "lambda_scale": float(config.get(f"{prefix}.lambda_scale", lambda_scale)),
        }

    def update(self, costmap_info, control_sequence, current_speed=None):
        """Update cached perception/control snapshots."""
        if costmap_info is None:
            self.costmap_tensor = None
            self.width = 0
            self.height = 0
        else:
            cm = costmap_info.get("costmap_tensor", None)
            # Move the costmap to this module's device once per tick (the optimizer
            # keeps it on GPU). All subsequent sector/gap math then runs locally
            # without per-element GPU<->CPU synchronizations.
            if cm is not None and cm.device != self.device:
                cm = cm.to(self.device)
            self.costmap_tensor = cm
            self.resolution = max(float(costmap_info.get("resolution", 0.05)), 1e-4)
            self.origin_x = float(costmap_info.get("origin_x", 0.0))
            self.origin_y = float(costmap_info.get("origin_y", 0.0))
            self.width = int(costmap_info.get("width", 0))
            self.height = int(costmap_info.get("height", 0))

        if control_sequence is not None:
            self.control_sequence = control_sequence.detach().to(self.device).clone()

        if current_speed is not None:
            self.current_speed = abs(float(current_speed))

    def evaluate(self, robot_state):
        """Evaluate raw and stabilized context, then return adaptation profile."""
        rs = robot_state.detach().cpu().tolist() if hasattr(robot_state, "detach") else list(robot_state)
        x = float(rs[0])
        y = float(rs[1])
        theta = float(rs[2])

        current_speed = self.current_speed
        if current_speed <= 1e-6 and self.control_sequence is not None and self.control_sequence.shape[0] > 0:
            current_speed = abs(float(self.control_sequence[0, 0].item()))

        braking_dist = (current_speed * current_speed) / (2.0 * self.max_decel)
        lookahead_dist = current_speed * self.lookahead_time
        safety_range = max(braking_dist + self.sensor_error, lookahead_dist + self.sensor_error)
        safety_range = min(max(safety_range, self.min_safety_range), self.max_safety_range)

        # External-axis input range: vehicle max speed * dt * horizon over the costmap.
        eval_range = self.max_speed * self.dt * float(self.horizon)
        context_range = min(max(eval_range, self.min_safety_range), self.max_safety_range)

        reachable_half_angle = min(
            (context_range * math.tan(self.max_delta)) / self.L,
            self.max_reachable_half_angle,
            math.pi,
        )
        # Two situation axes (see situations.py): external (density) / internal (rotation).
        s_external, reachable_density, min_obs_dist = external_score(
            self, x, y, theta, context_range, reachable_half_angle
        )
        s_internal, curvature_deg = internal_score(self)
        scores = {"internal": s_internal, "external": s_external}
        raw_context = self._classify_raw(scores)
        stable_context = self._apply_hysteresis_and_dwell(raw_context=raw_context, scores=scores)
        blended_adaptation = self._blend_adaptation(scores)
        adaptation = self._smooth_adaptation(blended_adaptation)
        # Adaptive max-speed from a braking-distance control barrier (GS-MPPI, arXiv:2410.02154):
        # require the robot be able to stop before the nearest obstacle, v <= sqrt(2*a_max*(d_clear - eps)).
        # This DERIVES the speed adaptation from vehicle physics (a_max, sensor margin eps) instead of
        # hand-tuned per-situation speed_scale constants — composite with the situation profile (only ever
        # slows further), floored to keep the robot moving. d_clear = forward nearest-obstacle distance.
        d_clear = min_obs_dist if math.isfinite(min_obs_dist) else context_range
        v_cbf = math.sqrt(2.0 * self.max_decel * max(d_clear - self.sensor_error, 0.0))
        cbf_speed_scale = min(v_cbf / max(self.max_speed, 1e-6), 1.0)
        adaptation = dict(adaptation)
        adaptation["speed_scale"] = max(min(adaptation.get("speed_scale", 1.0), cbf_speed_scale), 0.12)
        normal_score = clamp01((1.0 - scores["internal"]) * (1.0 - scores["external"]))

        return {
            "raw_situation": raw_context,
            "situation": stable_context,
            "adaptation": adaptation,
            "safety_range": safety_range,
            "context_range": context_range,
            "eval_range": context_range,
            "reachable_half_angle_deg": math.degrees(reachable_half_angle),
            "current_speed": current_speed,
            "braking_distance": braking_dist,
            "lookahead_distance": lookahead_dist,
            "reachable_density": reachable_density,
            "min_obstacle_dist": min_obs_dist,
            "cumulative_curvature_deg": curvature_deg,
            "normal_score": normal_score,
            # Interface compat: external axis surfaces as crowded_score, internal
            # axis as curved_score, blocked retired (always 0). Downstream debug
            # publisher fills forward_density / *_gap with .get() defaults (0/inf).
            "crowded_score": scores["external"],
            "curved_score": scores["internal"],
            "blocked_score": 0.0,
            "internal_score": scores["internal"],
            "external_score": scores["external"],
        }

    def _classify_raw(self, scores):
        """Dominant axis for debug/visualization (not the adaptation path)."""
        s_int = scores.get("internal", 0.0)
        s_ext = scores.get("external", 0.0)
        if max(s_int, s_ext) < self.context_activation_score:
            return self.NORMAL
        return self.CURVED if s_int >= s_ext else self.CROWDED

    def _apply_hysteresis_and_dwell(self, raw_context, scores):
        current_score = self._context_score(self.current_context, scores)
        if (self.current_context != self.NORMAL) and (current_score >= self.context_exit_score):
            target = self.current_context
        else:
            target = raw_context

        if target == self.current_context:
            self._pending_context = self.current_context
            self._pending_ticks = 0
            return self.current_context

        if target != self._pending_context:
            self._pending_context = target
            self._pending_ticks = 1
            return self.current_context

        self._pending_ticks += 1
        if self._pending_ticks >= self.switch_dwell_ticks:
            self.current_context = target
            self._pending_ticks = 0
        return self.current_context

    def _smooth_adaptation(self, target_profile):
        a = self.adaptation_smoothing_alpha
        for key, target_val in target_profile.items():
            prev = float(self._smoothed_adapt.get(key, target_val))
            self._smoothed_adapt[key] = a * prev + (1.0 - a) * float(target_val)
        return dict(self._smoothed_adapt)

    def _blend_adaptation(self, scores):
        """Additive two-axis blending around the Normal base profile.

        theta(x) = base + s_int*(internal - base) + s_ext*(external - base)

        Independent axes (rotation + density) can both be active, so deviations
        from the base add up. s_int=s_ext=0 -> base (identity) -> baseline MPPI.
        """
        base = self.profiles[self.NORMAL]
        internal = self.profiles[self.CURVED]    # internal axis profile
        external = self.profiles[self.CROWDED]   # external axis profile

        s_int = clamp01(scores.get("internal", 0.0))
        s_ext = clamp01(scores.get("external", 0.0))

        blended = {}
        for key in base.keys():
            b = float(base[key])
            blended[key] = b + s_int * (float(internal[key]) - b) + s_ext * (float(external[key]) - b)
        return blended

    def _context_score(self, context_name, scores):
        # Curved <-> internal axis, Crowded <-> external axis (2-axis refactor).
        if context_name == self.CROWDED:
            return float(scores.get("external", 0.0))
        if context_name == self.CURVED:
            return float(scores.get("internal", 0.0))
        return clamp01(
            (1.0 - scores.get("internal", 0.0)) * (1.0 - scores.get("external", 0.0))
        )
