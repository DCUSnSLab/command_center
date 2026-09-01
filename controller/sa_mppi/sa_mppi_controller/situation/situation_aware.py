"""Situation-aware context classification and heuristic adaptation for SA-MPPI."""
import math
import torch

from .situations import (
    external_score,
    internal_score,
    planned_curvature,
    DynamicLayer,
    dynamic_sector,
    external_score_dynamic,
    rasterize_points_mask,
)


def clamp01(value):
    return min(max(float(value), 0.0), 1.0)


class SituationAware:
    """Rule-based context estimator with hysteresis and adaptation smoothing."""

    NORMAL = "Normal"
    CROWDED = "Crowded"   # external axis (surrounding density)
    CURVED = "Curved"     # internal axis (rotational motion)
    ESCAPE = "Escape"     # internal axis 2 (standstill escape)

    def __init__(self, config, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

        self.L = max(float(config.get("L", 0.65)), 1e-3)
        self.max_delta = float(config.get("max_delta", 0.461))
        self.dt = max(float(config.get("dt", 0.1)), 1e-4)
        self.max_speed = max(float(config.get("max_speed", 1.0)), 1e-3)
        self.horizon = max(int(config.get("horizon", 20)), 1)

        self.max_decel = max(float(config.get("max_decel", 1.5)), 1e-3)
        self.enable_cbf = bool(config.get("enable_cbf", True))  # ablation toggle
        # Internal axis 1: lateral-acceleration limit.  a_lat = v^2 * kappa, so
        # the geometry of the planned turn bounds a safe speed:
        #     v <= sqrt(a_lat_max / kappa).
        # This DERIVES the cornering speed from vehicle physics (skid/rollover
        # margin) instead of the hand-tuned per-situation speed_scale, mirroring
        # the external axis' braking-distance cap.  At max_v = 1.5 m/s and the
        # Hunter's max steering the cap never binds (sqrt(2.0/0.766) = 1.6 m/s),
        # which is why it is validated on a high-speed (2.0 m/s) curvy run.
        self.enable_lat_acc = bool(config.get("enable_lat_acc", False))
        self.max_lat_acc = max(float(config.get("max_lat_acc", 2.0)), 1e-3)
        # Dynamic-aware gating (paper core): a single tracker-free motion label
        # (ego-motion-compensated costmap temporal difference) gates the two
        # conservatism knobs INDEPENDENTLY, enabling the 2x2 gate ablation:
        #   - dynamic_gate_cbf:        braking-distance speed cap keys off moving
        #                              cells only (False -> legacy: any obstacle)
        #   - dynamic_gate_covariance: crowded/external axis (sampling covariance
        #                              raise) driven by moving cells only
        #                              (False -> legacy: static density)
        # enable_dynamic_aware is the master switch for computing the motion
        # mask at all; with it False both knobs behave legacy (ungated).
        self.enable_dynamic_aware = bool(config.get("enable_dynamic_aware", True))  # ablation master
        # Label-source ablation: 'detector' (temporal-difference, deployable) or
        # 'oracle' (ground-truth agent positions rasterized as disks — sim-only
        # perfect-perception upper bound; separates gate concept from detector).
        self.dynamic_mask_source = str(config.get("dynamic_mask_source", "detector"))
        self.oracle_disk_radius = float(config.get("oracle_disk_radius", 0.35))
        self.oracle_points = []
        self.dynamic_gate_cbf = bool(config.get("dynamic_gate_cbf", True))
        self.dynamic_gate_covariance = bool(config.get("dynamic_gate_covariance", True))
        self.dynamic_min_cluster = max(int(config.get("dynamic_min_cluster", 3)), 1)
        self.dynamic_hold_frames = max(int(config.get("dynamic_hold_frames", 25)), 1)
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
            self.ESCAPE: self._parse_profile(config, "escape", 2.5, 1.8, 1.0, 1.0, 0.0, 2.0),
        }

        # Internal axis 2: standstill escape.  Every controller in mixed_long
        # (vanilla 28%, oracle-labeled gate 20%, detector gate 68% of trials)
        # dies in the SAME static zigzag pockets: the robot sits at |v|~0 with
        # healthy ESS while the softmax average cancels around a concave dead
        # end.  Escape reallocates exploration (sigma_v up, lambda up) ONLY
        # when (a) the robot has been standing still and (b) the external axis
        # is clear — a stop near a moving agent is safety, not a pocket.  That
        # cross-axis approval rule (external-clear gates internal-escape) is
        # the composition SA-MPPI's situation set exists to express.
        # Stuck detection is PROGRESS-based, not speed-based: "no
        # escape_progress_m of displacement within the trailing
        # escape_progress_window_s".  A speed-threshold version (stuck below
        # 0.08 m/s, released above 0.30) latched permanently in a crowd —
        # creeping inside the 0.08-0.30 dead zone neither accumulated nor
        # released, so escape stayed on for 55% of ticks in ped_corridor and
        # its own widened sigma_v sustained the creep (goal 56%->28%).
        # Displacement has no dead zone: creeping 0.2 m/s covers 0.6 m in 3 s
        # and releases immediately, while a pocket shows ~0 m.
        self.enable_escape = bool(config.get("enable_escape", False))
        self.escape_progress_window_s = max(float(config.get("escape_progress_window_s", 3.0)), self.dt)
        self.escape_progress_m = max(float(config.get("escape_progress_m", 0.5)), 1e-3)
        # Trigger is deliberately LONG.  Kinematics alone cannot tell "crowd
        # crawl" (correct behaviour) from "static pocket" (needs escape) —
        # measured no-progress run lengths: ped_corridor median 3.1 s (p90
        # 55 s) vs mixed_long zigzag pockets median 208 s.  Duration is the
        # only separating evidence, so escape acts as a last resort: at 30 s
        # the zigzag keeps 86% of its escape time while corridor exposure
        # drops from ~57 s to ~21 s per trial.
        self.escape_trigger_s = max(float(config.get("escape_trigger_s", 30.0)), 0.0)
        self.escape_ramp_s = max(float(config.get("escape_ramp_s", 2.0)), self.dt)
        self._esc_buf = []      # trailing [(t, x, y)] within the progress window
        self._esc_t = 0.0       # local clock (ticks * dt)
        self._noprog_s = 0.0    # seconds without progress

        # Cached observations
        self.costmap_tensor = None
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.width = 0
        self.height = 0
        # Dynamic-aware perception (temporal-difference moving-cell detector)
        self.dynamic_layer = DynamicLayer(
            occupied_thresh=self.occupied_thresh,
            min_cluster=self.dynamic_min_cluster,
            device=self.device,
            hold_frames=self.dynamic_hold_frames,
        )
        self.dynamic_mask = None
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

    def set_oracle_points(self, points_xy):
        """Latest ground-truth dynamic-agent positions [(x, y), ...] (oracle source)."""
        self.oracle_points = list(points_xy) if points_xy else []

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
            # Update the moving-cell mask (ego-motion-compensated temporal diff,
            # or GT rasterization when running the oracle label source).
            if self.enable_dynamic_aware and cm is not None:
                if self.dynamic_mask_source == "oracle":
                    self.dynamic_mask = rasterize_points_mask(
                        cm.shape, self.origin_x, self.origin_y, self.resolution,
                        self.oracle_points, self.oracle_disk_radius,
                        device=self.device,
                    )
                else:
                    self.dynamic_mask = self.dynamic_layer.update(
                        cm, self.origin_x, self.origin_y, self.resolution
                    )
            else:
                self.dynamic_mask = None

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
        # Static external score is always computed for min_obstacle_dist (debug/legacy).
        s_external_static, reachable_density, min_obs_dist = external_score(
            self, x, y, theta, context_range, reachable_half_angle
        )
        # Dynamic-aware external axis: "crowded" driven by MOVING obstacles only,
        # so static clutter keeps baseline sampling variance (avoids the
        # noise_scale_delta steering washout in tight static geometry).
        # The moving-cell statistics are computed whenever the mask exists; each
        # of the two conservatism knobs then independently chooses gated/legacy
        # (dynamic_gate_covariance / dynamic_gate_cbf) for the 2x2 ablation.
        d_dyn = float("inf")
        n_dynamic = 0
        s_external_dyn = 0.0
        _mask_ok = self.enable_dynamic_aware and self.dynamic_mask is not None
        if _mask_ok:
            s_external_dyn, dyn_density, d_dyn, n_dynamic = external_score_dynamic(
                self, x, y, theta, context_range, reachable_half_angle
            )
        if _mask_ok and self.dynamic_gate_covariance:
            s_external = s_external_dyn
        else:
            s_external = s_external_static
        s_internal, curvature_deg = internal_score(self)

        # Internal axis 2 (standstill escape): ramps up while the robot stands
        # still, is vetoed by ANY external presence (moving agent nearby ->
        # standing still is the correct behaviour, not a pocket), and resets
        # as soon as the robot moves again.
        s_escape = 0.0
        if self.enable_escape:
            self._esc_t += self.dt
            self._esc_buf.append((self._esc_t, x, y))
            win = self.escape_progress_window_s
            while self._esc_buf and (self._esc_t - self._esc_buf[0][0]) > win:
                self._esc_buf.pop(0)
            if self._esc_buf and (self._esc_t - self._esc_buf[0][0]) >= 0.8 * win:
                t0, x0, y0 = self._esc_buf[0]
                if math.hypot(x - x0, y - y0) >= self.escape_progress_m:
                    self._noprog_s = 0.0
                else:
                    self._noprog_s += self.dt
            external_present = (n_dynamic > 0) if _mask_ok else (s_external > 0.10)
            if not external_present:
                s_escape = clamp01(
                    (self._noprog_s - self.escape_trigger_s) / self.escape_ramp_s
                )

        scores = {"internal": s_internal, "external": s_external, "escape": s_escape}
        raw_context = self._classify_raw(scores)
        stable_context = self._apply_hysteresis_and_dwell(raw_context=raw_context, scores=scores)
        blended_adaptation = self._blend_adaptation(scores)
        adaptation = self._smooth_adaptation(blended_adaptation)
        # Adaptive max-speed from a braking-distance control barrier (GS-MPPI, arXiv:2410.02154):
        # require the robot be able to stop before the nearest obstacle, v <= sqrt(2*a_max*(d_clear - eps)).
        # This DERIVES the speed adaptation from vehicle physics (a_max, sensor margin eps) instead of
        # hand-tuned per-situation speed_scale constants — composite with the situation profile (only ever
        # slows further), floored to keep the robot moving.
        #
        # Dynamic-aware (paper direction A): the braking distance that matters is
        # to *moving* obstacles; static geometry is already avoided by the
        # obstacle cost, so braking for it just stalls the robot in clutter.
        # d_safety = nearest MOVING-obstacle distance; if none is moving, the CBF
        # is inactive (cbf_speed_scale = 1) and speed is governed by the profile.
        # d_dyn / n_dynamic already computed above by external_score_dynamic.
        if _mask_ok and self.dynamic_gate_cbf:
            d_safety = d_dyn  # nearest MOVING obstacle (inf if none -> no braking)
        else:
            # legacy behaviour: brake for the nearest obstacle of any kind
            d_safety = min_obs_dist if math.isfinite(min_obs_dist) else context_range
        if math.isfinite(d_safety):
            v_cbf = math.sqrt(2.0 * self.max_decel * max(d_safety - self.sensor_error, 0.0))
            cbf_speed_scale = min(v_cbf / max(self.max_speed, 1e-6), 1.0)
        else:
            cbf_speed_scale = 1.0  # no moving obstacle -> no braking-distance cap
        # Internal axis 1 (lateral-acceleration cap): v <= sqrt(a_lat,max/kappa)
        # from the planned turn geometry.  Composes by min() with the external
        # braking cap — every axis may only ever slow the robot further.
        kappa = planned_curvature(self) if self.enable_lat_acc else 0.0
        if self.enable_lat_acc and kappa > 1e-6:
            v_lat = math.sqrt(self.max_lat_acc / kappa)
            lat_speed_scale = min(v_lat / max(self.max_speed, 1e-6), 1.0)
        else:
            lat_speed_scale = 1.0

        adaptation = dict(adaptation)
        _prof_ss = adaptation.get("speed_scale", 1.0)
        # ablation: enable_cbf=False -> profile speed_scale only (no braking-distance cap)
        _ss = min(_prof_ss, cbf_speed_scale) if self.enable_cbf else _prof_ss
        adaptation["speed_scale"] = max(min(_ss, lat_speed_scale), 0.12)
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
            "min_dynamic_dist": d_dyn,
            "n_dynamic_cells": n_dynamic,
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
            "escape_score": scores["escape"],
            "planned_curvature": kappa,
            "lat_speed_scale": lat_speed_scale,
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
        escape = self.profiles[self.ESCAPE]      # internal axis 2 (standstill escape)

        s_int = clamp01(scores.get("internal", 0.0))
        s_ext = clamp01(scores.get("external", 0.0))
        s_esc = clamp01(scores.get("escape", 0.0))

        blended = {}
        for key in base.keys():
            b = float(base[key])
            blended[key] = (b + s_int * (float(internal[key]) - b)
                            + s_ext * (float(external[key]) - b)
                            + s_esc * (float(escape[key]) - b))
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
