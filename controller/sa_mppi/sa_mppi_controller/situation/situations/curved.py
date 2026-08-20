"""Internal situation axis: Curved (rotational motion).

Self-contained, like a cost file: it owns its small helpers, its descriptor
(cumulative curvature of the current control plan), and its score.
``internal_score`` reads what it needs from the ``SituationAware`` instance.
"""
import math

import torch


def clamp01(value):
    return min(max(float(value), 0.0), 1.0)


def linear_ramp(value, start, end):
    if end <= start + 1e-6:
        return 1.0 if value >= end else 0.0
    return clamp01((float(value) - float(start)) / (float(end) - float(start)))


def _cumulative_curvature_deg(sa):
    """Max |heading change| over a short window of the current control plan (deg)."""
    cs = sa.control_sequence
    if cs is None or cs.shape[0] == 0:
        return 0.0
    v = cs[:, 0]
    delta = cs[:, 1]
    theta_dot = (v / sa.L) * torch.tan(delta)
    steps = min(int(v.shape[0]), sa.curved_window_steps)
    if steps <= 0:
        return 0.0
    heading_delta = torch.cumsum(theta_dot[:steps] * sa.dt, dim=0)
    return math.degrees(float(torch.max(torch.abs(heading_delta)).item()))


def internal_score(sa):
    """Curved score in [0, 1]: ramp of |cumulative curvature| of the control
    plan. Returns (score, curvature_deg)."""
    curvature_deg = _cumulative_curvature_deg(sa)
    score = clamp01(linear_ramp(abs(curvature_deg), sa.curved_entry_deg, sa.curved_saturation_deg))
    return score, curvature_deg


def planned_curvature(sa):
    """Max path curvature |kappa| = |tan(delta)| / L over the near plan (1/m).

    Curvature, unlike the cumulative-heading descriptor above, is a *geometric*
    quantity independent of speed, which is what a lateral-acceleration limit
    needs: a_lat = v^2 * kappa.  Taken as the maximum over the near window so
    the cap anticipates the tightest part of the upcoming turn.
    """
    cs = sa.control_sequence
    if cs is None or cs.shape[0] == 0:
        return 0.0
    steps = max(min(int(cs.shape[0]), sa.curved_window_steps), 1)
    delta = cs[:steps, 1]
    kappa = torch.abs(torch.tan(delta)) / sa.L
    return float(torch.max(kappa).item())
