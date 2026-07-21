"""External situation axis: Crowded (surrounding density).

Self-contained, like a cost file: it owns its small helpers, its descriptor
(reachable-sector occupancy density + nearest-obstacle distance), and its
score. ``external_score`` reads the parameters/cached perception it needs from
the ``SituationAware`` instance (passed as ``sa``).
"""
import math

import torch


def clamp01(value):
    return min(max(float(value), 0.0), 1.0)


def linear_ramp(value, start, end):
    if end <= start + 1e-6:
        return 1.0 if value >= end else 0.0
    return clamp01((float(value) - float(start)) / (float(end) - float(start)))


def _sector_density(sa, x, y, theta, range_m, half_angle, threshold):
    """Distance-weighted occupancy density + nearest occupied distance in a
    forward sector of the cached costmap. Returns (density, min_dist[m])."""
    if sa.costmap_tensor is None or sa.width <= 0 or sa.height <= 0:
        return 0.0, float("inf")

    range_cells = range_m / sa.resolution
    gx_center = (x - sa.origin_x) / sa.resolution
    gy_center = (y - sa.origin_y) / sa.resolution

    margin = int(range_cells) + 2
    gx_min = max(0, int(gx_center) - margin)
    gx_max = min(sa.width, int(gx_center) + margin + 1)
    gy_min = max(0, int(gy_center) - margin)
    gy_max = min(sa.height, int(gy_center) + margin + 1)
    if gx_min >= gx_max or gy_min >= gy_max:
        return 0.0, float("inf")

    gy_idx, gx_idx = torch.meshgrid(
        torch.arange(gy_min, gy_max, device=sa.device, dtype=sa.dtype),
        torch.arange(gx_min, gx_max, device=sa.device, dtype=sa.dtype),
        indexing="ij",
    )
    dx = gx_idx - gx_center
    dy = gy_idx - gy_center
    dist_cells = torch.sqrt(dx * dx + dy * dy)

    cell_angle = torch.atan2(dy, dx) - theta
    cell_angle = (cell_angle + math.pi) % (2.0 * math.pi) - math.pi

    in_sector = (dist_cells <= range_cells) & (dist_cells > 0.5) & (torch.abs(cell_angle) <= half_angle)
    if int(in_sector.sum().item()) == 0:
        return 0.0, float("inf")

    local_costmap = sa.costmap_tensor[gy_min:gy_max, gx_min:gx_max]
    sector_values = local_costmap[in_sector]

    threshold_val = min(max(float(threshold), 0.0), 100.0)
    denom = max(100.0 - threshold_val, 1e-6)
    occupancy_score = torch.clamp((sector_values - threshold_val) / denom, min=0.0, max=1.0)

    # Distance-weighted: nearer occupied cells contribute more risk.
    dist_m = dist_cells[in_sector] * sa.resolution
    near_weight = torch.clamp(1.0 - (dist_m / max(range_m, 1e-6)), min=0.1, max=1.0)
    density = float((occupancy_score * near_weight).sum().item() / near_weight.sum().item())

    occupied_in_sector = in_sector & (local_costmap >= sa.occupied_thresh)
    if bool(occupied_in_sector.any()):
        min_dist = float(dist_cells[occupied_in_sector].min().item() * sa.resolution)
    else:
        min_dist = float("inf")
    return density, min_dist


def external_score(sa, x, y, theta, context_range, reachable_half_angle):
    """Crowded score in [0, 1]: proximity to nearest obstacle blended with
    reachable-sector occupancy density. Returns (score, density, min_obs_dist)."""
    density, min_obs = _sector_density(
        sa, x, y, theta, context_range, reachable_half_angle, sa.crowded_cost_threshold
    )
    if math.isfinite(min_obs):
        prox = clamp01((sa.crowded_near_distance - min_obs) / max(sa.crowded_near_distance, 1e-6))
    else:
        prox = 0.0
    dens = linear_ramp(density, sa.crowded_entry, sa.crowded_saturation)
    score = clamp01(
        sa.crowded_proximity_weight * prox + (1.0 - sa.crowded_proximity_weight) * dens
    )
    return score, density, min_obs
