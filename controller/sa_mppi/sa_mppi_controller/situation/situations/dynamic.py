"""Dynamic-aware perception layer for the external situation axis.

Motivation (paper direction A): the braking-distance CBF that governs SA-MPPI's
speed cap keys off ``d_clear`` = distance to the *nearest obstacle of any kind*.
That makes it brake for static walls exactly as for a moving pedestrian, which
crushes speed in static clutter (empirically: 4/5 timeouts in mixed_long, and a
speed_scale retune from 0.70->0.85 did NOT help because the CBF term, not the
profile, is the binding constraint).

Fix: distinguish *moving* obstacles from *static* geometry and let the CBF react
only to moving ones; static geometry is already avoided by the obstacle cost, so
no speed cap is needed for it.  Detection is sensor-based (costmap temporal
differencing), so it transfers to a real robot -- no ground-truth obstacle state.

The local costmap is a rolling window published in the (fixed, axis-aligned)
``odom`` frame with ``origin = robot_pose - size/2``.  Between two ticks the grid
only *translates* (no rotation, orientation.w = 1), so a cell at world (wx, wy)
maps between frames by an integer cell shift derived from the origin change.  A
cell that is occupied now but was free at the *same world location* last tick is
the leading edge of something that moved -> dynamic.  Static walls keep the same
world location occupied across ticks -> not flagged.
"""
import math

import torch

try:
    from scipy.ndimage import label as _cc_label
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


class DynamicLayer:
    """Ego-motion-compensated temporal-difference detector of moving cells.

    ``update`` returns a boolean mask (same shape as the costmap) marking cells
    that just became occupied at their world location = moving-obstacle fronts.
    """

    def __init__(self, occupied_thresh=80.0, min_cluster=3, device="cpu"):
        self.occupied_thresh = float(occupied_thresh)
        self.min_cluster = int(min_cluster)   # sector must hold >= this many dyn cells to be trusted
        self.device = device
        self._prev = None          # previous costmap tensor (occupancy, [H, W])
        self._prev_ox = None
        self._prev_oy = None
        self._prev_res = None

    def reset(self):
        self._prev = None
        self._prev_ox = self._prev_oy = self._prev_res = None

    def update(self, costmap_tensor, origin_x, origin_y, resolution):
        """Return a boolean [H, W] mask of newly-occupied (dynamic) cells.

        Returns ``None`` when no valid comparison is possible yet (first frame,
        shape/resolution change).
        """
        if costmap_tensor is None:
            self.reset()
            return None
        curr = costmap_tensor
        occ_curr = curr >= self.occupied_thresh

        prev = self._prev
        need_reset = (
            prev is None
            or prev.shape != curr.shape
            or self._prev_res is None
            or abs(self._prev_res - resolution) > 1e-9
        )
        # cache current for next tick regardless
        store = curr.detach()
        if need_reset:
            self._prev = store
            self._prev_ox, self._prev_oy, self._prev_res = origin_x, origin_y, resolution
            return None

        h, w = curr.shape
        # cell shift so that prev[i + s] is the same world location as curr[i]
        sx = int(round((origin_x - self._prev_ox) / resolution))
        sy = int(round((origin_y - self._prev_oy) / resolution))

        if abs(sx) >= w or abs(sy) >= h:
            # moved further than the window in one tick -> no overlap
            self._prev = store
            self._prev_ox, self._prev_oy, self._prev_res = origin_x, origin_y, resolution
            return None

        # aligned_prev[i] = prev[i + s]  via roll by -s
        aligned_prev = torch.roll(prev, shifts=(-sy, -sx), dims=(0, 1))
        occ_prev = aligned_prev >= self.occupied_thresh

        # validity: rolled-in wrap region is garbage -> invalidate a border margin
        valid = torch.ones_like(occ_curr, dtype=torch.bool)
        if sx > 0:
            valid[:, w - sx:] = False
        elif sx < 0:
            valid[:, : -sx] = False
        if sy > 0:
            valid[h - sy:, :] = False
        elif sy < 0:
            valid[: -sy, :] = False

        # dynamic = occupied now, free at same world loc last tick, inside valid overlap
        dynamic = occ_curr & (~occ_prev) & valid

        # Reject scattered perception flicker: a real moving obstacle forms a
        # coherent blob (connected component), whereas costmap/lidar noise near
        # static walls appears as isolated cells.  Keep only components whose
        # size >= min_cluster.
        if _HAVE_SCIPY and self.min_cluster > 1 and bool(dynamic.any()):
            dyn_np = dynamic.detach().to("cpu").numpy()
            lab, n = _cc_label(dyn_np)
            if n > 0:
                import numpy as _np
                sizes = _np.bincount(lab.ravel())
                sizes[0] = 0  # background
                keep = sizes >= self.min_cluster
                dyn_np = keep[lab]
                dynamic = torch.from_numpy(dyn_np).to(device=dynamic.device)

        self._prev = store
        self._prev_ox, self._prev_oy, self._prev_res = origin_x, origin_y, resolution
        return dynamic


def dynamic_sector(sa, x, y, theta, range_m, half_angle):
    """Forward-sector statistics over *dynamic* (moving) cells only.

    Mirrors the sector geometry of ``crowded._sector_density`` but queries the
    dynamic mask cached on the SituationAware instance (``sa.dynamic_mask``).
    Returns (min_dist_m, n_dynamic, density) where density is the distance-
    weighted fraction of the reachable sector occupied by moving cells (in
    [0,1]).  When no moving cell is present: (+inf, 0, 0.0).
    """
    mask = getattr(sa, "dynamic_mask", None)
    if mask is None or sa.width <= 0 or sa.height <= 0:
        return float("inf"), 0, 0.0

    range_cells = range_m / sa.resolution
    gx_center = (x - sa.origin_x) / sa.resolution
    gy_center = (y - sa.origin_y) / sa.resolution

    margin = int(range_cells) + 2
    gx_min = max(0, int(gx_center) - margin)
    gx_max = min(sa.width, int(gx_center) + margin + 1)
    gy_min = max(0, int(gy_center) - margin)
    gy_max = min(sa.height, int(gy_center) + margin + 1)
    if gx_min >= gx_max or gy_min >= gy_max:
        return float("inf"), 0, 0.0

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
    n_sector = int(in_sector.sum().item())
    if n_sector == 0:
        return float("inf"), 0, 0.0

    local_mask = mask[gy_min:gy_max, gx_min:gx_max]
    dyn_in_sector = in_sector & local_mask
    n_dyn = int(dyn_in_sector.sum().item())
    if n_dyn <= 0:
        return float("inf"), 0, 0.0

    dist_m = dist_cells[in_sector] * sa.resolution
    near_w_all = torch.clamp(1.0 - (dist_m / max(range_m, 1e-6)), min=0.1, max=1.0)
    dyn_dist_m = dist_cells[dyn_in_sector] * sa.resolution
    near_w_dyn = torch.clamp(1.0 - (dyn_dist_m / max(range_m, 1e-6)), min=0.1, max=1.0)
    density = float(near_w_dyn.sum().item() / max(near_w_all.sum().item(), 1e-6))
    min_dist = float(dyn_dist_m.min().item())
    return min_dist, n_dyn, density


def external_score_dynamic(sa, x, y, theta, range_m, half_angle):
    """Crowded score from *moving* obstacles only (dynamic-aware external axis).

    Same blend formula as ``crowded.external_score`` (proximity + density) but
    keyed off the dynamic mask, so static clutter does NOT raise the crowded
    score -> in static tight geometry the robot keeps baseline sampling variance
    and commits through gaps instead of freezing (the noise_scale_delta washout).
    Returns (score, density, min_dyn_dist, n_dynamic).
    """
    min_dyn, n_dyn, density = dynamic_sector(sa, x, y, theta, range_m, half_angle)
    if n_dyn < getattr(sa, "dynamic_min_cluster", 1):  # reject perception flicker
        return 0.0, 0.0, float("inf"), 0
    if math.isfinite(min_dyn):
        prox = min(max((sa.crowded_near_distance - min_dyn) / max(sa.crowded_near_distance, 1e-6), 0.0), 1.0)
    else:
        prox = 0.0
    if sa.crowded_saturation <= sa.crowded_entry + 1e-6:
        dens = 1.0 if density >= sa.crowded_saturation else 0.0
    else:
        dens = min(max((density - sa.crowded_entry) / (sa.crowded_saturation - sa.crowded_entry), 0.0), 1.0)
    score = min(max(sa.crowded_proximity_weight * prox + (1.0 - sa.crowded_proximity_weight) * dens, 0.0), 1.0)
    return score, density, min_dyn, n_dyn
