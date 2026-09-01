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

    def __init__(self, occupied_thresh=80.0, min_cluster=3, device="cpu",
                 hold_frames=25):
        self.occupied_thresh = float(occupied_thresh)
        self.min_cluster = int(min_cluster)   # sector must hold >= this many dyn cells to be trusted
        # Motion PERSISTENCE: a cell that moved stays classified dynamic for
        # hold_frames costmap updates as long as it remains occupied.  Without
        # this, an agent that slows to a stop near the robot (social-forces
        # pedestrians do exactly that) becomes invisible to frame differencing
        # at the precise moment the controller most needs to treat it as alive
        # (measured in ped_corridor: the un-held gate closed inside crowd
        # pockets and success collapsed 56%->32% vs the ungated cell).
        # Still tracker-free: this is a per-cell countdown, not object state.
        self.hold_frames = max(int(hold_frames), 1)
        # Vacancy-pairing radius (cells): an appearance counts as motion only if
        # some cell was vacated within this radius the same tick (disocclusion
        # rejection; ~0.6 m at 0.1 m grid covers a pedestrian step).
        self.vacancy_radius = 6
        # Stable-halo radius (cells): appearances this close to STRUCTURE are
        # wall-edge aliasing, not motion.  Structure = occupancy that has held
        # the same world cell for >= stable_age_frames consecutive frames.
        # The age condition is essential: a walking pedestrian's footprint
        # overlaps itself between frames, so its own body is "stable across 2
        # frames" and a naive 2-frame halo rejects the leading edge -> the
        # detector goes fully blind to movers <~2 m/s (measured: 0/13 frames
        # on a 1 m/s cylinder; mixed_long dynamic-region gate activity fell
        # 42%->2%).  A cell under a passing pedestrian stays occupied at most
        # 2r/v (~0.7 s at 1 m/s); walls hold indefinitely -> age separates
        # them.  Trade-off: movers slower than ~2r/age_thresh (~0.5 m/s) can
        # graduate into structure; occupancy-conditioned persistence keeps
        # them gated if they were ever seen moving.
        self.halo_radius = 2
        self.stable_age_frames = 15  # ~1.5 s @ 10 Hz costmap
        self.device = device
        self._prev = None          # previous costmap tensor (occupancy, [H, W])
        self._prev_ox = None
        self._prev_oy = None
        self._prev_res = None
        self._hold = None          # int16 countdown grid [H, W] (world-aligned)
        self._stable_age = None    # int16 consecutive-occupancy age grid (world-aligned)

    def reset(self):
        self._prev = None
        self._prev_ox = self._prev_oy = self._prev_res = None
        self._hold = None
        self._stable_age = None

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

        # VACANCY PAIRING (disocclusion filter): a real mover occupies new cells
        # AND vacates its previous cells nearby; a wall revealed by the lidar
        # shadow sweeping past a corner only APPEARS (nothing vacates).  Without
        # this test, cornering in a twisty corridor floods the mask with
        # "moving walls" and (amplified 25x by the persistence hold) the gated
        # CBF brakes continuously -> measured mixed_long G/G 0/25 with healthy
        # ESS (~1100) and freezing 0.62.  Keep an appearance only if some cell
        # was vacated within vacancy_radius of it this same tick.
        if bool(dynamic.any()):
            vacated = occ_prev & (~occ_curr) & valid
            if bool(vacated.any()):
                r = self.vacancy_radius
                k = 2 * r + 1
                near_vac = torch.nn.functional.max_pool2d(
                    vacated.to(torch.float32).unsqueeze(0).unsqueeze(0),
                    kernel_size=k, stride=1, padding=r,
                ).squeeze(0).squeeze(0) > 0.5
                dynamic = dynamic & near_vac
            else:
                dynamic = torch.zeros_like(dynamic)

        # Consecutive-occupancy age (world-aligned, for the structure test
        # below).  Rolled like _hold; rolled-in border cells restart at 0.
        if self._stable_age is None or self._stable_age.shape != curr.shape:
            self._stable_age = torch.zeros_like(curr, dtype=torch.int16)
        else:
            self._stable_age = torch.roll(self._stable_age, shifts=(-sy, -sx), dims=(0, 1))
            if sx > 0:
                self._stable_age[:, w - sx:] = 0
            elif sx < 0:
                self._stable_age[:, : -sx] = 0
            if sy > 0:
                self._stable_age[h - sy:, :] = 0
            elif sy < 0:
                self._stable_age[: -sy, :] = 0
        held_both = occ_prev & occ_curr & valid
        self._stable_age = torch.where(
            held_both,
            torch.clamp(self._stable_age + 1, max=self.stable_age_frames),
            torch.zeros_like(self._stable_age),
        )

        # STABLE-HALO rejection (wall-edge aliasing filter): as the robot moves,
        # lidar sampling jitter makes STATIC wall boundaries flicker — one edge
        # cell vacates while a neighbour appears, which forms a legal
        # appearance+vacancy pair and slips past the vacancy test (measured:
        # 58-155 'Crowded' ticks at x=31-47 of mixed_long where nothing moves,
        # re-enabling the washout the gate exists to prevent).  A real mover
        # travels through FREE space; boundary jitter is by definition adjacent
        # to STRUCTURE — occupancy old enough that no passing agent could have
        # produced it (see stable_age_frames).  Reject appearances within
        # halo_radius of structure only; a mover's own 2-frame-stable body
        # never qualifies, so walking pedestrians stay visible.
        if bool(dynamic.any()):
            structure = self._stable_age >= self.stable_age_frames
            if bool(structure.any()):
                hr = self.halo_radius
                hk = 2 * hr + 1
                halo = torch.nn.functional.max_pool2d(
                    structure.to(torch.float32).unsqueeze(0).unsqueeze(0),
                    kernel_size=hk, stride=1, padding=hr,
                ).squeeze(0).squeeze(0) > 0.5
                dynamic = dynamic & (~halo)

        # Reject scattered perception flicker: a real moving obstacle forms a
        # coherent blob (connected component), whereas costmap/lidar noise near
        # static walls appears as isolated cells.  Keep only components whose
        # size >= min_cluster.
        if _HAVE_SCIPY and self.min_cluster > 1 and bool(dynamic.any()):
            dyn_np = dynamic.detach().to("cpu").numpy()
            # 8-connectivity: a mover's leading edge is a thin diagonal arc
            # (one cell per row); under default 4-connectivity it shatters
            # into size-1..3 fragments and min_cluster rejects the whole
            # pedestrian (measured on a synthetic 0.35 m cylinder at 1 m/s:
            # 0/13 frames detected).  Diagonal adjacency is the correct
            # notion of "coherent blob" for arcs.
            import numpy as _np
            lab, n = _cc_label(dyn_np, structure=_np.ones((3, 3), dtype=bool))
            if n > 0:
                import numpy as _np
                sizes = _np.bincount(lab.ravel())
                sizes[0] = 0  # background
                keep = sizes >= self.min_cluster
                dyn_np = keep[lab]
                dynamic = torch.from_numpy(dyn_np).to(device=dynamic.device)

        # Motion persistence: refresh the countdown where motion was seen;
        # decay elsewhere; clear wherever the cell is no longer occupied (an
        # agent that walked away leaves free space -> label drops immediately;
        # an agent that STOPPED stays occupied -> label held for hold_frames).
        # The countdown grid is world-aligned, so roll it by the same shift.
        if self._hold is None or self._hold.shape != curr.shape:
            self._hold = torch.zeros_like(curr, dtype=torch.int16)
        else:
            self._hold = torch.roll(self._hold, shifts=(-sy, -sx), dims=(0, 1))
            # rolled-in border is garbage -> zero it
            if sx > 0:
                self._hold[:, w - sx:] = 0
            elif sx < 0:
                self._hold[:, : -sx] = 0
            if sy > 0:
                self._hold[h - sy:, :] = 0
            elif sy < 0:
                self._hold[: -sy, :] = 0
        self._hold = torch.clamp(self._hold - 1, min=0)
        self._hold[dynamic] = self.hold_frames
        self._hold[~occ_curr] = 0
        dynamic_held = self._hold > 0

        self._prev = store
        self._prev_ox, self._prev_oy, self._prev_res = origin_x, origin_y, resolution
        return dynamic_held


def rasterize_points_mask(shape, origin_x, origin_y, resolution, points_xy,
                          radius_m, device="cpu"):
    """Oracle motion mask for the label-source ablation.

    Stamps a disk of ``radius_m`` at each ground-truth dynamic-agent position
    (world frame == costmap odom frame in sim) instead of running the
    temporal-difference detector.  Downstream consumers (CBF distance,
    covariance gate) are identical, so comparing detector vs oracle isolates
    label quality (C3) from the gating concept (C2): the oracle is the
    perfect-perception upper bound, independent of lidar visibility.
    """
    h, w = int(shape[0]), int(shape[1])
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    if not points_xy or resolution <= 0.0:
        return mask
    # ceil so the disk always covers the physical agent footprint (round would
    # quantize 0.35 m / 0.1 m down to 3 cells through FP representation)
    r_cells = max(int(math.ceil(float(radius_m) / float(resolution) - 1e-6)), 1)
    r2 = r_cells * r_cells
    for px, py in points_xy:
        gx = int(round((float(px) - origin_x) / resolution))
        gy = int(round((float(py) - origin_y) / resolution))
        x0, x1 = max(gx - r_cells, 0), min(gx + r_cells + 1, w)
        y0, y1 = max(gy - r_cells, 0), min(gy + r_cells + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue  # agent outside the local window
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=device),
            torch.arange(x0, x1, device=device),
            indexing="ij",
        )
        d2 = (yy - gy) ** 2 + (xx - gx) ** 2
        mask[y0:y1, x0:x1] |= d2 <= r2
    return mask


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
