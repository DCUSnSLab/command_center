"""C4 — map-anchor interlock: don't drive on an unanchored map frame.

Until the map EKF fuses its first absolute position (receiver still cold-
starting, or every fix dropped by gps_fix_gate), it believes it is AT the
datum: navsat_transform uses a fixed datum = graph-map node[0] and the filter
initialises at the map origin. So the map-frame pose is wrong by exactly the
datum -> actual-start-position offset, and the route and corridor keepout —
which live in the map frame — are displaced by that same amount. The L2
corridor keepout, the layer that keeps the vehicle off the road, is therefore
void for as long as this lasts.

Measured in the chamber (2026-07-17, continuous probe, cold-start scenario):
spawned 3.0 m from the datum, pre-anchor map error was 3.01 m mean / 3.03 m
max, collapsing to 0.21 m after the first gate-passed fix — i.e. the error IS
the offset, exactly. Spawned ON the datum it is 0.03 m, which is why this
never showed up before: the sim used to start exactly at node[0]. The real
vehicle never does — on 2026-07-14 it started ~150 m from the map route.

The interlock only covers STARTUP: hold until the map frame has been anchored
once. It deliberately does NOT re-engage on later GPS loss — driving through
outages on FAST-LIO + wheel + IMU is the whole point of the dual-EKF design,
and re-blocking mid-run would strand the vehicle in exactly the tree-canopy
zones it is meant to traverse.

Time is injected through update(now, ...) so the logic is unit-testable
without ROS (same pattern as BlockedWaitMonitor).
"""


class LocalizationAnchorMonitor:
    def __init__(self, global_timeout: float = 1.0):
        # /odometry/global older than this counts as "no localization"
        self.global_timeout = float(global_timeout)
        self.anchored = False          # latches once the map frame is anchored
        self.last_anchor_t = None      # last accepted (gate-passed) fix
        self.last_global_t = None      # last /odometry/global message
        self._announced = False

    def on_anchor_fix(self, now: float):
        """A fix that PASSED gps_fix_gate — i.e. a sane absolute position."""
        self.last_anchor_t = now

    def on_global_odom(self, now: float):
        self.last_global_t = now

    def update(self, now: float):
        """-> (allow_drive, reason). reason is None once driving is allowed."""
        if self.anchored:
            return True, None
        if self.last_anchor_t is None:
            return False, 'map frame never anchored (no gate-passed GNSS fix yet)'
        if self.last_global_t is None:
            return False, 'no /odometry/global yet'
        age = now - self.last_global_t
        if age > self.global_timeout:
            return False, f'/odometry/global stale ({age:.1f}s)'
        self.anchored = True           # latch: outages after this are expected
        return True, None
