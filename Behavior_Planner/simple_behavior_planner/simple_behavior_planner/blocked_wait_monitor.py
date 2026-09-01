#!/usr/bin/env python3
"""
Blocked-Wait Monitor Module

Sidewalk policy for "obstacle ahead, no in-corridor avoidance path":
instead of letting the optimizer squeeze past toward the curb, the vehicle
holds, waits (pedestrians usually clear on their own), then escalates:

    NORMAL --(no progress for blocked_detect_sec while it should be moving)-->
    BLOCKED_WAIT --(progress resumes)--> NORMAL
                 --(wait_timeout)------> CREEP   (creep-speed retry)
    CREEP        --(progress resumes)--> NORMAL  (restore behavior params)
                 --(creep_timeout)-----> ASSIST  (announce, back to waiting)
    ASSIST       --(progress resumes)--> NORMAL

The monitor never publishes velocity itself; it only
  - announces state on /behavior_status (std_msgs/String)
  - switches MPPI to creep speed via the existing MPPIParams channel
  - requests operator attention on /blocked_assist_request in ASSIST.
MPPI keeps optimizing the whole time, so as soon as the blocker moves and
trajectory costs drop, odometry progress resumes and the monitor restores
normal parameters.

Time is injected through update(now, ...) so the state machine is unit-testable
without ROS.
"""

from typing import Optional


class BlockedWaitMonitor:
    NORMAL = 'NORMAL'
    BLOCKED_WAIT = 'BLOCKED_WAIT'
    CREEP = 'CREEP'
    ASSIST = 'ASSIST'

    def __init__(self,
                 blocked_detect_sec: float = 4.0,
                 progress_eps: float = 0.15,
                 wait_timeout: float = 12.0,
                 creep_timeout: float = 10.0,
                 creep_speed: float = 0.3,
                 assist_repeat_sec: float = 15.0,
                 probe_enabled: bool = False,
                 probe_max_dist: float = 2.0,
                 probe_max_fails: int = 2,
                 probe_warm_wait: float = 3.0,
                 probe_warm_window: float = 30.0):
        self.blocked_detect_sec = float(blocked_detect_sec)
        self.progress_eps = float(progress_eps)
        self.wait_timeout = float(wait_timeout)
        self.creep_timeout = float(creep_timeout)
        self.creep_speed = float(creep_speed)
        self.assist_repeat_sec = float(assist_repeat_sec)

        # probe advance (2026-08-22): 원거리 가짜 벽(램프 오판)은 접근하면
        # 소멸한다 — far_wall 패턴일 때 CREEP 을 '탐침 전진'으로 승격해
        # 예산(probe_max_dist)만큼 진행을 NORMAL 복귀 없이 유지한다.
        # 근거리 안전은 컨트롤러(probe 중 근거리 lethal hard 유지)가 보장.
        self.probe_enabled = bool(probe_enabled)
        self.probe_max_dist = float(probe_max_dist)
        self.probe_max_fails = int(probe_max_fails)
        self.probe_warm_wait = float(probe_warm_wait)
        self.probe_warm_window = float(probe_warm_window)
        self.probe_active = False
        self._probe_start_xy = None
        self._probe_fails = 0
        self._warm_until: Optional[float] = None

        self.state = self.NORMAL
        self._state_since: Optional[float] = None
        self._anchor_xy = None          # position when the stall window started
        self._anchor_t: Optional[float] = None
        self._last_assist_t: Optional[float] = None

    # ---------- helpers ----------

    @staticmethod
    def _dist2(a, b) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def _enter(self, state: str, now: float):
        self.state = state
        self._state_since = now

    def _reset_anchor(self, now: float, xy):
        self._anchor_xy = xy
        self._anchor_t = now

    # ---------- main tick ----------

    def update(self, now: float, xy, should_be_moving: bool,
               far_wall: bool = False) -> list:
        """Advance the state machine one tick.

        Args:
            now: monotonic seconds
            xy: (x, y) robot position, or None if unknown
            should_be_moving: path following is active, no node-type/safety
                pause, goal not reached — i.e. a stall means "blocked".
            far_wall: 차단 패턴이 '근거리 청정 + 원거리 전폭 lethal'
                (호스트가 costmap 으로 분류) — probe 승격 조건.

        Returns:
            list of action strings for the host node:
              'announce:<STATE>'   state changed (publish /behavior_status)
              'creep_on'           switch MPPI to creep speed
              'creep_off'          restore behavior parameters
              'assist'             publish /blocked_assist_request
              'probe_on'           원거리 lethal 완화 요청 (컨트롤러)
              'probe_off'          probe 해제
        """
        actions = []
        if xy is None:
            return actions

        if not should_be_moving:
            # planner-intended stop (pause node, safety stop, goal) — not a block
            if self.state != self.NORMAL:
                actions += self._to_normal(now)
            self._reset_anchor(now, xy)
            return actions

        if self._anchor_xy is None:
            self._reset_anchor(now, xy)
            return actions

        progressed = self._dist2(xy, self._anchor_xy) > self.progress_eps ** 2
        if progressed and self.probe_active:
            # probe 중 진행은 예산까지 유지 (0.15 m 마다 NORMAL 복귀하면
            # 벽이 재형성될 때마다 4+12 s 를 다시 기다려 램프를 못 오른다)
            moved2 = self._dist2(xy, self._probe_start_xy)
            if moved2 >= self.probe_max_dist ** 2:
                # 예산 소진 + 실제 전진 = 성공 종료
                self._probe_fails = 0
                self._warm_until = now + self.probe_warm_window
                actions += self._to_normal(now)
                self._reset_anchor(now, xy)
            return actions
        if progressed:
            # moving again — sliding anchor & recover state
            if self.state != self.NORMAL:
                actions += self._to_normal(now)
            self._reset_anchor(now, xy)
            return actions

        stalled_for = now - self._anchor_t

        if self.state == self.NORMAL:
            if stalled_for >= self.blocked_detect_sec:
                self._enter(self.BLOCKED_WAIT, now)
                actions.append(f'announce:{self.BLOCKED_WAIT}')

        elif self.state == self.BLOCKED_WAIT:
            eff_wait = self.wait_timeout
            if (self.probe_enabled and far_wall
                    and self._warm_until is not None and now < self._warm_until):
                eff_wait = self.probe_warm_wait   # 직전 probe 성공 — 빠른 재진입
            if now - self._state_since >= eff_wait:
                self._enter(self.CREEP, now)
                actions.append(f'announce:{self.CREEP}')
                actions.append('creep_on')
                if (self.probe_enabled and far_wall
                        and self._probe_fails < self.probe_max_fails):
                    self.probe_active = True
                    self._probe_start_xy = xy
                    actions.append('probe_on')

        elif self.state == self.CREEP:
            if now - self._state_since >= self.creep_timeout:
                if self.probe_active:
                    self._probe_fails += 1     # 예산 내 전진 실패 (실벽 추정)
                    self.probe_active = False
                    self._probe_start_xy = None
                    actions.append('probe_off')
                self._enter(self.ASSIST, now)
                actions.append(f'announce:{self.ASSIST}')
                actions.append('creep_off')
                actions.append('assist')
                self._last_assist_t = now

        elif self.state == self.ASSIST:
            if (self._last_assist_t is None or
                    now - self._last_assist_t >= self.assist_repeat_sec):
                actions.append('assist')
                self._last_assist_t = now

        return actions

    def _to_normal(self, now: float) -> list:
        actions = []
        if self.probe_active:
            self.probe_active = False
            self._probe_start_xy = None
            actions.append('probe_off')
        if self.state == self.CREEP:
            actions.append('creep_off')
        actions.append(f'announce:{self.NORMAL}')
        self._enter(self.NORMAL, now)
        self._last_assist_t = None
        return actions
