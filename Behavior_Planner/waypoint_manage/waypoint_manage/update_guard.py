"""가드형 waypoint 갱신 판정 (ROS 비의존 — 단위테스트 대상).

localization(map->odom) 보정을 컨트롤러 목표에 반영하되, FGO 가 틀렸을 때의
방어선 역할을 한다:

  새 타깃(노드 전환)           -> 즉시 교체 (보정이 아니라 새 목표)
  같은 타깃의 재변환(보정)      -> 아래 가드 통과분만, 슬루 제한으로 반영
    1) 상태 게이트: /fgo/status != OK -> 동결 (마지막 좌표 유지)
    2) 데드밴드:   |Δ| < deadband    -> 무시 (노이즈로 목표 흔들지 않음)
    3) 새너티:     |Δ| > sanity      -> 거부 + 경고 (localization 점프 의심)
    4) 슬루 제한:  통과분도 초당 slew_rate 이내로만 이동 (스텝 -> 램프)
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

Point = Tuple[float, float]


class GuardDecision(Enum):
    PUBLISH_NEW = "publish_new"       # 새 목표 (즉시 교체)
    HOLD = "hold"                     # 갱신 없음 (기존 좌표 재발행/유지)
    SLEW = "slew"                     # 보정 반영 (슬루 적용된 좌표)
    REJECT = "reject"                 # 새너티 위반 (기존 유지 + 경고)
    FROZEN = "frozen"                 # 상태 게이트 (기존 유지)


@dataclass
class GuardParams:
    deadband_m: float = 0.3
    sanity_m: float = 3.0
    slew_rate_mps: float = 0.5
    freeze_when_not_ok: bool = True


@dataclass
class GuardResult:
    decision: GuardDecision
    # SLEW/PUBLISH_NEW 일 때: 발행할 current goal 위치 (map->odom 보정 반영분)
    position: Optional[Point] = None
    # 진단용
    delta_m: float = 0.0
    note: str = ""


class UpdateGuard:
    """current goal 위치에 대한 갱신 판정. next waypoints 는 같은 보정 벡터를 공유하므로
    current goal 의 판정/슬루 계수를 그대로 적용하면 된다 (호출측 책임)."""

    def __init__(self, params: GuardParams):
        self.params = params

    def decide(
        self,
        prev_position: Optional[Point],       # 마지막으로 발행한 current goal (odom)
        candidate_position: Point,            # 최신 map->odom 으로 재변환한 좌표 (odom)
        same_target: bool,                    # 노드 ID 가 같은가 (보정) vs 다른가 (새 목표)
        localization_ok: bool,                # /fgo/status == "OK"
        dt_sec: float,                        # 직전 판정 이후 경과 시간
    ) -> GuardResult:
        if prev_position is None or not same_target:
            return GuardResult(GuardDecision.PUBLISH_NEW, candidate_position)

        dx = candidate_position[0] - prev_position[0]
        dy = candidate_position[1] - prev_position[1]
        delta = math.hypot(dx, dy)

        if self.params.freeze_when_not_ok and not localization_ok:
            return GuardResult(
                GuardDecision.FROZEN, delta_m=delta,
                note="localization not OK — waypoint frozen")

        if delta < self.params.deadband_m:
            return GuardResult(GuardDecision.HOLD, delta_m=delta)

        if delta > self.params.sanity_m:
            return GuardResult(
                GuardDecision.REJECT, delta_m=delta,
                note=f"correction {delta:.2f}m > sanity {self.params.sanity_m}m — "
                     "localization jump suspected")

        # 슬루: dt 동안 최대 slew_rate*dt 만 이동
        max_step = self.params.slew_rate_mps * max(dt_sec, 0.0)
        if delta <= max_step or max_step <= 0.0:
            new_pos = candidate_position
        else:
            ratio = max_step / delta
            new_pos = (prev_position[0] + dx * ratio, prev_position[1] + dy * ratio)
        return GuardResult(GuardDecision.SLEW, new_pos, delta_m=delta)

    def correction_ratio(self, result: GuardResult, prev: Point, candidate: Point) -> float:
        """SLEW 결과가 후보 대비 얼마나 이동했는지 (next waypoints 에 동일 적용용)."""
        if result.decision != GuardDecision.SLEW or result.position is None:
            return 1.0 if result.decision == GuardDecision.PUBLISH_NEW else 0.0
        full = math.hypot(candidate[0] - prev[0], candidate[1] - prev[1])
        if full < 1e-9:
            return 1.0
        done = math.hypot(result.position[0] - prev[0], result.position[1] - prev[1])
        return min(done / full, 1.0)
