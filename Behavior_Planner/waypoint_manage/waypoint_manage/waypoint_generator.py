"""waypoint 재계산/생성 (ROS 비의존 골격).

입력: map 프레임의 노드 시퀀스 (current + lookahead)
출력: map 프레임의 waypoint 시퀀스 (컨트롤러에 줄 최종 기하)

v0 은 passthrough. 재계산 '결정'(어떤 모드를 쓸지)은 BP/BT 가 TargetWaypoints.recalc_mode
로 내려보내고, 여기서는 '실행'만 한다.
"""

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MapWaypoint:
    node_id: str
    x: float            # map 프레임
    y: float
    z: float
    yaw: float          # map 프레임 [rad]
    node_type: int


class WaypointGenerator:
    def generate(self, mode: str, waypoints: List[MapWaypoint]) -> List[MapWaypoint]:
        if not waypoints:
            return []
        if mode in ("", "passthrough"):
            return waypoints
        if mode == "densify":
            return self._densify(waypoints)
        if mode == "offset":
            return self._offset(waypoints)
        # 미지 모드는 안전하게 passthrough
        return waypoints

    # ---- TODO: 구현 골격 ----

    def _densify(self, waypoints: List[MapWaypoint],
                 spacing_m: float = 1.0) -> List[MapWaypoint]:
        """노드 간격(~5m)을 spacing 간격 중간점으로 촘촘히.
        TODO: 링크 곡률 반영(현재는 선형 보간), yaw 는 진행방향으로 재계산."""
        out: List[MapWaypoint] = []
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            seg = math.hypot(b.x - a.x, b.y - a.y)
            n = max(int(seg / spacing_m), 1)
            for i in range(n):
                t = i / n
                yaw = math.atan2(b.y - a.y, b.x - a.x)
                out.append(MapWaypoint(
                    node_id=f"{a.node_id}+{i}" if i else a.node_id,
                    x=a.x + (b.x - a.x) * t, y=a.y + (b.y - a.y) * t,
                    z=a.z + (b.z - a.z) * t, yaw=yaw, node_type=a.node_type))
        out.append(waypoints[-1])
        return out

    def _offset(self, waypoints: List[MapWaypoint],
                lateral_m: float = 0.0) -> List[MapWaypoint]:
        """경로 횡방향 오프셋 (장애물 회피 보조 등).
        TODO: BT 가 내려주는 오프셋 양/방향 파라미터화, costmap 연동."""
        if abs(lateral_m) < 1e-6:
            return waypoints
        out = []
        for w in waypoints:
            nx, ny = -math.sin(w.yaw), math.cos(w.yaw)   # 좌측 법선
            out.append(MapWaypoint(
                w.node_id, w.x + nx * lateral_m, w.y + ny * lateral_m,
                w.z, w.yaw, w.node_type))
        return out
