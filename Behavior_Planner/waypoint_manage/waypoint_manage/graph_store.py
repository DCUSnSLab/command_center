"""graph(노드 좌표)와 datum 을 보관하고 노드 ID -> map 좌표를 해석.

소스: /map_provider_node/graph (GraphLayer, latched — 변경 시 재수신)
      /map_provider_node/utm   (UtmLayer, latched)
좌표 규약: MapNode 는 절대 UTM. map 프레임 = UTM - datum.
"""

import math
from typing import Dict, List, Optional

from .waypoint_generator import MapWaypoint


class GraphStore:
    def __init__(self):
        self._nodes: Dict[str, dict] = {}
        self._origin_easting: Optional[float] = None
        self._origin_northing: Optional[float] = None

    # ---- 콜백에서 호출 ----
    def set_graph(self, graph_msg) -> int:
        """map_interfaces/GraphLayer 수신. 반환: 노드 수."""
        self._nodes = {
            n.id: {
                "easting": n.easting,
                "northing": n.northing,
                "heading_deg": n.heading_deg,
                "node_type": int(n.node_type),
            }
            for n in graph_msg.nodes
        }
        return len(self._nodes)

    def set_datum(self, utm_msg) -> None:
        self._origin_easting = utm_msg.origin_easting
        self._origin_northing = utm_msg.origin_northing

    # ---- 조회 ----
    def ready(self) -> bool:
        return bool(self._nodes) and self._origin_easting is not None

    def resolve(self, node_id: str) -> Optional[MapWaypoint]:
        """노드 ID -> map 프레임 waypoint. 미지 ID 는 None."""
        n = self._nodes.get(node_id)
        if n is None or self._origin_easting is None:
            return None
        # heading(deg, 그래프 규약) -> map yaw [rad]
        #   기존 waypoint_publisher._convert_geographic_to_odom_heading 이식:
        #   단순 % 360 후 라디안 (그래프 heading 이 이미 ENU 기준이라는 전제).
        #   TODO: heading 규약(진북 CW vs ENU CCW) 확정 후 정리.
        yaw = math.radians(n["heading_deg"] % 360.0)
        return MapWaypoint(
            node_id=node_id,
            x=n["easting"] - self._origin_easting,
            y=n["northing"] - self._origin_northing,
            z=0.0,
            yaw=yaw,
            node_type=n["node_type"],
        )

    def resolve_many(self, node_ids: List[str]) -> List[MapWaypoint]:
        out = []
        for nid in node_ids:
            w = self.resolve(nid)
            if w is not None:
                out.append(w)
        return out
