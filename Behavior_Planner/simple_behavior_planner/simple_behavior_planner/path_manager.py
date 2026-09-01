#!/usr/bin/env python3
"""
Path Manager Module
경로 데이터 관리 및 노드 추적을 담당
"""

import math
from typing import Optional, List, Dict, Any
from command_center_interfaces.msg import PlannedPath


class PathManager:
    """경로 관리 클래스"""

    def __init__(self):
        self.planned_path: Optional[PlannedPath] = None
        self.path_nodes: List[Dict[str, Any]] = []
        self.current_target_index = 0
        self.is_path_following = False
        self.last_completed_goal_id = None
        # 실제로 곁을 지나온 노드의 최대 인덱스. -1 = 아직 아무 데도 안 지남.
        self.max_passed_index = -1

    def update_path(self, planned_path: PlannedPath) -> None:
        """새로운 경로로 업데이트"""
        self.planned_path = planned_path
        self.path_nodes = self._extract_path_nodes(planned_path)
        self._reset_path_state()

    def _extract_path_nodes(self, planned_path: PlannedPath) -> List[Dict[str, Any]]:
        """PlannedPath 메시지에서 노드 정보 추출"""
        nodes = []
        for node in planned_path.path_data.nodes:
            # map_interfaces/GraphLayer MapNode: 좌표 평탄(easting/northing), heading_deg, alt 없음
            node_data = {
                'id': node.id,
                'x': node.easting,
                'y': node.northing,
                'z': 0.0,
                'node_type': node.node_type,
                'heading': node.heading_deg
            }
            nodes.append(node_data)
        return nodes

    def _reset_path_state(self) -> None:
        """경로 상태 초기화"""
        self.current_target_index = 0
        self.is_path_following = True
        self.last_completed_goal_id = None
        self.max_passed_index = -1

    def note_position(self, x: float, y: float, pass_radius: float = 2.0) -> int:
        """현재 위치를 통과 이력에 반영. 지나온 노드의 최대 인덱스를 돌려준다.

        "합류해도 되는 앞쪽 노드"와 "가깝기만 한 앞쪽 노드"는 기하만으로
        구분되지 않는다. 둘 다 최근접이 앞쪽에 있기 때문이다:

          2026-08-06 필드  경로 옆 20 m 주행 -> 앞 노드들을 **지나온 게 맞다**
          2026-08-11 챔버  경로 옆 6 m 이격  -> 중간 노드를 **지나온 적 없다**

        갈라놓는 것은 통과 이력이다. 차량이 pass_radius 안으로 들어와 본
        노드만 '지나왔다'고 보고, 합류는 거기서 한 칸까지만 허용한다.
        """
        for i, nd in enumerate(self.path_nodes):
            if i <= self.max_passed_index:
                continue
            if math.hypot(nd['x'] - x, nd['y'] - y) <= pass_radius:
                self.max_passed_index = i
        return self.max_passed_index

    def align_to_position(self, x: float, y: float,
                          max_approach_m: float = 25.0,
                          approach_weight: float = 1.5,
                          max_skip_nodes: int = 2,
                          max_skip_from_current: int = 0,
                          approach_clear=None,
                          limit_to_passed: bool = True,
                          pass_radius: float = 2.0) -> int:
        """시작 노드 미지정 시 경로에 합류할 노드를 고른다.

        최근접이 아니라 **접근거리 + 그 노드부터 목표까지 남은 경로거리**가
        최소인 노드를 고른다. 최근접 규칙은 목표 방향을 보지 않기 때문에
        경로의 '꼬리'가 우연히 가까우면 목표 반대편으로 먼저 달리게 된다 —
        2026-08-05 필드 실측: 차량 (55.2, 40.8), 목표 N0009 는 북서 +151.6도
        인데 최근접 노드 N0020 은 남서 -157.6도라 자율 전환 직후 좌회전해
        목표에서 멀어졌다. 총비용은 N0020 75.4 m vs 최적 65.4 m 로 10 m 손해.

        max_approach_m: 접근거리 상한. 합류점을 너무 멀리 잡으면 경로 이탈
        구간이 길어져 corridor keepout 과 충돌한다. 상한을 넘는 후보는
        고려하지 않되, 전부 초과하면 최근접으로 되돌아간다(합류 불가 방지).

        approach_weight: 경로 밖 주행에 붙이는 가중치. 1.0(무가중)이면 규칙이
        **항상 경로를 가로지른다** — 꺾인 폴리라인보다 직선이 늘 짧기 때문이다
        (단위테스트에서 잡힌 회귀: 차량이 경로 머리 2 m 옆에 있는데도 18 m
        떨어진 앞쪽 노드로 합류). 경로 밖은 코리도 보호가 없어 실제로 더
        비싸므로 1 보다 크게 둔다. 1.5 는 위 회귀를 막으면서 8/5 기하에서는
        여전히 앞쪽 노드를 고르는 값이다.

        max_skip_from_current: 0 이면 끈다(기본). >0 이면 후보를 현재 진행
        인덱스 기준 그만큼으로 묶는다. **기본이 0 인 이유가 있다** — 이 상한은
        2026-08-06 필드가 필요로 하는 '따라잡기'를 막는다. 그날 차량은 경로
        옆을 20 m RC 주행했고(노드 간격 1.8 m) 노드에 닿지 않아 진행 인덱스가
        0 인 채였다. 여기서 0+2 로 묶으면 자율 전환 시 경로 머리로 되돌아가는
        그날 오전의 실패가 그대로 재현된다. 챔버에서 드러난 반대쪽 결함(경로를
        통째로 건너뛰고 지름길을 시도)에는 approach_clear 쪽이 맞다.

        approach_clear: (x0,y0,x1,y1) -> bool. 현재 위치에서 후보 노드까지
        직선이 통행 가능한지 묻는다. None 이면 검사하지 않는다(기본). 챔버
        2026-08-11 실측: 차량 (6,1) 에서 최근접이 경로 끝 P3(3,5) 라
        max_skip_nodes 가 아무것도 막지 못했고, 그 직선 위 장애물 3개
        (obs_226/227/228, x 4.75~5.25, y 2.25~2.75) 때문에 3/3 미도달했다.
        경로 P0->P1->P2->P3 는 바로 그 구역을 우회하려고 그려진 것이다.
        """
        if not self.path_nodes:
            return 0
        n = len(self.path_nodes)
        if limit_to_passed:
            self.note_position(x, y, pass_radius)
        # 각 노드에서 경로 끝(목표)까지 남은 거리
        rem = [0.0] * n
        for i in range(n - 2, -1, -1):
            a, b = self.path_nodes[i], self.path_nodes[i + 1]
            rem[i] = rem[i + 1] + math.hypot(b['x'] - a['x'], b['y'] - a['y'])

        # 최근접 노드 — 절대 상한만 두면 직선 경로에서 규칙이 목표 끝까지
        # 건너뛴다(경로가 직선이면 앞으로 갈수록 항상 총비용이 준다).
        # 최근접 기준 몇 노드까지만 허용해 거동을 예측 가능하게 묶는다.
        d2 = [(nd['x'] - x) ** 2 + (nd['y'] - y) ** 2 for nd in self.path_nodes]
        i_near = d2.index(min(d2))
        i_max = min(n - 1, i_near + max_skip_nodes)
        i_lo = i_near
        # 통과 이력이 **있을 때만** 게이트한다. 이력이 없다는 건 아직 경로에
        # 붙어 본 적이 없다는 뜻이고, 그때는 앞쪽 합류가 맞다 — 2026-08-05
        # 필드가 그 경우다(차량이 경로 옆에 놓인 채 기동, 경로 머리는 뒤쪽).
        # 여기까지 게이트하면 그날 고친 '경로 머리로 되돌아가기'가 되살아난다.
        # 게이트의 대상은 **이동 후 따라잡기**이지 최초 합류가 아니다.
        gate = limit_to_passed and self.max_passed_index >= 0
        if gate:
            # 지나온 마지막 노드의 **다음 칸**까지만 허용한다.
            i_max = min(i_max, self.max_passed_index + 1)
            i_lo = min(i_lo, i_max)
        if max_skip_from_current > 0:
            i_lo = min(i_lo, self.current_target_index)
            i_max = min(i_max, self.current_target_index + max_skip_from_current)

        def pick(lo, hi, use_clear):
            bi, bc = None, None
            for i in range(lo, hi + 1):
                node = self.path_nodes[i]
                d = math.hypot(node['x'] - x, node['y'] - y)
                if d > max_approach_m:
                    continue
                # 접근 직선이 막혀 있으면 후보에서 뺀다. 합류란 곧 "여기서 저
                # 노드까지 경로 밖을 직진한다"는 뜻이므로, 그 직선이 통행
                # 불가면 그 노드는 합류점이 될 수 없다.
                if use_clear and not approach_clear(x, y, node['x'], node['y']):
                    continue
                cost = approach_weight * d + rem[i]
                if bc is None or cost < bc:
                    bi, bc = i, cost
            return bi

        use_clear = approach_clear is not None
        best_i = pick(i_lo, i_max, use_clear)
        if best_i is None and use_clear:
            # 창 안이 전부 막혔다 — 창을 풀고 **닿을 수 있는** 노드를 찾는다.
            # 여기서 최근접으로 바로 떨어지면 방금 막혀서 뺀 그 노드로
            # 되돌아간다(챔버 실측: 최근접이 곧 막힌 P3 라 검사가 무효화됐다).
            best_i = pick(0, n - 1, True)
        if best_i is None:
            # 통과 후보 없음 — 합류 불가로 멎지 않게 되돌린다. 다만 최근접으로
            # 그냥 떨어지면 통과 게이트가 무효가 된다(실측: 경로에서 27 m 떨어진
            # 위치가 max_approach 를 넘겨 전부 탈락하자 지나온 적 없는 경로 끝이
            # 선택됐다). 게이트가 켜져 있으면 그 상한 안에서 되돌린다.
            best_i = (min(i_near, self.max_passed_index + 1)
                      if gate else i_near)
        self.current_target_index = best_i
        return self.current_target_index

    def get_current_target_node(self) -> Optional[Dict[str, Any]]:
        """현재 목표 노드 반환"""
        if not self.path_nodes or self.current_target_index >= len(self.path_nodes):
            return None
        return self.path_nodes[self.current_target_index]

    def get_next_nodes(self, count: int = 3) -> List[Dict[str, Any]]:
        """다음 노드들 반환 (multiple waypoints용)"""
        if not self.path_nodes:
            return []

        next_nodes = []
        for i in range(1, min(count + 1, len(self.path_nodes) - self.current_target_index)):
            next_idx = self.current_target_index + i
            if next_idx < len(self.path_nodes):
                next_nodes.append(self.path_nodes[next_idx])
        return next_nodes

    def advance_to_next_node(self) -> bool:
        """다음 노드로 진행"""
        if self.current_target_index < len(self.path_nodes) - 1:
            self.current_target_index += 1
            return True
        else:
            # 경로 완주
            self.is_path_following = False
            return False

    def mark_goal_completed(self, goal_id: str) -> None:
        """목표 완료 처리"""
        self.last_completed_goal_id = goal_id

    def get_path_info(self) -> Dict[str, Any]:
        """경로 정보 반환"""
        return {
            'path_id': self.planned_path.path_id if self.planned_path else "",
            'total_nodes': len(self.path_nodes),
            'current_index': self.current_target_index,
            'is_final_node': self.current_target_index == len(self.path_nodes) - 1,
            'is_following': self.is_path_following
        }

    def get_node_types(self) -> List[int]:
        """경로의 모든 노드 타입 반환"""
        return [node.get('node_type', 1) for node in self.path_nodes]