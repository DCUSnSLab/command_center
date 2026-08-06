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

    def update_path(self, planned_path: PlannedPath) -> None:
        """새로운 경로로 업데이트"""
        self.planned_path = planned_path
        self.path_nodes = self._extract_path_nodes(planned_path)
        self._reset_path_state()

    def _extract_path_nodes(self, planned_path: PlannedPath) -> List[Dict[str, Any]]:
        """PlannedPath 메시지에서 노드 정보 추출"""
        nodes = []
        for node in planned_path.path_data.nodes:
            node_data = {
                'id': node.id,
                'x': node.utm_info.easting,
                'y': node.utm_info.northing,
                'z': node.gps_info.alt,
                'node_type': node.node_type,
                'heading': node.heading
            }
            nodes.append(node_data)
        return nodes

    def _reset_path_state(self) -> None:
        """경로 상태 초기화"""
        self.current_target_index = 0
        self.is_path_following = True
        self.last_completed_goal_id = None

    def align_to_position(self, x: float, y: float,
                          max_approach_m: float = 25.0,
                          approach_weight: float = 1.5,
                          max_skip_nodes: int = 2) -> int:
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
        """
        if not self.path_nodes:
            return 0
        n = len(self.path_nodes)
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

        best_i, best_cost = None, None
        for i, node in enumerate(self.path_nodes):
            if i < i_near or i > i_max:
                continue
            d = math.hypot(node['x'] - x, node['y'] - y)
            if d > max_approach_m:
                continue
            cost = approach_weight * d + rem[i]
            if best_cost is None or cost < best_cost:
                best_i, best_cost = i, cost
        if best_i is None:      # 전부 상한 초과 — 최근접으로 합류
            best_i = i_near
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