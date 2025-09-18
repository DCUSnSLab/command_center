#!/usr/bin/env python3
"""
Path Manager Module
경로 데이터 관리 및 노드 추적을 담당
"""

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