#!/usr/bin/env python3
"""
Safety Monitor Module
안전 감시 및 pause command 전송을 담당하는 모듈
"""

import time
from typing import Optional
from rclpy.node import Node
from std_msgs.msg import Header
from command_center_interfaces.msg import PauseCommand


class SafetyMonitor:
    """안전 감시 클래스"""

    def __init__(self, node: Node):
        self.node = node

        # Safety state
        self.current_stop_flag = False
        self.current_traffic_light_state = 0

        # Timing control
        self.last_safety_pause_time = 0.0
        self.safety_pause_interval = 0.5  # seconds

        # Current behavior type (set by main node)
        self.current_behavior_type = 1

        # Pause command publisher will be set by main node
        self.pause_command_pub = None

    def set_pause_publisher(self, publisher):
        """Pause command publisher 설정"""
        self.pause_command_pub = publisher

    def update_stop_flag(self, stop_flag: bool):
        """장애물 감지 플래그 업데이트"""
        self.current_stop_flag = stop_flag
        self.node.get_logger().debug(f'Stop flag updated: {stop_flag}')

    def update_traffic_light_state(self, state: int):
        """신호등 상태 업데이트"""
        self.current_traffic_light_state = state
        self.node.get_logger().debug(f'Traffic light state updated: {state}')

    def update_behavior_type(self, behavior_type: int):
        """현재 행동 타입 업데이트"""
        self.current_behavior_type = behavior_type

    def check_safety_conditions(self):
        """안전 조건 확인 및 필요시 pause command 전송"""
        should_pause = self._should_send_pause_command()

        if should_pause:
            current_time = time.time()
            if current_time - self.last_safety_pause_time >= self.safety_pause_interval:
                self._send_pause_command()
                self.last_safety_pause_time = current_time

    def _should_send_pause_command(self) -> bool:
        """Pause command를 보내야 하는지 판단"""
        if self.current_behavior_type == 9:
            # Node type 9: Only check obstacle detection
            return self.current_stop_flag

        elif self.current_behavior_type == 10:
            # Node type 10: Only check traffic light conditions
            return self.current_traffic_light_state in [1]  # Green light or left turn

        else:
            # Other node types: No safety stop conditions
            return False

    def _send_pause_command(self):
        """Pause command 전송"""
        if not self.pause_command_pub:
            self.node.get_logger().warn("Pause command publisher not set")
            return

        try:
            pause_msg = PauseCommand()
            pause_msg.header = Header()
            pause_msg.header.stamp = self.node.get_clock().now().to_msg()
            pause_msg.header.frame_id = 'safety_monitor'
            pause_msg.pause_duration = 2.0  # 2 second pause
            pause_msg.node_id = f"safety_stop_type_{self.current_behavior_type}"

            # Determine reason for pause
            reason = self._get_pause_reason()
            pause_msg.reason = reason

            self.pause_command_pub.publish(pause_msg)
            self.node.get_logger().info(f"Safety pause command sent: {reason}")

        except Exception as e:
            self.node.get_logger().error(f"Failed to send safety pause command: {e}")

    def _get_pause_reason(self) -> str:
        """Pause 이유 문자열 생성"""
        reasons = []

        if self.current_behavior_type == 9 and self.current_stop_flag:
            reasons.append("obstacle detected (node_type 9)")

        elif self.current_behavior_type == 10:
            if self.current_traffic_light_state == 3:
                reasons.append("green light (node_type 10)")
            if self.current_traffic_light_state == 4:
                reasons.append("left turn signal (node_type 10)")

        if reasons:
            return f"Safety stop: {', '.join(reasons)}"
        else:
            return f"Safety stop (node_type {self.current_behavior_type})"

    def get_safety_status(self) -> dict:
        """안전 상태 정보 반환"""
        return {
            'stop_flag': self.current_stop_flag,
            'traffic_light_state': self.current_traffic_light_state,
            'behavior_type': self.current_behavior_type,
            'should_pause': self._should_send_pause_command()
        }