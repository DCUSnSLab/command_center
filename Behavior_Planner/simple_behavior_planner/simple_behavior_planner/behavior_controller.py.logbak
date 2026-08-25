#!/usr/bin/env python3
"""
Behavior Controller Module
행동별 파라미터 제어를 담당하는 모듈
"""

import os
import yaml
from typing import Optional
from rclpy.node import Node
from std_msgs.msg import Header
from command_center_interfaces.msg import MPPIParams

from .behavior_parameter_manager import BehaviorParameterManager


class BehaviorController:
    """행동 제어 클래스"""

    def __init__(self, node: Node, config_path: str = 'behavior_modifiers.yaml'):
        self.node = node
        self.current_node_type = 1
        self.previous_node_type = 1

        # Parameter manager 초기화
        self.param_manager: Optional[BehaviorParameterManager] = None
        self._init_parameter_manager(config_path)

        # MPPI parameter publisher will be set by main node
        self.mppi_param_pub = None

    def set_mppi_publisher(self, publisher):
        """MPPI 파라미터 publisher 설정"""
        self.mppi_param_pub = publisher

    def _init_parameter_manager(self, config_path: str):
        """파라미터 매니저 초기화"""
        try:
            # Package path 해결
            full_config_path = self._resolve_config_path(config_path)

            if not os.path.exists(full_config_path):
                self.node.get_logger().error(f"Behavior config file not found: {full_config_path}")
                return

            # Load behavior config
            with open(full_config_path, 'r') as file:
                config = yaml.safe_load(file)

            behavior_config = config.get('/**', {}).get('ros__parameters', {})
            smppi_config_path = behavior_config.get('smppi_config_path', 'smppi_params.yaml')

            # Initialize parameter manager
            self.param_manager = BehaviorParameterManager(smppi_config_path, behavior_config)

            self.node.get_logger().info(f"Behavior controller initialized with config: {full_config_path}")
            self._log_available_behaviors()

        except Exception as e:
            self.node.get_logger().error(f"Failed to initialize behavior controller: {e}")
            self.param_manager = None

    def _resolve_config_path(self, config_path: str) -> str:
        """설정 파일 경로 해결"""
        try:
            from ament_index_python.packages import get_package_share_directory
            package_share_path = get_package_share_directory('simple_behavior_planner')
            return os.path.join(package_share_path, 'config', config_path)
        except Exception:
            # Fallback to relative path
            package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(package_path, 'config', config_path)

    def _log_available_behaviors(self):
        """사용 가능한 행동들 로깅"""
        if not self.param_manager:
            return

        behaviors = self.param_manager.get_available_behaviors()
        self.node.get_logger().info(f"Available behaviors: {len(behaviors)}")
        for behavior_type, description in behaviors.items():
            self.node.get_logger().info(f"  {behavior_type}: {description}")

    def update_behavior(self, node_type: int) -> bool:
        """행동 업데이트"""
        if not self.param_manager or not self.mppi_param_pub:
            return False

        if node_type == self.current_node_type:
            return False  # No change needed

        try:
            # Get behavior parameters
            behavior_params = self.param_manager.get_behavior_params(node_type)

            # Validate parameters
            if not self.param_manager.validate_behavior_params(behavior_params):
                self.node.get_logger().error(f"Invalid parameters for behavior {node_type}")
                return False

            # Log parameter change
            self._log_parameter_change(node_type, behavior_params)

            # Send parameters to MPPI
            self._send_mppi_parameters(behavior_params)

            # Update state
            self.previous_node_type = self.current_node_type
            self.current_node_type = node_type

            return True

        except Exception as e:
            self.node.get_logger().error(f"Failed to update behavior: {e}")
            return False

    def _log_parameter_change(self, node_type: int, params: dict):
        """파라미터 변경 로깅"""
        if not self.param_manager:
            return

        behavior_desc = self.param_manager.get_behavior_description(node_type)

        # Log key parameter changes
        key_params = {
            'max_linear_velocity': params.get('max_linear_velocity', 'N/A'),
            'min_linear_velocity': params.get('min_linear_velocity', 'N/A'),
            'respect_reverse_heading': params.get('respect_reverse_heading', False),
            'goal_weight': params.get('goal_weight', 'N/A')
        }

        self.node.get_logger().info(f"🔄 [BEHAVIOR UPDATE] {self.current_node_type}->{node_type}: {behavior_desc}")
        for param, value in key_params.items():
            if value != 'N/A':
                self.node.get_logger().info(f"   {param}: {value}")

    def _send_mppi_parameters(self, behavior_params: dict):
        """MPPI 파라미터 전송"""
        try:
            msg = MPPIParams()
            msg.header = Header()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = f"behavior_{behavior_params.get('behavior_type', 1)}"

            # Vehicle parameters
            if 'max_linear_velocity' in behavior_params:
                msg.update_vehicle = True
                msg.max_linear_velocity = behavior_params['max_linear_velocity']
                msg.min_linear_velocity = behavior_params.get('min_linear_velocity', 0.0)
                msg.max_angular_velocity = behavior_params.get('max_angular_velocity', 1.16)
                msg.min_angular_velocity = behavior_params.get('min_angular_velocity', -1.16)
                msg.wheelbase = behavior_params.get('wheelbase', 0.65)
                msg.max_steering_angle = behavior_params.get('max_steering_angle', 0.3665)
                msg.radius = behavior_params.get('radius', 0.6)
                msg.footprint_padding = behavior_params.get('footprint_padding', 0.15)

            # Cost weights
            if 'goal_weight' in behavior_params or 'obstacle_weight' in behavior_params:
                msg.update_costs = True
                msg.goal_weight = behavior_params.get('goal_weight', 30.0)
                msg.obstacle_weight = behavior_params.get('obstacle_weight', 100.0)

            # Lookahead parameters
            lookahead_params = ['lookahead_base_distance', 'lookahead_velocity_factor',
                              'lookahead_min_distance', 'lookahead_max_distance']
            if any(param in behavior_params for param in lookahead_params):
                msg.update_lookahead = True
                msg.lookahead_base_distance = behavior_params.get('lookahead_base_distance', 1.0)
                msg.lookahead_velocity_factor = behavior_params.get('lookahead_velocity_factor', 0.4)
                msg.lookahead_min_distance = behavior_params.get('lookahead_min_distance', 1.0)
                msg.lookahead_max_distance = behavior_params.get('lookahead_max_distance', 6.0)

            # Goal critic parameters
            msg.update_goal_critic = True
            msg.respect_reverse_heading = behavior_params.get('respect_reverse_heading', False)

            # Control parameters
            msg.update_control = True
            msg.goal_reached_threshold = behavior_params.get('goal_reached_threshold', 2.0)
            msg.control_frequency = behavior_params.get('control_frequency', 20.0)
            msg.force_stop = False  # Resume normal operation

            # Behavior info
            msg.current_behavior_type = behavior_params.get('behavior_type', 1)
            msg.current_behavior_desc = behavior_params.get('behavior_description', 'Unknown behavior')

            # Publish
            self.mppi_param_pub.publish(msg)

        except Exception as e:
            self.node.get_logger().error(f"Failed to send MPPI parameters: {e}")

    def get_current_behavior_type(self) -> int:
        """현재 행동 타입 반환"""
        return self.current_node_type

    def is_enabled(self) -> bool:
        """행동 제어가 활성화되어 있는지 확인"""
        return self.param_manager is not None