#!/usr/bin/env python3
"""
Simple Behavior Planner Node (Refactored)
깔끔하고 모듈화된 새로운 구조
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from typing import Optional

# ROS2 messages
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String, Header, Int32

from command_center_interfaces.msg import (
    PlannedPath, ControllerGoalStatus, MultipleWaypoints,
    MPPIParams, PauseCommand
)

# Local modules
from simple_behavior_planner.path_manager import PathManager
from simple_behavior_planner.waypoint_publisher import WaypointPublisher
from simple_behavior_planner.behavior_controller import BehaviorController
from simple_behavior_planner.safety_monitor import SafetyMonitor


class SimpleBehaviorPlannerNode(Node):
    """새로운 모듈화된 Simple Behavior Planner Node"""

    def __init__(self):
        super().__init__('simple_behavior_planner')

        # Declare parameters
        self._declare_parameters()
        self._load_parameters()

        # Initialize core modules
        self.path_manager = PathManager()
        self.waypoint_publisher = WaypointPublisher(self, self.waypoint_mode)
        self.behavior_controller = BehaviorController(
            self, self.behavior_config_path) if self.enable_behavior_control else None
        self.safety_monitor = SafetyMonitor(self)

        # State variables
        self.current_pose: Optional[PoseStamped] = None
        self.subgoal_published = False
        self.emergency_stop_requested = False
        self.pause_signal_sent = False

        # Setup QoS profiles
        self._setup_qos_profiles()

        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()

        # Link modules with publishers
        self._link_module_publishers()

        # Main planning timer
        self.create_timer(0.1, self.planning_callback)  # 10Hz

        self.get_logger().info('Simple Behavior Planner Node (Refactored) initialized')
        self._log_configuration()

    def _declare_parameters(self):
        """파라미터 선언"""
        # Topic parameters
        self.declare_parameter('current_position_topic', '/odom')
        self.declare_parameter('planned_path_topic', '/planned_path_detailed')
        self.declare_parameter('goal_status_topic', '/goal_status')
        self.declare_parameter('subgoal_topic', '/subgoal')
        self.declare_parameter('multiple_waypoints_topic', '/multiple_waypoints')
        self.declare_parameter('emergency_stop_topic', '/emergency_stop')
        self.declare_parameter('stop_flag_topic', '/stop_flag')
        self.declare_parameter('traffic_light_topic', '/tl/state_id')

        # Behavior parameters
        self.declare_parameter('pause_trigger_distance', 0.8)
        self.declare_parameter('waypoint_mode', 'multiple')
        self.declare_parameter('behavior_config_path', 'behavior_modifiers.yaml')
        self.declare_parameter('enable_behavior_control', True)

    def _load_parameters(self):
        """파라미터 로드"""
        self.current_position_topic = self.get_parameter('current_position_topic').value
        self.planned_path_topic = self.get_parameter('planned_path_topic').value
        self.goal_status_topic = self.get_parameter('goal_status_topic').value
        self.subgoal_topic = self.get_parameter('subgoal_topic').value
        self.multiple_waypoints_topic = self.get_parameter('multiple_waypoints_topic').value
        self.emergency_stop_topic = self.get_parameter('emergency_stop_topic').value
        self.stop_flag_topic = self.get_parameter('stop_flag_topic').value
        self.traffic_light_topic = self.get_parameter('traffic_light_topic').value

        self.pause_trigger_distance = self.get_parameter('pause_trigger_distance').value
        self.waypoint_mode = self.get_parameter('waypoint_mode').value
        self.behavior_config_path = self.get_parameter('behavior_config_path').value
        self.enable_behavior_control = self.get_parameter('enable_behavior_control').value

        # Validate waypoint mode
        if self.waypoint_mode not in ['single', 'multiple']:
            self.get_logger().warn(f"Invalid waypoint_mode '{self.waypoint_mode}', defaulting to 'multiple'")
            self.waypoint_mode = 'multiple'

    def _setup_qos_profiles(self):
        """QoS 프로파일 설정"""
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

    def _setup_subscribers(self):
        """구독자 설정"""
        self.current_pose_sub = self.create_subscription(
            Odometry, self.current_position_topic,
            self.current_pose_callback, self.best_effort_qos)

        self.planned_path_sub = self.create_subscription(
            PlannedPath, self.planned_path_topic,
            self.planned_path_callback, self.reliable_qos)

        self.goal_status_sub = self.create_subscription(
            ControllerGoalStatus, self.goal_status_topic,
            self.goal_status_callback, self.reliable_qos)

        self.stop_flag_sub = self.create_subscription(
            Bool, self.stop_flag_topic,
            self.stop_flag_callback, self.reliable_qos)

        self.traffic_light_sub = self.create_subscription(
            Int32, self.traffic_light_topic,
            self.traffic_light_callback, self.reliable_qos)

    def _setup_publishers(self):
        """발행자 설정"""
        self.emergency_stop_pub = self.create_publisher(
            Bool, self.emergency_stop_topic, self.reliable_qos)

        self.pause_command_pub = self.create_publisher(
            PauseCommand, '/pause_command', self.reliable_qos)

        # Waypoint publishers based on mode
        if self.waypoint_mode in ['single', 'multiple']:
            if self.waypoint_mode == 'single' or self.waypoint_mode == 'multiple':
                self.subgoal_pub = self.create_publisher(
                    PoseStamped, self.subgoal_topic, self.reliable_qos)

            if self.waypoint_mode == 'multiple':
                self.multiple_waypoints_pub = self.create_publisher(
                    MultipleWaypoints, self.multiple_waypoints_topic, self.reliable_qos)

        # Behavior control publisher
        if self.enable_behavior_control:
            self.mppi_param_pub = self.create_publisher(
                MPPIParams, '/mppi_update_params', self.reliable_qos)

    def _link_module_publishers(self):
        """모듈에 발행자 연결"""
        # Waypoint publisher
        single_pub = getattr(self, 'subgoal_pub', None)
        multiple_pub = getattr(self, 'multiple_waypoints_pub', None)
        self.waypoint_publisher.set_publishers(single_pub, multiple_pub)

        # Behavior controller
        if self.behavior_controller:
            mppi_pub = getattr(self, 'mppi_param_pub', None)
            self.behavior_controller.set_mppi_publisher(mppi_pub)

        # Safety monitor
        self.safety_monitor.set_pause_publisher(self.pause_command_pub)

    def _log_configuration(self):
        """설정 로깅"""
        self.get_logger().info(f'Waypoint mode: {self.waypoint_mode}')
        self.get_logger().info(f'Behavior control: {"enabled" if self.enable_behavior_control else "disabled"}')
        self.get_logger().info(f'Planned path topic: {self.planned_path_topic}')
        self.get_logger().info(f'Goal status topic: {self.goal_status_topic}')

    # ===== Callback Methods =====

    def current_pose_callback(self, msg: Odometry):
        """현재 위치 콜백"""
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.current_pose = pose_stamped

    def planned_path_callback(self, msg: PlannedPath):
        """경로 계획 콜백"""
        self.path_manager.update_path(msg)
        self.subgoal_published = False

        # Log path info
        path_info = self.path_manager.get_path_info()
        node_types = self.path_manager.get_node_types()

        self.get_logger().info(f'Received path with {path_info["total_nodes"]} nodes')
        self.get_logger().info(f'Path ID: {path_info["path_id"]}')
        self.get_logger().info(f'Node types: {list(set(node_types))}')

        # Update initial behavior
        if self.behavior_controller and self.path_manager.path_nodes:
            first_node = self.path_manager.get_current_target_node()
            if first_node:
                first_node_type = first_node.get('node_type', 1)
                self.behavior_controller.update_behavior(first_node_type)
                self.safety_monitor.update_behavior_type(first_node_type)

    def goal_status_callback(self, msg: ControllerGoalStatus):
        """목표 상태 콜백"""
        current_target = self.path_manager.get_current_target_node()
        if not current_target or msg.goal_id != current_target['id']:
            return

        # Check for pause trigger (before goal reached)
        self._check_pause_trigger(msg)

        if msg.goal_reached and msg.status_code == 1:  # SUCCEEDED
            self._handle_goal_success(msg)
        elif msg.status_code == 2:  # FAILED
            self._handle_goal_failure(msg)
        elif msg.status_code == 3:  # ABORTED
            self._handle_goal_abort(msg)

    def stop_flag_callback(self, msg: Bool):
        """장애물 감지 플래그 콜백"""
        self.safety_monitor.update_stop_flag(msg.data)
        self.get_logger().info(f'Stop flag received: {msg.data}')

    def traffic_light_callback(self, msg: Int32):
        """신호등 상태 콜백"""
        self.safety_monitor.update_traffic_light_state(msg.data)
        self.get_logger().debug(f'Traffic light state received: {msg.data}')

    # ===== Planning Logic =====

    def planning_callback(self):
        """메인 계획 루프 (10Hz)"""
        if not self._is_ready_for_planning():
            return

        if self.emergency_stop_requested:
            self._publish_emergency_stop()
            return

        # Safety monitoring
        self.safety_monitor.check_safety_conditions()

        # Behavior control
        self._update_behavior_if_needed()

        # Waypoint publishing
        if not self.subgoal_published:
            self._publish_waypoints()

    def _is_ready_for_planning(self) -> bool:
        """계획 준비 상태 확인"""
        return (self.path_manager.is_path_following and
                self.current_pose is not None and
                self.path_manager.path_nodes)

    def _update_behavior_if_needed(self):
        """필요시 행동 업데이트"""
        if not self.behavior_controller:
            return

        current_target = self.path_manager.get_current_target_node()
        if current_target:
            current_node_type = current_target.get('node_type', 1)
            if self.behavior_controller.update_behavior(current_node_type):
                self.safety_monitor.update_behavior_type(current_node_type)

    def _publish_waypoints(self):
        """웨이포인트 발행"""
        current_target = self.path_manager.get_current_target_node()
        if not current_target:
            return

        next_nodes = self.path_manager.get_next_nodes()
        path_info = self.path_manager.get_path_info()

        self.waypoint_publisher.publish_waypoints(current_target, next_nodes, path_info)
        self.subgoal_published = True

        self.get_logger().info(f'Published waypoints: current={current_target["id"]}, '
                             f'next_count={len(next_nodes)}')

    # ===== Goal Status Handlers =====

    def _check_pause_trigger(self, msg: ControllerGoalStatus):
        """Pause 트리거 확인"""
        current_target = self.path_manager.get_current_target_node()
        if not current_target or self.pause_signal_sent:
            return

        if msg.distance_to_goal <= self.pause_trigger_distance:
            node_type = current_target.get('node_type', 1)

            if node_type in [7, 8]:  # Pause nodes
                pause_duration = 2.0 if node_type == 7 else 4.0
                self._send_pause_command(pause_duration, current_target['id'], f"Node type {node_type} pause")
                self.pause_signal_sent = True

                self.get_logger().info(f"Pause command sent: {pause_duration}s for node {msg.goal_id}")

    def _handle_goal_success(self, msg: ControllerGoalStatus):
        """목표 성공 처리"""
        self.path_manager.mark_goal_completed(msg.goal_id)
        self.pause_signal_sent = False
        self.subgoal_published = False

        if self.path_manager.advance_to_next_node():
            next_target = self.path_manager.get_current_target_node()
            if next_target:
                self.get_logger().info(f'Advanced to next node: {next_target["id"]} '
                                     f'({self.path_manager.current_target_index + 1}/'
                                     f'{len(self.path_manager.path_nodes)})')
        else:
            self.get_logger().info('Path following completed!')

    def _handle_goal_failure(self, msg: ControllerGoalStatus):
        """목표 실패 처리"""
        self.get_logger().warn(f'Goal {msg.goal_id} failed! Distance: {msg.distance_to_goal:.3f}m')
        self.subgoal_published = False

    def _handle_goal_abort(self, msg: ControllerGoalStatus):
        """목표 중단 처리"""
        self.get_logger().warn(f'Goal {msg.goal_id} aborted! Distance: {msg.distance_to_goal:.3f}m')
        self.emergency_stop_requested = True

    # ===== Utility Methods =====

    def _send_pause_command(self, duration: float, node_id: str, reason: str):
        """Pause command 전송"""
        try:
            pause_msg = PauseCommand()
            pause_msg.header = Header()
            pause_msg.header.stamp = self.get_clock().now().to_msg()
            pause_msg.header.frame_id = 'behavior_planner'
            pause_msg.pause_duration = duration
            pause_msg.node_id = node_id
            pause_msg.reason = reason

            self.pause_command_pub.publish(pause_msg)
            self.get_logger().info(f"Pause command sent: {reason}")

        except Exception as e:
            self.get_logger().error(f"Failed to send pause command: {e}")

    def _publish_emergency_stop(self):
        """긴급 정지 발행"""
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)
        self.get_logger().warn('Emergency stop published!')


def main(args=None):
    rclpy.init(args=args)

    try:
        node = SimpleBehaviorPlannerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()