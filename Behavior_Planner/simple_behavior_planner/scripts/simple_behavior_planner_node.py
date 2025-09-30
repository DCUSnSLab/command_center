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
    MPPIParams, PauseCommand, RequestReplan
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

        # Dynamic replanning state
        self.current_route_type = "A"  # Default to route A
        self.path_availability = True  # Assume path is available initially
        self.last_goal_node_id = ""   # Track last goal for replanning

        # One-time trigger flags
        self.path_query_sent = False  # Flag to prevent multiple queries for same node
        self.current_trigger_node_id = ""  # Track which node triggered the query
        self.replan_request_sent = False  # Flag to prevent multiple replan requests

        # Setup QoS profiles
        self._setup_qos_profiles()

        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()

        # Link modules with publishers
        self._link_module_publishers()

        # Main planning timer
        self.create_timer(0.05, self.planning_callback)  # 20Hz

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
        self.declare_parameter('traffic_light_topic', '/tl/state')

        # Dynamic replanning parameters
        self.declare_parameter('path_availability_topic', '/path_availability')
        self.declare_parameter('request_replan_topic', '/request_replan')
        self.declare_parameter('node_type_triggers', [12, 13])  # node types that trigger path check

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

        # Dynamic replanning parameters
        self.path_availability_topic = self.get_parameter('path_availability_topic').value
        self.request_replan_topic = self.get_parameter('request_replan_topic').value
        self.node_type_triggers = self.get_parameter('node_type_triggers').value

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

        # Dynamic replanning subscribers
        self.path_availability_sub = self.create_subscription(
            Bool, self.path_availability_topic,
            self.path_availability_callback, self.reliable_qos)

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

        # Dynamic replanning publisher
        self.request_replan_pub = self.create_publisher(
            RequestReplan, self.request_replan_topic, self.reliable_qos)

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
        self.get_logger().info(f'Path availability topic: {self.path_availability_topic}')
        self.get_logger().info(f'Request replan topic: {self.request_replan_topic}')
        self.get_logger().info(f'Node type triggers: {self.node_type_triggers}')
        self.get_logger().info(f'Initial route type: {self.current_route_type}')

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

        # Update last goal node ID for replanning
        self._update_last_goal_node_id(msg)

        # Update route type based on received path (if it was a replan response)
        if self.replan_request_sent:
            # Determine route type from path_id or start_node_id
            new_route_type = self._determine_route_type_from_path(msg)
            if new_route_type and new_route_type != self.current_route_type:
                old_route = self.current_route_type
                self.current_route_type = new_route_type
                self.get_logger().info(f'Route type updated: {old_route} -> {self.current_route_type}')

        # Log path info
        path_info = self.path_manager.get_path_info()
        node_types = self.path_manager.get_node_types()

        self.get_logger().info(f'Received path with {path_info["total_nodes"]} nodes')
        self.get_logger().info(f'Path ID: {path_info["path_id"]}')
        self.get_logger().info(f'Current route type: {self.current_route_type}')
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

    def path_availability_callback(self, msg: Bool):
        """경로 가용성 콜백 - 인지 모듈로부터 받는 응답"""
        # Only process if we're currently at a trigger node and haven't sent replan request yet
        if not self.path_query_sent or self.replan_request_sent:
            return

        self.path_availability = msg.data
        self.get_logger().info(f'Path availability received: {self.path_availability} for node {self.current_trigger_node_id}')

        # Only trigger replanning if path is NOT available (false)
        if not self.path_availability:
            self.get_logger().info(f'Current route {self.current_route_type} is NOT available, triggering replanning...')
            # Get current trigger node type
            current_target = self.path_manager.get_current_target_node()
            trigger_node_type = current_target.get('node_type') if current_target else None
            self._trigger_replanning(trigger_node_type)
        else:
            self.get_logger().info(f'Current route {self.current_route_type} is available, continuing with existing path')
            # Mark as processed but don't trigger replanning
            self.replan_request_sent = True  # Prevent further processing for this trigger node

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

        # Check for node type trigger (dynamic replanning)
        self._check_node_type_trigger()

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

    # ===== Dynamic Replanning Logic =====

    def _check_node_type_trigger(self):
        """Node type 트리거 확인 - 경로 변경 지점에서 호출"""
        current_target = self.path_manager.get_current_target_node()
        if not current_target:
            return

        current_node_type = current_target.get('node_type')
        current_node_id = current_target['id']

        # Check if current node type matches any trigger (12 or 13)
        if current_node_type in self.node_type_triggers:
            # Check if we already processed this specific node
            if self.current_trigger_node_id == current_node_id and self.path_query_sent:
                # Already processed this node, skip
                return

            # New trigger node detected
            self.current_trigger_node_id = current_node_id
            self.path_query_sent = True
            self.replan_request_sent = False  # Reset replan flag for new trigger

            self.get_logger().info(f'Node type trigger detected: {current_node_type} at node {current_node_id}')
            self.get_logger().info(f'Querying perception system for route availability...')
            # Here we would query perception system, but for now we assume it responds via path_availability_callback
        else:
            # Not a trigger node, reset flags when moving to different node type
            if self.current_trigger_node_id != current_node_id:
                self.path_query_sent = False
                self.current_trigger_node_id = ""
                self.replan_request_sent = False

    def _trigger_replanning(self, trigger_node_type=None):
        """재계획 트리거 - 경로가 사용 불가능할 때 호출"""
        # Prevent multiple replan requests for same trigger
        if self.replan_request_sent:
            self.get_logger().debug('Replan request already sent for this trigger node')
            return

        if not self.path_manager.path_nodes:
            self.get_logger().warn('No path available for replanning')
            return

        # Determine route type based on trigger node type
        if trigger_node_type == 12:
            new_route_type = "T"
        elif trigger_node_type == 13:
            new_route_type = "P"
        else:
            # Fallback for unknown node types
            new_route_type = "B"
            self.get_logger().warn(f'Unknown trigger node type: {trigger_node_type}, using fallback route type B')

        # Get current position in path for start_node_id
        current_target = self.path_manager.get_current_target_node()
        start_node_id = current_target['id'] if current_target else ""

        # Use the last goal from current path
        goal_node_id = self.last_goal_node_id or self._get_final_node_id()

        self._send_replan_request(start_node_id, goal_node_id, new_route_type)

    def _send_replan_request(self, start_node_id: str, goal_node_id: str, route_type: str):
        """재계획 요청 메시지 전송"""
        try:
            replan_msg = RequestReplan()
            replan_msg.header = Header()
            replan_msg.header.stamp = self.get_clock().now().to_msg()
            replan_msg.header.frame_id = 'behavior_planner'

            replan_msg.start_node_id = start_node_id
            replan_msg.goal_node_id = goal_node_id
            replan_msg.route_type = route_type

            self.request_replan_pub.publish(replan_msg)

            # Mark replan request as sent
            self.replan_request_sent = True

            # NOTE: Don't update current_route_type here - wait until new path is received
            # self.current_route_type will be updated when planned_path_callback receives new path

            self.get_logger().info(f'Replan request sent: route_type={route_type}, '
                                 f'start: {start_node_id}, goal: {goal_node_id}')
            self.get_logger().info(f'Waiting for new planned path from Global Planner...')

        except Exception as e:
            self.get_logger().error(f'Failed to send replan request: {e}')

    def _get_final_node_id(self) -> str:
        """현재 경로의 마지막 노드 ID 반환"""
        if self.path_manager.path_nodes:
            return self.path_manager.path_nodes[-2]['id']
        return ""

    def _update_last_goal_node_id(self, path_msg: PlannedPath):
        """마지막 목표 노드 ID 업데이트"""
        self.last_goal_node_id = path_msg.goal_node_id

    def _determine_route_type_from_path(self, path_msg: PlannedPath) -> str:
        """수신된 경로로부터 route type 결정"""
        # Path ID나 start_node_id를 기반으로 route type 판단
        path_id = path_msg.path_id.lower()
        start_node_id = path_msg.start_node_id

        # Path ID에서 route type 추출 시도
        if 'route_a' in path_id or '_a_' in path_id or path_id.endswith('_a'):
            return "A"
        elif 'route_b' in path_id or '_b_' in path_id or path_id.endswith('_b'):
            return "B"

        # 현재와 다른 타입으로 추정 (toggle)
        return "B" if self.current_route_type == "A" else "A"


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