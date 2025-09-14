#!/usr/bin/env python3
"""
Simple Behavior Planner Node
차량의 전반적인 행동 결정을 담당하는 노드
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import math
import os
from typing import Optional

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String, Header

import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs.tf2_geometry_msgs as tf2_geometry_msgs

from command_center_interfaces.msg import PlannedPath, ControllerGoalStatus, MultipleWaypoints, MPPIParams, PauseCommand

import yaml
from simple_behavior_planner.behavior_parameter_manager import BehaviorParameterManager


class SimpleBehaviorPlannerNode(Node):
    """Simple Behavior Planner Node for autonomous vehicle path following"""
    
    def __init__(self):
        super().__init__('simple_behavior_planner')
        
        # Parameters
        self.declare_parameter('current_position_topic', '/odom')
        self.declare_parameter('planned_path_topic', '/planned_path_detailed')
        self.declare_parameter('perception_topic', '/perception')
        self.declare_parameter('goal_status_topic', '/goal_status')
        self.declare_parameter('subgoal_topic', '/subgoal')
        self.declare_parameter('multiple_waypoints_topic', '/multiple_waypoints')
        self.declare_parameter('emergency_stop_topic', '/emergency_stop')
        self.declare_parameter('lookahead_distance', 10.0)  # meters
        self.declare_parameter('goal_tolerance', 2.0)  # meters
        self.declare_parameter('pause_trigger_distance', 0.8)  # meters - distance to trigger pause command
        self.declare_parameter('waypoint_mode', 'multiple')  # 'single' or 'multiple'
        self.declare_parameter('behavior_config_path', 'behavior_modifiers.yaml')  # behavior config file
        self.declare_parameter('enable_behavior_control', True)  # enable behavior-based parameter control
        
        # Get parameters
        self.current_position_topic = self.get_parameter('current_position_topic').get_parameter_value().string_value
        self.planned_path_topic = self.get_parameter('planned_path_topic').get_parameter_value().string_value
        self.perception_topic = self.get_parameter('perception_topic').get_parameter_value().string_value
        self.goal_status_topic = self.get_parameter('goal_status_topic').get_parameter_value().string_value
        self.subgoal_topic = self.get_parameter('subgoal_topic').get_parameter_value().string_value
        self.multiple_waypoints_topic = self.get_parameter('multiple_waypoints_topic').get_parameter_value().string_value
        self.emergency_stop_topic = self.get_parameter('emergency_stop_topic').get_parameter_value().string_value
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.pause_trigger_distance = self.get_parameter('pause_trigger_distance').get_parameter_value().double_value
        self.waypoint_mode = self.get_parameter('waypoint_mode').get_parameter_value().string_value
        self.behavior_config_path = self.get_parameter('behavior_config_path').get_parameter_value().string_value
        self.enable_behavior_control = self.get_parameter('enable_behavior_control').get_parameter_value().bool_value
        
        if self.waypoint_mode not in ['single', 'multiple']:
            self.get_logger().warn(f"Invalid waypoint_mode '{self.waypoint_mode}', defaulting to 'multiple'")
            self.waypoint_mode = 'multiple'
        
        # State variables
        self.current_pose: Optional[PoseStamped] = None
        self.planned_path: Optional[PlannedPath] = None
        self.current_target_node_index = 0
        self.path_nodes = []
        self.is_path_following = False
        self.emergency_stop_requested = False
        self.subgoal_published = False
        self.last_completed_goal_id = None
        
        # Behavior control state
        self.current_node_type = 1
        self.previous_node_type = 1
        self.is_paused = False
        self.pause_timer = None
        self.pause_start_time = None
        
        # Pause command state tracking
        self.pause_signal_sent = False
        
        # Initialize behavior parameter manager
        self.behavior_param_manager = None
        if self.enable_behavior_control:
            self._init_behavior_parameter_manager()
        
        # TF2 for coordinate transformation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )
        
        # Subscribers
        self.current_pose_sub = self.create_subscription(
            Odometry, 
            self.current_position_topic, 
            self.current_pose_callback, 
            best_effort_qos
        )
        
        self.planned_path_sub = self.create_subscription(
            PlannedPath, 
            self.planned_path_topic, 
            self.planned_path_callback, 
            reliable_qos
        )
        
        self.perception_sub = self.create_subscription(
            String,  # Placeholder - 실제 인지 메시지 타입으로 변경 필요
            self.perception_topic, 
            self.perception_callback, 
            reliable_qos
        )
        
        self.goal_status_sub = self.create_subscription(
            ControllerGoalStatus,  # 실제 목표 상태 메시지 타입 사용
            self.goal_status_topic, 
            self.goal_status_callback, 
            reliable_qos
        )
        
        # Publishers - create based on waypoint mode
        self.emergency_stop_pub = self.create_publisher(
            Bool,  # Placeholder - 실제 긴급정지 메시지 타입으로 변경 필요
            self.emergency_stop_topic, 
            reliable_qos
        )
        
        # MPPI parameter publisher for behavior control
        if self.enable_behavior_control:
            self.mppi_param_pub = self.create_publisher(
                MPPIParams,
                '/mppi_update_params',
                reliable_qos
            )
        
        # Pause command publisher
        self.pause_command_pub = self.create_publisher(
            PauseCommand,
            '/pause_command',
            reliable_qos
        )
        
        if self.waypoint_mode == 'single':
            self.subgoal_pub = self.create_publisher(
                PoseStamped, 
                self.subgoal_topic, 
                reliable_qos
            )
            self.get_logger().info(f"Single waypoint mode: publishing to {self.subgoal_topic}")
        elif self.waypoint_mode == 'multiple':
            self.multiple_waypoints_pub = self.create_publisher(
                MultipleWaypoints,
                self.multiple_waypoints_topic,
                reliable_qos
            )
            self.get_logger().info(f"Multiple waypoints mode: publishing to {self.multiple_waypoints_topic}")
        else:
            # Fallback - create both publishers
            self.subgoal_pub = self.create_publisher(
                PoseStamped, 
                self.subgoal_topic, 
                reliable_qos
            )
            self.multiple_waypoints_pub = self.create_publisher(
                MultipleWaypoints,
                self.multiple_waypoints_topic,
                reliable_qos
            )
            self.get_logger().warn("Unknown waypoint mode, creating both publishers")
        
        # Timer for main behavior planning loop
        self.planning_timer = self.create_timer(0.1, self.planning_callback)  # 10Hz
        
        self.get_logger().info('Simple Behavior Planner Node initialized')
        self.get_logger().info(f'Subscribed to: {self.planned_path_topic}')
        self.get_logger().info(f'Publishing subgoals to: {self.subgoal_topic}')
        if self.enable_behavior_control:
            self.get_logger().info('Behavior-based parameter control enabled')
        else:
            self.get_logger().info('Behavior-based parameter control disabled')
    
    def _init_behavior_parameter_manager(self):
        """Initialize behavior parameter manager"""
        try:
            # Get the package path using ament_index
            from ament_index_python.packages import get_package_share_directory
            try:
                package_share_path = get_package_share_directory('simple_behavior_planner')
                config_full_path = os.path.join(package_share_path, 'config', self.behavior_config_path)
            except Exception as e:
                # Fallback to relative path
                self.get_logger().warn(f"Could not find package share directory, using fallback path: {e}")
                package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_full_path = os.path.join(package_path, 'config', self.behavior_config_path)
            
            if not os.path.exists(config_full_path):
                self.get_logger().error(f"Behavior config file not found: {config_full_path}")
                self.enable_behavior_control = False
                return
            
            # Load behavior modifiers config
            with open(config_full_path, 'r') as file:
                config = yaml.safe_load(file)
            
            behavior_config = config.get('/**', {}).get('ros__parameters', {})
            # Use simple_behavior_planner config directory for smppi_params.yaml
            smppi_config_path = behavior_config.get('smppi_config_path', 'smppi_params.yaml')
            
            # Initialize parameter manager
            self.behavior_param_manager = BehaviorParameterManager(smppi_config_path, behavior_config)
            
            self.get_logger().info(f"Behavior parameter manager initialized with config: {config_full_path}")
            
            # Log available behaviors
            behaviors = self.behavior_param_manager.get_available_behaviors()
            self.get_logger().info(f"Available behaviors: {len(behaviors)}")
            for behavior_type, description in behaviors.items():
                self.get_logger().info(f"  {behavior_type}: {description}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to initialize behavior parameter manager: {e}")
            self.enable_behavior_control = False
            self.behavior_param_manager = None
    
    def current_pose_callback(self, msg: Odometry):
        """Process current position from odometry"""
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.current_pose = pose_stamped
    
    def planned_path_callback(self, msg: PlannedPath):
        """Process received planned path"""
        self.planned_path = msg
        self.path_nodes = self._extract_path_nodes(msg)
        self._reset_path_state()
        self._update_initial_behavior()
        self._log_path_info(msg)
    
    def _extract_path_nodes(self, msg: PlannedPath) -> list:
        """Extract node information from planned path message"""
        nodes = []
        for node in msg.path_data.nodes:
            node_pose = {
                'id': node.id,
                'x': node.utm_info.easting,
                'y': node.utm_info.northing,
                'z': node.gps_info.alt,
                'node_type': node.node_type,
                'heading': node.heading
            }
            nodes.append(node_pose)
        return nodes
    
    def _reset_path_state(self):
        """Reset state variables for new path"""
        self.current_target_node_index = 0
        self.is_path_following = True
        self.subgoal_published = False
        self.last_completed_goal_id = None
    
    def _update_initial_behavior(self):
        """Update behavior parameters for first node if needed"""
        if self.enable_behavior_control and self.path_nodes:
            first_node_type = self.path_nodes[0].get('node_type', 1)
            if first_node_type != self.current_node_type:
                self._update_behavior_parameters(first_node_type)
    
    def _log_path_info(self, msg: PlannedPath):
        """Log information about received path"""
        self.get_logger().info(f'Received path with {len(self.path_nodes)} nodes')
        self.get_logger().info(f'Path ID: {msg.path_id}, Start: {msg.start_node_id}, Goal: {msg.goal_node_id}')
        
        if self.enable_behavior_control and self.path_nodes:
            node_types = [node.get('node_type', 1) for node in self.path_nodes]
            unique_types = list(set(node_types))
            self.get_logger().info(f'Path contains behavior types: {unique_types}')
    
    def perception_callback(self, msg: String):
        """Handle perception data (placeholder)"""
        self.get_logger().debug(f'Perception data received: {msg.data}')
    
    def goal_status_callback(self, msg: ControllerGoalStatus):
        """Handle goal status updates from controller"""
        if not self._is_valid_goal_status(msg):
            return
        
        # Check for pause command trigger (before goal reached)
        self._check_pause_trigger(msg)
            
        if msg.goal_reached and msg.status_code == 1:  # SUCCEEDED
            self._handle_goal_success(msg)
        elif msg.status_code == 2:  # FAILED
            self._handle_goal_failure(msg)
        elif msg.status_code == 3:  # ABORTED
            self._handle_goal_abort(msg)
    
    def _is_valid_goal_status(self, msg: ControllerGoalStatus) -> bool:
        """Check if goal status message is valid for current state"""
        if not self.path_nodes or self.current_target_node_index >= len(self.path_nodes):
            return False
            
        current_target_id = self.path_nodes[self.current_target_node_index]['id']
        if msg.goal_id != current_target_id or self.last_completed_goal_id == msg.goal_id:
            return False
            
        return True
    
    def _check_pause_trigger(self, msg: ControllerGoalStatus):
        #self.get_logger().info(f"@@@@@@@@@@@@@{msg.distance_to_goal} < {self.pause_trigger_distance}@@@@@@@@@{self.path_nodes[self.current_target_node_index]}@@@@")
        """Check if we should send pause command based on distance and node type"""
        if (self.current_target_node_index < len(self.path_nodes) and
            not self.pause_signal_sent and
            msg.distance_to_goal <= self.pause_trigger_distance):
            
            current_node = self.path_nodes[self.current_target_node_index]
            node_type = current_node.get('node_type', 1)

    
            # Check if current node is a pause node (type 7 or 8)
            if node_type in [7, 8]:
                pause_duration = 2.0 if node_type == 7 else 4.0
                
                # Send pause command
                pause_msg = PauseCommand()
                pause_msg.header.stamp = self.get_clock().now().to_msg()
                pause_msg.header.frame_id = 'behavior_planner'
                pause_msg.pause_duration = pause_duration
                pause_msg.node_id = msg.goal_id
                pause_msg.reason = f"Node type {node_type} pause ({pause_duration}s)"
                
                self.pause_command_pub.publish(pause_msg)
                self.pause_signal_sent = True
                
                # Also update behavior parameters at pause trigger point
                if self.enable_behavior_control and node_type != self.current_node_type:
                    self._update_behavior_parameters(node_type)
                
                self.get_logger().info(f"Pause command sent: {pause_duration}s for node {msg.goal_id} (distance: {msg.distance_to_goal:.2f}m)")
                self.get_logger().info(f"Behavior parameters updated to type {node_type} at pause trigger")
    
    def _handle_goal_success(self, msg: ControllerGoalStatus):
        """Handle successful goal completion"""
        self.last_completed_goal_id = msg.goal_id
        self.pause_signal_sent = False  # Reset for next goal
        self.advance_to_next_node()
        self.subgoal_published = False
    
    def _handle_goal_failure(self, msg: ControllerGoalStatus):
        """Handle goal failure"""
        self.get_logger().warn(f'Goal {msg.goal_id} failed! Distance: {msg.distance_to_goal:.3f}m')
        self.subgoal_published = False
    
    def _handle_goal_abort(self, msg: ControllerGoalStatus):
        """Handle goal abort"""
        self.get_logger().warn(f'Goal {msg.goal_id} aborted! Distance: {msg.distance_to_goal:.3f}m')
        self.request_emergency_stop()
    
    def planning_callback(self):
        """메인 행동 계획 루프"""
        if not self.is_path_following or not self.current_pose or not self.path_nodes:
            return
        
        if self.emergency_stop_requested:
            self.publish_emergency_stop()
            return
        
        # Removed pause behavior handling - now handled by SMPPI controller
        
        # Check current node behavior and update parameters if needed
        if self.enable_behavior_control and self.path_nodes and self.current_target_node_index < len(self.path_nodes):
            current_node = self.path_nodes[self.current_target_node_index]
            current_node_type = current_node.get('node_type', 1)
            
            # Update behavior parameters if node type changed
            if current_node_type != self.current_node_type:
                self._update_behavior_parameters(current_node_type)
        
        # Find and publish next subgoal (only if not already published)
        if not self.subgoal_published:
            next_subgoal = self.find_next_subgoal()
            if next_subgoal:
                # Publish based on waypoint mode
                if self.waypoint_mode == 'single':
                    self.publish_subgoal(next_subgoal)
                elif self.waypoint_mode == 'multiple':
                    self.publish_multiple_waypoints()
                else:
                    # Fallback - publish both (multiple waypoints takes priority)
                    self.publish_multiple_waypoints()
                    self.publish_subgoal(next_subgoal)
                
                self.subgoal_published = True  # 서브골 발행 완료 표시
    
    def find_next_subgoal(self) -> Optional[dict]:
        """순서대로 정렬된 경로에서 다음 서브골 찾기"""
        if not self.path_nodes:
            return None
        
        # planned_path_detailed의 노드들은 이미 순서대로 정렬되어 있으므로
        # 단순히 current_target_node_index를 사용하여 다음 노드 선택
        if self.current_target_node_index < len(self.path_nodes):
            target_node = self.path_nodes[self.current_target_node_index]
            
            self.get_logger().debug(f'Target node {self.current_target_node_index}: {target_node["id"]} '
                                  f'at ({target_node["x"]:.2f}, {target_node["y"]:.2f}) '
                                  f'type={target_node.get("node_type", 1)}')
            
            return target_node
        else:
            # 모든 노드를 완주했으면 경로 추종 종료
            self.is_path_following = False
            self.get_logger().info('All path nodes completed!')
            return None
    
    def _create_map_pose(self, target_node: dict) -> PoseStamped:
        """Create pose in map frame from target node"""
        map_pose = PoseStamped()
        map_pose.header = Header()
        map_pose.header.stamp = self.get_clock().now().to_msg()
        map_pose.header.frame_id = 'map'
        
        map_pose.pose.position.x = target_node['x']
        map_pose.pose.position.y = target_node['y'] 
        map_pose.pose.position.z = target_node['z']
        map_pose.pose.orientation.w = 1.0
        
        return map_pose
    
    def _set_pose_orientation(self, pose: PoseStamped, target_node: dict):
        """Set pose orientation based on node heading or current position"""
        if 'heading' in target_node and target_node['heading'] is not None:
            odom_heading_rad = self._convert_geographic_to_odom_heading(target_node['heading'])
            pose.pose.orientation.z = math.sin(odom_heading_rad / 2.0)
            pose.pose.orientation.w = math.cos(odom_heading_rad / 2.0)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
        elif self.current_pose:
            dx = pose.pose.position.x - self.current_pose.pose.position.x
            dy = pose.pose.position.y - self.current_pose.pose.position.y
            yaw = math.atan2(dy, dx)
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0

    def publish_subgoal(self, target_node: dict):
        """Publish subgoal with coordinate transformation"""
        try:
            map_pose = self._create_map_pose(target_node)
            transform = self.tf_buffer.lookup_transform('odom', 'map', rclpy.time.Time())
            odom_pose = tf2_geometry_msgs.do_transform_pose_stamped(map_pose, transform)
            
            odom_pose.header.frame_id = target_node['id']
            self._set_pose_orientation(odom_pose, target_node)
            
            self.subgoal_pub.publish(odom_pose)
            self.get_logger().info(f'Published subgoal: Node {target_node["id"]} at odom({odom_pose.pose.position.x:.2f}, {odom_pose.pose.position.y:.2f})')
        
        except Exception as e:
            self._publish_fallback_subgoal(target_node, str(e))
    
    def _publish_fallback_subgoal(self, target_node: dict, error_msg: str):
        """Publish subgoal without TF transformation as fallback"""
        self.get_logger().warn(f'TF transformation failed: {error_msg}')
        
        subgoal_msg = PoseStamped()
        subgoal_msg.header = Header()
        subgoal_msg.header.stamp = self.get_clock().now().to_msg()
        subgoal_msg.header.frame_id = target_node['id']
        
        subgoal_msg.pose.position.x = target_node['x']
        subgoal_msg.pose.position.y = target_node['y']
        subgoal_msg.pose.position.z = target_node['z']
        
        self._set_pose_orientation(subgoal_msg, target_node)
        
        if hasattr(self, 'subgoal_pub') and self.subgoal_pub is not None:
            self.subgoal_pub.publish(subgoal_msg)
            self.get_logger().warn(f'Published fallback subgoal: Node {target_node["id"]}')
        else:
            self.get_logger().warn(f'Subgoal publisher not available in {self.waypoint_mode} mode')
    
    def advance_to_next_node(self):
        """다음 노드로 이동"""
        if self.current_target_node_index < len(self.path_nodes) - 1:
            self.current_target_node_index += 1
            next_node = self.path_nodes[self.current_target_node_index]
            self.get_logger().info(f'Advanced to next node: {next_node["id"]} '
                                 f'({self.current_target_node_index + 1}/{len(self.path_nodes)}), node_type: {self.current_node_type}')
        else:
            self.get_logger().info('Reached final destination!')
            self.is_path_following = False
    
    def publish_emergency_stop(self):
        """긴급 정지 명령 발행"""
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)
        self.get_logger().warn('Emergency stop published!')
    
    def request_emergency_stop(self):
        """긴급 정지 요청"""
        self.emergency_stop_requested = True
        self.get_logger().warn('Emergency stop requested!')
    
    def clear_emergency_stop(self):
        """긴급 정지 해제"""
        self.emergency_stop_requested = False
        self.get_logger().info('Emergency stop cleared')
    
    def _convert_geographic_to_odom_heading(self, geographic_heading_deg: float) -> float:
        """
        지리학적 heading(북쪽 기준, 시계방향)을 odom heading(동쪽 기준, 반시계방향)으로 변환
        
        Args:
            geographic_heading_deg: 북쪽 기준 0-360도 (시계방향)
            
        Returns:
            odom heading in radians (동쪽 기준, 반시계방향)
        """
        # 북쪽 기준 → 동쪽 기준 변환
        # 지리학적: 북쪽=0°, 시계방향
        # 수학적(odom): 동쪽=0°, 반시계방향
        math_angle_deg = (geographic_heading_deg) % 360
        return math.radians(math_angle_deg)
    
    def publish_multiple_waypoints(self):
        """Multiple waypoints 발행 - 현재 목표 + 다음 목표들"""
        try:
            if not self.path_nodes or self.current_target_node_index >= len(self.path_nodes):
                return
            
            # 현재 목표
            current_node = self.path_nodes[self.current_target_node_index]
            
            # 다음 목표들 (최대 3개까지)
            lookahead_count = 3
            next_nodes = []
            for i in range(1, min(lookahead_count + 1, len(self.path_nodes) - self.current_target_node_index)):
                next_idx = self.current_target_node_index + i
                if next_idx < len(self.path_nodes):
                    next_nodes.append(self.path_nodes[next_idx])
            
            # MultipleWaypoints 메시지 생성
            waypoints_msg = MultipleWaypoints()
            waypoints_msg.header.stamp = self.get_clock().now().to_msg()
            waypoints_msg.header.frame_id = 'odom'
            
            # 현재 목표 설정
            waypoints_msg.current_goal = self.create_pose_stamped(current_node)
            
            # 현재 목표의 reverse heading 정보
            current_behavior_type = current_node.get('node_type', 1)
            waypoints_msg.current_goal_reverse_heading = self._is_reverse_behavior(current_behavior_type)
            waypoints_msg.current_goal_node_type = current_behavior_type
            
            # 다음 목표들 설정
            waypoints_msg.next_waypoints = []
            waypoints_msg.next_waypoints_reverse_heading = []
            waypoints_msg.next_waypoints_node_types = []
            for node in next_nodes:
                waypoints_msg.next_waypoints.append(self.create_pose_stamped(node))
                next_behavior_type = node.get('node_type', 1)
                waypoints_msg.next_waypoints_reverse_heading.append(self._is_reverse_behavior(next_behavior_type))
                waypoints_msg.next_waypoints_node_types.append(next_behavior_type)
            
            # 경로 정보 설정
            waypoints_msg.path_id = self.planned_path.path_id if self.planned_path else ""
            waypoints_msg.current_waypoint_index = self.current_target_node_index
            waypoints_msg.total_waypoints = len(self.path_nodes)
            waypoints_msg.is_final_waypoint = (self.current_target_node_index == len(self.path_nodes) - 1)
            
            # 발행
            if hasattr(self, 'multiple_waypoints_pub') and self.multiple_waypoints_pub is not None:
                self.multiple_waypoints_pub.publish(waypoints_msg)
                self.get_logger().info(f'Published MultipleWaypoints: current={current_node["id"]}, '
                                     f'next_count={len(next_nodes)}, final={waypoints_msg.is_final_waypoint}')
            else:
                self.get_logger().warn(f'Multiple waypoints publisher not available in {self.waypoint_mode} mode')
                                 
        except Exception as e:
            self.get_logger().warn(f'Failed to publish multiple waypoints: {str(e)}')
    
    def _is_reverse_behavior(self, behavior_type: int) -> bool:
        """Check if behavior type requires reverse heading"""
        if self.behavior_param_manager is None:
            return False
        
        try:
            behavior_params = self.behavior_param_manager.get_behavior_params(behavior_type)
            return behavior_params.get('respect_reverse_heading', False)
        except Exception as e:
            self.get_logger().warn(f"Could not check reverse behavior for type {behavior_type}: {e}")
            return False
    
    def create_pose_stamped(self, node: dict) -> PoseStamped:
        """Create PoseStamped message from node with coordinate transformation"""
        try:
            map_pose = self._create_map_pose(node)
            self._set_pose_orientation(map_pose, node)
            
            transform = self.tf_buffer.lookup_transform('odom', 'map', rclpy.time.Time())
            odom_pose = tf2_geometry_msgs.do_transform_pose_stamped(map_pose, transform)
            odom_pose.header.frame_id = node['id']
            
            return odom_pose
            
        except Exception as e:
            self.get_logger().warn(f'TF transform failed, using fallback: {str(e)}')
            fallback_pose = PoseStamped()
            fallback_pose.header.stamp = self.get_clock().now().to_msg()
            fallback_pose.header.frame_id = node['id']
            fallback_pose.pose.position.x = node['x']
            fallback_pose.pose.position.y = node['y']
            fallback_pose.pose.position.z = node['z']
            
            self._set_pose_orientation(fallback_pose, node)
            return fallback_pose
    
    def _update_behavior_parameters(self, node_type: int):
        """Update MPPI parameters based on behavior type"""
        if not self.enable_behavior_control or self.behavior_param_manager is None:
            return
        
        try:
            
            # Pause behaviors (7, 8) are now handled by SMPPI controller via pause commands
            # They use normal forward movement parameters
            
            # Get behavior-specific parameters
            behavior_params = self.behavior_param_manager.get_behavior_params(node_type)
            
            # Validate parameters
            if not self.behavior_param_manager.validate_behavior_params(behavior_params):
                self.get_logger().error(f"Invalid parameters for behavior {node_type}, skipping update")
                return
            
            # === DIAGNOSTIC LOGGING: Track parameter changes ===
            prev_type = self.current_node_type
            behavior_desc = self.behavior_param_manager.get_behavior_description(node_type)
            
            # Log key parameter changes
            key_params = {
                'max_linear_velocity': behavior_params.get('max_linear_velocity', 'N/A'),
                'min_linear_velocity': behavior_params.get('min_linear_velocity', 'N/A'),
                'respect_reverse_heading': behavior_params.get('respect_reverse_heading', False),
                'lookahead_base_distance': behavior_params.get('lookahead_base_distance', 'N/A'),
                'goal_weight': behavior_params.get('goal_weight', 'N/A')
            }
            
            self.get_logger().info(f"🔄 [PARAM UPDATE] {prev_type}->{node_type}: {behavior_desc}")
            for param, value in key_params.items():
                if value != 'N/A':
                    self.get_logger().info(f"   {param}: {value}")
            
            # Send parameters to MPPI
            self._send_mppi_parameters(behavior_params)
            
            # Update current behavior state
            self.previous_node_type = self.current_node_type
            self.current_node_type = node_type
            
        except Exception as e:
            self.get_logger().error(f"Failed to update behavior parameters: {e}")
    
    def _send_mppi_parameters(self, behavior_params: dict):
        """Send behavior parameters to MPPI controller"""
        try:
            msg = MPPIParams()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
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
            lookahead_updated = False
            if 'lookahead_base_distance' in behavior_params:
                msg.update_lookahead = True
                msg.lookahead_base_distance = behavior_params['lookahead_base_distance']
                lookahead_updated = True
            if 'lookahead_velocity_factor' in behavior_params:
                msg.update_lookahead = True
                msg.lookahead_velocity_factor = behavior_params['lookahead_velocity_factor']
                lookahead_updated = True
            if 'lookahead_min_distance' in behavior_params:
                msg.update_lookahead = True
                msg.lookahead_min_distance = behavior_params['lookahead_min_distance']
                lookahead_updated = True
            if 'lookahead_max_distance' in behavior_params:
                msg.update_lookahead = True
                msg.lookahead_max_distance = behavior_params['lookahead_max_distance']
                lookahead_updated = True
            
            # Goal critic parameters - always reset to ensure proper mode switching
            msg.update_goal_critic = True
            # Default to False, only True for reverse behaviors
            msg.respect_reverse_heading = behavior_params.get('respect_reverse_heading', False)
            
            # Control parameters - always set to ensure force_stop is properly managed
            msg.update_control = True
            msg.goal_reached_threshold = behavior_params.get('goal_reached_threshold', 2.0)
            msg.control_frequency = behavior_params.get('control_frequency', 20.0)
            msg.force_stop = False  # Resume normal operation (override any previous stop)
            
            # Current behavior mode information
            msg.current_behavior_type = behavior_params.get('behavior_type', 1)
            msg.current_behavior_desc = behavior_params.get('behavior_description', 'Normal forward movement')
            
            # Publish the parameter update
            self.mppi_param_pub.publish(msg)
            
            
        except Exception as e:
            self.get_logger().error(f"Failed to send MPPI parameters: {e}")
    
    # Removed pause behavior handling functions - now handled by SMPPI controller


def main(args=None):
    rclpy.init(args=args)
    
    node = SimpleBehaviorPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()