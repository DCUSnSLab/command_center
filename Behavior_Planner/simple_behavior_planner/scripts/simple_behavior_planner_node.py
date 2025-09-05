#!/usr/bin/env python3
"""
Simple Behavior Planner Node
차량의 전반적인 행동 결정을 담당하는 노드

Subscriptions:
- /planned_path_detailed: 전체 경로의 노드와 링크 정보
- /perception: 인지 결과 (현재는 구독만)
- /goal_status: 목적지 도달 상태 (현재는 구독만)

Publications:
- /subgoal: 현재 위치에서 가장 가까운 다음 노드의 UTM 좌표
- /emergency_stop: 긴급 정지 명령 (현재는 발행만)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import math
import numpy as np
from typing import Optional, List

# ROS2 messages
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String, Header

# TF2 for coordinate transformation
import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs.tf2_geometry_msgs as tf2_geometry_msgs

# Custom messages
from command_center_interfaces.msg import PlannedPath, ControllerGoalStatus, MultipleWaypoints, MPPIParams

# Behavior parameter management
import os
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
        self.subgoal_published = False  # 현재 서브골이 발행되었는지 추적
        self.last_completed_goal_id = None  # 마지막으로 완료된 goal_id 추적
        
        # Behavior control state
        self.current_node_type = 1  # 현재 행동 타입 (기본: 전진)
        self.previous_node_type = 1  # 이전 행동 타입
        self.is_paused = False  # 일시 정지 상태
        self.pause_timer = None  # 일시 정지 타이머
        self.pause_start_time = None
        
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
        """현재 위치 정보 수신 (Odometry에서 PoseStamped로 변환)"""
        # Odometry 메시지에서 PoseStamped로 변환
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.current_pose = pose_stamped
    
    def planned_path_callback(self, msg: PlannedPath):
        """계획된 경로 정보 수신"""
        self.planned_path = msg
        self.path_nodes = []
        
        # Extract node positions from planned path (in map frame - absolute coordinates)
        for node in msg.path_data.nodes:
            # Store nodes in map frame coordinates (absolute UTM) with node_type and heading
            node_pose = {
                'id': node.id,
                'x': node.utm_info.easting,  # Absolute UTM coordinates in map frame
                'y': node.utm_info.northing,
                'z': node.gps_info.alt,
                'node_type': node.node_type,  # Add node_type for behavior control
                'heading': node.heading  # Geographic heading (북쪽 기준 0-360도)
            }
            self.path_nodes.append(node_pose)
        
        # Reset target node index for new path
        self.current_target_node_index = 0
        self.is_path_following = True
        self.subgoal_published = False  # 새로운 경로에 대해 서브골 발행 준비
        self.last_completed_goal_id = None  # 새로운 경로 시작 시 완료된 goal_id 리셋
        
        # Check first node's behavior type and update MPPI parameters if needed
        if self.enable_behavior_control and self.path_nodes:
            first_node_type = self.path_nodes[0].get('node_type', 1)
            if first_node_type != self.current_node_type:
                self._update_behavior_parameters(first_node_type)
        
        self.get_logger().info(f'Received new planned path with {len(self.path_nodes)} nodes')
        self.get_logger().info(f'Path ID: {msg.path_id}, Start: {msg.start_node_id}, Goal: {msg.goal_node_id}')
        
        if self.enable_behavior_control and self.path_nodes:
            node_types = [node.get('node_type', 1) for node in self.path_nodes]
            unique_types = list(set(node_types))
            self.get_logger().info(f'Path contains behavior types: {unique_types}')
    
    def perception_callback(self, msg: String):
        """인지 결과 수신 (현재는 placeholder)"""
        # TODO: 실제 인지 결과에 따른 행동 계획 수정
        self.get_logger().debug(f'Perception data received: {msg.data}')
    
    def goal_status_callback(self, msg: ControllerGoalStatus):
        """목표 도달 상태 수신"""
        # 현재 목표 노드가 있는지 확인
        if not self.path_nodes or self.current_target_node_index >= len(self.path_nodes):
            self.get_logger().debug(f'Received goal_status for {msg.goal_id} but no current target node')
            return
            
        current_target_id = self.path_nodes[self.current_target_node_index]['id']
        
        # 현재 목표와 일치하는지 확인
        if msg.goal_id != current_target_id:
            self.get_logger().debug(f'Received goal_status for {msg.goal_id}, but current target is {current_target_id}. Ignoring.')
            return
            
        # 이미 처리된 goal인지 확인 (중복 방지)
        if self.last_completed_goal_id == msg.goal_id:
            self.get_logger().debug(f'Goal {msg.goal_id} already processed. Ignoring duplicate status.')
            return
        
        if msg.goal_reached and msg.status_code == 1:  # SUCCEEDED
            # self.get_logger().info(f'Goal {msg.goal_id} reached! Distance: {msg.distance_to_goal:.3f}m. Moving to next target node.')
            self.last_completed_goal_id = msg.goal_id  # 완료된 goal_id 기록
            self.advance_to_next_node()
            self.subgoal_published = False  # 다음 서브골 발행 준비
        elif msg.status_code == 2:  # FAILED
            self.get_logger().warn(f'Goal {msg.goal_id} failed! Distance: {msg.distance_to_goal:.3f}m')
            self.subgoal_published = False  # 재시도 준비
        elif msg.status_code == 3:  # ABORTED
            self.get_logger().warn(f'Goal {msg.goal_id} aborted! Distance: {msg.distance_to_goal:.3f}m')
            self.request_emergency_stop()
    
    def planning_callback(self):
        """메인 행동 계획 루프"""
        if not self.is_path_following or not self.current_pose or not self.path_nodes:
            return
        
        if self.emergency_stop_requested:
            self.publish_emergency_stop()
            return
        
        # Check if we need to handle pause behavior
        if self.is_paused:
            return  # Skip normal processing during pause
        
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
    
    def publish_subgoal(self, target_node: dict):
        """서브골 발행 - map 좌표를 odom 좌표로 변환하여 발행"""
        try:
            # Create pose in map frame
            map_pose = PoseStamped()
            map_pose.header = Header()
            map_pose.header.stamp = self.get_clock().now().to_msg()
            map_pose.header.frame_id = 'map'
            
            # Set position in map frame (absolute UTM coordinates)
            map_pose.pose.position.x = target_node['x']
            map_pose.pose.position.y = target_node['y'] 
            map_pose.pose.position.z = target_node['z']
            map_pose.pose.orientation.w = 1.0  # Default orientation
            
            # Transform from map to odom frame  
            transform = self.tf_buffer.lookup_transform('odom', 'map', rclpy.time.Time())
            odom_pose = tf2_geometry_msgs.do_transform_pose_stamped(map_pose, transform)
            
            # Set frame_id and goal_id
            odom_pose.header.frame_id = target_node['id']  # goal_id를 frame_id에 설정
            
            # 노드의 지리학적 heading을 odom 좌표계 heading으로 변환하여 orientation 설정
            if 'heading' in target_node and target_node['heading'] is not None:
                odom_heading_rad = self._convert_geographic_to_odom_heading(target_node['heading'])
                
                # Quaternion 변환 (yaw만 적용)
                odom_pose.pose.orientation.z = math.sin(odom_heading_rad / 2.0)
                odom_pose.pose.orientation.w = math.cos(odom_heading_rad / 2.0)
                odom_pose.pose.orientation.x = 0.0
                odom_pose.pose.orientation.y = 0.0
            else:
                # Fallback: 현재 위치에서 목표점으로의 방향으로 설정 (odom 좌표계에서)
                if self.current_pose:
                    dx = odom_pose.pose.position.x - self.current_pose.pose.position.x
                    dy = odom_pose.pose.position.y - self.current_pose.pose.position.y
                    yaw = math.atan2(dy, dx)
                    
                    # Quaternion 변환
                    odom_pose.pose.orientation.z = math.sin(yaw / 2.0)
                    odom_pose.pose.orientation.w = math.cos(yaw / 2.0)
                    odom_pose.pose.orientation.x = 0.0
                    odom_pose.pose.orientation.y = 0.0
            
            self.subgoal_pub.publish(odom_pose)
            
            # 서브골 발행 시마다 로그 출력
            self.get_logger().info(f'Published NEW subgoal: Node {target_node["id"]} '
                                 f'map({target_node["x"]:.2f}, {target_node["y"]:.2f}) -> '
                                 f'odom({odom_pose.pose.position.x:.2f}, {odom_pose.pose.position.y:.2f})')
        
        except Exception as e:
            self.get_logger().warn(f'Failed to transform subgoal from map to odom: {str(e)}')
            self.get_logger().warn(f'Error details: {type(e).__name__}')
            
            # Check what transforms are available
            all_frames = self.tf_buffer.all_frames_as_string()
            self.get_logger().warn(f'Available frames: {all_frames}')
            
            # Fallback: publish without transformation (assumes map==odom for now)
            subgoal_msg = PoseStamped()
            subgoal_msg.header = Header()
            subgoal_msg.header.stamp = self.get_clock().now().to_msg()
            subgoal_msg.header.frame_id = target_node['id']
            
            subgoal_msg.pose.position.x = target_node['x']
            subgoal_msg.pose.position.y = target_node['y']
            subgoal_msg.pose.position.z = target_node['z']
            
            # Fallback에서도 heading 적용
            if 'heading' in target_node and target_node['heading'] is not None:
                odom_heading_rad = self._convert_geographic_to_odom_heading(target_node['heading'])
                subgoal_msg.pose.orientation.z = math.sin(odom_heading_rad / 2.0)
                subgoal_msg.pose.orientation.w = math.cos(odom_heading_rad / 2.0)
                subgoal_msg.pose.orientation.x = 0.0
                subgoal_msg.pose.orientation.y = 0.0
            else:
                subgoal_msg.pose.orientation.w = 1.0
            
            if hasattr(self, 'subgoal_pub') and self.subgoal_pub is not None:
                self.subgoal_pub.publish(subgoal_msg)
                self.get_logger().warn(f'Published subgoal without TF transformation: Node {target_node["id"]} at absolute coords')
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
            
            # 다음 목표들 설정
            waypoints_msg.next_waypoints = []
            for node in next_nodes:
                waypoints_msg.next_waypoints.append(self.create_pose_stamped(node))
            
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
    
    def create_pose_stamped(self, node: dict) -> PoseStamped:
        """노드에서 PoseStamped 메시지 생성 (odom 좌표로 변환)"""
        try:
            # Create pose in map frame
            map_pose = PoseStamped()
            map_pose.header.stamp = self.get_clock().now().to_msg()
            map_pose.header.frame_id = 'map'
            map_pose.pose.position.x = node['x']
            map_pose.pose.position.y = node['y'] 
            map_pose.pose.position.z = node['z']
            
            # 노드의 지리학적 heading을 odom 좌표계 heading으로 변환하여 orientation 설정
            if 'heading' in node and node['heading'] is not None:
                odom_heading_rad = self._convert_geographic_to_odom_heading(node['heading'])
                
                # Quaternion 변환 (yaw만 적용)
                map_pose.pose.orientation.z = math.sin(odom_heading_rad / 2.0)
                map_pose.pose.orientation.w = math.cos(odom_heading_rad / 2.0)
                map_pose.pose.orientation.x = 0.0
                map_pose.pose.orientation.y = 0.0
            else:
                map_pose.pose.orientation.w = 1.0
            
            # Transform to odom frame
            transform = self.tf_buffer.lookup_transform('odom', 'map', rclpy.time.Time())
            odom_pose = tf2_geometry_msgs.do_transform_pose_stamped(map_pose, transform)
            
            # Set goal ID in frame_id for compatibility
            odom_pose.header.frame_id = node['id']
            
            return odom_pose
            
        except Exception as e:
            self.get_logger().warn(f'TF transform failed, using fallback: {str(e)}')
            # Fallback without transformation
            fallback_pose = PoseStamped()
            fallback_pose.header.stamp = self.get_clock().now().to_msg()
            fallback_pose.header.frame_id = node['id']
            fallback_pose.pose.position.x = node['x']
            fallback_pose.pose.position.y = node['y']
            fallback_pose.pose.position.z = node['z']
            
            # Fallback에서도 heading 적용
            if 'heading' in node and node['heading'] is not None:
                odom_heading_rad = self._convert_geographic_to_odom_heading(node['heading'])
                fallback_pose.pose.orientation.z = math.sin(odom_heading_rad / 2.0)
                fallback_pose.pose.orientation.w = math.cos(odom_heading_rad / 2.0)
                fallback_pose.pose.orientation.x = 0.0
                fallback_pose.pose.orientation.y = 0.0
            else:
                fallback_pose.pose.orientation.w = 1.0
            return fallback_pose
    
    def _update_behavior_parameters(self, node_type: int):
        """Update MPPI parameters based on behavior type"""
        if not self.enable_behavior_control or self.behavior_param_manager is None:
            return
        
        try:
            # DEBUG: Log behavior transition
            self.get_logger().info(f"=== BEHAVIOR TRANSITION DEBUG ===")
            self.get_logger().info(f"Previous behavior: {self.previous_node_type}")
            self.get_logger().info(f"Current behavior: {self.current_node_type}")  
            self.get_logger().info(f"Requested behavior: {node_type}")
            
            # Handle pause behaviors
            if self.behavior_param_manager.is_pause_behavior(node_type):
                pause_duration = self.behavior_param_manager.get_pause_duration(node_type)
                self._handle_pause_behavior(pause_duration)
                return
            
            # Get behavior-specific parameters
            behavior_params = self.behavior_param_manager.get_behavior_params(node_type)
            
            # Validate parameters
            if not self.behavior_param_manager.validate_behavior_params(behavior_params):
                self.get_logger().error(f"Invalid parameters for behavior {node_type}, skipping update")
                return
            
            # Send parameters to MPPI
            self._send_mppi_parameters(behavior_params)
            
            # Update current behavior state
            self.previous_node_type = self.current_node_type
            self.current_node_type = node_type
            
            behavior_desc = self.behavior_param_manager.get_behavior_description(node_type)
            self.get_logger().info(f"Updated to behavior {node_type}: {behavior_desc}")
            
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
            
            # Goal critic parameters - only respect_reverse_heading (from config)
            if 'respect_reverse_heading' in behavior_params:
                msg.update_goal_critic = True
                msg.respect_reverse_heading = behavior_params['respect_reverse_heading']
            
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
            
            self.get_logger().debug(f"Sent MPPI parameters for behavior {behavior_params.get('behavior_type', 1)}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to send MPPI parameters: {e}")
    
    def _handle_pause_behavior(self, pause_duration: float):
        """Handle pause behaviors (node_type 7, 8)"""
        if self.is_paused:
            self.get_logger().debug("Already in pause state, ignoring new pause request")
            return
        
        self.is_paused = True
        self.pause_start_time = self.get_clock().now().nanoseconds / 1e9
        
        # Send zero velocity to MPPI
        self._send_pause_parameters()
        
        # Create timer for resume
        if self.pause_timer is not None:
            self.pause_timer.cancel()
        
        self.pause_timer = self.create_timer(pause_duration, self._resume_from_pause)
        
        self.get_logger().info(f"Started pause behavior for {pause_duration} seconds")
    
    def _send_pause_parameters(self):
        """Send pause parameters to MPPI (force stop flag)"""
        try:
            msg = MPPIParams()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "pause_behavior"
            
            # Set force stop flag
            msg.update_control = True
            msg.force_stop = True
            msg.control_frequency = 20.0
            msg.goal_reached_threshold = 2.0
            
            # Get current node type for pause behavior info
            current_node_type = self.current_node_type
            if current_node_type == 7:
                msg.current_behavior_type = 7
                msg.current_behavior_desc = "Pause for 1 second"
            elif current_node_type == 8:
                msg.current_behavior_type = 8
                msg.current_behavior_desc = "Pause for 4 seconds"
            else:
                # Fallback - should not happen
                msg.current_behavior_type = 7
                msg.current_behavior_desc = "Pause behavior"
            
            self.mppi_param_pub.publish(msg)
            self.get_logger().debug(f"Sent pause (force_stop=True) parameters to MPPI, behavior_type: {msg.current_behavior_type}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to send pause parameters: {e}")
    
    def _resume_from_pause(self):
        """Resume from pause behavior"""
        if not self.is_paused:
            return
        
        self.is_paused = False
        self.pause_start_time = None
        
        # Cancel pause timer
        if self.pause_timer is not None:
            self.pause_timer.cancel()
            self.pause_timer = None
        
        # Restore previous behavior parameters if available
        if self.enable_behavior_control and self.path_nodes and self.current_target_node_index < len(self.path_nodes):
            current_node = self.path_nodes[self.current_target_node_index]
            current_node_type = current_node.get('node_type', 1)
            
            # If current node is still a pause node, move to next
            if self.behavior_param_manager.is_pause_behavior(current_node_type):
                self.advance_to_next_node()
                self.subgoal_published = False
            else:
                # Restore normal behavior parameters
                self._update_behavior_parameters(current_node_type)
        
        self.get_logger().info("Resumed from pause behavior")


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