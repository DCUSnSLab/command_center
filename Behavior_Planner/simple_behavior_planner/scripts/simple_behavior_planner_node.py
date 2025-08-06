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

# Custom messages
from command_center_interfaces.msg import PlannedPath, ControllerGoalStatus


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
        self.declare_parameter('emergency_stop_topic', '/emergency_stop')
        self.declare_parameter('lookahead_distance', 10.0)  # meters
        self.declare_parameter('goal_tolerance', 2.0)  # meters
        
        # Get parameters
        self.current_position_topic = self.get_parameter('current_position_topic').get_parameter_value().string_value
        self.planned_path_topic = self.get_parameter('planned_path_topic').get_parameter_value().string_value
        self.perception_topic = self.get_parameter('perception_topic').get_parameter_value().string_value
        self.goal_status_topic = self.get_parameter('goal_status_topic').get_parameter_value().string_value
        self.subgoal_topic = self.get_parameter('subgoal_topic').get_parameter_value().string_value
        self.emergency_stop_topic = self.get_parameter('emergency_stop_topic').get_parameter_value().string_value
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        
        # State variables
        self.current_pose: Optional[PoseStamped] = None
        self.planned_path: Optional[PlannedPath] = None
        self.current_target_node_index = 0
        self.path_nodes = []
        self.is_path_following = False
        self.emergency_stop_requested = False
        self.subgoal_published = False  # 현재 서브골이 발행되었는지 추적
        
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
        
        # Publishers
        self.subgoal_pub = self.create_publisher(
            PoseStamped, 
            self.subgoal_topic, 
            reliable_qos
        )
        
        self.emergency_stop_pub = self.create_publisher(
            Bool,  # Placeholder - 실제 긴급정지 메시지 타입으로 변경 필요
            self.emergency_stop_topic, 
            reliable_qos
        )
        
        # Timer for main behavior planning loop
        self.planning_timer = self.create_timer(0.1, self.planning_callback)  # 10Hz
        
        self.get_logger().info('Simple Behavior Planner Node initialized')
        self.get_logger().info(f'Subscribed to: {self.planned_path_topic}')
        self.get_logger().info(f'Publishing subgoals to: {self.subgoal_topic}')
    
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
        
        # Extract node positions from planned path
        for node in msg.path_data.nodes:
            # UTM 좌표를 사용
            node_pose = {
                'id': node.id,
                'x': node.utm_info.easting,
                'y': node.utm_info.northing,
                'z': node.gps_info.alt
            }
            self.path_nodes.append(node_pose)
        
        # Reset target node index for new path
        self.current_target_node_index = 0
        self.is_path_following = True
        self.subgoal_published = False  # 새로운 경로에 대해 서브골 발행 준비
        
        self.get_logger().info(f'Received new planned path with {len(self.path_nodes)} nodes')
        self.get_logger().info(f'Path ID: {msg.path_id}, Start: {msg.start_node_id}, Goal: {msg.goal_node_id}')
    
    def perception_callback(self, msg: String):
        """인지 결과 수신 (현재는 placeholder)"""
        # TODO: 실제 인지 결과에 따른 행동 계획 수정
        self.get_logger().debug(f'Perception data received: {msg.data}')
    
    def goal_status_callback(self, msg: ControllerGoalStatus):
        """목표 도달 상태 수신"""
        if msg.goal_reached and msg.status_code == 1:  # SUCCEEDED
            self.get_logger().info(f'Goal {msg.goal_id} reached! Distance: {msg.distance_to_goal:.3f}m. Moving to next target node.')
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
        
        # Find and publish next subgoal (only if not already published)
        if not self.subgoal_published:
            next_subgoal = self.find_next_subgoal()
            if next_subgoal:
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
                                  f'at ({target_node["x"]:.2f}, {target_node["y"]:.2f})')
            
            return target_node
        else:
            # 모든 노드를 완주했으면 경로 추종 종료
            self.is_path_following = False
            self.get_logger().info('All path nodes completed!')
            return None
    
    def publish_subgoal(self, target_node: dict):
        """서브골 발행"""
        subgoal_msg = PoseStamped()
        subgoal_msg.header = Header()
        subgoal_msg.header.stamp = self.get_clock().now().to_msg()
        subgoal_msg.header.frame_id = target_node['id']  # goal_id를 frame_id에 설정
        
        # UTM 좌표를 그대로 사용
        subgoal_msg.pose.position.x = target_node['x']
        subgoal_msg.pose.position.y = target_node['y']
        subgoal_msg.pose.position.z = target_node['z']
        
        # 방향은 현재 위치에서 목표점으로의 방향으로 설정
        if self.current_pose:
            dx = target_node['x'] - self.current_pose.pose.position.x
            dy = target_node['y'] - self.current_pose.pose.position.y
            yaw = math.atan2(dy, dx)
            
            # Quaternion 변환
            subgoal_msg.pose.orientation.z = math.sin(yaw / 2.0)
            subgoal_msg.pose.orientation.w = math.cos(yaw / 2.0)
        else:
            subgoal_msg.pose.orientation.w = 1.0
        
        self.subgoal_pub.publish(subgoal_msg)
        
        # 서브골 발행 시마다 로그 출력 (반복 발행이 없으므로)
        self.get_logger().info(f'Published NEW subgoal: Node {target_node["id"]} '
                             f'at UTM ({target_node["x"]:.2f}, {target_node["y"]:.2f})')
    
    def advance_to_next_node(self):
        """다음 노드로 이동"""
        if self.current_target_node_index < len(self.path_nodes) - 1:
            self.current_target_node_index += 1
            next_node = self.path_nodes[self.current_target_node_index]
            self.get_logger().info(f'Advanced to next node: {next_node["id"]} '
                                 f'({self.current_target_node_index + 1}/{len(self.path_nodes)})')
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