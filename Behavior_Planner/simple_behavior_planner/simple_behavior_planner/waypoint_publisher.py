#!/usr/bin/env python3
"""
Waypoint Publisher Module
웨이포인트 발행을 담당하는 모듈
"""

import math
from typing import Dict, Any, List
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from command_center_interfaces.msg import MultipleWaypoints
from map_interfaces.msg import UtmLayer
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs.tf2_geometry_msgs as tf2_geometry_msgs


class WaypointPublisher:
    """웨이포인트 발행 클래스"""

    def __init__(self, node: Node, waypoint_mode: str = 'multiple'):
        self.node = node
        self.waypoint_mode = waypoint_mode

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        # datum (map 프레임 원점) — map_provider가 /map_provider_node/utm 로 latched 발행.
        # PlannedPath 노드 좌표는 절대 UTM 이므로 datum 을 빼서 map(datum-local) 좌표로 변환한다.
        self.origin_easting = None
        self.origin_northing = None
        datum_qos = QoSProfile(depth=1)
        datum_qos.reliability = ReliabilityPolicy.RELIABLE
        datum_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.datum_sub = node.create_subscription(
            UtmLayer, '/map_provider_node/utm', self._datum_callback, datum_qos)

        # Publishers setup will be done by main node
        self.single_waypoint_pub = None
        self.multiple_waypoints_pub = None

    def _datum_callback(self, msg: UtmLayer):
        self.origin_easting = msg.origin_easting
        self.origin_northing = msg.origin_northing

    def _datum_ready(self) -> bool:
        return self.origin_easting is not None and self.origin_northing is not None

    def set_publishers(self, single_pub=None, multiple_pub=None):
        """Publisher 설정"""
        self.single_waypoint_pub = single_pub
        self.multiple_waypoints_pub = multiple_pub

    def publish_waypoints(self, current_node: Dict[str, Any], next_nodes: List[Dict[str, Any]] = None,
                         path_info: Dict[str, Any] = None):
        """웨이포인트 발행"""
        # datum 미수신 시 발행 스킵 (절대 UTM 웨이포인트가 컨트롤러로 새는 것 방지)
        if not self._datum_ready():
            self.node.get_logger().warn(
                'datum(/map_provider_node/utm) 미수신 — waypoint 발행 대기', throttle_duration_sec=5.0)
            return
        if self.waypoint_mode == 'single':
            self._publish_single_waypoint(current_node)
        elif self.waypoint_mode == 'multiple':
            self._publish_multiple_waypoints(current_node, next_nodes or [], path_info or {})
        else:
            # Fallback - publish both
            self._publish_multiple_waypoints(current_node, next_nodes or [], path_info or {})
            self._publish_single_waypoint(current_node)

    def _publish_single_waypoint(self, node: Dict[str, Any]):
        """단일 웨이포인트 발행"""
        if not self.single_waypoint_pub:
            return

        try:
            pose_stamped = self._create_pose_stamped(node)
            if pose_stamped is None:
                return
            self.single_waypoint_pub.publish(pose_stamped)
            self.node.get_logger().debug(f'Published single waypoint: Node {node["id"]}')
        except Exception as e:
            self.node.get_logger().error(f'Failed to publish single waypoint: {e}')

    def _publish_multiple_waypoints(self, current_node: Dict[str, Any],
                                   next_nodes: List[Dict[str, Any]],
                                   path_info: Dict[str, Any]):
        """다중 웨이포인트 발행"""
        if not self.multiple_waypoints_pub:
            return

        try:
            waypoints_msg = MultipleWaypoints()
            waypoints_msg.header.stamp = self.node.get_clock().now().to_msg()
            waypoints_msg.header.frame_id = 'odom'

            # 현재 목표 설정
            current_goal = self._create_pose_stamped(current_node)
            if current_goal is None:
                return   # map->odom 미가용 -> 발행 스킵
            waypoints_msg.current_goal = current_goal
            waypoints_msg.current_goal_node_type = current_node.get('node_type', 1)
            waypoints_msg.current_goal_reverse_heading = self._is_reverse_behavior(
                current_node.get('node_type', 1))

            # 다음 목표들 설정
            waypoints_msg.next_waypoints = []
            waypoints_msg.next_waypoints_node_types = []
            waypoints_msg.next_waypoints_reverse_heading = []

            for node in next_nodes:
                wp = self._create_pose_stamped(node)
                if wp is None:
                    continue   # 병렬 배열 정합 위해 세 항목 함께 스킵
                waypoints_msg.next_waypoints.append(wp)
                node_type = node.get('node_type', 1)
                waypoints_msg.next_waypoints_node_types.append(node_type)
                waypoints_msg.next_waypoints_reverse_heading.append(
                    self._is_reverse_behavior(node_type))

            # 경로 정보 설정
            waypoints_msg.path_id = path_info.get('path_id', '')
            waypoints_msg.current_waypoint_index = path_info.get('current_index', 0)
            waypoints_msg.total_waypoints = path_info.get('total_nodes', 0)
            waypoints_msg.is_final_waypoint = path_info.get('is_final_node', False)

            self.multiple_waypoints_pub.publish(waypoints_msg)
            self.node.get_logger().debug(f'Published multiple waypoints: current={current_node["id"]}, '
                                       f'next_count={len(next_nodes)}')

        except Exception as e:
            self.node.get_logger().error(f'Failed to publish multiple waypoints: {e}')

    def _create_pose_stamped(self, node: Dict[str, Any]) -> PoseStamped:
        """노드 정보로부터 PoseStamped 생성"""
        try:
            # Create pose in map frame
            map_pose = PoseStamped()
            map_pose.header.stamp = self.node.get_clock().now().to_msg()
            map_pose.header.frame_id = 'map'

            # 절대 UTM -> map(datum-local): datum 뺄셈
            map_pose.pose.position.x = node['x'] - self.origin_easting
            map_pose.pose.position.y = node['y'] - self.origin_northing
            map_pose.pose.position.z = node['z']

            # Set orientation
            self._set_pose_orientation(map_pose, node)

            # Transform to odom frame
            transform = self.tf_buffer.lookup_transform('odom', 'map', rclpy.time.Time())
            odom_pose = tf2_geometry_msgs.do_transform_pose_stamped(map_pose, transform)
            odom_pose.header.frame_id = node['id']

            return odom_pose

        except Exception as e:
            # TF(map->odom) 미가용 시 잘못된 좌표를 컨트롤러로 보내지 않도록 발행하지 않음.
            self.node.get_logger().warn(
                f'map->odom TF 미가용, waypoint 스킵: {e}', throttle_duration_sec=5.0)
            return None

    def _set_pose_orientation(self, pose: PoseStamped, node: Dict[str, Any]):
        """포즈 방향 설정"""
        if 'heading' in node and node['heading'] is not None:
            # Convert geographic heading to odom heading
            odom_heading_rad = self._convert_geographic_to_odom_heading(node['heading'])
            pose.pose.orientation.z = math.sin(odom_heading_rad / 2.0)
            pose.pose.orientation.w = math.cos(odom_heading_rad / 2.0)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
        else:
            # Default orientation (facing forward)
            pose.pose.orientation.w = 1.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0

    def _convert_geographic_to_odom_heading(self, geographic_heading_deg: float) -> float:
        """지리학적 heading을 odom heading으로 변환"""
        math_angle_deg = (geographic_heading_deg) % 360
        return math.radians(math_angle_deg)

    def _is_reverse_behavior(self, behavior_type: int) -> bool:
        """행동 타입이 후진인지 확인"""
        # 후진 행동 타입들 (2, 4)
        return behavior_type in [2, 4]