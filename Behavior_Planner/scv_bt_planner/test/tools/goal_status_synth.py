#!/usr/bin/env python3
"""bag 재생용 ControllerGoalStatus 합성기.

실차 bag 에는 /goal_status(제어기 → BP)가 녹화돼 있지 않아 BP 가 노드를 넘길 수 없다.
제어기의 도달 판정을 같은 규칙으로 흉내 내 두 BP(simple, bt)에 각각 공급한다.

판정은 **map 프레임**에서 한다: 현재 목표 노드 ID(current_goal.header.frame_id, BP 규약)로
graph.json 의 UtmInfo 를 찾아 datum 을 빼고, /odometry/global(map) 과 비교한다.
odom 프레임 좌표(TF 스냅샷 시점 의존)를 쓰면 앵커 이동 시 판정이 흔들리므로 피한다.
  - 도달 = 거리 <= goal_reached_threshold (마지막 MPPIParams 값, 기본 1.6)
         또는 통과: 목표가 기수 뒤(진행방향 내적 < 0)이고 거리 < passed_goal_max_distance(8.0)
           (최종 노드·pause 노드 7/8 제외 — smppi 규약)
사용: goal_status_synth.py --ros-args -p prefix:=/bt -p map_file:=... ('none' = 접두 없음 = simple BP)
"""
import json
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from command_center_interfaces.msg import ControllerGoalStatus, MPPIParams, MultipleWaypoints


class Synth(Node):
    def __init__(self):
        super().__init__('goal_status_synth')
        self.declare_parameter('prefix', 'none')         # 'none' = 접두 없음 (빈 문자열·'-' 는 rcl 인자로 못 넘긴다)
        self.declare_parameter('map_file', '')
        self.declare_parameter('datum_utm_easting', 482232.7)
        self.declare_parameter('datum_utm_northing', 3974384.47)
        self.declare_parameter('position_topic', '/odometry/global')
        self.declare_parameter('default_threshold', 1.6)
        self.declare_parameter('passed_goal_max_distance', 8.0)
        self.declare_parameter('rate_hz', 10.0)
        p = self.get_parameter('prefix').value
        p = '' if p in ('none', '-', '') else p
        self.thr = float(self.get_parameter('default_threshold').value)
        self.passed_max = float(self.get_parameter('passed_goal_max_distance').value)
        de = float(self.get_parameter('datum_utm_easting').value)
        dn = float(self.get_parameter('datum_utm_northing').value)
        g = json.load(open(self.get_parameter('map_file').value))
        self.nodes = {n['ID']: (n['UtmInfo']['Easting'] - de, n['UtmInfo']['Northing'] - dn)
                      for n in (g.get('Node') or g.get('nodes'))}
        self.wp = None
        self.pose = None
        self.reached_sent_for = None
        self.create_subscription(MultipleWaypoints, p + '/multiple_waypoints', self.wp_cb, 10)
        self.create_subscription(MPPIParams, p + '/mppi_update_params', self.params_cb, 10)
        self.create_subscription(Odometry, self.get_parameter('position_topic').value, self.pose_cb, 20)
        self.pub = self.create_publisher(ControllerGoalStatus, p + '/goal_status', 10)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self.tick)
        self.get_logger().info(f'goal_status synth prefix="{p}" thr={self.thr} nodes={len(self.nodes)}')

    def wp_cb(self, m):
        if self.wp is None or m.current_goal.header.frame_id != self.wp.current_goal.header.frame_id:
            self.reached_sent_for = None
        self.wp = m

    def params_cb(self, m):
        if m.update_control and m.goal_reached_threshold > 0:
            self.thr = float(m.goal_reached_threshold)

    def pose_cb(self, m):
        self.pose = m

    def tick(self):
        if self.wp is None or self.pose is None:
            return
        gid = self.wp.current_goal.header.frame_id
        if gid not in self.nodes:
            self.get_logger().warn(f'unknown goal node id {gid}', throttle_duration_sec=5.0)
            return
        gx, gy = self.nodes[gid]
        o = self.pose.pose.pose
        dx, dy = gx - o.position.x, gy - o.position.y
        d = math.hypot(dx, dy)
        q = o.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        ahead = dx * math.cos(yaw) + dy * math.sin(yaw)
        ntype = self.wp.current_goal_node_type
        passed = (ahead < 0.0 and d < self.passed_max and not self.wp.is_final_waypoint
                  and ntype not in (7, 8))
        reached = d <= self.thr or passed
        m = ControllerGoalStatus()
        m.header.stamp = self.get_clock().now().to_msg()
        m.goal_id = gid
        m.distance_to_goal = d
        if reached and self.reached_sent_for != gid:
            m.goal_reached = True
            m.status_code = 1
            self.reached_sent_for = gid
            self.get_logger().info(f'REACHED {gid} d={d:.2f} passed={passed}')
        else:
            m.goal_reached = False
            m.status_code = 0
        self.pub.publish(m)


def main():
    rclpy.init()
    n = Synth()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
