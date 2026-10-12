#!/usr/bin/env python3
"""scv_bt_planner 통합 스모크 (ROS 그래프 필요, colcon test 대상 아님).

합성 입력을 넣고 /bt/* 출력을 관찰한다:
  - /map_provider_node/utm  datum (latched)
  - map->odom TF (항등)
  - /odometry/global        차량 위치
  - /planned_path_detailed  4노드 경로 (N0001 -> N0002(road) -> N0003(crosswalk, type3) -> N0004(gps_denied), 90도 좌회전 포함)
  - /goal_status            현재 목표까지 거리 (회전 창·pause 트리거용)

기대:
  1) /bt/multiple_waypoints 가 발행되고 current_goal.header.frame_id == 'N0001'
  2) /bt/mppi_params 가 발행되고 current_behavior_type == 1 (Zone 없음 → 동등성 경로)
  3) 목표를 N0002 로 넘기면(goal_status reached) road 오버레이(코드 21, max_v 1.25×)로 바뀐다
  4) N0003 접근(goal_distance 0.5) 시 좌회전 창 → turn_left(코드 25) 또는 crosswalk pause 발행
사용: ros2 run scv_bt_planner bt_planner_node --ros-args -p mode:=shadow -p map_file_path:=<graph_zones.json> &
      python3 shadow_smoke.py
"""
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from command_center_interfaces.msg import (ControllerGoalStatus, MPPIParams, MultipleWaypoints,
                                           PauseCommand, PlannedPath)
from map_interfaces.msg import GraphLayer, MapLink, MapNode, UtmLayer

OE, ON = 482278.238, 3974410.535   # datum


class Smoke(Node):
    def __init__(self):
        super().__init__('bt_shadow_smoke')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.datum_pub = self.create_publisher(UtmLayer, '/map_provider_node/utm', latched)
        self.odom_pub = self.create_publisher(Odometry, '/odometry/global', 10)
        self.path_pub = self.create_publisher(PlannedPath, '/planned_path_detailed', 10)
        self.goal_pub = self.create_publisher(ControllerGoalStatus, '/goal_status', 10)
        self.tf = StaticTransformBroadcaster(self)
        self.wps, self.params, self.behav, self.pauses = [], [], [], []
        self.create_subscription(MultipleWaypoints, '/bt/multiple_waypoints', lambda m: self.wps.append(m), 10)
        self.create_subscription(MPPIParams, '/bt/mppi_params', lambda m: self.params.append(m), 10)
        self.create_subscription(String, '/bt/behavior', lambda m: self.behav.append(m.data), 10)
        self.create_subscription(PauseCommand, '/bt/pause_command', lambda m: self.pauses.append(m), 10)
        self.pose = (0.0, 0.0)
        # 순간이동 재정렬(점프 > max(2 m, 5 m/s·dt))과 정체 감지(4 s 무진행)를 피하려면
        # 위치를 점진적으로 움직여야 한다 — 이 둘은 원본과 같은 정상 거동이다.
        self.target = (0.0, 0.0)
        self.step = 0.05   # m per odom message (20 Hz → 1 m/s)

    def send_static(self):
        u = UtmLayer(); u.utm_zone = 52; u.zone_letter = 'S'; u.northern = True
        u.origin_easting = OE; u.origin_northing = ON
        self.datum_pub.publish(u)
        t = TransformStamped(); t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'; t.child_frame_id = 'odom'; t.transform.rotation.w = 1.0
        self.tf.sendTransform(t)

    def send_odom(self):
        dx, dy = self.target[0] - self.pose[0], self.target[1] - self.pose[1]
        d = math.hypot(dx, dy)
        if d > 1e-6:
            s = min(self.step, d)
            self.pose = (self.pose[0] + dx / d * s, self.pose[1] + dy / d * s)
        o = Odometry(); o.header.stamp = self.get_clock().now().to_msg(); o.header.frame_id = 'map'
        o.pose.pose.position.x, o.pose.pose.position.y = self.pose
        o.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(o)

    def send_path(self):
        p = PlannedPath(); p.header.stamp = self.get_clock().now().to_msg()
        p.path_id = 'smoke'; p.start_node_id = ''; p.goal_node_id = 'N0004'
        g = GraphLayer()
        coords = [('N0001', 0.0, 0.0, 1, 0.0), ('N0002', 5.0, 0.0, 1, 0.0),
                  ('N0003', 10.0, 0.0, 3, 0.0), ('N0004', 10.0, 5.0, 1, 90.0)]
        for nid, x, y, t, h in coords:
            n = MapNode(); n.id = nid; n.node_type = t; n.easting = OE + x; n.northing = ON + y
            n.heading_deg = h; n.source = 'slam'; g.nodes.append(n)
        for i in range(3):
            l = MapLink(); l.id = f'L{i}'; l.from_node_id = coords[i][0]; l.to_node_id = coords[i + 1][0]
            l.length = 5.0; l.bidirectional = True; g.links.append(l)
        p.path_data = g
        self.path_pub.publish(p)

    def send_goal_status(self, goal_id, dist, reached=False):
        m = ControllerGoalStatus(); m.header.stamp = self.get_clock().now().to_msg()
        m.goal_id = goal_id; m.distance_to_goal = dist; m.goal_reached = reached
        m.status_code = 1 if reached else 0
        self.goal_pub.publish(m)

    def spin_for(self, sec):
        end = time.time() + sec
        while time.time() < end:
            self.send_odom()
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    rclpy.init()
    n = Smoke()
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    n.send_static(); n.spin_for(1.0)
    n.send_path(); n.spin_for(1.5)
    check('waypoints published', n.wps, f'(n={len(n.wps)})')
    if n.wps:
        check('current goal = N0001', n.wps[-1].current_goal.header.frame_id == 'N0001',
              f'({n.wps[-1].current_goal.header.frame_id})')
        check('current goal odom x≈0', abs(n.wps[-1].current_goal.pose.position.x) < 1e-3)
    check('mppi params published (type 1)', n.params and n.params[-1].current_behavior_type == 1,
          f'(types={[p.current_behavior_type for p in n.params]})')

    # N0001 도달 → N0002(road). 차량은 N0002 쪽으로 서서히 이동(정체 감지 회피)
    n.target = (4.0, 0.0)
    n.send_goal_status('N0001', 0.2, reached=True); n.spin_for(1.0)
    n.send_goal_status('N0002', 4.5); n.spin_for(0.5)
    check('advanced to N0002', n.wps and n.wps[-1].current_goal.header.frame_id == 'N0002',
          f'({n.wps[-1].current_goal.header.frame_id if n.wps else "-"})')
    check('road overlay code 21', n.params and n.params[-1].current_behavior_type == 21,
          f'(last={n.params[-1].current_behavior_type if n.params else "-"})')

    # N0002 도달 → N0003 (crosswalk, type 3, 좌회전 90도 지점). 접근 4.5 m: 회전 창 6 m 안 → turn_left
    n.target = (9.0, 0.0)
    n.send_goal_status('N0002', 0.2, reached=True); n.spin_for(1.0)
    n.send_goal_status('N0003', 4.5); n.spin_for(0.6)
    check('turn_left overlay code 25 near N0003', n.params and n.params[-1].current_behavior_type == 25,
          f'(last={n.params[-1].current_behavior_type if n.params else "-"} max_v={n.params[-1].max_linear_velocity if n.params else 0:.2f})')
    # 회전 창 밖(7 m)이면 crosswalk 로 내려간다
    n.send_goal_status('N0003', 7.0); n.spin_for(0.6)
    check('crosswalk overlay code 22 outside turn window', n.params and n.params[-1].current_behavior_type == 22,
          f'(last={n.params[-1].current_behavior_type if n.params else "-"})')
    # pause 트리거 거리 안으로 → PauseBeforeEntry 1회
    n.send_goal_status('N0003', 0.5); n.spin_for(0.6)
    check('crosswalk entry pause sent once', len(n.pauses) == 1, f'(n={len(n.pauses)})')
    n.send_goal_status('N0003', 0.4); n.spin_for(0.4)
    check('pause not repeated', len(n.pauses) == 1, f'(n={len(n.pauses)})')

    print('behavior trace:')
    for b in n.behav:
        print('  ', b)
    n.destroy_node(); rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
