#!/usr/bin/env python3
"""시나리오 웨이포인트 러너 + GPS 안정화 시작 게이트.

동작:
  1) GATE — GPS σ(=sqrt(position_covariance[0]))를 감시하며 출발을 보류한다.
     출발 조건 (min_wait 경과 후):
       - 안정화: σ <= sigma_ok 가 stable_hold 초 연속 유지, 또는
       - 횡보: 최근 plateau_window 초 동안 최저 σ 개선폭 < plateau_eps (임계 미달로
         더 좋아지지 않는 상태 — 기다려도 소용없으므로 출발), 또는
       - max_wait 경과 (경고 후 출발).
     min_wait 기본 175 s = 보유 31개 bag 중 σ<=0.15 도달 16개 세션의 평균 소요.
  2) RUN — waypoints 를 순서대로 /goal_node_id 로 발행. /odometry/global 이
     현재 웨이포인트 반경 arrive_radius 안에 arrive_hold 초 머물면 다음으로.
  3) DONE — 마지막 웨이포인트 도달 후 상태만 유지.

좌표: 그래프 UtmInfo − (datum_utm_easting, datum_utm_northing) = 주행 스택
map 프레임 (sequential planner 와 동일 규약 — DCU_0819 운용 시 구 d2 datum).
"""
import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix


class ScenarioRunner(Node):
    def __init__(self):
        super().__init__('scenario_runner')
        self.declare_parameter('waypoints', '')            # "N0426,NX0004,..."
        self.declare_parameter('map_file', '')             # graph.json 절대경로
        self.declare_parameter('goal_topic', '/goal_node_id')
        self.declare_parameter('position_topic', '/odometry/global')
        self.declare_parameter('gps_topic', '/ublox_gps_node/fix')
        self.declare_parameter('datum_utm_easting', 482232.7)
        self.declare_parameter('datum_utm_northing', 3974384.47)
        self.declare_parameter('sigma_ok', 0.15)
        self.declare_parameter('stable_hold', 5.0)
        self.declare_parameter('min_wait', 175.0)
        self.declare_parameter('plateau_window', 30.0)
        self.declare_parameter('plateau_eps', 0.03)
        self.declare_parameter('max_wait', 420.0)
        self.declare_parameter('arrive_radius', 0.6)
        self.declare_parameter('arrive_hold', 2.0)
        self.declare_parameter('goal_repeat', 4)           # 목표 재발행 횟수(1 Hz)

        g = lambda k: self.get_parameter(k).value
        self.waypoints = [w.strip() for w in str(g('waypoints')).split(',') if w.strip()]
        if not self.waypoints:
            raise RuntimeError('waypoints 파라미터가 비어 있음')
        de, dn = float(g('datum_utm_easting')), float(g('datum_utm_northing'))
        graph = json.load(open(str(g('map_file'))))
        self.node_xy = {}
        for n in graph['Node']:
            u = n.get('UtmInfo') or {}
            self.node_xy[n['ID']] = (u['Easting'] - de, u['Northing'] - dn)
        missing = [w for w in self.waypoints if w not in self.node_xy]
        if missing:
            raise RuntimeError(f'그래프에 없는 웨이포인트: {missing}')

        self.sigma_ok = float(g('sigma_ok'))
        self.stable_hold = float(g('stable_hold'))
        self.min_wait = float(g('min_wait'))
        self.plateau_window = float(g('plateau_window'))
        self.plateau_eps = float(g('plateau_eps'))
        self.max_wait = float(g('max_wait'))
        self.arrive_radius = float(g('arrive_radius'))
        self.arrive_hold = float(g('arrive_hold'))
        self.goal_repeat = int(g('goal_repeat'))

        self.goal_pub = self.create_publisher(String, str(g('goal_topic')), 10)
        self.status_pub = self.create_publisher(String, '/scenario/status', 10)
        self.create_subscription(NavSatFix, str(g('gps_topic')), self.gps_cb, 20)
        self.create_subscription(Odometry, str(g('position_topic')), self.odom_cb, 20)

        self.t0 = self.now()
        self.state = 'GATE'
        self.sig_hist = []           # (t, sigma)
        self.stable_since = None
        self.pos = None
        self.wp_i = -1
        self.arrive_since = None
        self.repeat_left = 0
        self.leg_t0 = None
        self.timer = self.create_timer(1.0, self.tick)
        self.get_logger().info(
            f'시나리오 {len(self.waypoints)}개 웨이포인트: {" → ".join(self.waypoints)} | '
            f'게이트: min_wait {self.min_wait:.0f}s, σ_ok {self.sigma_ok}, '
            f'plateau {self.plateau_window:.0f}s/{self.plateau_eps}m, max {self.max_wait:.0f}s')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def gps_cb(self, m):
        var = m.position_covariance[0]
        sig = math.sqrt(var) if var > 0 else 99.0
        if m.status.status < 0:
            sig = 99.0
        t = self.now()
        self.sig_hist.append((t, sig))
        cut = t - max(self.plateau_window, self.stable_hold) - 5.0
        while self.sig_hist and self.sig_hist[0][0] < cut:
            self.sig_hist.pop(0)
        if sig <= self.sigma_ok:
            if self.stable_since is None:
                self.stable_since = t
        else:
            self.stable_since = None

    def odom_cb(self, m):
        self.pos = (m.pose.pose.position.x, m.pose.pose.position.y)

    def gate_ready(self):
        t = self.now()
        el = t - self.t0
        if el < self.min_wait:
            return False, f'대기 {el:.0f}/{self.min_wait:.0f}s'
        if self.stable_since is not None and t - self.stable_since >= self.stable_hold:
            return True, f'GPS 안정화 (σ<= {self.sigma_ok}, {t-self.stable_since:.0f}s 유지)'
        if el >= self.max_wait:
            return True, f'max_wait {self.max_wait:.0f}s 초과 — 강제 출발'
        recent = [s for (ts, s) in self.sig_hist if ts >= t - self.plateau_window]
        older = [s for (ts, s) in self.sig_hist if ts < t - self.plateau_window]
        if len(recent) >= 10 and older:
            improve = min(older) - min(recent)
            if improve < self.plateau_eps:
                cur = min(recent)
                return True, f'σ 횡보 (최근 {self.plateau_window:.0f}s 개선 {improve:+.3f}m, 현재 σ~{cur:.2f}) — 출발'
        return False, f'σ 개선 대기 중 ({el:.0f}s)'

    def send_goal(self, wp):
        self.goal_pub.publish(String(data=wp))

    def tick(self):
        if self.state == 'GATE':
            ready, why = self.gate_ready()
            self.status_pub.publish(String(data=f'GATE: {why}'))
            if int(self.now() - self.t0) % 10 == 0:
                self.get_logger().info(f'[게이트] {why}')
            if ready:
                self.get_logger().info(f'[게이트 통과] {why}')
                self.state = 'RUN'
                self.advance()
            return
        if self.state == 'RUN':
            if self.repeat_left > 0:
                self.send_goal(self.waypoints[self.wp_i])
                self.repeat_left -= 1
            wp = self.waypoints[self.wp_i]
            tx, ty = self.node_xy[wp]
            if self.pos is None:
                self.status_pub.publish(String(data=f'RUN {wp}: 위치 미수신'))
                return
            d = math.hypot(self.pos[0] - tx, self.pos[1] - ty)
            self.status_pub.publish(
                String(data=f'RUN {self.wp_i+1}/{len(self.waypoints)} {wp}: {d:.1f} m'))
            t = self.now()
            if d <= self.arrive_radius:
                if self.arrive_since is None:
                    self.arrive_since = t
                elif t - self.arrive_since >= self.arrive_hold:
                    self.get_logger().info(
                        f'[도달] {wp} (잔여거리 {d:.1f} m, 구간 {t-self.leg_t0:.0f}s)')
                    self.advance()
            else:
                self.arrive_since = None
            return
        # DONE
        self.status_pub.publish(String(data='DONE'))

    def advance(self):
        self.wp_i += 1
        self.arrive_since = None
        if self.wp_i >= len(self.waypoints):
            self.state = 'DONE'
            self.get_logger().info('[완료] 모든 웨이포인트 도달')
            return
        wp = self.waypoints[self.wp_i]
        self.repeat_left = self.goal_repeat
        self.leg_t0 = self.now()
        self.send_goal(wp)
        self.get_logger().info(f'[목표 {self.wp_i+1}/{len(self.waypoints)}] {wp} 발행')


def main():
    rclpy.init()
    rclpy.spin(ScenarioRunner())


if __name__ == '__main__':
    main()
