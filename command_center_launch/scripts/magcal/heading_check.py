#!/usr/bin/env python3
"""보정 후 검증: 이동 중 [GPS 실이동 방위 - IMU yaw] 를 10초마다 출력.
드라이버+ublox 필요(전체 스택이면 그대로 됨). RC 직선 주행하며 관찰:
 - 오프셋이 상수(자편각 약 -9도 부근)면 성공 — autocal 이 그 잔차를 흡수
 - 헤딩(주행 방향)에 따라 수십 도씩 출렁이면 경자성 잔차 — HSI 재보정 필요"""
import math, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix

def yaw_of(q):
    return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

LAT0, LON0 = 35.91361, 128.80308
KX = 111320.0*math.cos(math.radians(LAT0)); KY = 110540.0

rclpy.init()
n = Node("heading_check")
S = {"yaw": None, "fixes": []}
n.create_subscription(Imu, "/vectornav/imu",
    lambda m: S.__setitem__("yaw", yaw_of(m.orientation)), 50)
def on_fix(m):
    S["fixes"].append((time.monotonic(), (m.longitude-LON0)*KX,
                       (m.latitude-LAT0)*KY, S["yaw"], m.status.status))
    if len(S["fixes"]) > 200:
        S["fixes"].pop(0)
n.create_subscription(NavSatFix, "/ublox_gps_node/fix", on_fix, 30)
last = 0.0
print("수집 중... (이동해야 측정됨, Ctrl-C 종료)")
while rclpy.ok():
    rclpy.spin_once(n, timeout_sec=0.2)
    now = time.monotonic()
    if now - last < 10.0:
        continue
    last = now
    fx = [f for f in S["fixes"] if now - f[0] < 12.0 and f[3] is not None]
    if len(fx) < 5:
        continue
    dx = fx[-1][1] - fx[0][1]; dy = fx[-1][2] - fx[0][2]
    d = math.hypot(dx, dy)
    if d < 1.0:
        print("정지 중 (변위 %.2f m)" % d)
        continue
    bear = math.atan2(dy, dx)
    yaw = fx[len(fx)//2][3]
    off = math.degrees(math.atan2(math.sin(bear-yaw), math.cos(bear-yaw)))
    rtk = sum(1 for f in fx if f[4] == 2)
    print("오프셋(실이동-IMU): %+7.1f도  변위 %.1fm  RTK %d/%d" % (
        off, d, rtk, len(fx)))
