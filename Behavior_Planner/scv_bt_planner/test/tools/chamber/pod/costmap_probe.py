#!/usr/bin/env python3
"""costmap + TF(map->costmap frame) + odom 를 함께 떠서, BT 위험 경유점 판정(점 반경 0.75, 치명>=90)을 노드별로 재현."""
import sys, json, math, time
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
import tf2_ros
G = json.load(open(sys.argv[1]))
nodes = [(n['ID'], n['UtmInfo']['Easting'], n['UtmInfo']['Northing']) for n in G['Node']]
class N(Node):
    def __init__(s):
        super().__init__('costmap_probe'); s.m = None; s.od = None
        s.buf = tf2_ros.Buffer(); s.tl = tf2_ros.TransformListener(s.buf, s)
        s.create_subscription(OccupancyGrid, '/costmap', lambda m: setattr(s, 'm', m), 1)
        s.create_subscription(Odometry, '/odometry/global', lambda m: setattr(s, 'od', m), 1)
rclpy.init(); n = N(); t0 = time.time()
tr = None
while time.time() - t0 < 25:
    rclpy.spin_once(n, timeout_sec=0.2)
    if n.m is not None and n.od is not None:
        try:
            tr = n.buf.lookup_transform(n.m.header.frame_id, 'map', rclpy.time.Time()); break
        except Exception: pass
if tr is None: print('missing', n.m is None, n.od is None); sys.exit(1)
m = n.m; i = m.info
a = np.array(m.data, dtype=np.int16).reshape(i.height, i.width)
q = tr.transform.rotation; th = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
tx, ty = tr.transform.translation.x, tr.transform.translation.y
c, s = math.cos(th), math.sin(th)
def cost(x, y):
    gx = c*x - s*y + tx; gy = s*x + c*y + ty
    col = int(math.floor((gx - i.origin.position.x)/i.resolution)); row = int(math.floor((gy - i.origin.position.y)/i.resolution))
    if col < 0 or row < 0 or col >= i.width or row >= i.height: return None
    return int(a[row, col])
p = n.od.pose.pose.position
print(f'veh map ({p.x:.2f},{p.y:.2f})  tf map->{m.header.frame_id}: t=({tx:.2f},{ty:.2f}) yaw={math.degrees(th):.1f}')
for nid, x, y in nodes:
    if math.hypot(x - p.x, y - p.y) > 9: continue
    ring = []
    for r in (0.0, 0.375, 0.75):
        for k in (range(8) if r else [0]):
            v = cost(x + r*math.cos(k*math.pi/4), y + r*math.sin(k*math.pi/4)); ring.append(v)
    bad = [v for v in ring if v is not None and v >= 90]
    print(f'{nid} ({x:6.1f},{y:5.1f}) d={math.hypot(x-p.x,y-p.y):4.1f}  center={cost(x,y)}  ring_max={max([v for v in ring if v is not None], default=None)}  lethal_samples={len(bad)}')
np.save('/tmp/cm_probe.npy', a)
