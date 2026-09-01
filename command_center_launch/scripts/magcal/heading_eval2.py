import math, sys
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import Imu, NavSatFix
from hunter_msgs.msg import HunterStatus

r = SequentialReader()
r.open(StorageOptions(uri=sys.argv[1], storage_id="sqlite3"),
       ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
imu, fix, hs = [], [], []
while r.has_next():
    topic, data, ts = r.read_next()
    t = ts/1e9
    if topic == "/vectornav/imu":
        m = deserialize_message(data, Imu); q = m.orientation
        imu.append((t, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))))
    elif topic == "/ublox_gps_node/fix":
        m = deserialize_message(data, NavSatFix)
        fix.append((t, m.latitude, m.longitude, m.status.status))
    elif topic == "/hunter_status":
        m = deserialize_message(data, HunterStatus)
        hs.append((t, m.linear_velocity))
I, F, H = np.array(imu), np.array(fix), np.array(hs)
LAT0, LON0 = F[0,1], F[0,2]
KX = 111320.0*math.cos(math.radians(LAT0)); KY = 110540.0
ft = F[:,0]; fx = (F[:,2]-LON0)*KX; fy = (F[:,1]-LAT0)*KY
iy = np.unwrap(I[:,1])
out = []
i = 0
while i < len(ft)-1:
    j = i
    while j < len(ft)-1 and math.hypot(fx[j]-fx[i], fy[j]-fy[i]) < 6.0:
        j += 1
        if ft[j]-ft[i] > 20.0: break
    d = math.hypot(fx[j]-fx[i], fy[j]-fy[i]); dt = ft[j]-ft[i]
    if d >= 6.0 and dt <= 20.0:
        v = np.interp(np.linspace(ft[i], ft[j], 20), H[:,0], H[:,1])
        fwd = (v > 0.05).mean(); rev = (v < -0.05).mean()
        bear = math.atan2(fy[j]-fy[i], fx[j]-fx[i])
        yaw = float(np.interp((ft[i]+ft[j])/2, I[:,0], iy))
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        off = math.atan2(math.sin(bear-yaw), math.cos(bear-yaw))
        rtk = (F[i:j,3] == 2).mean()
        if fwd > 0.8: kind = "전진"
        elif rev > 0.8: kind = "후진"
        else: kind = "혼합"
        out.append((kind, math.degrees(bear), math.degrees(yaw), math.degrees(off), d, rtk))
        i = j
    else:
        i += 1
print("구간 %d개 (전진 %d, 후진 %d, 혼합 %d)" % (len(out),
      sum(1 for o in out if o[0]=="전진"), sum(1 for o in out if o[0]=="후진"),
      sum(1 for o in out if o[0]=="혼합")))
print("종류  GPS방위   IMUyaw   오프셋  변위  RTK%")
for k,b,y,o,d,rk in out:
    print("%s  %+7.1f  %+7.1f  %+7.1f  %4.1f  %3.0f" % (k,b,y,o,d,100*rk))
fw = [o[3] for o in out if o[0]=="전진"]
if fw:
    a = np.radians(fw)
    m = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))
    R = math.hypot(np.cos(a).mean(), np.sin(a).mean())
    res = np.degrees(np.arctan2(np.sin(a-math.radians(m)), np.cos(a-math.radians(m))))
    print("\n[전진 %d구간] 오프셋 평균 %+.1f도, 잔차 RMS %.1f도 (R=%.3f)" % (
        len(fw), m, np.sqrt((res**2).mean()), R))
    print("판정:", "상수 오프셋 — autocal 흡수 가능 = 성공" if np.sqrt((res**2).mean()) < 15
          else "방향 의존 — 철기 잔차 남음")
    print("권장 yaw_offset = %.4f rad (%.1f도)" % (math.radians(m), m))
