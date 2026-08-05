#!/usr/bin/env python3
"""온보드 HSI(경자성) 추정을 직접 구동한다 — 드라이버 액션이 조기 수렴한 경우의 대안.

드라이버의 mag_cal 액션은 readCalculatedMagnetometerCalibration() 값이
'변하지 않으면' 수렴으로 판정한다. 추정이 아예 돌지 않아 0 만 반환하면
11 초 만에 SUCCEEDED 로 끝나고 결과는 전부 0 이 된다(2026-08-05 현장 실측).
여기서는 reg44 로 HSI 추정을 RUN 으로 켜고 reg47(계산 결과)을 직접
폴링해 실제로 항등에서 벗어나는지 관찰한다.

사용: hsi_run.py [관측초=180]
이후 적용은 hsi_apply.py.
"""
import re
import sys
import time

import serial


def _ck(s):
    c = 0
    for ch in s:
        c ^= ord(ch)
    return c


def cmd(s):
    return ("$%s*%02X\r\n" % (s, _ck(s))).encode()


def open_sensor():
    """보드레이트 자동 감지 — 설정은 460800 이지만 센서 NV 는 921600 이고
    vnproglib connect() 가 기동 때마다 바꿔놓는다(현장 실측)."""
    for baud in (460800, 921600, 115200, 230400):
        s = serial.Serial("/dev/rs232", baud, timeout=0.8)
        s.reset_input_buffer()
        time.sleep(0.4)
        raw = s.read_all()
        s.write(cmd("VNRRG,35"))
        time.sleep(0.5)
        buf = (raw + s.read_all()).decode("latin-1", "replace")
        if re.search(r"\$VNRRG,35,", buf):
            print("보드레이트 %d" % baud, flush=True)
            return s
        s.close()
    sys.exit("센서 무응답")


def rd(p, reg, tries=4):
    pat = re.compile(r"\$VNRRG,%d,[ -~]*?\*[0-9A-Fa-f]{2}" % reg)
    for _ in range(tries):
        p.write(cmd("VNWRG,75,0,1,00"))   # 바이너리 출력 정지(응답 묻힘 방지)
        time.sleep(0.3)
        p.reset_input_buffer()
        p.write(cmd("VNRRG,%d" % reg))
        time.sleep(0.45)
        m = pat.search(p.read_all().decode("latin-1", "replace"))
        if m:
            return m.group(0)
    return None


DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
p = open_sensor()
p.write(cmd("VNWRG,75,0,1,00"))
time.sleep(0.4)
p.reset_input_buffer()
# hsiMode=1(RUN), hsiOutput=1(계산만 — 아직 미적용), 수렴율 5(빠름)
p.write(cmd("VNWRG,44,1,1,5"))
time.sleep(0.6)
print("HSI 추정 RUN ->", rd(p, 44), flush=True)
print("RC 원 주행 계속 — %.0f초 관측\n" % DUR, flush=True)
t0 = time.time()
while time.time() - t0 < DUR:
    r47 = rd(p, 47)
    if r47:
        f = r47.split("*")[0].split(",")[2:]
        ident = f[:9] == ["1", "0", "0", "0", "1", "0", "0", "0", "1"]
        print("%5.0fs  %s  %s" % (time.time() - t0,
                                  "항등(미수렴)" if ident else "비항등!",
                                  ",".join(f[:12])), flush=True)
    time.sleep(10)
p.close()
