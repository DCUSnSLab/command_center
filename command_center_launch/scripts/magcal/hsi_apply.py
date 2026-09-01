#!/usr/bin/env python3
"""수렴한 HSI 보정을 적용하고 비휘발 메모리에 저장한다.

절차: 추정 정지(hsiMode=0) + 온보드 적용(hsiOutput=3) -> VNWNV(NV 저장)
     -> 재조회로 확인. VPE headingMode 는 이미 ABSOLUTE 여야 한다.
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
    for baud in (460800, 921600, 115200, 230400):
        s = serial.Serial("/dev/rs232", baud, timeout=0.8)
        s.reset_input_buffer()
        time.sleep(0.4)
        raw = s.read_all()
        s.write(cmd("VNRRG,35"))
        time.sleep(0.5)
        if re.search(r"\$VNRRG,35,",
                     (raw + s.read_all()).decode("latin-1", "replace")):
            print("보드레이트 %d" % baud)
            return s
        s.close()
    sys.exit("센서 무응답")


def rd(p, reg, tries=4):
    pat = re.compile(r"\$VNRRG,%d,[ -~]*?\*[0-9A-Fa-f]{2}" % reg)
    for _ in range(tries):
        p.write(cmd("VNWRG,75,0,1,00"))
        time.sleep(0.3)
        p.reset_input_buffer()
        p.write(cmd("VNRRG,%d" % reg))
        time.sleep(0.45)
        m = pat.search(p.read_all().decode("latin-1", "replace"))
        if m:
            return m.group(0)
    return None


p = open_sensor()
p.write(cmd("VNWRG,75,0,1,00"))
time.sleep(0.4)
p.reset_input_buffer()

print("적용 전 reg47:", rd(p, 47))
# hsiMode=0(추정 정지 — 계산값 동결), hsiOutput=3(USEONBOARD 적용)
p.write(cmd("VNWRG,44,0,3,5"))
time.sleep(0.8)
print("reg44 설정 ->", rd(p, 44))
# VPE 도 명시적으로 ABSOLUTE 확정 (enable=1, headingMode=0, filt/tune=1)
p.write(cmd("VNWRG,35,1,0,1,1"))
time.sleep(0.8)
print("reg35 설정 ->", rd(p, 35))
# 비휘발 저장
p.write(cmd("VNWNV"))
time.sleep(2.5)
print("NV 저장 완료")

for r in (35, 44, 47):
    print("최종 reg%d:" % r, rd(p, r))
p.close()
