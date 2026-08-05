#!/usr/bin/env python3
"""VectorNav 레지스터 질의 (드라이버 정지 후 실행).
판정: reg35 headingMode=0(ABSOLUTE) / reg44 hsiOutput=3(USEONBOARD) /
      reg47 비항등(HSI 보정 존재).
무응답은 성공이 아니다 — 명시적으로 실패로 보고하고 종료코드 2 를 낸다
(초판은 무응답 시 파싱 실패가 "비항등 OK" 로 새어 나갔다)."""
import serial, sys, time

def cmd(s):
    ck = 0
    for c in s:
        ck ^= ord(c)
    return ("$%s*%02X\r\n" % (s, ck)).encode()

try:
    p = serial.Serial("/dev/rs232", 460800, timeout=1.0)
except Exception as e:
    print("시리얼 열기 실패: %s" % e)
    print("드라이버가 떠 있으면 포트를 물고 있다 — 스택 정지 후 재시도.")
    sys.exit(2)

import re

def rd(reg, tries=5):
    """바이너리 비동기 출력(BO1)이 켜져 있으면 ASCII 응답이 그 사이에
    섞여 들어온다 — 줄 단위 파싱은 깨진다(실측: 드라이버 기동 이력이
    있는 세션에서 전 레지스터 무응답). 원시 버퍼 전체를 정규식으로
    훑는다."""
    pat = re.compile(r"\$VNRRG,%d,[ -~]*?\*[0-9A-Fa-f]{2}" % reg)
    for _ in range(tries):
        # VNASY,0 은 ASCII 비동기만 끈다. 드라이버가 한 번이라도 떴던
        # 세션에서는 바이너리 출력(BO1, reg 75)이 계속 흘러 응답이 묻힌다
        # (실측: 전 레지스터 무응답, VNRST 로도 안 멎음). BO1 을 끈다 —
        # 드라이버가 다음 기동 때 config 대로 다시 켜므로 안전하다.
        p.write(cmd("VNWRG,75,0,1,00")); time.sleep(0.4)
        p.write(cmd("VNASY,0")); time.sleep(0.3)
        p.reset_input_buffer()
        p.write(cmd("VNRRG,%d" % reg)); time.sleep(0.5)
        buf = p.read_all().decode("latin-1", errors="replace")
        m = pat.search(buf)
        if m:
            return m.group(0)
    return None

vals = {r: rd(r) for r in (35, 44, 47, 21)}
p.write(cmd("VNASY,1")); p.close()

names = {35: "VPE", 44: "HSI ctl", 47: "HSI 결과", 21: "기준벡터"}
for r in (35, 44, 47, 21):
    print("reg%-3d %-9s: %s" % (r, names[r], vals[r] or "(무응답)"))

missing = [r for r in (35, 44, 47) if vals[r] is None]
if missing:
    print()
    print("판정 불가 — 무응답 레지스터 %s. 센서 통신을 먼저 확인할 것." % missing)
    sys.exit(2)

f35 = vals[35].split("*")[0].split(",")
f44 = vals[44].split("*")[0].split(",")
f47 = vals[47].split("*")[0].split(",")[2:]
ok35 = len(f35) > 3 and f35[3] == "0"
ok44 = len(f44) > 3 and f44[3] == "3"
ident = f47[:9] == ["1","0","0","0","1","0","0","0","1"]
print()
print("headingMode ABSOLUTE:", "OK" if ok35 else "아님 (1=RELATIVE)")
print("HSI 온보드 적용     :", "OK" if ok44 else "아님 (1=계산만, 미적용)")
print("HSI 보정 존재       :", "없음(항등행렬)" if ident else "OK(비항등)")
sys.exit(0 if (ok35 and ok44 and not ident) else 1)
