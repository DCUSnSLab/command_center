#!/bin/bash
# 전체 스택 정지.
#
# 함정 둘을 모두 피한다:
#  1) ssh 원라이너로 `pkill -f` 를 쓰면 패턴이 자기 명령줄과 매칭돼 스스로를
#     죽이고 나머지가 남는다. 그래서 스크립트 파일로 둔다.
#  2) 스크립트로 옮겨도 **호출한 셸의 명령줄**에 'ros2 launch' 같은 문자열이
#     들어 있으면 그 셸까지 죽는다(실제로 당했다). 그래서 자기 자신뿐 아니라
#     **조상 프로세스 전체**를 대상에서 뺀다.
#
# fastlio_mapping 은 반드시 포함한다 — 이전 런의 FAST-LIO 고아가 다음 런의
# odom 을 km 급으로 오염시킨 사고가 있었다(2026-07-30).
set -uo pipefail

PAT='ros2 launch|ros2 bag record|field_drive|sequential_planner|path_planner_node|gmserver|ekf_node|ekf_filter|navsat_transform|map_anchor_node|fastlio_mapping|laser_mapping|curb_detection_node|costmap_node|corridor_keepout|costmap_processor|mppi_main|simple_behavior_planner|wheel_odom_adapter|gps_fix_gate|velodyne|vectornav|sllidar|realsense|ublox|ntrip|hunter'

# 자기 자신 + 조상 전부 (이들을 죽이면 teardown 이 중간에 끊긴다)
SAFE=" "
p=$$
while [ -n "$p" ] && [ "$p" != "1" ] && [ "$p" != "0" ]; do
  SAFE="$SAFE$p "
  p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
done

targets() {
  for pid in $(pgrep -f "$PAT" 2>/dev/null); do
    case "$SAFE" in *" $pid "*) continue ;; esac
    echo "$pid"
  done
}

echo "[teardown] 정지 대상:"
for pid in $(targets); do
  echo "  $pid $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-110)"
done

# 기록부터 멈춘다 — bag 이 깨끗이 닫히도록
for pid in $(targets); do
  grep -qa 'ros2 bag record' /proc/$pid/cmdline 2>/dev/null && kill "$pid" 2>/dev/null
done
sleep 3

for pid in $(targets); do kill "$pid" 2>/dev/null; done
sleep 5

LEFT=$(targets)
if [ -n "$LEFT" ]; then
  echo "[teardown] 잔존 — SIGKILL: $(echo $LEFT | tr '\n' ' ')"
  for pid in $LEFT; do kill -9 "$pid" 2>/dev/null; done
  sleep 2
fi

REMAIN=$(targets)
if [ -z "$REMAIN" ]; then
  echo "[teardown] 전부 종료"
else
  echo "[teardown] 남음: $(echo $REMAIN | tr '\n' ' ')"
fi
