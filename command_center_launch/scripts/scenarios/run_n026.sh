#!/bin/bash
# 주차장 구석(N026) 자율주행 시나리오 — 2026-08-06 판.
#
# 8/5 실패에서 바뀐 것:
#   * 지도: d2_unha (목표 N026 이 기준점과 0.04 m 일치). 8/5 에 이 파일이
#     차량에 없어 다른 지도로 대체했다가 목표가 13 m 어긋나고 진입 방향이
#     반대가 됐다. field_bringup 이 이제 지도 부재 시 중단한다.
#   * 합류 규칙: 최근접 -> 비용 기반(접근*1.5 + 잔여). 챔버 A/B 에서
#     최근접은 뒤쪽 노드로 합류하다 미도달, 비용 기반은 도달.
#   * LiDAR 링크 사전 점검 (8/5 에 enp7s0 이 DOWN 이라 전면 LETHAL 이었다).
#   * yaw 시드 +103.2도 + 온라인 autocal.
#
# 사용: run_n026.sh              # 기록 포함 기동
#       run_n026.sh --no-camera # 카메라 제외 기록(실내->실외 이동 시 권장)
#       run_n026.sh --dry       # 사전 점검만 (스택 미기동)
set -o pipefail
# 지도 정본은 저장소(GraphMap_Server/maps/d2_unha.json)다. 차량 사본이 없으면
# field_bringup 이 중단하므로, 정본을 배포 경로로 복사해 두고 쓴다.
MAP=${SCV_MAP:-/home/scv/MAP/D2/d2_unha.json}
GOAL=N026
S=/home/scv/SCV_park/src/command_center/command_center_launch/scripts

echo "=== 사전 점검 ==="
[ -f "$MAP" ] && echo "  지도 OK: $MAP" || { echo "  !! 지도 없음: $MAP"; exit 1; }
ip link show enp7s0 2>/dev/null | grep -q LOWER_UP \
  && echo "  LiDAR 링크 OK" || echo "  !! LiDAR 링크 DOWN — 자율주행 불가"
ping -c 2 -W 2 192.168.1.201 > /dev/null 2>&1 \
  && echo "  LiDAR 응답 OK" || echo "  !! LiDAR 192.168.1.201 무응답"
ip link show can0 2>/dev/null | grep -q "state UP" \
  && echo "  can0 UP" || echo "  can0 DOWN (기동 시 자동으로 올린다)"
df -h /home | awk 'NR==2 {print "  디스크 여유 " $4}'

[ "${1:-}" = "--dry" ] && { echo "(사전 점검만 수행)"; exit 0; }
EXTRA=()
[ "${1:-}" = "--no-camera" ] && EXTRA+=(--no-camera)

echo
echo "=== 기동: 목표 $GOAL (주차장 구석) ==="
# "${EXTRA[@]:-}" 로 쓰면 안 된다: 배열이 비었을 때 :- 가 **빈 문자열 인자
# 하나**를 만들어 내고, 그게 field_bringup 을 지나 ros2 launch 까지 가서
# "malformed launch argument ''" 로 기동이 통째로 죽는다. --no-camera 없이
# 부를 때만 터지는 조용한 함정이라 실측 전까지 드러나지 않았다.
"$S/field_bringup.sh" --record ${EXTRA[@]+"${EXTRA[@]}"} \
  map_file_path:="$MAP" \
  route_source:=sequential goal_node:="$GOAL" \
  corridor_half_width:=25.0

cat <<'MSG'

────────────────────────────────────────────────────────
자율 전환 전 필수 절차 (yaw 게이트)
  1) RC 수동으로 전진 10 m  (RTK 고정이면 5 m 로 충분)
  2) 아래 중 하나로 보정 발동 확인
       ros2 topic echo /map_anchor/yaw_corr
       grep "yaw autocal" ~/field_*/field_drive.log | tail -3
  3) ENGAGED 확인 후 자율 전환

정지는  ~/SCV_park/src/command_center/command_center_launch/scripts/teardown.sh
────────────────────────────────────────────────────────
MSG
