#!/bin/bash
# VectorNav 자기 보정 현장 절차. 사용:
#   ./magcal_run.sh pre     # (드라이버 정지 상태) 보정 전 레지스터 기록
#   ./magcal_run.sh cal     # (드라이버/스택 기동 상태) 보정 액션 — 실행 즉시
#                           #  RC 저속(<=0.5 m/s) 원/8자 주행 2~3바퀴 유지
#   ./magcal_run.sh post    # (드라이버 정지 상태) 보정 결과 검증
# 성공 기준: post 에서 ABSOLUTE / USEONBOARD / 비항등 모두 OK.
# 이후 스택 재기동, heading_check.py 로 직선 주행 오프셋이 상수인지 확인.
set -o pipefail
D="$(cd "$(dirname "$0")" && pwd)"
case "${1:-}" in
  pre)
    python3 "$D/vn_reg.py" | tee "$D/regs_pre_$(date +%m%d_%H%M).log" ;;
  cal)
    source /opt/ros/humble/setup.bash
    source /home/scv/SCV_park/install/setup.bash
    echo "[cal] 보정 시작 — 지금부터 RC 저속 원/8자 주행 (수렴 시 자동 종료, 최대 1000표본)"
    ros2 action send_goal --feedback /vectornav/mag_cal \
      vectornav_msgs/action/MagCal "{}" | tee "$D/cal_$(date +%m%d_%H%M).log" ;;
  post)
    python3 "$D/vn_reg.py" | tee "$D/regs_post_$(date +%m%d_%H%M).log" ;;
  *) sed -n 2,9p "$0" ;;
esac
