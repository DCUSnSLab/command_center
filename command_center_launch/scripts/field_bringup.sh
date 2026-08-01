#!/bin/bash
# 필드 기동 한 방 — can0 를 올린 뒤 field_drive.launch.py 를 띄운다.
#
# can0 를 런치 파일에 넣지 않은 이유: `ip link set` 은 sudo 가 필요한데 런치에서
# sudo 를 부르면 비밀번호 프롬프트가 런치 로그에 묻혀 조용히 멈춘다. 그래서
# 특권이 필요한 한 줄만 여기서 처리하고 나머지는 전부 순정 ROS 런치로 둔다.
#
# 사용:
#   field_bringup.sh                                  # 기본(시작노드 미지정 경로)
#   field_bringup.sh --record                         # 기동 + bag 기록
#   field_bringup.sh route_source:=graph              # 인자는 그대로 런치에 전달
#   field_bringup.sh --record map_file_path:=/x.json
set -uo pipefail

WS=${SCV_WS:-/home/scv/SCV_park}
LOGDIR=${SCV_LOGDIR:-/home/scv/field_$(date +%Y%m%d_%H%M%S)}
RECORD=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --record) RECORD=1 ;;
    *)        ARGS+=("$a") ;;
  esac
done

mkdir -p "$LOGDIR"
echo "[bringup] 로그 $LOGDIR"

# --- 1. can0 --------------------------------------------------------------
if ip link show can0 2>/dev/null | grep -q "state UP"; then
  echo "[bringup] can0 이미 UP"
else
  echo "[bringup] can0 올리는 중 (sudo)"
  sudo ip link set can0 up type can bitrate 500000 || {
    echo "[bringup] can0 실패 — 차량 베이스가 응답하지 않는다. 중단."; exit 1; }
fi
ip link show can0 | grep -o "state [A-Z]*" | sed 's/^/[bringup] can0 /'

# --- 2. 스택 --------------------------------------------------------------
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"

echo "[bringup] field_drive.launch.py ${ARGS[*]:-(기본 인자)}"
nohup ros2 launch command_center_launch field_drive.launch.py "${ARGS[@]}" \
  > "$LOGDIR/field_drive.log" 2>&1 &
LAUNCH_PID=$!
echo "[bringup] launch pid $LAUNCH_PID"

# --- 3. 기록 (선택) --------------------------------------------------------
# 기동 직후가 아니라 스택이 다 뜬 뒤에 시작한다. 초기화 과도구간(FAST-LIO 수렴
# 전 수백 m 스파이크)이 bag 앞머리에 들어가면 재생 분석에서 매번 걸린다.
if [ "$RECORD" = 1 ]; then
  echo "[bringup] 스택 기동 대기 20 s 후 기록 시작"
  sleep 20
  nohup ros2 bag record -o "$LOGDIR/bag" \
    /clock /tf /tf_static /odom /odometry/global /odometry/filtered \
    /gps/fix /gps/filtered /vectornav/imu /velodyne_points /velodyne_points_curb \
    /costmap /costmap_keepout /planned_path_detailed /cmd_vel \
    /hunter_status /joint_states /behavior_state \
    > "$LOGDIR/record.log" 2>&1 &
  echo "[bringup] record pid $!"
fi

sleep 20
echo "[bringup] 기동 후 노드 목록:"
ros2 node list 2>/dev/null | sort | sed 's/^/  /'
echo "[bringup] BRINGUP_DONE — 정지는 teardown.sh"
