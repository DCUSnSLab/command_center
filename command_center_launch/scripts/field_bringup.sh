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
#
# set -u 는 colcon setup.bash 소싱 **후에** 켠다. 먼저 켜면 setup.bash 의
# unbound 변수에서 조용히 죽는다 — 이 저장소에서 네 번째로 밟은 함정.
set -o pipefail

WS=${SCV_WS:-/home/scv/SCV_park}
LOGDIR=${SCV_LOGDIR:-/home/scv/field_$(date +%Y%m%d_%H%M%S)}
RECORD=0
NOCAM=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --record)    RECORD=1 ;;
    # 카메라 원본이 bag 의 대부분이다 — 2026-08-06 실측 4.8 GB/분 중
    # 거의 전부. 실내→실외 이동처럼 긴 구간을 기록할 때 빼면 90% 이상
    # 준다(30분 이동 = 145 GB -> 10 GB 남짓).
    --no-camera) NOCAM=1 ;;
    *)           ARGS+=("$a") ;;
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

# --- 1b. LiDAR 이더넷 ------------------------------------------------------
# 2026-08-05 필드: enp7s0 이 재부팅 후 DOWN 인 채로 기동했다. 증상이 아주
# 나쁘게 나온다 — 스택은 정상 기동하고 자율 전환도 되지만, 포인트클라우드가
# 없으니 local_costmap 이 안전측으로 전면 LETHAL(40,000셀 전부)을 발행하고
# MPPI 가 매 주기 비상정지한다. 로그 겉면은 "장애물 때문에 못 감"처럼 보여
# 현장에서 원인 찾는 데 6분을 썼다. 그래서 여기서 먼저 잡고, 못 잡으면
# 조용히 진행하지 말고 크게 경고한다.
LIDAR_IF=${SCV_LIDAR_IF:-enp7s0}
LIDAR_IP=${SCV_LIDAR_IP:-192.168.1.201}
if ip link show "$LIDAR_IF" 2>/dev/null | grep -q "LOWER_UP"; then
  echo "[bringup] $LIDAR_IF 링크 정상"
else
  echo "[bringup] $LIDAR_IF 내려가 있음 — 올리는 중 (sudo)"
  sudo ip link set "$LIDAR_IF" up 2>/dev/null
  sleep 3
fi
if ip link show "$LIDAR_IF" 2>/dev/null | grep -q "NO-CARRIER"; then
  echo "[bringup] !! $LIDAR_IF NO-CARRIER — 케이블/전원 확인 필요."
  echo "[bringup] !! 이대로 기동하면 코스트맵이 전면 LETHAL 이라 자율주행 불가."
elif ping -c 2 -W 2 "$LIDAR_IP" > /dev/null 2>&1; then
  echo "[bringup] LiDAR $LIDAR_IP 응답 OK"
else
  echo "[bringup] !! LiDAR $LIDAR_IP 무응답 (링크는 있음) — IP 설정/전원 확인."
  echo "[bringup] !! 이대로 기동하면 코스트맵이 전면 LETHAL 이라 자율주행 불가."
fi

# --- 1c. 시나리오 지도 존재 확인 -------------------------------------------
# 2026-08-05: 쓰려던 d2_unha.json 이 차량에 없어 현장에서 다른 지도로 대체
# 기동했다. 그 지도에는 목표 지점(주차장 구석)에 대응하는 노드가 없어 13 m
# 떨어진 노드를 목표로 삼았고, 차량 위치가 그 지도 경로의 '꼬리 뒤'라
# 자율 전환 직후 목표 반대쪽으로 향했다(올바른 지도였다면 최근접 노드가
# 목표와 같은 방향 +160도, 12 m 였다). 지도가 없으면 조용히 대체하지 말고
# 여기서 멈춘다.
for a in "${ARGS[@]:-}"; do
  case "$a" in
    map_file_path:=*)
      MF="${a#map_file_path:=}"
      if [ ! -f "$MF" ]; then
        echo "[bringup] !! 지도 파일 없음: $MF"
        echo "[bringup] !! 다른 지도로 대체하지 말 것 — 목표 노드가 달라진다."
        exit 1
      fi
      echo "[bringup] 지도 확인: $MF ($(python3 -c "
import json,sys
d=json.load(open('$MF'))
print('노드 %d, 링크 %d' % (len(d.get('Node',[])), len(d.get('Link',[]))))
" 2>/dev/null || echo '파싱 실패'))"
      ;;
  esac
done

# --- 2. 스택 --------------------------------------------------------------
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u

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
  # 토픽 목록은 field_20260730/record.sh 에서 가져왔다(실주행으로 검증된 이름).
  # 추측한 이름을 쓰면 ros2 bag record 가 조용히 아무것도 안 남긴다 — 실제로
  # /gps/fix, /behavior_state, /odometry/filtered 는 존재하지 않는 이름이었다.
  # 여기에 경로 추종 진단용 토픽 셋을 더했다.
  CAM_TOPICS="/camera/camera/color/camera_info /camera/camera/color/image_raw \
    /camera/camera/depth/camera_info /camera/camera/depth/image_rect_raw"
  [ "$NOCAM" = 1 ] && { CAM_TOPICS=""; echo "[bringup] 카메라 토픽 제외 기록"; }
  # shellcheck disable=SC2086
  nohup ros2 bag record -o "$LOGDIR/bag" $CAM_TOPICS \
    /vectornav/imu /vectornav/pose /vectornav/magnetic \
    /ublox_gps_node/fix /ublox_gps_node/fix_velocity \
    /tf /tf_static /robot_description \
    /cmd_vel /current_speed /current_steer_angle /front/scan /rear/scan \
    /hunter/velocity /hunter_status /velodyne_points \
    /gps/fix_gated /odometry/global /map_anchor/mode /map_anchor/yaw_corr \
    /odom /odometry/fast_lio \
    /costmap /costmap_keepout /behavior_status \
    /planned_path_detailed /multiple_waypoints \
    /vehicle/mux_status /vehicle/estop_status /final_cmd /remote/cmd_mode \
    /navpvt /rxmrtcm \
    > "$LOGDIR/record.log" 2>&1 &
  echo "[bringup] record pid $!"
fi

sleep 20
echo "[bringup] 기동 후 노드 목록:"
ros2 node list 2>/dev/null | sort | sed 's/^/  /'
echo "[bringup] BRINGUP_DONE — 정지는 teardown.sh"
