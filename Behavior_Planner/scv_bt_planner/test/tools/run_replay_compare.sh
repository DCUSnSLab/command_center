#!/bin/bash
# bag 재생 shadow 비교: simple_behavior_planner(active) vs scv_bt_planner(shadow), 같은 입력.
#
#   run_replay_compare.sh <bag_dir> <graph.json> <goal_node> <out.json> [rate=1.0] [domain=95]
#
# bag 은 /odometry/global /odom /tf /tf_static /hunter_status /costmap 를 담고 있어야 한다
# (8/11 bag 137 을 BagArchive topics 필터로 받은 것). /goal_status 는 녹화에 없으므로
# goal_status_synth.py 가 두 플래너에 각각 합성해 준다.
BAG=${1:?bag dir}; MAP=${2:?graph.json}; GOAL=${3:?goal node}; OUT=${4:?out.json}
RATE=${5:-1.0}; DOM=${6:-95}
WS=${SCV_WS:-$HOME/SCV/vehicle_ws}
T=$WS/src/command_center/Behavior_Planner/scv_bt_planner/test/tools
LOGD=$(dirname "$OUT")/logs_$(basename "$OUT" .json); rm -rf "$LOGD"; mkdir -p "$LOGD"
DUR_S=$(python3 -c "import yaml,sys; m=yaml.safe_load(open('$BAG/metadata.yaml')); print(int(m['rosbag2_bagfile_information']['duration']['nanoseconds']/1e9))")
DUR=$(python3 -c "print(int($DUR_S / $RATE) + 25)")

# set -u 는 ROS setup.bash 뒤에 — setup 스크립트가 미정의 변수를 참조한다(AMENT_TRACE_SETUP_FILES)
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
set -u
export ROS_DOMAIN_ID=$DOM ROS_LOCALHOST_ONLY=1 PYTHONUNBUFFERED=1
export RCUTILS_LOGGING_BUFFERED_STREAM=0 RCUTILS_LOGGING_USE_STDOUT=1

cleanup() {
  pkill -x bt_planner_node 2>/dev/null
  # 주의: 패턴이 이 스크립트 자신의 명령줄(run_replay_compare.sh)에 매칭되면 자살한다 — .py 로 한정
  for pat in "[s]imple_behavior_planner_node" "[s]equential_planner_node" "[g]oal_status_synth\.py" "[r]eplay_compare\.py" "[r]os2 bag play" "[r]os2 topic pub"; do
    pkill -f "$pat" 2>/dev/null
  done
}
cleanup; sleep 1
echo "bag=$BAG dur=${DUR_S}s rate=$RATE -> recorder $DUR s; map=$MAP goal=$GOAL"

# datum (latched, 1 Hz 반복 발행)
ros2 topic pub --qos-durability transient_local --qos-reliability reliable \
  /map_provider_node/utm map_interfaces/msg/UtmLayer \
  "{utm_zone: 52, zone_letter: 'S', northern: true, origin_easting: 482232.7, origin_northing: 3974384.47}" \
  > "$LOGD/datum.log" 2>&1 &

# 경로: 최근접 노드에서 goal 까지 (8/11 운용과 동일: goal_node 지정, 시작 미지정)
ros2 run sequential_global_planner sequential_planner_node.py --ros-args -p use_sim_time:=true \
  -p map_file:="$MAP" -p auto_start:=true -p explicit_endpoints:=false \
  -p ${GOAL_PARAM:-initial_goal_node}:="$GOAL" -p map_origin_source:=datum \
  -p datum_utm_easting:=482232.7 -p datum_utm_northing:=3974384.47 > "$LOGD/planner.log" 2>&1 &

# simple BP (active)
ros2 run simple_behavior_planner simple_behavior_planner_node.py --ros-args -p use_sim_time:=true \
  -p current_position_topic:=/odometry/global ${SCV_BP_ARGS:-} > "$LOGD/simple.log" 2>&1 &
# BT (shadow): goal_status 는 자기 몫의 합성기에서
"$WS/install/scv_bt_planner/lib/scv_bt_planner/bt_planner_node" --ros-args -p use_sim_time:=true \
  -p mode:=shadow -p goal_status_topic:=/bt/goal_status -p map_file_path:="$MAP" \
  -p current_position_topic:=/odometry/global ${SCV_BT_ARGS:-} > "$LOGD/bt.log" 2>&1 &
# goal_status 합성기 ×2
python3 "$T/goal_status_synth.py" --ros-args -p use_sim_time:=true -p prefix:=none -p map_file:="$MAP" > "$LOGD/synth_simple.log" 2>&1 &
python3 "$T/goal_status_synth.py" --ros-args -p use_sim_time:=true -p prefix:=/bt -p map_file:="$MAP" > "$LOGD/synth_bt.log" 2>&1 &
# 기록기
python3 "$T/replay_compare.py" "$OUT" "$DUR" > "$LOGD/compare.log" 2>&1 &
REC=$!
sleep 4

# 재생 (녹화된 /behavior_status 가 있으면 /rec/ 로 리맵)
ros2 bag play "$BAG" --clock 100 -r "$RATE" \
  --topics /odometry/global /odom /tf /tf_static /hunter_status /costmap /behavior_status \
  --remap /behavior_status:=/rec/behavior_status > "$LOGD/play.log" 2>&1
echo "bag play finished; waiting recorder"
wait $REC 2>/dev/null
cleanup
echo "=== SUMMARY ($OUT)"
python3 -c "import json; s=json.load(open('$OUT'))['summary']; [print(k, ':', v) for k, v in s.items() if k != 'escalation']; print('escalation:', s['escalation'])"
