#!/bin/bash
# scv_system.launch.py 를 simple/bt/shadow 로 각각 짧게 띄워 노드 집합·BT 파라미터·토픽을 확인 (지도·측위 없이 가용성만)
# 인자: <install setup.bash> [domain]
source /opt/ros/humble/setup.bash; source "$1"; export ROS_DOMAIN_ID=${2:-99} ROS_LOCALHOST_ONLY=1
LOG=${LOGDIR:-/tmp/scvsys_modes}; mkdir -p $LOG
for m in simple bt shadow; do
  echo "=== behavior_planner:=$m"
  timeout 25 ros2 launch command_center_launch scv_system.launch.py use_sim_time:=false behavior_planner:=$m bt_log_file:=$LOG/bt_$m.btlog route_tool:=false > $LOG/launch_$m.log 2>&1 &
  LP=$!
  sleep 14
  ros2 node list 2>/dev/null | grep -E "behavior|bt_planner|path_planner|costmap|mppi|waypoint" | sort | tr '\n' ' '; echo
  if ros2 node list 2>/dev/null | grep -q bt_planner; then
    echo "  bt mode=$(ros2 param get /bt_planner mode 2>/dev/null | cut -d: -f2) waypoint_mode=$(ros2 param get /bt_planner waypoint_mode 2>/dev/null | cut -d: -f2) pos_topic=$(ros2 param get /bt_planner current_position_topic 2>/dev/null | cut -d: -f2) log=$(ros2 param get /bt_planner bt_log_file 2>/dev/null | cut -d: -f2)"
    echo "  bt pubs: $(ros2 node info /bt_planner 2>/dev/null | sed -n '/Publishers/,/Service Servers/p' | grep -o -E '/[a-z_/]*(waypoints|params|behavior|pause|status|assist|hint)[a-z_/]*' | sort -u | tr '\n' ' ')"
  fi
  wait $LP 2>/dev/null
  grep -i -E "error|exception|Traceback|not found" $LOG/launch_$m.log | grep -v "DEBUG" | head -3
  ls -la $LOG/bt_$m.btlog 2>/dev/null | awk '{print "  btlog:", $5, "bytes"}'
  sleep 3
done
echo "=== package absent, default simple (expect OK):"
export AMENT_PREFIX_PATH=$(echo "$AMENT_PREFIX_PATH" | tr ':' '\n' | grep -v scv_bt_planner | paste -sd:)
timeout 12 ros2 launch command_center_launch scv_system.launch.py use_sim_time:=false route_tool:=false > $LOG/launch_absent.log 2>&1; echo "  exit=$? (124=정상 가동 후 timeout)"; grep -c -E "scv_bt_planner.*not found|PackageNotFound" $LOG/launch_absent.log
echo "=== package absent, bt (expect PackageNotFound):"
timeout 12 ros2 launch command_center_launch scv_system.launch.py use_sim_time:=false behavior_planner:=bt route_tool:=false > $LOG/launch_absent_bt.log 2>&1; echo "  exit=$?"; grep -m1 -o -E "package 'scv_bt_planner' not found" $LOG/launch_absent_bt.log
echo MODES_DONE
