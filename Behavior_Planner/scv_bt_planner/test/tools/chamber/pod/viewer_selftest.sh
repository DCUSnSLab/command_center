#!/bin/bash
# BT 뷰어 자가시험: 시뮬 없이 BT 노드만(도메인 98, groot2 1777) 띄우고 xvfb 에서 뷰어 화면을 캡처
source /opt/ros/humble/setup.bash; source ~/vehicle_ws/install/setup.bash
export ROS_DOMAIN_ID=98 ROS_LOCALHOST_ONLY=1 FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/bt_chamber/fastdds_udp_only.xml
OUT=~/bt_chamber/selftest; mkdir -p $OUT/logs_selftest
~/vehicle_ws/install/scv_bt_planner/lib/scv_bt_planner/bt_planner_node --ros-args -p mode:=shadow -p groot2_port:=1777 > $OUT/logs_selftest/bt.log 2>&1 &
BP=$!
sleep 4
xvfb-run -a -s "-screen 0 600x1020x24" bash -c "python3 ~/bt_chamber/bt_viewer.py --port 1777 --log-glob '$OUT/**/bt.log' --geometry 540x1000+0+0 & V=\$!; sleep 6; import -window root $OUT/shot.png; kill \$V" 2>&1 | tail -n 5
kill $BP; wait $BP 2>/dev/null
echo "--- bt.log"; grep -i "groot\|error" $OUT/logs_selftest/bt.log | head -n 5
ls -l $OUT/shot.png
