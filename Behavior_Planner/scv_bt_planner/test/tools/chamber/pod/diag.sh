#!/bin/bash
# 런 진행 중 DDS/제어 경로 진단 (하네스와 동일 env, domain 96)
source ~/bt_chamber/env.sh 2>/dev/null
source ~/vehicle_ws/install/setup.bash 2>/dev/null
export ROS_DOMAIN_ID=${SCV_DOMAIN:-96}
echo "== time $(date +%s)"
echo "== lo multicast: $(ip -o link show lo | grep -o 'MULTICAST' || echo NONE)"; ip maddr show lo 2>/dev/null | grep -c 239 | sed 's/^/   239.* groups on lo: /'
echo "== ros2 daemon procs: $(ps -eo args | grep -c '_ros2_daemo[n]')"
echo "== node list (no daemon, 6s)"; timeout 12 ros2 node list --no-daemon --spin-time 6 2>&1 | sort | tr '\n' ' '; echo
echo "== hz groundtruth"; timeout 8 ros2 topic hz --no-daemon --window 20 /groundtruth/odom 2>&1 | grep -m1 average || echo "   no groundtruth"
echo "== hz cmd_vel"; timeout 8 ros2 topic hz --no-daemon --window 20 /cmd_vel 2>&1 | grep -m1 average || echo "   no cmd_vel"
echo "== hz rc_cmd"; timeout 6 ros2 topic hz --no-daemon --window 20 /rc_cmd 2>&1 | grep -m1 average || echo "   no rc_cmd"
echo "== hunter_status once"; timeout 6 ros2 topic echo --no-daemon --once /hunter_status 2>&1 | grep -E "control_mode" || echo "   none"
echo "== rc_watch sample"; timeout 8 python3 ~/scv_ws/tools/gazebo/rc_watch.py 0 0 0 2>&1 | tail -n 2
echo "== participants (udp ports 31400-32000 bound)"; ss -lun | awk '{print $5}' | grep -cE ':(31[4-9][0-9][0-9]|32[0-9]{3})$'
echo "== gz paused?"; timeout 5 gz stats -p 2>&1 | head -n 2
echo "== load: $(uptime | sed s/.*load/load/)"
ps -eo pcpu,rss,comm --sort=-pcpu | head -n 6
