#!/bin/bash
# scv_bt_planner 만 재빌드 (소스는 ppub 에서 cat 스트림으로 갱신)
source /opt/ros/humble/setup.bash
cd ~/vehicle_ws && colcon build --packages-select scv_bt_planner --cmake-args -DBTCPP_EXAMPLES=OFF -DBTCPP_UNIT_TESTS=OFF -DBTCPP_BUILD_TOOLS=OFF 2>&1 | tail -n 3
echo "EXIT ${PIPESTATUS[0]}"
