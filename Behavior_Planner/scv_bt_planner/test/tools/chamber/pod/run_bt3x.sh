#!/bin/bash
# bt 3런 재시행(run3): RC 정지 결정화(rc_watch --drive) + shadow resync 반영 뒤
mkdir -p ~/scv_sim/bt_ab/run3
for i in 1 2 3; do echo "===== bt$i 시작 $(date +%T)"; ~/bt_chamber/run_one.sh bt ~/scv_sim/bt_ab/run3/bt$i.json 420 > ~/scv_sim/bt_ab/run3/bt$i.stdout 2>&1; python3 -c "import json; d=json.load(open(\"$HOME/scv_sim/bt_ab/run3/bt$i.json\")); print(\"      완료\", d[\"verdict\"], d[\"distance\"], d[\"min_goal_dist\"])"; done
echo "===== shadow 1런"; ~/bt_chamber/run_one.sh shadow ~/scv_sim/bt_ab/run3/shadow1.json 420 > ~/scv_sim/bt_ab/run3/shadow1.stdout 2>&1; echo DONE
