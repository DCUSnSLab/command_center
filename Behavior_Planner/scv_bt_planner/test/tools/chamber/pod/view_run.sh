#!/bin/bash
# 뷰어 루프 재시작 + 런 1개: view_run.sh <simple|bt|shadow> <이름> [dur=420]
# (exec 명령줄에 뷰어·노드 이름을 넣지 않으려고 스크립트로 감싼다 — stop 의 pkill 자기매칭 방지)
~/bt_chamber/watch.sh stop >/dev/null; sleep 2; ~/bt_chamber/watch.sh start
cd ~ && nohup ~/bt_chamber/run_one.sh "$1" ~/scv_sim/bt_ab/$2.json "${3:-420}" > ~/scv_sim/bt_ab/$2.stdout 2>&1 < /dev/null &
echo "started $2 $(date +%s)"
