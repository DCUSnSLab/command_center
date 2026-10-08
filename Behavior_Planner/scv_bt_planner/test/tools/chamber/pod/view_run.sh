#!/bin/bash
# 뷰어 루프 재시작 + 런 1개: view_run.sh <simple|bt|shadow> <이름> [dur=420] [설계 JSON]
# (exec 명령줄에 뷰어·노드 이름을 넣지 않으려고 스크립트로 감싼다 — stop 의 pkill 자기매칭 방지)
# 설계를 주면 그 챔버로 돌고(시나리오 포함), 뷰어 지도도 그 설계를 그린다(~/bt_chamber/current_design).
DESIGN=${4:-$(bash -c 'source ~/bt_chamber/env.sh >/dev/null 2>&1; echo "$SCV_DESIGN"')}
echo "$DESIGN" > ~/bt_chamber/current_design
# 시나리오 챔버(설계에 scenario 가 있으면)에서는 위험 지대 경유점 재배치를 켠다(BT 기본은 비활성)
grep -q '"scenario"' "$DESIGN" 2>/dev/null && export SCV_BT_EXTRA="-p hazard.enable:=true"
~/bt_chamber/watch.sh stop >/dev/null; sleep 2; ~/bt_chamber/watch.sh start
cd ~ && SCV_DESIGN="$DESIGN" nohup ~/bt_chamber/run_one.sh "$1" ~/scv_sim/bt_ab/$2.json "${3:-420}" > ~/scv_sim/bt_ab/$2.stdout 2>&1 < /dev/null &
echo "started $2 $(date +%s) design=$DESIGN"
