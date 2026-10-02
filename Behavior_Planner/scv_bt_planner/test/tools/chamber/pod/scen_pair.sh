#!/bin/bash
# 시나리오 챔버에서 bt → simple 순서로 1런씩 (녹화 포함). scen_pair.sh <접두> [dur=480]
D=~/scv_sim/chamber/scen_hazard.design.json
for bp in bt simple; do
  ~/bt_chamber/view_run.sh $bp ${1}_$bp ${2:-480} $D
  until [ -f ~/scv_sim/bt_ab/${1}_$bp.json ] && ! ss -ltn | grep -q ":11355 "; do sleep 10; done
  sleep 8
done
echo PAIR_DONE
