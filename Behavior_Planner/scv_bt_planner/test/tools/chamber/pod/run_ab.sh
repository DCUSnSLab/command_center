#!/bin/bash
# 사용: run_ab.sh <out_dir> [n=3] [duration=420]
# simple / bt 각 n 런 + shadow 1 런. 결과 <out_dir>/{simple,bt}N.json, shadow1.json, 로그 logs_*
source $HOME/bt_chamber/env.sh
C=$SCV_WS/src/command_center/Behavior_Planner/scv_bt_planner/test/tools/chamber
exec bash "$C/run_chamber_ab.sh" "${1:?out_dir}" "${2:-3}" "${3:-420}" "$SCV_DESIGN"
