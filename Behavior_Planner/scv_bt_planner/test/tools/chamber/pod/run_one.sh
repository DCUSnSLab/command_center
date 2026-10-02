#!/bin/bash
# 단일 런: run_one.sh <simple|bt|shadow> <result.json> [duration=420]
source $HOME/bt_chamber/env.sh
C=$SCV_WS/src/command_center/Behavior_Planner/scv_bt_planner/test/tools/chamber
SCV_BP=${1:?simple|bt|shadow} exec bash "$C/run_field_sequence_bt.sh" "${2:?result.json}" "${3:-420}" true
