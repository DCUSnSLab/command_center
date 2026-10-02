#!/bin/bash
# 런 상태 점검 — 하니스 kill_stack 의 패턴 자기매칭을 피하려고 스크립트 파일로 실행한다.
R=${1:?result.json}; L=$(dirname "$R")/logs_$(basename "$R" .json)
echo "result: $([ -f "$R" ] && echo YES || echo no)   gz: $(pgrep -x gzserver | wc -l)  judge: $(pgrep -fc gz_judge)"
echo "gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
p=$(pgrep -x gzserver | head -1); [ -n "$p" ] && echo "gzserver nvidia fds: $(ls -l /proc/$p/fd 2>/dev/null | grep -c nvidia)"
echo "--- stdout"; tail -8 "$(dirname "$R")/$(basename "$R" .json).stdout" 2>/dev/null
echo "--- rc_phase"; tail -4 "$L/rc_phase.log" 2>/dev/null
echo "--- behavior"; grep -h "join idx\|Advanced\|BLOCKED\|completed" "$L/behavior.log" "$L/bt.log" 2>/dev/null | tail -6
echo "--- judge"; tail -3 "$L/judge.log" 2>/dev/null
[ -f "$R" ] && python3 ~/bt_chamber/summ.py "$R"
