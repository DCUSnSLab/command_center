#!/bin/bash
# Gazebo 챔버 A/B: simple_behavior_planner vs scv_bt_planner(active), 같은 설계 챔버·같은 필드 절차.
#
#   run_chamber_ab.sh <out_dir> [n=3] [duration=420] [design=chamber_20260803.design.json]
#
# 각 런은 run_field_sequence_bt.sh (필드 절차: 목표 전송 → RC 전진 → 3→1 자율 전환 → 자율 주행)를
# SCV_BP=simple / SCV_BP=bt 로 번갈아 돌린다. CHAMBER.md 규칙대로 verdict 만 보지 않고
# 연속 지표(min_goal_dist, distance, duration)를 함께 본다. 추가로 SCV_BP=shadow 1회를 돌려
# 같은 주행에서 두 플래너의 결정을 나란히 기록한다.
set -u
OUT=${1:?out dir}; N=${2:-3}; DUR=${3:-420}
DESIGN=${4:-$HOME/SCV/scv_sim/chamber/chamber_20260803.design.json}
HERE=$(cd "$(dirname "$0")" && pwd)
export SCV_WS=${SCV_WS:-$HOME/SCV/vehicle_ws}
export SCV_DESIGN=$DESIGN
export SCV_ZONES=${SCV_ZONES:-$HOME/scv_ws/tools/gazebo/zones/rtk_clean.yaml}
mkdir -p "$OUT"
for i in $(seq 1 "$N"); do
  for arm in simple bt; do
    f="$OUT/${arm}${i}.json"
    [ -f "$f" ] && { echo "== ${arm}${i} 이미 있음 — 건너뜀"; continue; }
    echo "===== ${arm}${i} 시작 $(date +%T)"
    # DDS 기동 플레이크(rc_phase 가 RC 전진을 못 채움 → 거리 ~0)는 무효 런 — 최대 2회 재시도
    for attempt in 1 2 3; do
      SCV_BP=$arm bash "$HERE/run_field_sequence_bt.sh" "$f" "$DUR" true > "$OUT/${arm}${i}.stdout" 2>&1
      if grep -q "목표 거리" "$OUT/logs_${arm}${i}/rc_phase.log" 2>/dev/null; then break; fi
      echo "      !! RC 전진 미달성(플레이크 의심) — 재시도 $attempt"; mv "$f" "$f.flake$attempt" 2>/dev/null
    done
    echo "      완료 $(date +%T): $(python3 -c "import json;d=json.load(open('$f'));print(d['verdict'],'dist',d['distance'],'min_goal',d['min_goal_dist'],'dur',d['duration'])" 2>/dev/null || echo '결과 없음')"
  done
done
if [ ! -f "$OUT/shadow1.json" ]; then
  echo "===== shadow1 (simple 명령 + BT 결정 기록) 시작 $(date +%T)"
  for attempt in 1 2 3; do
    SCV_BP=shadow bash "$HERE/run_field_sequence_bt.sh" "$OUT/shadow1.json" "$DUR" true > "$OUT/shadow1.stdout" 2>&1
    grep -q "목표 거리" "$OUT/logs_shadow1/rc_phase.log" 2>/dev/null && break
    echo "      !! RC 전진 미달성(플레이크 의심) — 재시도 $attempt"; mv "$OUT/shadow1.json" "$OUT/shadow1.json.flake$attempt" 2>/dev/null
  done
  echo "      완료 $(date +%T)"
fi
python3 "$HERE/chamber_ab_report.py" "$OUT"
