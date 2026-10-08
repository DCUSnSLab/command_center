#!/usr/bin/env python3
"""scv_sim_tools 의 Gazebo 챔버 하니스를 BT A/B 용으로 변형한 사본을 만든다.

원본(~/scv_ws/tools/gazebo/run_gazebo_experiment.sh, run_field_sequence.sh)은 건드리지 않는다.
변형 내용:
  - 워크스페이스: ~/scv_ws → ${SCV_WS} (기본 ~/SCV/vehicle_ws, scv_bt_planner 가 빌드된 곳)
  - 행동 플래너 선택: SCV_BP=simple|bt|shadow (simple 만 / BT active 만 / simple + BT shadow)
  - datum 발행: sequential planner 경로(SCV_GRAPH_MAP)일 때 /map_provider_node/utm 을 (0,0) 으로
    latched 발행 — 현행 BP 의 datum 게이트 통과 (SCV_PLANNER_ARGS 의 datum 0,0 과 일치)
  - kill 패턴에 bt_planner_node 추가, rc_phase 의 "join idx" 감지를 bt.log 에도 적용
사용: make_variants.py [원본 gazebo 디렉토리] [출력 디렉토리]
"""
import os
import re
import stat
import sys

SRC = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/scv_ws/tools/gazebo')
DST = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__)))


def must_replace(text, old, new, name):
    if old not in text:
        raise SystemExit(f'anchor not found in {name}: {old[:60]!r}')
    return text.replace(old, new)


# ---------------- run_gazebo_experiment_bt.sh ----------------
t = open(os.path.join(SRC, 'run_gazebo_experiment.sh')).read()
n = 'run_gazebo_experiment.sh'
t = must_replace(t, 'source ~/scv_ws/install/setup.bash', 'source ${SCV_WS:-$HOME/SCV/vehicle_ws}/install/setup.bash', n)
t = must_replace(t, 'B=~/scv_ws/install/pcd_ground_filter/share/pcd_ground_filter',
                 'B=${SCV_WS:-$HOME/SCV/vehicle_ws}/install/pcd_ground_filter/share/pcd_ground_filter', n)
t = must_replace(t, 'S=~/scv_ws/install/smppi/share/smppi', 'S=${SCV_WS:-$HOME/SCV/vehicle_ws}/install/smppi/share/smppi', n)
t = must_replace(t, 'simple_behavior_planner_node|', 'simple_behavior_planner_node|bt_planner_node|', n)
t = must_replace(t, 'ros2 run smppi corridor_keepout_node.py --ros-args', '[ -f "$S/../../lib/smppi/corridor_keepout_node.py" ] && ros2 run smppi corridor_keepout_node.py --ros-args', n)
t = must_replace(t, 'ros2 run smppi costmap_processor_node.py --ros-args', '[ -f "$S/../../lib/smppi/costmap_processor_node.py" ] && ros2 run smppi costmap_processor_node.py --ros-args', n)

bp_old = ('ros2 run simple_behavior_planner simple_behavior_planner_node.py \\\n'
          '  --ros-args -p use_sim_time:=true ${SCV_BP_ARGS:-} > "$LOGD/behavior.log" 2>&1 &\n')
bp_new = '''# datum (BP waypoint 게이트 통과용). 지도 원점 = 월드 원점 (SCV_PLANNER_ARGS datum 0,0 과 일치).
ros2 topic pub --qos-durability transient_local --qos-reliability reliable \\
  /map_provider_node/utm map_interfaces/msg/UtmLayer \\
  "{utm_zone: 52, zone_letter: 'S', northern: true, origin_easting: ${SCV_DATUM_E:-0.0}, origin_northing: ${SCV_DATUM_N:-0.0}}" \\
  > "$LOGD/datum.log" 2>&1 &
UTM_PID=$!   # latched pub 는 kill_stack 패턴 밖 — 종료 시 PID 로 정리(런마다 1개씩 누수 → 도메인 디스커버리 불능, 10/01 Pod 실측)
# SCV_BP: simple(현행) | bt(scv_bt_planner active, simple 미기동) | shadow(simple 명령 + BT 결정만)
BP_MODE=${SCV_BP:-simple}
if [ "$BP_MODE" != "bt" ]; then
  ros2 run simple_behavior_planner simple_behavior_planner_node.py \\
    --ros-args -p use_sim_time:=true ${SCV_BP_ARGS:-} > "$LOGD/behavior.log" 2>&1 &
fi
if [ "$BP_MODE" != "simple" ]; then
  BTMODE=shadow; [ "$BP_MODE" = "bt" ] && BTMODE=active
  ${SCV_WS:-$HOME/SCV/vehicle_ws}/install/scv_bt_planner/lib/scv_bt_planner/bt_planner_node \\
    --ros-args -p use_sim_time:=true -p mode:=$BTMODE ${SCV_BP_ARGS:-} ${SCV_BT_ARGS:-} > "$LOGD/bt.log" 2>&1 &
fi
'''
t = must_replace(t, bp_old, bp_new, n)
t = must_replace(t, 'kill_stack\necho "=== RESULT ==="', 'kill "${UTM_PID:-}" 2>/dev/null; kill_stack\necho "=== RESULT ==="', n)
# 실험 도메인 선택(SCV_DOMAIN). 누수 참여자가 쌓인 도메인은 새 노드 디스커버리가 깨진다 — 새 도메인으로 피할 수 있게.
t = must_replace(t, 'export ROS_DOMAIN_ID=96 ROS_LOCALHOST_ONLY=1', 'export ROS_DOMAIN_ID=${SCV_DOMAIN:-96} ROS_LOCALHOST_ONLY=1', n)
out = os.path.join(DST, 'run_gazebo_experiment_bt.sh')
open(out, 'w').write(t)
os.chmod(out, os.stat(out).st_mode | stat.S_IXUSR)

# ---------------- run_field_sequence_bt.sh ----------------
t = open(os.path.join(SRC, 'run_field_sequence.sh')).read()
n = 'run_field_sequence.sh'
t = must_replace(t, 'source ~/scv_ws/install/setup.bash', 'source ${SCV_WS:-$HOME/SCV/vehicle_ws}/install/setup.bash', n)
t = must_replace(t, 'if grep -q "join idx" "$LOGD/behavior.log" 2>/dev/null; then got=1; break; fi',
                 'if grep -q "join idx" "$LOGD/behavior.log" "$LOGD/bt.log" 2>/dev/null; then got=1; break; fi', n)
t = must_replace(t, '''$(grep -o 'join idx [0-9]* ([A-Za-z0-9]*)' "$LOGD/behavior.log" | tail -1)''',
                 '''$(grep -ho 'join idx [0-9]* ([A-Za-z0-9]*)' "$LOGD/behavior.log" "$LOGD/bt.log" 2>/dev/null | tail -1)''', n)
# main smppi 호환 (8/25 closedloop b973ea5 와 동일): 출력 토픽은 리맵이 아니라 파라미터로 바꾼다 —
# YAML 이 /final_cmd(차량 작업본) 이라 '-r /cmd_vel:=' 리맵은 아무것도 바꾸지 못한다.
t = must_replace(t, 'export SCV_MPPI_ARGS="-r /cmd_vel:=/autonomy_cmd ${SCV_MPPI_EXTRA:-}"',
                 'export SCV_MPPI_ARGS="-r /cmd_vel:=/autonomy_cmd -p topics.output.cmd_vel:=/autonomy_cmd ${SCV_MPPI_EXTRA:-}"', n)
t = must_replace(t, '"$G/run_gazebo_experiment.sh" "$RESULT" "$DUR" 0',
                 '"$(dirname "$0")/run_gazebo_experiment_bt.sh" "$RESULT" "$DUR" 0', n)
t = must_replace(t, 'export ROS_DOMAIN_ID=96 ROS_LOCALHOST_ONLY=1', 'export ROS_DOMAIN_ID=${SCV_DOMAIN:-96} ROS_LOCALHOST_ONLY=1', n)
# rc_watch 의 stderr 를 버리지 않는다 — 디스커버리 실패 때 "(x,y)=(,)" 만 남아 원인 추적이 막혔다(10/01).
t, k = re.subn(r'(python3 "\$G/rc_watch\.py" [^\n]*?) 2>/dev/null\)"', r'\1 2>>"$LOGD/rc_watch.err")"', t)
if k < 1:
    raise SystemExit(f'rc_watch stderr anchors: expected >=1, got {k}')
out = os.path.join(DST, 'run_field_sequence_bt.sh')
open(out, 'w').write(t)
os.chmod(out, os.stat(out).st_mode | stat.S_IXUSR)
print('generated:', DST)
