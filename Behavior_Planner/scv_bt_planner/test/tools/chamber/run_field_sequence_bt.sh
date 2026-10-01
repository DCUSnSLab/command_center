#!/bin/bash
# 필드 운용 절차를 그대로 밟는 챔버 실험 — 2026-08-06 오후 시나리오 재현.
#
# 실차 절차(FIELD_PROCEDURE.md §1-2)와 같은 순서로 간다:
#   기동 -> 목표 전송(경로 수신, 합류 노드 선택) -> RC 수동 전진 ~10 m
#   -> control_mode 3->1 자율 전환 -> 목표까지 자율 주행
#
# 이 순서가 핵심이다. 종전 챔버 스크립트는 RC 주행을 목표 전송 **전에**
# 했는데, 그러면 합류 노드가 이동 후에 정해져 8/6 오전 결함(기동 시점에
# 고른 합류 노드가 RC 이동으로 낡아 뒤쪽을 가리킴)이 아예 재현되지 않는다.
# 실차는 목표를 먼저 받고(기동 인자 goal_node) RC 로 움직인다.
#
# 또 하나: 재선택은 /hunter_status 의 control_mode 3->1 전이로 발동한다.
# hunter_msgs 가 없으면 behavior planner 가 그 게이트를 통째로 생략하므로
# (ImportError 분기) 챔버에서 수정이 동작하는지 확인할 수 없었다. 여기서는
# hunter_msgs 를 빌드해 두고 RC/자율 상태를 실제로 발행한다.
#
# 기하 — RC 가 경로를 **벗어나는** 방향이어야 8/6 결함이 재현된다.
#   경로 P0(0,1) -> P1(0,3) -> P2(0,5) -> P3(3,5). 로봇은 P0 에서 출발해
#   동쪽으로 6 m RC 주행 — 경로는 북쪽으로 가므로 멀어지기만 한다.
#
#   차선을 y=+1 로 올린 이유: y=0 으로 달리면 x 가 커질수록 연석에 붙는다
#   (보도 하단이 x=0 에서 -2.0, x>=2 에서 -1.0, x=8 에서 -0.5). 실측으로
#   로봇 남쪽 끝이 연석에서 0.17 m 까지 붙어 선회 여유가 사라졌고, 연석
#   검출이 세운 벽(매 주기 wall_pts 2,700)에 갇혀 MPPI 가 v~0 을 냈다.
#   y=+1 이면 연석 여유가 최소 2.0 m 다.
#
#   지도 원점은 SCV_PLANNER_ARGS 로 월드 원점에 고정한다 — 기본 first_node
#   는 node[0] 이 원점일 때만 맞고, 아니면 지도 프레임이 통째로 밀린다.
#
#   RC 가 경로를 **따라가면** MPPI 가 RC 중에도 /goal_status 를 발행해
#   노드가 자동 소진되고(실측: B 에서 RC 구간에 P1~P4 소진) 자율 전환
#   시점에는 합류가 이미 전진해 있어 낡은 상태가 만들어지지 않는다.
#
#   전환지점 (2,0) 기준 방위:
#     낡은 합류 P0(-5,3)  +156.8도  7.6 m   <- 재선택 OFF
#     재선택 합류 P4(3,3)  +71.6도  3.2 m   <- 재선택 ON
#     목표      P6(3,7)   +81.9도  7.1 m
#   85도 어긋난다 = 8/6 필드(-72.8도 vs +139.4도)와 같은 성격.
#
# 사용: run_field_sequence.sh <결과.json> [지속s] [realign=true|false]
set -u
G=~/scv_ws/tools/gazebo
RESULT=${1:?결과 json 경로}
DUR=${2:-300}
REALIGN=${3:-true}
LOGD=$(dirname "$RESULT")/logs_$(basename "$RESULT" .json)

# --- 실차와 맞춘 설정 -------------------------------------------------------
# SCV_URDF 를 미리 주면 존중한다 (GPU 노드에서는 scv_sim_robot_chamber_gpu.urdf + SCV_VGL=1).
export SCV_URDF=${SCV_URDF:-$G/scv_sim_robot_chamber.urdf} SCV_XVFB=1
export SCV_ZONES=$G/zones/rtk_clean.yaml
# 종전 chamber_graph(y=0, x 0~21)는 **물리적으로 통행 불가**였다. x=8.75 에서
# 보도 폭이 y>=-0.5 로 좁아지는데 obs_255 가 y 0~0.5 를 차지해 남는 폭이
# 0.5 m — 로봇(축간 0.66 m)이 못 지나간다. 실측으로 x=8.01 에서 완전히 멎었고
# (궤적 5,718 표본의 끝이 같은 좌표) 자율 구간에서도 회복하지 못했다.
# 이전 챔버 A/B 가 목표를 G09(x=9)로 잡았던 것도 같은 제약으로 보인다 —
# 즉 9 m 이상 주행은 이 챔버에서 검증된 적이 없다.
#
# 시험 구간이 x -4 ~ +7 (11 m)로 좁은 데는 두 가지 상한이 겹친다:
#   위쪽 - x=8.25 의 통행 불가 병목(위 참조)
#   아래쪽 - 판정기의 추락 기준이 **절대** z < -0.10 인데 이 월드는 1.5 %
#            경사 램프(보도 상면 -0.366@x=-24 ~ +0.363@x=+24)라 x <~ -6 은
#            연석과 무관하게 자동 추락 판정이 난다(실측: x=-23 스폰이 1.5 s
#            만에 FAIL_FELL_OFF_CURB, z=-0.255).
# 판정기는 여러 실험이 공유하는 계측기라 고치지 않는다 — 고치면 과거 결과와
# 비교가 깨진다. 대신 RC 전진을 7 m 로 잡는다(절차서의 10 m 보다 짧다).
export SCV_GRAPH_MAP=$G/chamber_graph_lane.json
export SCV_GOAL_NODE=P3
export SCV_GOAL_X=3 SCV_GOAL_Y=5 SCV_REACH_D=1.5
# datum 위 스폰은 미고정 상태의 위험을 은폐한다 (CHAMBER.md) — 실차는
# 절대 node[0] 위에서 출발하지 않는다.
export SCV_SPAWN_X=0.0 SCV_SPAWN_Y=1.0

# --- SCV_DESIGN: design_chamber.py 가 만든 챔버를 그대로 태운다 --------------
# 위 값들은 차선 챔버(chamber_graph_lane)에 손으로 맞춘 것이라 다른 월드에
# 그대로 쓸 수 없다. 설계 파일이 주어지면 월드·경로·출발점·RC 거리를 전부
# 거기서 읽는다 — 사람이 좌표를 옮겨 적다 틀리는 경로를 없앤다.
if [ -n "${SCV_DESIGN:-}" ]; then
  eval "$(python3 - "$SCV_DESIGN" <<'PYEOF'
import json, math, sys
d = json.load(open(sys.argv[1]))
r, s = d['route'], d['spawn']
print(f"export SCV_WORLD={d['world']}")
print(f"export SCV_GRAPH_MAP={d['graph']}")
print(f"export SCV_GOAL_NODE={r['goal_node']}")
print(f"export SCV_GOAL_X={r['goal'][0]} SCV_GOAL_Y={r['goal'][1]} SCV_REACH_D=1.5")
# SCV_SPAWN_YAW 는 환경으로 덮어쓸 수 있다 — 역방향 스폰(재정렬 기동 시나리오)용
print(f"export SCV_SPAWN_YAW=${{SCV_SPAWN_YAW:-{s['yaw_to_route']}}}")
print(f"export SCV_SPAWN_X={s['x']} SCV_SPAWN_Y={s['y']}")
print(f"export RC_DIST=${{RC_DIST:-{s['rc_distance_m']}}}")
# 존(SCV_ZONES)은 일부러 건드리지 않는다 — 설계 파일이 GNSS 시나리오까지
# 정하면 기준선 런과 열화 런의 구분이 사라진다. 호출자가 고른다.
PYEOF
)"
  echo "설계 챔버: $SCV_DESIGN"
  echo "  world=$SCV_WORLD graph=$SCV_GRAPH_MAP goal=$SCV_GOAL_NODE"
  echo "  spawn=($SCV_SPAWN_X,$SCV_SPAWN_Y) yaw=$SCV_SPAWN_YAW RC=$RC_DIST m"
fi
# 필드와 동일: 행동계획은 map 프레임 포즈를 본다
# 3번째 인자로 join_check_approach 를 켜고 끈다(재선택은 항상 ON).
# 2026-08-11 챔버에서 재선택이 경로를 통째로 건너뛰어 지름길을 시도하다
# 막힌 것이 드러났고, 그 대책이 접근 직선 통행성 검사다.
export SCV_BP_ARGS="-p current_position_topic:=/odometry/global -p realign_on_engage:=true -p join_check_approach:=$REALIGN"
export SCV_CORRIDOR_ARGS="-p corridor_half_width:=25.0"
# mppi 출력을 중재 노드 뒤로 돌린다 — base_mux_sim.py 참조
# SCV_MPPI_EXTRA: 제어기 파라미터 추가 주입 (예: -p v_max:=1.5) — 튜닝 스윕용
export SCV_MPPI_ARGS="-r /cmd_vel:=/autonomy_cmd -p topics.output.cmd_vel:=/autonomy_cmd ${SCV_MPPI_EXTRA:-}"
# 지도 원점을 월드 원점에 고정한다 — 기본 first_node 는 node[0] 이 원점이어야만
# 맞는다(run_gazebo_experiment.sh 주석 참조). 이걸로 경로를 아무 데나 그릴 수 있다.
export SCV_PLANNER_ARGS="-p map_origin_source:=datum -p datum_utm_easting:=0.0 -p datum_utm_northing:=0.0"
# yaw 시드. 시뮬 IMU 에는 오프셋이 없으므로 실차 기본값(1.8012=+103.2도)을
# 그대로 넣으면 그건 곧 **103도 결함 주입**이다. 처음에 그렇게 돌렸다가
# A/B 가 오염됐다: autocal 이 103도를 걷어내는 순간 map->odom 이 크게 돌아
# 포즈가 점프하고, 그 점프가 **순간이동 재정렬**(realign_on_engage 와 무관한
# 경로)을 발동시켜 재선택이 꺼진 B 에서도 합류 노드가 갱신됐다.
#
# 8/6 필드는 자기 앵커가 복원된 뒤라 시드가 이미 맞았고 autocal 보정은
# -5~+1도로 작았다 — 점프가 없었고, 그래서 낡은 합류 노드가 그대로 남아
# 실패했다. 그 상태를 재현하려면 시드 결함이 없어야 한다.
YAW_SEED=${YAW_SEED:-0.0}
# SCV_LOC_EXTRA: 이 줄이 SCV_LOC_ARGS 를 통째로 덮어쓰므로(8/13 실측: 외부에서
# 준 SCV_LOC_ARGS 가 조용히 사라져 결함주입 A/B 가 무결함 런이 됐다), 추가
# 위치추정 인자는 이 변수로 넘긴다. 예: SCV_LOC_EXTRA="scan_yaw_init:=1"
export SCV_LOC_ARGS="anchor_yaw_offset:=$YAW_SEED anchor_yaw_autocal:=true ${SCV_LOC_EXTRA:-}"
# 하니스의 고정 지연 목표 재발행을 **끈다**(도달 불가능한 지연값).
# 재발행은 플래너가 매번 현재 최근접 노드에서 경로를 새로 만들게 하고,
# 그 경로 수신이 곧 합류 노드 재선택이라 realign_on_engage 와 무관하게
# 합류가 갱신된다 — 실측으로 A/B 가 통째로 무의미해졌다(B 도 P0->P4).
# 실차는 기동 인자 goal_node 로 **한 번만** 보낸다. 아래 rc_phase 에서
# 받아들여질 때까지만 재시도하고 즉시 멈추는 방식으로 대체한다.
export SCV_GOAL_DELAYS="99999"

RC_START=${RC_START:-15}     # 경로 확정 후 안정화 대기
# 절차서가 요구하는 것은 "전진 10 m" 이지 "몇 초" 가 아니다. 고정 시간으로
# 두면 시뮬 속도가 바뀔 때마다 이동거리가 달라져 실험이 흔들린다 — 실제로
# 그렇게 됐다. 거리로 끊고 시간은 안전망으로만 둔다.
# (이 챔버 로봇의 실측 속도는 1.0 m/s 를 지령해도 0.28 m/s 다. 10 m 는
#  시뮬 ~36 s. RTF 는 ~1.0 이라 벽시계도 비슷하다.)
RC_DIST=${RC_DIST:-6.0}
RC_SECS=${RC_SECS:-120}
mkdir -p "$LOGD"

# --- RC 수동 주행 + 제어모드 발행 (실차 흉내) --------------------------------
# groundtruth x 한 표본. topic echo 는 FastRTPS 공유메모리 경고를 섞어
# 뱉을 때가 있어(실측) 숫자만 걸러낸다.
gt_x() {
  # timeout 필수: --once 는 표본이 안 오면 무한 대기한다. 실제로 A1 런이
  # 여기서 멎어 RC 단계가 통째로 실행되지 않았고, mux 가 RC 모드에 갇혀
  # 베이스가 자율 명령을 영영 받지 못했다(주행 0.01 m, PASS_BLOCKED).
  timeout 10 ros2 topic echo /groundtruth/odom nav_msgs/msg/Odometry \
    --field pose.pose.position.x --once 2>/dev/null \
    | grep -oE '^-?[0-9]+\.?[0-9]*' | head -1
}

gt_y() {
  timeout 10 ros2 topic echo /groundtruth/odom nav_msgs/msg/Odometry \
    --field pose.pose.position.y --once 2>/dev/null \
    | grep -oE '^-?[0-9]+\.?[0-9]*' | head -1
}

rc_phase() {
  set +u
  source /opt/ros/humble/setup.bash
  source ${SCV_WS:-$HOME/SCV/vehicle_ws}/install/setup.bash
  export ROS_DOMAIN_ID=${SCV_DOMAIN:-96} ROS_LOCALHOST_ONLY=1
  local log="$LOGD/rc_phase.log"

  until ros2 topic list 2>/dev/null | grep -q groundtruth; do sleep 3; done
  # 대기(0) 상태를 먼저 알린다 — 실차도 기동 직후는 대기다
  python3 "$G/hunter_status_pub.py" 0 --ros-args -p use_sim_time:=true > "$LOGD/hs0.log" 2>&1 &
  local HS=$!

  # 베이스 명령 중재 — control_mode 에 따라 자율/RC 를 통과시킨다.
  python3 "$G/base_mux_sim.py" --ros-args -p use_sim_time:=true \
    > "$LOGD/base_mux.log" 2>&1 &
  echo "[rc] base_mux 기동 (자율 전환 전까지 베이스 미청취)" > "$log"

  # 목표 전송: 경로가 실릴 때까지만 재시도하고 실리면 즉시 멈춘다.
  # 첫 발행은 위치추정 수렴 전이라 플래너가 의도적으로 거절한다
  # (부트스트랩 포즈로 라우팅하면 조용히 엉뚱한 출발 노드가 잡힌다).
  local got=0
  for _ in $(seq 1 15); do
    if grep -q "join idx" "$LOGD/behavior.log" "$LOGD/bt.log" 2>/dev/null; then got=1; break; fi
    # 따옴표 중첩을 피해 YAML 평문으로 넘긴다 — 'P6' 를 따옴표로 감싸려다
    # {data: '"P6"'} 가 되어 노드 이름이 어긋나는 함정이 있다(실측).
    ros2 topic pub --once /goal_node_id std_msgs/String \
      "{data: $SCV_GOAL_NODE}" > /dev/null 2>&1
    sleep 8
  done
  if [ "$got" = 1 ]; then
    echo "[rc] 경로 수신·합류 노드 확정: $(grep -ho 'join idx [0-9]* ([A-Za-z0-9]*)' "$LOGD/behavior.log" "$LOGD/bt.log" 2>/dev/null | tail -1)" >> "$log"
  else
    echo "[rc] !! 경로 미수신 — 이 런은 무효" >> "$log"
  fi
  sleep "$RC_START"

  echo "[rc] === RC 수동 전환 (control_mode 3) $(date +%s) ===" >> "$log"
  kill $HS 2>/dev/null
  python3 "$G/hunter_status_pub.py" 3 --ros-args -p use_sim_time:=true > "$LOGD/hs3.log" 2>&1 &
  HS=$!

  # 실차에서 RC 는 mux 를 거치지 않고 CAN 레벨에서 베이스를 잡는다 — 그
  # 동안 자율 파이프라인의 /cmd_vel 은 베이스가 **무시한다**. 시뮬은 둘 다
  # 같은 토픽으로 들어가 경쟁하므로, 첫 시행에서 30 s 전진이 0.88 m 에
  # 그쳤다(mppi 의 0-명령이 절반을 먹었다). RC 창 동안 mppi 를 정지시켜
  # "베이스가 자율 명령을 듣지 않는다"를 그대로 모사한다.
  sleep 1

  read -r x0 y0 <<< "$(timeout 15 python3 "$G/rc_watch.py" 0 0 0 2>>"$LOGD/rc_watch.err")"
  echo "[rc] 전진 시작 (x,y)=($x0,$y0)" >> "$log"
  # 실차 RC 전진 속도대(최고 1.3 m/s)에서 1.0 m/s 를 지령한다 — 지령도 rc_watch 가 낸다(--drive).
  # 10/01: 별도 `topic pub` + 도달 후 kill 은 kill 지연 동안 1.0/0 이 섞여 정지 거리가 1.7~3.6 m 로 흔들렸다.
  # 10 m 를 채우면 바로 멈춘다 — 운전자가 절차상 거리를 보고 손을 떼는 것과 같다
  # 이동량은 **직선거리**로 잰다. x 증분만 보면 기수방위가 +x 가 아닌 챔버
  # (설계된 경로는 아무 방향으로나 뻗는다)에서 영영 조건을 못 채우고 RC 창이
  # 타임아웃까지 늘어진다. 종전 차선 챔버는 +x 로 달렸으므로 결과가 같다.
  # 2026-09-29: 폴링(sleep 2 + topic echo ×2 ≈ 5~7 s) → 구독 즉시 판정(rc_watch.py).
  # RTF ~0.8 인 V100 Pod 에서 폴링은 한 번에 4~5 m 를 지나쳐 10 m RC 가 14 m 가 됐다.
  # rc_watch 는 도달 즉시 /rc_cmd 정지도 직접 발행한다(정지 명령 스폰 지연 제거).
  local t0=$SECONDS xnow="$x0" ynow="$y0"
  read -r xnow ynow <<< "$(timeout "$RC_SECS" python3 "$G/rc_watch.py" "$x0" "$y0" "$RC_DIST" --drive 1.0 2>>"$LOGD/rc_watch.err")"
  [ -n "$xnow" ] && echo "[rc] 목표 거리 $RC_DIST m 달성 ((x,y)=($xnow,$ynow), ${SECONDS}-${t0}s)" >> "$log" \
    || echo "[rc] !! RC 거리 미달성 (RC_SECS=$RC_SECS 초과)" >> "$log"
  # 정지 명령을 명시적으로 한 번 — RC 를 놓으면 실차도 선다
  ros2 topic pub --once /rc_cmd geometry_msgs/msg/Twist '{linear: {x: 0.0}}' \
    > /dev/null 2>&1
  read -r x1 y1 <<< "$(timeout 15 python3 "$G/rc_watch.py" 0 0 0 2>>"$LOGD/rc_watch.err")"
  echo "[rc] 전진 종료 (x,y)=($x1,$y1)" >> "$log"

  echo "[rc] === 자율 전환 (control_mode 3 -> 1) $(date +%s) ===" >> "$log"
  kill $HS 2>/dev/null; sleep 0.5
  python3 "$G/hunter_status_pub.py" 1 --ros-args -p use_sim_time:=true > "$LOGD/hs1.log" 2>&1 &
  echo "[rc] 자율 상태 유지" >> "$log"
}

rc_phase & RCPID=$!
"$(dirname "$0")/run_gazebo_experiment_bt.sh" "$RESULT" "$DUR" 0
kill $RCPID 2>/dev/null
pkill -f hunter_status_pub.py 2>/dev/null
pkill -f base_mux_sim.py 2>/dev/null
exit 0
