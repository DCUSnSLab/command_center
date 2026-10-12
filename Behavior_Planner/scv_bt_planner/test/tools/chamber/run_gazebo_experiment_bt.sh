#!/bin/bash
# Full Gazebo closed-loop experiment: log-derived world + real autonomy stack.
# Usage: run_gazebo_experiment.sh <result.json> [duration=120] [ped=1|0]
source /opt/ros/humble/setup.bash
source ${SCV_WS:-$HOME/SCV/vehicle_ws}/install/setup.bash
export ROS_DOMAIN_ID=${SCV_DOMAIN:-96} ROS_LOCALHOST_ONLY=1
# 파이썬 노드 로그를 줄 단위로 즉시 흘린다 — rc_phase 가 behavior.log 의
# "join idx" 를 폴링하는데, 블록 버퍼링이면 grep 이 헛돌아 목표를 최대 15회
# 재발행하고 주행 중 재합류 churn 을 만든다(8/12 실측).
export PYTHONUNBUFFERED=1

RESULT=${1:-~/scv_sim/gazebo/exp_result.json}
DUR=${2:-120}
PED=${3:-1}
G=~/scv_ws/tools/gazebo
LOGD=$(dirname "$RESULT")/logs_$(basename "$RESULT" .json)
rm -rf "$LOGD"; mkdir -p "$LOGD"; rm -f "$RESULT"

kill_stack() {
  # frenet_mpc_node 누락으로 SCV_CTRL=mpc 런마다 컨트롤러가 유출돼 이후 모든
  # 런의 /autonomy_cmd 를 오염시켰다(8/11-12 실측: 스테일 9개 동시 발행).
  local pat='gzserver|spawn_entity|robot_state_publisher|sim_relays|scv_dual_ekf.launch|ekf_node|navsat_transform_node|map_anchor_node|wheel_odom_adapter|gps_fix_gate|gps_zone_sim|pcd_sim|curb_detection_node|costmap_node|corridor_keepout|costmap_processor_node|mppi_main_node|frenet_mpc_node|maneuver_node|obstacle_dropper|base_mux_sim|hunter_status_pub|simple_behavior_planner_node|bt_planner_node|scenario_pub|sequential_planner_node|metrics_logger|gz_judge|ped_director'
  for pid in $(pgrep -f "$pat"); do
    [ "$pid" = "$$" ] && continue
    ps -o args= -p "$pid" 2>/dev/null | grep -q 'run_gazebo_experiment' && continue
    kill -9 "$pid" 2>/dev/null
  done
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* /dev/shm/_port* /dev/shm/sem._port* 2>/dev/null
}
kill_stack; sleep 1

# ---- 1) gazebo + robot ----
WORLD_FILE=${SCV_WORLD:-~/scv_sim/gazebo/scv_curb.world}
# chamber mode: SCV_URDF selects the sensor-extended robot,
# SCV_ZONES enables the zone-based GNSS simulator (owns /ublox_gps_node/fix)
URDF_FILE=${SCV_URDF:-$G/scv_sim_robot.urdf}
# SCV_XVFB=1: wrap gzserver in a virtual X server — required for the chamber
# URDF's depth camera (headless gzserver renders nothing without X; P2-verified)
GZ_PREFIX=""
[ "${SCV_XVFB:-0}" = "1" ] && GZ_PREFIX="xvfb-run -a"
# SCV_VGL=1: OGRE 렌더를 GPU 로 돌린다(VirtualGL → EGL). xvfb-run 만으로는 llvmpipe
# 소프트웨어 래스터라이저다 — V100 Pod 실측(8/27): xvfb-run 단독 llvmpipe / +vglrun -d egl
# 은 Tesla V100·OpenGL 4.6. gpu_ray·depth 카메라는 이 경로여야 GPU 를 쓴다.
[ "${SCV_VGL:-0}" = "1" ] && GZ_PREFIX="$GZ_PREFIX vglrun -d ${VGL_DISPLAY:-egl}"
setsid nohup $GZ_PREFIX gzserver -s libgazebo_ros_init.so -s libgazebo_ros_factory.so \
  "$WORLD_FILE" > "$LOGD/gzserver.log" 2>&1 < /dev/null &
sleep 12
# SCV_SPAWN_X/Y: start away from the map datum (world origin). The real
# vehicle almost never starts exactly on graph-map node[0] — on 2026-07-14 it
# started ~150 m away — and the map EKF initialises at the datum, so the
# map-frame pose is wrong by exactly this offset until the first GNSS anchor.
# SCV_SPAWN_YAW: 초기 기수방위 [rad]. 기본 0 = 종전과 동일(+x 를 본다).
# 설계된 챔버는 경로가 아무 방향으로나 뻗으므로 RC 진입 방향을 맞추려면
# 기수방위가 필요하다 — 0 으로 고정하면 RC 전진이 경로에서 멀어진다.
ros2 run gazebo_ros spawn_entity.py -entity scv -file "$URDF_FILE" \
  -x ${SCV_SPAWN_X:-0} -y ${SCV_SPAWN_Y:-0} -z ${SCV_SPAWN_Z:-0.05} -Y ${SCV_SPAWN_YAW:-0} \
  > "$LOGD/spawn.log" 2>&1
# NB: inline -p robot_description:=<urdf> breaks rcl argument parsing
# (silent rsp crash -> no sensor TF -> empty costmap). Use a params file.
python3 - "$URDF_FILE" "$LOGD/rsp_params.yaml" <<'PYEOF'
import sys
urdf = open(sys.argv[1]).read()
body = '\n'.join('      ' + l for l in urdf.splitlines())
with open(sys.argv[2], 'w') as f:
    f.write('robot_state_publisher:\n  ros__parameters:\n'
            '    use_sim_time: true\n    robot_description: |\n' + body + '\n')
PYEOF
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args --params-file "$LOGD/rsp_params.yaml" \
  > "$LOGD/rsp.log" 2>&1 &
python3 $G/sim_relays.py --ros-args -p use_sim_time:=true > "$LOGD/relays.log" 2>&1 &

# chamber: zone-based GNSS simulator replaces the stock GPS plugin as the
# /ublox_gps_node/fix publisher (chamber URDF remaps the plugin to /sim/gps_true)
if [ -n "${SCV_ZONES:-}" ]; then
  python3 $G/gps_zone_sim.py --ros-args -p use_sim_time:=true \
    -p config:="$SCV_ZONES" > "$LOGD/gps_zone.log" 2>&1 &
fi
# SCV_PCD_SIM=1: simulated PCD map-matcher (sparse ground-truth poses in the
# main map frame) so hybrid-anchor scenarios run without the real matcher
if [ "${SCV_PCD_SIM:-0}" = "1" ]; then
  python3 $G/pcd_sim.py --ros-args -p use_sim_time:=true \
    > "$LOGD/pcd_sim.log" 2>&1 &
fi

# ---- 2) localization (real chain; FAST-LIO substituted by relay) ----
# SCV_LOC_ARGS: extra launch args (e.g. gate_max_radius_m:=1e9 as the
# chamber's negative control that neutralizes the GNSS sanity gate)
ros2 launch robot_localization scv_dual_ekf.launch.py \
  use_sim_time:=true with_fastlio:=false ${SCV_LOC_ARGS:-} > "$LOGD/loc.log" 2>&1 &
sleep 8

# ---- 3) safety stack (all real nodes) ----
B=${SCV_WS:-$HOME/SCV/vehicle_ws}/install/pcd_ground_filter/share/pcd_ground_filter
S=${SCV_WS:-$HOME/SCV/vehicle_ws}/install/smppi/share/smppi
# SCV_CURB_ARGS: 연석 노드 추가 인자 (예: -p method:=ring — 2026-08-22 A/B)
ros2 run pcd_ground_filter curb_detection_node --ros-args -p use_sim_time:=true \
  --params-file "$B/config/curb_params.yaml" ${SCV_CURB_ARGS:-} > "$LOGD/curb.log" 2>&1 &
ros2 run local_costmap costmap_node --ros-args -p use_sim_time:=true \
  -p point_cloud_topic:=/velodyne_points_curb -p update_frequency:=10.0 > "$LOGD/costmap.log" 2>&1 &
[ -f "$S/../../lib/smppi/corridor_keepout_node.py" ] && ros2 run smppi corridor_keepout_node.py --ros-args -p use_sim_time:=true \
  --params-file "$S/config/smppi_params.yaml" ${SCV_CORRIDOR_ARGS:-} > "$LOGD/corridor.log" 2>&1 &
[ -f "$S/../../lib/smppi/costmap_processor_node.py" ] && ros2 run smppi costmap_processor_node.py --ros-args -p use_sim_time:=true \
  --params-file "$S/config/smppi_params.yaml" > "$LOGD/cproc.log" 2>&1 &
# SCV_CTRL: 제어기 선택. 기본 mppi(종전과 동일). mpc = Frenet MPC 프로토타입
# (tools/mpc/frenet_mpc_node.py, MPPI 와 토픽 계약 동일 — A/B 비교용).
if [ "${SCV_CTRL:-mppi}" = "mpc" ]; then
  python3 ~/scv_ws/tools/mpc/frenet_mpc_node.py --ros-args -p use_sim_time:=true \
    ${SCV_MPPI_ARGS:-} > "$LOGD/mppi.log" 2>&1 &
  # SCV_MANEUVER=1: Hybrid A* 기동 계층 병행 기동 (3분할 구조의 '기동')
  if [ "${SCV_MANEUVER:-0}" = "1" ]; then
    python3 ~/scv_ws/tools/mpc/maneuver_node.py --ros-args -p use_sim_time:=true \
      ${SCV_MPPI_ARGS:-} > "$LOGD/maneuver.log" 2>&1 &
  fi
else
  ros2 run smppi mppi_main_node.py --ros-args -p use_sim_time:=true \
    --params-file "$S/config/smppi_params.yaml" ${SCV_MPPI_ARGS:-} > "$LOGD/mppi.log" 2>&1 &
fi
# SCV_DROP="tx,ty,tr,bx,by": 주행 중 장애물 투하 (막힘 탈출 시나리오)
if [ -n "${SCV_DROP:-}" ]; then
  IFS=',' read -r DTX DTY DTR DBX DBY <<< "$SCV_DROP"
  python3 $G/obstacle_dropper.py --ros-args -p use_sim_time:=true \
    -p trigger_x:=$DTX -p trigger_y:=$DTY -p trigger_r:=$DTR \
    -p box_x:=$DBX -p box_y:=$DBY > "$LOGD/dropper.log" 2>&1 &
fi
# SCV_BP_ARGS: 시작노드-미지정 시나리오는 필드와 동일하게
#   -p current_position_topic:=/odometry/global 을 넘긴다 (기본 /odom 은
#   odom 프레임이라 map 프레임 노드와의 최근접 정렬이 스폰 오프셋만큼 틀어진다)
# datum (BP waypoint 게이트 통과용). 지도 원점 = 월드 원점 (SCV_PLANNER_ARGS datum 0,0 과 일치).
ros2 topic pub --qos-durability transient_local --qos-reliability reliable \
  /map_provider_node/utm map_interfaces/msg/UtmLayer \
  "{utm_zone: 52, zone_letter: 'S', northern: true, origin_easting: ${SCV_DATUM_E:-0.0}, origin_northing: ${SCV_DATUM_N:-0.0}}" \
  > "$LOGD/datum.log" 2>&1 &
UTM_PID=$!   # latched pub 는 kill_stack 패턴 밖 — 종료 시 PID 로 정리(런마다 1개씩 누수 → 도메인 디스커버리 불능, 10/01 Pod 실측)
# SCV_BP: simple(현행) | bt(scv_bt_planner active, simple 미기동) | shadow(simple 명령 + BT 결정만)
BP_MODE=${SCV_BP:-simple}
if [ "$BP_MODE" != "bt" ]; then
  ros2 run simple_behavior_planner simple_behavior_planner_node.py \
    --ros-args -p use_sim_time:=true ${SCV_BP_ARGS:-} > "$LOGD/behavior.log" 2>&1 &
fi
if [ "$BP_MODE" != "simple" ]; then
  BTMODE=shadow; [ "$BP_MODE" = "bt" ] && BTMODE=active
  ${SCV_WS:-$HOME/SCV/vehicle_ws}/install/scv_bt_planner/lib/scv_bt_planner/bt_planner_node \
    --ros-args -p use_sim_time:=true -p mode:=$BTMODE ${SCV_BP_ARGS:-} ${SCV_BT_ARGS:-} > "$LOGD/bt.log" 2>&1 &
fi
sleep 6

# ---- fault injection (C3 test): kill the curb node mid-run ----
if [ -n "${FAULT_T:-}" ]; then
  (
    sleep "$FAULT_T"
    pkill -9 -f curb_detection_node
    echo "FAULT: curb_detection_node killed at T+${FAULT_T}s" >> "$LOGD/fault.log"
  ) &
fi

# ---- 4) scenario: route + (optional) pedestrian director + judge ----
# SCV_GRAPH_MAP: 합성 경로 대신 실제 그래프 지도 + sequential planner 를 쓴다.
# SCV_GOAL_NODE 를 주면 auto_start 를 끄고 목적지 라우팅으로 기동한다 —
# 로봇은 goal 수신 전까지 대기하고, 수신 시 최근접→목적지 최단 경로(무방향)로
# 출발한다. goal 은 두 번 보낸다: 첫 발행이 위치추정 수렴 전에 도착해
# 거절될 수 있다(플래너가 의도적으로 거절 — 부트스트랩 포즈 라우팅 방지).
if [ -n "${SCV_GRAPH_MAP:-}" ]; then
  AUTO=true; [ -n "${SCV_GOAL_NODE:-}" ] && AUTO=false
  # SCV_PLANNER_ARGS: 플래너 파라미터 추가/덮어쓰기. 기본 first_node 는
  # **지도 원점을 node[0] 의 UtmInfo 로 잡으므로 node[0] 이 월드 원점이 아니면
  # 지도 프레임 전체가 그만큼 밀린다.** chamber_graph.json 은 G00 이 (0,0)
  # 이라 우연히 맞았고, 다른 좌표에 경로를 그리자마자 이격이 설계와 달라져
  # 실험이 조용히 어긋났다(실측: 3 m 로 잡은 이격이 1.89 m). 원점 제약 없이
  # 쓰려면
  #   SCV_PLANNER_ARGS="-p map_origin_source:=datum -p datum_utm_easting:=0.0 -p datum_utm_northing:=0.0"
  # 로 넘겨 지도 원점을 월드 원점에 고정한다.
  ros2 run sequential_global_planner sequential_planner_node.py --ros-args \
    -p use_sim_time:=true -p map_file:="$SCV_GRAPH_MAP" \
    -p auto_start:=$AUTO -p explicit_endpoints:=false \
    -p map_origin_source:=first_node ${SCV_PLANNER_ARGS:-} \
    > "$LOGD/planner.log" 2>&1 &
  if [ -n "${SCV_GOAL_NODE:-}" ]; then
    ( for d in ${SCV_GOAL_DELAYS:-40 75}; do
        sleep "$d"
        ros2 topic pub --once /goal_node_id std_msgs/String "{data: '$SCV_GOAL_NODE'}"
      done ) > "$LOGD/goalpub.log" 2>&1 &
  fi
else
  python3 ~/scv_ws/tools/closedloop/scenario_pub.py > "$LOGD/scenario.log" 2>&1 &
fi
if [ "$PED" = "1" ]; then
  python3 $G/ped_director.py > "$LOGD/ped.log" 2>&1 &
fi
JUDGE_ARGS=""
[ -n "${SCV_ZONES:-}" ] && JUDGE_ARGS="--zones $SCV_ZONES"
[ -n "${SCV_SCENARIO:-}" ] && JUDGE_ARGS="$JUDGE_ARGS --scenario $SCV_SCENARIO"
timeout $((DUR + 30)) python3 $G/gz_judge.py "$RESULT" $DUR $JUDGE_ARGS > "$LOGD/judge.log" 2>&1
RC=$?

kill "${UTM_PID:-}" 2>/dev/null; kill_stack
echo "=== RESULT ==="
cat "$RESULT" 2>/dev/null || tail -8 "$LOGD/judge.log"
exit $RC
