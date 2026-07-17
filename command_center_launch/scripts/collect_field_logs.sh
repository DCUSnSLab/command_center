#!/bin/bash
# Field diagnostic snapshot — run AFTER (or during) a field attempt so the
# run can be analysed later without the vehicle present.
#
#   collect_field_logs.sh [out_dir]        (default ~/field_logs)
#
# Produces <out_dir>/scv_diag_<timestamp>.tar.gz — text only, a few hundred
# KB. Safe to run over LTE. It records NOTHING that moves the vehicle.
#
# Why each item is here (all bit us at least once):
#   env/git      : which code actually ran (SCV_park branches + dirty files)
#   ros_logs     : "process has died" — how the 2026-07-14 hunter_base
#                  SIGABRT and the rsp crash were both found
#   can/net      : can0 DOWN => hunter_base dies => no actuation, no wheel odom
#   gnss         : cold-start garbage fixes => map EKF divergence (gps_fix_gate)
#   topics/tf    : silent perception starvation (missing TF => empty costmap)
#   bags         : what was recorded, so it can be uploaded from the lab later
set -u
OUT=${1:-~/field_logs}
TS=$(date +%Y%m%d_%H%M%S)
D=$(mktemp -d)/scv_diag_$TS
mkdir -p "$D" "$OUT"
WS=${SCV_WS:-/home/scv/SCV_park}

say() { echo "  - $1"; }
echo "collecting into $D"

# ---- 1. environment / code identity -----------------------------------
say "env + git"
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "uptime: $(uptime -p)"
  echo "workspace: $WS"
  echo; echo "=== branches / commits ==="
  for d in src/perception src/command_center src/localization/robot_localization \
           src/vehicle/hunter_ros2 src/vehicle/hunter2_description; do
    [ -d "$WS/$d" ] || continue
    printf '%-46s %-26s %s\n' "$d" \
      "$(git -C "$WS/$d" branch --show-current 2>/dev/null)" \
      "$(git -C "$WS/$d" log --oneline -1 2>/dev/null)"
  done
  echo; echo "=== uncommitted (pyc excluded) ==="
  for d in src/perception src/command_center src/localization/robot_localization \
           src/vehicle/hunter_ros2 src/vehicle/hunter2_description \
           src/sensor_pkg/sllidar_ros2 src/sensor_pkg/ntrip_client_ros2; do
    [ -d "$WS/$d" ] || continue
    m=$(git -C "$WS/$d" status -s 2>/dev/null | grep -v pycache)
    [ -n "$m" ] && { echo "--- $d"; echo "$m"; }
  done
  echo; echo "=== key params ==="
  grep -hE 'min_obstacle_height|max_obstacle_height|staleness_timeout' \
    "$WS"/src/command_center/costmap/local_costmap/config/*.yaml 2>/dev/null
  grep -hE '^\s*(datum|wait_for_datum):' \
    "$WS"/src/localization/robot_localization/params/scv_dual_ekf.yaml 2>/dev/null
} > "$D/00_env_git.txt" 2>&1

# ---- 2. ROS launch logs (today) ---------------------------------------
say "ros launch logs"
mkdir -p "$D/ros_logs"
for d in $(ls -td ~/.ros/log/$(date +%Y-%m-%d)* 2>/dev/null | head -4); do
  n=$(basename "$d")
  # launch.log is the one that names dead processes; node stdout can be huge
  [ -f "$d/launch.log" ] && cp "$d/launch.log" "$D/ros_logs/$n.launch.log" 2>/dev/null
done
{
  echo "=== 'process has died' across today's launches ==="
  grep -h 'process has died' "$D"/ros_logs/*.launch.log 2>/dev/null |
    sed -E 's/.*\[([^]]+)\]: process has died \[pid ([0-9]+), exit code ([-0-9]+).*/\1  pid=\2 exit=\3/' |
    sort | uniq -c | sort -rn
  echo; echo "=== errors/warnings (deduped) ==="
  grep -hiE '\[(ERROR|WARN)\]' "$D"/ros_logs/*.launch.log 2>/dev/null |
    sed -E 's/[0-9]{10}\.[0-9]+//; s/\[[0-9]+\.[0-9]+\]//; s/pid [0-9]+/pid N/' |
    sort | uniq -c | sort -rn | head -40
} > "$D/01_node_failures.txt" 2>&1

# ---- 3. vehicle bus / network -----------------------------------------
say "can + network"
{
  echo "=== can0 ==="; ip -det link show can0 2>&1 | head -5
  echo "(state ERROR-ACTIVE + UP = healthy; DOWN => hunter_base will SIGABRT)"
  echo; echo "--- 3 s of CAN traffic ---"; timeout 3 candump can0 -n 8 2>&1 | head -8
  echo; echo "=== interfaces ==="; ip -br a | grep -vE 'docker|veth|^lo'
  echo; echo "=== routes ==="; ip route | head -6
  echo; echo "=== policy routing (tether/wired coexistence) ==="
  ip rule | grep -E '^100:' || echo "(rule missing — campus inbound breaks when tethering)"
  ip route show table 100 2>/dev/null
  echo; echo "=== velodyne link ==="
  ip -br a | grep enp7s0
  timeout 3 python3 - <<'PY' 2>&1
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(2.5)
try:
    s.bind(('', 2368)); d, a = s.recvfrom(2000)
    print(f"velodyne UDP OK: {len(d)}B from {a[0]}")
except socket.timeout: print("velodyne: NO UDP DATA (check power/cable/enp7s0)")
except OSError as e: print(f"velodyne: port busy (driver running?) — {e}")
PY
  echo; echo "=== tailscale ==="; tailscale status 2>&1 | head -3
} > "$D/02_can_network.txt" 2>&1

# ---- 4. live ROS graph (only if a stack is already running) ------------
# Whole section is time-boxed: ros2 CLI can block indefinitely spinning up
# its daemon on a quiet graph, and this script must never hang in the field.
say "ros graph (live topics/TF)  [max 120 s]"
timeout 120 bash <<'GRAPH' > "$D/03_ros_graph.txt" 2>&1
  WS=${SCV_WS:-/home/scv/SCV_park}
  source /opt/ros/humble/setup.bash 2>/dev/null
  source "$WS/install/setup.bash" 2>/dev/null
  export ROS_DAEMON_TIMEOUT=5
  NODES=$(timeout 10 ros2 node list --no-daemon 2>/dev/null)
  if [ -z "$NODES" ]; then
    echo "(no ROS graph — stack not running; run this again while field_drive is up)"
  else
    echo "=== nodes ==="; echo "$NODES"
    echo; echo "=== rates (5 s each) ==="
    for t in /velodyne_points /velodyne_points_curb /vectornav/imu /ublox_gps_node/fix \
             /gps/fix_gated /hunter/velocity /front/scan /rear/scan /odom \
             /odometry/global /costmap /costmap_keepout /cmd_vel; do
      printf '%-26s ' "$t"
      timeout 6 ros2 topic hz "$t" --window 30 2>/dev/null |
        grep -m1 'average rate' || echo 'NO DATA'
    done
    echo; echo "=== GPS fix (status 2 = RTK fixed) ==="
    timeout 6 ros2 topic echo /ublox_gps_node/fix --once 2>/dev/null |
      grep -E 'status:|latitude:|longitude:|position_covariance:' | head -5
    echo; echo "=== TF chain ==="
    for f in odom base_link velodyne gnss_antenna front_lidar rear_lidar; do
      printf 'map->%-14s ' "$f"
      timeout 4 ros2 run tf2_ros tf2_echo map "$f" 2>/dev/null |
        grep -m1 'Translation' || echo 'NO TF'
    done
    echo; echo "=== localization vs GPS anchor (should agree within a few m) ==="
    timeout 5 ros2 topic echo /odometry/global --once 2>/dev/null | grep -A3 'position:' | head -4
    timeout 5 ros2 topic echo /odometry/gps --once 2>/dev/null | grep -A3 'position:' | head -4
    echo; echo "=== behavior state ==="
    timeout 5 ros2 topic echo /behavior_status --once 2>/dev/null | head -3
  fi
GRAPH
[ -s "$D/03_ros_graph.txt" ] || echo "(ros graph section timed out)" > "$D/03_ros_graph.txt"

# ---- 5. recorded bags (inventory only — never copy the data) -----------
say "bag inventory"
{
  echo "=== ~/bags (newest first) ==="
  ls -lt ~/bags 2>/dev/null | head -12
  echo; echo "=== today's bags: topics + counts ==="
  for b in $(ls -td ~/bags/*$(date +%Y%m%d)* 2>/dev/null | head -3); do
    echo "--- $(basename "$b")  ($(du -sh "$b" | cut -f1))"
    python3 - "$b" <<'PY' 2>/dev/null
import sys, yaml
i = yaml.safe_load(open(sys.argv[1] + '/metadata.yaml'))['rosbag2_bagfile_information']
print("    duration %.0fs, %d msgs" % (i['duration']['nanoseconds']/1e9, i['message_count']))
for t in sorted(i['topics_with_message_count'], key=lambda x: -x['message_count'])[:14]:
    print(f"    {t['topic_metadata']['name']:38s} {t['message_count']}")
PY
  done
  echo; echo "NOTE: upload bags from the LAB (wired), not over LTE:"
  echo "  bash ~/SCV_park/src/command_center/command_center_launch/scripts/upload_bags.sh <bag_dir>..."
} > "$D/04_bags.txt" 2>&1

# ---- 6. system ---------------------------------------------------------
say "system"
{
  df -h /home /; echo; free -h
  echo; echo "=== USB (sensors) ==="; lsusb | grep -iE 'ublox|realsense|silicon|future|cypress'
  echo; echo "=== serial ==="; ls -l /dev/ttyUSB* /dev/ttyACM* /dev/rplidar_* 2>/dev/null
  echo; echo "=== dmesg (usb/can/eth) ==="
  dmesg -T 2>/dev/null | grep -iE 'usb|can0|enp7s0|disconnect' | tail -12 ||
    echo "(dmesg needs root)"
} > "$D/05_system.txt" 2>&1

# ---- pack --------------------------------------------------------------
tar czf "$OUT/scv_diag_$TS.tar.gz" -C "$(dirname "$D")" "scv_diag_$TS"
rm -rf "$(dirname "$D")"
echo
echo "=== SUMMARY ==="
tar xzOf "$OUT/scv_diag_$TS.tar.gz" "scv_diag_$TS/01_node_failures.txt" 2>/dev/null |
  sed -n '/process has died/,/^$/p' | head -6
echo "saved: $OUT/scv_diag_$TS.tar.gz  ($(du -h "$OUT/scv_diag_$TS.tar.gz" | cut -f1))"
echo "share it: scp scv@100.102.222.82:$OUT/scv_diag_$TS.tar.gz ."
