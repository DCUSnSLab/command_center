#include "scv_bt_planner/bt_nodes.hpp"

#include <cstdio>

#include <cmath>
#include <limits>

namespace scv_bt_planner {

// ---------------- Context ----------------

void Context::resetTick()
{
  request_stop = false;
  stop_reason.clear();
  overlay.clear();
  behavior_name.clear();
  reverse_forced = false;
  need_publish_waypoints = false;
  safety_pause_due = false;
  blocked_actions.clear();
  pause_requests.clear();
  localization_hint.reset();
  hazard_log.clear();
}

std::string Context::zone() const
{
  const PathNode* t = target();
  if (!t) return ZoneTable::DEFAULT_ZONE;
  if (!zones) return ZoneTable::DEFAULT_ZONE;
  return zones->zoneOf(t->id);
}

int Context::effectiveNodeType() const
{
  if (reverse_forced) return 2;
  const PathNode* t = target();
  return t ? t->node_type : 1;
}

double Context::distToTarget() const
{
  const PathNode* t = target();
  if (!t || !pose) return std::numeric_limits<double>::infinity();
  // 노드 좌표는 절대 UTM, pose 는 map(datum-local). 상대 비교는 goal_distance 가 정확하고,
  // 없으면 경로 내 기하만으로는 못 구한다 → goal_distance 우선.
  if (goal_distance) return *goal_distance;
  return std::numeric_limits<double>::infinity();
}

double Context::turnDegAtTarget() const
{
  const auto& nodes = path.nodes();
  const int i = path.currentIndex();
  if (nodes.empty() || i < 0 || i + 1 >= static_cast<int>(nodes.size())) return 0.0;
  const PathNode& cur = nodes[i];
  const PathNode& nxt = nodes[i + 1];
  double in_x, in_y;
  if (i > 0) {
    in_x = cur.x - nodes[i - 1].x;
    in_y = cur.y - nodes[i - 1].y;
  } else {
    // 첫 노드: 진입 방향은 노드 heading (ENU) 으로 대체
    in_x = std::cos(cur.heading_deg * M_PI / 180.0);
    in_y = std::sin(cur.heading_deg * M_PI / 180.0);
  }
  const double out_x = nxt.x - cur.x, out_y = nxt.y - cur.y;
  const double cross = in_x * out_y - in_y * out_x;
  const double dot = in_x * out_x + in_y * out_y;
  if (std::hypot(in_x, in_y) < 1e-6 || std::hypot(out_x, out_y) < 1e-6) return 0.0;
  return std::atan2(cross, dot) * 180.0 / M_PI;   // +좌(CCW)
}

bool Context::shouldBeMoving() const
{
  // simple_behavior_planner._tick_blocked_monitor 와 동일 조건
  return path.isFollowing() && !emergency_stop && now >= pause_until && !safety_pause_due &&
         baseListening() && (!goal_distance || *goal_distance > near_goal_hold_off);
}

// ---------------- 조건 ----------------

BT::NodeStatus IsEmergencyStop::tick()
{
  return ctx_.emergency_stop ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

BT::NodeStatus MissionReady::tick()
{
  const bool ready = ctx_.path.isFollowing() && ctx_.pose.has_value() && ctx_.path.hasPath() &&
                     !ctx_.pending_align;
  return ready ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

BT::NodeStatus NeedsReverse::tick()
{
  const PathNode* t = ctx_.target();
  if (!t) return BT::NodeStatus::FAILURE;
  if (t->node_type == 2 || t->node_type == 4) return BT::NodeStatus::SUCCESS;
  if (ctx_.allow_behind_reverse && ctx_.pose) {
    // 후방 목표 판정은 map 프레임 좌표가 필요한데 노드는 절대 UTM 이라 여기서는 판단하지 않는다.
    // (ROS 노드가 datum 을 알고 있으므로 향후 ctx 에 map-frame 목표를 넣어 확장)
    return BT::NodeStatus::FAILURE;
  }
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus IsTurn::tick()
{
  std::string dir = "left";
  double min_deg = 35.0, window_m = 6.0;
  getInput("dir", dir);
  getInput("min_deg", min_deg);
  getInput("window_m", window_m);
  const double deg = ctx_.turnDegAtTarget();
  if (std::fabs(deg) < min_deg) return BT::NodeStatus::FAILURE;
  if (ctx_.distToTarget() > window_m) return BT::NodeStatus::FAILURE;
  const bool left = deg > 0;
  if ((dir == "left") == left) return BT::NodeStatus::SUCCESS;
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus ZoneIs::tick()
{
  std::string zone;
  if (!getInput("zone", zone)) return BT::NodeStatus::FAILURE;
  return ctx_.zone() == zone ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

// ---------------- 액션 ----------------

BT::NodeStatus Stop::tick()
{
  std::string reason = "guard";
  getInput("reason", reason);
  ctx_.request_stop = true;
  ctx_.stop_reason = reason;
  ctx_.behavior_name = "STOP:" + reason;
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus SafetyPause::tick()
{
  // safety_monitor._should_send_pause_command 와 동일: node_type 9 + stop_flag 만
  const PathNode* t = ctx_.target();
  if (t && t->node_type == 9 && ctx_.stop_flag) ctx_.safety_pause_due = true;
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus PauseBeforeEntry::tick()
{
  double sec = 2.0;
  getInput("sec", sec);
  const PathNode* t = ctx_.target();
  if (!t || !ctx_.goal_distance) return BT::NodeStatus::SUCCESS;
  if (*ctx_.goal_distance > ctx_.pause_trigger_distance) return BT::NodeStatus::SUCCESS;
  if (ctx_.pause_sent_for.count(t->id)) return BT::NodeStatus::SUCCESS;
  ctx_.pause_requests.push_back({sec, t->id, "zone entry pause (" + ctx_.zone() + ")"});
  ctx_.pause_sent_for.insert(t->id);
  ctx_.pause_until = ctx_.now + sec + 1.0;
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus Drive::tick()
{
  std::string profile = "sidewalk";
  getInput("profile", profile);
  // sidewalk / reverse 는 node_type 프로필 그대로 (동등성 경로). 그 외는 오버레이.
  ctx_.overlay = (profile == "sidewalk" || profile == "reverse") ? "" : profile;
  ctx_.behavior_name = profile;
  ctx_.last_selected_zone = ctx_.zone();
  ctx_.localization_hint = (ctx_.last_selected_zone == "gps_denied") ? "pcd_hold" : "gps";
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus RefreshWaypoints::tick()
{
  double move_m = 0.5, yaw_deg = 5.0;
  getInput("move_m", move_m);
  getInput("yaw_deg", yaw_deg);
  if (ctx_.waypoints_published &&
      (ctx_.anchor_moved_m > move_m || std::fabs(ctx_.anchor_moved_deg) > yaw_deg)) {
    ctx_.need_publish_waypoints = true;
  }
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus AvoidHazardWaypoints::tick()
{
  if (!ctx_.hazard_blocked || !ctx_.hazard_segment_clear || !ctx_.pose_utm || !ctx_.path.hasPath() ||
      !ctx_.path.isFollowing()) {
    return BT::NodeStatus::SUCCESS;
  }
  HazardParams hp = ctx_.hazard.params();
  getInput("clear_radius", hp.clear_radius);
  getInput("max_shift", hp.max_shift);
  getInput("lookahead", hp.lookahead);
  getInput("min_dist", hp.min_dist);
  getInput("max_dist", hp.max_dist);
  bool allow_skip = true;
  getInput("allow_skip", allow_skip);
  ctx_.hazard.setParams(hp);

  const auto& nodes = ctx_.path.nodes();
  const int cur = ctx_.path.currentIndex();
  std::vector<PathNode> upcoming(nodes.begin() + cur, nodes.end());
  const PathNode* prev = cur > 0 ? &nodes[cur - 1] : nullptr;
  const auto pls = ctx_.hazard.plan(upcoming, prev, ctx_.pose_utm->first, ctx_.pose_utm->second,
                                    ctx_.hazard_blocked, ctx_.hazard_segment_clear);
  bool republish = false;
  for (size_t i = 0; i < pls.size(); ++i) {
    const auto& pl = pls[i];
    const PathNode& nd = upcoming[i];
    if (pl.changed && !pl.skip) {
      char buf[200];
      std::snprintf(buf, sizeof(buf), "shift %s by (%+.2f, %+.2f) m -> at (%.2f, %.2f)", pl.id.c_str(), pl.dx, pl.dy,
                    nd.x + pl.dx, nd.y + pl.dy);
      ctx_.hazard_log.push_back(pl.dx == 0.0 && pl.dy == 0.0 ? "restore " + pl.id + " (clear)" : buf);
      republish = true;
    }
    if (i == 0 && pl.skip && !pl.out_of_range) {
      if (allow_skip && !ctx_.path.isFinalNode()) {
        ctx_.hazard_log.push_back("skip " + pl.id + ": lethal, no free spot within " +
                                  std::to_string(hp.max_shift).substr(0, 4) + " m");
        ctx_.path.markGoalCompleted(pl.id);
        ctx_.path.advanceToNextNode();
        ctx_.goal_distance.reset();
        republish = true;
        break;   // 다음 틱에 새 목표 기준으로 다시 본다
      }
      ctx_.hazard_log.push_back("target " + pl.id + " lethal, no free spot (final or skip disabled) — kept");
    }
  }
  if (republish) ctx_.waypoints_published = false;
  return BT::NodeStatus::SUCCESS;
}

// ---------------- 데코레이터 ----------------

BT::NodeStatus BlockedEscalation::tick()
{
  const BT::NodeStatus s = child_node_->executeTick();
  if (s != BT::NodeStatus::RUNNING) resetChild();
  std::optional<BlockedWaitMonitor::XY> xy;
  if (ctx_.pose) xy = BlockedWaitMonitor::XY{ctx_.pose->x, ctx_.pose->y};
  auto actions = ctx_.blocked.update(ctx_.now, xy, ctx_.shouldBeMoving(), ctx_.probe_far_wall);
  ctx_.blocked_actions.insert(ctx_.blocked_actions.end(), actions.begin(), actions.end());
  return s;
}

// ---------------- 등록 ----------------

void registerScvNodes(BT::BehaviorTreeFactory& factory, Context& ctx)
{
  factory.registerNodeType<IsEmergencyStop>("IsEmergencyStop", std::ref(ctx));
  factory.registerNodeType<MissionReady>("MissionReady", std::ref(ctx));
  factory.registerNodeType<NeedsReverse>("NeedsReverse", std::ref(ctx));
  factory.registerNodeType<IsTurn>("IsTurn", std::ref(ctx));
  factory.registerNodeType<ZoneIs>("ZoneIs", std::ref(ctx));
  factory.registerNodeType<Stop>("Stop", std::ref(ctx));
  factory.registerNodeType<SafetyPause>("SafetyPause", std::ref(ctx));
  factory.registerNodeType<PauseBeforeEntry>("PauseBeforeEntry", std::ref(ctx));
  factory.registerNodeType<Drive>("Drive", std::ref(ctx));
  factory.registerNodeType<RefreshWaypoints>("RefreshWaypoints", std::ref(ctx));
  factory.registerNodeType<AvoidHazardWaypoints>("AvoidHazardWaypoints", std::ref(ctx));
  factory.registerNodeType<BlockedEscalation>("BlockedEscalation", std::ref(ctx));
}

}  // namespace scv_bt_planner
