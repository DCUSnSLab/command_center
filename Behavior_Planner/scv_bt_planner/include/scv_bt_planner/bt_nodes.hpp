// BT 노드와 공유 Context.
//
// 설계 원칙 (2026-09-28): 모든 노드는 즉시 SUCCESS/FAILURE 를 돌려주는 결정 트리다.
// 상태(에스컬레이션 타이머, pause 발동 이력, 발행 이력)는 Context 에 두고 ROS 노드가
// 매 틱 입력을 채운 뒤 트리를 한 번 돌리고 출력을 소비한다. RUNNING 은 쓰지 않는다.
#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <behaviortree_cpp/bt_factory.h>

#include "scv_bt_planner/blocked_wait_monitor.hpp"
#include "scv_bt_planner/hazard_waypoints.hpp"
#include "scv_bt_planner/path_manager.hpp"
#include "scv_bt_planner/profiles.hpp"
#include "scv_bt_planner/zone_table.hpp"

namespace scv_bt_planner {

struct Pose2D {
  double x = 0.0, y = 0.0, yaw = 0.0;   // map 프레임 (/odometry/global)
};

struct PauseRequest {
  double duration = 0.0;
  std::string node_id;
  std::string reason;
};

struct Context {
  // ---------- 입력 (ROS 노드가 채움) ----------
  std::optional<Pose2D> pose;
  PathManager path;
  bool pending_align = false;
  int control_mode = -1;             // /hunter_status.control_mode, -1 = 미상(청취 중으로 간주)
  bool emergency_stop = false;
  bool stop_flag = false;
  std::optional<double> goal_distance;   // /goal_status.distance_to_goal (현재 목표 일치 시)
  double now = 0.0;                  // steady 초
  double pause_until = 0.0;
  double pause_trigger_distance = 0.8;
  double anchor_moved_m = 0.0;       // 마지막 웨이포인트 발행 이후 map→odom 병진 이동량
  double anchor_moved_deg = 0.0;     // 동 회전량
  bool waypoints_published = false;
  const ZoneTable* zones = nullptr;
  const Profiles* profiles = nullptr;
  bool allow_behind_reverse = false;
  double near_goal_hold_off = 0.8;
  bool probe_far_wall = false;
  BlockedWaitMonitor blocked;
  // 위험 지대 경유점 재배치 (AvoidHazardWaypoints). 코스트맵 질의는 노드가 채워 넣는다(미설정이면 비활성).
  HazardWaypoints hazard;
  HazardWaypoints::BlockedAt hazard_blocked;          // 절대 UTM 점이 치명인가
  HazardWaypoints::SegmentClear hazard_segment_clear; // 절대 UTM 직선이 통행 가능한가
  std::optional<std::pair<double, double>> pose_utm;   // 차량 절대 UTM (datum 수신 후)
  std::vector<std::string> hazard_log;                // 이번 틱의 재배치·건너뜀 기록(노드가 로그로 출력)
  double hazard_seen_t = -1e9;                        // 판정 범위 안에서 위험(재배치·빈 곳 없음)을 마지막으로 본 시각(steady)
  bool hazardRecent(double window = 3.0) const { return now - hazard_seen_t <= window; }

  // ---------- 출력 (트리가 채우고 ROS 노드가 소비) ----------
  bool request_stop = false;
  std::string stop_reason;
  std::string overlay;               // "" = node_type 프로필만 (동등성 경로)
  std::string behavior_name;         // 로그·A/B 비교용
  bool reverse_forced = false;       // 후방 목표 자동 후진 (allow_behind_reverse 일 때만)
  bool need_publish_waypoints = false;
  bool safety_pause_due = false;     // node_type 9 + stop_flag
  std::vector<std::string> blocked_actions;
  std::vector<PauseRequest> pause_requests;
  std::optional<std::string> localization_hint;
  std::set<std::string> pause_sent_for;  // PauseBeforeEntry 발동 이력 (노드 ID)
  std::string last_selected_zone;

  // 틱마다 초기화되는 출력만 지운다 (이력·상태기계는 유지)
  void resetTick();

  // ---------- 파생 조회 ----------
  // zone()·turnDegAtTarget() 은 한 틱에 여러 노드(ZoneIs ×5, IsTurn ×2, Drive, 발행부)가 부르므로 목표 노드별로
  // 한 번만 계산해 둔다. 캐시는 resetTick() 에서 비우고, 같은 틱 안에서 목표가 바뀌면(위험 경유점 건너뜀) 목표 포인터
  // 불일치로 다시 계산한다. 경로 교체(setPath)는 콜백에서 일어나고 틱과 같은 스레드라 틱 중간에는 바뀌지 않는다.
  const PathNode* target() const { return path.currentTarget(); }
  const std::string& zone() const;          // 현재 목표 노드의 Zone (기본 sidewalk)
  int effectiveNodeType() const;            // reverse_forced 면 2
  // 현재 목표 노드에서의 진행 방향 변화 (도, +좌/-우). 다음 노드 없으면 0.
  double turnDegAtTarget() const;
  double distToTarget() const;              // pose 기준, 미상이면 +inf
  bool baseListening() const { return control_mode < 0 || control_mode == 1; }
  bool shouldBeMoving() const;

private:
  double computeTurnDeg() const;
  void refreshCache() const;
  mutable const PathNode* cache_target_ = nullptr;   // 캐시가 유효한 목표 노드(nullptr = 무효)
  mutable std::string cache_zone_ = ZoneTable::DEFAULT_ZONE;
  mutable double cache_turn_deg_ = 0.0;
};

// 포트는 XML 리터럴(정적)이라 첫 틱에 한 번만 읽는다 — getInput 은 호출마다 포트 표 탐색 + 문자열 파싱을 한다.
// 블랙보드 재매핑({var})을 쓰면 첫 틱 값만 반영되므로, 틱마다 바뀌는 값은 Context 로 넣을 것.
template <class T>
T readPortOnce(const BT::TreeNode& n, const char* key, T dflt)
{
  T v = dflt;
  n.getInput(key, v);
  return v;
}

// ---- 조건 ----
class IsEmergencyStop : public BT::ConditionNode {
public:
  IsEmergencyStop(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::ConditionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;
private: Context& ctx_;
};

class MissionReady : public BT::ConditionNode {
public:
  MissionReady(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::ConditionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;
private: Context& ctx_;
};

// node_type 2/4 (후진·정밀 후진) 이면 SUCCESS. 후방 목표 자동 후진(allow_behind_reverse)은 map 프레임 목표가 필요해
// 아직 판정하지 않는다(README "미이식 / 보류").
class NeedsReverse : public BT::ConditionNode {
public:
  NeedsReverse(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::ConditionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;
private: Context& ctx_;
};

class IsTurn : public BT::ConditionNode {
public:
  IsTurn(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::ConditionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<std::string>("dir", "left", "left | right"),
            BT::InputPort<double>("min_deg", 35.0, "회전 판정 최소 각도"),
            BT::InputPort<double>("window_m", 6.0, "목표 노드까지 이 거리 안에서만 회전으로 본다")};
  }
  BT::NodeStatus tick() override;
private:
  Context& ctx_;
  struct Cfg { bool want_left; double min_deg; double window_m; };
  std::optional<Cfg> cfg_;
};

class ZoneIs : public BT::ConditionNode {
public:
  ZoneIs(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::ConditionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {BT::InputPort<std::string>("zone")}; }
  BT::NodeStatus tick() override;
private:
  Context& ctx_;
  std::optional<std::string> zone_;
};

// ---- 액션 ----
// 목표·다음 경유점이 코스트맵 치명 영역 안이면 경로 옆으로 비켜 놓거나(재배치) 현재 목표를 건너뛴다. 항상 SUCCESS.
class AvoidHazardWaypoints : public BT::SyncActionNode {
public:
  AvoidHazardWaypoints(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::SyncActionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<double>("clear_radius", 0.75, "경유점 주변 무치명 반경 [m]"),
            BT::InputPort<double>("max_shift", 2.0, "경로 법선 방향 최대 이동 [m]"),
            BT::InputPort<int>("lookahead", 3, "현재 목표 뒤로 검사할 노드 수"),
            BT::InputPort<double>("min_dist", 1.5, "차량에서 이 거리보다 가까운 노드는 판정 안 함 [m]"),
            BT::InputPort<double>("max_dist", 6.5, "차량에서 이 거리보다 먼 노드는 판정 안 함 [m]"),
            BT::InputPort<bool>("allow_skip", true, "빈 곳이 없으면 현재 목표를 건너뛴다(최종 노드 제외)")};
  }
  BT::NodeStatus tick() override;
private:
  Context& ctx_;
  std::string last_hold_;   // 같은 '정지 유지' 기록을 틱마다 반복하지 않는다
  double last_hold_pause_t_ = -1e9;
  std::map<std::string, int> skip_streak_;   // 노드별 연속 회피 불가 판정 틱 수
  int hold_persist_ticks_ = 6;               // 0.3 s @20 Hz
};

class Stop : public BT::SyncActionNode {
public:
  Stop(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::SyncActionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {BT::InputPort<std::string>("reason", "guard", "")}; }
  BT::NodeStatus tick() override;   // 항상 FAILURE — 이번 틱의 나머지를 중단
private:
  Context& ctx_;
  std::optional<std::string> reason_, behavior_;
};

class SafetyPause : public BT::SyncActionNode {
public:
  SafetyPause(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::SyncActionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;
private: Context& ctx_;
};

class PauseBeforeEntry : public BT::SyncActionNode {
public:
  PauseBeforeEntry(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::SyncActionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {BT::InputPort<double>("sec", 2.0, "진입 전 정지 시간")}; }
  BT::NodeStatus tick() override;
private:
  Context& ctx_;
  std::optional<double> sec_;
};

class Drive : public BT::SyncActionNode {
public:
  Drive(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::SyncActionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {BT::InputPort<std::string>("profile", "sidewalk", "")}; }
  BT::NodeStatus tick() override;
private:
  Context& ctx_;
  struct Cfg { std::string profile; std::string overlay; };   // overlay "" = node_type 프로필만 (동등성 경로)
  std::optional<Cfg> cfg_;
};

class RefreshWaypoints : public BT::SyncActionNode {
public:
  RefreshWaypoints(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::SyncActionNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<double>("move_m", 0.5, ""), BT::InputPort<double>("yaw_deg", 5.0, "")};
  }
  BT::NodeStatus tick() override;
private:
  Context& ctx_;
  struct Cfg { double move_m; double yaw_deg; };
  std::optional<Cfg> cfg_;
};

// ---- 데코레이터: 정체 에스컬레이션 ----
class BlockedEscalation : public BT::DecoratorNode {
public:
  BlockedEscalation(const std::string& n, const BT::NodeConfig& c, Context& ctx) : BT::DecoratorNode(n, c), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;
private: Context& ctx_;
};

void registerScvNodes(BT::BehaviorTreeFactory& factory, Context& ctx);

}  // namespace scv_bt_planner
