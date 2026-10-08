// simple_behavior_planner/blocked_wait_monitor.py 의 C++ 이식.
//
//   NORMAL --(정지 blocked_detect_sec, 움직여야 하는 상황)--> BLOCKED_WAIT
//   BLOCKED_WAIT --(진행 재개)--> NORMAL / --(wait_timeout)--> CREEP
//   CREEP        --(진행 재개)--> NORMAL / --(creep_timeout)--> ASSIST
//   ASSIST       --(진행 재개)--> NORMAL
//
// 시간은 update(now, ...) 로 주입해 ROS 없이 시험한다. probe(탐침 전진)는
// 원본 파라미터를 그대로 두되 호스트가 far_wall 분류를 넘겨줘야 동작한다.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace scv_bt_planner {

class BlockedWaitMonitor {
public:
  static constexpr const char* NORMAL = "NORMAL";
  static constexpr const char* BLOCKED_WAIT = "BLOCKED_WAIT";
  static constexpr const char* CREEP = "CREEP";
  static constexpr const char* ASSIST = "ASSIST";

  struct Params {
    double blocked_detect_sec = 4.0;
    double progress_eps = 0.15;
    double wait_timeout = 12.0;
    double creep_timeout = 10.0;
    double creep_speed = 0.3;
    double assist_repeat_sec = 15.0;
    bool probe_enabled = false;
    double probe_max_dist = 2.0;
    int probe_max_fails = 2;
    double probe_warm_wait = 3.0;
    double probe_warm_window = 30.0;
  };

  using XY = std::pair<double, double>;

  // 주의: 중첩 struct 의 NSDMI 는 클래스 정의 끝까지 완성되지 않아 기본 인자로 못 쓴다(GCC)
  BlockedWaitMonitor() = default;
  explicit BlockedWaitMonitor(const Params& p) : p_(p) {}

  // 반환: 호스트가 처리할 액션 문자열 목록
  //   "announce:<STATE>", "creep_on", "creep_off", "assist", "probe_on", "probe_off"
  std::vector<std::string> update(double now, const std::optional<XY>& xy,
                                  bool should_be_moving, bool far_wall = false);

  const std::string& state() const { return state_; }
  const Params& params() const { return p_; }
  bool probeActive() const { return probe_active_; }

private:
  static double dist2(const XY& a, const XY& b)
  {
    const double dx = a.first - b.first, dy = a.second - b.second;
    return dx * dx + dy * dy;
  }
  void enter(const char* s, double now) { state_ = s; state_since_ = now; }
  void resetAnchor(double now, const XY& xy) { anchor_xy_ = xy; anchor_t_ = now; }
  std::vector<std::string> toNormal(double now);

  Params p_{};
  std::string state_ = NORMAL;
  std::optional<double> state_since_;
  std::optional<XY> anchor_xy_;
  std::optional<double> anchor_t_;
  std::optional<double> last_assist_t_;
  bool probe_active_ = false;
  std::optional<XY> probe_start_xy_;
  int probe_fails_ = 0;
  std::optional<double> warm_until_;
};

}  // namespace scv_bt_planner
