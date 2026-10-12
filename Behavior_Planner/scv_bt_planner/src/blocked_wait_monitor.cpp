#include "scv_bt_planner/blocked_wait_monitor.hpp"

namespace scv_bt_planner {

std::vector<std::string> BlockedWaitMonitor::update(double now, const std::optional<XY>& xy,
                                                    bool should_be_moving, bool far_wall)
{
  std::vector<std::string> actions;
  if (!xy) return actions;

  if (!should_be_moving) {
    // 계획된 정지(pause 노드·안전 정지·목표 도달) — 차단이 아니다
    if (state_ != NORMAL) {
      auto a = toNormal(now);
      actions.insert(actions.end(), a.begin(), a.end());
    }
    resetAnchor(now, *xy);
    return actions;
  }

  if (!anchor_xy_) {
    resetAnchor(now, *xy);
    return actions;
  }

  const bool progressed = dist2(*xy, *anchor_xy_) > p_.progress_eps * p_.progress_eps;
  if (progressed && probe_active_) {
    // probe 중 진행은 예산까지 NORMAL 복귀 없이 유지
    const double moved2 = dist2(*xy, *probe_start_xy_);
    if (moved2 >= p_.probe_max_dist * p_.probe_max_dist) {
      probe_fails_ = 0;
      warm_until_ = now + p_.probe_warm_window;
      auto a = toNormal(now);
      actions.insert(actions.end(), a.begin(), a.end());
      resetAnchor(now, *xy);
    }
    return actions;
  }
  if (progressed) {
    if (state_ != NORMAL) {
      auto a = toNormal(now);
      actions.insert(actions.end(), a.begin(), a.end());
    }
    resetAnchor(now, *xy);
    return actions;
  }

  const double stalled_for = now - *anchor_t_;

  if (state_ == NORMAL) {
    if (stalled_for >= p_.blocked_detect_sec) {
      enter(BLOCKED_WAIT, now);
      actions.emplace_back(std::string("announce:") + BLOCKED_WAIT);
    }
  } else if (state_ == BLOCKED_WAIT) {
    double eff_wait = p_.wait_timeout;
    if (p_.probe_enabled && far_wall && warm_until_ && now < *warm_until_) {
      eff_wait = p_.probe_warm_wait;
    }
    if (now - *state_since_ >= eff_wait) {
      enter(CREEP, now);
      actions.emplace_back(std::string("announce:") + CREEP);
      actions.emplace_back("creep_on");
      if (p_.probe_enabled && far_wall && probe_fails_ < p_.probe_max_fails) {
        probe_active_ = true;
        probe_start_xy_ = xy;
        actions.emplace_back("probe_on");
      }
    }
  } else if (state_ == CREEP) {
    if (now - *state_since_ >= p_.creep_timeout) {
      if (probe_active_) {
        ++probe_fails_;
        probe_active_ = false;
        probe_start_xy_.reset();
        actions.emplace_back("probe_off");
      }
      enter(ASSIST, now);
      actions.emplace_back(std::string("announce:") + ASSIST);
      actions.emplace_back("creep_off");
      actions.emplace_back("assist");
      last_assist_t_ = now;
    }
  } else if (state_ == ASSIST) {
    if (!last_assist_t_ || now - *last_assist_t_ >= p_.assist_repeat_sec) {
      actions.emplace_back("assist");
      last_assist_t_ = now;
    }
  }
  return actions;
}

std::vector<std::string> BlockedWaitMonitor::toNormal(double now)
{
  std::vector<std::string> actions;
  if (probe_active_) {
    probe_active_ = false;
    probe_start_xy_.reset();
    actions.emplace_back("probe_off");
  }
  if (state_ == CREEP) actions.emplace_back("creep_off");
  actions.emplace_back(std::string("announce:") + NORMAL);
  enter(NORMAL, now);
  last_assist_t_.reset();
  return actions;
}

}  // namespace scv_bt_planner
