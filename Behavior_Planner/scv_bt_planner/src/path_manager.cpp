#include "scv_bt_planner/path_manager.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scv_bt_planner {

void PathManager::setPath(std::vector<PathNode> nodes, std::string path_id)
{
  nodes_ = std::move(nodes);
  path_id_ = std::move(path_id);
  current_index_ = 0;
  is_following_ = true;
  last_completed_goal_id_.reset();
  max_passed_index_ = -1;
}

void PathManager::clear()
{
  nodes_.clear();
  path_id_.clear();
  current_index_ = 0;
  is_following_ = false;
  last_completed_goal_id_.reset();
  max_passed_index_ = -1;
}

int PathManager::notePosition(double x, double y, double pass_radius)
{
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (static_cast<int>(i) <= max_passed_index_) continue;
    if (std::hypot(nodes_[i].x - x, nodes_[i].y - y) <= pass_radius) {
      max_passed_index_ = static_cast<int>(i);
    }
  }
  return max_passed_index_;
}

int PathManager::alignToPosition(double x, double y, double max_approach_m,
                                 double approach_weight, int max_skip_nodes,
                                 int max_skip_from_current,
                                 const ApproachClear& approach_clear,
                                 bool limit_to_passed, double pass_radius)
{
  if (nodes_.empty()) return 0;
  const int n = static_cast<int>(nodes_.size());
  if (limit_to_passed) notePosition(x, y, pass_radius);

  // 각 노드에서 경로 끝까지 남은 거리
  std::vector<double> rem(n, 0.0);
  for (int i = n - 2; i >= 0; --i) {
    const auto& a = nodes_[i];
    const auto& b = nodes_[i + 1];
    rem[i] = rem[i + 1] + std::hypot(b.x - a.x, b.y - a.y);
  }

  // 최근접 노드 (python: d2.index(min(d2)) — 동률이면 앞 인덱스)
  int i_near = 0;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    const double dx = nodes_[i].x - x, dy = nodes_[i].y - y;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) { best_d2 = d2; i_near = i; }
  }
  int i_max = std::min(n - 1, i_near + max_skip_nodes);
  int i_lo = i_near;
  const bool gate = limit_to_passed && max_passed_index_ >= 0;
  if (gate) {
    i_max = std::min(i_max, max_passed_index_ + 1);
    i_lo = std::min(i_lo, i_max);
  }
  if (max_skip_from_current > 0) {
    i_lo = std::min(i_lo, current_index_);
    i_max = std::min(i_max, current_index_ + max_skip_from_current);
  }

  const bool use_clear = static_cast<bool>(approach_clear);
  auto pick = [&](int lo, int hi, bool clear_check) -> std::optional<int> {
    std::optional<int> bi;
    double bc = 0.0;
    for (int i = lo; i <= hi; ++i) {
      const auto& node = nodes_[i];
      const double d = std::hypot(node.x - x, node.y - y);
      if (d > max_approach_m) continue;
      if (clear_check && !approach_clear(x, y, node.x, node.y)) continue;
      const double cost = approach_weight * d + rem[i];
      if (!bi || cost < bc) { bi = i; bc = cost; }
    }
    return bi;
  };

  std::optional<int> best = pick(i_lo, i_max, use_clear);
  if (!best && use_clear) {
    best = pick(0, n - 1, true);
  }
  if (!best) {
    best = gate ? std::min(i_near, max_passed_index_ + 1) : i_near;
  }
  current_index_ = *best;
  return current_index_;
}

const PathNode* PathManager::currentTarget() const
{
  if (nodes_.empty() || current_index_ >= static_cast<int>(nodes_.size())) return nullptr;
  return &nodes_[current_index_];
}

std::vector<PathNode> PathManager::nextNodes(int count) const
{
  std::vector<PathNode> out;
  if (nodes_.empty()) return out;
  const int n = static_cast<int>(nodes_.size());
  const int limit = std::min(count + 1, n - current_index_);
  for (int i = 1; i < limit; ++i) {
    const int idx = current_index_ + i;
    if (idx < n) out.push_back(nodes_[idx]);
  }
  return out;
}

bool PathManager::advanceToNextNode()
{
  if (current_index_ < static_cast<int>(nodes_.size()) - 1) {
    ++current_index_;
    return true;
  }
  is_following_ = false;
  return false;
}

void PathManager::markGoalCompleted(const std::string& goal_id)
{
  last_completed_goal_id_ = goal_id;
}

}  // namespace scv_bt_planner
