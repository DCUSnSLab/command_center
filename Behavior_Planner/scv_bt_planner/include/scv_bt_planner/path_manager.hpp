// simple_behavior_planner/path_manager.py 의 C++ 이식.
// 동작 동등성이 목적이므로 규칙·기본값·예외 처리를 원본과 같게 유지한다.
// 원본 docstring 의 필드 근거(2026-08-05/06/11)는 path_manager.py 참조.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace scv_bt_planner {

struct PathNode {
  std::string id;
  double x = 0.0;        // 절대 UTM easting (map 프레임 좌표가 아님)
  double y = 0.0;        // 절대 UTM northing
  int node_type = 1;
  double heading_deg = 0.0;  // ENU, 동쪽 0, 반시계 양
};

class PathManager {
public:
  // (x0,y0)->(x1,y1) 직선이 통행 가능한가. nullptr 이면 검사하지 않는다.
  using ApproachClear = std::function<bool(double, double, double, double)>;

  void setPath(std::vector<PathNode> nodes, std::string path_id);
  void clear();

  // 현재 위치를 통과 이력에 반영. 지나온 노드의 최대 인덱스를 돌려준다(-1 = 없음).
  int notePosition(double x, double y, double pass_radius = 2.0);

  // 합류 노드 선택 (접근거리*가중 + 잔여 경로거리 최소). 반환 = current_target_index.
  int alignToPosition(double x, double y,
                      double max_approach_m = 25.0,
                      double approach_weight = 1.5,
                      int max_skip_nodes = 2,
                      int max_skip_from_current = 0,
                      const ApproachClear& approach_clear = nullptr,
                      bool limit_to_passed = true,
                      double pass_radius = 2.0);

  const PathNode* currentTarget() const;
  std::vector<PathNode> nextNodes(int count = 3) const;
  bool advanceToNextNode();          // false = 경로 완주(is_following=false)
  void markGoalCompleted(const std::string& goal_id);

  bool hasPath() const { return !nodes_.empty(); }
  bool isFollowing() const { return is_following_; }
  int currentIndex() const { return current_index_; }
  bool isFinalNode() const { return hasPath() && current_index_ == static_cast<int>(nodes_.size()) - 1; }
  size_t size() const { return nodes_.size(); }
  const std::vector<PathNode>& nodes() const { return nodes_; }
  const std::string& pathId() const { return path_id_; }
  int maxPassedIndex() const { return max_passed_index_; }
  const std::optional<std::string>& lastCompletedGoalId() const { return last_completed_goal_id_; }

private:
  std::vector<PathNode> nodes_;
  std::string path_id_;
  int current_index_ = 0;
  bool is_following_ = false;
  std::optional<std::string> last_completed_goal_id_;
  int max_passed_index_ = -1;
};

}  // namespace scv_bt_planner
