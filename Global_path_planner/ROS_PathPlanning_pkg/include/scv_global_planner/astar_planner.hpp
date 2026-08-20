#ifndef SCV_GLOBAL_PLANNER__ASTAR_PLANNER_HPP_
#define SCV_GLOBAL_PLANNER__ASTAR_PLANNER_HPP_

#include <string>
#include <vector>

#include <rclcpp/logger.hpp>

#include "scv_global_planner/route_planner.hpp"

namespace scv_global_planner
{

// Shortest path by straight-line-heuristic A*. The default mode.
class AStarPlanner : public RoutePlanner
{
public:
    explicit AStarPlanner(rclcpp::Logger logger) : logger_(logger) {}

    std::vector<int> plan(const RouteGraph& graph, int start, int goal,
                          std::string& reason) override;
    const char* name() const override { return "astar"; }

    // Exposed because SequencePlanner uses it to bridge waypoints that are not
    // directly linked.
    static std::vector<int> search(const RouteGraph& graph, int start, int goal);

private:
    rclcpp::Logger logger_;
};

}  // namespace scv_global_planner

#endif  // SCV_GLOBAL_PLANNER__ASTAR_PLANNER_HPP_
