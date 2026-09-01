#include "scv_global_planner/sequence_planner.hpp"

#include "scv_global_planner/astar_planner.hpp"

namespace scv_global_planner
{

std::vector<int> SequencePlanner::resolveRoute(const RouteGraph& graph, std::string& reason) const
{
    std::vector<int> idx;
    std::string missing;
    for (const auto& id : route_nodes_) {
        const int i = graph.indexOf(id);
        if (i < 0) {
            missing += (missing.empty() ? "" : ", ") + id;
            continue;
        }
        // Skip a repeat of the previous waypoint: it contributes no travel and
        // would look like a zero-length gap later.
        if (!idx.empty() && idx.back() == i) {
            continue;
        }
        idx.push_back(i);
    }
    if (!missing.empty()) {
        reason = "route_nodes not in this map: " + missing;
        return {};
    }
    return idx;
}

std::vector<int> SequencePlanner::plan(const RouteGraph& graph, int start, int goal,
                                       std::string& reason)
{
    if (route_nodes_.empty()) {
        reason = "route_mode is 'sequence' but route_nodes is empty";
        return {};
    }

    std::vector<int> way = resolveRoute(graph, reason);
    if (way.empty()) {
        if (reason.empty()) {
            reason = "route_nodes resolved to nothing";
        }
        return {};
    }
    if (loop_ && way.size() > 1 && way.front() != way.back()) {
        way.push_back(way.front());
    }
    // Drive on from where the robot actually is, and finish where asked, so the
    // fixed route still connects to the current pose and the commanded goal.
    if (start >= 0 && start != way.front()) {
        way.insert(way.begin(), start);
    }
    if (!loop_ && goal >= 0 && goal != way.back()) {
        way.push_back(goal);
    }

    std::vector<int> path;
    path.push_back(way.front());
    for (size_t i = 1; i < way.size(); ++i) {
        const int a = way[i - 1];
        const int b = way[i];
        if (a == b) {
            continue;
        }
        if (graph.hasEdge(a, b)) {
            path.push_back(b);
            continue;
        }
        if (!fill_gaps_) {
            reason = "no direct link " + graph.idOf(a) + " -> " + graph.idOf(b) +
                     " and route_fill_gaps is false";
            return {};
        }
        auto bridge = AStarPlanner::search(graph, a, b);
        if (bridge.empty()) {
            reason = "cannot reach " + graph.idOf(b) + " from " + graph.idOf(a) +
                     " (check link direction / Bidirectional)";
            return {};
        }
        RCLCPP_DEBUG(logger_, "filled gap %s -> %s with %zu nodes",
                     graph.idOf(a).c_str(), graph.idOf(b).c_str(), bridge.size());
        path.insert(path.end(), bridge.begin() + 1, bridge.end());
    }

    RCLCPP_INFO(logger_, "sequence route: %zu waypoints -> %zu nodes%s",
                way.size(), path.size(), loop_ ? " (loop closed)" : "");
    return path;
}

}  // namespace scv_global_planner
