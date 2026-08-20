#include "scv_global_planner/astar_planner.hpp"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace scv_global_planner
{

std::vector<int> AStarPlanner::search(const RouteGraph& graph, int start, int goal)
{
    std::vector<int> out;
    if (!graph.node(start) || !graph.node(goal)) {
        return out;
    }
    if (start == goal) {
        out.push_back(start);
        return out;
    }

    const auto goal_pose = graph.poseOf(goal);

    std::priority_queue<std::shared_ptr<AStarNode>,
                        std::vector<std::shared_ptr<AStarNode>>,
                        AStarNodeComparator> open;
    std::unordered_map<int, double> best_g;
    std::unordered_map<int, int> came_from;
    std::unordered_set<int> closed;

    auto s = std::make_shared<AStarNode>(
        start, graph.poseOf(start), 0.0,
        distanceBetween(graph.poseOf(start), goal_pose), -1);
    open.push(s);
    best_g[start] = 0.0;

    while (!open.empty()) {
        auto cur = open.top();
        open.pop();
        if (closed.count(cur->id)) {
            continue;
        }
        closed.insert(cur->id);

        if (cur->id == goal) {
            for (int at = goal; at != -1; at = (came_from.count(at) ? came_from[at] : -1)) {
                out.push_back(at);
                if (at == start) {
                    break;
                }
            }
            std::reverse(out.begin(), out.end());
            return out;
        }

        for (const auto& link : graph.neighbours(cur->id)) {
            const int nxt = link.to_node_id;
            if (closed.count(nxt)) {
                continue;
            }
            const double g = cur->g_cost + link.length;
            auto it = best_g.find(nxt);
            if (it != best_g.end() && g >= it->second) {
                continue;
            }
            best_g[nxt] = g;
            came_from[nxt] = cur->id;
            open.push(std::make_shared<AStarNode>(
                nxt, graph.poseOf(nxt), g,
                distanceBetween(graph.poseOf(nxt), goal_pose), cur->id));
        }
    }
    return out;   // unreachable
}

std::vector<int> AStarPlanner::plan(const RouteGraph& graph, int start, int goal,
                                    std::string& reason)
{
    auto path = search(graph, start, goal);
    if (path.empty()) {
        reason = "no A* path from " + std::to_string(start) + " to " + std::to_string(goal) +
                 " (graph disconnected in the travel direction?)";
    }
    return path;
}

}  // namespace scv_global_planner
