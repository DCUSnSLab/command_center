#include "scv_global_planner/route_graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scv_global_planner
{

double distanceBetween(const geometry_msgs::msg::Pose& a, const geometry_msgs::msg::Pose& b)
{
    const double dx = a.position.x - b.position.x;
    const double dy = a.position.y - b.position.y;
    const double dz = a.position.z - b.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

RouteGraph::RouteGraph(rclcpp::Logger logger) : logger_(logger) {}

bool RouteGraph::setGraph(const map_interfaces::msg::GraphLayer& layer)
{
    graph_ = layer;
    buildPoses();
    buildAdjacency();
    return !node_map_.empty();
}

void RouteGraph::buildPoses()
{
    node_poses_.poses.clear();
    node_ids_.clear();
    node_types_.clear();
    node_id_to_index_.clear();

    for (size_t i = 0; i < graph_.nodes.size(); ++i) {
        const auto& n = graph_.nodes[i];
        geometry_msgs::msg::Pose pose;
        // Absolute UTM throughout; the node frame is the datum from UtmLayer.
        pose.position.x = n.easting;
        pose.position.y = n.northing;
        pose.position.z = 0.0;
        // heading_deg is ENU (east 0, CCW positive) -> plain z-axis yaw.
        const double yaw = n.heading_deg * M_PI / 180.0;
        pose.orientation.x = 0.0;
        pose.orientation.y = 0.0;
        pose.orientation.z = std::sin(yaw / 2.0);
        pose.orientation.w = std::cos(yaw / 2.0);

        node_poses_.poses.push_back(pose);
        node_ids_.push_back(n.id);
        node_types_.push_back(static_cast<short>(n.node_type));
        node_id_to_index_[n.id] = static_cast<int>(i);
    }
    node_poses_.header.frame_id = "map";
}

void RouteGraph::buildAdjacency()
{
    node_map_.clear();
    adjacency_list_.clear();
    has_temp_start_ = has_temp_goal_ = false;

    for (size_t i = 0; i < graph_.nodes.size(); ++i) {
        node_map_[static_cast<int>(i)] =
            std::make_shared<AStarNode>(static_cast<int>(i), node_poses_.poses[i]);
    }

    size_t bidir = 0;
    for (const auto& link : graph_.links) {
        const int from = indexOf(link.from_node_id);
        const int to = indexOf(link.to_node_id);
        if (from < 0 || to < 0) {
            RCLCPP_WARN(logger_, "Unknown node IDs in link: %s -> %s",
                        link.from_node_id.c_str(), link.to_node_id.c_str());
            continue;
        }
        const double d = (link.length > 0.0)
            ? link.length
            : distanceBetween(node_poses_.poses[from], node_poses_.poses[to]);

        // Forward edge always; reverse only when the map says the link is
        // bidirectional. Adding the reverse unconditionally would make one-way
        // roads impossible to express; dropping it entirely disconnects any
        // hand-drawn graph whose links were not all drawn in travel order.
        // Link(from, to, len): store it the way it reads. The old single-file
        // version passed these swapped and worked around it at every read site
        // ("whichever field is not me is the neighbour"), which is how a plain
        // to_node_id lookup silently returned self-loops.
        adjacency_list_[from].emplace_back(from, to, d);
        if (link.bidirectional) {
            adjacency_list_[to].emplace_back(to, from, d);
            ++bidir;
        }
    }

    RCLCPP_INFO(logger_, "graph: %zu nodes, %zu links (%zu bidirectional), %zu connected",
                graph_.nodes.size(), graph_.links.size(), bidir, adjacency_list_.size());
    if (bidir == 0 && !graph_.links.empty()) {
        RCLCPP_WARN(logger_,
                    "No bidirectional links in this map - every link is one-way. "
                    "If planning fails, check Bidirectional in graph.json.");
    }
    for (size_t i = 0; i < graph_.nodes.size(); ++i) {
        if (adjacency_list_.find(static_cast<int>(i)) == adjacency_list_.end()) {
            RCLCPP_WARN(logger_, "Node %zu (%s) is isolated (no outgoing links)",
                        i, graph_.nodes[i].id.c_str());
        }
    }
}

const std::vector<Link>& RouteGraph::neighbours(int index) const
{
    static const std::vector<Link> kNone;
    auto it = adjacency_list_.find(index);
    return (it == adjacency_list_.end()) ? kNone : it->second;
}

std::shared_ptr<AStarNode> RouteGraph::node(int index) const
{
    auto it = node_map_.find(index);
    return (it == node_map_.end()) ? nullptr : it->second;
}

geometry_msgs::msg::Pose RouteGraph::poseOf(int index) const
{
    auto n = node(index);
    return n ? n->pose : geometry_msgs::msg::Pose();
}

int RouteGraph::indexOf(const std::string& node_id) const
{
    auto it = node_id_to_index_.find(node_id);
    return (it == node_id_to_index_.end()) ? -1 : it->second;
}

std::string RouteGraph::idOf(int index) const
{
    if (index < 0 || index >= static_cast<int>(node_ids_.size())) {
        return std::string();
    }
    return node_ids_[index];
}

int RouteGraph::closestNode(double x, double y) const
{
    int best = -1;
    double best_d = std::numeric_limits<double>::max();
    for (size_t i = 0; i < node_poses_.poses.size(); ++i) {
        const double dx = node_poses_.poses[i].position.x - x;
        const double dy = node_poses_.poses[i].position.y - y;
        const double d = std::sqrt(dx * dx + dy * dy);
        if (d < best_d) {
            best_d = d;
            best = static_cast<int>(i);
        }
    }
    return best;
}

int RouteGraph::addTemporary(int temp_id, double x, double y, const char* what)
{
    geometry_msgs::msg::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = 0.0;
    pose.orientation.w = 1.0;

    const int closest = closestNode(x, y);
    if (closest < 0) {
        RCLCPP_ERROR(logger_, "No map nodes to attach temporary %s node to", what);
        return -1;
    }
    node_map_[temp_id] = std::make_shared<AStarNode>(temp_id, pose);
    const double d = distanceBetween(node_poses_.poses[closest], pose);
    // Always both ways: this is a query artefact, not a road, so it must never
    // be the reason a route is unreachable.
    adjacency_list_[temp_id].emplace_back(temp_id, closest, d);
    adjacency_list_[closest].emplace_back(closest, temp_id, d);

    RCLCPP_DEBUG(logger_, "temporary %s node %d at (%.2f, %.2f) -> node %d (%.2f m)",
                 what, temp_id, x, y, closest, d);
    return temp_id;
}

bool RouteGraph::nearestEdge(double x, double y, EdgeHit& hit) const
{
    bool found = false;
    double best = std::numeric_limits<double>::max();
    for (const auto& entry : adjacency_list_) {
        const int from = entry.first;
        if (from < 0) {
            continue;                       // 임시 노드가 만든 간선은 제외
        }
        for (const auto& link : entry.second) {
            const int to = link.to_node_id;
            if (to < 0 || from == to) {
                continue;
            }
            const auto& a = node_poses_.poses[from].position;
            const auto& b = node_poses_.poses[to].position;
            const double dx = b.x - a.x;
            const double dy = b.y - a.y;
            const double l2 = dx * dx + dy * dy;
            if (l2 < 1e-9) {
                continue;
            }
            double t = ((x - a.x) * dx + (y - a.y) * dy) / l2;
            t = std::max(0.0, std::min(1.0, t));
            const double px = a.x + t * dx;
            const double py = a.y + t * dy;
            const double d = std::hypot(x - px, y - py);
            if (d < best) {
                best = d;
                hit.from = from;
                hit.to = to;
                hit.t = t;
                hit.dist = d;
                hit.length = std::sqrt(l2);
                found = true;
            }
        }
    }
    if (found) {
        hit.bidirectional = hasEdge(hit.to, hit.from);
    }
    return found;
}

int RouteGraph::addTemporaryStart(double x, double y)
{
    EdgeHit hit;
    if (!nearestEdge(x, y, hit) || hit.dist > kSnapMaxM) {
        if (hit.from >= 0) {
            RCLCPP_WARN(logger_,
                        "vehicle is %.1f m from the nearest link (limit %.1f) - "
                        "falling back to the nearest node; check localisation",
                        hit.dist, kSnapMaxM);
        }
        const int id = addTemporary(kTempStartId, x, y, "start");
        has_temp_start_ = (id == kTempStartId);
        return id;
    }

    // 끝점에 거의 붙어 있으면 그 노드를 그대로 쓴다.
    const double from_dist = hit.t * hit.length;
    const double to_dist = (1.0 - hit.t) * hit.length;
    if (from_dist < kSnapEndpointM || to_dist < kSnapEndpointM) {
        const int id = addTemporary(kTempStartId, x, y, "start");
        has_temp_start_ = (id == kTempStartId);
        return id;
    }

    const auto& a = node_poses_.poses[hit.from].position;
    const auto& b = node_poses_.poses[hit.to].position;
    geometry_msgs::msg::Pose pose;
    pose.position.x = a.x + hit.t * (b.x - a.x);
    pose.position.y = a.y + hit.t * (b.y - a.y);
    pose.position.z = 0.0;
    pose.orientation.w = 1.0;
    node_map_[kTempStartId] = std::make_shared<AStarNode>(kTempStartId, pose);

    // 나가는 간선만 만든다 (출발점이므로 들어올 일이 없다).
    // 진행 방향은 항상 열고, 뒤로는 그 링크가 양방향일 때만 - 일방통행을 거슬러
    // 출발하게 만들면 안 된다.
    adjacency_list_[kTempStartId].emplace_back(kTempStartId, hit.to, to_dist);
    if (hit.bidirectional) {
        adjacency_list_[kTempStartId].emplace_back(kTempStartId, hit.from, from_dist);
    }
    has_temp_start_ = true;

    RCLCPP_INFO(logger_,
                "start snapped onto link %s->%s at %.0f%% (%.1f m off), "
                "forward %.1f m / back %.1f m%s",
                idOf(hit.from).c_str(), idOf(hit.to).c_str(), hit.t * 100.0, hit.dist,
                to_dist, from_dist, hit.bidirectional ? "" : " (one-way: forward only)");
    return kTempStartId;
}

int RouteGraph::addTemporaryGoal(double x, double y)
{
    const int id = addTemporary(kTempGoalId, x, y, "goal");
    has_temp_goal_ = (id >= 0 || id == kTempGoalId);
    return id;
}

void RouteGraph::clearTemporaryNodes()
{
    for (int temp_id : {kTempStartId, kTempGoalId}) {
        node_map_.erase(temp_id);
        adjacency_list_.erase(temp_id);
        // Drop the back-links the real nodes gained.
        for (auto& entry : adjacency_list_) {
            auto& v = entry.second;
            v.erase(std::remove_if(v.begin(), v.end(),
                                   [temp_id](const Link& l) { return l.to_node_id == temp_id; }),
                    v.end());
        }
    }
    has_temp_start_ = has_temp_goal_ = false;
}

bool RouteGraph::hasEdge(int from, int to) const
{
    for (const auto& l : neighbours(from)) {
        if (l.to_node_id == to) {
            return true;
        }
    }
    return false;
}

}  // namespace scv_global_planner
