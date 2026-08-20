#ifndef SCV_GLOBAL_PLANNER__ROUTE_GRAPH_HPP_
#define SCV_GLOBAL_PLANNER__ROUTE_GRAPH_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include <geometry_msgs/msg/pose_array.hpp>
#include <map_interfaces/msg/graph_layer.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>

#include "scv_global_planner/graph_types.hpp"

namespace scv_global_planner
{

// The road graph, shared by every planning mode.
//
// Node coordinates are absolute UTM (easting/northing), matching what
// map_provider publishes. Indices are positions in GraphLayer.nodes; the two
// negative ids below are reserved for the temporary start/goal that get spliced
// in for a single query.
class RouteGraph
{
public:
    static constexpr int kTempGoalId = -2;
    static constexpr int kTempStartId = -3;

    explicit RouteGraph(rclcpp::Logger logger);

    // Replace the graph. Returns false if the layer has no usable nodes.
    bool setGraph(const map_interfaces::msg::GraphLayer& layer);

    bool empty() const { return node_map_.empty(); }
    size_t nodeCount() const { return graph_.nodes.size(); }
    size_t linkCount() const { return graph_.links.size(); }

    const map_interfaces::msg::GraphLayer& layer() const { return graph_; }
    const geometry_msgs::msg::PoseArray& nodePoses() const { return node_poses_; }
    const std::vector<std::string>& nodeIds() const { return node_ids_; }
    const std::vector<short>& nodeTypes() const { return node_types_; }
    const std::unordered_map<int, std::shared_ptr<AStarNode>>& nodes() const { return node_map_; }
    const std::unordered_map<int, std::vector<Link>>& adjacency() const { return adjacency_list_; }

    // Neighbours of a node. Empty vector if the node has none.
    const std::vector<Link>& neighbours(int index) const;

    std::shared_ptr<AStarNode> node(int index) const;
    geometry_msgs::msg::Pose poseOf(int index) const;

    // -1 when the id is unknown.
    int indexOf(const std::string& node_id) const;
    // "" when the index is not a real map node (e.g. a temporary one).
    std::string idOf(int index) const;

    // Nearest map node to a point, by straight-line distance. -1 if empty.
    int closestNode(double x, double y) const;

    //: 이보다 멀면 링크에 붙이지 않는다. 도로를 벗어났거나 측위가 틀어진 상태에서
    //: 링크에 스냅하는 건 사실이 아닌 위치를 지어내는 것이다.
    static constexpr double kSnapMaxM = 25.0;
    //: 투영점이 끝점에서 이만큼 안쪽이면 그냥 그 노드를 쓴다 (겹치는 임시 노드 방지).
    static constexpr double kSnapEndpointM = 1.0;

    // Splice a one-off endpoint into the graph, linked both ways to its nearest
    // node so a query can start or finish off-graph. Returns the temp id, or -1.
    //
    // 시작점은 **가장 가까운 링크 위로 정사영**한다. 차량은 노드가 아니라 링크 위에
    // 있는데 가장 가까운 노드에 붙이면, A-B 사이에 있어도 경로가 A 부터 시작해
    // 뒤로 돌아가게 된다. 투영점을 양 끝에 부분 거리로 이으면 A* 가 알아서
    // 앞쪽(B)을 고른다 - 목표가 정말 뒤에 있으면 그때는 A 로 가는 게 맞다.
    //
    // 링크가 kSnapMaxM 보다 멀면 예전처럼 가장 가까운 노드에 붙고 경고를 낸다.
    int addTemporaryStart(double x, double y);
    int addTemporaryGoal(double x, double y);
    void clearTemporaryNodes();

    // True when a direct edge exists (used by SequencePlanner to tell a real
    // link from a gap it has to fill).
    bool hasEdge(int from, int to) const;

private:
    // 점을 링크 선분에 정사영. 가장 가까운 것을 찾으면 true.
    struct EdgeHit {
        int from = -1;
        int to = -1;
        double t = 0.0;        // 선분 위 0~1 위치
        double dist = 0.0;     // 점에서 선분까지 거리
        double length = 0.0;
        bool bidirectional = false;
    };
    bool nearestEdge(double x, double y, EdgeHit& hit) const;

    void buildPoses();
    void buildAdjacency();
    int addTemporary(int temp_id, double x, double y, const char* what);

    rclcpp::Logger logger_;
    map_interfaces::msg::GraphLayer graph_;
    geometry_msgs::msg::PoseArray node_poses_;
    std::vector<std::string> node_ids_;
    std::vector<short> node_types_;
    std::unordered_map<std::string, int> node_id_to_index_;
    std::unordered_map<int, std::shared_ptr<AStarNode>> node_map_;
    std::unordered_map<int, std::vector<Link>> adjacency_list_;
    bool has_temp_start_ = false;
    bool has_temp_goal_ = false;
};

}  // namespace scv_global_planner

#endif  // SCV_GLOBAL_PLANNER__ROUTE_GRAPH_HPP_
