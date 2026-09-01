#ifndef SCV_GLOBAL_PLANNER__GRAPH_TYPES_HPP_
#define SCV_GLOBAL_PLANNER__GRAPH_TYPES_HPP_

#include <memory>

#include <geometry_msgs/msg/pose.hpp>

namespace scv_global_planner
{

// Search node. Carries the pose so planners never have to reach back into the
// graph for geometry, and the cost fields A* fills in.
struct AStarNode
{
    int id;
    geometry_msgs::msg::Pose pose;
    double g_cost;   // cost from start
    double h_cost;   // heuristic to goal
    double f_cost;   // g + h
    int parent_id;   // for path reconstruction

    AStarNode(int node_id, geometry_msgs::msg::Pose node_pose)
        : id(node_id), pose(node_pose), g_cost(0.0), h_cost(0.0), f_cost(0.0), parent_id(-1) {}

    AStarNode(int node_id, geometry_msgs::msg::Pose node_pose, double g, double h, int parent)
        : id(node_id), pose(node_pose), g_cost(g), h_cost(h), f_cost(g + h), parent_id(parent) {}
};

// Min-heap on f_cost for the open set.
struct AStarNodeComparator
{
    bool operator()(const std::shared_ptr<AStarNode>& a, const std::shared_ptr<AStarNode>& b) const
    {
        return a->f_cost > b->f_cost;
    }
};

// Directed edge in the adjacency list. A bidirectional map link produces two of
// these; see RouteGraph::build().
struct Link
{
    int from_node_id;
    int to_node_id;
    double length;

    Link(int from, int to, double len) : from_node_id(from), to_node_id(to), length(len) {}
};

double distanceBetween(const geometry_msgs::msg::Pose& a, const geometry_msgs::msg::Pose& b);

}  // namespace scv_global_planner

#endif  // SCV_GLOBAL_PLANNER__GRAPH_TYPES_HPP_
