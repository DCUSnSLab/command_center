#ifndef SCV_GLOBAL_PLANNER__SEQUENCE_PLANNER_HPP_
#define SCV_GLOBAL_PLANNER__SEQUENCE_PLANNER_HPP_

#include <string>
#include <vector>

#include <rclcpp/logger.hpp>

#include "scv_global_planner/route_planner.hpp"

namespace scv_global_planner
{

// Follows a route the operator wrote out, instead of the shortest one.
//
// Why this exists: A* always cuts the corner. A patrol loop is defined by the
// way round, not by length, and a loop cannot be expressed as one A* query at
// all because start == goal. Listing the nodes states the intent directly.
//
// fill_gaps (default on): consecutive waypoints that share no direct link get
// bridged by A*, so only the junctions where the route could go two ways have
// to be listed. Turn it off to demand that every step be a real link - useful
// when validating that a route is exactly what was drawn.
//
// loop: closes the ring by appending the first waypoint. The node that drives
// this planner re-issues the route when the robot reaches the end.
class SequencePlanner : public RoutePlanner
{
public:
    SequencePlanner(rclcpp::Logger logger, std::vector<std::string> route_nodes,
                    bool loop, bool fill_gaps)
        : logger_(logger), route_nodes_(std::move(route_nodes)),
          loop_(loop), fill_gaps_(fill_gaps) {}

    std::vector<int> plan(const RouteGraph& graph, int start, int goal,
                          std::string& reason) override;
    const char* name() const override { return "sequence"; }

    void setRoute(std::vector<std::string> route_nodes) { route_nodes_ = std::move(route_nodes); }
    const std::vector<std::string>& route() const { return route_nodes_; }

private:
    // Resolve the configured ids to indices; reports every id it cannot find.
    std::vector<int> resolveRoute(const RouteGraph& graph, std::string& reason) const;

    rclcpp::Logger logger_;
    std::vector<std::string> route_nodes_;
    bool loop_;
    bool fill_gaps_;
};

}  // namespace scv_global_planner

#endif  // SCV_GLOBAL_PLANNER__SEQUENCE_PLANNER_HPP_
