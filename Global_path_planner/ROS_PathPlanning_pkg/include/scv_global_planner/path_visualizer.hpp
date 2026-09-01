#ifndef SCV_GLOBAL_PLANNER__PATH_VISUALIZER_HPP_
#define SCV_GLOBAL_PLANNER__PATH_VISUALIZER_HPP_

#include <builtin_interfaces/msg/time.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace scv_global_planner
{

// RViz markers for the planned route.
//
// Deliberately only the route: map_provider already publishes the graph itself
// on ~/graph/markers, and the old in-node graph markers were a duplicate of it
// (they had been commented out for that reason).
class PathVisualizer
{
public:
    // A line through the route plus a marker at each end.
    static visualization_msgs::msg::MarkerArray pathMarkers(
        const nav_msgs::msg::Path& path, const builtin_interfaces::msg::Time& stamp);
};

}  // namespace scv_global_planner

#endif  // SCV_GLOBAL_PLANNER__PATH_VISUALIZER_HPP_
