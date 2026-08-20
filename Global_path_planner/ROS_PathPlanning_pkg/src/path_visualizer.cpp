#include "scv_global_planner/path_visualizer.hpp"

namespace scv_global_planner
{

namespace
{
visualization_msgs::msg::Marker base(const nav_msgs::msg::Path& path,
                                     const builtin_interfaces::msg::Time& stamp,
                                     const char* ns, int id, int32_t type)
{
    visualization_msgs::msg::Marker m;
    m.header.frame_id = path.header.frame_id.empty() ? "map" : path.header.frame_id;
    m.header.stamp = stamp;
    m.ns = ns;
    m.id = id;
    m.type = type;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.orientation.w = 1.0;
    m.color.a = 1.0;
    return m;
}
}  // namespace

visualization_msgs::msg::MarkerArray PathVisualizer::pathMarkers(
    const nav_msgs::msg::Path& path, const builtin_interfaces::msg::Time& stamp)
{
    visualization_msgs::msg::MarkerArray arr;

    auto line = base(path, stamp, "route", 0, visualization_msgs::msg::Marker::LINE_STRIP);
    line.scale.x = 0.35;
    line.color.r = 0.15f;
    line.color.g = 0.85f;
    line.color.b = 1.0f;
    for (const auto& ps : path.poses) {
        line.points.push_back(ps.pose.position);
    }
    // An empty LINE_STRIP is rejected by RViz, so clear instead of publishing one.
    if (line.points.size() < 2) {
        line.action = visualization_msgs::msg::Marker::DELETE;
    }
    arr.markers.push_back(line);

    if (!path.poses.empty()) {
        auto start = base(path, stamp, "route", 1, visualization_msgs::msg::Marker::SPHERE);
        start.pose = path.poses.front().pose;
        start.scale.x = start.scale.y = start.scale.z = 1.2;
        start.color.g = 1.0f;
        arr.markers.push_back(start);

        auto goal = base(path, stamp, "route", 2, visualization_msgs::msg::Marker::SPHERE);
        goal.pose = path.poses.back().pose;
        goal.scale.x = goal.scale.y = goal.scale.z = 1.2;
        goal.color.r = 1.0f;
        goal.color.g = 0.35f;
        arr.markers.push_back(goal);
    }
    return arr;
}

}  // namespace scv_global_planner
