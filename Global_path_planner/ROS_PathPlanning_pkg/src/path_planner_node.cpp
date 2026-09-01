// Global path planner node.
//
// ROS wiring only: the graph lives in RouteGraph and the search strategy behind
// the RoutePlanner interface, chosen once from the route_mode parameter.
//
//   route_mode: astar     shortest path (default, unchanged behaviour)
//   route_mode: sequence  follow route_nodes in order; see SequencePlanner
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <map_interfaces/msg/graph_layer.hpp>
#include <map_interfaces/msg/map_node.hpp>
#include <map_interfaces/msg/map_link.hpp>
#include <map_interfaces/msg/utm_layer.hpp>
#include <command_center_interfaces/msg/planned_path.hpp>
#include <command_center_interfaces/msg/route_request.hpp>
#include <command_center_interfaces/msg/route_status.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "scv_global_planner/astar_planner.hpp"
#include "scv_global_planner/path_visualizer.hpp"
#include "scv_global_planner/route_graph.hpp"
#include "scv_global_planner/sequence_planner.hpp"

using namespace scv_global_planner;

class PathPlannerNode : public rclcpp::Node
{
public:
    PathPlannerNode()
    : Node("global_path_planner_node"), graph_(rclcpp::get_logger("route_graph"))
    {
        goal_received_ = false;
        path_planned_for_current_goal_ = false;
        datum_initialized_ = false;
        graph_received_ = false;

        route_mode_ = this->declare_parameter<std::string>("route_mode", "astar");
        // A comma-separated string, not a string array: ROS cannot infer the
        // element type of an empty array override, so "route_nodes:=[]" - the
        // natural default for a launch file - aborts the node at startup.
        route_nodes_ = splitIds(this->declare_parameter<std::string>("route_nodes", ""));
        route_loop_ = this->declare_parameter<bool>("route_loop", false);
        route_fill_gaps_ = this->declare_parameter<bool>("route_fill_gaps", true);
        makePlanner();

        graph_sub_ = this->create_subscription<map_interfaces::msg::GraphLayer>(
            "/map_provider_node/graph", rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&PathPlannerNode::graphCallback, this, std::placeholders::_1));
        utm_layer_sub_ = this->create_subscription<map_interfaces::msg::UtmLayer>(
            "/map_provider_node/utm", rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&PathPlannerNode::utmLayerCallback, this, std::placeholders::_1));
        // 경로 요청 (latched): 재시작해도 지금 유효한 경로를 다시 받는다.
        route_subscriber_ = this->create_subscription<command_center_interfaces::msg::RouteRequest>(
            "route_request", rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&PathPlannerNode::routeRequestCallback, this, std::placeholders::_1));

        goal_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "goal_pose", 10,
            std::bind(&PathPlannerNode::goalCallback, this, std::placeholders::_1));

        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("planned_path", 10);
        planned_path_publisher_ =
            this->create_publisher<command_center_interfaces::msg::PlannedPath>(
                "planned_path_detailed", 10);
        route_status_publisher_ = this->create_publisher<command_center_interfaces::msg::RouteStatus>(
            "route_status", rclcpp::QoS(10).transient_local());
        path_viz_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "planned_path_markers", rclcpp::QoS(1).transient_local());

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

        timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&PathPlannerNode::checkAndPlanPath, this));

        RCLCPP_INFO(this->get_logger(), "global path planner ready (mode=%s)",
                    planner_ ? planner_->name() : "none");
    }

private:
    // "N0394, N0426 NX0011" -> {"N0394","N0426","NX0011"}. Commas or spaces.
    static std::vector<std::string> splitIds(const std::string& csv)
    {
        std::vector<std::string> out;
        std::string cur;
        for (char c : csv) {
            if (c == ',' || c == ' ' || c == '\t' || c == '[' || c == ']' || c == '"') {
                if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) { out.push_back(cur); }
        return out;
    }

    void makePlanner()
    {
        if (route_mode_ == "sequence") {
            planner_ = std::make_unique<SequencePlanner>(
                this->get_logger(), route_nodes_, route_loop_, route_fill_gaps_);
            RCLCPP_INFO(this->get_logger(),
                        "sequence mode: %zu route_nodes, loop=%s, fill_gaps=%s",
                        route_nodes_.size(), route_loop_ ? "true" : "false",
                        route_fill_gaps_ ? "true" : "false");
            if (route_nodes_.empty()) {
                RCLCPP_WARN(this->get_logger(),
                            "route_mode is 'sequence' but route_nodes is empty - nothing to follow");
            }
        } else {
            if (route_mode_ != "astar") {
                RCLCPP_WARN(this->get_logger(),
                            "unknown route_mode '%s', falling back to 'astar'", route_mode_.c_str());
            }
            planner_ = std::make_unique<AStarPlanner>(this->get_logger());
        }
    }

    // map_provider graph layer (latched) -> rebuild the graph
    void graphCallback(const map_interfaces::msg::GraphLayer::SharedPtr msg)
    {
        // Rebuild only when the map actually changed. A latched topic re-delivers
        // on every new match, and a stray second map_provider makes that a loop -
        // without this the whole graph is rebuilt several times a second.
        const size_t sig = msg->nodes.size() * 1000003u + msg->links.size();
        if (graph_received_ && sig == graph_signature_) {
            return;
        }
        graph_signature_ = sig;
        graph_received_ = graph_.setGraph(*msg);
        // A new map invalidates whatever we planned - and may make a previously
        // impossible route possible, so clear the failure latch too.
        path_planned_for_current_goal_ = false;
        request_failed_ = false;
    }

    // map_provider datum (UtmLayer, latched) -> map frame origin in UTM
    void utmLayerCallback(const map_interfaces::msg::UtmLayer::SharedPtr msg)
    {
        datum_easting_ = msg->origin_easting;
        datum_northing_ = msg->origin_northing;
        utm_zone_ = msg->utm_zone;
        northern_ = msg->northern;
        datum_initialized_ = true;
        RCLCPP_DEBUG(this->get_logger(), "datum: zone=%d %s e=%.3f n=%.3f",
                     static_cast<int>(utm_zone_), northern_ ? "north" : "south",
                     datum_easting_, datum_northing_);
    }

    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (!datum_initialized_ || !graph_received_) {
            RCLCPP_WARN(this->get_logger(), "Goal received, but datum/graph not initialized yet. Waiting for map_provider.");
            return;
        }

        // Transform goal to map frame based on frame_id
        geometry_msgs::msg::PoseStamped goal_in_map_frame = *msg;

        RCLCPP_DEBUG(this->get_logger(), "datum east %f", datum_easting_);
        RCLCPP_DEBUG(this->get_logger(), "datum north %f", datum_northing_);

        if (msg->header.frame_id == "map") {
            RCLCPP_DEBUG(this->get_logger(), "frame_id : map");
            // Goal is already in map frame, convert to absolute UTM coordinates
            goal_pose_ = *msg;
            goal_pose_.pose.position.x += datum_easting_;
            goal_pose_.pose.position.y += datum_northing_;
            
            RCLCPP_DEBUG(this->get_logger(), 
                       "Goal received in map frame - Relative: (%.2f, %.2f) -> Absolute UTM: (%.2f, %.2f)", 
                       msg->pose.position.x, msg->pose.position.y,
                       goal_pose_.pose.position.x, goal_pose_.pose.position.y);
        } 
        else if (msg->header.frame_id == "odom") {
            RCLCPP_DEBUG(this->get_logger(), "frame_id : odom");
            // Goal is in odom frame, transform to map frame first
            try {
                // Transform from odom to map frame
                geometry_msgs::msg::PoseStamped goal_in_map;
                // tf_buffer_->transform(*msg, goal_in_map, "map", tf2::durationFromSec(1.0));
                
                RCLCPP_DEBUG(this->get_logger(), "transform output x %f", goal_pose_.pose.position.x);
                RCLCPP_DEBUG(this->get_logger(), "transform output y %f", goal_pose_.pose.position.y);
                
                // Convert to absolute UTM coordinates
                // goal_pose_ = goal_in_map;
                goal_pose_ = *msg;
                goal_pose_.pose.position.x += datum_easting_;
                goal_pose_.pose.position.y += datum_northing_;
                
                RCLCPP_DEBUG(this->get_logger(), 
                           "Goal received in odom frame - Odom: (%.2f, %.2f) -> Map: (%.2f, %.2f) -> Absolute UTM: (%.2f, %.2f)", 
                           msg->pose.position.x, msg->pose.position.y,
                           goal_in_map.pose.position.x, goal_in_map.pose.position.y,
                           goal_pose_.pose.position.x, goal_pose_.pose.position.y);
            }
            catch (const tf2::TransformException& ex) {
                RCLCPP_ERROR(this->get_logger(), 
                           "Failed to transform goal from odom to map frame: %s", ex.what());
                return;
            }
        }
        else {
            RCLCPP_DEBUG(this->get_logger(), "frame_id : none");
            // Unsupported frame_id, try to transform to map frame
            try {
                geometry_msgs::msg::PoseStamped goal_in_map;
                tf_buffer_->transform(*msg, goal_in_map, "map", tf2::durationFromSec(1.0));
                
                // Convert to absolute UTM coordinates
                goal_pose_ = goal_in_map;
                goal_pose_.pose.position.x += datum_easting_;
                goal_pose_.pose.position.y += datum_northing_;
                
                RCLCPP_DEBUG(this->get_logger(), 
                           "Goal received in %s frame - Transformed to Map: (%.2f, %.2f) -> Absolute UTM: (%.2f, %.2f)", 
                           msg->header.frame_id.c_str(),
                           goal_in_map.pose.position.x, goal_in_map.pose.position.y,
                           goal_pose_.pose.position.x, goal_pose_.pose.position.y);
            }
            catch (const tf2::TransformException& ex) {
                RCLCPP_ERROR(this->get_logger(), 
                           "Failed to transform goal from %s to map frame: %s. Treating as map frame.", 
                           msg->header.frame_id.c_str(), ex.what());
                
                // Fallback: treat as map frame
                goal_pose_ = *msg;
                goal_pose_.pose.position.x += datum_easting_;
                goal_pose_.pose.position.y += datum_northing_;
            }
        }
        RCLCPP_DEBUG(this->get_logger(), "goalCallback goal pose x %f", goal_pose_.pose.position.x);
        RCLCPP_DEBUG(this->get_logger(), "goalCallback goal pose y %f", goal_pose_.pose.position.y);
        
        goal_received_ = true;
        path_planned_for_current_goal_ = false; // Reset flag for new goal
        request_failed_ = false;
        
        // Trigger immediate path planning
        planRoute();
    }



    command_center_interfaces::msg::PlannedPath createDetailedPlannedPath(
        const std::vector<std::shared_ptr<AStarNode>>& path_nodes,
        int start_node_id, int goal_node_id)
    {
        command_center_interfaces::msg::PlannedPath detailed_path;

        // Set header
        detailed_path.header.frame_id = "map";
        detailed_path.header.stamp = this->get_clock()->now();
        
        // Set path metadata
        detailed_path.path_id = "path_" + std::to_string(this->get_clock()->now().nanoseconds());
        detailed_path.start_node_id = (start_node_id == RouteGraph::kTempStartId) ? "GPS_START" : 
                                     (start_node_id < static_cast<int>(graph_.nodeIds().size()) ? graph_.nodeIds()[start_node_id] : "UNKNOWN");
        detailed_path.goal_node_id = (goal_node_id == RouteGraph::kTempGoalId) ? "GPS_GOAL" : 
                                    (goal_node_id < static_cast<int>(graph_.nodeIds().size()) ? graph_.nodeIds()[goal_node_id] : "UNKNOWN");
        
        // Calculate total distance
        double total_distance = 0.0;
        for (size_t i = 1; i < path_nodes.size(); ++i) {
            total_distance += distanceBetween(path_nodes[i-1]->pose, path_nodes[i]->pose);
        }
        detailed_path.total_distance = total_distance;
        detailed_path.total_time = total_distance / 10.0; // 평균 속도 10m/s 가정
        
        // Convert path nodes to MapNode messages (map_interfaces, 절대 UTM 유지)
        detailed_path.path_data.nodes.clear();
        for (size_t i = 0; i < path_nodes.size(); ++i) {
            map_interfaces::msg::MapNode map_node;

            int node_idx = path_nodes[i]->id;

            // Temporary nodes에 대한 처리
            if (node_idx == RouteGraph::kTempStartId) {
                map_node.id = "GPS_START";
                map_node.source = "gps";
            } else if (node_idx == RouteGraph::kTempGoalId) {
                map_node.id = "GPS_GOAL";
                map_node.source = "gps";
            } else if (node_idx >= 0 && node_idx < static_cast<int>(graph_.layer().nodes.size())) {
                // 실제 맵 노드에서 정보 복사 (id/node_type/easting/northing/lat/lon/heading_deg/source)
                map_node = graph_.layer().nodes[node_idx];
            } else {
                // Fallback for unknown nodes
                map_node.id = "NODE_" + std::to_string(node_idx);
            }

            if (node_idx >= 0 && node_idx < static_cast<int>(graph_.layer().nodes.size()) &&
                node_idx != RouteGraph::kTempStartId && node_idx != RouteGraph::kTempGoalId) {
                // 실제 맵 노드: 원본 절대 UTM/GPS/heading 유지 (datum 빼지 않음)
                map_node.easting = graph_.layer().nodes[node_idx].easting;
                map_node.northing = graph_.layer().nodes[node_idx].northing;
                map_node.latitude = graph_.layer().nodes[node_idx].latitude;
                map_node.longitude = graph_.layer().nodes[node_idx].longitude;
                map_node.heading_deg = graph_.layer().nodes[node_idx].heading_deg;
                // If heading is -1.0, calculate from previous node in path
                if (std::abs(map_node.heading_deg + 1.0) < 1e-6 && i > 0) {
                    double dx = path_nodes[i]->pose.position.x - path_nodes[i-1]->pose.position.x;
                    double dy = path_nodes[i]->pose.position.y - path_nodes[i-1]->pose.position.y;
                    double heading_rad = std::atan2(dy, dx);
                    double heading_deg = heading_rad * 180.0 / M_PI;
                    if (heading_deg < 0) {
                        heading_deg += 360.0;
                    }
                    map_node.heading_deg = heading_deg;
                }
            } else {
                // Temporary 노드: pose(절대 UTM)에서 값 채움
                map_node.easting = path_nodes[i]->pose.position.x;
                map_node.northing = path_nodes[i]->pose.position.y;
                map_node.latitude = 0.0; // GPS 역변환은 생략
                map_node.longitude = 0.0;

                // Calculate heading from previous node if available
                if (i > 0) {
                    double dx = path_nodes[i]->pose.position.x - path_nodes[i-1]->pose.position.x;
                    double dy = path_nodes[i]->pose.position.y - path_nodes[i-1]->pose.position.y;
                    double heading_rad = std::atan2(dy, dx);
                    double heading_deg = heading_rad * 180.0 / M_PI;
                    if (heading_deg < 0) {
                        heading_deg += 360.0;
                    }
                    map_node.heading_deg = heading_deg;
                } else {
                    map_node.heading_deg = 0.0;
                }
            }

            detailed_path.path_data.nodes.push_back(map_node);
        }

        // Create links between consecutive path nodes
        detailed_path.path_data.links.clear();
        for (size_t i = 1; i < path_nodes.size(); ++i) {
            map_interfaces::msg::MapLink map_link;

            int from_node_idx = path_nodes[i-1]->id;
            int to_node_idx = path_nodes[i]->id;

            // Set link metadata
            map_link.id = "PATH_LINK_" + std::to_string(i-1) + "_" + std::to_string(i);
            map_link.from_node_id = detailed_path.path_data.nodes[i-1].id;
            map_link.to_node_id = detailed_path.path_data.nodes[i].id;

            // Calculate link length (meters)
            double distance = distanceBetween(path_nodes[i-1]->pose, path_nodes[i]->pose);
            map_link.length = distance;

            // Try to find existing link in graph to reuse its length
            if (from_node_idx >= 0 && from_node_idx < static_cast<int>(graph_.layer().nodes.size()) &&
                to_node_idx >= 0 && to_node_idx < static_cast<int>(graph_.layer().nodes.size()) &&
                from_node_idx != RouteGraph::kTempStartId && from_node_idx != RouteGraph::kTempGoalId &&
                to_node_idx != RouteGraph::kTempStartId && to_node_idx != RouteGraph::kTempGoalId) {

                std::string from_id = graph_.layer().nodes[from_node_idx].id;
                std::string to_id = graph_.layer().nodes[to_node_idx].id;

                // Find existing link in GraphLayer
                for (const auto& original_link : graph_.layer().links) {
                    if ((original_link.from_node_id == from_id && original_link.to_node_id == to_id) ||
                        (original_link.from_node_id == to_id && original_link.to_node_id == from_id)) {
                        map_link.id = original_link.id;
                        map_link.length = original_link.length;
                        map_link.from_node_id = from_id;
                        map_link.to_node_id = to_id;
                        break;
                    }
                }
            }

            detailed_path.path_data.links.push_back(map_link);
        }
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "Created detailed path: %zu nodes, %zu links, total distance: %.2f m",
                    detailed_path.path_data.nodes.size(), detailed_path.path_data.links.size(), total_distance);
        
        return detailed_path;
    }
    

    // 현재 ROS 버전에서 특정 메시지형의 doTransform 미지원으로 인한 변환 함수 구현.
    // 다른 코드에서도 쓸 수 있게 향후 tools 같은 디렉토리에 별도 클래스로 작성하는게 좋긴함

    // 요청 내용이 방식을 정한다: via_nodes 가 있으면 sequence, 없으면 astar.
    // map 프레임(datum 상대) -> 절대 UTM. 그래프 노드가 절대 UTM 이라 반드시 거쳐야
    // 한다. goalCallback 과 routeRequestCallback 이 각자 변환하다 한쪽이 빠지면
    // 목표가 지도 밖으로 날아가고, 가장 가까운 노드로 남서쪽 끝이 잡힌다.
    geometry_msgs::msg::PoseStamped toAbsoluteUtm(
        const geometry_msgs::msg::PoseStamped& in) const
    {
        geometry_msgs::msg::PoseStamped out = in;
        out.pose.position.x += datum_easting_;
        out.pose.position.y += datum_northing_;
        return out;
    }

    void routeRequestCallback(const command_center_interfaces::msg::RouteRequest::SharedPtr msg)
    {
        active_request_id_ = msg->request_id.empty()
            ? ("req_" + std::to_string(this->get_clock()->now().nanoseconds()))
            : msg->request_id;

        route_nodes_ = msg->via_nodes;
        route_loop_ = msg->loop;
        route_fill_gaps_ = msg->fill_gaps;
        route_mode_ = route_nodes_.empty() ? "astar" : "sequence";
        makePlanner();

        if (msg->use_goal) {
            if (!datum_initialized_) {
                publishStatus(false, "datum not received yet - cannot place goal", 0);
                return;
            }
            // 요청의 goal 은 map 프레임이다 (툴/웹이 datum 상대로 보낸다).
            goal_pose_ = toAbsoluteUtm(msg->goal);
            goal_received_ = true;
            RCLCPP_INFO(this->get_logger(),
                        "goal: map (%.2f, %.2f) -> UTM (%.2f, %.2f)",
                        msg->goal.pose.position.x, msg->goal.pose.position.y,
                        goal_pose_.pose.position.x, goal_pose_.pose.position.y);
        } else if (route_nodes_.empty()) {
            publishStatus(false, "via_nodes is empty and use_goal is false - nothing to plan", 0);
            return;
        }
        // 새 요청이므로 이전 계획도, 이전 실패도 무효.
        path_planned_for_current_goal_ = false;
        request_failed_ = false;
        RCLCPP_INFO(this->get_logger(), "route request '%s': %zu via_nodes, mode=%s",
                    active_request_id_.c_str(), route_nodes_.size(), route_mode_.c_str());
    }

    void publishStatus(bool accepted, const std::string& reason, uint32_t node_count)
    {
        command_center_interfaces::msg::RouteStatus st;
        st.header.stamp = this->get_clock()->now();
        st.request_id = active_request_id_;
        st.accepted = accepted;
        st.reason = reason;
        st.node_count = node_count;
        st.mode = planner_ ? planner_->name() : "";
        route_status_publisher_->publish(st);
        if (!accepted) {
            RCLCPP_WARN(this->get_logger(), "route '%s' rejected: %s",
                        active_request_id_.c_str(), reason.c_str());
        }
    }

    void checkAndPlanPath()
    {
        // request_failed_: 계획 자체가 불가능한 요청(없는 노드 ID 등)은 다시 시도해도
        // 결과가 같다. 새 요청이 올 때까지 멈춘다 - 안 그러면 2초마다 같은 거부를 반복한다.
        // (TF 미수신 같은 일시적 사유는 planRoute 안에서 따로 빠져나가므로 여기 안 걸린다.)
        if (!datum_initialized_ || !graph_received_ ||
            path_planned_for_current_goal_ || request_failed_) {
            return;
        }
        // A fixed route carries its own destination, so it does not wait for a goal.
        if (goal_received_ || route_mode_ == "sequence") {
            planRoute();
        }
    }

    // Current vehicle position in absolute UTM, from TF map->base_link.
    bool getCurrentUtm(double& e, double& n)
    {
        try {
            geometry_msgs::msg::TransformStamped tf =
                tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
            e = tf.transform.translation.x + datum_easting_;
            n = tf.transform.translation.y + datum_northing_;
            return true;
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "map->base_link unavailable: %s", ex.what());
            return false;
        }
    }

    // Run the selected planner and turn indices back into search nodes, so the
    // rest of the node keeps working with poses the way it always did.
    std::vector<std::shared_ptr<AStarNode>> planNodes(int start_id, int goal_id)
    {
        std::vector<std::shared_ptr<AStarNode>> nodes;
        if (!planner_) {
            return nodes;
        }
        std::string reason;
        auto idx = planner_->plan(graph_, start_id, goal_id, reason);
        if (idx.empty()) {
            const std::string why = reason.empty() ? "no route found" : reason;
            RCLCPP_WARN(this->get_logger(), "%s planner found no route: %s",
                        planner_->name(), why.c_str());
            publishStatus(false, why, 0);
            return nodes;
        }
        nodes.reserve(idx.size());
        for (int i : idx) {
            if (auto n = graph_.node(i)) {
                nodes.push_back(n);
            }
        }
        return nodes;
    }

    void planRoute()
    {
        if (graph_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No graph available for path planning");
            return;
        }
        graph_.clearTemporaryNodes();

        double start_e, start_n;
        if (!getCurrentUtm(start_e, start_n)) {
            return;
        }
        const int start_id = graph_.addTemporaryStart(start_e, start_n);
        if (start_id == -1) {
            RCLCPP_ERROR(this->get_logger(), "Could not attach temporary start node");
            return;
        }

        // In sequence mode the route ends where route_nodes says; a goal is optional.
        int goal_id = -1;
        if (goal_received_) {
            goal_id = graph_.addTemporaryGoal(goal_pose_.pose.position.x,
                                              goal_pose_.pose.position.y);
            if (goal_id == -1) {
                RCLCPP_ERROR(this->get_logger(), "Could not attach temporary goal node");
                graph_.clearTemporaryNodes();
                return;
            }
        }

        auto path_nodes = planNodes(start_id, goal_id);
        if (!path_nodes.empty()) {
            publishPath(path_nodes, start_id, goal_id);
            path_planned_for_current_goal_ = true;
        } else {
            request_failed_ = true;   // 같은 요청으로는 다시 시도하지 않는다
        }
        graph_.clearTemporaryNodes();
    }

    void publishPath(const std::vector<std::shared_ptr<AStarNode>>& path_nodes,
                     int start_id, int goal_id)
    {
        nav_msgs::msg::Path planned_path;
        planned_path.header.frame_id = "map";
        planned_path.header.stamp = this->get_clock()->now();
        for (const auto& node : path_nodes) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = planned_path.header;
            ps.pose = node->pose;
            // Published in the map frame, i.e. relative to the datum.
            ps.pose.position.x -= datum_easting_;
            ps.pose.position.y -= datum_northing_;
            ps.pose.position.z = 0.0;
            planned_path.poses.push_back(ps);
        }
        path_publisher_->publish(planned_path);

        auto detailed = createDetailedPlannedPath(path_nodes, start_id, goal_id);
        planned_path_publisher_->publish(detailed);

        path_viz_publisher_->publish(
            PathVisualizer::pathMarkers(planned_path, this->get_clock()->now()));

        RCLCPP_INFO(this->get_logger(), "%s path: %zu waypoints",
                    planner_->name(), planned_path.poses.size());
        publishStatus(true, "", static_cast<uint32_t>(planned_path.poses.size()));
    }

    // ------------------------------------------------------------------ state
    rclcpp::Subscription<map_interfaces::msg::GraphLayer>::SharedPtr graph_sub_;
    rclcpp::Subscription<map_interfaces::msg::UtmLayer>::SharedPtr utm_layer_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_subscriber_;
    rclcpp::Subscription<command_center_interfaces::msg::RouteRequest>::SharedPtr route_subscriber_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    rclcpp::Publisher<command_center_interfaces::msg::PlannedPath>::SharedPtr planned_path_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr path_viz_publisher_;
    rclcpp::Publisher<command_center_interfaces::msg::RouteStatus>::SharedPtr route_status_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    RouteGraph graph_;
    std::unique_ptr<RoutePlanner> planner_;
    std::string route_mode_;
    std::vector<std::string> route_nodes_;
    bool route_loop_{false};
    bool route_fill_gaps_{true};
    std::string active_request_id_;

    geometry_msgs::msg::PoseStamped goal_pose_;
    bool goal_received_;
    bool path_planned_for_current_goal_;
    bool request_failed_{false};

    double datum_easting_{0.0};
    double datum_northing_{0.0};
    int64_t utm_zone_{52};
    bool northern_{true};
    bool datum_initialized_;
    bool graph_received_;
    size_t graph_signature_{0};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathPlannerNode>());
    rclcpp::shutdown();
    return 0;
}
