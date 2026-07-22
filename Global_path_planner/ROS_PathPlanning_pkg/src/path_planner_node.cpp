#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <map_interfaces/msg/graph_layer.hpp>
#include <map_interfaces/msg/map_node.hpp>
#include <map_interfaces/msg/map_link.hpp>
#include <map_interfaces/msg/utm_layer.hpp>
#include <command_center_interfaces/msg/planned_path.hpp>
#include <command_center_interfaces/msg/request_replan.hpp>
#include <vector>
#include <memory>
#include <chrono>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>

// A* Node structure
struct AStarNode {
    int id;
    geometry_msgs::msg::Pose pose;
    double g_cost;  // Cost from start
    double h_cost;  // Heuristic cost to goal
    double f_cost;  // Total cost (g + h)
    int parent_id;  // Parent node ID for path reconstruction
    
    AStarNode(int node_id, geometry_msgs::msg::Pose node_pose) 
        : id(node_id), pose(node_pose), g_cost(0.0), h_cost(0.0), f_cost(0.0), parent_id(-1) {}
    
    AStarNode(int node_id, geometry_msgs::msg::Pose node_pose, double g, double h, int parent)
        : id(node_id), pose(node_pose), g_cost(g), h_cost(h), f_cost(g + h), parent_id(parent) {}
};

// Comparator for priority queue (min-heap based on f_cost)
struct AStarNodeComparator {
    bool operator()(const std::shared_ptr<AStarNode>& a, const std::shared_ptr<AStarNode>& b) {
        return a->f_cost > b->f_cost;  // Min-heap
    }
};

// Link structure for graph representation
struct Link {
    int from_node_id;
    int to_node_id;
    double length;
    
    Link(int from, int to, double len) : from_node_id(from), to_node_id(to), length(len) {}
};

// No need for constant - will use dynamic ID based on existing nodes

class PathPlannerNode : public rclcpp::Node
{
public:
    PathPlannerNode() : Node("global_path_planner_node")
    {
        // Initialize state variables
        goal_received_ = false;
        path_planned_for_current_goal_ = false;
        has_temp_goal_node_ = false;
        has_temp_start_node_ = false;
        temp_goal_node_id_ = -2;
        temp_start_node_id_ = -3;
        datum_initialized_ = false;
        graph_received_ = false;

        // Subscribe to map_provider graph layer (latched, received once / on change)
        graph_sub_ = this->create_subscription<map_interfaces::msg::GraphLayer>(
            "/map_provider_node/graph", rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&PathPlannerNode::graphCallback, this, std::placeholders::_1));

        // Subscribe to map_provider datum (UtmLayer, latched)
        utm_layer_sub_ = this->create_subscription<map_interfaces::msg::UtmLayer>(
            "/map_provider_node/utm", rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&PathPlannerNode::utmLayerCallback, this, std::placeholders::_1));

        goal_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "goal_pose", 10,
            std::bind(&PathPlannerNode::goalCallback, this, std::placeholders::_1));

        branch_subscriber_ = this->create_subscription<command_center_interfaces::msg::RequestReplan>( // 전역 경로 계획 중 분기점 Node에서 향후 경로 선택을 위한 Subscriber
            "/request_replan", 10,
            std::bind(&PathPlannerNode::branchCallback, this, std::placeholders::_1));

        // Create publishers
        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("planned_path", 10);
        planned_path_publisher_ = this->create_publisher<command_center_interfaces::msg::PlannedPath>("planned_path_detailed", 10);
        nodes_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>("map_nodes_viz", 10);
        links_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>("map_links_viz", 10);
        map_viz_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("map_graph_viz", 10);

        // Create TF listener for vehicle pose (map->base_link) lookup
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

        // Create timer for checking path planning conditions
        timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&PathPlannerNode::checkAndPlanPath, this));

        RCLCPP_INFO(this->get_logger(), "Global path planner node initialized with A* algorithm");
    }

private:
    // map_provider graph layer 수신 -> 그래프 구성. latched 이므로 1회 수신(변경 시 갱신).
    void graphCallback(const map_interfaces::msg::GraphLayer::SharedPtr msg)
    {
        graph_map_ = *msg;
        graph_received_ = true;

        RCLCPP_INFO(this->get_logger(),
                   "Graph received: %zu nodes, %zu links",
                   graph_map_.nodes.size(), graph_map_.links.size());

        // Convert GraphLayer to PoseArray for compatibility with existing visualization
        convertGraphMapToPoseArrays();

        // Build graph from map data using actual connectivity
        buildGraph();

        // Publish visualization data only after datum is initialized
        if (datum_initialized_) {
            publishVisualizationData();
        }
    }

    // map_provider datum(UtmLayer) 수신 -> map frame 원점 UTM 갱신. latched.
    void utmLayerCallback(const map_interfaces::msg::UtmLayer::SharedPtr msg)
    {
        datum_easting_ = msg->origin_easting;
        datum_northing_ = msg->origin_northing;
        utm_zone_ = msg->utm_zone;
        northern_ = msg->northern;
        datum_initialized_ = true;

        RCLCPP_INFO(this->get_logger(),
                   "datum received: UTM zone=%d %s e=%.3f n=%.3f",
                   static_cast<int>(utm_zone_), northern_ ? "north" : "south",
                   datum_easting_, datum_northing_);

        // Publish visualization data now that datum is initialized
        if (graph_received_) {
            publishVisualizationData();
        }
    }

    void convertGraphMapToPoseArrays()
    {
        // Convert nodes to PoseArray
        map_nodes_.poses.clear();
        node_ids_.clear();
        node_types_.clear();
        
        for (const auto& node : graph_map_.nodes) {
            geometry_msgs::msg::Pose pose;

            // Always use absolute UTM coordinates
            pose.position.x = node.easting;
            pose.position.y = node.northing;
            pose.position.z = 0.0;

            // Convert heading from degrees to quaternion
            double heading_rad = node.heading_deg * M_PI / 180.0;
            pose.orientation.x = 0.0;
            pose.orientation.y = 0.0;
            pose.orientation.z = sin(heading_rad / 2.0);
            pose.orientation.w = cos(heading_rad / 2.0);

            map_nodes_.poses.push_back(pose);
            node_ids_.push_back(node.id);
            node_types_.push_back(static_cast<short>(node.node_type));
        }
        
        map_nodes_.header.frame_id = "map";
        map_nodes_.header.stamp = this->get_clock()->now();
        
        RCLCPP_INFO(this->get_logger(), "Converted %zu nodes from GraphMap to PoseArray", 
                   map_nodes_.poses.size());
    }
    
    void buildGraph()
    {
        // Clear existing graph
        node_map_.clear();
        adjacency_list_.clear();
        node_id_to_index_.clear();
        
        // Build node map and ID mapping
        for (size_t i = 0; i < graph_map_.nodes.size(); ++i) {
            const auto& map_node = graph_map_.nodes[i];
            node_map_[static_cast<int>(i)] = std::make_shared<AStarNode>(static_cast<int>(i), map_nodes_.poses[i]);
            node_id_to_index_[map_node.id] = static_cast<int>(i);
        }
        
        // Build adjacency list using GraphMap links
        for (const auto& link : graph_map_.links) {
            // Find node indices from string IDs
            auto from_it = node_id_to_index_.find(link.from_node_id);
            auto to_it = node_id_to_index_.find(link.to_node_id);
            
            if (from_it != node_id_to_index_.end() && to_it != node_id_to_index_.end()) {
                int from_node_idx = from_it->second;
                int to_node_idx = to_it->second;
                
                // map_interfaces/MapLink.length 는 미터 단위(map_provider). 없으면 유클리드 거리(m).
                double distance = (link.length > 0.0) ? link.length :
                                 calculateDistance(map_nodes_.poses[from_node_idx], map_nodes_.poses[to_node_idx]);
                
                // Add bidirectional links (roads can be traversed in both directions)
                adjacency_list_[from_node_idx].emplace_back(to_node_idx, from_node_idx, distance);
                // adjacency_list_[to_node_idx].emplace_back(from_node_idx, to_node_idx, distance);
                
                RCLCPP_DEBUG(this->get_logger(), "Connected nodes %d (%s) <-> %d (%s) (distance: %.2f)", 
                           from_node_idx, link.from_node_id.c_str(), to_node_idx, link.to_node_id.c_str(), distance);
            } else {
                RCLCPP_WARN(this->get_logger(), "Unknown node IDs in link: %s -> %s", 
                           link.from_node_id.c_str(), link.to_node_id.c_str());
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Graph built with %zu nodes and connectivity for %zu nodes", 
                   node_map_.size(), adjacency_list_.size());
        
        // Debug: Print adjacency list
        for (const auto& pair : adjacency_list_) {
            std::string connections;
            for (const auto& link : pair.second) {
                int neighbor = (link.from_node_id == pair.first) ? link.to_node_id : link.from_node_id;
                connections += std::to_string(neighbor) + " ";
            }
            RCLCPP_DEBUG(this->get_logger(), "Node %d connected to: %s", pair.first, connections.c_str());
        }
        
        // Check for isolated nodes
        for (size_t i = 0; i < graph_map_.nodes.size(); ++i) {
            if (adjacency_list_.find(static_cast<int>(i)) == adjacency_list_.end()) {
                RCLCPP_WARN(this->get_logger(), "Node %zu (%s) is isolated (no connections)", 
                           i, graph_map_.nodes[i].id.c_str());
            }
        }
    }
    
    void publishVisualizationData()
    {
        visualization_msgs::msg::MarkerArray viz_graph;
        visualization_msgs::msg::Marker viz_marker;

        // geometry_msgs::msg::TransformStamped transform = 
        //     tf_buffer_->lookupTransform(
        //         "odom",      // target frame
        //         "map",    // source frame
        //         tf2::TimePointZero);

        int i = 0;

        RCLCPP_INFO(this->get_logger(), "datum east\t%f", datum_easting_);
        RCLCPP_INFO(this->get_logger(), "datum north\t%f", datum_northing_);

        // Adjust map nodes for RViz visualization
        if (!map_nodes_.poses.empty()) {
            RCLCPP_INFO(this->get_logger(), "nodes viz init");
            geometry_msgs::msg::PoseArray viz_nodes = map_nodes_;
            for (auto& pose : viz_nodes.poses) {
                pose.position.x -= datum_easting_;
                pose.position.y -= datum_northing_;

                viz_marker.header.frame_id = "map";
                viz_marker.header.stamp = this->get_clock()->now();
                viz_marker.ns = "graph";
                viz_marker.id = i;
                viz_marker.type = visualization_msgs::msg::Marker::ARROW;
                viz_marker.scale.x = 1.5;
                // viz_marker.scale.x = 0.5 + (1.0 * node_types_[i]);
                viz_marker.scale.y = 0.5;
                viz_marker.scale.z = 0.5;
                viz_marker.color.a = 0.35;

                auto color = getRGBColor(node_types_[i]);

                viz_marker.color.r = std::get<0>(color);
                viz_marker.color.g = std::get<1>(color);
                viz_marker.color.b = std::get<2>(color);
                viz_marker.pose.position.x = pose.position.x;
                viz_marker.pose.position.y = pose.position.y;
                viz_marker.pose.position.z = 0;

                viz_marker.pose.orientation.x = pose.orientation.x;
                viz_marker.pose.orientation.y = pose.orientation.y;
                viz_marker.pose.orientation.z = pose.orientation.z;
                viz_marker.pose.orientation.w = pose.orientation.w;

                viz_graph.markers.push_back(viz_marker);

                i++;
            }
            viz_nodes.header.stamp = this->get_clock()->now();

            //geometry_msgs::msg::PoseArray transformed_viz_nodes;
            //transformPoseArray(viz_nodes, transformed_viz_nodes, transform);

            nodes_publisher_->publish(viz_nodes);
        }
        
        // Adjust map links for RViz visualization
        // 링크 시각화는 불필요해 보이므로 우선 비활성화 하겠음
        // if (!map_links_.poses.empty()) {
        //     RCLCPP_INFO(this->get_logger(), "links viz init");
        //     geometry_msgs::msg::PoseArray viz_links = map_links_;
        //     for (auto& pose : viz_links.poses) {
        //         //pose.position.x -= datum_easting_;
        //         //pose.position.y -= datum_northing_;
        //         pose.position.x -= map_utm_easting_;
        //         pose.position.y -= map_utm_northing_;

        //         viz_marker.header.frame_id = "map";
        //         viz_marker.header.stamp = this->get_clock()->now();
        //         viz_marker.ns = "graph";
        //         viz_marker.id = i;
        //         viz_marker.type = visualization_msgs::msg::Marker::CUBE;
        //         viz_marker.scale.x = 3.0;
        //         viz_marker.scale.y = 3.0;
        //         viz_marker.scale.z = 3.0;
        //         viz_marker.color.a = 1.0;
        //         viz_marker.color.r = 1.0;
        //         viz_marker.color.g = 0.0;
        //         viz_marker.color.b = 0.0;
        //         viz_marker.pose.position.x = pose.position.x;
        //         viz_marker.pose.position.y = pose.position.y;
        //         viz_marker.pose.position.z = 0;

        //         viz_marker.pose.orientation.x = pose.orientation.x;
        //         viz_marker.pose.orientation.y = pose.orientation.y;
        //         viz_marker.pose.orientation.z = pose.orientation.z;
        //         viz_marker.pose.orientation.w = pose.orientation.w;

        //         viz_graph.markers.push_back(viz_marker);

        //         i++;
        //     }
        //     viz_links.header.stamp = this->get_clock()->now();

        //     links_publisher_->publish(viz_links);
            
        // }

        //visualization_msgs::msg::MarkerArray transformed_viz_graph;
        //transformMarkerArray(viz_graph, transformed_viz_graph, transform);

        map_viz_publisher_->publish(viz_graph);
        
        RCLCPP_INFO(this->get_logger(), "Published visualization data");
    }
    
    std::tuple<int, int, int> getRGBColor(int index) {
        if (index < 1 || index > 13) {
            throw std::invalid_argument("Index must be between 1 and 13");
        }

        switch (index) {
            case 1:  return std::make_tuple(255, 0, 0);     // Red
            case 2:  return std::make_tuple(0, 255, 0);     // Green
            case 3:  return std::make_tuple(0, 0, 255);     // Blue
            case 4:  return std::make_tuple(255, 255, 0);   // Yellow
            case 5:  return std::make_tuple(255, 0, 255);   // Magenta
            case 6:  return std::make_tuple(0, 255, 255);   // Cyan
            case 7:  return std::make_tuple(255, 165, 0);   // Orange
            case 8:  return std::make_tuple(128, 0, 128);   // Purple
            case 9:  return std::make_tuple(255, 192, 203); // Pink
            case 10: return std::make_tuple(165, 42, 42);   // Brown
            case 11: return std::make_tuple(128, 128, 128); // Gray
            case 12: return std::make_tuple(75, 0, 130);    // Indigo - for dynamic replanning trigger
            case 13: return std::make_tuple(255, 20, 147);  // Deep Pink - for dynamic replanning trigger
            default: return std::make_tuple(0, 0, 0);       // Black (fallback)
        }
    }

    void createMarkertext(visualization_msgs::msg::MarkerArray graph, visualization_msgs::msg::Marker origin, int sequence)
    {
        visualization_msgs::msg::Marker text_marker;

        text_marker.header.frame_id = origin.header.frame_id;
        text_marker.header.stamp = origin.header.stamp;
        text_marker.ns = origin.ns;
        text_marker.id = origin.id;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        
        text_marker.scale.z = 1.0;

        text_marker.pose.position.x = origin.pose.position.x;
        text_marker.pose.position.y = origin.pose.position.y;

        // text_marker.text = node_types_[sequence];
        text_marker.text = "test";

        graph.markers.push_back(text_marker);
    }

    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (!datum_initialized_ || !graph_received_) {
            RCLCPP_WARN(this->get_logger(), "Goal received, but datum/graph not initialized yet. Waiting for map_provider.");
            return;
        }

        // Transform goal to map frame based on frame_id
        geometry_msgs::msg::PoseStamped goal_in_map_frame = *msg;

        RCLCPP_INFO(this->get_logger(), "datum east %f", datum_easting_);
        RCLCPP_INFO(this->get_logger(), "datum north %f", datum_northing_);

        if (msg->header.frame_id == "map") {
            RCLCPP_INFO(this->get_logger(), "frame_id : map");
            // Goal is already in map frame, convert to absolute UTM coordinates
            goal_pose_ = *msg;
            goal_pose_.pose.position.x += datum_easting_;
            goal_pose_.pose.position.y += datum_northing_;
            
            RCLCPP_INFO(this->get_logger(), 
                       "Goal received in map frame - Relative: (%.2f, %.2f) -> Absolute UTM: (%.2f, %.2f)", 
                       msg->pose.position.x, msg->pose.position.y,
                       goal_pose_.pose.position.x, goal_pose_.pose.position.y);
        } 
        else if (msg->header.frame_id == "odom") {
            RCLCPP_INFO(this->get_logger(), "frame_id : odom");
            // Goal is in odom frame, transform to map frame first
            try {
                // Transform from odom to map frame
                geometry_msgs::msg::PoseStamped goal_in_map;
                // tf_buffer_->transform(*msg, goal_in_map, "map", tf2::durationFromSec(1.0));
                
                RCLCPP_INFO(this->get_logger(), "transform output x %f", goal_pose_.pose.position.x);
                RCLCPP_INFO(this->get_logger(), "transform output y %f", goal_pose_.pose.position.y);
                
                // Convert to absolute UTM coordinates
                // goal_pose_ = goal_in_map;
                goal_pose_ = *msg;
                goal_pose_.pose.position.x += datum_easting_;
                goal_pose_.pose.position.y += datum_northing_;
                
                RCLCPP_INFO(this->get_logger(), 
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
            RCLCPP_INFO(this->get_logger(), "frame_id : none");
            // Unsupported frame_id, try to transform to map frame
            try {
                geometry_msgs::msg::PoseStamped goal_in_map;
                tf_buffer_->transform(*msg, goal_in_map, "map", tf2::durationFromSec(1.0));
                
                // Convert to absolute UTM coordinates
                goal_pose_ = goal_in_map;
                goal_pose_.pose.position.x += datum_easting_;
                goal_pose_.pose.position.y += datum_northing_;
                
                RCLCPP_INFO(this->get_logger(), 
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
        RCLCPP_INFO(this->get_logger(), "goalCallback goal pose x %f", goal_pose_.pose.position.x);
        RCLCPP_INFO(this->get_logger(), "goalCallback goal pose y %f", goal_pose_.pose.position.y);
        
        goal_received_ = true;
        path_planned_for_current_goal_ = false; // Reset flag for new goal
        
        // Trigger immediate path planning
        planPathFromGpsToGoal();
    }

    void branchCallback(const command_center_interfaces::msg::RequestReplan msg) {
        /*
        아래 코드 블록에서 경로 분기를 위해 하드코딩된 조건식이 다수 존재합니다.
        각각의 분기 시점은 만도 자율주행 대회에서 T자 주차, 평행 주차 수행을 위한 시점에 해당하며
        사용한 맵 파일 mando-merge-base.json 기준
        N136 (T자 주차)
        -> N0347(A코스)
        -> N0393(B코스)
        N0414 (평행 주차)
        -> N0397(A코스)
        -> N0415(B코스)
        에 해당합니다.

        동작 방식

        A 경로를 default로 주행을 시작하며, 주행 중 해당 콜백이 호출되는 경우 분기점에 해당하는 B 경로를 사용

        향후 기능 구현 과정에서 참고 바랍니다.
         */

        // Search for node with the given name in the graph

        auto node_it = node_id_to_index_.find(msg.start_node_id);

        // Output next nodes' IDs and lengths for the start_node_id
        if (node_it != node_id_to_index_.end()) {
            int start_node_index = node_it->second;
            RCLCPP_INFO(this->get_logger(), "=== Next nodes from start_node_id '%s' (index %d) ===",
                       msg.start_node_id.c_str(), start_node_index);

            if (adjacency_list_.find(start_node_index) != adjacency_list_.end()) {
                const auto& links = adjacency_list_[start_node_index];
                if (links.empty()) {
                    RCLCPP_INFO(this->get_logger(), "No next nodes found for start_node_id '%s'", msg.start_node_id.c_str());
                } else {
                    for (const auto& link : links) {
                        int next_node_id = (link.from_node_id == start_node_index) ? link.to_node_id : link.from_node_id;
                        double length = link.length;

                        // Get the string ID for the next node
                        std::string next_node_string_id = "UNKNOWN";
                        if (next_node_id >= 0 && next_node_id < static_cast<int>(node_ids_.size())) {
                            next_node_string_id = node_ids_[next_node_id];
                        }

                        RCLCPP_INFO(this->get_logger(), "  -> Next Node ID: %s (index: %d), Length: %.2f meters",
                                   next_node_string_id.c_str(), next_node_id, length);
                    }
                }
            } else {
                RCLCPP_INFO(this->get_logger(), "Node '%s' not found in adjacency list", msg.start_node_id.c_str());
            }
            RCLCPP_INFO(this->get_logger(), "=== End of next nodes list ===");
        } else {
            RCLCPP_INFO(this->get_logger(), "start_node_id '%s' not found in node_id_to_index_ map", msg.start_node_id.c_str());
        }

        // Find the node with the bigger cost and store it in bigger_node
        std::string bigger_node = "";
        if (node_it != node_id_to_index_.end()) {
            int start_node_index = node_it->second;
            if (adjacency_list_.find(start_node_index) != adjacency_list_.end()) {
                const auto& links = adjacency_list_[start_node_index];
                double max_length = 0.0;
                int bigger_node_index = -1;

                for (const auto& link : links) {
                    int next_node_id = (link.from_node_id == start_node_index) ? link.to_node_id : link.from_node_id;
                    if (link.length > max_length) {
                        max_length = link.length;
                        bigger_node_index = next_node_id;
                    }
                }

                // Get the string ID for the bigger cost node
                if (bigger_node_index >= 0 && bigger_node_index < static_cast<int>(node_ids_.size())) {
                    bigger_node = node_ids_[bigger_node_index];
                    RCLCPP_INFO(this->get_logger(), "Bigger cost node found: %s (index: %d), Length: %.2f meters",
                               bigger_node.c_str(), bigger_node_index, max_length);
                }
            }
        }

        node_it = node_id_to_index_.find(bigger_node);
        RCLCPP_INFO(this->get_logger(), "Using bigger cost node '%s' for routing", bigger_node.c_str());

        int branch_node_index = node_it->second;
        RCLCPP_INFO(this->get_logger(), "Found branch node '%s' at index %d", msg.start_node_id.c_str(), branch_node_index);

        // Plan path from current position to branch node, then from branch node to goal
        // This ensures the branch node is included in the path
        planPathWithMandatoryNode(branch_node_index);
    }
    
    void checkAndPlanPath()
    {
        // Only plan when datum + graph ready, goal set, and not yet planned for current goal
        if (datum_initialized_ && graph_received_ && goal_received_ && !path_planned_for_current_goal_) {
            planPathFromGpsToGoal();
        }
    }

    // TF map->base_link 조회로 현재 차량의 절대 UTM 위치를 계산
    bool getCurrentUtm(double& e, double& n)
    {
        try {
            geometry_msgs::msg::TransformStamped tf =
                tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
            e = tf.transform.translation.x + datum_easting_;
            n = tf.transform.translation.y + datum_northing_;
            return true;
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(),
                       "Could not lookup map->base_link transform: %s", ex.what());
            return false;
        }
    }

    void planPathFromGpsToGoal()
    {
        if (node_map_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No map data available for path planning");
            return;
        }

        if (!datum_initialized_ || !graph_received_ || !goal_received_) {
            RCLCPP_WARN(this->get_logger(), "Datum, graph or Goal not available for path planning");
            return;
        }

        // Clean up any existing temporary nodes first
        removeTemporaryNodes();

        // Current vehicle absolute UTM from TF map->base_link
        double start_utm_easting, start_utm_northing;
        if (!getCurrentUtm(start_utm_easting, start_utm_northing)) {
            RCLCPP_WARN(this->get_logger(), "Current vehicle pose unavailable (TF); aborting planning");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "start utm east %f", start_utm_easting);
        RCLCPP_INFO(this->get_logger(), "start utm north %f", start_utm_northing);

        // Goal has been converted to UTM coordinates in goalCallback
        double goal_x = goal_pose_.pose.position.x;
        double goal_y = goal_pose_.pose.position.y;

        RCLCPP_INFO(this->get_logger(), "planPathFromGpsToGoal goal pose x %f", goal_pose_.pose.position.x);
        RCLCPP_INFO(this->get_logger(), "planPathFromGpsToGoal goal pose y %f", goal_pose_.pose.position.y);
        
        // Create temporary start node at GPS position
        int start_node_id = createTemporaryStartNode(start_utm_easting, start_utm_northing);
        if (start_node_id == -1) {
            RCLCPP_ERROR(this->get_logger(), "Could not create temporary start node");
            return;
        }
        
        // Create temporary goal node and connect it to the closest existing node
        int goal_node_id = createTemporaryGoalNode(goal_x, goal_y);
        if (goal_node_id == -1) {
            RCLCPP_ERROR(this->get_logger(), "Could not create temporary goal node");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Planning path from GPS (UTM: %.2f, %.2f) -> temp node %d to Goal (%.2f, %.2f) -> temp node %d", 
                   start_utm_easting, start_utm_northing, start_node_id, goal_x, goal_y, goal_node_id);
        
        // Plan path using A*
        auto path_nodes = planAStarPath(start_node_id, goal_node_id);
        
        if (!path_nodes.empty()) {
            // Convert to ROS Path message and adjust for RViz
            nav_msgs::msg::Path planned_path;
            planned_path.header.frame_id = "map";
            planned_path.header.stamp = this->get_clock()->now();
            
            for (const auto& node : path_nodes) {
                geometry_msgs::msg::PoseStamped pose_stamped;
                pose_stamped.header.frame_id = "map";
                pose_stamped.header.stamp = this->get_clock()->now();
                pose_stamped.pose = node->pose;
                pose_stamped.pose.position.x -= datum_easting_;
                pose_stamped.pose.position.y -= datum_northing_;
                pose_stamped.pose.position.z = 0;
                
                planned_path.poses.push_back(pose_stamped);
            }
            
            // Publish planned path (기존 nav_msgs::Path)
            path_publisher_->publish(planned_path);
            
            // Create and publish detailed planned path (새로운 custom message)
            auto detailed_path = createDetailedPlannedPath(path_nodes, start_node_id, goal_node_id);
            planned_path_publisher_->publish(detailed_path);
            
            RCLCPP_INFO(this->get_logger(), "Published A* path with %zu waypoints (detailed: %zu nodes, %zu links)", 
                       planned_path.poses.size(), detailed_path.path_data.nodes.size(), detailed_path.path_data.links.size());
            
            // Mark that path has been planned for current goal
            path_planned_for_current_goal_ = true;
        } else {
            RCLCPP_WARN(this->get_logger(), "No path found from node %d to node %d", start_node_id, goal_node_id);
        }
        
        // Clean up temporary nodes after path planning
        removeTemporaryNodes();
    }

    void planPathWithMandatoryNode(int mandatory_node_index) {
        if (node_map_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No map data available for path planning");
            return;
        }

        if (!datum_initialized_ || !graph_received_ || !goal_received_) {
            RCLCPP_WARN(this->get_logger(), "Datum, graph or Goal not available for path planning");
            return;
        }

        // Clean up any existing temporary nodes first
        removeTemporaryNodes();

        // Current vehicle absolute UTM from TF map->base_link
        double start_utm_easting, start_utm_northing;
        if (!getCurrentUtm(start_utm_easting, start_utm_northing)) {
            RCLCPP_WARN(this->get_logger(), "Current vehicle pose unavailable (TF); aborting planning");
            return;
        }

        // Goal has been converted to UTM coordinates in goalCallback
        double goal_x = goal_pose_.pose.position.x;
        double goal_y = goal_pose_.pose.position.y;

        // Create temporary start node at GPS position
        int start_node_id = createTemporaryStartNode(start_utm_easting, start_utm_northing);
        if (start_node_id == -1) {
            RCLCPP_ERROR(this->get_logger(), "Could not create temporary start node");
            return;
        }

        // Create temporary goal node and connect it to the closest existing node
        int goal_node_id = createTemporaryGoalNode(goal_x, goal_y);
        if (goal_node_id == -1) {
            RCLCPP_ERROR(this->get_logger(), "Could not create temporary goal node");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Planning path from GPS -> temp node %d -> mandatory node %d -> temp goal %d",
                   start_node_id, mandatory_node_index, goal_node_id);

        // Plan path from start to mandatory node
        auto path_to_branch = planAStarPath(start_node_id, mandatory_node_index);
        if (path_to_branch.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No path found from start to mandatory node %d", mandatory_node_index);
            removeTemporaryNodes();
            return;
        }

        // Plan path from mandatory node to goal
        auto path_from_branch = planAStarPath(mandatory_node_index, goal_node_id);
        if (path_from_branch.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No path found from mandatory node %d to goal", mandatory_node_index);
            removeTemporaryNodes();
            return;
        }

        // Combine paths, avoiding duplication of mandatory node
        std::vector<std::shared_ptr<AStarNode>> combined_path = path_to_branch;
        // Skip the first node of path_from_branch since it's the same as the last node of path_to_branch
        for (size_t i = 1; i < path_from_branch.size(); ++i) {
            combined_path.push_back(path_from_branch[i]);
        }

        if (!combined_path.empty()) {
            // Convert to ROS Path message and adjust for RViz
            nav_msgs::msg::Path planned_path;
            planned_path.header.frame_id = "map";
            planned_path.header.stamp = this->get_clock()->now();

            for (const auto& node : combined_path) {
                geometry_msgs::msg::PoseStamped pose_stamped;
                pose_stamped.header.frame_id = "map";
                pose_stamped.header.stamp = this->get_clock()->now();
                pose_stamped.pose = node->pose;

                // Adjust position for datum offset (map frame metric = absolute UTM - datum)
                pose_stamped.pose.position.x -= datum_easting_;
                pose_stamped.pose.position.y -= datum_northing_;

                planned_path.poses.push_back(pose_stamped);
            }

            path_publisher_->publish(planned_path);

            // Create detailed path with nodes and links
            auto detailed_path = createDetailedPlannedPath(combined_path, start_node_id, goal_node_id);
            planned_path_publisher_->publish(detailed_path);

            RCLCPP_INFO(this->get_logger(), "Published path with mandatory node: %zu poses, %zu detailed nodes, %zu links",
                       planned_path.poses.size(), detailed_path.path_data.nodes.size(), detailed_path.path_data.links.size());

            // Mark that path has been planned for current goal
            path_planned_for_current_goal_ = true;
        } else {
            RCLCPP_WARN(this->get_logger(), "Combined path is empty");
        }

        // Clean up temporary nodes after path planning
        removeTemporaryNodes();
    }

    // A* path planning algorithm implementation
    std::vector<std::shared_ptr<AStarNode>> planAStarPath(int start_id, int goal_id)
    {
        if (node_map_.find(start_id) == node_map_.end() || 
            node_map_.find(goal_id) == node_map_.end()) {
            RCLCPP_ERROR(this->get_logger(), "Invalid start or goal node ID");
            return {};
        }
        
        // Priority queue for open set (min-heap)
        std::priority_queue<std::shared_ptr<AStarNode>, 
                           std::vector<std::shared_ptr<AStarNode>>, 
                           AStarNodeComparator> open_set;
        
        // Sets to track visited nodes
        std::unordered_set<int> open_set_ids;
        std::unordered_set<int> closed_set;
        
        // Map to store best g_cost for each node
        std::unordered_map<int, double> best_g_cost;
        
        // Map to store parent relationships for path reconstruction
        std::unordered_map<int, int> parent_map;
        
        // Initialize start node
        auto start_node = std::make_shared<AStarNode>(*node_map_[start_id]);
        start_node->g_cost = 0.0;
        start_node->h_cost = calculateHeuristic(start_node->pose, node_map_[goal_id]->pose);
        start_node->f_cost = start_node->g_cost + start_node->h_cost;
        start_node->parent_id = -1;
        
        open_set.push(start_node);
        open_set_ids.insert(start_id);
        best_g_cost[start_id] = 0.0;
        
        while (!open_set.empty()) {
            // Get node with lowest f_cost
            auto current = open_set.top();
            open_set.pop();
            open_set_ids.erase(current->id);
            
            // Add to closed set
            closed_set.insert(current->id);
            
            // Check if we reached the goal
            if (current->id == goal_id) {
                RCLCPP_INFO(this->get_logger(), "A* path found with cost: %.2f", current->f_cost);
                return reconstructPath(current, parent_map);
            }
            
            // Explore neighbors
            if (adjacency_list_.find(current->id) != adjacency_list_.end()) {
                for (const auto& link : adjacency_list_[current->id]) {
                    int neighbor_id = link.from_node_id == current->id ? link.to_node_id : link.from_node_id;
                    
                    // Skip if in closed set
                    if (closed_set.find(neighbor_id) != closed_set.end()) {
                        continue;
                    }
                    
                    // Calculate tentative g_cost
                    double tentative_g = current->g_cost + link.length;
                    
                    // Check if this path to neighbor is better
                    if (best_g_cost.find(neighbor_id) == best_g_cost.end() || 
                        tentative_g < best_g_cost[neighbor_id]) {
                        
                        // Update best g_cost and parent
                        best_g_cost[neighbor_id] = tentative_g;
                        parent_map[neighbor_id] = current->id;
                        
                        // Create neighbor node
                        auto neighbor = std::make_shared<AStarNode>(*node_map_[neighbor_id]);
                        neighbor->g_cost = tentative_g;
                        neighbor->h_cost = calculateHeuristic(neighbor->pose, node_map_[goal_id]->pose);
                        neighbor->f_cost = neighbor->g_cost + neighbor->h_cost;
                        neighbor->parent_id = current->id;
                        
                        // Add to open set if not already there
                        if (open_set_ids.find(neighbor_id) == open_set_ids.end()) {
                            open_set.push(neighbor);
                            open_set_ids.insert(neighbor_id);
                        }
                    }
                }
            }
        }
        
        RCLCPP_WARN(this->get_logger(), "No A* path found from %d to %d", start_id, goal_id);
        return {};
    }
    
    // Reconstruct path from goal to start using parent relationships
    std::vector<std::shared_ptr<AStarNode>> reconstructPath(
        std::shared_ptr<AStarNode> goal_node,
        const std::unordered_map<int, int>& parent_map)
    {
        std::vector<std::shared_ptr<AStarNode>> path;
        int current_id = goal_node->id;
        
        // Build path backwards from goal to start
        while (current_id != -1) {
            path.push_back(node_map_[current_id]);
            
            auto parent_it = parent_map.find(current_id);
            current_id = (parent_it != parent_map.end()) ? parent_it->second : -1;
        }
        
        // Reverse to get path from start to goal
        std::reverse(path.begin(), path.end());
        
        return path;
    }
    
    // Calculate heuristic (Euclidean distance)
    double calculateHeuristic(const geometry_msgs::msg::Pose& a, 
                             const geometry_msgs::msg::Pose& b)
    {
        return calculateDistance(a, b);
    }
    
    // Calculate Euclidean distance between two poses
    double calculateDistance(const geometry_msgs::msg::Pose& a, 
                           const geometry_msgs::msg::Pose& b)
    {
        double dx = a.position.x - b.position.x;
        double dy = a.position.y - b.position.y;
        double dz = a.position.z - b.position.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    
    
    // Find closest node to given UTM coordinates
    int findClosestNode(double utm_x, double utm_y)
    {
        if (node_map_.empty()) {
            return -1;
        }
        
        int closest_id = -1;
        double min_distance = std::numeric_limits<double>::max();
        
        for (const auto& pair : node_map_) {
            int node_id = pair.first;
            const auto& node = pair.second;
            
            double dx = node->pose.position.x - utm_x;
            double dy = node->pose.position.y - utm_y;
            double distance = std::sqrt(dx*dx + dy*dy);
            
            if (distance < min_distance) {
                min_distance = distance;
                closest_id = node_id;
            }
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Closest node to (%.2f, %.2f) is node %d at distance %.2f", 
                    utm_x, utm_y, closest_id, min_distance);
        
        return closest_id;
    }
    
    // Helper function for buildGraph - find closest node to a position
    int findClosestNodeToPosition(double x, double y)
    {
        if (map_nodes_.poses.empty()) {
            return -1;
        }
        
        int closest_id = -1;
        double min_distance = std::numeric_limits<double>::max();
        
        for (size_t i = 0; i < map_nodes_.poses.size(); ++i) {
            double dx = map_nodes_.poses[i].position.x - x;
            double dy = map_nodes_.poses[i].position.y - y;
            double distance = std::sqrt(dx*dx + dy*dy);
            
            if (distance < min_distance) {
                min_distance = distance;
                closest_id = static_cast<int>(i);
            }
        }
        
        return closest_id;
    }
    
    int createTemporaryGoalNode(double goal_x, double goal_y)
    {
        temp_goal_node_id_ = -2;
        
        // Create temporary goal node pose
        geometry_msgs::msg::Pose goal_pose;
        goal_pose.position.x = goal_x;
        goal_pose.position.y = goal_y;
        goal_pose.position.z = 0.0;
        goal_pose.orientation.w = 1.0;
        
        // Add temporary node to node map
        auto temp_node = std::make_shared<AStarNode>(temp_goal_node_id_, goal_pose);
        node_map_[temp_goal_node_id_] = temp_node;
        
        // Find closest existing node
        int closest_node_id = findClosestNodeToPosition(goal_x, goal_y);
        if (closest_node_id == -1) {
            RCLCPP_ERROR(this->get_logger(), "No existing nodes found to connect temporary goal node");
            return -1;
        }
        
        // Calculate distance to closest node
        double distance = calculateDistance(map_nodes_.poses[closest_node_id], goal_pose);
        
        // Create bidirectional links between goal node and closest existing node
        adjacency_list_[temp_goal_node_id_].emplace_back(closest_node_id, temp_goal_node_id_, distance);
        adjacency_list_[closest_node_id].emplace_back(temp_goal_node_id_, closest_node_id, distance);
        
        has_temp_goal_node_ = true;
        
        RCLCPP_INFO(this->get_logger(), 
                   "Created temporary goal node %d at (%.2f, %.2f) connected to node %d (distance: %.2f)",
                   temp_goal_node_id_, goal_x, goal_y, closest_node_id, distance);
        
        return temp_goal_node_id_;
    }
    
    int createTemporaryStartNode(double start_x, double start_y)
    {   
        // Create temporary start node pose
        geometry_msgs::msg::Pose start_pose;
        start_pose.position.x = start_x;
        start_pose.position.y = start_y;
        start_pose.position.z = 0.0;
        start_pose.orientation.w = 1.0;
        
        // Add temporary node to node map
        auto temp_node = std::make_shared<AStarNode>(temp_start_node_id_, start_pose);
        node_map_[temp_start_node_id_] = temp_node;
        
        // Find closest existing node
        int closest_node_id = findClosestNodeToPosition(start_x, start_y);
        if (closest_node_id == -1) {
            RCLCPP_ERROR(this->get_logger(), "No existing nodes found to connect temporary start node");
            return -1;
        }
        
        // Calculate distance to closest node
        double distance = calculateDistance(map_nodes_.poses[closest_node_id], start_pose);
        
        // Create bidirectional links between start node and closest existing node
        adjacency_list_[temp_start_node_id_].emplace_back(closest_node_id, temp_start_node_id_, distance);
        adjacency_list_[closest_node_id].emplace_back(temp_start_node_id_, closest_node_id, distance);
        
        has_temp_start_node_ = true;
        
        RCLCPP_INFO(this->get_logger(), 
                   "Created temporary start node %d at (%.2f, %.2f) connected to node %d (distance: %.2f)",
                   temp_start_node_id_, start_x, start_y, closest_node_id, distance);
        
        return temp_start_node_id_;
    }
    
    void removeTemporaryNodes()
    {
        // Remove temporary goal node
        if (has_temp_goal_node_) {
            auto temp_it = node_map_.find(temp_goal_node_id_);
            if (temp_it != node_map_.end()) {
                node_map_.erase(temp_it);
            }
            
            auto adj_it = adjacency_list_.find(temp_goal_node_id_);
            if (adj_it != adjacency_list_.end()) {
                for (const auto& link : adj_it->second) {
                    int connected_node_id = (link.from_node_id == temp_goal_node_id_) ? link.to_node_id : link.from_node_id;
                    auto& connected_links = adjacency_list_[connected_node_id];
                    connected_links.erase(
                        std::remove_if(connected_links.begin(), connected_links.end(),
                            [this](const Link& l) { 
                                return l.from_node_id == temp_goal_node_id_ || l.to_node_id == temp_goal_node_id_; 
                            }),
                        connected_links.end()
                    );
                }
                adjacency_list_.erase(adj_it);
            }
            
            has_temp_goal_node_ = false;
            RCLCPP_DEBUG(this->get_logger(), "Removed temporary goal node from graph");
        }
        
        // Remove temporary start node  
        if (has_temp_start_node_) {
            auto temp_it = node_map_.find(temp_start_node_id_);
            if (temp_it != node_map_.end()) {
                node_map_.erase(temp_it);
            }
            
            auto adj_it = adjacency_list_.find(temp_start_node_id_);
            if (adj_it != adjacency_list_.end()) {
                for (const auto& link : adj_it->second) {
                    int connected_node_id = (link.from_node_id == temp_start_node_id_) ? link.to_node_id : link.from_node_id;
                    auto& connected_links = adjacency_list_[connected_node_id];
                    connected_links.erase(
                        std::remove_if(connected_links.begin(), connected_links.end(),
                            [this](const Link& l) { 
                                return l.from_node_id == temp_start_node_id_ || l.to_node_id == temp_start_node_id_; 
                            }),
                        connected_links.end()
                    );
                }
                adjacency_list_.erase(adj_it);
            }
            
            has_temp_start_node_ = false;
            RCLCPP_DEBUG(this->get_logger(), "Removed temporary start node from graph");
        }
    }
    
    // Create detailed planned path with nodes and links
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
        detailed_path.start_node_id = (start_node_id == temp_start_node_id_) ? "GPS_START" : 
                                     (start_node_id < static_cast<int>(node_ids_.size()) ? node_ids_[start_node_id] : "UNKNOWN");
        detailed_path.goal_node_id = (goal_node_id == temp_goal_node_id_) ? "GPS_GOAL" : 
                                    (goal_node_id < static_cast<int>(node_ids_.size()) ? node_ids_[goal_node_id] : "UNKNOWN");
        
        // Calculate total distance
        double total_distance = 0.0;
        for (size_t i = 1; i < path_nodes.size(); ++i) {
            total_distance += calculateDistance(path_nodes[i-1]->pose, path_nodes[i]->pose);
        }
        detailed_path.total_distance = total_distance;
        detailed_path.total_time = total_distance / 10.0; // 평균 속도 10m/s 가정
        
        // Convert path nodes to MapNode messages (map_interfaces, 절대 UTM 유지)
        detailed_path.path_data.nodes.clear();
        for (size_t i = 0; i < path_nodes.size(); ++i) {
            map_interfaces::msg::MapNode map_node;

            int node_idx = path_nodes[i]->id;

            // Temporary nodes에 대한 처리
            if (node_idx == temp_start_node_id_) {
                map_node.id = "GPS_START";
                map_node.source = "gps";
            } else if (node_idx == temp_goal_node_id_) {
                map_node.id = "GPS_GOAL";
                map_node.source = "gps";
            } else if (node_idx >= 0 && node_idx < static_cast<int>(graph_map_.nodes.size())) {
                // 실제 맵 노드에서 정보 복사 (id/node_type/easting/northing/lat/lon/heading_deg/source)
                map_node = graph_map_.nodes[node_idx];
            } else {
                // Fallback for unknown nodes
                map_node.id = "NODE_" + std::to_string(node_idx);
            }

            if (node_idx >= 0 && node_idx < static_cast<int>(graph_map_.nodes.size()) &&
                node_idx != temp_start_node_id_ && node_idx != temp_goal_node_id_) {
                // 실제 맵 노드: 원본 절대 UTM/GPS/heading 유지 (datum 빼지 않음)
                map_node.easting = graph_map_.nodes[node_idx].easting;
                map_node.northing = graph_map_.nodes[node_idx].northing;
                map_node.latitude = graph_map_.nodes[node_idx].latitude;
                map_node.longitude = graph_map_.nodes[node_idx].longitude;
                map_node.heading_deg = graph_map_.nodes[node_idx].heading_deg;
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
            double distance = calculateDistance(path_nodes[i-1]->pose, path_nodes[i]->pose);
            map_link.length = distance;

            // Try to find existing link in graph to reuse its length
            if (from_node_idx >= 0 && from_node_idx < static_cast<int>(graph_map_.nodes.size()) &&
                to_node_idx >= 0 && to_node_idx < static_cast<int>(graph_map_.nodes.size()) &&
                from_node_idx != temp_start_node_id_ && from_node_idx != temp_goal_node_id_ &&
                to_node_idx != temp_start_node_id_ && to_node_idx != temp_goal_node_id_) {

                std::string from_id = graph_map_.nodes[from_node_idx].id;
                std::string to_id = graph_map_.nodes[to_node_idx].id;

                // Find existing link in GraphLayer
                for (const auto& original_link : graph_map_.links) {
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
    void transformPoseArray(
        const geometry_msgs::msg::PoseArray& input,
        geometry_msgs::msg::PoseArray& output,
        const geometry_msgs::msg::TransformStamped& transform)
    {
        output.header = input.header;
        output.poses.clear();
        output.poses.reserve(input.poses.size());
        
        for (const auto& pose : input.poses) {
            geometry_msgs::msg::Pose transformed_pose;
            tf2::doTransform(pose, transformed_pose, transform);
            output.poses.push_back(transformed_pose);
        }
    }

    void transformMarkerArray(
        const visualization_msgs::msg::MarkerArray& input,
        visualization_msgs::msg::MarkerArray& output,
        const geometry_msgs::msg::TransformStamped& transform)
    {
        output.markers.clear();
        output.markers.reserve(input.markers.size());
        
        for (const auto& marker : input.markers) {
            visualization_msgs::msg::Marker transformed_marker = marker;
            
            // Marker의 pose 변환
            tf2::doTransform(marker.pose, transformed_marker.pose, transform);
            
            // Points가 있는 경우 (LINE_STRIP, LINE_LIST, POINTS 등)
            if (!marker.points.empty()) {
                transformed_marker.points.clear();
                transformed_marker.points.reserve(marker.points.size());
                
                for (const auto& point : marker.points) {
                    geometry_msgs::msg::Point transformed_point;
                    tf2::doTransform(point, transformed_point, transform);
                    transformed_marker.points.push_back(transformed_point);
                }
            }
            
            // Frame ID 업데이트 (필요한 경우)
            transformed_marker.header.frame_id = transform.header.frame_id;
            
            output.markers.push_back(transformed_marker);
        }
    }
    
    // Member variables
    rclcpp::Subscription<map_interfaces::msg::GraphLayer>::SharedPtr graph_sub_;
    rclcpp::Subscription<map_interfaces::msg::UtmLayer>::SharedPtr utm_layer_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_subscriber_;
    rclcpp::Subscription<command_center_interfaces::msg::RequestReplan>::SharedPtr branch_subscriber_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    rclcpp::Publisher<command_center_interfaces::msg::PlannedPath>::SharedPtr planned_path_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr nodes_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr links_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr map_viz_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // Graph layer data from map_provider
    map_interfaces::msg::GraphLayer graph_map_;

    // Converted data for compatibility with existing visualization
    geometry_msgs::msg::PoseArray map_nodes_;
    geometry_msgs::msg::PoseArray map_links_;
    std::vector<std::string> node_ids_;
    std::vector<short> node_types_;

    // Node ID to index mapping for efficient lookup
    std::unordered_map<std::string, int> node_id_to_index_;

    // Goal state
    geometry_msgs::msg::PoseStamped goal_pose_;
    bool goal_received_;
    bool path_planned_for_current_goal_; // Flag to ensure single path planning per goal

    // datum (map frame origin in absolute UTM) from map_provider UtmLayer
    double datum_easting_{0.0};
    double datum_northing_{0.0};
    uint8_t utm_zone_{52};
    bool northern_{true};

    bool datum_initialized_; // datum(UtmLayer) received
    bool graph_received_;    // graph layer received

    // A* algorithm data structures
    std::unordered_map<int, std::shared_ptr<AStarNode>> node_map_;
    std::unordered_map<int, std::vector<Link>> adjacency_list_;
    
    // Temporary node management
    int temp_goal_node_id_;
    int temp_start_node_id_;
    bool has_temp_goal_node_;
    bool has_temp_start_node_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PathPlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}