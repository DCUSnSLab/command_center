#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <gmserver/srv/load_map.hpp>
#include <gmserver/msg/graph_map.hpp>
#include <gmserver/msg/map_data.hpp>
#include <gmserver/msg/map_node.hpp>
#include <gmserver/msg/map_link.hpp>
#include <gmserver/msg/gps_info.hpp>
#include <gmserver/msg/utm_info.hpp>
#include <command_center_interfaces/msg/planned_path.hpp>
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
        // Declare parameters
        this->declare_parameter<std::string>("map_file_path", 
            "/home/ros2/ros2_ws/src/gmserver/maps/3x3_map.json");
        
        // Initialize state variables
        current_gps_received_ = false;
        current_imu_received_ = false;
        goal_received_ = false;
        path_planned_for_current_goal_ = false;
        has_temp_goal_node_ = false;
        has_temp_start_node_ = false;
        temp_goal_node_id_ = -2;
        temp_start_node_id_ = -3;
        gps_ref_initialized_ = false;
        
        // Create service client for map loading
        map_client_ = this->create_client<gmserver::srv::LoadMap>("load_map");
        
        // Create subscribers
        gps_subscriber_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "gps/fix", 10,
            std::bind(&PathPlannerNode::gpsCallback, this, std::placeholders::_1));
            
        imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu/data", 10,
            std::bind(&PathPlannerNode::imuCallback, this, std::placeholders::_1));
            
        goal_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "goal_pose", 10,
            std::bind(&PathPlannerNode::goalCallback, this, std::placeholders::_1));
        
        // Create publishers
        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("planned_path", 10);
        planned_path_publisher_ = this->create_publisher<command_center_interfaces::msg::PlannedPath>("planned_path_detailed", 10);
        nodes_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>("map_nodes_viz", 10);
        links_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>("map_links_viz", 10);
        map_viz_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("map_graph_viz", 10);
        
        // Create TF broadcaster for map->odom transform
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        
        // Create TF listener for coordinate transformations
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Create timer for checking path planning conditions
        timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&PathPlannerNode::checkAndPlanPath, this));
        
        RCLCPP_INFO(this->get_logger(), "Global path planner node initialized with A* algorithm");
        
        // Load map initially
        loadMapData();
    }

private:
    void loadMapData()
    {
        RCLCPP_INFO(this->get_logger(), "Starting map data loading...");
        
        // Wait for service to be available
        if (!map_client_->wait_for_service(std::chrono::seconds(10))) {
            RCLCPP_ERROR(this->get_logger(), "Map service not available after 10 seconds");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Map service is available");
        
        // Get map file path parameter
        std::string map_file_path;
        this->get_parameter("map_file_path", map_file_path);
        
        // Create service request
        auto request = std::make_shared<gmserver::srv::LoadMap::Request>();
        request->map_file_path = map_file_path;
        
        RCLCPP_INFO(this->get_logger(), "Requesting map data from: %s", map_file_path.c_str());
        
        // Call service asynchronously
        auto future = map_client_->async_send_request(request);
        
        RCLCPP_INFO(this->get_logger(), "Map service request sent, waiting for response...");
        
        // Wait for response
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "Map service response received: success=%s", 
                       response->success ? "true" : "false");
            
            if (response->success) {
                // Store GraphMap data
                graph_map_ = response->graph_map;
                
                // Convert GraphMap to PoseArray for compatibility with existing visualization
                convertGraphMapToPoseArrays();
                
                RCLCPP_INFO(this->get_logger(), 
                           "Map data stored: %zu nodes, %zu links", 
                           graph_map_.map_data.nodes.size(), graph_map_.map_data.links.size());
                
                // Build graph from map data using actual connectivity
                buildGraph();
                
                RCLCPP_INFO(this->get_logger(), 
                           "Map loaded successfully: %zu nodes, %zu links", 
                           graph_map_.map_data.nodes.size(), graph_map_.map_data.links.size());
                
                // Publish visualization data
                publishVisualizationData();
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to load map: %s", 
                            response->message.c_str());
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to call map service - timeout or error");
        }
    }
    
    void convertGraphMapToPoseArrays()
    {
        // Convert nodes to PoseArray
        map_nodes_.poses.clear();
        node_ids_.clear();
        
        for (const auto& node : graph_map_.map_data.nodes) {
            geometry_msgs::msg::Pose pose;
            
            // Use UTM coordinates if available, otherwise use GPS
            if (!node.utm_info.zone.empty()) {
                pose.position.x = node.utm_info.easting;
                pose.position.y = node.utm_info.northing;
                pose.position.z = node.gps_info.alt;
            } else {
                // Simple GPS to local coordinate conversion as fallback
                pose.position.x = node.gps_info.longitude * 111320.0;
                pose.position.y = node.gps_info.lat * 110540.0;
                pose.position.z = node.gps_info.alt;
            }
            
            pose.orientation.w = 1.0; // No rotation for nodes
            
            map_nodes_.poses.push_back(pose);
            node_ids_.push_back(node.id);
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
        for (size_t i = 0; i < graph_map_.map_data.nodes.size(); ++i) {
            const auto& map_node = graph_map_.map_data.nodes[i];
            node_map_[static_cast<int>(i)] = std::make_shared<AStarNode>(static_cast<int>(i), map_nodes_.poses[i]);
            node_id_to_index_[map_node.id] = static_cast<int>(i);
        }
        
        // Build adjacency list using GraphMap links
        for (const auto& link : graph_map_.map_data.links) {
            // Find node indices from string IDs
            auto from_it = node_id_to_index_.find(link.from_node_id);
            auto to_it = node_id_to_index_.find(link.to_node_id);
            
            if (from_it != node_id_to_index_.end() && to_it != node_id_to_index_.end()) {
                int from_node_idx = from_it->second;
                int to_node_idx = to_it->second;
                
                // Use link length from GraphMap, or calculate if not available
                double distance = (link.length > 0.0) ? link.length * 1000.0 : // Convert km to m
                                 calculateDistance(map_nodes_.poses[from_node_idx], map_nodes_.poses[to_node_idx]);
                
                // Add bidirectional links (roads can be traversed in both directions)
                adjacency_list_[from_node_idx].emplace_back(to_node_idx, from_node_idx, distance);
                adjacency_list_[to_node_idx].emplace_back(from_node_idx, to_node_idx, distance);
                
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
        for (size_t i = 0; i < graph_map_.map_data.nodes.size(); ++i) {
            if (adjacency_list_.find(static_cast<int>(i)) == adjacency_list_.end()) {
                RCLCPP_WARN(this->get_logger(), "Node %zu (%s) is isolated (no connections)", 
                           i, graph_map_.map_data.nodes[i].id.c_str());
            }
        }
    }
    
    void publishVisualizationData()
    {
        visualization_msgs::msg::MarkerArray viz_graph;
        visualization_msgs::msg::Marker viz_marker;

        int i = 0;

        // Adjust map nodes for RViz visualization
        if (!map_nodes_.poses.empty()) {
            geometry_msgs::msg::PoseArray viz_nodes = map_nodes_;
            for (auto& pose : viz_nodes.poses) {
                pose.position.x -= gps_ref_utm_easting_;
                pose.position.y -= gps_ref_utm_northing_;

                viz_marker.header.frame_id = "map";
                viz_marker.header.stamp = this->get_clock()->now();
                viz_marker.ns = "graph";
                viz_marker.id = i;
                viz_marker.type = visualization_msgs::msg::Marker::CUBE;
                viz_marker.scale.x = 3.0;
                viz_marker.scale.y = 3.0;
                viz_marker.scale.z = 3.0;
                viz_marker.color.a = 1.0;
                viz_marker.color.r = 0.0;
                viz_marker.color.g = 1.0;
                viz_marker.color.b = 0.0;
                viz_marker.pose.position.x = pose.position.x;
                viz_marker.pose.position.y = pose.position.y;
                viz_marker.pose.position.z = 0;
                viz_graph.markers.push_back(viz_marker);

                i++;
            }
            viz_nodes.header.stamp = this->get_clock()->now();
            nodes_publisher_->publish(viz_nodes);
        }
        
        // Adjust map links for RViz visualization
        if (!map_links_.poses.empty()) {
            geometry_msgs::msg::PoseArray viz_links = map_links_;
            for (auto& pose : viz_links.poses) {
                pose.position.x -= gps_ref_utm_easting_;
                pose.position.y -= gps_ref_utm_northing_;

                viz_marker.header.frame_id = "map";
                viz_marker.header.stamp = this->get_clock()->now();
                viz_marker.ns = "graph";
                viz_marker.id = i;
                viz_marker.type = visualization_msgs::msg::Marker::SPHERE;
                viz_marker.scale.x = 3.0;
                viz_marker.scale.y = 3.0;
                viz_marker.scale.z = 3.0;
                viz_marker.color.a = 1.0;
                viz_marker.color.r = 1.0;
                viz_marker.color.g = 0.0;
                viz_marker.color.b = 0.0;
                viz_marker.pose.position.x = pose.position.x;
                viz_marker.pose.position.y = pose.position.y;
                viz_marker.pose.position.z = 0;
                viz_graph.markers.push_back(viz_marker);

                i++;
            }
            viz_links.header.stamp = this->get_clock()->now();
            links_publisher_->publish(viz_links);
            
        }

        map_viz_publisher_->publish(viz_graph);
        
        RCLCPP_DEBUG(this->get_logger(), "Published visualization data");
    }
    
    void gpsCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        if (msg->status.status < 0) {
            return; // Invalid GPS fix
        }
        
        // Initialize GPS reference coordinates from first valid GPS reading
        if (!gps_ref_initialized_) {
            gps_ref_lat_ = msg->latitude;
            gps_ref_lon_ = msg->longitude;
            gps_ref_alt_ = msg->altitude;
            
            // Calculate GPS reference UTM coordinates for goal transformation
            gpsToUTM(gps_ref_lat_, gps_ref_lon_, gps_ref_utm_easting_, gps_ref_utm_northing_);
            
            gps_ref_initialized_ = true;
            
            RCLCPP_INFO(this->get_logger(), 
                       "GPS reference initialized from first GPS reading: lat=%.6f, lon=%.6f, alt=%.2f -> UTM(%.2f, %.2f)", 
                       gps_ref_lat_, gps_ref_lon_, gps_ref_alt_,
                       gps_ref_utm_easting_, gps_ref_utm_northing_);
        }
        
        current_gps_ = *msg;
        current_gps_received_ = true;
        
        // Publish GPS-based map->odom transform
        // publishMapToOdomTransform(*msg); // 우선은 임시로 비활성화, tiny_localization에 있는 map->odom 변환을 사용하기로..
        
        RCLCPP_DEBUG(this->get_logger(), "GPS received: lat=%.6f, lon=%.6f", 
                    msg->latitude, msg->longitude);
    }
    
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        current_imu_ = *msg;
        current_imu_received_ = true;
        
        // Update map->odom transform with new orientation if GPS is also available
        if (current_gps_received_) {
            // publishMapToOdomTransform(current_gps_); # 위의 gps 콜백과 마찬가지 내용
        }
        
        RCLCPP_DEBUG(this->get_logger(), "IMU received: orientation(%.3f, %.3f, %.3f, %.3f)", 
                    msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
    }
    
    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (!gps_ref_initialized_) {
            RCLCPP_WARN(this->get_logger(), "Goal received, but GPS reference not initialized yet. Waiting for first GPS reading.");
            return;
        }

        // Transform goal to map frame based on frame_id
        geometry_msgs::msg::PoseStamped goal_in_map_frame = *msg;
        
        if (msg->header.frame_id == "map") {
            // Goal is already in map frame, convert to absolute UTM coordinates
            goal_pose_ = *msg;
            goal_pose_.pose.position.x += gps_ref_utm_easting_;
            goal_pose_.pose.position.y += gps_ref_utm_northing_;
            
            RCLCPP_INFO(this->get_logger(), 
                       "Goal received in map frame - Relative: (%.2f, %.2f) -> Absolute UTM: (%.2f, %.2f)", 
                       msg->pose.position.x, msg->pose.position.y,
                       goal_pose_.pose.position.x, goal_pose_.pose.position.y);
        } 
        else if (msg->header.frame_id == "odom") {
            // Goal is in odom frame, transform to map frame first
            try {
                // Transform from odom to map frame
                geometry_msgs::msg::PoseStamped goal_in_map;
                tf_buffer_->transform(*msg, goal_in_map, "map", tf2::durationFromSec(1.0));
                
                // Convert to absolute UTM coordinates
                goal_pose_ = goal_in_map;
                goal_pose_.pose.position.x += gps_ref_utm_easting_;
                goal_pose_.pose.position.y += gps_ref_utm_northing_;
                
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
            // Unsupported frame_id, try to transform to map frame
            try {
                geometry_msgs::msg::PoseStamped goal_in_map;
                tf_buffer_->transform(*msg, goal_in_map, "map", tf2::durationFromSec(1.0));
                
                // Convert to absolute UTM coordinates
                goal_pose_ = goal_in_map;
                goal_pose_.pose.position.x += gps_ref_utm_easting_;
                goal_pose_.pose.position.y += gps_ref_utm_northing_;
                
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
                goal_pose_.pose.position.x += gps_ref_utm_easting_;
                goal_pose_.pose.position.y += gps_ref_utm_northing_;
            }
        }
        
        goal_received_ = true;
        path_planned_for_current_goal_ = false; // Reset flag for new goal
        
        // Trigger immediate path planning
        planPathFromGpsToGoal();
    }
    
    void checkAndPlanPath()
    {
        // Only plan path when we have GPS reference initialized, current GPS and goal, and haven't planned for current goal yet
        if (gps_ref_initialized_ && current_gps_received_ && goal_received_ && !path_planned_for_current_goal_) {
            planPathFromGpsToGoal();
        }
    }
    
    void planPathFromGpsToGoal()
    {
        if (node_map_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No map data available for path planning");
            return;
        }
        
        if (!gps_ref_initialized_ || !current_gps_received_ || !goal_received_) {
            RCLCPP_WARN(this->get_logger(), "GPS reference, current GPS or Goal not available for path planning");
            return;
        }
        
        // Clean up any existing temporary nodes first
        removeTemporaryNodes();
        
        // Convert GPS to UTM coordinates
        double start_utm_easting, start_utm_northing;
        gpsToUTM(current_gps_.latitude, current_gps_.longitude, start_utm_easting, start_utm_northing);
        
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
                pose_stamped.pose.position.x -= gps_ref_utm_easting_;
                pose_stamped.pose.position.y -= gps_ref_utm_northing_;
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
    
    // Universal GPS to UTM conversion
    void gpsToUTM(double lat, double lon, double& easting, double& northing)
    {
        // WGS84 ellipsoid parameters
        const double a = 6378137.0;           // Semi-major axis
        const double f = 1.0 / 298.257223563; // Flattening
        const double k0 = 0.9996;            // UTM scale factor

        // Determine the UTM zone
        int zone = static_cast<int>((lon + 180.0) / 6.0) + 1;

        // Calculate the central meridian for the zone
        double lon0_deg = (zone - 1) * 6 - 180 + 3;
        double lon0_rad = lon0_deg * M_PI / 180.0;

        // False easting and northing
        double false_easting = 500000.0;
        double false_northing = (lat < 0) ? 10000000.0 : 0.0; // Southern hemisphere

        // Convert lat/lon to radians
        double lat_rad = lat * M_PI / 180.0;
        double lon_rad = lon * M_PI / 180.0;

        // Equations for conversion
        double e2 = 2 * f - f * f;
        double e_prime_sq = e2 / (1.0 - e2);

        double N = a / std::sqrt(1.0 - e2 * std::sin(lat_rad) * std::sin(lat_rad));
        double T = std::tan(lat_rad) * std::tan(lat_rad);
        double C = e_prime_sq * std::cos(lat_rad) * std::cos(lat_rad);
        double A = std::cos(lat_rad) * (lon_rad - lon0_rad);

        double M = a * ((1.0 - e2/4.0 - 3.0*e2*e2/64.0 - 5.0*e2*e2*e2/256.0) * lat_rad
                       - (3.0*e2/8.0 + 3.0*e2*e2/32.0 + 45.0*e2*e2*e2/1024.0) * std::sin(2.0*lat_rad)
                       + (15.0*e2*e2/256.0 + 45.0*e2*e2*e2/1024.0) * std::sin(4.0*lat_rad)
                       - (35.0*e2*e2*e2/3072.0) * std::sin(6.0*lat_rad));

        easting = false_easting + k0 * N * (A + (1.0 - T + C) * std::pow(A, 3) / 6.0
                                           + (5.0 - 18.0*T + T*T + 72.0*C - 58.0*e_prime_sq) * std::pow(A, 5) / 120.0);

        northing = false_northing + k0 * (M + N * std::tan(lat_rad) * (std::pow(A, 2) / 2.0
                                                                      + (5.0 - T + 9.0*C + 4.0*C*C) * std::pow(A, 4) / 24.0
                                                                      + (61.0 - 58.0*T + T*T + 600.0*C - 330.0*e_prime_sq) * std::pow(A, 6) / 720.0));
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
        temp_start_node_id_ = -3;
        
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
        detailed_path.header.frame_id = "odom";
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
        
        // Convert path nodes to MapNode messages
        detailed_path.path_data.nodes.clear();
        for (size_t i = 0; i < path_nodes.size(); ++i) {
            gmserver::msg::MapNode map_node;
            
            int node_idx = path_nodes[i]->id;
            
            // Temporary nodes에 대한 처리
            if (node_idx == temp_start_node_id_) {
                map_node.id = "GPS_START";
                map_node.remark = "Temporary start node from GPS position";
            } else if (node_idx == temp_goal_node_id_) {
                map_node.id = "GPS_GOAL";
                map_node.remark = "Temporary goal node from RViz goal";
            } else if (node_idx >= 0 && node_idx < static_cast<int>(graph_map_.map_data.nodes.size())) {
                // 실제 맵 노드에서 정보 복사
                map_node = graph_map_.map_data.nodes[node_idx];
            } else {
                // Fallback for unknown nodes
                map_node.id = "NODE_" + std::to_string(node_idx);
                map_node.remark = "Unknown node";
            }
            
            // UTM 좌표는 그대로 유지 (visualization용은 따로 조정됨)
            if (node_idx >= 0 && node_idx < static_cast<int>(graph_map_.map_data.nodes.size()) &&
                node_idx != temp_start_node_id_ && node_idx != temp_goal_node_id_) {
                // 실제 맵 노드의 경우 GPS 정보는 원본 유지, UTM은 odom frame으로 변환
                map_node.gps_info = graph_map_.map_data.nodes[node_idx].gps_info;
                map_node.utm_info = graph_map_.map_data.nodes[node_idx].utm_info;
                // UTM 좌표를 odom frame으로 변환 (일관성을 위해)
                map_node.utm_info.easting -= gps_ref_utm_easting_;
                map_node.utm_info.northing -= gps_ref_utm_northing_;
            } else {
                // Temporary 노드의 경우 pose에서 역산
                map_node.gps_info.lat = 0.0; // GPS 역변환은 복잡하므로 생략
                map_node.gps_info.longitude = 0.0;
                map_node.gps_info.alt = path_nodes[i]->pose.position.z;
                // UTM 좌표를 odom frame으로 변환 (gps_ref_utm offset 제거)
                map_node.utm_info.easting = path_nodes[i]->pose.position.x - gps_ref_utm_easting_;
                map_node.utm_info.northing = path_nodes[i]->pose.position.y - gps_ref_utm_northing_;
                map_node.utm_info.zone = "52N"; // K-City 기본 zone
            }
            
            detailed_path.path_data.nodes.push_back(map_node);
        }
        
        // Create links between consecutive path nodes
        detailed_path.path_data.links.clear();
        for (size_t i = 1; i < path_nodes.size(); ++i) {
            gmserver::msg::MapLink map_link;
            
            int from_node_idx = path_nodes[i-1]->id;
            int to_node_idx = path_nodes[i]->id;
            
            // Set link metadata
            map_link.id = "PATH_LINK_" + std::to_string(i-1) + "_" + std::to_string(i);
            map_link.from_node_id = detailed_path.path_data.nodes[i-1].id;
            map_link.to_node_id = detailed_path.path_data.nodes[i].id;
            
            // Calculate link length
            double distance = calculateDistance(path_nodes[i-1]->pose, path_nodes[i]->pose);
            map_link.length = distance / 1000.0; // Convert to km
            
            // Try to find existing link in graph for more details
            bool found_existing_link = false;
            if (from_node_idx >= 0 && from_node_idx < static_cast<int>(graph_map_.map_data.nodes.size()) &&
                to_node_idx >= 0 && to_node_idx < static_cast<int>(graph_map_.map_data.nodes.size()) &&
                from_node_idx != temp_start_node_id_ && from_node_idx != temp_goal_node_id_ &&
                to_node_idx != temp_start_node_id_ && to_node_idx != temp_goal_node_id_) {
                
                std::string from_id = graph_map_.map_data.nodes[from_node_idx].id;
                std::string to_id = graph_map_.map_data.nodes[to_node_idx].id;
                
                // Find existing link in GraphMap
                for (const auto& original_link : graph_map_.map_data.links) {
                    if ((original_link.from_node_id == from_id && original_link.to_node_id == to_id) ||
                        (original_link.from_node_id == to_id && original_link.to_node_id == from_id)) {
                        // Copy original link information
                        map_link = original_link;
                        // Ensure correct direction
                        map_link.from_node_id = from_id;
                        map_link.to_node_id = to_id;
                        found_existing_link = true;
                        break;
                    }
                }
            }
            
            // If no existing link found, use calculated data
            if (!found_existing_link) {
                map_link.admin_code = "PATH";
                map_link.road_rank = 1;
                map_link.road_type = 1;
                map_link.link_type = 3;
                map_link.lane_no = 2;
                map_link.maker = "Path Planner";
                map_link.remark = "Generated path link";
            }
            
            detailed_path.path_data.links.push_back(map_link);
        }
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "Created detailed path: %zu nodes, %zu links, total distance: %.2f m",
                    detailed_path.path_data.nodes.size(), detailed_path.path_data.links.size(), total_distance);
        
        return detailed_path;
    }
    
    void publishMapToOdomTransform(const sensor_msgs::msg::NavSatFix& gps)
    {
        // Convert GPS to UTM coordinates
        double utm_easting, utm_northing;
        gpsToUTM(gps.latitude, gps.longitude, utm_easting, utm_northing);
        
        // Create transform from map to odom based on GPS position
        geometry_msgs::msg::TransformStamped transform_stamped;
        
        transform_stamped.header.stamp = this->get_clock()->now();
        transform_stamped.header.frame_id = "map";
        transform_stamped.child_frame_id = "odom";
        
        // Set translation to GPS UTM position relative to reference point
        transform_stamped.transform.translation.x = utm_easting - gps_ref_utm_easting_;
        transform_stamped.transform.translation.y = utm_northing - gps_ref_utm_northing_;
        transform_stamped.transform.translation.z = gps.altitude - gps_ref_alt_;
        
        // Set rotation from IMU if available, otherwise no rotation
        if (current_imu_received_) {
            // Use IMU orientation directly
            transform_stamped.transform.rotation = current_imu_.orientation;
        } else {
            // No rotation if IMU not available
            tf2::Quaternion q;
            q.setRPY(0, 0, 0);
            transform_stamped.transform.rotation.x = q.x();
            transform_stamped.transform.rotation.y = q.y();
            transform_stamped.transform.rotation.z = q.z();
            transform_stamped.transform.rotation.w = q.w();
        }
        
        // Broadcast the transform
        // tf_broadcaster_->sendTransform(transform_stamped);
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "Published map->odom transform: GPS(%.6f, %.6f) -> UTM(%.2f, %.2f) -> offset(%.2f, %.2f), IMU: %s",
                    gps.latitude, gps.longitude, utm_easting, utm_northing,
                    transform_stamped.transform.translation.x, transform_stamped.transform.translation.y,
                    current_imu_received_ ? "available" : "not available");
    }
    
    // Member variables
    rclcpp::Client<gmserver::srv::LoadMap>::SharedPtr map_client_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_subscriber_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    rclcpp::Publisher<command_center_interfaces::msg::PlannedPath>::SharedPtr planned_path_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr nodes_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr links_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr map_viz_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // GraphMap data from gmserver
    gmserver::msg::GraphMap graph_map_;
    
    // Converted data for compatibility with existing visualization
    geometry_msgs::msg::PoseArray map_nodes_;
    geometry_msgs::msg::PoseArray map_links_;
    std::vector<std::string> node_ids_;
    
    // Node ID to index mapping for efficient lookup
    std::unordered_map<std::string, int> node_id_to_index_;
    
    // GPS, IMU and Goal state
    sensor_msgs::msg::NavSatFix current_gps_;
    sensor_msgs::msg::Imu current_imu_;
    geometry_msgs::msg::PoseStamped goal_pose_;
    bool current_gps_received_;
    bool current_imu_received_;
    bool goal_received_;
    bool path_planned_for_current_goal_; // Flag to ensure single path planning per goal
    
    // GPS reference coordinates for goal transformation  
    double gps_ref_lat_;
    double gps_ref_lon_;
    double gps_ref_alt_;
    double gps_ref_utm_easting_;
    double gps_ref_utm_northing_;
    bool gps_ref_initialized_; // Flag to track GPS reference initialization
    
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