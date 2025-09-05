#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <gmserver/srv/load_map.hpp>
#include <gmserver/msg/graph_map.hpp>
#include <gmserver/msg/map_data.hpp>
#include <gmserver/msg/map_node.hpp>
#include <gmserver/msg/map_link.hpp>
#include <gmserver/msg/gps_info.hpp>
#include <gmserver/msg/utm_info.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>

using json = nlohmann::json;

class MapServiceNode : public rclcpp::Node
{
public:
    MapServiceNode() : Node("map_service_node")
    {
        // Create service for loading map data
        service_ = this->create_service<gmserver::srv::LoadMap>(
            "load_map",
            std::bind(&MapServiceNode::handleLoadMapService, this,
                     std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(this->get_logger(), "Map service node initialized. Service '/load_map' is ready.");
    }

private:
    // Helper function to calculate heading between two UTM points
    double calculateHeading(double from_easting, double from_northing, 
                           double to_easting, double to_northing)
    {
        double dx = to_easting - from_easting;
        double dy = to_northing - from_northing;
        
        // Calculate heading in radians (atan2 returns -pi to pi)
        double heading_rad = std::atan2(dy, dx);
        
        // Convert to degrees and normalize to 0-360
        double heading_deg = heading_rad * 180.0 / M_PI;
        if (heading_deg < 0) {
            heading_deg += 360.0;
        }
        
        return heading_deg;
    }
    
    // Calculate headings for nodes with heading = 0.0 based on link connectivity
    void calculateNodeHeadings(std::shared_ptr<gmserver::srv::LoadMap::Response> response)
    {
        auto& nodes = response->graph_map.map_data.nodes;
        const auto& links = response->graph_map.map_data.links;
        
        // Build a map of node ID to index for fast lookup
        std::unordered_map<std::string, size_t> node_id_to_index;
        for (size_t i = 0; i < nodes.size(); ++i) {
            node_id_to_index[nodes[i].id] = i;
        }
        
        // Build adjacency map to find incoming links (previous nodes)
        std::unordered_map<std::string, std::string> incoming_node_map;
        for (const auto& link : links) {
            incoming_node_map[link.to_node_id] = link.from_node_id;
        }
        
        // Calculate headings for nodes with heading = 0.0
        for (auto& node : nodes) {
            if (std::abs(node.heading + 1.0) < 1e-6) {  // heading is approximately 0.0
                // Find the previous node via incoming link
                auto incoming_it = incoming_node_map.find(node.id);
                if (incoming_it != incoming_node_map.end()) {
                    const std::string& prev_node_id = incoming_it->second;
                    auto prev_node_it = node_id_to_index.find(prev_node_id);
                    
                    if (prev_node_it != node_id_to_index.end()) {
                        const auto& prev_node = nodes[prev_node_it->second];
                        
                        // Calculate heading from previous node to current node
                        double calculated_heading = calculateHeading(
                            prev_node.utm_info.easting, prev_node.utm_info.northing,
                            node.utm_info.easting, node.utm_info.northing
                        );
                        
                        node.heading = calculated_heading;
                        
                        RCLCPP_DEBUG(this->get_logger(), 
                                   "Calculated heading for node %s: %.2f degrees (from node %s)",
                                   node.id.c_str(), calculated_heading, prev_node_id.c_str());
                    }
                } else {
                    RCLCPP_DEBUG(this->get_logger(), 
                               "Node %s has heading=0.0 but no incoming link found", 
                               node.id.c_str());
                }
            }
        }
    }
    
    void handleLoadMapService(
        const std::shared_ptr<gmserver::srv::LoadMap::Request> request,
        std::shared_ptr<gmserver::srv::LoadMap::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Loading map from: %s", request->map_file_path.c_str());
        
        try {
            if (loadMapFromJSON(request->map_file_path, response)) {
                response->success = true;
                response->message = "Map loaded successfully";
                RCLCPP_INFO(this->get_logger(), "Map loaded successfully: %zu nodes, %zu links", 
                           response->graph_map.map_data.nodes.size(), response->graph_map.map_data.links.size());
            } else {
                response->success = false;
                response->message = "Failed to load map file";
                RCLCPP_ERROR(this->get_logger(), "Failed to load map from: %s", 
                            request->map_file_path.c_str());
            }
        } catch (const std::exception& e) {
            response->success = false;
            response->message = std::string("Exception: ") + e.what();
            RCLCPP_ERROR(this->get_logger(), "Exception while loading map: %s", e.what());
        }
    }
    
    bool loadMapFromJSON(const std::string& file_path, 
                        std::shared_ptr<gmserver::srv::LoadMap::Response> response)
    {
        try {
            std::ifstream file(file_path);
            if (!file.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Cannot open file: %s", file_path.c_str());
                return false;
            }
            
            json map_json;
            file >> map_json;
            
            // Initialize GraphMap
            response->graph_map.map_name = "3x3_map";
            response->graph_map.map_version = "2021";
            response->graph_map.creation_date = "20250418";
            response->graph_map.description = "3x3 Grid Map for ROS2";
            
            // Clear map data arrays
            response->graph_map.map_data.nodes.clear();
            response->graph_map.map_data.links.clear();
            
            // Parse Node array from JSON
            if (map_json.contains("Node") && map_json["Node"].is_array()) {
                for (const auto& node_json : map_json["Node"]) {
                    gmserver::msg::MapNode map_node;
                    
                    // Basic node information
                    map_node.id = node_json.value("ID", "");
                    map_node.admin_code = node_json.value("AdminCode", "");
                    map_node.node_type = node_json.value("NodeType", 0);
                    map_node.its_node_id = node_json.value("ITSNodeID", "");
                    map_node.maker = node_json.value("Maker", "");
                    map_node.update_date = node_json.value("UpdateDate", "");
                    map_node.version = node_json.value("Version", "");
                    map_node.remark = node_json.value("Remark", "");
                    map_node.hist_type = node_json.value("HistType", "");
                    map_node.hist_remark = node_json.value("HistRemark", "");
                    // Handle heading field - use value if present, default to 0.0 if missing
                    if (node_json.contains("Heading")) {
                        map_node.heading = node_json.value("Heading", 0.0);
                    } else {
                        map_node.heading = 0.0;
                        RCLCPP_DEBUG(this->get_logger(), "Node %s has no Heading key, using default 0.0", 
                                   map_node.id.c_str());
                    }
                    
                    // GPS Information
                    if (node_json.contains("GpsInfo")) {
                        const auto& gps = node_json["GpsInfo"];
                        map_node.gps_info.lat = gps.value("Lat", 0.0);
                        map_node.gps_info.longitude = gps.value("Long", 0.0);
                        map_node.gps_info.alt = gps.value("Alt", 0.0);
                    }
                    
                    // UTM Information
                    if (node_json.contains("UtmInfo")) {
                        const auto& utm = node_json["UtmInfo"];
                        map_node.utm_info.easting = utm.value("Easting", 0.0);
                        map_node.utm_info.northing = utm.value("Northing", 0.0);
                        map_node.utm_info.zone = utm.value("Zone", "");
                    }
                    
                    response->graph_map.map_data.nodes.push_back(map_node);
                }
            }
            
            // Parse Link array from JSON
            if (map_json.contains("Link") && map_json["Link"].is_array()) {
                for (const auto& link_json : map_json["Link"]) {
                    gmserver::msg::MapLink map_link;
                    
                    // Basic link information
                    map_link.id = link_json.value("ID", "");
                    map_link.admin_code = link_json.value("AdminCode", "");
                    map_link.road_rank = link_json.value("RoadRank", 0);
                    map_link.road_type = link_json.value("RoadType", 0);
                    map_link.road_no = link_json.value("RoadNo", "");
                    map_link.link_type = link_json.value("LinkType", 0);
                    map_link.lane_no = link_json.value("LaneNo", 0);
                    map_link.r_link_id = link_json.value("R_LinkID", "");
                    map_link.l_link_id = link_json.value("L_LinkID", "");
                    map_link.from_node_id = link_json.value("FromNodeID", "");
                    map_link.to_node_id = link_json.value("ToNodeID", "");
                    map_link.section_id = link_json.value("SectionID", "");
                    map_link.length = link_json.value("Length", 0.0);
                    map_link.its_link_id = link_json.value("ITSLinkID", "");
                    map_link.maker = link_json.value("Maker", "");
                    map_link.update_date = link_json.value("UpdateDate", "");
                    map_link.version = link_json.value("Version", "");
                    map_link.remark = link_json.value("Remark", "");
                    map_link.hist_type = link_json.value("HistType", "");
                    map_link.hist_remark = link_json.value("HistRemark", "");
                    
                    response->graph_map.map_data.links.push_back(map_link);
                }
            }
            
            // Calculate headings for nodes with heading = 0.0
            calculateNodeHeadings(response);
            
            RCLCPP_INFO(this->get_logger(), "Parsed %zu nodes and %zu links from JSON", 
                       response->graph_map.map_data.nodes.size(), response->graph_map.map_data.links.size());
            return true;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error parsing JSON: %s", e.what());
            return false;
        }
    }
    
    rclcpp::Service<gmserver::srv::LoadMap>::SharedPtr service_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapServiceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}