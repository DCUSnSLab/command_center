#ifndef LOCAL_COSTMAP_2D__COSTMAP_PUBLISHER_HPP_
#define LOCAL_COSTMAP_2D__COSTMAP_PUBLISHER_HPP_

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
// #include <rclcpp_lifecycle/lifecycle_node.hpp>  // Not needed for regular node
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <map_msgs/msg/occupancy_grid_update.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "local_costmap_2d/costmap_2d.hpp"

namespace local_costmap_2d
{

class CostmapPublisher
{
public:
  struct Config
  {
    Config() = default;
    
    std::string global_frame = "map";
    bool always_send_full_costmap = false;
    double publish_frequency = 10.0;
    
    // Visualization settings
    bool enable_obstacle_markers = true;
    bool enable_filtered_cloud = true;
    double marker_lifetime = 1.0;
    double obstacle_marker_size = 0.1;
    
    // Topic names
    std::string costmap_topic = "costmap";
    std::string costmap_updates_topic = "costmap_updates";
    std::string obstacle_markers_topic = "obstacle_markers";
    std::string filtered_cloud_topic = "filtered_pointcloud";
  };

  template<typename NodeType>
  CostmapPublisher(
    std::shared_ptr<NodeType> node,
    std::shared_ptr<SimpleCostmap2D> costmap,
    const Config& config)
  : costmap_(costmap), config_(config),
    grid_msg_(std::make_unique<nav_msgs::msg::OccupancyGrid>()),
    grid_update_msg_(std::make_unique<map_msgs::msg::OccupancyGridUpdate>()),
    x0_(0), xn_(0), y0_(0), yn_(0),
    saved_origin_x_(0.0), saved_origin_y_(0.0),
    has_updated_data_(false), active_(false),
    logger_(rclcpp::get_logger("local_costmap_2d.publisher"))
  {
    // Extract node interfaces
    node_base_ = node->get_node_base_interface();
    node_topics_ = node->get_node_topics_interface();
    node_logging_ = node->get_node_logging_interface();
    node_clock_ = node->get_node_clock_interface();
    
    // Initialize cost translation table
    if (cost_translation_table_.empty()) {
      initCostTranslationTable();
    }
    
    // Create publishers using node topics interface
    costmap_pub_ = rclcpp::create_publisher<nav_msgs::msg::OccupancyGrid>(
      node_topics_, config_.costmap_topic, rclcpp::QoS(1).transient_local());
    
    costmap_update_pub_ = rclcpp::create_publisher<map_msgs::msg::OccupancyGridUpdate>(
      node_topics_, config_.costmap_updates_topic, rclcpp::QoS(10));
    
    if (config_.enable_obstacle_markers) {
      obstacle_markers_pub_ = rclcpp::create_publisher<visualization_msgs::msg::MarkerArray>(
        node_topics_, config_.obstacle_markers_topic, rclcpp::QoS(1));
    }
    
    if (config_.enable_filtered_cloud) {
      filtered_cloud_pub_ = rclcpp::create_publisher<sensor_msgs::msg::PointCloud2>(
        node_topics_, config_.filtered_cloud_topic, rclcpp::QoS(1));
    }
    
    // Reset bounds
    resetBounds();
  }
  
  
  ~CostmapPublisher();
  
  // Publishing functions
  void publishCostmap();
  void publishCostmapUpdate(
    unsigned int x0, unsigned int y0, 
    unsigned int width, unsigned int height);
  void publishObstacleMarkers(
    const std::vector<geometry_msgs::msg::Point>& obstacle_points);
  void publishFilteredPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud);
  
  // Configuration
  void updateConfig(const Config& config);
  const Config& getConfig() const { return config_; }
  
  // Lifecycle support
  void on_activate();
  void on_deactivate();
  
  // Update bounds for incremental updates
  void updateBounds(unsigned int x0, unsigned int xn, unsigned int y0, unsigned int yn);
  void resetBounds();
  
  // Check if publisher is active
  bool isActive() const { return active_; }

private:
  // Helper functions
  void prepareGrid();
  void prepareGridUpdate(
    unsigned int x0, unsigned int y0, 
    unsigned int width, unsigned int height);
  void createObstacleMarkers(
    const std::vector<geometry_msgs::msg::Point>& obstacle_points,
    visualization_msgs::msg::MarkerArray& markers);
  
  // ROS node interfaces
  rclcpp::node_interfaces::NodeBaseInterface::SharedPtr node_base_;
  rclcpp::node_interfaces::NodeTopicsInterface::SharedPtr node_topics_;
  rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr node_logging_;
  rclcpp::node_interfaces::NodeClockInterface::SharedPtr node_clock_;
  
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_pub_;
  rclcpp::Publisher<map_msgs::msg::OccupancyGridUpdate>::SharedPtr costmap_update_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacle_markers_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
  
  // Data
  std::shared_ptr<SimpleCostmap2D> costmap_;
  Config config_;
  
  // Grid message cache
  std::unique_ptr<nav_msgs::msg::OccupancyGrid> grid_msg_;
  std::unique_ptr<map_msgs::msg::OccupancyGridUpdate> grid_update_msg_;
  
  // Update bounds tracking
  unsigned int x0_, xn_, y0_, yn_;
  double saved_origin_x_, saved_origin_y_;
  bool has_updated_data_;
  
  // State
  bool active_;
  
  // Cost translation table (costmap values to occupancy grid values)
  static std::vector<int8_t> cost_translation_table_;
  static void initCostTranslationTable();
  
  // Logging
  rclcpp::Logger logger_;
};

}  // namespace local_costmap_2d

#endif  // LOCAL_COSTMAP_2D__COSTMAP_PUBLISHER_HPP_