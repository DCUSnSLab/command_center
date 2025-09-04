#include <memory>
#include <chrono>
#include <string>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/time.h>

#include "local_costmap_2d/costmap_2d.hpp"
#include "local_costmap_2d/pointcloud_processor.hpp"
#include "local_costmap_2d/costmap_publisher.hpp"

using namespace std::chrono_literals;

namespace local_costmap_2d
{

class LocalCostmapNode : public rclcpp::Node
{
public:
  LocalCostmapNode()
  : rclcpp::Node("local_costmap"),
    tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
    tf_listener_(std::make_shared<tf2_ros::TransformListener>(*tf_buffer_)),
    initialized_(false),
    robot_x_(0.0), robot_y_(0.0),
    has_odom_(false)
  {
    RCLCPP_INFO(this->get_logger(), "LocalCostmapNode constructed");
    
    // Declare parameters
    declareParameters();
    
    // Delay initialization using a timer
    init_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&LocalCostmapNode::initializeDelayed, this));
  }

private:
  void initializeDelayed()
  {
    if (initialized_) {
      return;
    }
    
    try {
      initialize();
      initialized_ = true;
      init_timer_.reset();  // Stop the initialization timer
      RCLCPP_INFO(this->get_logger(), "LocalCostmapNode initialized successfully");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize: %s", e.what());
      // Keep trying with the timer
    }
  }

  void initialize()
  {
    // Load parameters
    loadParameters();
    
    // Initialize costmap
    costmap_ = std::make_shared<SimpleCostmap2D>(
      static_cast<unsigned int>(map_width_ / resolution_),
      static_cast<unsigned int>(map_height_ / resolution_),
      resolution_,
      -map_width_ / 2.0,  // Center the map on robot
      -map_height_ / 2.0,
      default_value_);
    
    // Initialize processor
    PointCloudProcessor::Config processor_config;
    loadProcessorConfig(processor_config);
    processor_ = std::make_unique<PointCloudProcessor>(
      costmap_, tf_buffer_, processor_config);
    
    // Initialize publisher
    CostmapPublisher::Config publisher_config;
    loadPublisherConfig(publisher_config);
    publisher_ = std::make_unique<CostmapPublisher>(
      shared_from_this(), costmap_, publisher_config);
    
    // Activate publisher immediately
    publisher_->on_activate();
    
    // Create subscriptions
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      pointcloud_topic_, rclcpp::QoS(1),
      std::bind(&LocalCostmapNode::pointcloudCallback, this, std::placeholders::_1));
    
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(10),
      std::bind(&LocalCostmapNode::odomCallback, this, std::placeholders::_1));
    
    // Create timer for publishing
    publish_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / publish_frequency_)),
      std::bind(&LocalCostmapNode::publishTimerCallback, this));
  }

private:
  void declareParameters()
  {
    // Costmap parameters
    this->declare_parameter("resolution", 0.05);
    this->declare_parameter("width", 20.0);
    this->declare_parameter("height", 20.0);
    this->declare_parameter("publish_frequency", 10.0);
    this->declare_parameter("update_frequency", 20.0);
    
    // Topic names
    this->declare_parameter("pointcloud_topic", "/points");
    this->declare_parameter("odom_topic", "/odom");
    this->declare_parameter("global_frame", "odom");
    this->declare_parameter("robot_frame", "base_link");
    this->declare_parameter("sensor_frame", "velodyne");
    
    // Processing parameters
    this->declare_parameter("max_range", 100.0);
    this->declare_parameter("min_range", 0.1);
    this->declare_parameter("min_obstacle_height", 0.1);
    this->declare_parameter("max_obstacle_height", 2.0);
    this->declare_parameter("voxel_size", 0.05);
    this->declare_parameter("enable_ground_removal", true);
    this->declare_parameter("enable_temporal_decay", true);
    this->declare_parameter("decay_rate", 0.95);
    
    // Robot footprint
    this->declare_parameter("robot_footprint", std::vector<double>{
      0.49, 0.3725, 0.49, -0.3725, -0.49, -0.3725, -0.49, 0.3725});
  }

  void loadParameters()
  {
    resolution_ = this->get_parameter("resolution").as_double();
    map_width_ = this->get_parameter("width").as_double();
    map_height_ = this->get_parameter("height").as_double();
    publish_frequency_ = this->get_parameter("publish_frequency").as_double();
    update_frequency_ = this->get_parameter("update_frequency").as_double();
    
    pointcloud_topic_ = this->get_parameter("pointcloud_topic").as_string();
    odom_topic_ = this->get_parameter("odom_topic").as_string();
    global_frame_ = this->get_parameter("global_frame").as_string();
    robot_frame_ = this->get_parameter("robot_frame").as_string();
    sensor_frame_ = this->get_parameter("sensor_frame").as_string();
    
    default_value_ = static_cast<unsigned char>(
      local_costmap_2d::FREE_SPACE);  // Default to free space
    
    RCLCPP_INFO(this->get_logger(), 
                "Loaded parameters: resolution=%.3f, size=%.1fx%.1f, freq=%.1f Hz",
                resolution_, map_width_, map_height_, publish_frequency_);
  }

  void loadProcessorConfig(PointCloudProcessor::Config& config)
  {
    config.max_range = this->get_parameter("max_range").as_double();
    config.min_range = this->get_parameter("min_range").as_double();
    config.min_obstacle_height = this->get_parameter("min_obstacle_height").as_double();
    config.max_obstacle_height = this->get_parameter("max_obstacle_height").as_double();
    config.voxel_size = this->get_parameter("voxel_size").as_double();
    config.enable_ground_removal = this->get_parameter("enable_ground_removal").as_bool();
    config.enable_temporal_decay = this->get_parameter("enable_temporal_decay").as_bool();
    config.decay_rate = this->get_parameter("decay_rate").as_double();
    
    config.sensor_frame = sensor_frame_;
    config.robot_frame = robot_frame_;
    config.global_frame = global_frame_;
    
    // Load robot footprint
    auto footprint_param = this->get_parameter("robot_footprint").as_double_array();
    config.robot_footprint.clear();
    for (size_t i = 0; i + 1 < footprint_param.size(); i += 2) {
      config.robot_footprint.emplace_back(footprint_param[i], footprint_param[i + 1]);
    }
  }

  void loadPublisherConfig(CostmapPublisher::Config& config)
  {
    config.global_frame = global_frame_;
    config.publish_frequency = publish_frequency_;
    config.always_send_full_costmap = false;
    config.enable_obstacle_markers = true;
    config.enable_filtered_cloud = true;
  }

  void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!processor_) {
      return;
    }
    
    rclcpp::Time start_time = this->get_clock()->now();
    
    bool success = processor_->processPointCloud(msg, this->get_clock()->now());
    
    if (success) {
      rclcpp::Duration processing_time = this->get_clock()->now() - start_time;
      RCLCPP_DEBUG(this->get_logger(), 
                   "Processed point cloud in %.1f ms with %zu points",
                   processing_time.seconds() * 1000.0,
                   processor_->getFilteredCloud()->points.size());
    } else {
      RCLCPP_WARN(this->get_logger(), "Failed to process point cloud");
    }
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Update robot position from odometry
    robot_x_ = msg->pose.pose.position.x;
    robot_y_ = msg->pose.pose.position.y;
    has_odom_ = true;
    
    RCLCPP_DEBUG(this->get_logger(), "Updated robot pose: (%.2f, %.2f)", robot_x_, robot_y_);
  }

  void publishTimerCallback()
  {
    if (!initialized_ || !publisher_ || !publisher_->isActive()) {
      return;
    }
    
    // Update costmap origin based on robot position
    updateCostmapOrigin();
    
    // Publish costmap
    publisher_->publishCostmap();
    
    // Publish obstacle markers and filtered point cloud if processor is available
    if (processor_) {
      publisher_->publishObstacleMarkers(processor_->getObstaclePoints());
      publisher_->publishFilteredPointCloud(processor_->getFilteredCloud());
    }
  }

  void updateCostmapOrigin()
  {
    if (!costmap_ || !has_odom_) {
      return;
    }
    
    // Calculate new origin to center the map on robot
    double new_origin_x = robot_x_ - map_width_ / 2.0;
    double new_origin_y = robot_y_ - map_height_ / 2.0;
    
    // Get current origin
    double current_origin_x = costmap_->getOriginX();
    double current_origin_y = costmap_->getOriginY();
    
    // Update only if robot moved significantly (avoid unnecessary updates)
    double distance_moved = sqrt(
      pow(new_origin_x - current_origin_x, 2) + 
      pow(new_origin_y - current_origin_y, 2));
    
    if (distance_moved > resolution_) {  // Move threshold: 1 cell
      // Resize costmap with new origin
      costmap_->resizeMap(
        static_cast<unsigned int>(map_width_ / resolution_),
        static_cast<unsigned int>(map_height_ / resolution_),
        resolution_,
        new_origin_x,
        new_origin_y);
      
      RCLCPP_DEBUG(this->get_logger(), 
        "Updated costmap origin to (%.2f, %.2f) for robot at (%.2f, %.2f)",
        new_origin_x, new_origin_y, robot_x_, robot_y_);
    }
  }

  // Parameters
  double resolution_;
  double map_width_, map_height_;
  double publish_frequency_, update_frequency_;
  std::string pointcloud_topic_, odom_topic_, global_frame_, robot_frame_, sensor_frame_;
  unsigned char default_value_;
  
  // Robot pose from odometry
  double robot_x_, robot_y_;
  bool has_odom_;
  
  // Core components
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<SimpleCostmap2D> costmap_;
  std::unique_ptr<PointCloudProcessor> processor_;
  std::unique_ptr<CostmapPublisher> publisher_;
  
  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr init_timer_;
  
  // Initialization state
  bool initialized_;
};

}  // namespace local_costmap_2d

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<local_costmap_2d::LocalCostmapNode>();
  
  RCLCPP_INFO(node->get_logger(), "Starting LocalCostmapNode");
  
  rclcpp::spin(node);
  
  rclcpp::shutdown();
  return 0;
}