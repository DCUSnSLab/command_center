#ifndef LOCAL_COSTMAP__COSTMAP_NODE_HPP_
#define LOCAL_COSTMAP__COSTMAP_NODE_HPP_

#include <memory>
#include <string>
#include <mutex>
#include <atomic>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "local_costmap/costmap_2d.hpp"
#include "local_costmap/point_cloud_processor.hpp"
#include "local_costmap/inflation_layer.hpp"
#include "local_costmap/denoise_layer.hpp"

namespace local_costmap
{

struct RobotPose
{
  double x;
  double y;
  double z;
  double yaw;
  bool valid;
};

class CostmapNode : public rclcpp::Node
{
public:
  explicit CostmapNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~CostmapNode() override = default;

private:
  // Parameters
  void declareParameters();
  void loadParameters();

  // Callbacks
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg);
  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg);
  void mainLoopCallback();

  // Costmap operations
  void updateCostmap();
  void publishCostmap();

  // TF lookup
  bool lookupTransform(
    const std::string& target_frame,
    const std::string& source_frame,
    geometry_msgs::msg::TransformStamped& transform);

  // Parameters
  std::string point_cloud_topic_;
  std::string odom_topic_;
  std::string odom_frame_;
  std::string base_frame_;
  std::string sensor_frame_;

  double costmap_width_;
  double costmap_height_;
  double costmap_resolution_;
  double min_obstacle_height_;
  double max_obstacle_height_;
  double update_frequency_;
  double inflation_radius_;
  double cost_scaling_factor_;
  double sensor_timeout_;
  double raytrace_max_range_;
  double obstacle_max_range_;
  bool track_unknown_space_;
  int denoise_minimal_group_size_;
  std::vector<double> robot_footprint_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_pub_;
  rclcpp::TimerBase::SharedPtr main_timer_;

  // TF
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Core components
  std::unique_ptr<Costmap2D> obstacle_map_;  // persistent marking/clearing grid
  std::unique_ptr<Costmap2D> costmap_;       // published grid (obstacles + inflation)
  std::unique_ptr<PointCloudProcessor> pc_processor_;
  std::unique_ptr<InflationLayer> inflation_layer_;
  DenoiseLayer denoise_layer_;

  // State
  std::mutex pc_mutex_;
  ProcessedCloud latest_cloud_;
  double sensor_origin_x_;
  double sensor_origin_y_;
  std::atomic<bool> has_new_points_;
  bool cloud_received_;
  rclcpp::Time last_cloud_time_;

  std::mutex odom_mutex_;
  RobotPose robot_pose_;

  // Pre-allocated message for publishing
  nav_msgs::msg::OccupancyGrid costmap_msg_;
};

}  // namespace local_costmap

#endif  // LOCAL_COSTMAP__COSTMAP_NODE_HPP_
