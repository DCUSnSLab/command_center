#ifndef LOCAL_COSTMAP_2D__POINTCLOUD_PROCESSOR_HPP_
#define LOCAL_COSTMAP_2D__POINTCLOUD_PROCESSOR_HPP_

#include <memory>
#include <vector>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "local_costmap_2d/costmap_2d.hpp"
#include "local_costmap_2d/pointcloud_filters.hpp"
#include "local_costmap_2d/cost_values.hpp"

namespace local_costmap_2d
{

class PointCloudProcessor
{
public:
  struct Config
  {
    Config() = default;
    
    // Range filtering
    double max_range = 100.0;
    double min_range = 0.1;
    
    // Height filtering
    double min_obstacle_height = 0.1;
    double max_obstacle_height = 2.0;
    double ground_height_threshold = 0.05;
    
    // Filtering parameters
    double voxel_size = 0.05;
    double noise_threshold = 0.02;
    int min_cluster_size = 10;
    
    // Statistical filtering
    bool enable_statistical_filter = true;
    int statistical_mean_k = 50;
    double statistical_std_dev = 1.0;
    
    // Ground removal
    bool enable_ground_removal = true;
    double ground_distance_threshold = 0.1;
    int ground_max_iterations = 1000;
    
    // Temporal decay
    bool enable_temporal_decay = true;
    double decay_rate = 0.95;
    
    // TF frames
    std::string sensor_frame = "velodyne";
    std::string robot_frame = "base_link";
    std::string global_frame = "map";
    
    // Robot footprint (in robot_frame coordinates)
    std::vector<std::pair<double, double>> robot_footprint = {
      {0.49, 0.3725}, {0.49, -0.3725}, {-0.49, -0.3725}, {-0.49, 0.3725}
    };
    double footprint_padding = 0.15;
  };

  PointCloudProcessor(
    std::shared_ptr<SimpleCostmap2D> costmap,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer,
    const Config& config);
  
  PointCloudProcessor(
    std::shared_ptr<SimpleCostmap2D> costmap,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer);
  
  ~PointCloudProcessor();
  
  // Main processing function
  bool processPointCloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg,
    const rclcpp::Time& current_time = rclcpp::Time(0));
  
  // Configuration
  void updateConfig(const Config& config);
  const Config& getConfig() const { return config_; }
  
  // Get processed obstacle points
  const std::vector<geometry_msgs::msg::Point>& getObstaclePoints() const;
  
  // Get filtered point cloud (for debugging/visualization)
  pcl::PointCloud<pcl::PointXYZ>::Ptr getFilteredCloud() const { return filtered_cloud_; }

private:
  // Processing pipeline steps
  bool transformPointCloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr input,
    sensor_msgs::msg::PointCloud2& output,
    const rclcpp::Time& target_time);
  
  void applyFilters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  void projectToGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  void clearRobotFootprint();
  
  // Helper functions
  bool getRobotPose(
    geometry_msgs::msg::TransformStamped& transform,
    const rclcpp::Time& target_time);
  
  void updateObstaclePoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  
  // Member variables
  std::shared_ptr<SimpleCostmap2D> costmap_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  Config config_;
  
  // Processing data
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_;
  std::vector<geometry_msgs::msg::Point> obstacle_points_;
  
  // Robot state
  geometry_msgs::msg::TransformStamped last_robot_pose_;
  rclcpp::Time last_update_time_;
  bool robot_pose_valid_;
  
  // Logging
  rclcpp::Logger logger_;
};

}  // namespace local_costmap_2d

#endif  // LOCAL_COSTMAP_2D__POINTCLOUD_PROCESSOR_HPP_