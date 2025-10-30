#include "local_costmap_2d/pointcloud_processor.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace local_costmap_2d
{

PointCloudProcessor::PointCloudProcessor(
  std::shared_ptr<SimpleCostmap2D> costmap,
  std::shared_ptr<tf2_ros::Buffer> tf_buffer,
  const Config& config)
: costmap_(costmap), tf_buffer_(tf_buffer), config_(config),
  filtered_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
  robot_pose_valid_(false),
  logger_(rclcpp::get_logger("local_costmap_2d.pointcloud_processor"))
{
  RCLCPP_INFO(logger_, "PointCloudProcessor initialized");
}

PointCloudProcessor::PointCloudProcessor(
  std::shared_ptr<SimpleCostmap2D> costmap,
  std::shared_ptr<tf2_ros::Buffer> tf_buffer)
: PointCloudProcessor(costmap, tf_buffer, Config())
{
}

PointCloudProcessor::~PointCloudProcessor()
{
  RCLCPP_INFO(logger_, "PointCloudProcessor destroyed");
}

bool PointCloudProcessor::processPointCloud(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg,
  const rclcpp::Time& current_time)
{
  if (!msg || msg->data.empty()) {
    RCLCPP_WARN(logger_, "Empty point cloud message received");
    return false;
  }
  
  rclcpp::Time target_time = (current_time.nanoseconds() == 0) ? 
    rclcpp::Time(msg->header.stamp) : current_time;
  
  // Transform point cloud to global frame
  sensor_msgs::msg::PointCloud2 transformed_cloud;
  if (!transformPointCloud(msg, transformed_cloud, target_time)) {
    RCLCPP_WARN(logger_, "Failed to transform point cloud");
    return false;
  }
  
  // Convert to PCL format
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(transformed_cloud, *cloud);
  
  if (cloud->points.empty()) {
    RCLCPP_WARN(logger_, "Empty point cloud after conversion");
    return false;
  }
  
  RCLCPP_DEBUG(logger_, "Processing point cloud with %zu points", cloud->points.size());
  
  // Apply filtering pipeline
  applyFilters(cloud);
  
  if (cloud->points.empty()) {
    RCLCPP_DEBUG(logger_, "All points filtered out");
    return true;
  }
  
  RCLCPP_DEBUG(logger_, "After filtering: %zu points remain", cloud->points.size());
  
  // Store filtered cloud for debugging
  *filtered_cloud_ = *cloud;
  
  // Apply temporal decay if enabled
  if (config_.enable_temporal_decay) {
    costmap_->applyTemporalDecay(config_.decay_rate);
  }
  
  // Project points to costmap grid
  projectToGrid(cloud);
  
  // Clear robot footprint
  clearRobotFootprint();
  
  // Update obstacle points list
  updateObstaclePoints(cloud);
  
  last_update_time_ = target_time;
  
  return true;
}

bool PointCloudProcessor::transformPointCloud(
  const sensor_msgs::msg::PointCloud2::SharedPtr input,
  sensor_msgs::msg::PointCloud2& output,
  const rclcpp::Time& target_time)
{
  try {
    // Check if transform is available
    if (!tf_buffer_->canTransform(
        config_.global_frame, input->header.frame_id, 
        target_time, rclcpp::Duration::from_nanoseconds(100000000))) {  // 100ms timeout
      RCLCPP_WARN(logger_, "Transform not available from %s to %s at time %f",
                  input->header.frame_id.c_str(), config_.global_frame.c_str(),
                  target_time.seconds());
      return false;
    }
    
    // Get transform
    geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
      config_.global_frame, input->header.frame_id, target_time);
    
    // Transform point cloud
    tf2::doTransform(*input, output, transform);
    
    return true;
    
  } catch (const tf2::TransformException& e) {
    RCLCPP_ERROR(logger_, "Transform exception: %s", e.what());
    return false;
  }
}

void PointCloudProcessor::applyFilters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  // 1. Range filtering
  PointCloudFilters::filterByRange(cloud, config_.max_range, config_.min_range);
  
  if (cloud->points.empty()) return;
  
  // 2. Height filtering
  PointCloudFilters::filterByHeight(
    cloud, config_.min_obstacle_height, config_.max_obstacle_height);
  
  if (cloud->points.empty()) return;
  
  // 3. Ground plane removal
  if (config_.enable_ground_removal) {
    PointCloudFilters::removeGroundPlane(
      cloud, config_.ground_distance_threshold, config_.ground_max_iterations);
  }
  
  if (cloud->points.empty()) return;
  
  // 4. Voxel grid downsampling
  if (config_.voxel_size > 0.0) {
    PointCloudFilters::voxelGridDownsample(cloud, config_.voxel_size);
  }
  
  if (cloud->points.empty()) return;
  
  // 5. Statistical outlier removal
  if (config_.enable_statistical_filter && 
      cloud->points.size() >= static_cast<size_t>(config_.statistical_mean_k)) {
    PointCloudFilters::removeOutliers(
      cloud, config_.statistical_mean_k, config_.statistical_std_dev);
  }
  
  if (cloud->points.empty()) return;
  
  // 6. Euclidean clustering to remove small noise clusters
  if (config_.min_cluster_size > 0) {
    PointCloudFilters::euclideanClustering(cloud, config_.min_cluster_size);
  }
  
  // 7. Filter robot footprint (if robot pose is available)
  if (robot_pose_valid_) {
    double robot_x = last_robot_pose_.transform.translation.x;
    double robot_y = last_robot_pose_.transform.translation.y;
    
    // Convert quaternion to yaw
    tf2::Quaternion q;
    tf2::fromMsg(last_robot_pose_.transform.rotation, q);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    
    PointCloudFilters::filterRobotFootprint(
      cloud, config_.robot_footprint, robot_x, robot_y, yaw);
  }
}

void PointCloudProcessor::projectToGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  for (const auto& point : cloud->points) {
    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
      costmap_->markObstacleWithHeight(point.x, point.y, point.z, LETHAL_OBSTACLE);
    }
  }
}

void PointCloudProcessor::clearRobotFootprint()
{
  if (!robot_pose_valid_) {
    return;
  }
  
  // Convert footprint to geometry_msgs format
  std::vector<geometry_msgs::msg::Point> footprint_points;
  for (const auto& point : config_.robot_footprint) {
    geometry_msgs::msg::Point p;
    p.x = point.first;
    p.y = point.second;
    p.z = 0.0;
    footprint_points.push_back(p);
  }
  
  double robot_x = last_robot_pose_.transform.translation.x;
  double robot_y = last_robot_pose_.transform.translation.y;
  
  // Convert quaternion to yaw
  tf2::Quaternion q;
  tf2::fromMsg(last_robot_pose_.transform.rotation, q);
  tf2::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  
  costmap_->clearRobotFootprint(footprint_points, robot_x, robot_y, yaw);
}

bool PointCloudProcessor::getRobotPose(
  geometry_msgs::msg::TransformStamped& transform,
  const rclcpp::Time& target_time)
{
  try {
    transform = tf_buffer_->lookupTransform(
      config_.global_frame, config_.robot_frame, target_time,
      rclcpp::Duration::from_nanoseconds(100000000));  // 100ms timeout
    
    last_robot_pose_ = transform;
    robot_pose_valid_ = true;
    return true;
    
  } catch (const tf2::TransformException& e) {
    static rclcpp::Clock clock;
    RCLCPP_WARN_THROTTLE(logger_, clock, 1000,
                          "Failed to get robot pose: %s", e.what());
    robot_pose_valid_ = false;
    return false;
  }
}

void PointCloudProcessor::updateObstaclePoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  obstacle_points_.clear();
  obstacle_points_.reserve(cloud->points.size());
  
  for (const auto& point : cloud->points) {
    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
      geometry_msgs::msg::Point obstacle_point;
      obstacle_point.x = point.x;
      obstacle_point.y = point.y;
      obstacle_point.z = point.z;
      obstacle_points_.push_back(obstacle_point);
    }
  }
}

void PointCloudProcessor::updateConfig(const Config& config)
{
  config_ = config;
  RCLCPP_INFO(logger_, "PointCloudProcessor configuration updated");
}

const std::vector<geometry_msgs::msg::Point>& PointCloudProcessor::getObstaclePoints() const
{
  return obstacle_points_;
}

}  // namespace local_costmap_2d