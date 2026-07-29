#include "local_costmap/costmap_node.hpp"

#include <chrono>
#include <cmath>
#include <functional>

using namespace std::chrono_literals;
using std::placeholders::_1;

namespace local_costmap
{

CostmapNode::CostmapNode(const rclcpp::NodeOptions& options)
  : Node("local_costmap_node", options),
    sensor_origin_x_(0.0),
    sensor_origin_y_(0.0),
    has_new_points_(false),
    cloud_received_(false)
{
  // Initialize robot pose
  robot_pose_.valid = false;

  // Declare and load parameters
  declareParameters();
  loadParameters();

  // Initialize TF
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Initialize costmaps: persistent obstacle grid + published grid
  double origin_x = -costmap_width_ / 2.0;
  double origin_y = -costmap_height_ / 2.0;
  obstacle_map_ = std::make_unique<Costmap2D>(
    costmap_width_, costmap_height_, costmap_resolution_,
    origin_x, origin_y);
  obstacle_map_->reset(track_unknown_space_ ? Costmap2D::UNKNOWN : Costmap2D::FREE_SPACE);
  costmap_ = std::make_unique<Costmap2D>(
    costmap_width_, costmap_height_, costmap_resolution_,
    origin_x, origin_y);

  // Initialize point cloud processor
  pc_processor_ = std::make_unique<PointCloudProcessor>();
  pc_processor_->setHeightFilter(min_obstacle_height_, max_obstacle_height_);
  pc_processor_->setRobotFootprint(robot_footprint_);

  // Initialize inflation layer
  inflation_layer_ = std::make_unique<InflationLayer>();
  inflation_layer_->initialize(inflation_radius_, cost_scaling_factor_, costmap_resolution_);

  // Initialize denoise layer
  denoise_layer_.initialize(denoise_minimal_group_size_);

  // QoS settings
  rclcpp::QoS qos(10);
  qos.reliability(rclcpp::ReliabilityPolicy::Reliable);

  // Create subscribers
  // Sensor drivers typically publish best_effort; SensorDataQoS matches both
  pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    point_cloud_topic_, rclcpp::SensorDataQoS(),
    std::bind(&CostmapNode::pointCloudCallback, this, _1));

  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_, qos,
    std::bind(&CostmapNode::odomCallback, this, _1));

  // Create publisher
  costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/costmap", qos);

  // Pre-allocate costmap message
  costmap_msg_.header.frame_id = odom_frame_;
  costmap_msg_.info.resolution = static_cast<float>(costmap_resolution_);
  costmap_msg_.info.width = costmap_->getWidthCells();
  costmap_msg_.info.height = costmap_->getHeightCells();
  costmap_msg_.info.origin.orientation.w = 1.0;
  costmap_msg_.data.resize(costmap_->getDataSize());

  // Create main loop timer
  double timer_period = 1.0 / update_frequency_;
  main_timer_ = this->create_wall_timer(
    std::chrono::duration<double>(timer_period),
    std::bind(&CostmapNode::mainLoopCallback, this));

  RCLCPP_INFO(this->get_logger(), "Local costmap node initialized");
  RCLCPP_INFO(this->get_logger(), "  PointCloud topic: %s", point_cloud_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "  Odometry topic: %s", odom_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "  Costmap size: %.1f x %.1f m (%.0f x %.0f cells)",
    costmap_width_, costmap_height_,
    static_cast<double>(costmap_->getWidthCells()),
    static_cast<double>(costmap_->getHeightCells()));
}

void CostmapNode::declareParameters()
{
  // Topic configuration
  this->declare_parameter("point_cloud_topic", "/velodyne_points");
  this->declare_parameter("odom_topic", "/odom");

  // Frame IDs
  this->declare_parameter("odom_frame", "odom");
  this->declare_parameter("base_frame", "base_link");
  this->declare_parameter("sensor_frame", "velodyne");

  // Costmap dimensions
  this->declare_parameter("costmap_width", 20.0);
  this->declare_parameter("costmap_height", 20.0);
  this->declare_parameter("costmap_resolution", 0.1);

  // Obstacle detection
  this->declare_parameter("min_obstacle_height", 0.5);
  this->declare_parameter("max_obstacle_height", 2.0);

  // Update frequency
  this->declare_parameter("update_frequency", 10.0);

  // Sensor timeout: if no point cloud arrives within this time,
  // the costmap is cleared to UNKNOWN instead of publishing stale data
  this->declare_parameter("sensor_timeout", 1.0);

  // Raytracing / persistence (Nav2 obstacle_layer semantics)
  this->declare_parameter("raytrace_max_range", 20.0);
  this->declare_parameter("obstacle_max_range", 20.0);
  this->declare_parameter("track_unknown_space", false);

  // Denoise: remove obstacle groups smaller than this (<= 1 disables)
  this->declare_parameter("denoise_minimal_group_size", 2);

  // Robot footprint
  this->declare_parameter("robot_footprint",
    std::vector<double>{0.49, 0.3725, 0.49, -0.3725, -0.49, -0.3725, -0.49, 0.3725});

  // Inflation layer
  this->declare_parameter("inflation_radius", 1.0);
  this->declare_parameter("cost_scaling_factor", 5.0);
}

void CostmapNode::loadParameters()
{
  point_cloud_topic_ = this->get_parameter("point_cloud_topic").as_string();
  odom_topic_ = this->get_parameter("odom_topic").as_string();

  odom_frame_ = this->get_parameter("odom_frame").as_string();
  base_frame_ = this->get_parameter("base_frame").as_string();
  sensor_frame_ = this->get_parameter("sensor_frame").as_string();

  costmap_width_ = this->get_parameter("costmap_width").as_double();
  costmap_height_ = this->get_parameter("costmap_height").as_double();
  costmap_resolution_ = this->get_parameter("costmap_resolution").as_double();

  min_obstacle_height_ = this->get_parameter("min_obstacle_height").as_double();
  max_obstacle_height_ = this->get_parameter("max_obstacle_height").as_double();

  update_frequency_ = this->get_parameter("update_frequency").as_double();

  sensor_timeout_ = this->get_parameter("sensor_timeout").as_double();

  raytrace_max_range_ = this->get_parameter("raytrace_max_range").as_double();
  obstacle_max_range_ = this->get_parameter("obstacle_max_range").as_double();
  track_unknown_space_ = this->get_parameter("track_unknown_space").as_bool();

  denoise_minimal_group_size_ = static_cast<int>(
    this->get_parameter("denoise_minimal_group_size").as_int());

  robot_footprint_ = this->get_parameter("robot_footprint").as_double_array();

  inflation_radius_ = this->get_parameter("inflation_radius").as_double();
  cost_scaling_factor_ = this->get_parameter("cost_scaling_factor").as_double();
}

bool CostmapNode::lookupTransform(
  const std::string& target_frame,
  const std::string& source_frame,
  geometry_msgs::msg::TransformStamped& transform)
{
  try {
    transform = tf_buffer_->lookupTransform(
      target_frame, source_frame,
      tf2::TimePointZero);
    return true;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "TF lookup failed: %s", ex.what());
    return false;
  }
}

void CostmapNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
{
  // Get transform from sensor frame to odom frame
  std::string source_frame = msg->header.frame_id;
  if (source_frame.empty()) {
    source_frame = sensor_frame_;
  }
  // Remove leading '/' if present
  if (!source_frame.empty() && source_frame[0] == '/') {
    source_frame = source_frame.substr(1);
  }

  // Look up the transform at the cloud's timestamp so points are not smeared
  // while the robot moves; fall back to the latest transform if unavailable
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_->lookupTransform(
      odom_frame_, source_frame,
      msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "TF lookup at cloud time failed (%s), falling back to latest", ex.what());
    if (!lookupTransform(odom_frame_, source_frame, transform)) {
      return;
    }
  }

  // Get current robot pose
  RobotPose pose;
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    pose = robot_pose_;
  }

  if (!pose.valid) {
    return;
  }

  // Process point cloud
  auto processed = pc_processor_->processCloud(msg, transform, pose.x, pose.y, pose.z, pose.yaw);

  // Store processed points and the sensor origin (raytrace start)
  {
    std::lock_guard<std::mutex> lock(pc_mutex_);
    latest_cloud_ = std::move(processed);
    sensor_origin_x_ = transform.transform.translation.x;
    sensor_origin_y_ = transform.transform.translation.y;
    has_new_points_ = true;
    cloud_received_ = true;
    last_cloud_time_ = this->now();
  }
}

void CostmapNode::odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
{
  const auto& p = msg->pose.pose.position;
  const auto& q = msg->pose.pose.orientation;

  // Convert quaternion to yaw
  double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  double yaw = std::atan2(siny_cosp, cosy_cosp);

  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    robot_pose_.x = p.x;
    robot_pose_.y = p.y;
    robot_pose_.z = p.z;
    robot_pose_.yaw = yaw;
    robot_pose_.valid = true;
  }
}

void CostmapNode::mainLoopCallback()
{
  updateCostmap();
  publishCostmap();
}

void CostmapNode::updateCostmap()
{
  // Get robot pose
  RobotPose pose;
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    pose = robot_pose_;
  }

  if (!pose.valid) {
    return;
  }

  // Get latest points
  ProcessedCloud cloud;
  double sensor_x = 0.0;
  double sensor_y = 0.0;
  bool new_points = false;
  bool cloud_received = false;
  rclcpp::Time last_cloud_time;
  {
    std::lock_guard<std::mutex> lock(pc_mutex_);
    new_points = has_new_points_;
    cloud_received = cloud_received_;
    last_cloud_time = last_cloud_time_;
    if (new_points) {
      cloud = std::move(latest_cloud_);
      sensor_x = sensor_origin_x_;
      sensor_y = sensor_origin_y_;
      has_new_points_ = false;
    }
  }

  const double origin_x = pose.x - costmap_width_ / 2.0;
  const double origin_y = pose.y - costmap_height_ / 2.0;

  if (!new_points) {
    if (!cloud_received) {
      // No cloud processed yet since startup (sensor not up, or TF missing)
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "Waiting for first point cloud on %s, publishing UNKNOWN costmap",
        point_cloud_topic_.c_str());
      obstacle_map_->reset(Costmap2D::UNKNOWN);
      obstacle_map_->updateOrigin(origin_x, origin_y);
      costmap_->copyFrom(*obstacle_map_);
      return;
    }

    // Sensor data stopped: clear the costmap to UNKNOWN instead of
    // keeping stale obstacles while the robot may still be moving
    if ((this->now() - last_cloud_time).seconds() > sensor_timeout_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "No point cloud for %.1f s (timeout %.1f s), clearing costmap to UNKNOWN",
        (this->now() - last_cloud_time).seconds(), sensor_timeout_);
      obstacle_map_->reset(Costmap2D::UNKNOWN);
      obstacle_map_->updateOrigin(origin_x, origin_y);
      costmap_->copyFrom(*obstacle_map_);
    }
    return;
  }

  // Shift the rolling window, keeping previously observed cells
  const int8_t fill = track_unknown_space_ ? Costmap2D::UNKNOWN : Costmap2D::FREE_SPACE;
  obstacle_map_->shiftOrigin(origin_x, origin_y, fill);

  // Raytrace clearing: free every cell each beam passed through
  int sensor_mx, sensor_my;
  if (obstacle_map_->worldToMap(sensor_x, sensor_y, sensor_mx, sensor_my)) {
    for (const auto& p : cloud.clearing) {
      double ex = p.x;
      double ey = p.y;
      const double dx = ex - sensor_x;
      const double dy = ey - sensor_y;
      const double range = std::hypot(dx, dy);
      if (range < 1e-6) {
        continue;
      }
      if (range > raytrace_max_range_) {
        const double scale = raytrace_max_range_ / range;
        ex = sensor_x + dx * scale;
        ey = sensor_y + dy * scale;
      }
      int end_mx, end_my;
      obstacle_map_->worldToMapNoBounds(ex, ey, end_mx, end_my);
      obstacle_map_->raytraceSetLine(sensor_mx, sensor_my, end_mx, end_my,
                                     Costmap2D::FREE_SPACE);
    }
  }

  // Mark obstacles (after clearing, so marks from this scan survive)
  for (const auto& p : cloud.marking) {
    if (std::hypot(p.x - sensor_x, p.y - sensor_y) > obstacle_max_range_) {
      continue;
    }
    int mx, my;
    if (obstacle_map_->worldToMap(p.x, p.y, mx, my)) {
      obstacle_map_->setCost(mx, my, Costmap2D::OCCUPIED);
    }
  }

  // Remove isolated noise cells before they get inflated
  denoise_layer_.apply(*obstacle_map_);

  // Build the published grid: obstacles + inflation
  costmap_->copyFrom(*obstacle_map_);
  if (inflation_layer_->isInitialized()) {
    inflation_layer_->inflate(*costmap_);
  }
}

void CostmapNode::publishCostmap()
{
  // Update header
  costmap_msg_.header.stamp = this->now();
  costmap_msg_.header.frame_id = odom_frame_;

  // Update map info
  costmap_msg_.info.origin.position.x = costmap_->getOriginX();
  costmap_msg_.info.origin.position.y = costmap_->getOriginY();
  costmap_msg_.info.origin.position.z = 0.0;

  // Copy data directly (no tolist() conversion needed!)
  const auto& data = costmap_->getDataVector();
  std::memcpy(costmap_msg_.data.data(), data.data(), data.size());

  costmap_pub_->publish(costmap_msg_);
}

}  // namespace local_costmap
