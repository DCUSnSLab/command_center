#include "local_costmap_2d/costmap_publisher.hpp"
#include "local_costmap_2d/cost_values.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <algorithm>

namespace local_costmap_2d
{

std::vector<int8_t> CostmapPublisher::cost_translation_table_;


CostmapPublisher::~CostmapPublisher()
{
  RCLCPP_INFO(logger_, "CostmapPublisher destroyed");
}

void CostmapPublisher::initCostTranslationTable()
{
  cost_translation_table_.resize(256);
  
  // Translate costmap values to occupancy grid values
  for (int i = 0; i < 256; i++) {
    if (i == NO_INFORMATION) {
      cost_translation_table_[i] = -1;  // Unknown
    } else if (i == FREE_SPACE) {
      cost_translation_table_[i] = 0;   // Free
    } else {
      // Scale from 1-254 to 1-100
      cost_translation_table_[i] = static_cast<int8_t>(
        std::round(static_cast<double>(i) * 100.0 / LETHAL_OBSTACLE));
      cost_translation_table_[i] = std::min(cost_translation_table_[i], static_cast<int8_t>(100));
    }
  }
}

void CostmapPublisher::publishCostmap()
{
  if (!active_ || !costmap_) {
    return;
  }
  
  std::lock_guard<std::mutex> lock(*costmap_->getMutex());
  
  // Check if we need to send full costmap or just an update
  if (config_.always_send_full_costmap || !has_updated_data_ ||
      saved_origin_x_ != costmap_->getOriginX() || 
      saved_origin_y_ != costmap_->getOriginY()) {
    
    prepareGrid();
    costmap_pub_->publish(*grid_msg_);
    
    saved_origin_x_ = costmap_->getOriginX();
    saved_origin_y_ = costmap_->getOriginY();
    
  } else if (has_updated_data_) {
    // Send incremental update
    unsigned int width = xn_ - x0_;
    unsigned int height = yn_ - y0_;
    
    if (width > 0 && height > 0) {
      publishCostmapUpdate(x0_, y0_, width, height);
    }
  }
  
  has_updated_data_ = false;
  resetBounds();
}

void CostmapPublisher::publishCostmapUpdate(
  unsigned int x0, unsigned int y0, unsigned int width, unsigned int height)
{
  if (!active_ || !costmap_update_pub_) {
    return;
  }
  
  prepareGridUpdate(x0, y0, width, height);
  costmap_update_pub_->publish(*grid_update_msg_);
}

void CostmapPublisher::publishObstacleMarkers(
  const std::vector<geometry_msgs::msg::Point>& obstacle_points)
{
  if (!active_ || !config_.enable_obstacle_markers || !obstacle_markers_pub_) {
    return;
  }
  
  visualization_msgs::msg::MarkerArray markers;
  createObstacleMarkers(obstacle_points, markers);
  obstacle_markers_pub_->publish(markers);
}

void CostmapPublisher::publishFilteredPointCloud(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud)
{
  if (!active_ || !config_.enable_filtered_cloud || !filtered_cloud_pub_ || !filtered_cloud) {
    return;
  }
  
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*filtered_cloud, cloud_msg);
  cloud_msg.header.frame_id = config_.global_frame;
  cloud_msg.header.stamp = node_clock_->get_clock()->now();
  
  filtered_cloud_pub_->publish(cloud_msg);
}

void CostmapPublisher::prepareGrid()
{
  if (!costmap_) {
    return;
  }
  
  grid_msg_->header.frame_id = config_.global_frame;
  grid_msg_->header.stamp = node_clock_->get_clock()->now();
  
  grid_msg_->info.resolution = costmap_->getResolution();
  grid_msg_->info.width = costmap_->getSizeInCellsX();
  grid_msg_->info.height = costmap_->getSizeInCellsY();
  
  grid_msg_->info.origin.position.x = costmap_->getOriginX();
  grid_msg_->info.origin.position.y = costmap_->getOriginY();
  grid_msg_->info.origin.position.z = 0.0;
  grid_msg_->info.origin.orientation.w = 1.0;
  
  unsigned int size = costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY();
  grid_msg_->data.resize(size);
  
  unsigned char* data = costmap_->getCharMap();
  for (unsigned int i = 0; i < size; i++) {
    grid_msg_->data[i] = cost_translation_table_[data[i]];
  }
}

void CostmapPublisher::prepareGridUpdate(
  unsigned int x0, unsigned int y0, unsigned int width, unsigned int height)
{
  if (!costmap_) {
    return;
  }
  
  grid_update_msg_->header.frame_id = config_.global_frame;
  grid_update_msg_->header.stamp = node_clock_->get_clock()->now();
  
  grid_update_msg_->x = x0;
  grid_update_msg_->y = y0;
  grid_update_msg_->width = width;
  grid_update_msg_->height = height;
  
  grid_update_msg_->data.resize(width * height);
  
  unsigned char* data = costmap_->getCharMap();
  unsigned int costmap_width = costmap_->getSizeInCellsX();
  
  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      unsigned int costmap_index = (y0 + y) * costmap_width + (x0 + x);
      unsigned int update_index = y * width + x;
      grid_update_msg_->data[update_index] = cost_translation_table_[data[costmap_index]];
    }
  }
}

void CostmapPublisher::createObstacleMarkers(
  const std::vector<geometry_msgs::msg::Point>& obstacle_points,
  visualization_msgs::msg::MarkerArray& markers)
{
  // Clear previous markers
  visualization_msgs::msg::Marker clear_marker;
  clear_marker.header.frame_id = config_.global_frame;
  clear_marker.header.stamp = node_clock_->get_clock()->now();
  clear_marker.ns = "obstacle_points";
  clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(clear_marker);
  
  if (obstacle_points.empty()) {
    return;
  }
  
  // Create obstacle markers
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = config_.global_frame;
  marker.header.stamp = node_clock_->get_clock()->now();
  marker.ns = "obstacle_points";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::POINTS;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  
  marker.scale.x = config_.obstacle_marker_size;
  marker.scale.y = config_.obstacle_marker_size;
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 0.8;
  
  marker.lifetime = rclcpp::Duration::from_nanoseconds(
    static_cast<int64_t>(config_.marker_lifetime * 1e9));
  
  marker.points = obstacle_points;
  
  markers.markers.push_back(marker);
}

void CostmapPublisher::updateBounds(unsigned int x0, unsigned int xn, unsigned int y0, unsigned int yn)
{
  x0_ = std::min(x0_, x0);
  xn_ = std::max(xn_, xn);
  y0_ = std::min(y0_, y0);
  yn_ = std::max(yn_, yn);
  has_updated_data_ = true;
}

void CostmapPublisher::resetBounds()
{
  if (costmap_) {
    x0_ = costmap_->getSizeInCellsX();
    xn_ = 0;
    y0_ = costmap_->getSizeInCellsY();
    yn_ = 0;
  } else {
    x0_ = 0;
    xn_ = 0;
    y0_ = 0;
    yn_ = 0;
  }
  has_updated_data_ = false;
}

void CostmapPublisher::updateConfig(const Config& config)
{
  config_ = config;
  RCLCPP_INFO(logger_, "CostmapPublisher configuration updated");
}

void CostmapPublisher::on_activate()
{
  active_ = true;
  RCLCPP_INFO(logger_, "CostmapPublisher activated");
}

void CostmapPublisher::on_deactivate()
{
  active_ = false;
  RCLCPP_INFO(logger_, "CostmapPublisher deactivated");
}

}  // namespace local_costmap_2d