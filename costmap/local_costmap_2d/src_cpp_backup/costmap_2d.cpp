#include "local_costmap_2d/costmap_2d.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <rclcpp/rclcpp.hpp>

namespace local_costmap_2d
{

SimpleCostmap2D::SimpleCostmap2D(
  unsigned int cells_size_x, unsigned int cells_size_y, double resolution,
  double origin_x, double origin_y, unsigned char default_value)
: size_x_(cells_size_x), size_y_(cells_size_y), resolution_(resolution),
  origin_x_(origin_x), origin_y_(origin_y), costmap_(nullptr), default_value_(default_value)
{
  initMaps(size_x_, size_y_);
  resetMaps();
}

SimpleCostmap2D::SimpleCostmap2D(const nav_msgs::msg::OccupancyGrid & map)
: default_value_(FREE_SPACE)
{
  size_x_ = map.info.width;
  size_y_ = map.info.height;
  resolution_ = map.info.resolution;
  origin_x_ = map.info.origin.position.x;
  origin_y_ = map.info.origin.position.y;
  
  costmap_ = new unsigned char[size_x_ * size_y_];
  
  // Convert OccupancyGrid data to costmap format
  for (unsigned int i = 0; i < size_x_ * size_y_; i++) {
    int8_t data = map.data[i];
    if (data == -1) {
      costmap_[i] = NO_INFORMATION;
    } else {
      // Convert from 0-100 occupancy to 0-254 cost
      costmap_[i] = static_cast<unsigned char>(
        std::round(static_cast<double>(data) * LETHAL_OBSTACLE / 100.0));
    }
  }
}

SimpleCostmap2D::SimpleCostmap2D(const SimpleCostmap2D & map)
: costmap_(nullptr)
{
  *this = map;
}

SimpleCostmap2D & SimpleCostmap2D::operator=(const SimpleCostmap2D & map)
{
  if (this == &map) {
    return *this;
  }
  
  deleteMaps();
  
  size_x_ = map.size_x_;
  size_y_ = map.size_y_;
  resolution_ = map.resolution_;
  origin_x_ = map.origin_x_;
  origin_y_ = map.origin_y_;
  default_value_ = map.default_value_;
  
  initMaps(size_x_, size_y_);
  
  if (map.costmap_) {
    std::memcpy(costmap_, map.costmap_, size_x_ * size_y_ * sizeof(unsigned char));
  }
  
  obstacle_points_ = map.obstacle_points_;
  
  return *this;
}

SimpleCostmap2D::SimpleCostmap2D()
: size_x_(0), size_y_(0), resolution_(0.0), origin_x_(0.0), origin_y_(0.0), 
  costmap_(nullptr), default_value_(FREE_SPACE)
{
}

SimpleCostmap2D::~SimpleCostmap2D()
{
  deleteMaps();
}

void SimpleCostmap2D::deleteMaps()
{
  std::lock_guard<std::mutex> lock(access_mutex_);
  delete[] costmap_;
  costmap_ = nullptr;
}

void SimpleCostmap2D::initMaps(unsigned int size_x, unsigned int size_y)
{
  std::lock_guard<std::mutex> lock(access_mutex_);
  delete[] costmap_;
  costmap_ = new unsigned char[size_x * size_y];
}

void SimpleCostmap2D::resetMaps()
{
  std::lock_guard<std::mutex> lock(access_mutex_);
  std::memset(costmap_, default_value_, size_x_ * size_y_ * sizeof(unsigned char));
}

unsigned char SimpleCostmap2D::getCost(unsigned int mx, unsigned int my) const
{
  return costmap_[getIndex(mx, my)];
}

unsigned char SimpleCostmap2D::getCost(unsigned int index) const
{
  return costmap_[index];
}

void SimpleCostmap2D::setCost(unsigned int mx, unsigned int my, unsigned char cost)
{
  costmap_[getIndex(mx, my)] = cost;
}

void SimpleCostmap2D::mapToWorld(unsigned int mx, unsigned int my, double & wx, double & wy) const
{
  wx = origin_x_ + (mx + 0.5) * resolution_;
  wy = origin_y_ + (my + 0.5) * resolution_;
}

bool SimpleCostmap2D::worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my) const
{
  if (wx < origin_x_ || wy < origin_y_) {
    return false;
  }
  
  mx = static_cast<unsigned int>((wx - origin_x_) / resolution_);
  my = static_cast<unsigned int>((wy - origin_y_) / resolution_);
  
  if (mx < size_x_ && my < size_y_) {
    return true;
  }
  return false;
}

void SimpleCostmap2D::worldToMapNoBounds(double wx, double wy, int & mx, int & my) const
{
  mx = static_cast<int>((wx - origin_x_) / resolution_);
  my = static_cast<int>((wy - origin_y_) / resolution_);
}

void SimpleCostmap2D::worldToMapEnforceBounds(double wx, double wy, int & mx, int & my) const
{
  if (wx < origin_x_) {
    mx = 0;
  } else if (wx > resolution_ * size_x_ + origin_x_) {
    mx = size_x_ - 1;
  } else {
    mx = static_cast<int>((wx - origin_x_) / resolution_);
  }
  
  if (wy < origin_y_) {
    my = 0;
  } else if (wy > resolution_ * size_y_ + origin_y_) {
    my = size_y_ - 1;
  } else {
    my = static_cast<int>((wy - origin_y_) / resolution_);
  }
}

void SimpleCostmap2D::resetMap(unsigned int x0, unsigned int y0, unsigned int xn, unsigned int yn)
{
  resetMapToValue(x0, y0, xn, yn, default_value_);
}

void SimpleCostmap2D::resetMapToValue(
  unsigned int x0, unsigned int y0, unsigned int xn, unsigned int yn, unsigned char value)
{
  std::lock_guard<std::mutex> lock(access_mutex_);
  unsigned int len = xn - x0;
  for (unsigned int y = y0 * size_x_ + x0; y < yn * size_x_ + x0; y += size_x_) {
    std::memset(costmap_ + y, value, len * sizeof(unsigned char));
  }
}

void SimpleCostmap2D::resizeMap(
  unsigned int size_x, unsigned int size_y, double resolution, double origin_x, double origin_y)
{
  size_x_ = size_x;
  size_y_ = size_y;
  resolution_ = resolution;
  origin_x_ = origin_x;
  origin_y_ = origin_y;
  
  initMaps(size_x, size_y);
  resetMaps();
}

void SimpleCostmap2D::projectPoint(double x, double y, double z)
{
  unsigned int mx, my;
  if (worldToMap(x, y, mx, my)) {
    setCost(mx, my, LETHAL_OBSTACLE);
    
    // Store obstacle point for external use
    geometry_msgs::msg::Point point;
    point.x = x;
    point.y = y;
    point.z = z;
    obstacle_points_.push_back(point);
  }
}

void SimpleCostmap2D::markObstacleWithHeight(double x, double y, double z, unsigned char cost)
{
  unsigned int mx, my;
  if (worldToMap(x, y, mx, my)) {
    setCost(mx, my, cost);
    
    // Store obstacle point
    geometry_msgs::msg::Point point;
    point.x = x;
    point.y = y;
    point.z = z;
    obstacle_points_.push_back(point);
  }
}

void SimpleCostmap2D::processPointBatch(const std::vector<geometry_msgs::msg::Point>& points)
{
  obstacle_points_.clear();
  obstacle_points_.reserve(points.size());
  
  for (const auto& point : points) {
    projectPoint(point.x, point.y, point.z);
  }
}

void SimpleCostmap2D::applyTemporalDecay(double decay_rate)
{
  std::lock_guard<std::mutex> lock(access_mutex_);
  
  for (unsigned int i = 0; i < size_x_ * size_y_; i++) {
    if (costmap_[i] > FREE_SPACE && costmap_[i] < NO_INFORMATION) {
      unsigned char decayed = static_cast<unsigned char>(costmap_[i] * decay_rate);
      costmap_[i] = (decayed > FREE_SPACE) ? decayed : FREE_SPACE;
    }
  }
}

void SimpleCostmap2D::raytrace(double x0, double y0, double x1, double y1)
{
  unsigned int cell_x0, cell_y0, cell_x1, cell_y1;
  
  if (!worldToMap(x0, y0, cell_x0, cell_y0) || !worldToMap(x1, y1, cell_x1, cell_y1)) {
    return;
  }
  
  MarkCell marker(costmap_, FREE_SPACE);
  raytraceLine(marker, cell_x0, cell_y0, cell_x1, cell_y1);
}

template<class ActionType>
inline void SimpleCostmap2D::raytraceLine(
  ActionType at, unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1)
{
  int dx = x1 - x0;
  int dy = y1 - y0;
  
  int abs_dx = std::abs(dx);
  int abs_dy = std::abs(dy);
  
  int offset_dx = (dx > 0) ? 1 : -1;
  int offset_dy = (dy > 0) ? size_x_ : -size_x_;
  
  unsigned int offset = y0 * size_x_ + x0;
  
  if (abs_dx >= abs_dy) {
    int error_y = abs_dx / 2;
    for (int i = 0; i < abs_dx; ++i) {
      at(offset);
      offset += offset_dx;
      error_y += abs_dy;
      if (error_y >= abs_dx) {
        offset += offset_dy;
        error_y -= abs_dx;
      }
    }
    at(offset);
  } else {
    int error_x = abs_dy / 2;
    for (int i = 0; i < abs_dy; ++i) {
      at(offset);
      offset += offset_dy;
      error_x += abs_dx;
      if (error_x >= abs_dy) {
        offset += offset_dx;
        error_x -= abs_dy;
      }
    }
    at(offset);
  }
}

nav_msgs::msg::OccupancyGrid SimpleCostmap2D::toOccupancyGrid(const std::string& frame_id) const
{
  nav_msgs::msg::OccupancyGrid grid;
  
  grid.header.frame_id = frame_id;
  grid.header.stamp = rclcpp::Clock().now();
  
  grid.info.resolution = resolution_;
  grid.info.width = size_x_;
  grid.info.height = size_y_;
  grid.info.origin.position.x = origin_x_;
  grid.info.origin.position.y = origin_y_;
  grid.info.origin.position.z = 0.0;
  grid.info.origin.orientation.w = 1.0;
  
  grid.data.resize(size_x_ * size_y_);
  
  // Convert costmap data to occupancy grid format
  for (unsigned int i = 0; i < size_x_ * size_y_; i++) {
    if (costmap_[i] == NO_INFORMATION) {
      grid.data[i] = -1;
    } else {
      // Convert from 0-254 cost to 0-100 occupancy
      grid.data[i] = static_cast<int8_t>(
        std::round(static_cast<double>(costmap_[i]) * 100.0 / LETHAL_OBSTACLE));
    }
  }
  
  return grid;
}

void SimpleCostmap2D::clearRobotFootprint(const std::vector<geometry_msgs::msg::Point>& footprint, 
                                         double robot_x, double robot_y, double robot_yaw)
{
  if (footprint.empty()) {
    return;
  }
  
  // Transform footprint to world coordinates
  std::vector<geometry_msgs::msg::Point> world_footprint;
  world_footprint.reserve(footprint.size());
  
  double cos_yaw = cos(robot_yaw);
  double sin_yaw = sin(robot_yaw);
  
  for (const auto& point : footprint) {
    geometry_msgs::msg::Point world_point;
    world_point.x = robot_x + point.x * cos_yaw - point.y * sin_yaw;
    world_point.y = robot_y + point.x * sin_yaw + point.y * cos_yaw;
    world_point.z = 0.0;
    world_footprint.push_back(world_point);
  }
  
  // Find bounding box of the footprint
  double min_x = world_footprint[0].x, max_x = world_footprint[0].x;
  double min_y = world_footprint[0].y, max_y = world_footprint[0].y;
  
  for (const auto& point : world_footprint) {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
    min_y = std::min(min_y, point.y);
    max_y = std::max(max_y, point.y);
  }
  
  // Convert bounding box to map coordinates
  unsigned int min_mx, min_my, max_mx, max_my;
  if (!worldToMap(min_x, min_y, min_mx, min_my) || 
      !worldToMap(max_x, max_y, max_mx, max_my)) {
    return;  // Footprint is outside the map
  }
  
  // Clear cells within the footprint
  for (unsigned int mx = min_mx; mx <= max_mx && mx < size_x_; mx++) {
    for (unsigned int my = min_my; my <= max_my && my < size_y_; my++) {
      double wx, wy;
      mapToWorld(mx, my, wx, wy);
      
      // Check if point is inside the footprint polygon
      if (isPointInPolygon(wx, wy, world_footprint)) {
        setCost(mx, my, FREE_SPACE);
      }
    }
  }
}

bool SimpleCostmap2D::isPointInPolygon(double x, double y, 
                                      const std::vector<geometry_msgs::msg::Point>& polygon) const
{
  int n = polygon.size();
  bool inside = false;
  
  for (int i = 0, j = n - 1; i < n; j = i++) {
    if (((polygon[i].y > y) != (polygon[j].y > y)) &&
        (x < (polygon[j].x - polygon[i].x) * (y - polygon[i].y) / 
              (polygon[j].y - polygon[i].y) + polygon[i].x)) {
      inside = !inside;
    }
  }
  
  return inside;
}

}  // namespace local_costmap_2d