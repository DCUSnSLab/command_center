#include "local_costmap/point_cloud_processor.hpp"

#include <cmath>
#include <algorithm>

namespace local_costmap
{

PointCloudProcessor::PointCloudProcessor()
  : min_height_(0.1),
    max_height_(2.0),
    height_filter_enabled_(true),
    has_footprint_(false),
    last_point_count_(0),
    last_filtered_count_(0)
{
}

void PointCloudProcessor::setHeightFilter(double min_height, double max_height)
{
  min_height_ = min_height;
  max_height_ = max_height;
}

void PointCloudProcessor::setHeightFilterEnabled(bool enabled)
{
  height_filter_enabled_ = enabled;
}

void PointCloudProcessor::setRobotFootprint(const std::vector<double>& footprint_flat)
{
  footprint_.clear();

  if (footprint_flat.size() < 6 || footprint_flat.size() % 2 != 0) {
    has_footprint_ = false;
    return;
  }

  size_t num_vertices = footprint_flat.size() / 2;
  footprint_.reserve(num_vertices);

  for (size_t i = 0; i < num_vertices; ++i) {
    footprint_.emplace_back(
      static_cast<float>(footprint_flat[2 * i]),
      static_cast<float>(footprint_flat[2 * i + 1])
    );
  }

  has_footprint_ = true;
}

int PointCloudProcessor::findFieldOffset(
  const sensor_msgs::msg::PointCloud2& cloud,
  const std::string& field_name) const
{
  for (const auto& field : cloud.fields) {
    if (field.name == field_name) {
      return static_cast<int>(field.offset);
    }
  }
  return -1;
}

bool PointCloudProcessor::parsePointCloud(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
  std::vector<Point3D>& points)
{
  // Find field offsets
  int x_offset = findFieldOffset(*cloud, "x");
  int y_offset = findFieldOffset(*cloud, "y");
  int z_offset = findFieldOffset(*cloud, "z");

  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    return false;
  }

  const uint32_t point_step = cloud->point_step;
  const size_t num_points = cloud->width * cloud->height;
  const uint8_t* data_ptr = cloud->data.data();

  points.clear();
  points.reserve(num_points);

  for (size_t i = 0; i < num_points; ++i) {
    const uint8_t* point_ptr = data_ptr + i * point_step;

    float x, y, z;
    std::memcpy(&x, point_ptr + x_offset, sizeof(float));
    std::memcpy(&y, point_ptr + y_offset, sizeof(float));
    std::memcpy(&z, point_ptr + z_offset, sizeof(float));

    // Skip NaN points
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
      points.push_back({x, y, z});
    }
  }

  return true;
}

void PointCloudProcessor::transformPoints(
  std::vector<Point3D>& points,
  const geometry_msgs::msg::TransformStamped& transform)
{
  const auto& t = transform.transform.translation;
  const auto& r = transform.transform.rotation;

  // Convert quaternion to rotation matrix
  float qx = static_cast<float>(r.x);
  float qy = static_cast<float>(r.y);
  float qz = static_cast<float>(r.z);
  float qw = static_cast<float>(r.w);

  float xx = qx * qx, yy = qy * qy, zz = qz * qz;
  float xy = qx * qy, xz = qx * qz, yz = qy * qz;
  float wx = qw * qx, wy = qw * qy, wz = qw * qz;

  // Rotation matrix elements
  float r00 = 1.0f - 2.0f * (yy + zz);
  float r01 = 2.0f * (xy - wz);
  float r02 = 2.0f * (xz + wy);
  float r10 = 2.0f * (xy + wz);
  float r11 = 1.0f - 2.0f * (xx + zz);
  float r12 = 2.0f * (yz - wx);
  float r20 = 2.0f * (xz - wy);
  float r21 = 2.0f * (yz + wx);
  float r22 = 1.0f - 2.0f * (xx + yy);

  float tx = static_cast<float>(t.x);
  float ty = static_cast<float>(t.y);
  float tz = static_cast<float>(t.z);

  // Transform all points
  for (auto& p : points) {
    float x = p.x, y = p.y, z = p.z;
    p.x = r00 * x + r01 * y + r02 * z + tx;
    p.y = r10 * x + r11 * y + r12 * z + ty;
    p.z = r20 * x + r21 * y + r22 * z + tz;
  }
}

void PointCloudProcessor::filterByHeight(std::vector<Point3D>& points)
{
  auto new_end = std::remove_if(points.begin(), points.end(),
    [this](const Point3D& p) {
      return p.z < min_height_ || p.z > max_height_;
    });

  points.erase(new_end, points.end());
}

bool PointCloudProcessor::isInsidePolygon(
  float px, float py,
  const std::vector<Eigen::Vector2f>& polygon) const
{
  // Ray casting algorithm
  bool inside = false;
  size_t n = polygon.size();

  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    float xi = polygon[i].x(), yi = polygon[i].y();
    float xj = polygon[j].x(), yj = polygon[j].y();

    if (((yi > py) != (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }

  return inside;
}

void PointCloudProcessor::filterByFootprint(
  std::vector<Point3D>& points,
  double robot_x, double robot_y, double robot_yaw)
{
  if (!has_footprint_ || footprint_.empty()) {
    return;
  }

  // Transform footprint to world frame
  float cos_yaw = std::cos(static_cast<float>(robot_yaw));
  float sin_yaw = std::sin(static_cast<float>(robot_yaw));
  float rx = static_cast<float>(robot_x);
  float ry = static_cast<float>(robot_y);

  std::vector<Eigen::Vector2f> footprint_world;
  footprint_world.reserve(footprint_.size());

  for (const auto& v : footprint_) {
    float wx = cos_yaw * v.x() - sin_yaw * v.y() + rx;
    float wy = sin_yaw * v.x() + cos_yaw * v.y() + ry;
    footprint_world.emplace_back(wx, wy);
  }

  // Remove points inside footprint
  auto new_end = std::remove_if(points.begin(), points.end(),
    [this, &footprint_world](const Point3D& p) {
      return isInsidePolygon(p.x, p.y, footprint_world);
    });

  points.erase(new_end, points.end());
}

std::vector<Point3D> PointCloudProcessor::processCloud(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
  const geometry_msgs::msg::TransformStamped& transform,
  double robot_x, double robot_y, double robot_yaw)
{
  std::vector<Point3D> points;

  // Parse point cloud (zero-copy from raw data)
  if (!parsePointCloud(cloud, points)) {
    last_point_count_ = 0;
    last_filtered_count_ = 0;
    return points;
  }

  last_point_count_ = points.size();

  // Transform to target frame
  transformPoints(points, transform);

  // Apply height filter (only if enabled - disabled when using HeightAnalyzer)
  if (height_filter_enabled_) {
    filterByHeight(points);
  }

  // Filter robot footprint
  filterByFootprint(points, robot_x, robot_y, robot_yaw);

  last_filtered_count_ = points.size();

  return points;
}

}  // namespace local_costmap
