#ifndef LOCAL_COSTMAP__POINT_CLOUD_PROCESSOR_HPP_
#define LOCAL_COSTMAP__POINT_CLOUD_PROCESSOR_HPP_

#include <vector>
#include <array>
#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace local_costmap
{

struct Point3D
{
  float x, y, z;
};

class PointCloudProcessor
{
public:
  PointCloudProcessor();

  // Set filtering parameters
  void setHeightFilter(double min_height, double max_height);
  void setHeightFilterEnabled(bool enabled);
  void setRobotFootprint(const std::vector<double>& footprint_flat);

  // Process point cloud
  // Returns filtered points in target frame
  std::vector<Point3D> processCloud(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
    const geometry_msgs::msg::TransformStamped& transform,
    double robot_x, double robot_y, double robot_yaw);

  // Get point count from last processing
  size_t getLastPointCount() const { return last_point_count_; }
  size_t getLastFilteredCount() const { return last_filtered_count_; }

private:
  double min_height_;
  double max_height_;
  bool height_filter_enabled_;

  // Robot footprint as polygon vertices (N x 2)
  std::vector<Eigen::Vector2f> footprint_;
  bool has_footprint_;

  // Statistics
  size_t last_point_count_;
  size_t last_filtered_count_;

  // Internal methods
  bool parsePointCloud(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
    std::vector<Point3D>& points);

  void transformPoints(
    std::vector<Point3D>& points,
    const geometry_msgs::msg::TransformStamped& transform);

  void filterByHeight(std::vector<Point3D>& points);

  void filterByFootprint(
    std::vector<Point3D>& points,
    double robot_x, double robot_y, double robot_yaw);

  bool isInsidePolygon(
    float px, float py,
    const std::vector<Eigen::Vector2f>& polygon) const;

  // Find field offset in PointCloud2
  int findFieldOffset(
    const sensor_msgs::msg::PointCloud2& cloud,
    const std::string& field_name) const;
};

}  // namespace local_costmap

#endif  // LOCAL_COSTMAP__POINT_CLOUD_PROCESSOR_HPP_
