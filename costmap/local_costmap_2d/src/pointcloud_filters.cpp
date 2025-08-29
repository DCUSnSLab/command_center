#include "local_costmap_2d/pointcloud_filters.hpp"
#include <cmath>
#include <algorithm>

namespace local_costmap_2d
{

void PointCloudFilters::removeGroundPlane(
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
  double distance_threshold,
  int max_iterations)
{
  if (cloud->points.empty()) {
    return;
  }
  
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  
  // Create segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(max_iterations);
  seg.setDistanceThreshold(distance_threshold);
  
  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coefficients);
  
  if (inliers->indices.empty()) {
    return;  // No ground plane found
  }
  
  // Extract non-ground points
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);  // Extract everything except inliers (ground)
  extract.filter(*cloud);
}

void PointCloudFilters::filterByHeight(
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
  double min_height,
  double max_height)
{
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(min_height, max_height);
  pass.filter(*cloud);
}

void PointCloudFilters::filterByRange(
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
  double max_range,
  double min_range)
{
  cloud->points.erase(
    std::remove_if(cloud->points.begin(), cloud->points.end(),
      [min_range, max_range](const pcl::PointXYZ& point) {
        double range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        return range < min_range || range > max_range;
      }),
    cloud->points.end()
  );
  
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = false;
}

void PointCloudFilters::voxelGridDownsample(
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
  double voxel_size)
{
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
  voxel_filter.filter(*cloud);
}

void PointCloudFilters::removeOutliers(
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
  int mean_k,
  double std_dev_mul)
{
  if (cloud->points.size() < static_cast<size_t>(mean_k)) {
    return;  // Not enough points for statistical filtering
  }
  
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(mean_k);
  sor.setStddevMulThresh(std_dev_mul);
  sor.filter(*cloud);
}

void PointCloudFilters::euclideanClustering(
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
  int min_cluster_size,
  double cluster_tolerance,
  int max_cluster_size)
{
  if (cloud->points.empty()) {
    return;
  }
  
  // Create KdTree object for search method
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);
  
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance);
  ec.setMinClusterSize(min_cluster_size);
  ec.setMaxClusterSize(max_cluster_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);
  
  // Create new cloud with only valid clusters
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  
  for (const auto& cluster : cluster_indices) {
    for (const auto& idx : cluster.indices) {
      filtered_cloud->points.push_back(cloud->points[idx]);
    }
  }
  
  filtered_cloud->width = filtered_cloud->points.size();
  filtered_cloud->height = 1;
  filtered_cloud->is_dense = false;
  
  *cloud = *filtered_cloud;
}

void PointCloudFilters::filterRobotFootprint(
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
  const std::vector<std::pair<double, double>>& footprint,
  double robot_x,
  double robot_y,
  double robot_yaw)
{
  cloud->points.erase(
    std::remove_if(cloud->points.begin(), cloud->points.end(),
      [&](const pcl::PointXYZ& point) {
        // Transform point to robot coordinate frame
        double x = point.x - robot_x;
        double y = point.y - robot_y;
        transformPoint(x, y, 0.0, 0.0, -robot_yaw);
        
        return isPointInPolygon(x, y, footprint);
      }),
    cloud->points.end()
  );
  
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = false;
}

void PointCloudFilters::filterByRegion(
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
  double min_x, double max_x,
  double min_y, double max_y,
  double min_z, double max_z)
{
  cloud->points.erase(
    std::remove_if(cloud->points.begin(), cloud->points.end(),
      [min_x, max_x, min_y, max_y, min_z, max_z](const pcl::PointXYZ& point) {
        return point.x < min_x || point.x > max_x ||
               point.y < min_y || point.y > max_y ||
               point.z < min_z || point.z > max_z;
      }),
    cloud->points.end()
  );
  
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = false;
}

bool PointCloudFilters::isPointInPolygon(
  double x, double y,
  const std::vector<std::pair<double, double>>& polygon)
{
  bool inside = false;
  size_t n = polygon.size();
  
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    if (((polygon[i].second > y) != (polygon[j].second > y)) &&
        (x < (polygon[j].first - polygon[i].first) * (y - polygon[i].second) / 
         (polygon[j].second - polygon[i].second) + polygon[i].first)) {
      inside = !inside;
    }
  }
  
  return inside;
}

void PointCloudFilters::transformPoint(
  double& x, double& y,
  double trans_x, double trans_y, double yaw)
{
  double cos_yaw = std::cos(yaw);
  double sin_yaw = std::sin(yaw);
  
  double temp_x = x * cos_yaw - y * sin_yaw + trans_x;
  double temp_y = x * sin_yaw + y * cos_yaw + trans_y;
  
  x = temp_x;
  y = temp_y;
}

}  // namespace local_costmap_2d