#ifndef LOCAL_COSTMAP_2D__POINTCLOUD_FILTERS_HPP_
#define LOCAL_COSTMAP_2D__POINTCLOUD_FILTERS_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>

namespace local_costmap_2d
{

class PointCloudFilters
{
public:
  // Remove ground plane using RANSAC
  static void removeGroundPlane(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    double distance_threshold = 0.1,
    int max_iterations = 1000);
  
  // Filter points by height (Z-axis)
  static void filterByHeight(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    double min_height, 
    double max_height);
  
  // Filter points by distance from origin
  static void filterByRange(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    double max_range,
    double min_range = 0.0);
  
  // Voxel grid downsampling
  static void voxelGridDownsample(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    double voxel_size);
  
  // Remove statistical outliers
  static void removeOutliers(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    int mean_k = 50,
    double std_dev_mul = 1.0);
  
  // Euclidean clustering to remove small noise clusters
  static void euclideanClustering(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    int min_cluster_size,
    double cluster_tolerance = 0.3,
    int max_cluster_size = 25000);
  
  // Remove points inside robot footprint
  static void filterRobotFootprint(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    const std::vector<std::pair<double, double>>& footprint,
    double robot_x = 0.0,
    double robot_y = 0.0,
    double robot_yaw = 0.0);
  
  // Filter points in specific regions (e.g., ignore ceiling)
  static void filterByRegion(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    double min_x, double max_x,
    double min_y, double max_y,
    double min_z, double max_z);

private:
  // Helper function to check if point is inside polygon
  static bool isPointInPolygon(
    double x, double y,
    const std::vector<std::pair<double, double>>& polygon);
  
  // Helper function to transform point
  static void transformPoint(
    double& x, double& y,
    double trans_x, double trans_y, double yaw);
};

}  // namespace local_costmap_2d

#endif  // LOCAL_COSTMAP_2D__POINTCLOUD_FILTERS_HPP_