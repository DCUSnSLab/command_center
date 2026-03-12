#ifndef LOCAL_COSTMAP__HEIGHT_ANALYZER_HPP_
#define LOCAL_COSTMAP__HEIGHT_ANALYZER_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>

#include "local_costmap/costmap_2d.hpp"
#include "local_costmap/point_cloud_processor.hpp"

namespace local_costmap
{

// Cell height statistics
struct CellHeightInfo
{
  float z_min = std::numeric_limits<float>::max();
  float z_max = std::numeric_limits<float>::lowest();
  int point_count = 0;

  void addPoint(float z)
  {
    if (z < z_min) z_min = z;
    if (z > z_max) z_max = z;
    ++point_count;
  }

  float getHeightDiff() const
  {
    if (point_count == 0) return 0.0f;
    return z_max - z_min;
  }

  bool hasPoints() const { return point_count > 0; }
};

class HeightAnalyzer
{
public:
  HeightAnalyzer();

  // Initialize with parameters
  void initialize(double obstacle_height_threshold,
                  double negative_obstacle_threshold,
                  double min_point_height,
                  double max_point_height);

  // Analyze points and mark obstacles on costmap
  // Returns number of obstacle cells marked
  size_t analyzeAndMark(
    const std::vector<Point3D>& points,
    Costmap2D& costmap,
    double robot_z);

  // Getters
  double getObstacleHeightThreshold() const { return obstacle_height_threshold_; }
  double getNegativeObstacleThreshold() const { return negative_obstacle_threshold_; }

private:
  double obstacle_height_threshold_;      // Height diff threshold for obstacles (e.g., 0.3m)
  double negative_obstacle_threshold_;    // Z drop threshold for pits/cliffs (e.g., -0.3m)
  double min_point_height_;               // Min point height relative to robot (filter noise)
  double max_point_height_;               // Max point height relative to robot
  bool initialized_;

  // Per-cell height info (reused buffer)
  std::vector<CellHeightInfo> cell_heights_;

  // Reset cell height buffer
  void resetCellHeights(size_t size);
};

}  // namespace local_costmap

#endif  // LOCAL_COSTMAP__HEIGHT_ANALYZER_HPP_
