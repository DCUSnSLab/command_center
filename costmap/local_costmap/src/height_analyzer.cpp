#include "local_costmap/height_analyzer.hpp"

#include <cmath>
#include <algorithm>

namespace local_costmap
{

HeightAnalyzer::HeightAnalyzer()
  : obstacle_height_threshold_(0.3),
    negative_obstacle_threshold_(-0.3),
    min_point_height_(-2.0),
    max_point_height_(2.0),
    initialized_(false)
{
}

void HeightAnalyzer::initialize(double obstacle_height_threshold,
                                 double negative_obstacle_threshold,
                                 double min_point_height,
                                 double max_point_height)
{
  obstacle_height_threshold_ = obstacle_height_threshold;
  negative_obstacle_threshold_ = negative_obstacle_threshold;
  min_point_height_ = min_point_height;
  max_point_height_ = max_point_height;
  initialized_ = true;
}

void HeightAnalyzer::resetCellHeights(size_t size)
{
  if (cell_heights_.size() != size) {
    cell_heights_.resize(size);
  }

  // Reset all cells
  for (auto& cell : cell_heights_) {
    cell.z_min = std::numeric_limits<float>::max();
    cell.z_max = std::numeric_limits<float>::lowest();
    cell.point_count = 0;
  }
}

size_t HeightAnalyzer::analyzeAndMark(
  const std::vector<Point3D>& points,
  Costmap2D& costmap,
  double robot_z)
{
  if (!initialized_) {
    return 0;
  }

  const int width = costmap.getWidthCells();
  const int height = costmap.getHeightCells();
  const size_t total_cells = static_cast<size_t>(width * height);

  // Reset cell heights buffer
  resetCellHeights(total_cells);

  // Convert thresholds to float for comparison
  float robot_z_f = static_cast<float>(robot_z);
  float min_h = static_cast<float>(min_point_height_);
  float max_h = static_cast<float>(max_point_height_);

  // First pass: accumulate min/max z for each cell
  for (const auto& p : points) {
    // Height filtering relative to robot
    float relative_z = p.z - robot_z_f;
    if (relative_z < min_h || relative_z > max_h) {
      continue;
    }

    int mx, my;
    if (costmap.worldToMap(p.x, p.y, mx, my)) {
      size_t idx = static_cast<size_t>(my * width + mx);
      cell_heights_[idx].addPoint(p.z);
    }
  }

  // Second pass: analyze cells and mark obstacles
  size_t obstacle_count = 0;
  int8_t* data = costmap.getData();

  // Calculate reference ground height (robot's z position)
  float ground_reference = robot_z_f;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      size_t idx = static_cast<size_t>(y * width + x);
      const auto& cell = cell_heights_[idx];

      if (!cell.hasPoints()) {
        continue;
      }

      bool is_obstacle = false;

      // Check 1: Height difference within cell (positive obstacle)
      // e.g., a box, wall, person
      float height_diff = cell.getHeightDiff();
      if (height_diff > static_cast<float>(obstacle_height_threshold_)) {
        is_obstacle = true;
      }

      // Check 2: Negative obstacle (pit, cliff, stairs down)
      // If the minimum z in this cell is significantly below the ground reference
      float z_drop = cell.z_min - ground_reference;
      if (z_drop < static_cast<float>(negative_obstacle_threshold_)) {
        is_obstacle = true;
      }

      // Check 3: Positive obstacle above ground
      // If max height is significantly above ground (but not enough height diff in cell)
      float z_rise = cell.z_max - ground_reference;
      if (z_rise > static_cast<float>(obstacle_height_threshold_) &&
          cell.z_min > ground_reference + 0.1f) {
        // Points are all above ground - likely an obstacle
        is_obstacle = true;
      }

      if (is_obstacle) {
        data[idx] = Costmap2D::OCCUPIED;
        ++obstacle_count;
      }
    }
  }

  return obstacle_count;
}

}  // namespace local_costmap
