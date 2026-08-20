#include "local_costmap/semantic_layer.hpp"

#include <cmath>

namespace local_costmap
{

void SemanticLayer::initialize(int8_t crosswalk_cost, int8_t non_drivable_cost)
{
  crosswalk_cost_ = crosswalk_cost;
  non_drivable_cost_ = non_drivable_cost;
}

void SemanticLayer::setGrid(const nav_msgs::msg::OccupancyGrid& grid)
{
  std::lock_guard<std::mutex> lock(mutex_);
  data_ = grid.data;
  origin_x_ = grid.info.origin.position.x;
  origin_y_ = grid.info.origin.position.y;
  resolution_ = grid.info.resolution;
  width_ = static_cast<int>(grid.info.width);
  height_ = static_cast<int>(grid.info.height);
  has_grid_ = (resolution_ > 0.0 && width_ > 0 && height_ > 0 &&
               data_.size() == static_cast<size_t>(width_) * height_);
}

bool SemanticLayer::isReady() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return has_grid_;
}

int8_t SemanticLayer::costFor(int8_t v) const
{
  switch (v) {
    case VAL_NON_DRIVABLE:
      return non_drivable_cost_;
    case VAL_CROSSWALK:
      return crosswalk_cost_;
    default:
      // drivable(0) and unknown(-1) add nothing. Marking drivable as free here
      // would erase obstacles the sensor just saw.
      return 0;
  }
}

int SemanticLayer::apply(Costmap2D& costmap,
                         double map_to_odom_x, double map_to_odom_y,
                         double map_to_odom_yaw)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!has_grid_) {
    return -1;
  }

  const double cs = std::cos(map_to_odom_yaw);
  const double sn = std::sin(map_to_odom_yaw);
  const double res = costmap.getResolution();
  const double ox = costmap.getOriginX();
  const double oy = costmap.getOriginY();
  const int w = costmap.getWidthCells();
  const int h = costmap.getHeightCells();
  const double inv_res = 1.0 / resolution_;

  int raised = 0;
  for (int my = 0; my < h; ++my) {
    // Cell centre in the costmap (odom) frame
    const double ly = oy + (my + 0.5) * res;
    for (int mx = 0; mx < w; ++mx) {
      const double lx = ox + (mx + 0.5) * res;

      // odom -> map
      const double wx = cs * lx - sn * ly + map_to_odom_x;
      const double wy = sn * lx + cs * ly + map_to_odom_y;

      // map -> semantic grid cell (nearest; the semantic grid is coarser)
      const int gx = static_cast<int>((wx - origin_x_) * inv_res);
      const int gy = static_cast<int>((wy - origin_y_) * inv_res);
      if (gx < 0 || gx >= width_ || gy < 0 || gy >= height_) {
        continue;                       // outside the mapped area: leave as is
      }

      const int8_t add = costFor(data_[static_cast<size_t>(gy) * width_ + gx]);
      if (add <= 0) {
        continue;
      }
      const int8_t cur = costmap.getCost(mx, my);
      // UNKNOWN(-1) must lose to any real cost, so compare as "unknown is lowest".
      if (cur == Costmap2D::UNKNOWN || add > cur) {
        costmap.setCost(mx, my, add);
        ++raised;
      }
    }
  }
  return raised;
}

}  // namespace local_costmap
