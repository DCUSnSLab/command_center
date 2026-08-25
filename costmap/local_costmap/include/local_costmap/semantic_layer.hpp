#ifndef LOCAL_COSTMAP__SEMANTIC_LAYER_HPP_
#define LOCAL_COSTMAP__SEMANTIC_LAYER_HPP_

#include <cstdint>
#include <mutex>
#include <vector>

#include <nav_msgs/msg/occupancy_grid.hpp>

#include "local_costmap/costmap_2d.hpp"

namespace local_costmap
{

// Static semantic map from map_provider (~/semantic/grid), sampled into the
// rolling local costmap.
//
// Ordering: run AFTER inflation, just before publishing.
//   The semantic non-drivable region is a boundary a human drew on the map, not
//   a physical object the robot could hit. Inflating it by inflation_radius
//   (3 m) would eat 6 m out of every corridor and close narrow paths outright.
//
// Cost rule: raise only, never lower.
//   A cell marked drivable on the map may still hold a parked car right now.
//   Writing FREE_SPACE there would erase an obstacle the sensor actually saw,
//   so every cell takes max(existing, semantic).
class SemanticLayer
{
public:
  // Grid values published by map_provider (SEMANTIC_GRID_VALUE).
  static constexpr int8_t VAL_DRIVABLE = 0;
  static constexpr int8_t VAL_CROSSWALK = 50;
  static constexpr int8_t VAL_NON_DRIVABLE = 100;
  static constexpr int8_t VAL_UNKNOWN = -1;

  void initialize(int8_t crosswalk_cost, int8_t non_drivable_cost);

  // Store the latched grid (subscription callback, different thread).
  void setGrid(const nav_msgs::msg::OccupancyGrid& grid);

  bool isReady() const;

  // Sample the semantic grid into `costmap`.
  // (map_to_odom_x, map_to_odom_y, map_to_odom_yaw) transforms a point in the
  // costmap frame (odom) into the map frame. Returns cells raised, or -1 if
  // the grid has not arrived yet.
  int apply(Costmap2D& costmap,
            double map_to_odom_x, double map_to_odom_y, double map_to_odom_yaw);

private:
  int8_t costFor(int8_t semantic_value) const;

  mutable std::mutex mutex_;
  bool has_grid_ = false;
  std::vector<int8_t> data_;
  double origin_x_ = 0.0;
  double origin_y_ = 0.0;
  double resolution_ = 0.0;
  int width_ = 0;
  int height_ = 0;

  int8_t crosswalk_cost_ = 50;
  int8_t non_drivable_cost_ = Costmap2D::OCCUPIED;
};

}  // namespace local_costmap

#endif  // LOCAL_COSTMAP__SEMANTIC_LAYER_HPP_
