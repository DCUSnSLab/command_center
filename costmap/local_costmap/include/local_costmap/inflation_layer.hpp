#ifndef LOCAL_COSTMAP__INFLATION_LAYER_HPP_
#define LOCAL_COSTMAP__INFLATION_LAYER_HPP_

#include <vector>
#include <cstdint>

#include "local_costmap/costmap_2d.hpp"

namespace local_costmap
{

// BFS wavefront inflation (Nav2 inflation_layer algorithm):
// obstacle cells seed a breadth-first expansion ordered by distance, each
// visited cell gets the pre-computed cost for its distance to the nearest
// obstacle. O(inflated cells), independent of the number of obstacles.
class InflationLayer
{
public:
  InflationLayer();

  // Initialize with parameters (call once or when params change)
  void initialize(double inflation_radius, double cost_scaling_factor, double resolution);

  // Apply inflation to costmap
  void inflate(Costmap2D& costmap);

  // Getters
  double getInflationRadius() const { return inflation_radius_; }
  double getCostScalingFactor() const { return cost_scaling_factor_; }
  bool isInitialized() const { return initialized_; }

private:
  // Cell queued for expansion, tracking its nearest obstacle (sx, sy)
  struct CellData
  {
    int x;
    int y;
    int sx;
    int sy;
  };

  double inflation_radius_;
  double cost_scaling_factor_;
  double resolution_;
  bool initialized_;

  // Inflation radius in cells
  int cell_inflation_radius_;

  // Distance / cost lookup tables indexed by [|dy|][|dx|]
  std::vector<std::vector<double>> cached_distances_;
  std::vector<std::vector<int8_t>> cached_costs_;

  // Per-inflate() scratch (kept as members to avoid reallocation)
  std::vector<uint8_t> seen_;
  std::vector<std::vector<CellData>> bins_;

  // Pre-compute the distance/cost lookup tables
  void computeCaches();

  // Compute cost based on distance using exponential decay
  int8_t computeCost(double distance) const;

  // Queue a cell into the distance-ordered bins
  void enqueue(int x, int y, int sx, int sy, int width);
};

}  // namespace local_costmap

#endif  // LOCAL_COSTMAP__INFLATION_LAYER_HPP_
