#ifndef LOCAL_COSTMAP__INFLATION_LAYER_HPP_
#define LOCAL_COSTMAP__INFLATION_LAYER_HPP_

#include <vector>
#include <cstdint>

#include "local_costmap/costmap_2d.hpp"

namespace local_costmap
{

struct InflationCell
{
  int dx;      // Offset from obstacle cell
  int dy;
  int8_t cost; // Pre-computed cost at this distance
};

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
  double inflation_radius_;
  double cost_scaling_factor_;
  double resolution_;
  bool initialized_;

  // Pre-computed inflation kernel
  // Sorted by distance (closest first) for early termination
  std::vector<InflationCell> kernel_;

  // Inflation radius in cells
  int inflation_cells_;

  // Pre-compute the inflation kernel
  void computeKernel();

  // Compute cost based on distance using exponential decay
  int8_t computeCost(double distance) const;
};

}  // namespace local_costmap

#endif  // LOCAL_COSTMAP__INFLATION_LAYER_HPP_
