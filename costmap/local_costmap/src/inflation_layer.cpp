#include "local_costmap/inflation_layer.hpp"

#include <cmath>
#include <algorithm>

namespace local_costmap
{

InflationLayer::InflationLayer()
  : inflation_radius_(0.0),
    cost_scaling_factor_(10.0),
    resolution_(0.1),
    initialized_(false),
    inflation_cells_(0)
{
}

void InflationLayer::initialize(double inflation_radius, double cost_scaling_factor, double resolution)
{
  inflation_radius_ = inflation_radius;
  cost_scaling_factor_ = cost_scaling_factor;
  resolution_ = resolution;

  if (inflation_radius_ > 0.0) {
    computeKernel();
    initialized_ = true;
  } else {
    initialized_ = false;
  }
}

int8_t InflationLayer::computeCost(double distance) const
{
  if (distance <= 0.0) {
    return Costmap2D::OCCUPIED;
  }

  if (distance >= inflation_radius_) {
    return Costmap2D::FREE_SPACE;
  }

  // Exponential decay: cost = OCCUPIED * exp(-factor * distance / radius)
  double factor = cost_scaling_factor_ * distance / inflation_radius_;
  double cost = static_cast<double>(Costmap2D::OCCUPIED) * std::exp(-factor);

  // Clamp to valid range [1, OCCUPIED-1] for inflated cells
  // (0 is free, 100 is occupied)
  int8_t result = static_cast<int8_t>(std::round(cost));
  return std::max(static_cast<int8_t>(1), std::min(static_cast<int8_t>(Costmap2D::OCCUPIED - 1), result));
}

void InflationLayer::computeKernel()
{
  kernel_.clear();

  inflation_cells_ = static_cast<int>(std::ceil(inflation_radius_ / resolution_));

  // Pre-compute all cells within inflation radius
  std::vector<std::pair<double, InflationCell>> temp_kernel;

  for (int dy = -inflation_cells_; dy <= inflation_cells_; ++dy) {
    for (int dx = -inflation_cells_; dx <= inflation_cells_; ++dx) {
      // Skip center (obstacle cell itself)
      if (dx == 0 && dy == 0) {
        continue;
      }

      double distance = std::hypot(dx, dy) * resolution_;

      if (distance <= inflation_radius_) {
        int8_t cost = computeCost(distance);
        if (cost > 0) {
          temp_kernel.push_back({distance, {dx, dy, cost}});
        }
      }
    }
  }

  // Sort by distance (closest first) - helps with cache locality
  std::sort(temp_kernel.begin(), temp_kernel.end(),
    [](const auto& a, const auto& b) { return a.first < b.first; });

  // Extract sorted cells
  kernel_.reserve(temp_kernel.size());
  for (const auto& item : temp_kernel) {
    kernel_.push_back(item.second);
  }
}

void InflationLayer::inflate(Costmap2D& costmap)
{
  if (!initialized_ || kernel_.empty()) {
    return;
  }

  const int width = costmap.getWidthCells();
  const int height = costmap.getHeightCells();
  int8_t* data = costmap.getData();

  // First pass: find all obstacle cells
  std::vector<std::pair<int, int>> obstacles;
  obstacles.reserve(1000);  // Pre-allocate for typical case

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (data[y * width + x] >= Costmap2D::OCCUPIED) {
        obstacles.emplace_back(x, y);
      }
    }
  }

  // Second pass: apply inflation kernel around each obstacle
  for (const auto& obs : obstacles) {
    int ox = obs.first;
    int oy = obs.second;

    for (const auto& cell : kernel_) {
      int nx = ox + cell.dx;
      int ny = oy + cell.dy;

      // Bounds check
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        int idx = ny * width + nx;
        // Take maximum of existing cost and inflated cost
        if (cell.cost > data[idx]) {
          data[idx] = cell.cost;
        }
      }
    }
  }
}

}  // namespace local_costmap
