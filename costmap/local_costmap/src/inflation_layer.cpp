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
    cell_inflation_radius_(0)
{
}

void InflationLayer::initialize(double inflation_radius, double cost_scaling_factor, double resolution)
{
  inflation_radius_ = inflation_radius;
  cost_scaling_factor_ = cost_scaling_factor;
  resolution_ = resolution;

  if (inflation_radius_ > 0.0) {
    cell_inflation_radius_ = static_cast<int>(std::ceil(inflation_radius_ / resolution_));
    computeCaches();
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

void InflationLayer::computeCaches()
{
  const int size = cell_inflation_radius_ + 2;

  cached_distances_.assign(size, std::vector<double>(size, 0.0));
  cached_costs_.assign(size, std::vector<int8_t>(size, 0));

  for (int dy = 0; dy < size; ++dy) {
    for (int dx = 0; dx < size; ++dx) {
      const double distance_cells = std::hypot(dx, dy);
      cached_distances_[dy][dx] = distance_cells;
      cached_costs_[dy][dx] = computeCost(distance_cells * resolution_);
    }
  }

  // Half-cell quantized distance bins for BFS processing order
  bins_.resize(2 * (cell_inflation_radius_ + 2));
}

void InflationLayer::enqueue(int x, int y, int sx, int sy, int width)
{
  const int idx = y * width + x;
  if (seen_[idx]) {
    return;
  }

  const int adx = std::abs(x - sx);
  const int ady = std::abs(y - sy);
  if (adx > cell_inflation_radius_ + 1 || ady > cell_inflation_radius_ + 1) {
    return;
  }

  const double distance_cells = cached_distances_[ady][adx];
  if (distance_cells > cell_inflation_radius_) {
    return;
  }

  const size_t bin = static_cast<size_t>(distance_cells * 2.0);
  if (bin < bins_.size()) {
    bins_[bin].push_back({x, y, sx, sy});
  }
}

void InflationLayer::inflate(Costmap2D& costmap)
{
  if (!initialized_) {
    return;
  }

  const int width = costmap.getWidthCells();
  const int height = costmap.getHeightCells();
  int8_t* data = costmap.getData();

  seen_.assign(static_cast<size_t>(width) * height, 0);
  for (auto& bin : bins_) {
    bin.clear();
  }

  // Seed the wavefront with all obstacle cells
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (data[y * width + x] >= Costmap2D::OCCUPIED) {
        bins_[0].push_back({x, y, x, y});
      }
    }
  }

  // Expand in order of increasing distance; each cell is finalized on first
  // visit, which is (approximately) from its nearest obstacle
  for (size_t bin = 0; bin < bins_.size(); ++bin) {
    // Bins grow while being processed, so index instead of iterating
    for (size_t i = 0; i < bins_[bin].size(); ++i) {
      const CellData cell = bins_[bin][i];
      const int idx = cell.y * width + cell.x;

      if (seen_[idx]) {
        continue;
      }
      seen_[idx] = 1;

      const int adx = std::abs(cell.x - cell.sx);
      const int ady = std::abs(cell.y - cell.sy);
      const int8_t cost = cached_costs_[ady][adx];

      // Take max with existing cost; never touch UNKNOWN cells
      if (data[idx] >= 0 && cost > data[idx]) {
        data[idx] = cost;
      }

      // Expand 4-connected neighbors from the same source obstacle
      if (cell.x > 0) {
        enqueue(cell.x - 1, cell.y, cell.sx, cell.sy, width);
      }
      if (cell.x < width - 1) {
        enqueue(cell.x + 1, cell.y, cell.sx, cell.sy, width);
      }
      if (cell.y > 0) {
        enqueue(cell.x, cell.y - 1, cell.sx, cell.sy, width);
      }
      if (cell.y < height - 1) {
        enqueue(cell.x, cell.y + 1, cell.sx, cell.sy, width);
      }
    }
  }
}

}  // namespace local_costmap
