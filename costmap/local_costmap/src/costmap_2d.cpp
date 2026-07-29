#include "local_costmap/costmap_2d.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>

namespace local_costmap
{

Costmap2D::Costmap2D(double width, double height, double resolution,
                     double origin_x, double origin_y)
  : resolution_(resolution),
    origin_x_(origin_x),
    origin_y_(origin_y)
{
  width_cells_ = static_cast<int>(std::ceil(width / resolution));
  height_cells_ = static_cast<int>(std::ceil(height / resolution));

  // Initialize grid with free space
  data_.resize(width_cells_ * height_cells_, FREE_SPACE);
}

void Costmap2D::reset(int8_t value)
{
  std::memset(data_.data(), value, data_.size());
}

bool Costmap2D::worldToMap(double wx, double wy, int& mx, int& my) const
{
  worldToMapNoBounds(wx, wy, mx, my);
  return isValid(mx, my);
}

void Costmap2D::worldToMapNoBounds(double wx, double wy, int& mx, int& my) const
{
  // floor, not truncation: cells left/below the origin must stay negative
  mx = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
  my = static_cast<int>(std::floor((wy - origin_y_) / resolution_));
}

void Costmap2D::mapToWorld(int mx, int my, double& wx, double& wy) const
{
  wx = origin_x_ + (mx + 0.5) * resolution_;
  wy = origin_y_ + (my + 0.5) * resolution_;
}

void Costmap2D::setCost(int mx, int my, int8_t cost)
{
  if (isValid(mx, my)) {
    data_[getIndex(mx, my)] = cost;
  }
}

int8_t Costmap2D::getCost(int mx, int my) const
{
  if (isValid(mx, my)) {
    return data_[getIndex(mx, my)];
  }
  return UNKNOWN;
}

void Costmap2D::setCostWorld(double wx, double wy, int8_t cost)
{
  int mx, my;
  if (worldToMap(wx, wy, mx, my)) {
    setCost(mx, my, cost);
  }
}

void Costmap2D::updateOrigin(double new_origin_x, double new_origin_y)
{
  origin_x_ = new_origin_x;
  origin_y_ = new_origin_y;
}

void Costmap2D::shiftOrigin(double new_origin_x, double new_origin_y, int8_t fill_value)
{
  // Snap the shift to whole cells so retained data stays cell-aligned
  const int cell_dx = static_cast<int>(std::floor((new_origin_x - origin_x_) / resolution_));
  const int cell_dy = static_cast<int>(std::floor((new_origin_y - origin_y_) / resolution_));

  if (cell_dx == 0 && cell_dy == 0) {
    return;
  }

  const double snapped_ox = origin_x_ + cell_dx * resolution_;
  const double snapped_oy = origin_y_ + cell_dy * resolution_;

  if (std::abs(cell_dx) >= width_cells_ || std::abs(cell_dy) >= height_cells_) {
    // No overlap with the previous window
    reset(fill_value);
  } else {
    // New cell (x, y) corresponds to old cell (x + cell_dx, y + cell_dy)
    std::vector<int8_t> old_data = data_;
    std::memset(data_.data(), fill_value, data_.size());

    const int x_start = std::max(0, -cell_dx);
    const int x_end = std::min(width_cells_, width_cells_ - cell_dx);
    const int y_start = std::max(0, -cell_dy);
    const int y_end = std::min(height_cells_, height_cells_ - cell_dy);

    for (int y = y_start; y < y_end; ++y) {
      std::memcpy(&data_[getIndex(x_start, y)],
                  &old_data[getIndex(x_start + cell_dx, y + cell_dy)],
                  x_end - x_start);
    }
  }

  origin_x_ = snapped_ox;
  origin_y_ = snapped_oy;
}

void Costmap2D::raytraceSetLine(int x0, int y0, int x1, int y1, int8_t value)
{
  if (!isValid(x0, y0)) {
    return;
  }

  // Bresenham; the map is convex, so once the ray leaves it never re-enters
  const int dx = std::abs(x1 - x0);
  const int dy = std::abs(y1 - y0);
  const int sx = (x0 < x1) ? 1 : -1;
  const int sy = (y0 < y1) ? 1 : -1;
  int err = dx - dy;
  int x = x0;
  int y = y0;

  while (!(x == x1 && y == y1)) {  // endpoint cell excluded
    if (!isValid(x, y)) {
      return;
    }
    data_[getIndex(x, y)] = value;

    const int e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x += sx;
    }
    if (e2 < dx) {
      err += dx;
      y += sy;
    }
  }
}

void Costmap2D::copyFrom(const Costmap2D& other)
{
  if (other.width_cells_ != width_cells_ || other.height_cells_ != height_cells_) {
    return;
  }
  origin_x_ = other.origin_x_;
  origin_y_ = other.origin_y_;
  std::memcpy(data_.data(), other.data_.data(), data_.size());
}

}  // namespace local_costmap
