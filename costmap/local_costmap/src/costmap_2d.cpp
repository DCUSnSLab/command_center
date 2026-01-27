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
  mx = static_cast<int>((wx - origin_x_) / resolution_);
  my = static_cast<int>((wy - origin_y_) / resolution_);

  return isValid(mx, my);
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

}  // namespace local_costmap
