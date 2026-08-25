#include "local_costmap/denoise_layer.hpp"

namespace local_costmap
{

void DenoiseLayer::initialize(int minimal_group_size)
{
  minimal_group_size_ = minimal_group_size;
}

void DenoiseLayer::apply(Costmap2D& costmap)
{
  if (!isEnabled()) {
    return;
  }

  const int width = costmap.getWidthCells();
  const int height = costmap.getHeightCells();
  int8_t* data = costmap.getData();
  const size_t size = static_cast<size_t>(width) * height;

  visited_.assign(size, 0);

  for (int start = 0; start < static_cast<int>(size); ++start) {
    if (visited_[start] || data[start] < Costmap2D::OCCUPIED) {
      continue;
    }

    // Flood-fill the connected group (8-connectivity)
    stack_.clear();
    group_.clear();
    stack_.push_back(start);
    visited_[start] = 1;

    while (!stack_.empty()) {
      const int idx = stack_.back();
      stack_.pop_back();
      group_.push_back(idx);

      const int x = idx % width;
      const int y = idx / width;

      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) {
            continue;
          }
          const int nx = x + dx;
          const int ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            continue;
          }
          const int nidx = ny * width + nx;
          if (!visited_[nidx] && data[nidx] >= Costmap2D::OCCUPIED) {
            visited_[nidx] = 1;
            stack_.push_back(nidx);
          }
        }
      }
    }

    if (static_cast<int>(group_.size()) < minimal_group_size_) {
      for (const int idx : group_) {
        data[idx] = Costmap2D::FREE_SPACE;
      }
    }
  }
}

}  // namespace local_costmap
