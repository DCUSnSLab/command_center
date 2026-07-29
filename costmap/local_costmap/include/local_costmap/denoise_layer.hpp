#ifndef LOCAL_COSTMAP__DENOISE_LAYER_HPP_
#define LOCAL_COSTMAP__DENOISE_LAYER_HPP_

#include <vector>
#include <cstdint>

#include "local_costmap/costmap_2d.hpp"

namespace local_costmap
{

// Salt-and-pepper noise removal (Nav2 denoise_layer):
// connected groups of OCCUPIED cells (8-connectivity) smaller than
// minimal_group_size are cleared to FREE_SPACE. Run on the obstacle grid
// BEFORE inflation, so isolated lidar noise never becomes an obstacle.
class DenoiseLayer
{
public:
  // minimal_group_size <= 1 disables denoising
  void initialize(int minimal_group_size);

  // Remove small obstacle groups from the costmap
  void apply(Costmap2D& costmap);

  bool isEnabled() const { return minimal_group_size_ > 1; }

private:
  int minimal_group_size_ = 0;

  // Per-apply() scratch (kept as members to avoid reallocation)
  std::vector<uint8_t> visited_;
  std::vector<int> stack_;
  std::vector<int> group_;
};

}  // namespace local_costmap

#endif  // LOCAL_COSTMAP__DENOISE_LAYER_HPP_
