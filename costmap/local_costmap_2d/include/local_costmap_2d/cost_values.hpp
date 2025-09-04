#ifndef LOCAL_COSTMAP_2D__COST_VALUES_HPP_
#define LOCAL_COSTMAP_2D__COST_VALUES_HPP_

namespace local_costmap_2d
{

// Cost values for the costmap
static constexpr unsigned char FREE_SPACE = 0;
static constexpr unsigned char LETHAL_OBSTACLE = 254;
static constexpr unsigned char INSCRIBED_INFLATED_OBSTACLE = 253;
static constexpr unsigned char MAX_NON_OBSTACLE = 252;
static constexpr unsigned char NO_INFORMATION = 255;

// Additional cost levels for different obstacle types
static constexpr unsigned char LOW_OBSTACLE = 100;
static constexpr unsigned char MEDIUM_OBSTACLE = 150;
static constexpr unsigned char HIGH_OBSTACLE = 200;

}  // namespace local_costmap_2d

#endif  // LOCAL_COSTMAP_2D__COST_VALUES_HPP_