#ifndef LOCAL_COSTMAP__COSTMAP_2D_HPP_
#define LOCAL_COSTMAP__COSTMAP_2D_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

namespace local_costmap
{

class Costmap2D
{
public:
  // Cost values
  static constexpr int8_t FREE_SPACE = 0;
  static constexpr int8_t OCCUPIED = 100;
  static constexpr int8_t UNKNOWN = -1;  // 255 as uint8, -1 as int8

  Costmap2D(double width, double height, double resolution,
            double origin_x = 0.0, double origin_y = 0.0);

  // Reset all cells to given value
  void reset(int8_t value = FREE_SPACE);

  // Coordinate conversions
  bool worldToMap(double wx, double wy, int& mx, int& my) const;
  void mapToWorld(int mx, int my, double& wx, double& wy) const;

  // Cost access
  void setCost(int mx, int my, int8_t cost);
  int8_t getCost(int mx, int my) const;
  void setCostWorld(double wx, double wy, int8_t cost);

  // Update origin (for rolling window)
  void updateOrigin(double new_origin_x, double new_origin_y);

  // Data access
  int8_t* getData() { return data_.data(); }
  const int8_t* getData() const { return data_.data(); }
  const std::vector<int8_t>& getDataVector() const { return data_; }
  size_t getDataSize() const { return data_.size(); }

  // Getters
  double getResolution() const { return resolution_; }
  double getOriginX() const { return origin_x_; }
  double getOriginY() const { return origin_y_; }
  int getWidthCells() const { return width_cells_; }
  int getHeightCells() const { return height_cells_; }
  double getWidthMeters() const { return width_cells_ * resolution_; }
  double getHeightMeters() const { return height_cells_ * resolution_; }

private:
  double resolution_;
  double origin_x_;
  double origin_y_;
  int width_cells_;
  int height_cells_;
  std::vector<int8_t> data_;

  inline int getIndex(int mx, int my) const
  {
    return my * width_cells_ + mx;
  }

  inline bool isValid(int mx, int my) const
  {
    return mx >= 0 && mx < width_cells_ && my >= 0 && my < height_cells_;
  }
};

}  // namespace local_costmap

#endif  // LOCAL_COSTMAP__COSTMAP_2D_HPP_
