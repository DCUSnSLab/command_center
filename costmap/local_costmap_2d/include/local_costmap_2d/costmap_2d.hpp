#ifndef LOCAL_COSTMAP_2D__COSTMAP_2D_HPP_
#define LOCAL_COSTMAP_2D__COSTMAP_2D_HPP_

#include <vector>
#include <mutex>
#include <memory>
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "local_costmap_2d/cost_values.hpp"

namespace local_costmap_2d
{

struct MapLocation
{
  unsigned int x;
  unsigned int y;
};

class SimpleCostmap2D
{
public:
  SimpleCostmap2D(
    unsigned int cells_size_x, unsigned int cells_size_y, double resolution,
    double origin_x, double origin_y, unsigned char default_value = FREE_SPACE);
  
  explicit SimpleCostmap2D(const nav_msgs::msg::OccupancyGrid & map);
  
  SimpleCostmap2D(const SimpleCostmap2D & map);
  SimpleCostmap2D & operator=(const SimpleCostmap2D & map);
  
  SimpleCostmap2D();
  virtual ~SimpleCostmap2D();

  // Basic costmap operations
  unsigned char getCost(unsigned int mx, unsigned int my) const;
  unsigned char getCost(unsigned int index) const;
  void setCost(unsigned int mx, unsigned int my, unsigned char cost);
  
  // Coordinate transformations
  void mapToWorld(unsigned int mx, unsigned int my, double & wx, double & wy) const;
  bool worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my) const;
  void worldToMapNoBounds(double wx, double wy, int & mx, int & my) const;
  void worldToMapEnforceBounds(double wx, double wy, int & mx, int & my) const;
  
  // Index operations
  inline unsigned int getIndex(unsigned int mx, unsigned int my) const
  {
    return my * size_x_ + mx;
  }
  
  inline void indexToCells(unsigned int index, unsigned int & mx, unsigned int & my) const
  {
    my = index / size_x_;
    mx = index - (my * size_x_);
  }
  
  // Accessors
  unsigned char * getCharMap() const { return costmap_; }
  unsigned int getSizeInCellsX() const { return size_x_; }
  unsigned int getSizeInCellsY() const { return size_y_; }
  double getSizeInMetersX() const { return (size_x_ - 1 + 0.5) * resolution_; }
  double getSizeInMetersY() const { return (size_y_ - 1 + 0.5) * resolution_; }
  double getOriginX() const { return origin_x_; }
  double getOriginY() const { return origin_y_; }
  double getResolution() const { return resolution_; }
  
  // Map operations
  void resetMap(unsigned int x0, unsigned int y0, unsigned int xn, unsigned int yn);
  void resetMapToValue(unsigned int x0, unsigned int y0, unsigned int xn, unsigned int yn, unsigned char value);
  void resizeMap(unsigned int size_x, unsigned int size_y, double resolution, double origin_x, double origin_y);
  
  // PointCloud specific operations
  void projectPoint(double x, double y, double z = 0.0);
  void markObstacleWithHeight(double x, double y, double z, unsigned char cost = LETHAL_OBSTACLE);
  void processPointBatch(const std::vector<geometry_msgs::msg::Point>& points);
  
  // Temporal decay for dynamic obstacles
  void applyTemporalDecay(double decay_rate = 0.95);
  
  // Clear robot footprint
  void clearRobotFootprint(const std::vector<geometry_msgs::msg::Point>& footprint, 
                          double robot_x, double robot_y, double robot_yaw);
  
  // Ray tracing for clearing free space
  void raytrace(double x0, double y0, double x1, double y1);
  
  // Get obstacle points for external use
  const std::vector<geometry_msgs::msg::Point>& getObstaclePoints() const 
  { 
    return obstacle_points_; 
  }
  
  // Convert to OccupancyGrid message
  nav_msgs::msg::OccupancyGrid toOccupancyGrid(const std::string& frame_id) const;
  
  // Thread safety
  std::mutex* getMutex() { return &access_mutex_; }

protected:
  // Helper methods
  bool isPointInPolygon(double x, double y, const std::vector<geometry_msgs::msg::Point>& polygon) const;

private:
  void deleteMaps();
  void resetMaps();
  void initMaps(unsigned int size_x, unsigned int size_y);
  
  // Ray tracing implementation
  template<class ActionType>
  inline void raytraceLine(ActionType at, unsigned int x0, unsigned int y0, 
                          unsigned int x1, unsigned int y1);

  // Member variables
  unsigned int size_x_;
  unsigned int size_y_;
  double resolution_;
  double origin_x_;
  double origin_y_;
  unsigned char * costmap_;
  unsigned char default_value_;
  
  // PointCloud specific data
  std::vector<geometry_msgs::msg::Point> obstacle_points_;
  
  // Thread safety
  std::mutex access_mutex_;
  
  // Helper classes for ray tracing
  class MarkCell
  {
  public:
    MarkCell(unsigned char * costmap, unsigned char value)
    : costmap_(costmap), value_(value) {}
    
    inline void operator()(unsigned int offset)
    {
      costmap_[offset] = value_;
    }
    
  private:
    unsigned char * costmap_;
    unsigned char value_;
  };
};

}  // namespace local_costmap_2d

#endif  // LOCAL_COSTMAP_2D__COSTMAP_2D_HPP_