#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "local_costmap/costmap_node.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<local_costmap::CostmapNode>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
