cmake_minimum_required(VERSION 3.8)
project(loop_manager)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(map_interfaces REQUIRED)
find_package(command_center_interfaces REQUIRED)

install(PROGRAMS
  scripts/loop_manager_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
