cmake_minimum_required(VERSION 3.8)
project(local_costmap_2d)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
# find_package(rclcpp_lifecycle REQUIRED)  # Not needed for regular node
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(map_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})

# Create library
add_library(${PROJECT_NAME}
  src/costmap_2d.cpp
  src/pointcloud_processor.cpp
  src/pointcloud_filters.cpp
  src/costmap_publisher.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${PCL_LIBRARIES}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
# rclcpp_lifecycle
  sensor_msgs
  nav_msgs
  map_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  pcl_ros
  pcl_conversions
)

# Create executable
add_executable(local_costmap_node
  src/local_costmap_node.cpp
)

target_link_libraries(local_costmap_node
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
)

ament_target_dependencies(local_costmap_node
  rclcpp
# rclcpp_lifecycle
  sensor_msgs
  nav_msgs
  map_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  pcl_ros
  pcl_conversions
)

# Install targets
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS local_costmap_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(
  rclcpp
# rclcpp_lifecycle
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  pcl_ros
  pcl_conversions
)

ament_package()