cmake_minimum_required(VERSION 3.8)
project(local_costmap)

# C++17 standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

# Library
add_library(${PROJECT_NAME}_lib SHARED
  src/costmap_2d.cpp
  src/point_cloud_processor.cpp
  src/inflation_layer.cpp
  src/denoise_layer.cpp
  src/semantic_layer.cpp
  src/costmap_node.cpp
)

ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_sensor_msgs
)

target_link_libraries(${PROJECT_NAME}_lib
  Eigen3::Eigen
)

# Executable
add_executable(costmap_node src/main.cpp)
target_link_libraries(costmap_node ${PROJECT_NAME}_lib)

ament_target_dependencies(costmap_node
  rclcpp
)

# Install
install(TARGETS
  ${PROJECT_NAME}_lib
  costmap_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
