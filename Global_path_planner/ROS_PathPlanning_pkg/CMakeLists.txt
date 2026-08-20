cmake_minimum_required(VERSION 3.8)
project(scv_global_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(map_interfaces REQUIRED)
find_package(command_center_interfaces REQUIRED)
find_package(ament_cmake_python REQUIRED)

set(dependencies
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
  nav_msgs
  tf2_ros
  tf2
  tf2_geometry_msgs
  map_interfaces
  command_center_interfaces
)

# 계획 로직 (ROS 배선과 분리) — 그래프 + 모드별 플래너
add_library(${PROJECT_NAME}_core SHARED
  src/route_graph.cpp
  src/astar_planner.cpp
  src/sequence_planner.cpp
  src/path_visualizer.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME}_core ${dependencies})

# Add executable
add_executable(path_planner_node src/path_planner_node.cpp)
target_include_directories(path_planner_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(path_planner_node ${PROJECT_NAME}_core)
ament_target_dependencies(path_planner_node ${dependencies})

# Install executables
install(TARGETS
  path_planner_node
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS ${PROJECT_NAME}_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/
  DESTINATION include
)

# 경로 지시 PyQt 툴 (rclpy). map_editor 는 ROS 를 넣지 않기로 해서
# 렌더링에 필요한 부분만 가져와 여기에 뒀다.
ament_python_install_package(route_tool)
install(PROGRAMS
  scripts/route_tool
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Export dependencies
ament_export_dependencies(command_center_interfaces)

ament_package()
