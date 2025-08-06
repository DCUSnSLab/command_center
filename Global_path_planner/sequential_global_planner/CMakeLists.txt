cmake_minimum_required(VERSION 3.8)
project(sequential_global_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Install Python scripts
install(PROGRAMS
  scripts/sequential_planner_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install maps, config and utils
install(DIRECTORY
  maps/
  DESTINATION share/${PROJECT_NAME}/maps
)

install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY
  utils/
  DESTINATION share/${PROJECT_NAME}/utils
)

ament_package()