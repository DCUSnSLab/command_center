cmake_minimum_required(VERSION 3.8)
project(bae_mppi)

# --- dependencies ---
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)     # <-- 추가
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# --- messages ---
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ProcessedObstacles.msg"
  "msg/MPPIState.msg"
  "msg/OptimalPath.msg"
  "msg/HighCostPath.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# --- Python package 설치 (중복 호출 금지) ---
ament_python_install_package(bae_mppi_core)  # <-- 이 줄 '한 번만'

# --- executables ---
install(PROGRAMS
  scripts/sensor_processor_node.py
  scripts/mppi_core_node.py
  scripts/visualization_node.py
  scripts/steering_validation_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# --- data ---
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
