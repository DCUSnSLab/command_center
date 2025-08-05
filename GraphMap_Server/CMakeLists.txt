cmake_minimum_required(VERSION 3.8)
project(gmserver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/LoadMap.srv"
  "msg/GpsInfo.msg"
  "msg/UtmInfo.msg"
  "msg/MapNode.msg"
  "msg/MapLink.msg"
  "msg/MapData.msg"
  "msg/GraphMap.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

# Add executables
add_executable(map_service_node src/map_service_node.cpp)

# Specify libraries to link a library or executable target against
ament_target_dependencies(map_service_node
  rclcpp
  std_msgs
  geometry_msgs
)


target_link_libraries(map_service_node
  nlohmann_json::nlohmann_json
)

# Link generated interfaces
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(map_service_node "${cpp_typesupport_target}")

# Install executables
install(TARGETS
  map_service_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install maps directory
install(DIRECTORY
  maps/
  DESTINATION share/${PROJECT_NAME}/maps
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
ament_export_dependencies(rosidl_default_runtime)

ament_package()
