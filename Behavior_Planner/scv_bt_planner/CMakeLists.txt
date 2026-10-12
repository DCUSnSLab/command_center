cmake_minimum_required(VERSION 3.16)
project(scv_bt_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(command_center_interfaces REQUIRED)
find_package(map_interfaces REQUIRED)
# 외부 의존 2종은 command_center 저장소 밖에 있다(hunter_msgs: SCV 슈퍼프로젝트 hunter_ros2,
# behaviortree_cpp: apt ros-humble-behaviortree-cpp 4.x 또는 src/third_party 서브모듈).
# 없으면 패키지를 건너뛴다 — behavior_planner:=simple(기본) 인 팀원 워크스페이스 빌드가 깨지지 않게.
find_package(hunter_msgs QUIET)
find_package(behaviortree_cpp QUIET)
if(NOT hunter_msgs_FOUND OR NOT behaviortree_cpp_FOUND)
  set(_missing "")
  if(NOT hunter_msgs_FOUND)
    string(APPEND _missing " hunter_msgs(SCV 슈퍼프로젝트 src/vehicle/hunter_ros2)")
  endif()
  if(NOT behaviortree_cpp_FOUND)
    string(APPEND _missing " behaviortree_cpp(sudo apt install ros-humble-behaviortree-cpp)")
  endif()
  message(WARNING "scv_bt_planner: 의존 패키지 없음 —${_missing}. 이 패키지를 건너뜁니다 "
                  "(behavior_planner:=bt|shadow 사용 불가, simple 은 영향 없음).")
  return()
endif()
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(ament_index_cpp REQUIRED)

# ---- ROS 무의존 코어 (단위시험 대상) -------------------------------------
add_library(${PROJECT_NAME}_core
  src/path_manager.cpp
  src/hazard_waypoints.cpp
  src/blocked_wait_monitor.cpp
  src/zone_table.cpp
  src/profiles.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}_core PUBLIC
  nlohmann_json::nlohmann_json yaml-cpp)

# ---- BT 노드 + ROS 노드 ---------------------------------------------------
add_executable(bt_planner_node
  src/bt_nodes.cpp
  src/bt_planner_node.cpp
)
target_link_libraries(bt_planner_node ${PROJECT_NAME}_core behaviortree_cpp::behaviortree_cpp)
ament_target_dependencies(bt_planner_node
  rclcpp std_msgs geometry_msgs nav_msgs tf2 tf2_ros tf2_geometry_msgs
  command_center_interfaces map_interfaces hunter_msgs ament_index_cpp)

install(TARGETS ${PROJECT_NAME}_core
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS bt_planner_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY behavior_trees config launch DESTINATION share/${PROJECT_NAME})

# ---- 단위시험 -------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_path_manager test/test_path_manager.cpp)
  target_link_libraries(test_path_manager ${PROJECT_NAME}_core)
  ament_add_gtest(test_blocked_wait_monitor test/test_blocked_wait_monitor.cpp)
  target_link_libraries(test_blocked_wait_monitor ${PROJECT_NAME}_core)
  ament_add_gtest(test_zone_and_profiles test/test_zone_and_profiles.cpp)
  target_link_libraries(test_zone_and_profiles ${PROJECT_NAME}_core)
  ament_add_gtest(test_hazard_waypoints test/test_hazard_waypoints.cpp)
  target_link_libraries(test_hazard_waypoints ${PROJECT_NAME}_core)
  target_compile_definitions(test_zone_and_profiles PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_core)
ament_package()
