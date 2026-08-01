cmake_minimum_required(VERSION 3.8)
project(command_center_launch)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# Bring-up / teardown helpers (field_bringup.sh raises can0 then launches)
install(PROGRAMS
  scripts/field_bringup.sh
  scripts/teardown.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()