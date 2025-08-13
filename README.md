# ROS 2 실행 명령어

## Gazebo
```bash
ros2 launch scv_robot_gazebo hunter_test.launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/ackermann_like_controller/cmd_vel
```
## SCV
```bash
ros2 launch tiny_localization tiny_localization.launch.py
ros2 launch pointcloud_to_laserscan velodyne_to_scan.launch.py
```
## Command Center
```bash
ros2 launch gmserver map_service.launch.py
ros2 launch scv_global_planner path_planner.launch.py
ros2 launch simple_behavior_planner simple_behavior_planner.launch.py
ros2 launch bae_mppi mppi_distributed.launch.py
```
# 또는 (Sequential Global Planner 사용)
```bash
ros2 launch command_center_launch command_center.launch.py
```