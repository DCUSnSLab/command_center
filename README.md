
ros2 launch pointcloud_to_laserscan vel

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/ackermann_like_controller/cmd_vel

## Simulation env(Gazebo)

### commands

bash
```
# Run gazebo simulation
ros2 launch scv_robot_gazebo hunter_test.launch.py

# Mapserver & global planner
ros2 launch gmserver map_service.launch.py
ros2 launch scv_global_planner path_planner.launch.py
ros2 launch tiny_localization tiny_localization.launch.py

# 위 명령어 실행 후 수동 조작을 통해 odom -> base_link TF 변환 및 Localization 진행 필요
# 수동 조작 명령어
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/ackermann_like_controller/cmd_vel

# simple_planner 동 실행
ros2 launch bae_mppi mppi_distributed.launch.py
ros2 launch simple_behavior_planner simple_behavior_planner.launch.py
ros2 launch pointcloud_to_laserscan velodyne_to_scan.launch.py
```

### 알려진 이슈

- 시뮬레이션 환경에서 tiny_localization 장시간 실행 시 odom -> base_link TF 변환이 반시계축으로 회전하는 drift 발생
    - 해당 현상은 실제 차량에서는 발생하지 않음
- 때때로 Marker, Path 시각화 맞지 않음
    - Marker 갱신 문제이므로 Goal 지정 시 실제로는 정상적으로 동작하는듯?
- Rviz의 fixed_frame이 map인 경우 오동작
- Path planning 과정에서 Link의 존재 무시