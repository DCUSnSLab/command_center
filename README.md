
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