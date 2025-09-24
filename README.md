
ros2 launch pointcloud_to_laserscan vel

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/ackermann_like_controller/cmd_vel

## Simulation env(Gazebo)

### dependency

bash
```
sudo apt-get install nlohmann-json3-dev
```

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

## New localization option

- command_center_launch 경로에 있는 new_localization.launch.py 실행
    - 해당 launch에는 tiny_localization을 대체할 robot_localization이 포함되어 있음
        - robot_localization의 경우 아래 저장소의 humble_tf_option 브랜치를 워크스페이스 src 경로에 clone 후 빌드 진행하면 됨
            - git clone -b "humble_tf_option" https://github.com/junhp1234/SCV_Localization.git
    - pointcloud_to_laserscan 도 포함되어 있으니 해당 패키지 빌드 결과 적용 후 실행하면 더 편할듯?
- 주의사항
    - 이제부터 Graph Planner의 launch 가 아닌 command_center_launch 에서 사용할 map 파일 정의 가능
    - robot_localization만 따로 실행하는경우 TF가 UTM 좌표를 제대로 반영하지 못하므로 초기화 시 가급적 new_localization.launch.py 로 한번에 실행
    - 기존 tiny_localization과 달리 후진 과정에서 TF 축이 뒤집히지 않으니 해당 내용 고려해서 알고리즘 수정 필요