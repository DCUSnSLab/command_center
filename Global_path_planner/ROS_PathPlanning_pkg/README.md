아래 패키지 선행 설치 필요
https://github.com/DCUSnSLab/GraphMap_Server

이후 아래 명령어 순차 실행으로 path_planning 수행

```bash
# Map Server 실행
ros2 launch gmserver map_service.launch.py

# Global Planner 실행
ros2 launch scv_global_planner path_planner.launch.py
```

### Sub Topic
- Gps sensor
  - /gps/fix
  - sensor_msgs/msg/NavSatFix
- Imu sensor
  - /imu/data
  - sensor_msgs/msg/Imu

### Pub Topic
- Created Path
  - /planned_path
  - nav_msgs/Path Message
