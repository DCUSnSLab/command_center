
## Build the package:
```bash
cd /home/d2-521-30/repo/local_ws/
colcon build --packages-select bae_mppi
source install/setup.bash
```

## Usage

### Launch the controller:
```bash
ros2 launch bae_mppi mppi_controller.launch.py
```

### With custom config:
```bash
ros2 launch bae_mppi mppi_controller.launch.py config_file:=/path/to/custom_config.yaml
```

### With GPU acceleration:
```bash
ros2 launch bae_mppi mppi_controller.launch.py use_gpu:=true
```

## Topics

### Subscribed Topics
- `/odom` (nav_msgs/Odometry): Robot odometry
- `/scan` (sensor_msgs/LaserScan): Laser scan for obstacle detection  
- `/goal_pose` (geometry_msgs/PoseStamped): Goal pose

### Published Topics
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands

## Configuration

### MPPI Parameters
- `horizon_steps`: Prediction horizon length
- `num_samples`: Number of trajectory samples
- `lambda_`: Temperature parameter
- `sigma`: Control noise [linear, angular]

### Robot Constraints
- `max_linear_vel`: Maximum linear velocity (m/s)
- `max_angular_vel`: Maximum angular velocity (rad/s)

### Cost Function Weights
- `obstacle_cost.safety_radius`: Minimum safe distance (m)
- `goal_cost.goal_weight`: Goal attraction weight
- `control_cost.linear_weight`: Linear velocity penalty


## 남은일
### foot print