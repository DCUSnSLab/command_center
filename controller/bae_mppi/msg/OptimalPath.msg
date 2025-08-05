# Optimal path from MPPI controller
std_msgs/Header header
geometry_msgs/Point[] path_points  # Optimal trajectory points
float32 current_velocity          # Current commanded velocity
float32 current_steering_angle    # Current commanded steering angle

# High cost paths for visualization
HighCostPath[] high_cost_paths    # Array of high cost trajectory paths