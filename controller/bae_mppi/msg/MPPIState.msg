# Robot state for MPPI controller
std_msgs/Header header
geometry_msgs/Pose pose      # Current robot pose
geometry_msgs/Twist velocity # Current robot velocity
float64[3] state_vector      # [x, y, theta] for MPPI