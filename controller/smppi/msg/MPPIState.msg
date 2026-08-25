# MPPI state representation
std_msgs/Header header
float64[] state_vector    # [x, y, theta, v]
geometry_msgs/Pose pose   # robot pose
geometry_msgs/Twist velocity  # robot velocity