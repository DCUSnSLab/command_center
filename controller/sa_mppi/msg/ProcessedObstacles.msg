# Processed obstacle points for MPPI controller
std_msgs/Header header
geometry_msgs/Point[] obstacle_points
float64[] distances  # distance to each obstacle point
float64[] costs      # computed cost for each obstacle point