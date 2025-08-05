# Processed obstacle information
std_msgs/Header header
geometry_msgs/Point[] obstacle_points  # Obstacle points in map frame
float32[] ranges                        # Raw laser ranges
float32[] angles                        # Corresponding angles
float32 min_distance                    # Minimum distance to any obstacle