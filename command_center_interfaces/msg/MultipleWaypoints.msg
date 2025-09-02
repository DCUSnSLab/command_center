# Multiple waypoints message for smooth path following
# Provides current goal and lookahead waypoints to prevent speed reduction near goals

std_msgs/Header header

# Current target waypoint (what we're heading to now)
geometry_msgs/PoseStamped current_goal

# Next waypoints for lookahead (typically 1-3 waypoints ahead)
geometry_msgs/PoseStamped[] next_waypoints

# Path information
string path_id                    # Path identifier
int32 current_waypoint_index      # Index of current goal in full path
int32 total_waypoints             # Total number of waypoints in path
bool is_final_waypoint            # True if current_goal is the final destination