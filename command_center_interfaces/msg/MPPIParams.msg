# SMPPI Dynamic Parameter Update Message
std_msgs/Header header

# Optimizer parameters
bool update_optimizer
int32 batch_size
int32 time_steps
float64 model_dt
float64 temperature
float64 lambda_action
float64[] noise_std_u
float64[] omega_diag

# Vehicle parameters
bool update_vehicle
float64 wheelbase
float64 max_linear_velocity
float64 min_linear_velocity
float64 max_angular_velocity
float64 min_angular_velocity
float64 max_steering_angle
float64[] footprint
float64 footprint_padding
float64 radius

# Cost function weights
bool update_costs
float64 obstacle_weight
float64 goal_weight

# Lookahead parameters
bool update_lookahead
float64 lookahead_base_distance
float64 lookahead_velocity_factor
float64 lookahead_min_distance
float64 lookahead_max_distance

# Goal critic parameters
bool update_goal_critic
float64 xy_goal_tolerance
float64 yaw_goal_tolerance
float64 distance_scale
float64 angle_scale
float64 alignment_scale
float64 progress_scale
bool use_progress_reward
bool respect_reverse_heading
float64 yaw_blend_distance

# Obstacle critic parameters
bool update_obstacle_critic
float64 safety_radius
float64 collision_cost
float64 repulsion_factor
float64 vehicle_radius
float64 max_range

# Control parameters
bool update_control
float64 control_frequency
float64 goal_reached_threshold
bool force_stop  # Force vehicle to stop (emergency/pause)

# Multiple waypoints
bool update_waypoints
bool use_multiple_waypoints

# Debug parameters
bool update_debug
bool debug
int32 debug_level