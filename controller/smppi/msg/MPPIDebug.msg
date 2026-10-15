# Per-cycle MPPI diagnostics, published at the visualization rate (odom frame)
std_msgs/Header header

# Optimal plan (updated control sequence, re-simulated)
float32 goal_cost               # goal critic on the plan (cached carrot, no carrot update)
float32 obstacle_soft_cost      # obstacle_weight * (worst clearance shortfall)^2
float32 collision_severity      # [0,1] graded by first hit + depth; x collision_cost
float32 keepout_severity        # [0,1]; x keepout_cost
float32 min_clearance           # [m] min over future poses (disc cover to nearest blocked cell)
float32 min_clearance_time      # [s] when along the plan
float32 required_at_min         # [m] d_req(v) at that pose
float32 worst_shortfall         # [m] max over the plan of d_req(v) - clearance (0 = none)
float32[] plan_speed            # [m/s] T values, speed carrying pose t to t+1
float32[] plan_clearance        # [m] T+1 values (index 0 = current pose)
float32[] plan_required         # [m] T values, d_req for poses 1..T

# Rollout population of this cycle
int32 num_rollouts
float32 collide_fraction        # share of rollouts with any future collision
float32 first_hit_median        # [s] median first collision time among colliding rollouts (-1: none)
float32 cost_min
float32 cost_median
float32 ess                     # effective sample size 1/sum(w^2): num_rollouts = uniform, 1 = a single sample decides
float32 weight_entropy          # [nats]

# Command and inputs
float32 v_cmd                   # [m/s]
float32 w_cmd                   # [rad/s]
float32 costmap_age             # [s] node clock - costmap stamp
float32 compute_ms              # this cycle, prepare -> command published (wall)
