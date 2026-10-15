# Subsample of the MPPI rollouts for visualization (odom frame)
std_msgs/Header header
int32 num_samples
int32 num_poses                 # T+1
float32[] xy                    # num_samples * num_poses * 2, row-major (sample, pose, xy)
float32[] cost                  # total cost per sample (critics + action smoothing)
float32[] weight                # softmax weight per sample (sums to 1 over all rollouts)
