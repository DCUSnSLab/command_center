#!/usr/bin/env python3
"""
Core MPPI Controller Node
Pure MPPI computation without sensor processing or visualization
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import torch
import numpy as np
import time

# ROS2 messages
from geometry_msgs.msg import Twist, PoseStamped, Point
from std_msgs.msg import Header

# Custom messages
from bae_mppi.msg import ProcessedObstacles, MPPIState, OptimalPath, HighCostPath
from command_center_interfaces.msg import ControllerGoalStatus

# Local modules
from bae_mppi_core.pytorch_mppi import MPPI
from bae_mppi_core.dynamics import AckermannDynamics, TwistDynamics
from bae_mppi_core.cost_functions import CombinedCostFunction


class MPPICoreNode(Node):
    """Core MPPI controller - pure computation"""
    
    def __init__(self):
        super().__init__('mppi_core')
        
        # Topic parameters
        self.declare_parameter('topics.input.robot_state', 'state')
        self.declare_parameter('topics.input.processed_obstacles', 'obstacles')
        self.declare_parameter('topics.input.goal_pose', '/goal_pose')
        self.declare_parameter('topics.output.cmd_vel', '/ackermann_like_controller/cmd_vel')
        self.declare_parameter('topics.output.optimal_path', 'optimal_path')
        self.declare_parameter('topics.output.goal_status', '/goal_status')
        
        # Parameters
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('horizon_steps', 50)
        self.declare_parameter('num_samples', 5000)
        self.declare_parameter('lambda_', 1.0)
        self.declare_parameter('sigma', [0.8, 0.8])
        self.declare_parameter('max_linear_vel', 1.5)
        self.declare_parameter('max_steering_angle', 0.39)
        self.declare_parameter('max_angular_vel', 1.0)  # Added for twist model
        self.declare_parameter('wheelbase', 0.65)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('motion_model', 'twist')  # 'ackermann' or 'twist'
        
        # Visualization parameters
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('enable_path_viz', True)
        self.declare_parameter('enable_best_paths', True)
        self.declare_parameter('num_best_paths', 10)
        
        # Cost function parameters
        self.declare_parameter('obstacle_cost.safety_radius', 0.8)
        self.declare_parameter('obstacle_cost.max_range', 100.0)
        self.declare_parameter('obstacle_cost.penalty_weight', 1000.0)
        self.declare_parameter('obstacle_cost.exponential_factor', 3.0)
        self.declare_parameter('goal_cost.goal_weight', 0.3)
        self.declare_parameter('goal_cost.angle_weight', 0.5)
        
        # Get parameters
        use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        self.control_frequency = self.get_parameter('control_frequency').get_parameter_value().double_value
        self.enable_visualization = self.get_parameter('enable_visualization').get_parameter_value().bool_value
        self.enable_path_viz = self.get_parameter('enable_path_viz').get_parameter_value().bool_value
        self.enable_best_paths = self.get_parameter('enable_best_paths').get_parameter_value().bool_value
        self.num_best_paths = self.get_parameter('num_best_paths').get_parameter_value().integer_value
        horizon_steps = self.get_parameter('horizon_steps').get_parameter_value().integer_value
        num_samples = self.get_parameter('num_samples').get_parameter_value().integer_value
        lambda_ = self.get_parameter('lambda_').get_parameter_value().double_value
        sigma = self.get_parameter('sigma').get_parameter_value().double_array_value
        max_linear_vel = self.get_parameter('max_linear_vel').get_parameter_value().double_value
        max_steering_angle = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        max_angular_vel = self.get_parameter('max_angular_vel').get_parameter_value().double_value
        wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        dt = self.get_parameter('dt').get_parameter_value().double_value
        self.motion_model = self.get_parameter('motion_model').get_parameter_value().string_value
        
        # Setup device
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')
        
        # Initialize MPPI dynamics based on motion model
        if self.motion_model == 'twist':
            self.dynamics = TwistDynamics(dt=dt, device=self.device)
            self.get_logger().info('Using Twist dynamics model')
        else:  # 'ackermann'
            self.dynamics = AckermannDynamics(wheelbase=wheelbase, dt=dt, device=self.device)
            self.get_logger().info('Using Ackermann dynamics model')
        
        self.wheelbase = wheelbase
        
        # Cost function with parameters
        obstacle_params = {
            'safety_radius': self.get_parameter('obstacle_cost.safety_radius').get_parameter_value().double_value,
            'max_range': self.get_parameter('obstacle_cost.max_range').get_parameter_value().double_value,
            'penalty_weight': self.get_parameter('obstacle_cost.penalty_weight').get_parameter_value().double_value,
            'exponential_factor': self.get_parameter('obstacle_cost.exponential_factor').get_parameter_value().double_value,
        }
        self.cost_function = CombinedCostFunction(device=self.device, obstacle_params=obstacle_params)
        
        # Set goal cost parameters
        self.cost_function.goal_cost.goal_weight = self.get_parameter('goal_cost.goal_weight').get_parameter_value().double_value
        self.cost_function.goal_cost.angle_weight = self.get_parameter('goal_cost.angle_weight').get_parameter_value().double_value
        
        # Control bounds based on motion model
        nx = 3  # [x, y, theta]
        nu = 2  # [v, w] or [v, delta]
        
        if self.motion_model == 'twist':
            # Twist model: [vx, wz]
            u_min = torch.tensor([-max_linear_vel, -max_angular_vel], device=self.device)
            u_max = torch.tensor([max_linear_vel, max_angular_vel], device=self.device)
        else:
            # Ackermann model: [v_rear, delta]
            u_min = torch.tensor([-max_linear_vel, -max_steering_angle], device=self.device)
            u_max = torch.tensor([max_linear_vel, max_steering_angle], device=self.device)
        
        # Initialize MPPI
        noise_sigma = torch.diag(torch.tensor(sigma, device=self.device))
        
        # Debug MPPI parameters
        self.get_logger().info(f"[MPPI INIT] Samples: {num_samples}, Horizon: {horizon_steps}")
        self.get_logger().info(f"[MPPI INIT] Sigma: {sigma}, Lambda: {lambda_}")
        u_min_vals = [float(u_min[0]), float(u_min[1])]
        u_max_vals = [float(u_max[0]), float(u_max[1])]
        
        if self.motion_model == 'ackermann':
            self.get_logger().info(f"[MPPI INIT] Control bounds - Velocity: [{u_min_vals[0]:.2f}, {u_max_vals[0]:.2f}] m/s, Steering: [{u_min_vals[1]:.2f}, {u_max_vals[1]:.2f}] rad")
            self.get_logger().info(f"[MPPI INIT] Wheelbase: {wheelbase:.2f}m, Max steering: {max_steering_angle:.3f} rad ({max_steering_angle*57.3:.1f}°)")
        else:
            self.get_logger().info(f"[MPPI INIT] Control bounds - Linear: [{u_min_vals[0]:.2f}, {u_max_vals[0]:.2f}], Angular: [{u_min_vals[1]:.2f}, {u_max_vals[1]:.2f}]")
            
        sigma_diag = [float(noise_sigma[0,0]), float(noise_sigma[1,1])]
        self.get_logger().info(f"[MPPI INIT] Noise sigma diagonal: [{sigma_diag[0]:.1f}, {sigma_diag[1]:.1f}]")
        
        self.mppi = MPPI(
            dynamics=self.dynamics,
            running_cost=self.cost_function,
            nx=nx,
            noise_sigma=noise_sigma,
            num_samples=num_samples,
            horizon=horizon_steps,
            lambda_=lambda_,
            device=self.device,
            u_min=u_min,
            u_max=u_max
        )
        
        # Get topic names
        state_topic = self.get_parameter('topics.input.robot_state').get_parameter_value().string_value
        obstacles_topic = self.get_parameter('topics.input.processed_obstacles').get_parameter_value().string_value
        goal_topic = self.get_parameter('topics.input.goal_pose').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('topics.output.cmd_vel').get_parameter_value().string_value
        optimal_path_topic = self.get_parameter('topics.output.optimal_path').get_parameter_value().string_value
        goal_status_topic = self.get_parameter('topics.output.goal_status').get_parameter_value().string_value
        
        # State variables
        self.current_state = None
        self.goal_pose = None
        self.latest_obstacles = None
        self.current_goal_id = ""
        self.goal_reached_threshold = 0.5  # 50cm
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )
        
        # Subscribers
        self.state_sub = self.create_subscription(
            MPPIState, state_topic, self.state_callback, reliable_qos)
        self.obstacles_sub = self.create_subscription(
            ProcessedObstacles, obstacles_topic, self.obstacles_callback, reliable_qos)
        self.goal_sub = self.create_subscription(
            PoseStamped, goal_topic, self.goal_callback, reliable_qos)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, cmd_vel_topic, reliable_qos)
        self.optimal_path_pub = self.create_publisher(
            OptimalPath, optimal_path_topic, reliable_qos)
        self.goal_status_pub = self.create_publisher(
            ControllerGoalStatus, goal_status_topic, reliable_qos)
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.control_callback)
        
        self.get_logger().info('MPPI Core Node initialized')
    
    def state_callback(self, msg: MPPIState):
        """Receive processed robot state"""
        self.current_state = torch.tensor(msg.state_vector, dtype=torch.float32, device=self.device)
    
    def obstacles_callback(self, msg: ProcessedObstacles):
        """Receive processed obstacle information"""
        self.latest_obstacles = msg
        
        # Update cost function with obstacles using proper method
        if len(msg.ranges) > 0 and self.current_state is not None:
            # Create a mock laser message for update_laser_scan
            class MockLaserMsg:
                def __init__(self, ranges, angles):
                    self.ranges = ranges
                    self.angle_min = angles[0] if len(angles) > 0 else 0.0
                    if len(angles) > 1:
                        self.angle_increment = angles[1] - angles[0]
                    else:
                        self.angle_increment = 0.1  # Default increment
            
            mock_laser = MockLaserMsg(msg.ranges, msg.angles)
            robot_pose = self.current_state.cpu().numpy()
            
            # Use proper update method
            self.cost_function.update_laser_scan(mock_laser, robot_pose)
    
    def goal_callback(self, msg: PoseStamped):
        """Receive goal pose"""
        pose = msg.pose
        x = pose.position.x
        y = pose.position.y
        
        # Convert quaternion to yaw
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        self.goal_pose = [x, y, yaw]
        self.cost_function.set_goal(self.goal_pose)
        
        # Extract goal_id from frame_id if available, otherwise generate one
        if hasattr(msg.header, 'frame_id') and msg.header.frame_id:
            self.current_goal_id = msg.header.frame_id
        else:
            self.current_goal_id = f"goal_{int(time.time())}"
        
        self.get_logger().info(f'New goal received: {self.goal_pose}, ID: {self.current_goal_id}')
    
    def control_callback(self):
        """Main control computation"""
        if self.current_state is None or self.goal_pose is None:
            return
        
        start_time = time.time()
        
        try:
            # Compute MPPI control command
            action = self.mppi.command(self.current_state)
            
            # Debug sampling diversity every 20 calls
            if not hasattr(self, '_control_debug_counter'):
                self._control_debug_counter = 0
            self._control_debug_counter += 1
            
            if self._control_debug_counter % 20 == 0:
                if hasattr(self.mppi, 'actions') and self.mppi.actions is not None:
                    try:
                        # Handle 4D tensor: [rollout, samples, horizon, action_dim]
                        if len(self.mppi.actions.shape) == 4:
                            # Take first rollout, calculate std across samples at time step 0
                            actions_t0 = self.mppi.actions[0, :, 0, :]  # [samples, action_dim]
                            actions_std = torch.std(actions_t0, dim=0)  # [action_dim]
                            linear_std = float(actions_std[0])
                            angular_std = float(actions_std[1])
                        else:
                            # Fallback for other shapes
                            actions_std = torch.std(self.mppi.actions, dim=0)
                            linear_std = float(torch.mean(actions_std))
                            angular_std = linear_std
                            
                        self.get_logger().info(f"[DEBUG] Actions shape: {self.mppi.actions.shape}")
                        self.get_logger().info(f"[DEBUG] Actual samples used: {self.mppi.actions.shape[1] if len(self.mppi.actions.shape) > 1 else 'unknown'}")
                            
                        action_0 = float(action[0])
                        action_1 = float(action[1])
                        
                        if self.motion_model == 'ackermann':
                            self.get_logger().info(f"[MPPI SAMPLING] Action diversity - Velocity std: {linear_std:.3f}, Steering std: {angular_std:.3f}")
                            self.get_logger().info(f"[MPPI OUTPUT] Selected action: Velocity={action_0:.3f} m/s, Steering={action_1:.3f} rad ({action_1*57.3:.1f}°)")
                        else:
                            self.get_logger().info(f"[MPPI SAMPLING] Action diversity - Linear std: {linear_std:.3f}, Angular std: {angular_std:.3f}")
                            self.get_logger().info(f"[MPPI OUTPUT] Selected action: [{action_0:.3f}, {action_1:.3f}]")
                    except Exception as e:
                        self.get_logger().warn(f"[DEBUG] Actions debug failed: {e}")
            
            # Create cmd_vel based on motion model
            cmd_msg = Twist()
            
            if self.motion_model == 'twist':
                # Direct mapping for twist model - NO conversion needed!
                cmd_msg.linear.x = float(action[0])   # vx
                cmd_msg.angular.z = float(action[1])  # wz
            else:
                # Ackermann model conversion (original logic)
                rear_wheel_velocity = float(action[0])
                front_steering_angle = float(action[1])
                
                if abs(rear_wheel_velocity) > 0.01:
                    angular_velocity = (rear_wheel_velocity / self.wheelbase) * torch.tan(action[1])
                else:
                    angular_velocity = 0.0
                    
                cmd_msg.linear.x = rear_wheel_velocity
                cmd_msg.angular.z = float(angular_velocity)
            
            self.cmd_vel_pub.publish(cmd_msg)
            
            # Publish optimal path for visualization (if enabled)
            if self.enable_visualization and self.enable_path_viz:
                self.publish_optimal_path(action)
            
            # Check if goal reached and publish goal status
            goal_distance = torch.norm(self.current_state[:2] - torch.tensor(self.goal_pose[:2], device=self.device))
            goal_distance_float = float(goal_distance)
            
            # Publish goal status
            self.publish_goal_status(goal_distance_float)
            
            if goal_distance < self.goal_reached_threshold:
                self.get_logger().info(f'Goal reached! Distance: {goal_distance_float:.3f}m')
                stop_msg = Twist()
                self.cmd_vel_pub.publish(stop_msg)
            
        except Exception as e:
            self.get_logger().error(f'Control computation failed: {str(e)}')
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
        
        end_time = time.time()
        computation_time = (end_time - start_time) * 1000
        if computation_time > 150:  # Log if too slow
            self.get_logger().warn(f'MPPI computation: {computation_time:.1f}ms')
    
    def publish_goal_status(self, distance_to_goal: float):
        """Publish goal status information"""
        if self.goal_pose is None:
            return
            
        status_msg = ControllerGoalStatus()
        status_msg.header = Header()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.header.frame_id = 'odom'
        
        # Set goal information
        status_msg.goal_id = self.current_goal_id
        status_msg.distance_to_goal = distance_to_goal
        
        # Determine status
        if distance_to_goal <= self.goal_reached_threshold:
            status_msg.goal_reached = True
            status_msg.status_code = 1  # SUCCEEDED
        else:
            status_msg.goal_reached = False
            status_msg.status_code = 0  # PENDING
        
        self.goal_status_pub.publish(status_msg)
    
    def publish_optimal_path(self, action):
        """Publish optimal path for visualization"""
        try:
            # Create optimal trajectory
            optimal_trajectory = self.create_optimal_trajectory(self.current_state, action)
            
            if optimal_trajectory is not None:
                msg = OptimalPath()
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'odom'
                
                # Convert trajectory to points
                path_points = []
                for i in range(optimal_trajectory.shape[0]):
                    point = Point()
                    point.x = float(optimal_trajectory[i, 0])
                    point.y = float(optimal_trajectory[i, 1])
                    point.z = 0.0
                    path_points.append(point)
                
                msg.path_points = path_points
                
                if self.motion_model == 'twist':
                    msg.current_velocity = float(action[0])  # linear velocity
                    msg.current_steering_angle = 0.0  # Not applicable for twist model
                else:
                    msg.current_velocity = float(action[0])  # rear wheel velocity
                    msg.current_steering_angle = float(action[1])  # front steering angle
                
                # Add best paths if enabled
                if self.enable_best_paths:
                    msg.high_cost_paths = self.get_best_paths()
                else:
                    msg.high_cost_paths = []
                
                self.optimal_path_pub.publish(msg)
                
        except Exception as e:
            self.get_logger().debug(f'Optimal path publication failed: {str(e)}')
    
    def get_best_paths(self):
        """Extract best paths (lowest cost) from MPPI for visualization"""
        try:
            if (not hasattr(self.mppi, 'cost_total') or self.mppi.cost_total is None or
                not hasattr(self.mppi, 'states') or self.mppi.states is None):
                return []
            
            # Get cost values and sort to find lowest cost trajectories (best paths)
            costs = self.mppi.cost_total.cpu().numpy()
            sorted_indices = torch.argsort(self.mppi.cost_total, descending=False)
            
            # Get top N lowest cost paths (best trajectories)
            num_paths = min(self.num_best_paths, len(sorted_indices))
            best_paths = []
            
            # Use already computed states from MPPI rollouts
            # self.mppi.states shape: [M, K, T, nx] where M=rollout_samples, K=num_samples, T=horizon, nx=state_dim
            if self.mppi.states is not None and len(self.mppi.states.shape) >= 3:
                # If we have rollout samples (M > 1), take the mean across rollouts
                if len(self.mppi.states.shape) == 4:  # [M, K, T, nx]
                    states = self.mppi.states.mean(dim=0)  # [K, T, nx]
                else:  # [K, T, nx]
                    states = self.mppi.states
                
                for i in range(num_paths):
                    idx = sorted_indices[i].item()
                    cost_value = float(costs[idx])
                    
                    # Get the pre-computed trajectory states
                    trajectory_states = states[idx]  # [T, nx]
                    
                    # Convert to ROS message format
                    path_points = []
                    # Add current state as starting point
                    point = Point()
                    point.x = float(self.current_state[0])
                    point.y = float(self.current_state[1])
                    point.z = 0.0
                    path_points.append(point)
                    
                    # Add trajectory points
                    for t in range(min(trajectory_states.shape[0], 30)):
                        point = Point()
                        point.x = float(trajectory_states[t, 0])
                        point.y = float(trajectory_states[t, 1])
                        point.z = 0.0
                        path_points.append(point)
                    
                    # Create HighCostPath message (reusing message name)
                    best_path_msg = HighCostPath()
                    best_path_msg.path_points = path_points
                    best_path_msg.path_cost = cost_value
                    best_paths.append(best_path_msg)
            
            return best_paths
            
        except Exception as e:
            self.get_logger().debug(f'Failed to get best paths: {str(e)}')
            return []
    
    def create_optimal_trajectory(self, state, action):
        """Create optimal trajectory by rolling out current solution"""
        try:
            if hasattr(self.mppi, 'U'):
                optimal_actions = self.mppi.U
            else:
                optimal_actions = action.unsqueeze(0).repeat(30, 1)
            
            current_state = state.clone()
            trajectory = [current_state.clone()]
            
            for t in range(min(len(optimal_actions), 30)):
                next_state = self.dynamics(current_state.unsqueeze(0), 
                                         optimal_actions[t].unsqueeze(0))
                current_state = next_state.squeeze(0)
                trajectory.append(current_state.clone())
            
            return torch.stack(trajectory)
            
        except Exception as e:
            self.get_logger().debug(f'Failed to create optimal trajectory: {str(e)}')
            return None


def main(args=None):
    rclpy.init(args=args)
    
    node = MPPICoreNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()