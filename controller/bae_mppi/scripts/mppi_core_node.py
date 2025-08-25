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
import math

# ROS2 messages
from geometry_msgs.msg import Twist, PoseStamped, Point
from std_msgs.msg import Header

# Custom messages
from bae_mppi.msg import ProcessedObstacles, MPPIState, OptimalPath, HighCostPath
from command_center_interfaces.msg import ControllerGoalStatus

# Local modules
from bae_mppi_core.pytorch_mppi import MPPI, SMPPI
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
        self.declare_parameter('track_width', 0.0)  # Hunter track width parameter
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
        self.declare_parameter('motion_cost.allow_reverse', True)
        self.declare_parameter('motion_cost.reverse_penalty_weight', 10.0)
        self.declare_parameter('motion_cost.min_forward_speed_preference', 0.2)
        self.declare_parameter('motion_cost.reverse_max_speed', -1.0)
        self.declare_parameter('smoothness_cost.curvature_weight', 15.0)
        self.declare_parameter('smoothness_cost.acceleration_weight', 10.0)
        self.declare_parameter('goal_reached_threshold', 0.5)
        
        # Sampling control parameters
        self.declare_parameter('sampling_control.allow_forward_sampling', True)
        self.declare_parameter('sampling_control.allow_reverse_sampling', False)
        
        # SMPPI specific parameters
        self.declare_parameter('use_smppi', True)
        self.declare_parameter('smppi.w_action_seq_cost', 10.0)
        self.declare_parameter('smppi.delta_t', 0.05)
        self.declare_parameter('smppi.use_action_bounds', True)
        
        # Control smoothing parameters
        self.declare_parameter('control_smoothing.filter_window', 5)
        self.declare_parameter('control_smoothing.max_change_rate', 0.8)
        self.declare_parameter('control_smoothing.sg_window', 9)
        self.declare_parameter('control_smoothing.stability_threshold', 1.5)
        self.declare_parameter('control_smoothing.emergency_change_rate', 0.3)
        self.declare_parameter('control_smoothing.min_samples_for_filter', 3)
        self.declare_parameter('control_smoothing.moving_average_weight_start', 0.1)
        self.declare_parameter('control_smoothing.moving_average_weight_end', 0.4)
        
        # Debug parameters
        self.declare_parameter('debug.control_debug_interval', 20)
        self.declare_parameter('debug.velocity_threshold_debug', 1e-3)
        self.declare_parameter('debug.steering_threshold_debug', 1e-3)
        
        # Sensor processing parameters
        self.declare_parameter('sensor_processing.default_angle_increment', 0.1)
        
        # QoS parameters
        self.declare_parameter('qos.sensor_depth', 1)
        self.declare_parameter('qos.reliable_depth', 5)
        
        # Advanced cost function parameters
        self.declare_parameter('obstacle_cost_advanced.default_vehicle_radius', 0.5)
        self.declare_parameter('obstacle_cost_advanced.danger_zone_factor', 1.5)
        self.declare_parameter('obstacle_cost_advanced.laser_min_range_absolute', 0.1)
        self.declare_parameter('obstacle_cost_advanced.debug_counter_interval', 50)
        
        self.declare_parameter('motion_cost_advanced.backward_penalty', 1000.0)
        self.declare_parameter('motion_cost_advanced.fast_reverse_penalty', 500.0)
        self.declare_parameter('motion_cost_advanced.forward_bonus_multiplier', 2.0)
        
        # SMPPI advanced parameters
        self.declare_parameter('smppi_advanced.trajectory_decay_factor', 0.8)
        self.declare_parameter('smppi_advanced.perturbation_smoothing_factor', 0.7)
        self.declare_parameter('smppi_advanced.action_continuity_weight', 0.8)
        
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
        self.track_width = self.get_parameter('track_width').get_parameter_value().double_value
        dt = self.get_parameter('dt').get_parameter_value().double_value
        self.motion_model = self.get_parameter('motion_model').get_parameter_value().string_value
        
        # SMPPI parameters
        self.use_smppi = self.get_parameter('use_smppi').get_parameter_value().bool_value
        w_action_seq_cost = self.get_parameter('smppi.w_action_seq_cost').get_parameter_value().double_value
        smppi_delta_t = self.get_parameter('smppi.delta_t').get_parameter_value().double_value
        use_action_bounds = self.get_parameter('smppi.use_action_bounds').get_parameter_value().bool_value
        
        # Setup device
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')
        
        # Time synchronization check for SMPPI
        control_period = 1.0 / self.control_frequency
        if self.use_smppi:
            # For SMPPI, use smppi.delta_t parameter
            model_dt = smppi_delta_t
            if abs(control_period - model_dt) > 0.001:  # 1ms tolerance
                self.get_logger().warn(
                    f'Control period ({control_period:.3f}s) != SMPPI delta_t ({model_dt:.3f}s). '
                    f'This may cause temporal desynchronization. Consider setting them equal.'
                )
        else:
            model_dt = dt
        
        # Initialize MPPI dynamics based on motion model
        if self.motion_model == 'twist':
            self.dynamics = TwistDynamics(dt=model_dt, device=self.device)
            self.get_logger().info(f'Using Twist dynamics model with dt={model_dt:.3f}s')
        else:  # 'ackermann'
            self.dynamics = AckermannDynamics(wheelbase=wheelbase, dt=model_dt, device=self.device)
            self.get_logger().info(f'Using Ackermann dynamics model with dt={model_dt:.3f}s')
        
        self.wheelbase = wheelbase
        
        # Cost function with parameters
        obstacle_params = {
            'safety_radius': self.get_parameter('obstacle_cost.safety_radius').get_parameter_value().double_value,
            'max_range': self.get_parameter('obstacle_cost.max_range').get_parameter_value().double_value,
            'penalty_weight': self.get_parameter('obstacle_cost.penalty_weight').get_parameter_value().double_value,
            'exponential_factor': self.get_parameter('obstacle_cost.exponential_factor').get_parameter_value().double_value,
        }
        motion_params = {
            'allow_reverse': self.get_parameter('motion_cost.allow_reverse').get_parameter_value().bool_value,
            'reverse_penalty_weight': self.get_parameter('motion_cost.reverse_penalty_weight').get_parameter_value().double_value,
            'min_forward_speed_preference': self.get_parameter('motion_cost.min_forward_speed_preference').get_parameter_value().double_value,
            'reverse_max_speed': self.get_parameter('motion_cost.reverse_max_speed').get_parameter_value().double_value,
        }
        smoothness_params = {
            'curvature_weight': self.get_parameter('smoothness_cost.curvature_weight').get_parameter_value().double_value,
            'acceleration_weight': self.get_parameter('smoothness_cost.acceleration_weight').get_parameter_value().double_value,
        }
        
        # Advanced parameters for cost functions
        obstacle_advanced_params = {
            'default_vehicle_radius': self.get_parameter('obstacle_cost_advanced.default_vehicle_radius').get_parameter_value().double_value,
            'danger_zone_factor': self.get_parameter('obstacle_cost_advanced.danger_zone_factor').get_parameter_value().double_value,
            'laser_min_range_absolute': self.get_parameter('obstacle_cost_advanced.laser_min_range_absolute').get_parameter_value().double_value,
            'debug_counter_interval': self.get_parameter('obstacle_cost_advanced.debug_counter_interval').get_parameter_value().integer_value,
        }
        motion_advanced_params = {
            'backward_penalty': self.get_parameter('motion_cost_advanced.backward_penalty').get_parameter_value().double_value,
            'fast_reverse_penalty': self.get_parameter('motion_cost_advanced.fast_reverse_penalty').get_parameter_value().double_value,
            'forward_bonus_multiplier': self.get_parameter('motion_cost_advanced.forward_bonus_multiplier').get_parameter_value().double_value,
        }
        
        self.cost_function = CombinedCostFunction(
            device=self.device, 
            obstacle_params=obstacle_params, 
            motion_params=motion_params, 
            smoothness_params=smoothness_params,
            obstacle_advanced_params=obstacle_advanced_params,
            motion_advanced_params=motion_advanced_params
        )
        
        # Set goal cost parameters
        self.cost_function.goal_cost.goal_weight = self.get_parameter('goal_cost.goal_weight').get_parameter_value().double_value
        self.cost_function.goal_cost.angle_weight = self.get_parameter('goal_cost.angle_weight').get_parameter_value().double_value
        
        # Get sampling control parameters
        allow_forward_sampling = self.get_parameter('sampling_control.allow_forward_sampling').get_parameter_value().bool_value
        allow_reverse_sampling = self.get_parameter('sampling_control.allow_reverse_sampling').get_parameter_value().bool_value
        
        # Control bounds based on motion model and sampling preferences
        nx = 3  # [x, y, theta]
        nu = 2  # [v, w] or [v, delta]
        
        # Determine velocity bounds based on sampling preferences
        if allow_forward_sampling and allow_reverse_sampling:
            min_linear_vel = -max_linear_vel
        elif allow_forward_sampling and not allow_reverse_sampling:
            min_linear_vel = 0.0  # Only forward motion
        elif not allow_forward_sampling and allow_reverse_sampling:
            min_linear_vel = -max_linear_vel
            max_linear_vel = 0.0  # Only reverse motion
        else:
            # Neither allowed - default to stationary
            min_linear_vel = 0.0
            max_linear_vel = 0.0
        
        if self.motion_model == 'twist':
            # Twist model: [vx, wz]
            u_min = torch.tensor([min_linear_vel, -max_angular_vel], device=self.device)
            u_max = torch.tensor([max_linear_vel, max_angular_vel], device=self.device)
        else:
            # Ackermann model: [v_rear, delta]
            u_min = torch.tensor([min_linear_vel, -max_steering_angle], device=self.device)
            u_max = torch.tensor([max_linear_vel, max_steering_angle], device=self.device)
        
        # Initialize MPPI
        noise_sigma = torch.diag(torch.tensor(sigma, device=self.device))
        
        # Debug MPPI parameters
        self.get_logger().info(f"[MPPI INIT] Samples: {num_samples}, Horizon: {horizon_steps}")
        self.get_logger().info(f"[MPPI INIT] Sigma: {sigma}, Lambda: {lambda_}")
        self.get_logger().info(f"[SAMPLING CONTROL] Forward: {allow_forward_sampling}, Reverse: {allow_reverse_sampling}")
        
        u_min_vals = [float(u_min[0]), float(u_min[1])]
        u_max_vals = [float(u_max[0]), float(u_max[1])]
        
        if self.motion_model == 'ackermann':
            self.get_logger().info(f"[MPPI INIT] Control bounds - Velocity: [{u_min_vals[0]:.2f}, {u_max_vals[0]:.2f}] m/s, Steering: [{u_min_vals[1]:.2f}, {u_max_vals[1]:.2f}] rad")
            self.get_logger().info(f"[MPPI INIT] Wheelbase: {wheelbase:.2f}m, Max steering: {max_steering_angle:.3f} rad ({max_steering_angle*57.3:.1f}°)")
        else:
            self.get_logger().info(f"[MPPI INIT] Control bounds - Linear: [{u_min_vals[0]:.2f}, {u_max_vals[0]:.2f}], Angular: [{u_min_vals[1]:.2f}, {u_max_vals[1]:.2f}]")
            
        sigma_diag = [float(noise_sigma[0,0]), float(noise_sigma[1,1])]
        self.get_logger().info(f"[MPPI INIT] Noise sigma diagonal: [{sigma_diag[0]:.1f}, {sigma_diag[1]:.1f}]")
        
        # Initialize MPPI or SMPPI based on parameter
        if self.use_smppi:
            self.get_logger().info(f"[SMPPI INIT] Using Smooth MPPI with w_action_seq_cost={w_action_seq_cost:.1f}, delta_t={smppi_delta_t:.3f}s")
            
            # Set action bounds for SMPPI if enabled
            action_min = u_min if use_action_bounds else None
            action_max = u_max if use_action_bounds else None
            
            # Get SMPPI advanced parameters
            trajectory_decay_factor = self.get_parameter('smppi_advanced.trajectory_decay_factor').get_parameter_value().double_value
            perturbation_smoothing_factor = self.get_parameter('smppi_advanced.perturbation_smoothing_factor').get_parameter_value().double_value
            action_continuity_weight = self.get_parameter('smppi_advanced.action_continuity_weight').get_parameter_value().double_value
            
            self.mppi = SMPPI(
                dynamics=self.dynamics,
                running_cost=self.cost_function,
                nx=nx,
                noise_sigma=noise_sigma,
                num_samples=num_samples,
                horizon=horizon_steps,
                lambda_=lambda_,
                device=self.device,
                u_min=u_min,
                u_max=u_max,
                # SMPPI specific parameters
                w_action_seq_cost=w_action_seq_cost,
                delta_t=smppi_delta_t,
                action_min=action_min,
                action_max=action_max,
                # SMPPI advanced parameters
                trajectory_decay_factor=trajectory_decay_factor,
                perturbation_smoothing_factor=perturbation_smoothing_factor,
                action_continuity_weight=action_continuity_weight
            )
        else:
            self.get_logger().info("[MPPI INIT] Using standard MPPI")
            self.mppi = MPPI(
                dynamics=self.dynamics,
                running_cost=self.cost_function,
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
        self.goal_reached_threshold = self.get_parameter('goal_reached_threshold').get_parameter_value().double_value
        
        # Control smoothing variables - get from parameters
        self.control_history = []
        self.filter_window = self.get_parameter('control_smoothing.filter_window').get_parameter_value().integer_value
        self.last_published_action = None
        self.max_change_rate = self.get_parameter('control_smoothing.max_change_rate').get_parameter_value().double_value
        
        # Savitzky-Golay filter (Phase 3)
        self.sg_history = []  # For Savitzky-Golay filter
        self.sg_window = self.get_parameter('control_smoothing.sg_window').get_parameter_value().integer_value
        self.use_savitzky_golay = True
        
        # Additional control smoothing parameters
        self.stability_threshold = self.get_parameter('control_smoothing.stability_threshold').get_parameter_value().double_value
        self.emergency_change_rate = self.get_parameter('control_smoothing.emergency_change_rate').get_parameter_value().double_value
        self.min_samples_for_filter = self.get_parameter('control_smoothing.min_samples_for_filter').get_parameter_value().integer_value
        self.moving_average_weight_start = self.get_parameter('control_smoothing.moving_average_weight_start').get_parameter_value().double_value
        self.moving_average_weight_end = self.get_parameter('control_smoothing.moving_average_weight_end').get_parameter_value().double_value
        
        # Debug parameters
        self.control_debug_interval = self.get_parameter('debug.control_debug_interval').get_parameter_value().integer_value
        self.velocity_threshold_debug = self.get_parameter('debug.velocity_threshold_debug').get_parameter_value().double_value
        
        # Sensor processing parameters
        self.default_angle_increment = self.get_parameter('sensor_processing.default_angle_increment').get_parameter_value().double_value
        
        # QoS parameters for later use
        self.sensor_qos_depth = self.get_parameter('qos.sensor_depth').get_parameter_value().integer_value
        self.reliable_qos_depth = self.get_parameter('qos.reliable_depth').get_parameter_value().integer_value
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=self.reliable_qos_depth
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
    
    def apply_control_smoothing(self, raw_action):
        """Apply moving average filter to control commands"""
        # Convert to tensor if needed
        if not isinstance(raw_action, torch.Tensor):
            raw_action = torch.tensor(raw_action, device=self.device)
        
        # Add to history
        self.control_history.append(raw_action.clone())
        if len(self.control_history) > self.filter_window:
            self.control_history.pop(0)
        
        # Apply weighted moving average (more weight to recent values)
        if len(self.control_history) >= self.min_samples_for_filter:  # Need at least min samples
            weights = torch.linspace(self.moving_average_weight_start, self.moving_average_weight_end, len(self.control_history), device=self.device)
            weights = weights / weights.sum()  # Normalize
            
            smoothed = torch.zeros_like(raw_action)
            for i, (weight, control) in enumerate(zip(weights, self.control_history)):
                smoothed += weight * control
            
            return smoothed
        else:
            # Not enough history, return original
            return raw_action
    
    def apply_rate_limiter(self, action, last_action, max_change_rate=None):
        """Limit the rate of change in control commands"""
        if max_change_rate is None:
            max_change_rate = self.max_change_rate
            
        if last_action is not None:
            change = action - last_action
            change_magnitude = torch.norm(change)
            
            if change_magnitude > max_change_rate:
                # Limit the change rate
                limited_change = change * (max_change_rate / change_magnitude)
                action = last_action + limited_change
                
                # Log high change rates
                self.get_logger().debug(f"Rate limited: {change_magnitude:.3f} -> {max_change_rate:.3f}")
        
        return action
    
    def monitor_control_stability(self, current_action, last_action):
        """Monitor control stability and apply emergency smoothing if needed"""
        if last_action is not None:
            change_rate = torch.norm(current_action - last_action)
            if change_rate > self.stability_threshold:
                self.get_logger().warn(f"High control change rate detected: {change_rate:.3f}")
                # Apply emergency smoothing (stronger rate limiting)
                return self.apply_rate_limiter(current_action, last_action, max_change_rate=self.emergency_change_rate)
        
        return current_action
    
    def apply_savitzky_golay_filter(self, raw_action):
        """Apply Savitzky-Golay filter (Nav2 style)"""
        # Convert to tensor if needed
        if not isinstance(raw_action, torch.Tensor):
            raw_action = torch.tensor(raw_action, device=self.device)
        
        # Add to Savitzky-Golay history
        self.sg_history.append(raw_action.clone())
        if len(self.sg_history) > self.sg_window:
            self.sg_history.pop(0)
        
        # Need at least 9 points for the filter
        if len(self.sg_history) < self.sg_window:
            return raw_action
        
        # Savitzky-Golay coefficients for 9-point quadratic filter (Nav2 style)
        coeffs = torch.tensor([-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0], 
                             device=self.device) / 231.0
        
        # Apply filter to each dimension
        filtered_action = torch.zeros_like(raw_action)
        
        for dim in range(raw_action.shape[0]):  # For each control dimension
            # Extract values for this dimension
            values = torch.stack([hist[dim] for hist in self.sg_history])
            
            # Apply convolution
            filtered_value = torch.sum(values * coeffs)
            filtered_action[dim] = filtered_value
        
        return filtered_action
    
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
                        self.angle_increment = self.default_angle_increment  # Default increment
            
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
        # self.get_logger().info(f'control callback called')
        """Main control computation"""
        if self.current_state is None or self.goal_pose is None:
            return
        start_time = time.time()
        
        try:
            # Compute MPPI control command
            raw_action = self.mppi.command(self.current_state)
            
            # Apply control smoothing filters
            # Phase 3: Savitzky-Golay filter (highest priority)
            if self.use_savitzky_golay:
                sg_filtered_action = self.apply_savitzky_golay_filter(raw_action)
            else:
                sg_filtered_action = raw_action
            
            # Phase 2: Moving average filter (backup/secondary)
            smoothed_action = self.apply_control_smoothing(sg_filtered_action)
            
            # Phase 2: Rate limiter
            limited_action = self.apply_rate_limiter(smoothed_action, self.last_published_action)
            
            # Phase 2: Stability monitoring
            action = self.monitor_control_stability(limited_action, self.last_published_action)
            
            # Store for next iteration
            self.last_published_action = action.clone() if isinstance(action, torch.Tensor) else torch.tensor(action, device=self.device)

            # Debug sampling diversity every 20 calls
            if not hasattr(self, '_control_debug_counter'):
                self._control_debug_counter = 0
            self._control_debug_counter += 1
            
            if self._control_debug_counter % self.control_debug_interval == 0:
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
                # 이미 Twist 형태면 그대로 사용
                cmd_msg.linear.x = float(action[0])   # vx
                cmd_msg.angular.z = float(action[1])  # wz
            else:
                # Ackermann → Twist (표준 공식)
                v     = float(action[0])   # 차체 전진속도 [m/s]
                delta = float(action[1])   # 전륜 조향각 [rad]

                # 필요 시 데드밴드(선택)
                if abs(v) < self.velocity_threshold_debug:
                    omega = 0.0
                else:
                    omega = (v / self.wheelbase) * math.tan(delta)

                cmd_msg.linear.x  = v
                cmd_msg.angular.z = omega
            
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
        # if computation_time > 150:  # Log if too slow
        #     self.get_logger().warn(f'MPPI computation: {computation_time:.1f}ms')
    
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