#!/usr/bin/env python3
"""
SMPPI Main Controller Node
Core MPPI optimization and control logic
Subscribes to processed sensor data, publishes control commands
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import torch
import numpy as np
import time
from typing import Optional

# ROS2 messages
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from smppi.msg import ProcessedObstacles, MPPIState, OptimalPath
from command_center_interfaces.msg import ControllerGoalStatus, MultipleWaypoints, MPPIParams

# SMPPI modules
from smppi_controller.optimizer.smppi_optimizer import SMPPIOptimizer
from smppi_controller.critics.obstacle_critic import ObstacleCritic
from smppi_controller.critics.goal_critic import GoalCritic
from smppi_controller.motion_models.ackermann_model import AckermannModel
from smppi_controller.utils.transforms import Transforms


class MPPIMainNode(Node):
    """
    Main SMPPI Controller Node
    Pure optimization and control logic
    """
    
    def __init__(self):
        super().__init__('smppi_main_controller')
        
        # Initialize parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize components
        self._init_motion_model()
        self._init_optimizer()
        self._init_critics()
        
        # Setup ROS2 interfaces
        self._setup_topics()
        
        # State variables
        self.processed_obstacles: Optional[ProcessedObstacles] = None
        self.robot_state: Optional[MPPIState] = None
        self.latest_goal: Optional[PoseStamped] = None
        self.latest_path: Optional[Path] = None
        self.multiple_waypoints: Optional[MultipleWaypoints] = None
        
        self.goal_state: Optional[torch.Tensor] = None
        
        # Goal tracking
        self.current_goal_id = ""
        
        # Control loop
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, 
            self.control_callback
        )

        self.obstacle_critic = None
        self.goal_critic = None
        
        # Statistics
        self.last_control_time = time.time()
        self.control_count = 0
        
        self.get_logger().info("SMPPI Main Controller Node initialized")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        # Topic parameters
        self.declare_parameter('topics.input.processed_obstacles', '/smppi/processed_obstacles')
        self.declare_parameter('topics.input.robot_state', '/smppi/robot_state')
        self.declare_parameter('topics.input.goal_pose', '/goal_pose')
        self.declare_parameter('topics.input.multiple_waypoints', '/multiple_waypoints')
        self.declare_parameter('topics.output.cmd_vel', '/ackermann_like_controller/cmd_vel')
        self.declare_parameter('topics.output.optimal_path', '/smppi/optimal_path')
        self.declare_parameter('topics.output.goal_status', '/goal_status')
        
        # Control parameters
        self.declare_parameter('control_frequency', 20.0)
        
        # SMPPI optimizer parameters
        self.declare_parameter('optimizer.batch_size', 3000)
        self.declare_parameter('optimizer.time_steps', 30)
        self.declare_parameter('optimizer.model_dt', 0.1)
        self.declare_parameter('optimizer.temperature', 1.8)
        self.declare_parameter('optimizer.iteration_count', 1)
        self.declare_parameter('optimizer.lambda_action', 0.08)
        self.declare_parameter('optimizer.smoothing_factor', 0.8)
        self.declare_parameter('optimizer.noise_std_u', [0.40, 0.18])
        self.declare_parameter('optimizer.omega_diag', [0.6, 1.2])
        
        # Vehicle parameters
        self.declare_parameter('vehicle.footprint', [0.0, 0.0])
        self.declare_parameter('vehicle.footprint_padding', 0.0)
        self.declare_parameter('vehicle.use_polygon_collision', True)
        self.declare_parameter('vehicle.radius', 0.6)
        
        # Vehicle parameters
        self.declare_parameter('vehicle.wheelbase', 0.65)
        self.declare_parameter('vehicle.max_steering_angle', 0.4)
        self.declare_parameter('vehicle.min_turning_radius', 1.54)
        self.declare_parameter('vehicle.max_linear_velocity', 2.0)
        self.declare_parameter('vehicle.min_linear_velocity', 0.0)
        self.declare_parameter('vehicle.max_angular_velocity', 1.0)
        self.declare_parameter('vehicle.min_angular_velocity', -1.0)
        
        # Critic weights
        self.declare_parameter('costs.obstacle_weight', 100.0)
        self.declare_parameter('costs.goal_weight', 6.0)
        
        # Lookahead parameters
        self.declare_parameter('costs.lookahead.base_distance', 2.5)
        self.declare_parameter('costs.lookahead.velocity_factor', 1.2)
        self.declare_parameter('costs.lookahead.min_distance', 1.0)
        self.declare_parameter('costs.lookahead.max_distance', 6.0)
        
        # Goal tracking
        self.declare_parameter('goal_reached_threshold', 2.0)
        
        # Waypoint mode ('single' or 'multiple')
        self.declare_parameter('waypoint_mode', 'multiple')
        
        # QoS
        self.declare_parameter('qos.reliable_depth', 5)
    
    def _load_parameters(self):
        """Load parameters from ROS2 parameter server"""
        # Control frequency
        self.control_frequency = self.get_parameter('control_frequency').get_parameter_value().double_value
        
        # Topic names
        self.obstacles_topic = self.get_parameter('topics.input.processed_obstacles').get_parameter_value().string_value
        self.robot_state_topic = self.get_parameter('topics.input.robot_state').get_parameter_value().string_value
        self.goal_topic = self.get_parameter('topics.input.goal_pose').get_parameter_value().string_value
        self.multiple_waypoints_topic = self.get_parameter('topics.input.multiple_waypoints').get_parameter_value().string_value
        self.cmd_topic = self.get_parameter('topics.output.cmd_vel').get_parameter_value().string_value
        self.path_topic = self.get_parameter('topics.output.optimal_path').get_parameter_value().string_value
        self.goal_status_topic = self.get_parameter('topics.output.goal_status').get_parameter_value().string_value
        
        # Get max_steering_angle for proper w_min/w_max calculation
        max_steering_angle = self.get_parameter('vehicle.max_steering_angle').get_parameter_value().double_value
        
        # Optimizer parameters
        self.optimizer_params = {
            'batch_size': self.get_parameter('optimizer.batch_size').get_parameter_value().integer_value,
            'time_steps': self.get_parameter('optimizer.time_steps').get_parameter_value().integer_value,
            'model_dt': self.get_parameter('optimizer.model_dt').get_parameter_value().double_value,
            'temperature': self.get_parameter('optimizer.temperature').get_parameter_value().double_value,
            'iteration_count': self.get_parameter('optimizer.iteration_count').get_parameter_value().integer_value,
            'lambda_action': self.get_parameter('optimizer.lambda_action').get_parameter_value().double_value,
            'smoothing_factor': self.get_parameter('optimizer.smoothing_factor').get_parameter_value().double_value,
            'v_min': self.get_parameter('vehicle.min_linear_velocity').get_parameter_value().double_value,
            'v_max': self.get_parameter('vehicle.max_linear_velocity').get_parameter_value().double_value,
            'w_min': -max_steering_angle,  # Use steering angle limits instead of angular velocity limits
            'w_max': max_steering_angle,   # This represents delta (steering angle) limits, not omega limits
            'wheelbase': self.get_parameter('vehicle.wheelbase').get_parameter_value().double_value,
            'noise_std_u': self.get_parameter('optimizer.noise_std_u').get_parameter_value().double_array_value,
            'omega_diag': self.get_parameter('optimizer.omega_diag').get_parameter_value().double_array_value,
        }
        
        # Vehicle parameters
        self.vehicle_params = {
            'wheelbase': self.get_parameter('vehicle.wheelbase').get_parameter_value().double_value,
            'max_steering_angle': self.get_parameter('vehicle.max_steering_angle').get_parameter_value().double_value,
            'min_turning_radius': self.get_parameter('vehicle.min_turning_radius').get_parameter_value().double_value
        }
        
        # Critic weights
        self.critic_weights = {
            'obstacle_weight': self.get_parameter('costs.obstacle_weight').get_parameter_value().double_value,
            'goal_weight': self.get_parameter('costs.goal_weight').get_parameter_value().double_value,
        }
        
        # Lookahead parameters
        self.lookahead_params = {
            'base_distance': self.get_parameter('costs.lookahead.base_distance').get_parameter_value().double_value,
            'velocity_factor': self.get_parameter('costs.lookahead.velocity_factor').get_parameter_value().double_value,
            'min_distance': self.get_parameter('costs.lookahead.min_distance').get_parameter_value().double_value,
            'max_distance': self.get_parameter('costs.lookahead.max_distance').get_parameter_value().double_value,
        }
        
        # QoS parameters
        self.reliable_qos_depth = self.get_parameter('qos.reliable_depth').get_parameter_value().integer_value
        
        # Goal tracking parameters
        self.goal_reached_threshold = self.get_parameter('goal_reached_threshold').get_parameter_value().double_value
        
        # Waypoint mode
        self.waypoint_mode = self.get_parameter('waypoint_mode').get_parameter_value().string_value
        if self.waypoint_mode not in ['single', 'multiple']:
            self.get_logger().warn(f"Invalid waypoint_mode '{self.waypoint_mode}', defaulting to 'multiple'")
            self.waypoint_mode = 'multiple'
    
    def _init_motion_model(self):
        """Initialize motion model"""
        self.motion_model = AckermannModel(self.vehicle_params)
        self.get_logger().info("Ackermann motion model initialized")
    
    def _init_optimizer(self):
        """Initialize SMPPI optimizer"""
        self.optimizer = SMPPIOptimizer(self.optimizer_params)
        self.optimizer.set_motion_model(self.motion_model)
        self.get_logger().info("SMPPI optimizer initialized")
    
    def _init_critics(self):
        """Initialize critic functions"""
        # Obstacle critic
        obstacle_params = {
            'weight': self.critic_weights['obstacle_weight'],
            'safety_radius': 0.5,
            'collision_cost': 1000.0,
            'repulsion_factor': 2.0,
            'vehicle_radius': 0.3,
            'max_range': 5.0
        }
        obstacle_critic = ObstacleCritic(obstacle_params)
        self.optimizer.add_critic(obstacle_critic)
        
        # Goal critic
        goal_params = {
            'weight': self.critic_weights['goal_weight'],
            'xy_goal_tolerance': 0.25,
            'yaw_goal_tolerance': 0.25,
            'distance_scale': 1.0,
            'angle_scale': 1.0,
            # Lookahead parameters
            'lookahead_base_distance': self.lookahead_params['base_distance'],
            'lookahead_velocity_factor': self.lookahead_params['velocity_factor'],
            'lookahead_min_distance': self.lookahead_params['min_distance'],
            'lookahead_max_distance': self.lookahead_params['max_distance'],
            # Debug parameters
            'debug': self.get_parameter('costs.debug').get_parameter_value().bool_value if self.has_parameter('costs.debug') else False,
            'debug_level': self.get_parameter('costs.debug_level').get_parameter_value().integer_value if self.has_parameter('costs.debug_level') else 1,
        }
        self.goal_critic = GoalCritic(goal_params)
        self.optimizer.add_critic(self.goal_critic)
        
        self.get_logger().info("Critics initialized")
    
    
    def _setup_topics(self):
        """Setup ROS2 topics"""
        # QoS profiles
        reliable_qos = QoSProfile(
            depth=self.reliable_qos_depth,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # Subscribers (processed data from sensor node)
        self.obstacles_sub = self.create_subscription(
            ProcessedObstacles, self.obstacles_topic, self.obstacles_callback, reliable_qos)
        self.robot_state_sub = self.create_subscription(
            MPPIState, self.robot_state_topic, self.robot_state_callback, reliable_qos)
        
        # Goal subscribers based on waypoint mode
        if self.waypoint_mode == 'single':
            self.goal_sub = self.create_subscription(
                PoseStamped, self.goal_topic, self.goal_callback, reliable_qos)
            self.get_logger().info(f"Single waypoint mode: subscribed to {self.goal_topic}")
        elif self.waypoint_mode == 'multiple':
            self.multiple_waypoints_sub = self.create_subscription(
                MultipleWaypoints, self.multiple_waypoints_topic, self.multiple_waypoints_callback, reliable_qos)
            self.get_logger().info(f"Multiple waypoints mode: subscribed to {self.multiple_waypoints_topic}")
        else:
            # Fallback - subscribe to both but with priority to multiple waypoints
            self.goal_sub = self.create_subscription(
                PoseStamped, self.goal_topic, self.goal_callback, reliable_qos)
            self.multiple_waypoints_sub = self.create_subscription(
                MultipleWaypoints, self.multiple_waypoints_topic, self.multiple_waypoints_callback, reliable_qos)
            self.get_logger().warn("Unknown waypoint mode, subscribing to both topics")
        
        # Publishers
        self.cmd_pub = self.create_publisher(
            Twist, self.cmd_topic, reliable_qos)
        self.goal_status_pub = self.create_publisher(
            ControllerGoalStatus, self.goal_status_topic, reliable_qos)
        self.path_pub = self.create_publisher(
            OptimalPath, self.path_topic, reliable_qos)
        
        # Parameter update subscriber
        self.params_update_sub = self.create_subscription(
            MPPIParams, '/mppi_update_params', self.params_update_callback, reliable_qos)
        
        # Visualization publishers - publish lookahead point for visualization node
        from geometry_msgs.msg import PoseStamped, PointStamped
        self.lookahead_pub = self.create_publisher(
            PoseStamped, '/smppi_visualization/lookahead_point', reliable_qos)
        self.target_direction_pub = self.create_publisher(
            PointStamped, '/smppi_visualization/target_direction', reliable_qos)
        
        self.get_logger().info(f"Topics configured: obstacles={self.obstacles_topic}, state={self.robot_state_topic}")
    
    def obstacles_callback(self, msg: ProcessedObstacles):
        """Receive processed obstacles from sensor node"""
        self.processed_obstacles = msg
    
    def robot_state_callback(self, msg: MPPIState):
        """Receive robot state from sensor node"""
        self.robot_state = msg
    
    def goal_callback(self, msg: PoseStamped):
        """Process goal pose"""
        self.latest_goal = msg
        self.goal_state = Transforms.pose_to_tensor(
            msg, self.optimizer.device, self.optimizer.dtype)
        
        # Set goal ID
        if msg.header.frame_id:
            self.current_goal_id = msg.header.frame_id
        else:
            self.current_goal_id = f"goal_{int(time.time())}"
            
        self.get_logger().info(f"New goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}), ID: {self.current_goal_id}")
    
    def multiple_waypoints_callback(self, msg: MultipleWaypoints):
        """Process multiple waypoints"""
        self.multiple_waypoints = msg
        
        # Set current goal from multiple waypoints
        self.latest_goal = msg.current_goal
        self.goal_state = Transforms.pose_to_tensor(
            msg.current_goal, self.optimizer.device, self.optimizer.dtype)
        
        # Set goal ID
        if msg.current_goal.header.frame_id:
            self.current_goal_id = msg.current_goal.header.frame_id
        else:
            self.current_goal_id = f"waypoint_{msg.current_waypoint_index}"
        
        # Pass multiple waypoints to goal critic
        goal_critic = None
        if hasattr(self, 'goal_critic') and self.goal_critic is not None:
            goal_critic = self.goal_critic
        elif hasattr(self, 'optimizer') and self.optimizer is not None:
            # Find GoalCritic in optimizer's critics
            for critic in self.optimizer.critics:
                if hasattr(critic, 'set_multiple_waypoints'):
                    goal_critic = critic
                    break
        
        if goal_critic is not None:
            goal_critic.set_multiple_waypoints(msg)
        
        self.get_logger().info(f"New multiple waypoints: current={self.current_goal_id}, "
                             f"next_count={len(msg.next_waypoints)}, final={msg.is_final_waypoint}")
    
    def params_update_callback(self, msg: MPPIParams):
        """Handle dynamic parameter updates"""
        self.get_logger().info("Received MPPI parameter update request")
        
        try:
            # Validate and update optimizer parameters
            if msg.update_optimizer:
                self._update_optimizer_params(msg)
            
            # Validate and update vehicle parameters
            if msg.update_vehicle:
                self._update_vehicle_params(msg)
            
            # Validate and update cost weights
            if msg.update_costs:
                self._update_cost_weights(msg)
            
            # Validate and update lookahead parameters
            if msg.update_lookahead:
                self._update_lookahead_params(msg)
            
            # Update goal critic parameters
            if msg.update_goal_critic:
                self._update_goal_critic_params(msg)
            
            # Update obstacle critic parameters
            if msg.update_obstacle_critic:
                self._update_obstacle_critic_params(msg)
            
            # Update control parameters
            if msg.update_control:
                self._update_control_params(msg)
            
            # Update waypoints settings
            if msg.update_waypoints:
                self._update_waypoints_params(msg)
            
            # Update debug parameters
            if msg.update_debug:
                self._update_debug_params(msg)
                
            self.get_logger().info("MPPI parameters updated successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to update MPPI parameters: {e}")
    
    def control_callback(self):
        """Main control loop callback"""
        if not self.is_ready():
            return
        
        start_time = time.perf_counter()
        
        try:
            # Convert MPPIState to internal format
            robot_pose = PoseStamped()
            robot_pose.header = self.robot_state.header
            robot_pose.pose = self.robot_state.pose
            
            # Prepare optimizer state
            self.optimizer.prepare(
                robot_pose=robot_pose,
                robot_velocity=self.robot_state.velocity,
                path=self.latest_path,
                goal=self.latest_goal
            )
            
            # Set obstacles
            self.optimizer.set_obstacles(self.processed_obstacles)
            
            # Optimize
            control_sequence = self.optimizer.optimize()
            
            # Get control command
            cmd_vel = self.optimizer.get_control_command()
            
            # Apply velocity limits before publishing
            cmd_vel = self._apply_velocity_limits(cmd_vel)
            
            # Publish control command
            self.cmd_pub.publish(cmd_vel)
            
            # Calculate goal distance and publish status
            if self.goal_state is not None:
                robot_pos = torch.tensor([self.robot_state.state_vector[0], self.robot_state.state_vector[1]], 
                                       device=self.optimizer.device, dtype=self.optimizer.dtype)
                goal_distance = torch.norm(robot_pos - self.goal_state[:2])
                goal_distance_float = float(goal_distance)
                self.publish_goal_status(goal_distance_float)
                
                # Check if goal reached
                if goal_distance_float < self.goal_reached_threshold:
                    self.get_logger().info(f'Goal reached! Distance: {goal_distance_float:.3f}m, {self.current_goal_id}')
            
            # Shift control sequence for next iteration
            self.optimizer.shift_control_sequence()
            
            # Publish optimal path for visualization node
            self.publish_optimal_path()
            
            # Publish lookahead point for visualization
            self.publish_lookahead_point()
            
            # Statistics
            end_time = time.perf_counter()
            compute_time = (end_time - start_time) * 1000  # ms
            
            self.control_count += 1
            if self.control_count % 100 == 0:
                self.get_logger().info(f"Control loop: {compute_time:.2f}ms, commands published: {self.control_count}")
                
        except Exception as e:
            self.get_logger().error(f"Control loop error: {str(e)}")
            # Publish zero command in case of error
            cmd_vel = Twist()
            self.cmd_pub.publish(cmd_vel)
    
    def is_ready(self) -> bool:
        """Check if controller is ready to compute commands"""
        if self.robot_state is None:
            return False
        if self.processed_obstacles is None:
            return False
        return True
    
    def publish_optimal_path(self):
        """Publish optimal trajectory for visualization"""
        try:
            # Get optimal trajectory from optimizer
            optimal_trajectory = self.optimizer.getOptimizedTrajectory()
            
            if optimal_trajectory is not None and optimal_trajectory.shape[0] > 0:
                # Create OptimalPath message
                path_msg = OptimalPath()
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = "odom"
                
                # Convert trajectory to PoseStamped points
                pose_points = []
                for i in range(optimal_trajectory.shape[0]):
                    pose_stamped = PoseStamped()
                    pose_stamped.header = path_msg.header
                    pose_stamped.pose.position.x = float(optimal_trajectory[i, 0])
                    pose_stamped.pose.position.y = float(optimal_trajectory[i, 1])
                    pose_stamped.pose.position.z = 0.0
                    
                    # Set orientation from yaw (if available)
                    if optimal_trajectory.shape[1] >= 3:
                        yaw = float(optimal_trajectory[i, 2])
                        pose_stamped.pose.orientation.w = np.cos(yaw / 2.0)
                        pose_stamped.pose.orientation.z = np.sin(yaw / 2.0)
                    else:
                        pose_stamped.pose.orientation.w = 1.0
                    
                    pose_points.append(pose_stamped)
                
                path_msg.path_points = pose_points
                path_msg.total_cost = 0.0
                path_msg.costs = []
                self.path_pub.publish(path_msg)
        
        except Exception as e:
            self.get_logger().warn(f"Path publishing error: {str(e)}")
    
    def publish_lookahead_point(self):
        """Publish lookahead point for visualization node"""
        try:
            # Try self.goal_critic first, then fallback to optimizer's critic
            goal_critic = None
            if hasattr(self, 'goal_critic') and self.goal_critic is not None:
                goal_critic = self.goal_critic
            elif hasattr(self, 'optimizer') and self.optimizer is not None:
                # Find GoalCritic in optimizer's critics
                for critic in self.optimizer.critics:
                    if hasattr(critic, 'get_lookahead_point'):
                        goal_critic = critic
                        break
            
            if goal_critic is not None:
                lookahead_point = goal_critic.get_lookahead_point()
                lookahead_yaw = goal_critic.get_lookahead_yaw()
                target_direction = goal_critic.get_target_direction()
                
                if lookahead_point is not None:
                    from geometry_msgs.msg import PoseStamped
                    import math
                    
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = "odom"
                    
                    pose_msg.pose.position.x = float(lookahead_point[0])
                    pose_msg.pose.position.y = float(lookahead_point[1])
                    pose_msg.pose.position.z = 0.0
                    
                    # Add yaw orientation
                    if lookahead_yaw is not None:
                        yaw = float(lookahead_yaw)
                        pose_msg.pose.orientation.w = math.cos(yaw / 2.0)
                        pose_msg.pose.orientation.z = math.sin(yaw / 2.0)
                    else:
                        pose_msg.pose.orientation.w = 1.0
                    
                    self.lookahead_pub.publish(pose_msg)
                
                # Publish target direction as a point from robot position
                if target_direction is not None and hasattr(self, 'robot_state') and self.robot_state is not None:
                    from geometry_msgs.msg import PointStamped
                    
                    direction_msg = PointStamped()
                    direction_msg.header.stamp = self.get_clock().now().to_msg()
                    direction_msg.header.frame_id = "odom"
                    
                    # Robot current position from MPPIState
                    robot_x = float(self.robot_state.state_vector[0])
                    robot_y = float(self.robot_state.state_vector[1])
                    
                    # Target direction scaled for visualization (2m length)
                    direction_scale = 2.0
                    direction_msg.point.x = robot_x + float(target_direction[0]) * direction_scale
                    direction_msg.point.y = robot_y + float(target_direction[1]) * direction_scale
                    direction_msg.point.z = 0.0
                    
                    self.target_direction_pub.publish(direction_msg)
                    
        except Exception as e:
            self.get_logger().warn(f"Lookahead point publishing error: {str(e)}")
    
    # ===== Dynamic Parameter Update Methods =====
    
    def _validate_positive(self, value: float, name: str, min_val: float = 0.001) -> float:
        """Validate that a parameter is positive"""
        if value <= 0:
            self.get_logger().warn(f"Invalid {name}: {value}, using minimum {min_val}")
            return min_val
        return value
    
    def _validate_range(self, value: float, name: str, min_val: float, max_val: float) -> float:
        """Validate that a parameter is within range"""
        if value < min_val or value > max_val:
            clamped = max(min_val, min(max_val, value))
            self.get_logger().warn(f"Invalid {name}: {value}, clamped to {clamped}")
            return clamped
        return value
    
    def _update_optimizer_params(self, msg: MPPIParams):
        """Update SMPPI optimizer parameters"""
        if msg.batch_size > 0:
            self.optimizer_params['batch_size'] = msg.batch_size
        if msg.time_steps > 0:
            self.optimizer_params['time_steps'] = msg.time_steps
        if msg.model_dt > 0:
            self.optimizer_params['model_dt'] = msg.model_dt
        if msg.temperature > 0:
            self.optimizer_params['temperature'] = msg.temperature
        if msg.lambda_action >= 0:
            self.optimizer_params['lambda_action'] = msg.lambda_action
        if len(msg.noise_std_u) == 2:
            self.optimizer_params['noise_std_u'] = list(msg.noise_std_u)
        if len(msg.omega_diag) == 2:
            self.optimizer_params['omega_diag'] = list(msg.omega_diag)
            
        # Update optimizer with new parameters
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            if hasattr(self.optimizer, 'update_parameters'):
                self.optimizer.update_parameters(self.optimizer_params)
            else:
                self.get_logger().info("Optimizer doesn't support parameter updates - parameters stored for future use")
        
        self.get_logger().info("Optimizer parameters updated")
    
    def _update_vehicle_params(self, msg: MPPIParams):
        """Update vehicle parameters"""
        if msg.wheelbase > 0:
            self.vehicle_params['wheelbase'] = msg.wheelbase
        if msg.max_linear_velocity > 0:
            self.vehicle_params['max_linear_velocity'] = msg.max_linear_velocity
        if msg.min_linear_velocity >= 0:
            self.vehicle_params['min_linear_velocity'] = msg.min_linear_velocity
        if msg.max_angular_velocity > 0:
            self.vehicle_params['max_angular_velocity'] = msg.max_angular_velocity
        if msg.min_angular_velocity < 0:
            self.vehicle_params['min_angular_velocity'] = msg.min_angular_velocity
        if msg.max_steering_angle > 0:
            self.vehicle_params['max_steering_angle'] = msg.max_steering_angle
        if len(msg.footprint) >= 6:  # At least 3 points (6 values)
            self.vehicle_params['footprint'] = list(msg.footprint)
        if msg.footprint_padding >= 0:
            self.vehicle_params['footprint_padding'] = msg.footprint_padding
        if msg.radius > 0:
            self.vehicle_params['radius'] = msg.radius
            
        # Update motion model with new parameters
        if hasattr(self, 'motion_model') and self.motion_model is not None:
            if hasattr(self.motion_model, 'update_parameters'):
                self.motion_model.update_parameters(self.vehicle_params)
            else:
                # For AckermannModel, we need to update internal parameters manually
                if hasattr(self.motion_model, 'max_linear_velocity'):
                    self.motion_model.max_linear_velocity = self.vehicle_params.get('max_linear_velocity', 2.0)
                if hasattr(self.motion_model, 'max_angular_velocity'):
                    self.motion_model.max_angular_velocity = self.vehicle_params.get('max_angular_velocity', 1.16)
                if hasattr(self.motion_model, 'min_angular_velocity'):
                    self.motion_model.min_angular_velocity = self.vehicle_params.get('min_angular_velocity', -1.16)
                if hasattr(self.motion_model, 'wheelbase'):
                    self.motion_model.wheelbase = self.vehicle_params.get('wheelbase', 0.65)
        
        # Also update optimizer with new velocity limits for action sampling
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            if hasattr(self.optimizer, 'update_velocity_limits'):
                self.optimizer.update_velocity_limits(
                    max_v=self.vehicle_params.get('max_linear_velocity', 2.0),
                    max_w=self.vehicle_params.get('max_angular_velocity', 1.16),
                    min_w=self.vehicle_params.get('min_angular_velocity', -1.16)
                )
            elif hasattr(self.optimizer, 'set_action_bounds'):
                # Alternative method name
                self.optimizer.set_action_bounds(
                    v_bounds=[0.0, self.vehicle_params.get('max_linear_velocity', 2.0)],
                    w_bounds=[self.vehicle_params.get('min_angular_velocity', -1.16), 
                             self.vehicle_params.get('max_angular_velocity', 1.16)]
                )
        
        self.get_logger().info(f"Vehicle parameters updated: "
                             f"max_v={self.vehicle_params['max_linear_velocity']:.1f}m/s, "
                             f"max_w={self.vehicle_params['max_angular_velocity']:.2f}rad/s")
    
    def _update_goal_critic_params(self, msg: MPPIParams):
        """Update goal critic parameters"""
        goal_params = {}
        
        if msg.xy_goal_tolerance > 0:
            goal_params['xy_goal_tolerance'] = msg.xy_goal_tolerance
        if msg.yaw_goal_tolerance > 0:
            goal_params['yaw_goal_tolerance'] = msg.yaw_goal_tolerance
        if msg.distance_scale >= 0:
            goal_params['distance_scale'] = msg.distance_scale
        if msg.angle_scale >= 0:
            goal_params['angle_scale'] = msg.angle_scale
        if msg.alignment_scale >= 0:
            goal_params['alignment_scale'] = msg.alignment_scale
        if msg.progress_scale >= 0:
            goal_params['progress_scale'] = msg.progress_scale
            
        goal_params['use_progress_reward'] = msg.use_progress_reward
        goal_params['respect_reverse_heading'] = msg.respect_reverse_heading
        
        if msg.yaw_blend_distance > 0:
            goal_params['yaw_blend_distance'] = msg.yaw_blend_distance
        
        # Add lookahead parameters
        goal_params.update({
            'lookahead_base_distance': self.lookahead_params['base_distance'],
            'lookahead_velocity_factor': self.lookahead_params['velocity_factor'],
            'lookahead_min_distance': self.lookahead_params['min_distance'],
            'lookahead_max_distance': self.lookahead_params['max_distance']
        })
        
        # Find and update goal critic
        goal_critic = None
        if hasattr(self, 'goal_critic') and self.goal_critic is not None:
            goal_critic = self.goal_critic
        elif hasattr(self, 'optimizer') and self.optimizer is not None:
            for critic in self.optimizer.critics:
                if hasattr(critic, 'set_multiple_waypoints'):
                    goal_critic = critic
                    break
        
        if goal_critic is not None:
            goal_critic.update_parameters(goal_params)
            
        self.get_logger().info("Goal critic parameters updated")
    
    def _update_cost_weights(self, msg: MPPIParams):
        """Update cost function weights"""
        if msg.obstacle_weight >= 0:
            self.critic_weights['obstacle_weight'] = msg.obstacle_weight
        if msg.goal_weight >= 0:
            self.critic_weights['goal_weight'] = msg.goal_weight
            
        self.get_logger().info(f"Cost weights updated: obstacle={self.critic_weights['obstacle_weight']}, goal={self.critic_weights['goal_weight']}")
    
    def _update_lookahead_params(self, msg: MPPIParams):
        """Update lookahead parameters"""
        if msg.lookahead_base_distance > 0:
            self.lookahead_params['base_distance'] = msg.lookahead_base_distance
        if msg.lookahead_velocity_factor > 0:
            self.lookahead_params['velocity_factor'] = msg.lookahead_velocity_factor
        if msg.lookahead_min_distance > 0:
            self.lookahead_params['min_distance'] = msg.lookahead_min_distance
        if msg.lookahead_max_distance > msg.lookahead_min_distance:
            self.lookahead_params['max_distance'] = msg.lookahead_max_distance
            
        # Update goal critic with new lookahead parameters
        goal_critic = None
        if hasattr(self, 'goal_critic') and self.goal_critic is not None:
            goal_critic = self.goal_critic
        elif hasattr(self, 'optimizer') and self.optimizer is not None:
            for critic in self.optimizer.critics:
                if hasattr(critic, 'set_multiple_waypoints'):
                    goal_critic = critic
                    break
        
        if goal_critic is not None:
            goal_critic.update_parameters({
                'lookahead_base_distance': self.lookahead_params['base_distance'],
                'lookahead_velocity_factor': self.lookahead_params['velocity_factor'],
                'lookahead_min_distance': self.lookahead_params['min_distance'],
                'lookahead_max_distance': self.lookahead_params['max_distance']
            })
            
        self.get_logger().info(f"Lookahead parameters updated: base={self.lookahead_params['base_distance']}, "
                             f"vel_factor={self.lookahead_params['velocity_factor']}")
    
    def _update_obstacle_critic_params(self, msg: MPPIParams):
        """Update obstacle critic parameters"""
        obstacle_params = {}
        
        if msg.safety_radius > 0:
            obstacle_params['safety_radius'] = msg.safety_radius
        if msg.collision_cost >= 0:
            obstacle_params['collision_cost'] = msg.collision_cost
        if msg.repulsion_factor >= 0:
            obstacle_params['repulsion_factor'] = msg.repulsion_factor
        if msg.vehicle_radius > 0:
            obstacle_params['vehicle_radius'] = msg.vehicle_radius
        if msg.max_range > 0:
            obstacle_params['max_range'] = msg.max_range
        
        # Find and update obstacle critic
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for critic in self.optimizer.critics:
                if hasattr(critic, 'update_parameters') and not hasattr(critic, 'set_multiple_waypoints'):
                    critic.update_parameters(obstacle_params)
                    break
                    
        self.get_logger().info("Obstacle critic parameters updated")
    
    def _update_control_params(self, msg: MPPIParams):
        """Update control parameters"""
        if msg.control_frequency > 0:
            # Update timer frequency if needed
            self.control_frequency = msg.control_frequency
        if msg.goal_reached_threshold > 0:
            self.goal_reached_threshold = msg.goal_reached_threshold
            
        self.get_logger().info("Control parameters updated")
    
    def _update_waypoints_params(self, msg: MPPIParams):
        """Update waypoints parameters"""
        self.use_multiple_waypoints = msg.use_multiple_waypoints
        
        # Update goal critic waypoints setting
        goal_critic = None
        if hasattr(self, 'goal_critic') and self.goal_critic is not None:
            goal_critic = self.goal_critic
        elif hasattr(self, 'optimizer') and self.optimizer is not None:
            for critic in self.optimizer.critics:
                if hasattr(critic, 'set_multiple_waypoints'):
                    goal_critic = critic
                    break
        
        if goal_critic is not None:
            goal_critic.update_parameters({'use_multiple_waypoints': msg.use_multiple_waypoints})
            
        self.get_logger().info("Waypoints parameters updated")
    
    def _update_debug_params(self, msg: MPPIParams):
        """Update debug parameters"""
        debug_params = {
            'debug': msg.debug,
            'debug_level': msg.debug_level
        }
        
        # Update all critics with debug parameters
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for critic in self.optimizer.critics:
                if hasattr(critic, 'update_parameters'):
                    critic.update_parameters(debug_params)
                    
        self.get_logger().info("Debug parameters updated")
    
    def _apply_velocity_limits(self, cmd_vel: Twist) -> Twist:
        """Apply velocity limits to command"""
        # Get current limits from vehicle parameters
        max_v = self.vehicle_params.get('max_linear_velocity', 2.0)
        max_w = self.vehicle_params.get('max_angular_velocity', 1.16)
        min_w = self.vehicle_params.get('min_angular_velocity', -1.16)
        
        # Apply limits
        cmd_vel.linear.x = max(-max_v, min(max_v, cmd_vel.linear.x))
        cmd_vel.angular.z = max(min_w, min(max_w, cmd_vel.angular.z))
        
        return cmd_vel

    def publish_goal_status(self, distance_to_goal: float):
        """Publish goal status information"""
        if self.latest_goal is None:
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

    def update_param(self):
        """
            self.obstaclecritic, self.goalcritic 의 멤버 함수 update_parameters
        """
    
    def shutdown(self):
        """Shutdown controller"""
        self.get_logger().info("Shutting down SMPPI main controller")
        # Stop the robot
        cmd_vel = Twist()
        self.cmd_pub.publish(cmd_vel)


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        controller = MPPIMainNode()
        
        try:
            rclpy.spin(controller)
        except KeyboardInterrupt:
            pass
        finally:
            controller.shutdown()
            controller.destroy_node()
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()