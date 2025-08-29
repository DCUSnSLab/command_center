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
from command_center_interfaces.msg import ControllerGoalStatus

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
        
        self.goal_state: Optional[torch.Tensor] = None
        
        # Goal tracking
        self.current_goal_id = ""
        
        # Control loop
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, 
            self.control_callback
        )
        
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
        
        # Goal tracking
        self.declare_parameter('goal_reached_threshold', 2.0)
        
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
        self.cmd_topic = self.get_parameter('topics.output.cmd_vel').get_parameter_value().string_value
        self.path_topic = self.get_parameter('topics.output.optimal_path').get_parameter_value().string_value
        self.goal_status_topic = self.get_parameter('topics.output.goal_status').get_parameter_value().string_value
        
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
            'w_min': self.get_parameter('vehicle.min_angular_velocity').get_parameter_value().double_value,
            'w_max': self.get_parameter('vehicle.max_angular_velocity').get_parameter_value().double_value,
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
        
        # QoS parameters
        self.reliable_qos_depth = self.get_parameter('qos.reliable_depth').get_parameter_value().integer_value
        
        # Goal tracking parameters
        self.goal_reached_threshold = self.get_parameter('goal_reached_threshold').get_parameter_value().double_value
    
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
            'angle_scale': 1.0
        }
        goal_critic = GoalCritic(goal_params)
        self.optimizer.add_critic(goal_critic)
        
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
        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_topic, self.goal_callback, reliable_qos)
        
        # Publishers
        self.cmd_pub = self.create_publisher(
            Twist, self.cmd_topic, reliable_qos)
        self.goal_status_pub = self.create_publisher(
            ControllerGoalStatus, self.goal_status_topic, reliable_qos)
        self.path_pub = self.create_publisher(
            OptimalPath, self.path_topic, reliable_qos)
        
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