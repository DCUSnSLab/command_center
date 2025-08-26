#!/usr/bin/env python3
"""
SMPPI Controller Node
Unified ROS2 node combining Nav2 structure with SMPPI enhancements
Processes /scan, /odom, /goal_pose and outputs control commands
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

import torch
import numpy as np
import time
from typing import Optional

# ROS2 messages
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped, Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

# Custom messages
from smppi.msg import ProcessedObstacles, MPPIState, OptimalPath
from command_center_interfaces.msg import ControllerGoalStatus

# SMPPI modules
from smppi_controller.optimizer.smppi_optimizer import SMPPIOptimizer
from smppi_controller.critics.obstacle_critic import ObstacleCritic
from smppi_controller.critics.goal_critic import GoalCritic
from smppi_controller.critics.control_critic import ControlCritic
from smppi_controller.motion_models.ackermann_model import AckermannModel
from smppi_controller.utils.sensor_processor import SensorProcessor
from smppi_controller.utils.transforms import Transforms


class SMPPIControllerNode(Node):
    """
    Unified SMPPI Controller Node
    Nav2-based architecture with SMPPI enhancements
    """
    
    def __init__(self):
        super().__init__('smppi_controller')
        
        # Initialize parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize components
        self._init_sensor_processor()
        self._init_motion_model()
        self._init_optimizer()
        self._init_critics()
        
        # Setup ROS2 interfaces
        self._setup_topics()
        
        # State variables
        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_goal: Optional[PoseStamped] = None
        self.latest_path: Optional[Path] = None
        
        self.robot_state: Optional[torch.Tensor] = None
        self.goal_state: Optional[torch.Tensor] = None
        self.obstacles: Optional[ProcessedObstacles] = None
        
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
        
        self.get_logger().info("SMPPI Controller Node initialized")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters (BAE MPPI style)"""
        # Topic parameters
        self.declare_parameter('topics.input.laser_scan', '/scan')
        self.declare_parameter('topics.input.odometry', '/odom')
        self.declare_parameter('topics.input.goal_pose', '/goal_pose')
        self.declare_parameter('topics.output.cmd_vel', '/ackermann_like_controller/cmd_vel')
        self.declare_parameter('topics.output.optimal_path', '/mppi_optimal_path')
        self.declare_parameter('topics.output.goal_status', '/goal_status')
        
        # Control parameters
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('enable_visualization', True)
        
        # SMPPI optimizer parameters
        self.declare_parameter('optimizer.batch_size', 1000)
        self.declare_parameter('optimizer.time_steps', 30)
        self.declare_parameter('optimizer.model_dt', 0.1)
        self.declare_parameter('optimizer.temperature', 1.0)
        self.declare_parameter('optimizer.iteration_count', 1)
        self.declare_parameter('optimizer.lambda_action', 0.1)
        self.declare_parameter('optimizer.smoothing_factor', 0.8)
        
        # Vehicle parameters
        self.declare_parameter('vehicle.wheelbase', 1.0)
        self.declare_parameter('vehicle.max_linear_velocity', 2.0)
        self.declare_parameter('vehicle.max_angular_velocity', 1.0)
        self.declare_parameter('vehicle.min_linear_velocity', 0.0)
        self.declare_parameter('vehicle.min_angular_velocity', -1.0)
        
        # Critic weights
        self.declare_parameter('costs.obstacle_weight', 100.0)
        self.declare_parameter('costs.goal_weight', 10.0)
        self.declare_parameter('costs.control_weight', 1.0)
        
        # Sensor processing
        self.declare_parameter('sensor.laser_min_range', 0.1)
        self.declare_parameter('sensor.laser_max_range', 5.0)
        self.declare_parameter('sensor.downsample_factor', 1)
        
        # Goal tracking
        self.declare_parameter('goal_reached_threshold', 0.5)
        
        # QoS
        self.declare_parameter('qos.sensor_depth', 1)
        self.declare_parameter('qos.reliable_depth', 5)
    
    def _load_parameters(self):
        """Load parameters from ROS2 parameter server"""
        # Control frequency
        self.control_frequency = self.get_parameter('control_frequency').get_parameter_value().double_value
        self.enable_visualization = self.get_parameter('enable_visualization').get_parameter_value().bool_value
        
        # Topic names
        self.scan_topic = self.get_parameter('topics.input.laser_scan').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('topics.input.odometry').get_parameter_value().string_value
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
        }
        
        # Vehicle parameters
        self.vehicle_params = {
            'wheelbase': self.get_parameter('vehicle.wheelbase').get_parameter_value().double_value,
            'max_steering_angle': np.pi / 4,  # 45 degrees
            'min_turning_radius': 0.5
        }
        
        # Sensor parameters
        self.sensor_params = {
            'laser_min_range': self.get_parameter('sensor.laser_min_range').get_parameter_value().double_value,
            'laser_max_range': self.get_parameter('sensor.laser_max_range').get_parameter_value().double_value,
            'downsample_factor': self.get_parameter('sensor.downsample_factor').get_parameter_value().integer_value,
            'max_obstacles': 1000
        }
        
        # Critic weights
        self.critic_weights = {
            'obstacle_weight': self.get_parameter('costs.obstacle_weight').get_parameter_value().double_value,
            'goal_weight': self.get_parameter('costs.goal_weight').get_parameter_value().double_value,
            'control_weight': self.get_parameter('costs.control_weight').get_parameter_value().double_value,
        }
        
        # QoS parameters
        self.sensor_qos_depth = self.get_parameter('qos.sensor_depth').get_parameter_value().integer_value
        self.reliable_qos_depth = self.get_parameter('qos.reliable_depth').get_parameter_value().integer_value
        
        # Goal tracking parameters
        self.goal_reached_threshold = self.get_parameter('goal_reached_threshold').get_parameter_value().double_value
    
    def _init_sensor_processor(self):
        """Initialize sensor processor"""
        self.sensor_processor = SensorProcessor(self.sensor_params)
        self.get_logger().info("Sensor processor initialized")
    
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
            'max_range': self.sensor_params['laser_max_range']
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
        
        # Control critic
        control_params = {
            'weight': self.critic_weights['control_weight'],
            'linear_cost_weight': 1.0,
            'angular_cost_weight': 1.0,
            'linear_change_weight': 1.0,
            'angular_change_weight': 1.0
        }
        control_critic = ControlCritic(control_params)
        self.optimizer.add_critic(control_critic)
        
        self.get_logger().info("Critics initialized")
    
    def _setup_topics(self):
        """Setup ROS2 topics (BAE MPPI style)"""
        # QoS profiles
        sensor_qos = QoSProfile(
            depth=self.sensor_qos_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        reliable_qos = QoSProfile(
            depth=self.reliable_qos_depth,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, sensor_qos)
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, sensor_qos)
        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_topic, self.goal_callback, reliable_qos)
        
        # Publishers
        self.cmd_pub = self.create_publisher(
            Twist, self.cmd_topic, reliable_qos)
        self.goal_status_pub = self.create_publisher(
            ControllerGoalStatus, self.goal_status_topic, reliable_qos)
        
        if self.enable_visualization:
            self.path_pub = self.create_publisher(
                OptimalPath, self.path_topic, reliable_qos)
            self.marker_pub = self.create_publisher(
                MarkerArray, '/smppi_visualization', reliable_qos)
        
        self.get_logger().info(f"Topics configured: scan={self.scan_topic}, odom={self.odom_topic}")
    
    def scan_callback(self, msg: LaserScan):
        """Process laser scan data"""
        self.latest_scan = msg
        self.obstacles = self.sensor_processor.process_laser_scan(msg)
    
    def odom_callback(self, msg: Odometry):
        """Process odometry data"""
        self.latest_odom = msg
        robot_state_msg = self.sensor_processor.process_odometry(msg)
        self.robot_state = self.sensor_processor.state_to_tensor(robot_state_msg)
    
    def goal_callback(self, msg: PoseStamped):
        """Process goal pose"""
        self.latest_goal = msg
        self.goal_state = Transforms.pose_to_tensor(
            msg, self.optimizer.device, self.optimizer.dtype)
        
        # Set goal ID (from BAE MPPI)
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
            # Prepare optimizer state
            if self.latest_odom:
                # Create PoseStamped from Odometry
                pose_stamped = PoseStamped()
                pose_stamped.header = self.latest_odom.header  
                pose_stamped.pose = self.latest_odom.pose.pose
                
                self.optimizer.prepare(
                    robot_pose=pose_stamped,
                    robot_velocity=self.latest_odom.twist.twist,
                    path=self.latest_path,
                    goal=self.latest_goal
                )
            else:
                return
            
            # Set obstacles
            self.optimizer.set_obstacles(self.obstacles)
            
            # Optimize
            control_sequence = self.optimizer.optimize()
            
            # Get control command
            cmd_vel = self.optimizer.get_control_command()
            
            # Publish control command
            self.cmd_pub.publish(cmd_vel)
            
            # Calculate goal distance and publish status
            if self.robot_state is not None and self.goal_state is not None:
                goal_distance = torch.norm(self.robot_state[:2] - self.goal_state[:2])
                goal_distance_float = float(goal_distance)
                self.publish_goal_status(goal_distance_float)
                
                # Check if goal reached
                if goal_distance_float < self.goal_reached_threshold:
                    self.get_logger().info(f'Goal reached! Distance: {goal_distance_float:.3f}m')
            
            # Shift control sequence for next iteration
            self.optimizer.shift_control_sequence()
            
            # Visualization
            if self.enable_visualization:
                self.publish_visualization()
                self.publish_markers()
            
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
        if self.obstacles is None:
            return False
        # Goal is optional for some applications
        return True
    
    def publish_visualization(self):
        """Publish visualization data"""
        if not self.path_pub:
            return
        
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
                path_msg.total_cost = 0.0  # Set total cost from optimizer if available
                path_msg.costs = []  # Individual costs for each point (optional)
                
                self.path_pub.publish(path_msg)
        
        except Exception as e:
            self.get_logger().warn(f"Visualization error: {str(e)}")
    
    def publish_markers(self):
        """Publish RViz markers for goal and optimal trajectory"""
        try:
            marker_array = MarkerArray()
            
            # Goal marker
            if self.latest_goal is not None:
                goal_marker = Marker()
                goal_marker.header.frame_id = "odom"
                goal_marker.header.stamp = self.get_clock().now().to_msg()
                goal_marker.ns = "smppi_goal"
                goal_marker.id = 0
                goal_marker.type = Marker.ARROW
                goal_marker.action = Marker.ADD
                
                # Goal position and orientation
                goal_marker.pose = self.latest_goal.pose
                goal_marker.pose.position.z = 0.1
                
                # Arrow appearance (length, width, height)
                goal_marker.scale.x = 0.8  # Arrow length
                goal_marker.scale.y = 0.1  # Arrow width
                goal_marker.scale.z = 0.1  # Arrow height
                goal_marker.color.r = 0.0
                goal_marker.color.g = 1.0
                goal_marker.color.b = 0.0
                goal_marker.color.a = 0.8
                
                marker_array.markers.append(goal_marker)
            
            # Optimal trajectory marker
            optimal_trajectory = self.optimizer.getOptimizedTrajectory()
            if optimal_trajectory is not None and optimal_trajectory.shape[0] > 1:
                traj_marker = Marker()
                traj_marker.header.frame_id = "odom"
                traj_marker.header.stamp = self.get_clock().now().to_msg()
                traj_marker.ns = "smppi_trajectory"
                traj_marker.id = 1
                traj_marker.type = Marker.LINE_STRIP
                traj_marker.action = Marker.ADD
                
                # Trajectory points
                for i in range(optimal_trajectory.shape[0]):
                    point = Point()
                    point.x = float(optimal_trajectory[i, 0])
                    point.y = float(optimal_trajectory[i, 1])
                    point.z = 0.05
                    traj_marker.points.append(point)
                
                # Trajectory appearance
                traj_marker.scale.x = 0.05  # Line width
                traj_marker.color.r = 1.0
                traj_marker.color.g = 0.0
                traj_marker.color.b = 0.0
                traj_marker.color.a = 1.0
                
                marker_array.markers.append(traj_marker)
            
            # Publish markers
            if len(marker_array.markers) > 0:
                self.marker_pub.publish(marker_array)
                
        except Exception as e:
            self.get_logger().warn(f"Marker visualization error: {str(e)}")
    
    def publish_goal_status(self, distance_to_goal: float):
        """Publish goal status information (from BAE MPPI)"""
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
        self.get_logger().info("Shutting down SMPPI controller")
        # Stop the robot
        cmd_vel = Twist()
        self.cmd_pub.publish(cmd_vel)


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        # Create node
        controller = SMPPIControllerNode()
        
        # Use multi-threaded executor
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(controller)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            controller.shutdown()
            executor.shutdown()
            controller.destroy_node()
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()