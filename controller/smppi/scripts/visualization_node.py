#!/usr/bin/env python3
"""
SMPPI Visualization Node
Handles all RViz visualization without impacting control performance
Subscribes to processed data, publishes markers and visualization
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import time

import numpy as np
from typing import Optional

# ROS2 messages
from geometry_msgs.msg import PoseStamped, Point, Twist
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from smppi.msg import ProcessedObstacles, OptimalPath, MPPIState

# TF2 imports
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# Geometry utils
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from smppi_controller.utils.geometry import GeometryUtils


class VisualizationNode(Node):
    """
    Dedicated visualization node
    Lightweight, non-critical visualization without performance impact
    """
    
    def __init__(self):
        super().__init__('smppi_visualization')
        
        # Initialize parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize TF
        self._init_tf()
        
        # Setup ROS2 interfaces
        self._setup_topics()
        
        # State variables
        self.processed_obstacles: Optional[ProcessedObstacles] = None
        self.optimal_path: Optional[OptimalPath] = None
        self.latest_goal: Optional[PoseStamped] = None
        self.robot_state: Optional[MPPIState] = None
        
        # Visualization timer (runs at lower frequency)
        self.visualization_timer = self.create_timer(
            1.0 / self.visualization_frequency,
            self.visualization_callback
        )
        
        self.get_logger().info(f"SMPPI Visualization Node initialized at {self.visualization_frequency}Hz")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        # Topic parameters
        self.declare_parameter('topics.input.processed_obstacles', '/smppi/processed_obstacles')
        self.declare_parameter('topics.input.optimal_path', '/mppi_optimal_path')
        self.declare_parameter('topics.input.goal_pose', '/goal_pose')
        self.declare_parameter('topics.input.robot_state', '/smppi/robot_state')
        self.declare_parameter('topics.output.markers', '/smppi_visualization')
        
        # Visualization parameters
        self.declare_parameter('visualization_frequency', 20.0)
        self.declare_parameter('enable_obstacles', True)
        self.declare_parameter('enable_trajectory', True)
        self.declare_parameter('enable_goal', True)
        self.declare_parameter('enable_robot_footprint', True)
        
        # Vehicle footprint parameters
        self.declare_parameter('vehicle.footprint', [0.0, 0.0])
        self.declare_parameter('vehicle.footprint_padding', 0.0)
        
        # QoS
        self.declare_parameter('qos.reliable_depth', 5)
        
    
    def _load_parameters(self):
        """Load parameters from ROS2 parameter server"""
        # Topic names
        self.obstacles_topic = self.get_parameter('topics.input.processed_obstacles').get_parameter_value().string_value
        self.path_topic = self.get_parameter('topics.input.optimal_path').get_parameter_value().string_value
        self.goal_topic = self.get_parameter('topics.input.goal_pose').get_parameter_value().string_value
        self.robot_state_topic = self.get_parameter('topics.input.robot_state').get_parameter_value().string_value
        self.markers_topic = self.get_parameter('topics.output.markers').get_parameter_value().string_value
        
        # Visualization parameters
        self.visualization_frequency = self.get_parameter('visualization_frequency').get_parameter_value().double_value
        self.enable_obstacles = self.get_parameter('enable_obstacles').get_parameter_value().bool_value
        self.enable_trajectory = self.get_parameter('enable_trajectory').get_parameter_value().bool_value
        self.enable_goal = self.get_parameter('enable_goal').get_parameter_value().bool_value
        self.enable_robot_footprint = self.get_parameter('enable_robot_footprint').get_parameter_value().bool_value
        
        # Vehicle footprint parameters
        self.footprint = self.get_parameter('vehicle.footprint').get_parameter_value().double_array_value
        self.footprint_padding = self.get_parameter('vehicle.footprint_padding').get_parameter_value().double_value
        
        # QoS parameters
        self.reliable_qos_depth = self.get_parameter('qos.reliable_depth').get_parameter_value().integer_value
    
    def _init_tf(self):
        """Initialize TF listener"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().info("TF listener initialized for visualization")
    
    def _setup_topics(self):
        """Setup ROS2 topics"""
        # QoS profiles
        reliable_qos = QoSProfile(
            depth=self.reliable_qos_depth,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # Subscribers
        self.obstacles_sub = self.create_subscription(
            ProcessedObstacles, self.obstacles_topic, self.obstacles_callback, reliable_qos)
        self.path_sub = self.create_subscription(
            OptimalPath, self.path_topic, self.path_callback, reliable_qos)
        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_topic, self.goal_callback, reliable_qos)
        self.robot_state_sub = self.create_subscription(
            MPPIState, self.robot_state_topic, self.robot_state_callback, reliable_qos)
        
        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, self.markers_topic, reliable_qos)
        
        self.get_logger().info(f"Visualization topics configured")
    
    def obstacles_callback(self, msg: ProcessedObstacles):
        """Receive processed obstacles"""
        self.processed_obstacles = msg
    
    def path_callback(self, msg: OptimalPath):
        """Receive optimal path"""
        self.optimal_path = msg
    
    def goal_callback(self, msg: PoseStamped):
        """Receive goal pose"""
        self.latest_goal = msg
    
    def robot_state_callback(self, msg: MPPIState):
        """Receive robot state"""
        self.robot_state = msg
    
    def visualization_callback(self):
        """Main visualization callback (runs at lower frequency)"""
        try:
            marker_array = MarkerArray()
            
            # Goal marker
            if self.enable_goal and self.latest_goal is not None:
                goal_marker = self.create_goal_marker()
                if goal_marker:
                    marker_array.markers.append(goal_marker)
            
            # Optimal trajectory marker
            if self.enable_trajectory and self.optimal_path is not None:
                traj_marker = self.create_trajectory_marker()
                if traj_marker:
                    marker_array.markers.append(traj_marker)
            
            # Obstacle markers
            if self.enable_obstacles and self.processed_obstacles is not None:
                obstacle_markers = self.create_obstacle_markers()
                marker_array.markers.extend(obstacle_markers)
            
            # Robot footprint marker
            if self.enable_robot_footprint and self.robot_state is not None and len(self.footprint) > 0:
                footprint_marker = self.create_footprint_marker()
                if footprint_marker:
                    marker_array.markers.append(footprint_marker)
            
            # Publish markers
            if len(marker_array.markers) > 0:
                self.marker_pub.publish(marker_array)
                
        except Exception as e:
            self.get_logger().warn(f"Visualization callback error: {str(e)}")
    
    def create_goal_marker(self) -> Optional[Marker]:
        """Create goal visualization marker"""
        if not self.latest_goal:
            return None
            
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
        
        # Arrow appearance
        goal_marker.scale.x = 0.8  # Arrow length
        goal_marker.scale.y = 0.1  # Arrow width
        goal_marker.scale.z = 0.1  # Arrow height
        goal_marker.color.r = 0.0
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 0.8
        
        return goal_marker
    
    def create_trajectory_marker(self) -> Optional[Marker]:
        """Create trajectory visualization marker"""
        if not self.optimal_path or not self.optimal_path.path_points:
            return None
            
        traj_marker = Marker()
        traj_marker.header.frame_id = "odom"
        traj_marker.header.stamp = self.get_clock().now().to_msg()
        traj_marker.ns = "smppi_trajectory"
        traj_marker.id = 1
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD
        
        # Trajectory points
        for pose_stamped in self.optimal_path.path_points:
            point = Point()
            point.x = pose_stamped.pose.position.x
            point.y = pose_stamped.pose.position.y
            point.z = 0.05
            traj_marker.points.append(point)
        
        # Trajectory appearance
        traj_marker.scale.x = 0.05  # Line width
        traj_marker.color.r = 1.0
        traj_marker.color.g = 0.0
        traj_marker.color.b = 0.0
        traj_marker.color.a = 1.0
        
        return traj_marker
    
    def create_obstacle_markers(self) -> list:
        """Create obstacle visualization markers"""
        markers = []
        
        if not hasattr(self.processed_obstacles, 'obstacle_points') or not self.processed_obstacles.obstacle_points:
            return markers
        
        # Create points marker for all obstacles
        obstacle_marker = Marker()
        obstacle_marker.header.frame_id = "odom"
        obstacle_marker.header.stamp = self.get_clock().now().to_msg()
        obstacle_marker.ns = "smppi_obstacles"
        obstacle_marker.id = 10
        obstacle_marker.type = Marker.POINTS
        obstacle_marker.action = Marker.ADD
        
        # Add obstacle points (already in odom frame from sensor processor)
        for point in self.processed_obstacles.obstacle_points:
            p = Point()
            p.x = float(point.x)  # odom frame x
            p.y = float(point.y)  # odom frame y
            p.z = 0.1
            obstacle_marker.points.append(p)
        
        # Obstacle appearance
        obstacle_marker.scale.x = 0.1  # Point width
        obstacle_marker.scale.y = 0.1  # Point height
        obstacle_marker.color.r = 1.0  # Red color
        obstacle_marker.color.g = 0.0
        obstacle_marker.color.b = 0.0
        obstacle_marker.color.a = 0.8
        
        markers.append(obstacle_marker)
        return markers
    
    def create_footprint_marker(self) -> Optional[Marker]:
        """Create robot footprint visualization marker"""
        if not self.robot_state or len(self.footprint) == 0:
            return None
        
        try:
            # Extract robot pose from robot state
            robot_x, robot_y, robot_yaw = self.robot_state.state_vector
            robot_pose = (robot_x, robot_y, robot_yaw)
            
            # Create robot footprint at current pose
            robot_footprint = GeometryUtils.create_robot_footprint_at_pose(
                self.footprint, robot_pose, self.footprint_padding)
            
            # Create line strip marker for footprint boundary
            footprint_marker = Marker()
            footprint_marker.header.frame_id = "odom"
            footprint_marker.header.stamp = self.get_clock().now().to_msg()
            footprint_marker.ns = "robot_footprint"
            footprint_marker.id = 20
            footprint_marker.type = Marker.LINE_STRIP
            footprint_marker.action = Marker.ADD
            
            # Add footprint points to create closed polygon
            for vertex in robot_footprint:
                point = Point()
                point.x = float(vertex[0])
                point.y = float(vertex[1])
                point.z = 0.02  # Slightly above ground
                footprint_marker.points.append(point)
            
            # Close the polygon by adding first point at the end
            if len(footprint_marker.points) > 0:
                footprint_marker.points.append(footprint_marker.points[0])
            
            # Blue color for footprint boundary
            footprint_marker.scale.x = 0.05  # Line width
            footprint_marker.color.r = 0.0
            footprint_marker.color.g = 0.0
            footprint_marker.color.b = 1.0  # Blue
            footprint_marker.color.a = 0.8
            
            return footprint_marker
            
        except Exception as e:
            self.get_logger().warn(f"Footprint marker creation error: {str(e)}")
            return None


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = VisualizationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()