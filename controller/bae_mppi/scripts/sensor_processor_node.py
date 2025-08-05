#!/usr/bin/env python3
"""
Sensor Processing Node for bae_mppi
Processes raw sensor data (LaserScan, Odometry) and publishes processed information
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import time

# ROS2 messages
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Header

# Custom messages
from bae_mppi.msg import ProcessedObstacles, MPPIState

# TF2
import tf2_ros
from transforms3d.euler import quat2euler


class SensorProcessorNode(Node):
    """Node for processing sensor data for MPPI controller"""
    
    def __init__(self):
        super().__init__('sensor_processor')
        
        # Initialize TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Topic parameters
        self.declare_parameter('topics.input.laser_scan', '/scan')
        self.declare_parameter('topics.input.odometry', '/odom')
        self.declare_parameter('topics.output.processed_obstacles', 'obstacles')
        self.declare_parameter('topics.output.robot_state', 'state')
        
        # Parameters
        self.declare_parameter('laser.min_range', 0.1)
        self.declare_parameter('laser.max_range', 100.0)
        self.declare_parameter('processing_frequency', 20.0)
        
        min_range = self.get_parameter('laser.min_range').get_parameter_value().double_value
        max_range = self.get_parameter('laser.max_range').get_parameter_value().double_value
        processing_freq = self.get_parameter('processing_frequency').get_parameter_value().double_value
        
        self.min_range = min_range
        self.max_range = max_range
        
        # Get topic names
        laser_topic = self.get_parameter('topics.input.laser_scan').get_parameter_value().string_value
        odom_topic = self.get_parameter('topics.input.odometry').get_parameter_value().string_value
        obstacles_topic = self.get_parameter('topics.output.processed_obstacles').get_parameter_value().string_value
        state_topic = self.get_parameter('topics.output.robot_state').get_parameter_value().string_value
        
        # State storage
        self.current_pose = None
        self.current_velocity = None
        self.latest_laser = None
        
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )
        
        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, laser_topic, self.laser_callback, sensor_qos)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, sensor_qos)
        
        # Publishers
        self.obstacles_pub = self.create_publisher(
            ProcessedObstacles, obstacles_topic, reliable_qos)
        self.state_pub = self.create_publisher(
            MPPIState, state_topic, reliable_qos)
        
        # Processing timer
        self.processing_timer = self.create_timer(
            1.0 / processing_freq, self.process_and_publish)
        
        self.get_logger().info('Sensor Processor Node initialized')
    
    def laser_callback(self, msg: LaserScan):
        """Store latest laser scan"""
        self.latest_laser = msg
    
    def odom_callback(self, msg: Odometry):
        """Process odometry and update state"""
        pose = msg.pose.pose
        twist = msg.twist.twist
        
        # Extract pose
        x = pose.position.x
        y = pose.position.y
        
        # Convert quaternion to yaw
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        self.current_pose = [x, y, yaw]
        self.current_velocity = [twist.linear.x, twist.angular.z]
    
    def process_and_publish(self):
        """Main processing loop"""
        if self.current_pose is None or self.latest_laser is None:
            return
        
        try:
            # Process obstacles
            obstacles_msg = self.process_obstacles()
            if obstacles_msg:
                self.obstacles_pub.publish(obstacles_msg)
            
            # Process state
            state_msg = self.process_state()
            if state_msg:
                self.state_pub.publish(state_msg)
                
        except Exception as e:
            self.get_logger().warn(f'Processing failed: {str(e)}')
    
    def process_obstacles(self):
        """Process laser scan into obstacle information"""
        if self.latest_laser is None:
            return None
        
        laser_msg = self.latest_laser
        ranges = np.array(laser_msg.ranges)
        
        # Filter valid ranges
        valid_mask = (ranges >= self.min_range) & (ranges <= self.max_range) & np.isfinite(ranges)
        valid_ranges = ranges[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_ranges) == 0:
            return None
        
        # Convert to angles
        angles = laser_msg.angle_min + valid_indices * laser_msg.angle_increment
        
        # Convert to world frame points
        obstacle_points = []
        if self.current_pose is not None:
            robot_x, robot_y, robot_theta = self.current_pose
            
            for r, a in zip(valid_ranges, angles):
                # Local coordinates
                local_x = r * np.cos(a)
                local_y = r * np.sin(a)
                
                # Transform to world frame
                world_x = robot_x + local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)
                world_y = robot_y + local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)
                
                point = Point()
                point.x = float(world_x)
                point.y = float(world_y)
                point.z = 0.0
                obstacle_points.append(point)
        
        # Create message
        msg = ProcessedObstacles()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.obstacle_points = obstacle_points
        msg.ranges = valid_ranges.tolist()
        msg.angles = angles.tolist()
        msg.min_distance = float(np.min(valid_ranges)) if len(valid_ranges) > 0 else float('inf')
        
        return msg
    
    def process_state(self):
        """Process robot state for MPPI"""
        if self.current_pose is None:
            return None
        
        msg = MPPIState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        
        # Set pose
        msg.pose.position.x = self.current_pose[0]
        msg.pose.position.y = self.current_pose[1]
        msg.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        yaw = self.current_pose[2]
        msg.pose.orientation.z = np.sin(yaw / 2.0)
        msg.pose.orientation.w = np.cos(yaw / 2.0)
        
        # Set velocity
        if self.current_velocity is not None:
            msg.velocity.linear.x = self.current_velocity[0]
            msg.velocity.angular.z = self.current_velocity[1]
        
        # Set state vector for MPPI
        msg.state_vector = self.current_pose
        
        return msg


def main(args=None):
    rclpy.init(args=args)
    
    node = SensorProcessorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()