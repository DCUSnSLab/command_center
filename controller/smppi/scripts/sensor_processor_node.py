#!/usr/bin/env python3
"""
Sensor Processing Node for SMPPI
Processes /scan and /odom data, publishes processed obstacles and robot state
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np
from typing import Optional

# TF2 imports
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# ROS2 messages
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from smppi.msg import ProcessedObstacles, MPPIState

from smppi_controller.utils.sensor_processor import SensorProcessor


class SensorProcessorNode(Node):
    """
    Dedicated node for sensor data processing
    Handles TF transformations and data filtering
    """
    
    def __init__(self):
        super().__init__('smppi_sensor_processor')
        
        # Initialize parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize TF
        self._init_tf()
        
        # Initialize sensor processor
        self._init_sensor_processor()
        
        # Setup ROS2 interfaces
        self._setup_topics()
        
        # State variables
        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        
        self.get_logger().info("SMPPI Sensor Processor Node initialized")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        # Topic parameters
        self.declare_parameter('topics.input.laser_scan', '/ptl/scan')
        self.declare_parameter('topics.input.odometry', '/odom')
        self.declare_parameter('topics.output.processed_obstacles', '/smppi/processed_obstacles')
        self.declare_parameter('topics.output.robot_state', '/smppi/robot_state')
        
        # Sensor processing parameters
        self.declare_parameter('sensor.laser_min_range', 0.1)
        self.declare_parameter('sensor.laser_max_range', 5.0)
        self.declare_parameter('sensor.downsample_factor', 1)
        self.declare_parameter('sensor.max_obstacles', 1000)
        
        # Vehicle footprint parameters
        self.declare_parameter('vehicle.footprint', [0.0, 0.0])
        self.declare_parameter('vehicle.footprint_padding', 0.0)
        self.declare_parameter('vehicle.use_footprint_filtering', False)
        
        # TF frames
        self.declare_parameter('frames.target_frame', 'odom')
        self.declare_parameter('frames.laser_frame', 'base_link')
        
        # QoS
        self.declare_parameter('qos.sensor_depth', 1)
        self.declare_parameter('qos.reliable_depth', 5)
    
    def _load_parameters(self):
        """Load parameters from ROS2 parameter server"""
        # Topic names
        self.scan_topic = self.get_parameter('topics.input.laser_scan').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('topics.input.odometry').get_parameter_value().string_value
        self.obstacles_topic = self.get_parameter('topics.output.processed_obstacles').get_parameter_value().string_value
        self.robot_state_topic = self.get_parameter('topics.output.robot_state').get_parameter_value().string_value
        
        # Sensor parameters
        self.sensor_params = {
            'laser_min_range': self.get_parameter('sensor.laser_min_range').get_parameter_value().double_value,
            'laser_max_range': self.get_parameter('sensor.laser_max_range').get_parameter_value().double_value,
            'downsample_factor': self.get_parameter('sensor.downsample_factor').get_parameter_value().integer_value,
            'max_obstacles': self.get_parameter('sensor.max_obstacles').get_parameter_value().integer_value,
            'footprint': self.get_parameter('vehicle.footprint').get_parameter_value().double_array_value,
            'footprint_padding': self.get_parameter('vehicle.footprint_padding').get_parameter_value().double_value,
            'use_footprint_filtering': self.get_parameter('vehicle.use_footprint_filtering').get_parameter_value().bool_value,
        }
        
        # TF frame parameters
        self.target_frame = self.get_parameter('frames.target_frame').get_parameter_value().string_value
        self.laser_frame = self.get_parameter('frames.laser_frame').get_parameter_value().string_value
        
        # QoS parameters
        self.sensor_qos_depth = self.get_parameter('qos.sensor_depth').get_parameter_value().integer_value
        self.reliable_qos_depth = self.get_parameter('qos.reliable_depth').get_parameter_value().integer_value
    
    def _init_tf(self):
        """Initialize TF listener"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().info(f"TF listener initialized: {self.laser_frame} -> {self.target_frame}")
    
    def _init_sensor_processor(self):
        """Initialize sensor processor"""
        self.sensor_processor = SensorProcessor(self.sensor_params)
        # Set TF buffer for coordinate transformations
        self.sensor_processor.set_tf_buffer(self.tf_buffer)
        self.get_logger().info("Sensor processor initialized")
    
    def _setup_topics(self):
        """Setup ROS2 topics"""
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
        
        # Publishers
        self.obstacles_pub = self.create_publisher(
            ProcessedObstacles, self.obstacles_topic, reliable_qos)
        self.robot_state_pub = self.create_publisher(
            MPPIState, self.robot_state_topic, reliable_qos)
        
        self.get_logger().info(f"Topics configured: scan={self.scan_topic}, odom={self.odom_topic}")
    
    def scan_callback(self, msg: LaserScan):
        """Process laser scan data"""
        self.latest_scan = msg
        
        # Process and publish obstacles
        obstacles = self.sensor_processor.process_laser_scan(msg)
        self.obstacles_pub.publish(obstacles)
        
        # Debug log for coordinate verification
        if obstacles and len(obstacles.obstacle_points) > 0:
            first_obs = obstacles.obstacle_points[0]
            self.get_logger().debug(f"First obstacle in odom frame: ({first_obs.x:.2f}, {first_obs.y:.2f})")
    
    def odom_callback(self, msg: Odometry):
        """Process odometry data"""
        self.latest_odom = msg
        
        # Process and publish robot state
        robot_state = self.sensor_processor.process_odometry(msg)
        self.robot_state_pub.publish(robot_state)


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = SensorProcessorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()