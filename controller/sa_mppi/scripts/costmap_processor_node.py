#!/usr/bin/env python3
"""
Costmap Processing Node for SMPPI
Processes /costmap and /odom data, publishes processed obstacles and robot state
Replaces sensor_processor_node with costmap-based approach
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np
from typing import Optional

# ROS2 messages
from nav_msgs.msg import OccupancyGrid, Odometry
from sa_mppi.msg import ProcessedObstacles, MPPIState

from sa_mppi_controller.utils.costmap_processor import CostmapProcessor


class CostmapProcessorNode(Node):
    """
    Dedicated node for costmap data processing
    Handles costmap-based obstacle detection
    """

    def __init__(self):
        super().__init__('sa_mppi_costmap_processor')

        # Initialize parameters
        self._declare_parameters()
        self._load_parameters()

        # Initialize costmap processor
        self._init_costmap_processor()

        # Setup ROS2 interfaces
        self._setup_topics()

        # State variables
        self.latest_costmap: Optional[OccupancyGrid] = None
        self.latest_odom: Optional[Odometry] = None

        self.get_logger().info("SMPPI Costmap Processor Node initialized")

    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        # Topic parameters
        self.declare_parameter('topics.input.costmap', '/costmap')
        self.declare_parameter('topics.input.odometry', '/odom')
        self.declare_parameter('topics.output.processed_obstacles', '/sa_mppi/processed_obstacles')
        self.declare_parameter('topics.output.robot_state', '/sa_mppi/robot_state')

        # Costmap processing parameters
        self.declare_parameter('costmap.occupied_threshold', 50)
        self.declare_parameter('costmap.max_obstacles', 1000)
        self.declare_parameter('costmap.downsample_factor', 1)

        # QoS
        self.declare_parameter('qos.sensor_depth', 1)
        self.declare_parameter('qos.reliable_depth', 5)

    def _load_parameters(self):
        """Load parameters from ROS2 parameter server"""
        # Topic names
        self.costmap_topic = self.get_parameter('topics.input.costmap').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('topics.input.odometry').get_parameter_value().string_value
        self.obstacles_topic = self.get_parameter('topics.output.processed_obstacles').get_parameter_value().string_value
        self.robot_state_topic = self.get_parameter('topics.output.robot_state').get_parameter_value().string_value

        # Costmap parameters
        self.costmap_params = {
            'occupied_threshold': self.get_parameter('costmap.occupied_threshold').get_parameter_value().integer_value,
            'max_obstacles': self.get_parameter('costmap.max_obstacles').get_parameter_value().integer_value,
            'downsample_factor': self.get_parameter('costmap.downsample_factor').get_parameter_value().integer_value,
        }

        # QoS parameters
        self.sensor_qos_depth = self.get_parameter('qos.sensor_depth').get_parameter_value().integer_value
        self.reliable_qos_depth = self.get_parameter('qos.reliable_depth').get_parameter_value().integer_value

    def _init_costmap_processor(self):
        """Initialize costmap processor"""
        self.costmap_processor = CostmapProcessor(self.costmap_params)
        self.get_logger().info("Costmap processor initialized")

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
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, self.costmap_topic, self.costmap_callback, reliable_qos)
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, sensor_qos)

        # Publishers
        self.obstacles_pub = self.create_publisher(
            ProcessedObstacles, self.obstacles_topic, reliable_qos)
        self.robot_state_pub = self.create_publisher(
            MPPIState, self.robot_state_topic, reliable_qos)

        self.get_logger().info(f"Topics configured: costmap={self.costmap_topic}, odom={self.odom_topic}")

    def costmap_callback(self, msg: OccupancyGrid):
        """Process costmap message and publish processed obstacles"""
        self.latest_costmap = msg

        # Process costmap to extract obstacles
        obstacles = self.costmap_processor.process_costmap(msg)

        # Publish processed obstacles
        self.obstacles_pub.publish(obstacles)

    def odom_callback(self, msg: Odometry):
        """Process odometry message and publish robot state"""
        self.latest_odom = msg

        # Process odometry to extract robot state
        robot_state = self.costmap_processor.process_odometry(msg)

        # Publish robot state
        self.robot_state_pub.publish(robot_state)


def main(args=None):
    """Main function"""
    rclpy.init(args=args)

    try:
        node = CostmapProcessorNode()

        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()

    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
