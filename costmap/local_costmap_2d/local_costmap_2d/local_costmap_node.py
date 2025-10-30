"""Local costmap node for PointCloud-based obstacle detection."""

import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener

from .costmap_2d import SimpleCostmap2D
from .pointcloud_processor import PointCloudProcessor, PointCloudProcessorConfig
from .costmap_publisher import CostmapPublisher, CostmapPublisherConfig
from .cost_values import FREE_SPACE


class LocalCostmapNode(Node):
    """ROS2 node for local costmap generation from point clouds."""

    def __init__(self):
        """Initialize the node."""
        super().__init__('local_costmap')

        self.get_logger().info("LocalCostmapNode constructed")

        # Initialize TF
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)

        self.initialized_ = False
        self.robot_x_ = 0.0
        self.robot_y_ = 0.0
        self.has_odom_ = False

        # Declare parameters
        self._declare_parameters()

        # Delay initialization using a timer
        self.init_timer_ = self.create_timer(0.1, self._initialize_delayed)

    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        # Costmap parameters
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('width', 20.0)
        self.declare_parameter('height', 20.0)
        self.declare_parameter('publish_frequency', 10.0)
        self.declare_parameter('update_frequency', 20.0)

        # Topic names
        self.declare_parameter('pointcloud_topic', '/points')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('global_frame', 'odom')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('sensor_frame', 'base_link')

        # Processing parameters
        self.declare_parameter('max_range', 100.0)
        self.declare_parameter('min_range', 0.1)
        self.declare_parameter('min_obstacle_height', 0.1)
        self.declare_parameter('max_obstacle_height', 2.0)
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('enable_ground_removal', True)
        self.declare_parameter('enable_temporal_decay', True)
        self.declare_parameter('decay_rate', 0.95)

        # Robot footprint
        self.declare_parameter('robot_footprint', [
            0.49, 0.3725, 0.49, -0.3725, -0.49, -0.3725, -0.49, 0.3725
        ])

    def _initialize_delayed(self):
        """Delayed initialization to ensure parameters are loaded."""
        if self.initialized_:
            return

        try:
            self._initialize()
            self.initialized_ = True
            self.init_timer_.cancel()
            self.get_logger().info("LocalCostmapNode initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize: {e}")
            # Keep trying with the timer

    def _initialize(self):
        """Initialize the node components."""
        # Load parameters
        self._load_parameters()

        # Initialize costmap
        cells_x = int(self.map_width_ / self.resolution_)
        cells_y = int(self.map_height_ / self.resolution_)

        self.costmap_ = SimpleCostmap2D(
            cells_x, cells_y,
            self.resolution_,
            -self.map_width_ / 2.0,  # Center the map on robot
            -self.map_height_ / 2.0,
            self.default_value_)

        # Initialize processor
        processor_config = self._load_processor_config()
        self.processor_ = PointCloudProcessor(
            self.costmap_, self.tf_buffer_, processor_config, self.get_logger())

        # Initialize publisher
        publisher_config = self._load_publisher_config()
        self.publisher_ = CostmapPublisher(
            self, self.costmap_, publisher_config)

        # Activate publisher immediately
        self.publisher_.on_activate()

        # Create subscriptions
        self.pointcloud_sub_ = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic_,
            self._pointcloud_callback,
            1)

        self.odom_sub_ = self.create_subscription(
            Odometry,
            self.odom_topic_,
            self._odom_callback,
            10)

        # Create timer for publishing
        publish_period = 1.0 / self.publish_frequency_
        self.publish_timer_ = self.create_timer(
            publish_period, self._publish_timer_callback)

    def _load_parameters(self):
        """Load parameters from ROS2 parameter server."""
        self.resolution_ = self.get_parameter('resolution').value
        self.map_width_ = self.get_parameter('width').value
        self.map_height_ = self.get_parameter('height').value
        self.publish_frequency_ = self.get_parameter('publish_frequency').value
        self.update_frequency_ = self.get_parameter('update_frequency').value

        self.pointcloud_topic_ = self.get_parameter('pointcloud_topic').value
        self.odom_topic_ = self.get_parameter('odom_topic').value
        self.global_frame_ = self.get_parameter('global_frame').value
        self.robot_frame_ = self.get_parameter('robot_frame').value
        self.sensor_frame_ = self.get_parameter('sensor_frame').value

        self.default_value_ = FREE_SPACE

        self.get_logger().info(
            f"Loaded parameters: resolution={self.resolution_:.3f}, "
            f"size={self.map_width_:.1f}x{self.map_height_:.1f}, "
            f"freq={self.publish_frequency_:.1f} Hz")

    def _load_processor_config(self) -> PointCloudProcessorConfig:
        """Load processor configuration."""
        config = PointCloudProcessorConfig()

        config.max_range = self.get_parameter('max_range').value
        config.min_range = self.get_parameter('min_range').value
        config.min_obstacle_height = self.get_parameter(
            'min_obstacle_height').value
        config.max_obstacle_height = self.get_parameter(
            'max_obstacle_height').value
        config.voxel_size = self.get_parameter('voxel_size').value
        config.enable_ground_removal = self.get_parameter(
            'enable_ground_removal').value
        config.enable_temporal_decay = self.get_parameter(
            'enable_temporal_decay').value
        config.decay_rate = self.get_parameter('decay_rate').value

        config.sensor_frame = self.sensor_frame_
        config.robot_frame = self.robot_frame_
        config.global_frame = self.global_frame_

        # Load robot footprint
        footprint_param = self.get_parameter('robot_footprint').value
        config.robot_footprint = []
        for i in range(0, len(footprint_param) - 1, 2):
            config.robot_footprint.append(
                (footprint_param[i], footprint_param[i + 1]))

        return config

    def _load_publisher_config(self) -> CostmapPublisherConfig:
        """Load publisher configuration."""
        config = CostmapPublisherConfig()

        config.global_frame = self.global_frame_
        config.publish_frequency = self.publish_frequency_
        config.always_send_full_costmap = False
        config.enable_obstacle_markers = True
        config.enable_filtered_cloud = True

        return config

    def _pointcloud_callback(self, msg: PointCloud2):
        """Process incoming point cloud."""
        if not self.processor_:
            return

        start_time = self.get_clock().now()

        success = self.processor_.process_point_cloud(msg)

        if success:
            processing_time = (self.get_clock().now() - start_time).nanoseconds / 1e6
            num_points = len(self.processor_.get_filtered_cloud())
            self.get_logger().debug(
                f"Processed point cloud in {processing_time:.1f} ms with {num_points} points")
        else:
            self.get_logger().warning("Failed to process point cloud")

    def _odom_callback(self, msg: Odometry):
        """Update robot position from odometry."""
        self.robot_x_ = msg.pose.pose.position.x
        self.robot_y_ = msg.pose.pose.position.y
        self.has_odom_ = True

        # Pass odometry to processor for coordinate transformation
        if self.processor_:
            self.processor_.set_odom(msg)

        self.get_logger().debug(
            f"Updated robot pose: ({self.robot_x_:.2f}, {self.robot_y_:.2f})")

    def _publish_timer_callback(self):
        """Periodic publishing callback."""
        if not self.initialized_ or not self.publisher_ or not self.publisher_.is_active():
            return

        # Update costmap origin based on robot position
        self._update_costmap_origin()

        # Publish costmap
        self.publisher_.publish_costmap()

        # Publish obstacle markers and filtered point cloud
        if self.processor_:
            self.publisher_.publish_obstacle_markers(
                self.processor_.get_obstacle_points())
            self.publisher_.publish_filtered_pointcloud(
                self.processor_.get_filtered_cloud())

    def _update_costmap_origin(self):
        """Update costmap origin to keep robot centered."""
        if not self.costmap_ or not self.has_odom_:
            return

        # Calculate new origin to center the map on robot
        new_origin_x = self.robot_x_ - self.map_width_ / 2.0
        new_origin_y = self.robot_y_ - self.map_height_ / 2.0

        # Get current origin
        current_origin_x = self.costmap_.get_origin_x()
        current_origin_y = self.costmap_.get_origin_y()

        # Update only if robot moved significantly (avoid unnecessary updates)
        distance_moved = math.sqrt(
            (new_origin_x - current_origin_x) ** 2 +
            (new_origin_y - current_origin_y) ** 2)

        if distance_moved > self.resolution_:  # Move threshold: 1 cell
            # Resize costmap with new origin
            cells_x = int(self.map_width_ / self.resolution_)
            cells_y = int(self.map_height_ / self.resolution_)

            self.costmap_.resize_map(
                cells_x, cells_y,
                self.resolution_,
                new_origin_x,
                new_origin_y)

            self.get_logger().debug(
                f"Updated costmap origin to ({new_origin_x:.2f}, {new_origin_y:.2f}) "
                f"for robot at ({self.robot_x_:.2f}, {self.robot_y_:.2f})")


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = LocalCostmapNode()

    node.get_logger().info("Starting LocalCostmapNode")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
