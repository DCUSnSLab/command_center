"""PointCloud processing pipeline."""

import numpy as np
import math
from typing import List, Tuple
from dataclasses import dataclass, field

import rclpy
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import Buffer
from tf2_ros.transform_listener import TransformListener
import sensor_msgs_py.point_cloud2 as pc2

from .costmap_2d import SimpleCostmap2D
from .pointcloud_filters import PointCloudFilters
from .cost_values import LETHAL_OBSTACLE

from rclpy.duration import Duration
import rclpy


    
@dataclass
class PointCloudProcessorConfig:
    """Configuration for PointCloudProcessor."""

    # Range filtering
    max_range: float = 100.0
    min_range: float = 0.1

    # Height filtering
    min_obstacle_height: float = 0.1
    max_obstacle_height: float = 2.0
    ground_height_threshold: float = 0.05

    # Filtering parameters
    voxel_size: float = 0.05
    noise_threshold: float = 0.02
    min_cluster_size: int = 10

    # Statistical filtering
    enable_statistical_filter: bool = True
    statistical_mean_k: int = 50
    statistical_std_dev: float = 1.0

    # Ground removal
    enable_ground_removal: bool = True
    ground_distance_threshold: float = 0.1
    ground_max_iterations: int = 1000

    # Temporal decay
    enable_temporal_decay: bool = True
    decay_rate: float = 0.95

    # TF frames
    sensor_frame: str = "velodyne"
    robot_frame: str = "base_link"
    global_frame: str = "map"

    # Robot footprint (in robot_frame coordinates)
    robot_footprint: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.49, 0.3725), (0.49, -0.3725), (-0.49, -0.3725), (-0.49, 0.3725)
    ])
    footprint_padding: float = 0.15


class PointCloudProcessor:
    """Process point cloud data and update costmap."""

    def __init__(self, costmap: SimpleCostmap2D, tf_buffer: Buffer,
                 config: PointCloudProcessorConfig = None, logger=None):
        """Initialize the processor."""
        self.costmap_ = costmap
        self.tf_buffer_ = tf_buffer
        self.config_ = config if config else PointCloudProcessorConfig()

        self.filtered_cloud_ = np.array([]).reshape(0, 3)
        self.obstacle_points_ = []

        self.last_update_time_ = None
        self.latest_odom_ = None

        # Static transform: sensor_frame -> robot_frame
        self.sensor_to_robot_translation_ = None
        self.sensor_to_robot_rotation_matrix_ = None
        self.static_transform_ready_ = False

        if logger is None:
            self.logger_ = rclpy.logging.get_logger('pointcloud_processor')
        else:
            self.logger_ = logger

        # Get static transform once
        self._initialize_static_transform()

        self.logger_.info("PointCloudProcessor initialized")

    def _debug_time(self, source_frame, target_time):
        # 1) 현재 노드 시각
        now = self.node_.get_clock().now() if hasattr(self, "node_") else rclpy.clock.Clock().now()

        # 2) 최신 TF 시각 (Time(0) → 최신)
        latest_tf_time = None
        latest_tf_err = None
        try:
            zero = Time(seconds=0, nanoseconds=0, clock_type=target_time.clock_type)
            tr_latest = self.tf_buffer_.lookup_transform(
                self.config_.global_frame, source_frame, zero)
            latest_tf_time = rclpy.time.Time.from_msg(tr_latest.header.stamp)
        except Exception as e:
            latest_tf_err = repr(e)

        # 3) use_sim_time (파라미터가 없으면 False로)
        try:
            p = self.node_.get_parameter('use_sim_time')
            use_sim_time = p.get_parameter_value().bool_value if p.type_ != p.Type.NOT_SET else False
        except Exception:
            use_sim_time = False

        # 4) Time(0)으로는 변환 가능한지
        zero = Time(seconds=0, nanoseconds=0, clock_type=target_time.clock_type)
        can_latest = self.tf_buffer_.can_transform(
            self.config_.global_frame, source_frame, zero, timeout=Duration(seconds=0.2))

        # 5) 그냥 출력만 (계산 없음)
        def to_s(t):
            return f"{t.nanoseconds/1e9:.6f}" if t is not None else "None"

        self.logger_.warn(
            f"[TF DEBUG] src={source_frame} → tgt={self.config_.global_frame} | "
            f"use_sim_time={use_sim_time} | "
            f"pc_time={to_s(target_time)} | now={to_s(now)} | "
            f"latest_tf_time={to_s(latest_tf_time)} | "
            f"latest_can={can_latest} | latest_err={latest_tf_err}"
        )

    def _initialize_static_transform(self):
        """Get static transform from sensor to robot frame once at startup."""
        try:
            # Try to get static transform with generous timeout
            transform = self.tf_buffer_.lookup_transform(
                self.config_.robot_frame,
                self.config_.sensor_frame,
                Time(seconds=0),  # Latest available
                timeout=Duration(seconds=5.0))

            # Extract translation
            trans = transform.transform.translation
            self.sensor_to_robot_translation_ = np.array([trans.x, trans.y, trans.z])

            # Extract rotation and convert to matrix
            rot = transform.transform.rotation
            qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w

            # Rotation matrix from quaternion
            self.sensor_to_robot_rotation_matrix_ = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ])

            self.static_transform_ready_ = True
            self.logger_.info(
                f"Static transform {self.config_.sensor_frame} -> {self.config_.robot_frame} loaded: "
                f"translation={self.sensor_to_robot_translation_}")

        except Exception as e:
            self.logger_.error(f"Failed to get static transform: {e}")
            self.logger_.warn("Will retry on first point cloud message")
            self.static_transform_ready_ = False

    def set_odom(self, odom_msg):
        """Update latest odometry data from /odom topic."""
        self.latest_odom_ = odom_msg

    def process_point_cloud(self, msg: PointCloud2,
                           current_time: Time = None) -> bool:
        """
        Process a point cloud message.

        Args:
            msg: PointCloud2 message
            current_time: Current time (optional)

        Returns:
            True if processing successful
        """
        if msg is None or len(msg.data) == 0:
            self.logger_.warning("Empty point cloud message received")
            return False

        target_time = Time.from_msg(
            msg.header.stamp) if current_time is None else current_time

        # Convert to numpy array first
        cloud_array = self._pointcloud2_to_array(msg)

        if len(cloud_array) == 0:
            self.logger_.warning("Empty point cloud after conversion")
            return False

        # Transform point cloud to global frame
        cloud_array = self._transform_points(cloud_array, msg.header.frame_id, target_time)
        if cloud_array is None:
            self.logger_.warning("Failed to transform point cloud")
            return False

        self.logger_.debug(f"Processing point cloud with {len(cloud_array)} points")

        # Apply filtering pipeline
        cloud_array = self._apply_filters(cloud_array)

        if len(cloud_array) == 0:
            self.logger_.debug("All points filtered out")
            return True

        self.logger_.debug(f"After filtering: {len(cloud_array)} points remain")

        # Store filtered cloud
        self.filtered_cloud_ = cloud_array

        # Apply temporal decay if enabled
        if self.config_.enable_temporal_decay:
            self.costmap_.apply_temporal_decay(self.config_.decay_rate)

        # Project points to costmap grid
        self._project_to_grid(cloud_array)

        # Clear robot footprint
        self._clear_robot_footprint()

        # Update obstacle points list
        self._update_obstacle_points(cloud_array)

        self.last_update_time_ = target_time

        return True

    def _transform_points(self, points: np.ndarray, source_frame: str,
                         target_time: Time) -> np.ndarray:
        """Transform points from sensor frame to global frame using static transform + odom."""
        # Retry static transform if not ready
        if not self.static_transform_ready_:
            self._initialize_static_transform()
            if not self.static_transform_ready_:
                self.logger_.warning("Static transform not available yet")
                return None

        # Check if we have odometry data
        if self.latest_odom_ is None:
            self.logger_.warning("No odometry data available yet")
            return None

        try:
            # Step 1: Transform from sensor_frame to robot_frame (static transform)
            # Apply rotation then translation
            points_in_robot = np.dot(points, self.sensor_to_robot_rotation_matrix_.T) + self.sensor_to_robot_translation_

            # Step 2: Transform from robot_frame to global_frame (using /odom)
            # Extract odometry position and orientation
            odom_pos = self.latest_odom_.pose.pose.position
            odom_quat = self.latest_odom_.pose.pose.orientation

            # Convert odom quaternion to rotation matrix
            qx, qy, qz, qw = odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w
            odom_rot_matrix = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ])

            # Apply odom transformation
            odom_translation = np.array([odom_pos.x, odom_pos.y, odom_pos.z])
            points_in_global = np.dot(points_in_robot, odom_rot_matrix.T) + odom_translation

            return points_in_global

        except Exception as e:
            self.logger_.error(f"Transform exception: {e}")
            return None

    def _pointcloud2_to_array(self, cloud_msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 message to numpy array."""
        points_list = []

        for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])

        if points_list:
            return np.array(points_list)
        return np.array([]).reshape(0, 3)

    def _apply_filters(self, cloud: np.ndarray) -> np.ndarray:
        """Apply filtering pipeline to point cloud."""
        # 1. Range filtering
        cloud = PointCloudFilters.filter_by_range(
            cloud, self.config_.max_range, self.config_.min_range)

        if len(cloud) == 0:
            return cloud

        # 2. Height filtering
        cloud = PointCloudFilters.filter_by_height(
            cloud, self.config_.min_obstacle_height,
            self.config_.max_obstacle_height)

        if len(cloud) == 0:
            return cloud

        # 3. Ground plane removal
        if self.config_.enable_ground_removal:
            cloud = PointCloudFilters.remove_ground_plane(
                cloud, self.config_.ground_distance_threshold,
                self.config_.ground_max_iterations)

        if len(cloud) == 0:
            return cloud

        # 4. Voxel grid downsampling
        if self.config_.voxel_size > 0.0:
            cloud = PointCloudFilters.voxel_grid_downsample(
                cloud, self.config_.voxel_size)

        if len(cloud) == 0:
            return cloud

        # 5. Statistical outlier removal
        if self.config_.enable_statistical_filter and \
           len(cloud) >= self.config_.statistical_mean_k:
            cloud = PointCloudFilters.remove_outliers(
                cloud, self.config_.statistical_mean_k,
                self.config_.statistical_std_dev)

        if len(cloud) == 0:
            return cloud

        # 6. Euclidean clustering
        if self.config_.min_cluster_size > 0:
            cloud = PointCloudFilters.euclidean_clustering(
                cloud, self.config_.min_cluster_size)

        # 7. Filter robot footprint (if odometry is available)
        if self.latest_odom_ is not None:
            robot_x = self.latest_odom_.pose.pose.position.x
            robot_y = self.latest_odom_.pose.pose.position.y

            # Convert quaternion to yaw
            q = self.latest_odom_.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                           1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            cloud = PointCloudFilters.filter_robot_footprint(
                cloud, self.config_.robot_footprint, robot_x, robot_y, yaw)

        return cloud

    def _project_to_grid(self, cloud: np.ndarray):
        """Project point cloud to costmap grid."""
        for point in cloud:
            if np.isfinite(point[0]) and np.isfinite(point[1]) and np.isfinite(point[2]):
                self.costmap_.mark_obstacle_with_height(
                    point[0], point[1], point[2], LETHAL_OBSTACLE)

    def _clear_robot_footprint(self):
        """Clear robot footprint from costmap."""
        if self.latest_odom_ is None:
            return

        # Convert footprint to Point list
        footprint_points = []
        for x, y in self.config_.robot_footprint:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            footprint_points.append(p)

        robot_x = self.latest_odom_.pose.pose.position.x
        robot_y = self.latest_odom_.pose.pose.position.y

        # Convert quaternion to yaw
        q = self.latest_odom_.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                       1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        self.costmap_.clear_robot_footprint(footprint_points, robot_x, robot_y, yaw)

    def _get_robot_pose(self, target_time: Time) -> bool:
        """Get robot pose from TF."""
        try:
            transform = self.tf_buffer_.lookup_transform(
                self.config_.global_frame,
                self.config_.robot_frame,
                target_time,
                timeout=Duration(seconds=1.0))

            self.last_robot_pose_ = transform
            self.robot_pose_valid_ = True
            return True

        except Exception as e:
            self.logger_.warn_throttle(
                1000, f"Failed to get robot pose: {e}")
            self.robot_pose_valid_ = False
            return False

    def _update_obstacle_points(self, cloud: np.ndarray):
        """Update obstacle points list."""
        self.obstacle_points_.clear()

        for point in cloud:
            if np.isfinite(point[0]) and np.isfinite(point[1]) and np.isfinite(point[2]):
                obstacle_point = Point()
                obstacle_point.x = float(point[0])
                obstacle_point.y = float(point[1])
                obstacle_point.z = float(point[2])
                self.obstacle_points_.append(obstacle_point)

    def update_config(self, config: PointCloudProcessorConfig):
        """Update configuration."""
        self.config_ = config
        self.logger_.info("PointCloudProcessor configuration updated")

    def get_config(self) -> PointCloudProcessorConfig:
        """Get current configuration."""
        return self.config_

    def get_obstacle_points(self) -> List[Point]:
        """Get obstacle points list."""
        return self.obstacle_points_

    def get_filtered_cloud(self) -> np.ndarray:
        """Get filtered point cloud."""
        return self.filtered_cloud_
