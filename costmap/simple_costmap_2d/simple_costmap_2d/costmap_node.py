# pointcloud2_subscriber.py
from __future__ import annotations
from typing import Optional
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Pose

from .tf_manager import TFManager, quat_to_yaw
from .costmap_2d import Costmap2D


class CostmapNode(Node):
    def __init__(self):
        super().__init__('pointcloud2_subscriber')

        # === Params ===
        self.declare_parameter('point_cloud_topic', '/velodyne_points')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('sensor_frame', 'velodyne')

        # Costmap parameters
        self.declare_parameter('costmap_width', 20.0)
        self.declare_parameter('costmap_height', 20.0)
        self.declare_parameter('costmap_resolution', 0.1)
        self.declare_parameter('min_obstacle_height', 0.1)
        self.declare_parameter('max_obstacle_height', 2.0)
        self.declare_parameter('update_frequency', 10.0)

        # Inflation parameters
        self.declare_parameter('robot_radius', 0.5)
        self.declare_parameter('inflation_radius', 1.0)
        self.declare_parameter('cost_scaling_factor', 10.0)

        pc_topic   = self.get_parameter('point_cloud_topic').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        sensor_fr  = self.get_parameter('sensor_frame').get_parameter_value().string_value

        self.costmap_width = self.get_parameter('costmap_width').get_parameter_value().double_value
        self.costmap_height = self.get_parameter('costmap_height').get_parameter_value().double_value
        self.costmap_resolution = self.get_parameter('costmap_resolution').get_parameter_value().double_value
        self.min_obstacle_height = self.get_parameter('min_obstacle_height').get_parameter_value().double_value
        self.max_obstacle_height = self.get_parameter('max_obstacle_height').get_parameter_value().double_value
        update_freq = self.get_parameter('update_frequency').get_parameter_value().double_value

        self.robot_radius = self.get_parameter('robot_radius').get_parameter_value().double_value
        self.inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        self.cost_scaling_factor = self.get_parameter('cost_scaling_factor').get_parameter_value().double_value

        # TF 매니저
        self.tf_manager = TFManager(self, odom_frame=odom_frame, base_frame=base_frame, sensor_frame=sensor_fr)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # === Subscriptions ===
        self.sub = self.create_subscription(PointCloud2, pc_topic, self.pc_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, qos)

        # === Publishers ===
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/costmap', qos)

        # === Latest PC state (in odom) ===
        self.latest_count = 0
        self.xyz_range = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.last_pc_time = 0.0
        self.latest_points_odom = None  # Store latest points in odom frame

        # === Latest Odom state ===
        self.odom_pose = None   # (x, y, z, yaw)
        self.odom_twist = None  # (vx, vy, vz, wx, wy, wz)
        self.last_odom_time = 0.0

        # === Costmap ===
        # Initialize with robot at center (origin will be updated in rolling window mode)
        origin_x = -self.costmap_width / 2.0
        origin_y = -self.costmap_height / 2.0
        self.costmap = Costmap2D(
            width=self.costmap_width,
            height=self.costmap_height,
            resolution=self.costmap_resolution,
            origin_x=origin_x,
            origin_y=origin_y
        )

        # Main loop timer
        timer_period = 1.0 / update_freq if update_freq > 0 else 0.1
        self.timer = self.create_timer(timer_period, self.main_callback)

        self.get_logger().info(f'Listening PointCloud2 on: {pc_topic}')
        self.get_logger().info(f'Listening Odometry   on: {odom_topic}')

    def pc_callback(self, msg: PointCloud2):
        cloud_in_odom = self.tf_manager.cloud_to_odom(msg)
        if cloud_in_odom is None:
            return

        points_np = self._extract_xyz_from_cloud(cloud_in_odom)
        if points_np is None or len(points_np) == 0:
            return

        self.latest_points_odom = points_np

        min_xyz = points_np.min(axis=0)
        max_xyz = points_np.max(axis=0)
        self.xyz_range = (min_xyz[0], max_xyz[0], min_xyz[1], max_xyz[1], min_xyz[2], max_xyz[2])
        self.latest_count = len(points_np)
        self.last_pc_time = time.time()

    def _extract_xyz_from_cloud(self, cloud: PointCloud2) -> Optional[np.ndarray]:
        """Extract xyz as numpy array (N, 3) from PointCloud2"""
        try:
            # Read points as structured array
            gen = pc2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=True)
            points = np.array(list(gen), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])

            if len(points) == 0:
                return None

            # Extract xyz columns as regular array (N, 3)
            return np.column_stack([points['x'], points['y'], points['z']]).astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Failed to extract xyz from cloud: {e}")
            return None


    # ---------------- Odometry ----------------
    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        t = msg.twist.twist
        self.odom_pose  = (p.x, p.y, p.z, yaw)
        self.odom_twist = (t.linear.x, t.linear.y, t.linear.z,
                           t.angular.x, t.angular.y, t.angular.z)
        self.last_odom_time = time.time()

    # ---------------- Costmap Update ----------------
    def update_costmap(self):
        """Update costmap from latest point cloud data"""
        if self.latest_points_odom is None or self.odom_pose is None:
            return None

        # Reset costmap
        self.costmap.reset(Costmap2D.FREE_SPACE)

        # Update rolling window origin to center on robot
        robot_x, robot_y, _, _ = self.odom_pose
        origin_x = robot_x - self.costmap_width / 2.0
        origin_y = robot_y - self.costmap_height / 2.0
        self.costmap.update_origin(origin_x, origin_y)

        # Mark obstacles in costmap - VECTORIZED
        # latest_points_odom is already numpy array (N, 3) from pc_callback
        points = self.latest_points_odom

        # Height filtering (vectorized)
        z_valid = (points[:, 2] >= self.min_obstacle_height) & (points[:, 2] <= self.max_obstacle_height)
        filtered_points = points[z_valid]  # Shape: (M, 3) where M <= N

        if len(filtered_points) == 0:
            return 0

        # World to map conversion (vectorized)
        mx = ((filtered_points[:, 0] - origin_x) / self.costmap.resolution).astype(np.int32)
        my = ((filtered_points[:, 1] - origin_y) / self.costmap.resolution).astype(np.int32)

        # Bounds checking (vectorized)
        valid_mask = (mx >= 0) & (mx < self.costmap.width_cells) & (my >= 0) & (my < self.costmap.height_cells)
        mx_valid = mx[valid_mask]
        my_valid = my[valid_mask]

        # Update costmap (vectorized)
        self.costmap.data[my_valid, mx_valid] = Costmap2D.OCCUPIED

        obstacle_count = len(mx_valid)

        # Apply inflation layer
        if self.inflation_radius > 0:
            self.costmap.inflate(self.inflation_radius, self.cost_scaling_factor)

        return obstacle_count

    def publish_costmap(self):
        """Publish costmap as OccupancyGrid message"""
        msg = OccupancyGrid()

        # Header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.tf_manager.odom_frame

        # Map metadata
        msg.info.resolution = self.costmap.resolution
        msg.info.width = self.costmap.width_cells
        msg.info.height = self.costmap.height_cells

        # Origin (lower-left corner)
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.costmap.origin_x
        msg.info.origin.position.y = self.costmap.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # Map data (flattened, row-major)
        msg.data = self.costmap.get_data_flat().tolist()

        self.costmap_pub.publish(msg)

    # ---------------- Main Loop ----------------
    def main_callback(self):
        # Update costmap from latest point cloud
        obstacle_count = self.update_costmap()
        
        # Publish costmap
        if obstacle_count is not None:
            self.publish_costmap()

        # Costmap 요약
        if obstacle_count is not None:
            costmap_text = f'COSTMAP obstacles={obstacle_count:,d} cells={self.costmap.width_cells}x{self.costmap.height_cells}'
        else:
            costmap_text = 'COSTMAP: waiting for data'




def main():
    rclpy.init()
    node = CostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
