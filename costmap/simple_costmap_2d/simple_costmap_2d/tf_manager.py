from __future__ import annotations
from typing import Optional

from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2

from .utils import quat_to_yaw, _transform_xyz_intensity

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

class TFManager:
    def __init__(self, node: Node, odom_frame: str, base_frame: str, sensor_frame: str):
        self.node = node
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.sensor_frame = sensor_frame

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)

        self._static_base_to_sensor: Optional[TransformStamped] = None
        self._reported_static_once = False

        self.node.get_logger().info(
            f"[TFManager] target={self.odom_frame}, base={self.base_frame}, sensor={self.sensor_frame}"
        )

    # ---------- internal ----------
    def _lookup(self, target: str, source: str, when: Time) -> Optional[TransformStamped]:
        try:
            return self.buffer.lookup_transform(target, source, when)
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def cache_base_to_sensor_once(self) -> None:
        """base_link -> velodyne 정적 TF 1회 캐시 (성공 전까지는 호출시마다 시도)"""
        if self._static_base_to_sensor is not None:
            return

        tf = self._lookup(self.sensor_frame, self.base_frame, Time())
        if tf:
            self._static_base_to_sensor = tf
            if not self._reported_static_once:
                self.node.get_logger().info(
                    "[TF cached once] lookup result "
                    f"target={tf.header.frame_id}  source={tf.child_frame_id} | "
                )
                self._reported_static_once = True

    # ---------- public ----------
    @staticmethod
    def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        return quat_to_yaw(x, y, z, w)

    def cloud_to_odom(self, cloud: PointCloud2) -> Optional[PointCloud2]:
        self.cache_base_to_sensor_once()

        src_frame = (cloud.header.frame_id or self.sensor_frame).lstrip('/')

        if not self.buffer.can_transform(self.odom_frame, src_frame, Time()):
            self.node.get_logger().warn(
                f"[TFManager] no transform {self.odom_frame} <- {src_frame}",
                throttle_duration_sec=1.0
            )
            return None

        try:
            tf = self.buffer.lookup_transform(self.odom_frame, src_frame, Time())
        except Exception as e:
            self.node.get_logger().warn(f"[TFManager] TF lookup failed: {e}", throttle_duration_sec=1.0)
            return None

        # Use optimized transform directly
        try:
            return _transform_xyz_intensity(cloud, tf, self.odom_frame, logger=self.node.get_logger())
        except Exception as e:
            self.node.get_logger().error(f"[TFManager] Transform failed: {e}")
            return None
