"""Costmap publishing utilities."""

import numpy as np
from typing import List
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

from .costmap_2d import SimpleCostmap2D
from .cost_values import FREE_SPACE, NO_INFORMATION, LETHAL_OBSTACLE


@dataclass
class CostmapPublisherConfig:
    """Configuration for CostmapPublisher."""

    global_frame: str = "map"
    always_send_full_costmap: bool = False
    publish_frequency: float = 10.0

    # Visualization settings
    enable_obstacle_markers: bool = True
    enable_filtered_cloud: bool = True
    marker_lifetime: float = 1.0
    obstacle_marker_size: float = 0.1

    # Topic names
    costmap_topic: str = "costmap"
    costmap_updates_topic: str = "costmap_updates"
    obstacle_markers_topic: str = "obstacle_markers"
    filtered_cloud_topic: str = "filtered_pointcloud"


class CostmapPublisher:
    """Publisher for costmap and related visualization."""

    # Static cost translation table
    cost_translation_table_ = None

    def __init__(self, node: Node, costmap: SimpleCostmap2D,
                 config: CostmapPublisherConfig):
        """Initialize the publisher."""
        self.node_ = node
        self.costmap_ = costmap
        self.config_ = config

        self.grid_msg_ = OccupancyGrid()
        self.grid_update_msg_ = OccupancyGridUpdate()

        self.x0_ = 0
        self.xn_ = 0
        self.y0_ = 0
        self.yn_ = 0
        self.saved_origin_x_ = 0.0
        self.saved_origin_y_ = 0.0
        self.has_updated_data_ = False
        self.active_ = False

        self.logger_ = node.get_logger()

        # Initialize cost translation table
        if CostmapPublisher.cost_translation_table_ is None:
            CostmapPublisher._init_cost_translation_table()

        # Create publishers
        self.costmap_pub_ = node.create_publisher(
            OccupancyGrid, config.costmap_topic, 1)

        self.costmap_update_pub_ = node.create_publisher(
            OccupancyGridUpdate, config.costmap_updates_topic, 10)

        if config.enable_obstacle_markers:
            self.obstacle_markers_pub_ = node.create_publisher(
                MarkerArray, config.obstacle_markers_topic, 1)
        else:
            self.obstacle_markers_pub_ = None

        if config.enable_filtered_cloud:
            self.filtered_cloud_pub_ = node.create_publisher(
                PointCloud2, config.filtered_cloud_topic, 1)
        else:
            self.filtered_cloud_pub_ = None

        # Reset bounds
        self.reset_bounds()

        self.logger_.info("CostmapPublisher initialized")

    @staticmethod
    def _init_cost_translation_table():
        """Initialize cost translation table."""
        CostmapPublisher.cost_translation_table_ = np.zeros(256, dtype=np.int8)

        for i in range(256):
            if i == NO_INFORMATION:
                CostmapPublisher.cost_translation_table_[i] = -1
            elif i == FREE_SPACE:
                CostmapPublisher.cost_translation_table_[i] = 0
            else:
                # Scale from 1-254 to 1-100
                value = int(round(i * 100.0 / LETHAL_OBSTACLE))
                CostmapPublisher.cost_translation_table_[i] = min(value, 100)

    def publish_costmap(self):
        """Publish the costmap."""
        if not self.active_ or self.costmap_ is None:
            return

        # Check if we need to send full costmap or just an update
        if (self.config_.always_send_full_costmap or
            not self.has_updated_data_ or
            self.saved_origin_x_ != self.costmap_.get_origin_x() or
                self.saved_origin_y_ != self.costmap_.get_origin_y()):

            self._prepare_grid()
            self.costmap_pub_.publish(self.grid_msg_)

            self.saved_origin_x_ = self.costmap_.get_origin_x()
            self.saved_origin_y_ = self.costmap_.get_origin_y()

        elif self.has_updated_data_:
            # Send incremental update
            width = self.xn_ - self.x0_
            height = self.yn_ - self.y0_

            if width > 0 and height > 0:
                self.publish_costmap_update(self.x0_, self.y0_, width, height)

        self.has_updated_data_ = False
        self.reset_bounds()

    def publish_costmap_update(self, x0: int, y0: int,
                              width: int, height: int):
        """Publish costmap update."""
        if not self.active_ or self.costmap_update_pub_ is None:
            return

        self._prepare_grid_update(x0, y0, width, height)
        self.costmap_update_pub_.publish(self.grid_update_msg_)

    def publish_obstacle_markers(self, obstacle_points: List[Point]):
        """Publish obstacle visualization markers."""
        if (not self.active_ or
            not self.config_.enable_obstacle_markers or
                self.obstacle_markers_pub_ is None):
            return

        markers = MarkerArray()
        self._create_obstacle_markers(obstacle_points, markers)
        self.obstacle_markers_pub_.publish(markers)

    def publish_filtered_pointcloud(self, filtered_cloud: np.ndarray):
        """Publish filtered point cloud."""
        if (not self.active_ or
            not self.config_.enable_filtered_cloud or
            self.filtered_cloud_pub_ is None or
                len(filtered_cloud) == 0):
            return

        # Convert numpy array to PointCloud2
        header = Header()
        header.frame_id = self.config_.global_frame
        header.stamp = self.node_.get_clock().now().to_msg()

        # Create point cloud message
        points = [(float(p[0]), float(p[1]), float(p[2]))
                  for p in filtered_cloud]
        cloud_msg = pc2.create_cloud_xyz32(header, points)

        self.filtered_cloud_pub_.publish(cloud_msg)

    def _prepare_grid(self):
        """Prepare full costmap grid message."""
        if self.costmap_ is None:
            return

        self.grid_msg_.header.frame_id = self.config_.global_frame
        self.grid_msg_.header.stamp = self.node_.get_clock().now().to_msg()

        self.grid_msg_.info.resolution = self.costmap_.get_resolution()
        self.grid_msg_.info.width = self.costmap_.get_size_in_cells_x()
        self.grid_msg_.info.height = self.costmap_.get_size_in_cells_y()

        self.grid_msg_.info.origin.position.x = self.costmap_.get_origin_x()
        self.grid_msg_.info.origin.position.y = self.costmap_.get_origin_y()
        self.grid_msg_.info.origin.position.z = 0.0
        self.grid_msg_.info.origin.orientation.w = 1.0

        size = self.costmap_.get_size_in_cells_x() * self.costmap_.get_size_in_cells_y()
        data = self.costmap_.get_char_map()

        # Convert using translation table
        self.grid_msg_.data = [
            int(CostmapPublisher.cost_translation_table_[data[i // self.costmap_.get_size_in_cells_x(), i % self.costmap_.get_size_in_cells_x()]])
            for i in range(size)
        ]

    def _prepare_grid_update(self, x0: int, y0: int, width: int, height: int):
        """Prepare costmap update message."""
        if self.costmap_ is None:
            return

        self.grid_update_msg_.header.frame_id = self.config_.global_frame
        self.grid_update_msg_.header.stamp = self.node_.get_clock().now().to_msg()

        self.grid_update_msg_.x = x0
        self.grid_update_msg_.y = y0
        self.grid_update_msg_.width = width
        self.grid_update_msg_.height = height

        data = self.costmap_.get_char_map()
        costmap_width = self.costmap_.get_size_in_cells_x()

        # Fill update data
        self.grid_update_msg_.data = []
        for y in range(height):
            for x in range(width):
                costmap_y = y0 + y
                costmap_x = x0 + x
                cost = data[costmap_y, costmap_x]
                self.grid_update_msg_.data.append(
                    int(CostmapPublisher.cost_translation_table_[cost]))

    def _create_obstacle_markers(self, obstacle_points: List[Point],
                                 markers: MarkerArray):
        """Create obstacle visualization markers."""
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = self.config_.global_frame
        clear_marker.header.stamp = self.node_.get_clock().now().to_msg()
        clear_marker.ns = "obstacle_points"
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        if not obstacle_points:
            return

        # Create obstacle markers
        marker = Marker()
        marker.header.frame_id = self.config_.global_frame
        marker.header.stamp = self.node_.get_clock().now().to_msg()
        marker.ns = "obstacle_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.config_.obstacle_marker_size
        marker.scale.y = self.config_.obstacle_marker_size
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        marker.lifetime = Duration(
            seconds=self.config_.marker_lifetime).to_msg()

        marker.points = obstacle_points

        markers.markers.append(marker)

    def update_bounds(self, x0: int, xn: int, y0: int, yn: int):
        """Update bounds for incremental updates."""
        self.x0_ = min(self.x0_, x0)
        self.xn_ = max(self.xn_, xn)
        self.y0_ = min(self.y0_, y0)
        self.yn_ = max(self.yn_, yn)
        self.has_updated_data_ = True

    def reset_bounds(self):
        """Reset bounds."""
        if self.costmap_:
            self.x0_ = self.costmap_.get_size_in_cells_x()
            self.xn_ = 0
            self.y0_ = self.costmap_.get_size_in_cells_y()
            self.yn_ = 0
        else:
            self.x0_ = 0
            self.xn_ = 0
            self.y0_ = 0
            self.yn_ = 0
        self.has_updated_data_ = False

    def update_config(self, config: CostmapPublisherConfig):
        """Update configuration."""
        self.config_ = config
        self.logger_.info("CostmapPublisher configuration updated")

    def get_config(self) -> CostmapPublisherConfig:
        """Get configuration."""
        return self.config_

    def on_activate(self):
        """Activate publisher."""
        self.active_ = True
        self.logger_.info("CostmapPublisher activated")

    def on_deactivate(self):
        """Deactivate publisher."""
        self.active_ = False
        self.logger_.info("CostmapPublisher deactivated")

    def is_active(self) -> bool:
        """Check if publisher is active."""
        return self.active_
