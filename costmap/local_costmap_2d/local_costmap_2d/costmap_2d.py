"""SimpleCostmap2D class for 2D grid-based costmap representation."""

import numpy as np
import threading
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from builtin_interfaces.msg import Time
from rclpy.clock import Clock

from .cost_values import (
    FREE_SPACE, LETHAL_OBSTACLE, NO_INFORMATION
)


@dataclass
class MapLocation:
    """Map location in grid coordinates."""
    x: int
    y: int


class SimpleCostmap2D:
    """A 2D costmap for obstacle representation and navigation."""

    def __init__(self, cells_size_x: int = 0, cells_size_y: int = 0,
                 resolution: float = 0.0, origin_x: float = 0.0,
                 origin_y: float = 0.0, default_value: int = FREE_SPACE):
        """Initialize costmap with given parameters."""
        self.size_x_ = cells_size_x
        self.size_y_ = cells_size_y
        self.resolution_ = resolution
        self.origin_x_ = origin_x
        self.origin_y_ = origin_y
        self.default_value_ = default_value
        self.obstacle_points_ = []
        self.access_mutex_ = threading.Lock()

        if cells_size_x > 0 and cells_size_y > 0:
            self.costmap_ = np.full(
                (cells_size_y, cells_size_x), default_value, dtype=np.uint8)
        else:
            self.costmap_ = None

    @classmethod
    def from_occupancy_grid(cls, grid_msg: OccupancyGrid):
        """Create costmap from OccupancyGrid message."""
        instance = cls()
        instance.size_x_ = grid_msg.info.width
        instance.size_y_ = grid_msg.info.height
        instance.resolution_ = grid_msg.info.resolution
        instance.origin_x_ = grid_msg.info.origin.position.x
        instance.origin_y_ = grid_msg.info.origin.position.y
        instance.default_value_ = FREE_SPACE

        # Convert OccupancyGrid data to costmap format
        instance.costmap_ = np.zeros(
            (instance.size_y_, instance.size_x_), dtype=np.uint8)

        for i, data in enumerate(grid_msg.data):
            y = i // instance.size_x_
            x = i % instance.size_x_
            if data == -1:
                instance.costmap_[y, x] = NO_INFORMATION
            else:
                # Convert from 0-100 occupancy to 0-254 cost
                instance.costmap_[y, x] = int(
                    round(data * LETHAL_OBSTACLE / 100.0))

        return instance

    def get_cost(self, mx: int, my: int) -> int:
        """Get cost at map coordinates."""
        if 0 <= mx < self.size_x_ and 0 <= my < self.size_y_:
            return int(self.costmap_[my, mx])
        return NO_INFORMATION

    def get_cost_by_index(self, index: int) -> int:
        """Get cost by linear index."""
        my = index // self.size_x_
        mx = index % self.size_x_
        return self.get_cost(mx, my)

    def set_cost(self, mx: int, my: int, cost: int):
        """Set cost at map coordinates."""
        if 0 <= mx < self.size_x_ and 0 <= my < self.size_y_:
            self.costmap_[my, mx] = cost

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """Convert map coordinates to world coordinates."""
        wx = self.origin_x_ + (mx + 0.5) * self.resolution_
        wy = self.origin_y_ + (my + 0.5) * self.resolution_
        return wx, wy

    def world_to_map(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to map coordinates."""
        if wx < self.origin_x_ or wy < self.origin_y_:
            return None

        mx = int((wx - self.origin_x_) / self.resolution_)
        my = int((wy - self.origin_y_) / self.resolution_)

        if 0 <= mx < self.size_x_ and 0 <= my < self.size_y_:
            return mx, my
        return None

    def world_to_map_no_bounds(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world to map coordinates without bounds checking."""
        mx = int((wx - self.origin_x_) / self.resolution_)
        my = int((wy - self.origin_y_) / self.resolution_)
        return mx, my

    def world_to_map_enforce_bounds(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world to map coordinates with bounds enforcement."""
        if wx < self.origin_x_:
            mx = 0
        elif wx > self.resolution_ * self.size_x_ + self.origin_x_:
            mx = self.size_x_ - 1
        else:
            mx = int((wx - self.origin_x_) / self.resolution_)

        if wy < self.origin_y_:
            my = 0
        elif wy > self.resolution_ * self.size_y_ + self.origin_y_:
            my = self.size_y_ - 1
        else:
            my = int((wy - self.origin_y_) / self.resolution_)

        return mx, my

    def get_index(self, mx: int, my: int) -> int:
        """Get linear index from map coordinates."""
        return my * self.size_x_ + mx

    def index_to_cells(self, index: int) -> Tuple[int, int]:
        """Convert linear index to map coordinates."""
        my = index // self.size_x_
        mx = index - (my * self.size_x_)
        return mx, my

    def get_char_map(self) -> np.ndarray:
        """Get the costmap array."""
        return self.costmap_

    def get_size_in_cells_x(self) -> int:
        """Get width in cells."""
        return self.size_x_

    def get_size_in_cells_y(self) -> int:
        """Get height in cells."""
        return self.size_y_

    def get_size_in_meters_x(self) -> float:
        """Get width in meters."""
        return (self.size_x_ - 1 + 0.5) * self.resolution_

    def get_size_in_meters_y(self) -> float:
        """Get height in meters."""
        return (self.size_y_ - 1 + 0.5) * self.resolution_

    def get_origin_x(self) -> float:
        """Get origin X coordinate."""
        return self.origin_x_

    def get_origin_y(self) -> float:
        """Get origin Y coordinate."""
        return self.origin_y_

    def get_resolution(self) -> float:
        """Get resolution."""
        return self.resolution_

    def reset_map(self, x0: int, y0: int, xn: int, yn: int):
        """Reset map region to default value."""
        self.reset_map_to_value(x0, y0, xn, yn, self.default_value_)

    def reset_map_to_value(self, x0: int, y0: int, xn: int, yn: int, value: int):
        """Reset map region to specific value."""
        with self.access_mutex_:
            self.costmap_[y0:yn, x0:xn] = value

    def resize_map(self, size_x: int, size_y: int, resolution: float,
                   origin_x: float, origin_y: float):
        """Resize the costmap."""
        self.size_x_ = size_x
        self.size_y_ = size_y
        self.resolution_ = resolution
        self.origin_x_ = origin_x
        self.origin_y_ = origin_y

        with self.access_mutex_:
            self.costmap_ = np.full(
                (size_y, size_x), self.default_value_, dtype=np.uint8)

    def project_point(self, x: float, y: float, z: float = 0.0):
        """Project a point onto the costmap."""
        result = self.world_to_map(x, y)
        if result is not None:
            mx, my = result
            self.set_cost(mx, my, LETHAL_OBSTACLE)

            # Store obstacle point
            point = Point()
            point.x = x
            point.y = y
            point.z = z
            self.obstacle_points_.append(point)

    def mark_obstacle_with_height(self, x: float, y: float, z: float, cost: int = LETHAL_OBSTACLE):
        """Mark an obstacle with specific cost and height."""
        result = self.world_to_map(x, y)
        if result is not None:
            mx, my = result
            self.set_cost(mx, my, cost)

            # Store obstacle point
            point = Point()
            point.x = x
            point.y = y
            point.z = z
            self.obstacle_points_.append(point)

    def process_point_batch(self, points: List[Point]):
        """Process a batch of points."""
        self.obstacle_points_.clear()

        for point in points:
            self.project_point(point.x, point.y, point.z)

    def apply_temporal_decay(self, decay_rate: float = 0.95):
        """Apply temporal decay to costmap for dynamic obstacles."""
        with self.access_mutex_:
            mask = (self.costmap_ > FREE_SPACE) & (
                self.costmap_ < NO_INFORMATION)
            decayed = (self.costmap_ * decay_rate).astype(np.uint8)
            self.costmap_ = np.where(
                mask, np.maximum(decayed, FREE_SPACE), self.costmap_)

    def raytrace(self, x0: float, y0: float, x1: float, y1: float):
        """Raytrace a line and mark cells as free."""
        result0 = self.world_to_map(x0, y0)
        result1 = self.world_to_map(x1, y1)

        if result0 is None or result1 is None:
            return

        cell_x0, cell_y0 = result0
        cell_x1, cell_y1 = result1

        self._raytrace_line(cell_x0, cell_y0, cell_x1, cell_y1, FREE_SPACE)

    def _raytrace_line(self, x0: int, y0: int, x1: int, y1: int, value: int):
        """Bresenham's line algorithm for ray tracing."""
        dx = x1 - x0
        dy = y1 - y0

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        offset_dx = 1 if dx > 0 else -1
        offset_dy = 1 if dy > 0 else -1

        x, y = x0, y0

        if abs_dx >= abs_dy:
            error_y = abs_dx // 2
            for _ in range(abs_dx):
                self.set_cost(x, y, value)
                x += offset_dx
                error_y += abs_dy
                if error_y >= abs_dx:
                    y += offset_dy
                    error_y -= abs_dx
            self.set_cost(x, y, value)
        else:
            error_x = abs_dy // 2
            for _ in range(abs_dy):
                self.set_cost(x, y, value)
                y += offset_dy
                error_x += abs_dx
                if error_x >= abs_dy:
                    x += offset_dx
                    error_x -= abs_dy
            self.set_cost(x, y, value)

    def to_occupancy_grid(self, frame_id: str) -> OccupancyGrid:
        """Convert costmap to OccupancyGrid message."""
        grid = OccupancyGrid()

        grid.header.frame_id = frame_id
        grid.header.stamp = Clock().now().to_msg()

        grid.info.resolution = self.resolution_
        grid.info.width = self.size_x_
        grid.info.height = self.size_y_
        grid.info.origin.position.x = self.origin_x_
        grid.info.origin.position.y = self.origin_y_
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0

        # Convert costmap data to occupancy grid format
        grid.data = []
        for i in range(self.size_y_ * self.size_x_):
            y = i // self.size_x_
            x = i % self.size_x_
            cost = self.costmap_[y, x]

            if cost == NO_INFORMATION:
                grid.data.append(-1)
            else:
                # Convert from 0-254 cost to 0-100 occupancy
                occupancy = int(round(cost * 100.0 / LETHAL_OBSTACLE))
                grid.data.append(occupancy)

        return grid

    def clear_robot_footprint(self, footprint: List[Point],
                             robot_x: float, robot_y: float, robot_yaw: float):
        """Clear the robot footprint from the costmap."""
        if not footprint:
            return

        # Transform footprint to world coordinates
        world_footprint = []
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        for point in footprint:
            world_point = Point()
            world_point.x = robot_x + point.x * cos_yaw - point.y * sin_yaw
            world_point.y = robot_y + point.x * sin_yaw + point.y * cos_yaw
            world_point.z = 0.0
            world_footprint.append(world_point)

        # Find bounding box
        min_x = min(p.x for p in world_footprint)
        max_x = max(p.x for p in world_footprint)
        min_y = min(p.y for p in world_footprint)
        max_y = max(p.y for p in world_footprint)

        # Convert bounding box to map coordinates
        result_min = self.world_to_map(min_x, min_y)
        result_max = self.world_to_map(max_x, max_y)

        if result_min is None or result_max is None:
            return

        min_mx, min_my = result_min
        max_mx, max_my = result_max

        # Clear cells within the footprint
        for mx in range(min_mx, min(max_mx + 1, self.size_x_)):
            for my in range(min_my, min(max_my + 1, self.size_y_)):
                wx, wy = self.map_to_world(mx, my)

                if self._is_point_in_polygon(wx, wy, world_footprint):
                    self.set_cost(mx, my, FREE_SPACE)

    def _is_point_in_polygon(self, x: float, y: float, polygon: List[Point]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            if ((polygon[i].y > y) != (polygon[j].y > y)) and \
               (x < (polygon[j].x - polygon[i].x) * (y - polygon[i].y) /
                    (polygon[j].y - polygon[i].y) + polygon[i].x):
                inside = not inside
            j = i

        return inside

    def get_obstacle_points(self) -> List[Point]:
        """Get list of obstacle points."""
        return self.obstacle_points_

    def get_mutex(self):
        """Get the mutex lock."""
        return self.access_mutex_
