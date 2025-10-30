from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from scipy.ndimage import distance_transform_edt


class Costmap2D:
    """Simple 2D occupancy grid costmap"""

    # Cost values
    FREE_SPACE = 0
    OCCUPIED = 100
    UNKNOWN = 255

    def __init__(self, width: float, height: float, resolution: float, origin_x: float = 0.0, origin_y: float = 0.0):
        """
        Initialize costmap.

        Args:
            width: Width in meters
            height: Height in meters
            resolution: Cell size in meters
            origin_x: X coordinate of map origin (lower-left corner)
            origin_y: Y coordinate of map origin (lower-left corner)
        """
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        # Calculate grid dimensions
        self.width_cells = int(np.ceil(width / resolution))
        self.height_cells = int(np.ceil(height / resolution))

        # Initialize grid (0 = free, 100 = occupied, 255 = unknown)
        self.data = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)

    def reset(self, value: int = 0):
        """Reset all cells to given value"""
        self.data.fill(value)

    def world_to_map(self, wx: float, wy: float) -> Tuple[Optional[int], Optional[int]]:
        """
        Convert world coordinates to map cell indices.

        Returns:
            (mx, my): Cell indices, or (None, None) if out of bounds
        """
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)

        if 0 <= mx < self.width_cells and 0 <= my < self.height_cells:
            return (mx, my)
        return (None, None)

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """
        Convert map cell indices to world coordinates (cell center).

        Returns:
            (wx, wy): World coordinates
        """
        wx = self.origin_x + (mx + 0.5) * self.resolution
        wy = self.origin_y + (my + 0.5) * self.resolution
        return (wx, wy)

    def set_cost(self, mx: int, my: int, cost: int):
        """Set cost at cell (mx, my)"""
        if 0 <= mx < self.width_cells and 0 <= my < self.height_cells:
            self.data[my, mx] = cost

    def get_cost(self, mx: int, my: int) -> Optional[int]:
        """Get cost at cell (mx, my)"""
        if 0 <= mx < self.width_cells and 0 <= my < self.height_cells:
            return int(self.data[my, mx])
        return None

    def set_cost_world(self, wx: float, wy: float, cost: int):
        """Set cost at world coordinate (wx, wy)"""
        mx, my = self.world_to_map(wx, wy)
        if mx is not None and my is not None:
            self.set_cost(mx, my, cost)

    def get_data_flat(self) -> np.ndarray:
        """Get flattened data array for OccupancyGrid message (row-major)"""
        return self.data.flatten().astype(np.int8)

    def update_origin(self, new_origin_x: float, new_origin_y: float):
        """Update map origin (for rolling window)"""
        self.origin_x = new_origin_x
        self.origin_y = new_origin_y

    def inflate(self, inflation_radius: float, cost_scaling_factor: float = 10.0):
        """
        Apply inflation layer around obstacles.

        Args:
            inflation_radius: Maximum distance to inflate (in meters)
            cost_scaling_factor: Factor for exponential cost decay (higher = steeper decay)
        """
        if inflation_radius <= 0:
            return

        # Convert inflation radius to cells
        inflation_cells = inflation_radius / self.resolution

        # Create binary obstacle mask (True = obstacle)
        obstacle_mask = (self.data >= self.OCCUPIED)

        if not np.any(obstacle_mask):
            # No obstacles, nothing to inflate
            return

        # Compute distance transform (distance to nearest obstacle in cells)
        # ~obstacle_mask inverts: True where FREE, False where OCCUPIED
        distances_cells = distance_transform_edt(~obstacle_mask)

        # Convert to meters
        distances_m = distances_cells * self.resolution

        # Create inflation mask (cells within inflation radius)
        inflation_mask = distances_m <= inflation_radius

        # Compute inflated costs using exponential decay
        # cost = OCCUPIED * exp(-cost_scaling_factor * distance / inflation_radius)
        inflated_costs = self.OCCUPIED * np.exp(
            -cost_scaling_factor * distances_m[inflation_mask] / inflation_radius
        )

        # Round and clip to valid range [0, OCCUPIED]
        inflated_costs = np.clip(inflated_costs.astype(np.uint8), 0, self.OCCUPIED)

        # Update costmap: take maximum of existing cost and inflated cost
        # This preserves OCCUPIED (100) cells and adds inflation around them
        self.data[inflation_mask] = np.maximum(
            self.data[inflation_mask],
            inflated_costs
        )
