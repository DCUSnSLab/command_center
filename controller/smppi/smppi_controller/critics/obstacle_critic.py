#!/usr/bin/env python3
"""
Obstacle Avoidance Critic for SMPPI
Costmap-based grid collision detection (optimized for performance)
"""

import torch
import numpy as np
from typing import Optional, Any

from .base_critic import BaseCritic


class ObstacleCritic(BaseCritic):
    """
    Obstacle avoidance critic using costmap grid-based collision detection

    Costmap value mapping:
        0-49:   Free space    → cost = 0
        50-79:  Inflation     → cost = scaled repulsion
        80-100: Occupied      → cost = collision_cost
    """

    def __init__(self, params: dict):
        """Initialize costmap-based obstacle critic"""
        super().__init__("ObstacleCritic", params)

        # Collision cost parameters
        self.collision_cost = params.get('collision_cost', 1000.0)
        self.repulsion_factor = params.get('repulsion_factor', 2.0)

        # Hard-lethal mode: occupied cells (curb walls, corridor keepout) get a
        # near-infinite cost so no colliding trajectory can win the softmax --
        # a summed soft cost of 1000 can be traded away by a good goal score,
        # which is exactly how avoidance used to slip over the curb.
        self.lethal_hard = bool(params.get('lethal_hard', True))
        self.hard_lethal_cost = float(params.get('hard_lethal_cost', 1.0e6))
        # probe advance (2026-08-22): behavior 가 '원거리 가짜 벽' 패턴에서
        # 탐침 전진을 요청하면, 로봇에서 probe_near_r 밖의 occupied 만
        # 유한 비용으로 완화한다. 근거리 occupied·OOB 는 항상 hard —
        # 실벽/연석 접근 시 기존 안전 정지가 그대로 발동한다.
        self.probe_active = False
        self.probe_near_r = float(params.get('probe_near_r', 3.0))
        self.probe_far_cost = float(params.get('probe_far_cost', 500.0))

        # Costmap threshold parameters
        self.occupied_cost_threshold = params.get('occupied_cost_threshold', 80)  # Cost >= 80 is occupied
        self.inflation_zone_start = params.get('inflation_zone_start', 50)        # Cost >= 50 is inflation zone

        # Vehicle footprint parameters
        self.footprint = params.get('footprint', [0.49, 0.3725, 0.49, -0.3725, -0.49, -0.3725, -0.49, 0.3725])
        self.footprint_padding = params.get('footprint_padding', 0.15)
        self.use_polygon_collision = params.get('use_polygon_collision', True)

        # Parse footprint into list of (x, y) tuples
        self.footprint_points = []
        for i in range(0, len(self.footprint), 2):
            if i + 1 < len(self.footprint):
                self.footprint_points.append((self.footprint[i], self.footprint[i+1]))

        # Costmap cache (updated via set_costmap_info)
        self.costmap_info = None

        print(f"[ObstacleCritic] Costmap-based collision detection")
        print(f"[ObstacleCritic] collision_cost={self.collision_cost}, repulsion_factor={self.repulsion_factor}")
        print(f"[ObstacleCritic] thresholds: inflation>={self.inflation_zone_start}, occupied>={self.occupied_cost_threshold}")
        print(f"[ObstacleCritic] footprint: {self.footprint_points}, padding={self.footprint_padding}, use_polygon={self.use_polygon_collision}")

    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                    robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                    obstacles: Optional[Any]) -> torch.Tensor:
        """
        Compute obstacle avoidance cost using costmap grid lookup

        Args:
            trajectories: [K, T+1, 3] sampled trajectories (x, y, theta)
            controls: [K, T, 2] control sequences (unused)
            robot_state: [5] robot state (unused)
            goal_state: [3] goal state (unused)
            obstacles: Obstacle data (unused in costmap mode)

        Returns:
            costs: [K] total obstacle costs per trajectory
        """
        if not self.enabled:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        # Check if costmap is available
        if self.costmap_info is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        return self.compute_cost_from_costmap(trajectories)

    def compute_cost_from_costmap(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Compute collision costs using vectorized costmap grid lookup (O(1) per point)

        Algorithm (with polygon footprint):
            1. Flatten all trajectory poses to [K*(T+1), 3] (x, y, theta)
            2. Rotate footprint for each pose → [K*(T+1), V, 2]
            3. Convert all vertices to grid coords (vectorized)
            4. Lookup costmap values for all vertices
            5. Take max cost per pose (most conservative)
            6. Reshape and sum per trajectory

        Args:
            trajectories: [K, T+1, 3] tensor (x, y, theta)

        Returns:
            costs: [K] tensor of total costs per trajectory
        """
        K, T_plus_1, _ = trajectories.shape

        # Extract costmap metadata
        resolution = self.costmap_info['resolution']
        origin_x = self.costmap_info['origin_x']
        origin_y = self.costmap_info['origin_y']
        width = self.costmap_info['width']
        height = self.costmap_info['height']
        costmap_data = self.costmap_info['data']  # (height, width) numpy array

        if self.use_polygon_collision and len(self.footprint_points) > 0:
            # === POLYGON FOOTPRINT MODE ===
            # Extract all poses: [K, T+1, 3] → [K*(T+1), 3]
            poses = trajectories.detach().cpu().numpy().reshape(-1, 3)  # [N, 3] where N = K*(T+1)
            N = poses.shape[0]

            # Rotate footprint for all poses → [N, V, 2]
            rotated_footprint = self._rotate_footprint(self.footprint_points, poses)  # [N, V, 2]
            V = rotated_footprint.shape[1]  # Number of vertices

            # Flatten to [N*V, 2] for vectorized grid conversion
            vertices_flat = rotated_footprint.reshape(-1, 2)  # [N*V, 2]

            # World to grid conversion (vectorized)
            gx = ((vertices_flat[:, 0] - origin_x) / resolution).astype(np.int32)
            gy = ((vertices_flat[:, 1] - origin_y) / resolution).astype(np.int32)

            # Bounds checking
            valid_mask = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)

            # Initialize costs (out-of-bounds = lethal)
            oob_cost = self.hard_lethal_cost if self.lethal_hard else self.collision_cost
            vertex_costs = np.full(len(gx), oob_cost, dtype=np.float32)

            # Get costmap values for valid vertices
            costmap_values = costmap_data[gy[valid_mask], gx[valid_mask]]  # Range: [0, 100]

            # Map costmap values to costs based on thresholds
            free_space_mask = costmap_values < self.inflation_zone_start
            inflation_mask = (costmap_values >= self.inflation_zone_start) & \
                            (costmap_values < self.occupied_cost_threshold)
            occupied_mask = costmap_values >= self.occupied_cost_threshold

            # Compute costs for valid points
            valid_costs = np.zeros(len(costmap_values), dtype=np.float32)

            # Free space: no cost
            valid_costs[free_space_mask] = 0.0

            # Inflation zone: scaled repulsion cost
            if np.any(inflation_mask):
                inflation_range = self.occupied_cost_threshold - self.inflation_zone_start
                normalized_values = (costmap_values[inflation_mask] - self.inflation_zone_start) / inflation_range
                valid_costs[inflation_mask] = self.repulsion_factor * (normalized_values ** 2) * 100.0

            # Occupied: collision cost (probe 시 원거리만 완화)
            occ_cost = (self.hard_lethal_cost
                        if self.lethal_hard else self.collision_cost)
            if self.probe_active and np.any(occupied_mask):
                robot_xy = poses[0, :2]
                vdist = np.hypot(vertices_flat[valid_mask][:, 0] - robot_xy[0],
                                 vertices_flat[valid_mask][:, 1] - robot_xy[1])
                far_occ = occupied_mask & (vdist > self.probe_near_r)
                near_occ = occupied_mask & ~far_occ
                valid_costs[near_occ] = occ_cost
                valid_costs[far_occ] = self.probe_far_cost
            else:
                valid_costs[occupied_mask] = occ_cost

            # Assign computed costs
            vertex_costs[valid_mask] = valid_costs

            # Reshape to [N, V] and take max cost per pose
            vertex_costs_reshaped = vertex_costs.reshape(N, V)  # [N, V]
            pose_costs = vertex_costs_reshaped.max(axis=1)  # [N] - max cost among vertices

            # Reshape to [K, T+1] and sum per trajectory
            pose_costs_reshaped = pose_costs.reshape(K, T_plus_1)  # [K, T+1]
            total_costs = pose_costs_reshaped.sum(axis=1)  # [K]

        else:
            # === SINGLE POINT MODE (original behavior) ===
            # Extract trajectory positions: [K, T+1, 2] → [K*(T+1), 2]
            traj_xy = trajectories[:, :, :2].detach().cpu().numpy()
            traj_xy_flat = traj_xy.reshape(-1, 2)  # Shape: [K*(T+1), 2]

            # World to grid conversion (vectorized)
            gx = ((traj_xy_flat[:, 0] - origin_x) / resolution).astype(np.int32)
            gy = ((traj_xy_flat[:, 1] - origin_y) / resolution).astype(np.int32)

            # Bounds checking
            valid_mask = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)

            # Initialize costs (out-of-bounds = lethal)
            oob_cost = self.hard_lethal_cost if self.lethal_hard else self.collision_cost
            point_costs = np.full(len(gx), oob_cost, dtype=np.float32)

            # Get costmap values for valid points (O(1) lookup per point)
            costmap_values = costmap_data[gy[valid_mask], gx[valid_mask]]  # Range: [0, 100]

            # Map costmap values to costs based on thresholds
            free_space_mask = costmap_values < self.inflation_zone_start
            inflation_mask = (costmap_values >= self.inflation_zone_start) & \
                            (costmap_values < self.occupied_cost_threshold)
            occupied_mask = costmap_values >= self.occupied_cost_threshold

            # Compute costs for valid points
            valid_costs = np.zeros(len(costmap_values), dtype=np.float32)

            # Free space: no cost
            valid_costs[free_space_mask] = 0.0

            # Inflation zone: scaled repulsion cost
            # Map costmap value [50, 80) to [0, 1] → apply quadratic penalty
            if np.any(inflation_mask):
                inflation_range = self.occupied_cost_threshold - self.inflation_zone_start
                normalized_values = (costmap_values[inflation_mask] - self.inflation_zone_start) / inflation_range
                valid_costs[inflation_mask] = self.repulsion_factor * (normalized_values ** 2) * 100.0

            # Occupied: collision cost
            occ_cost = (self.hard_lethal_cost
                        if self.lethal_hard else self.collision_cost)
            if self.probe_active and np.any(occupied_mask):
                robot_xy = traj_xy_flat[0]
                pdist = np.hypot(traj_xy_flat[valid_mask][:, 0] - robot_xy[0],
                                 traj_xy_flat[valid_mask][:, 1] - robot_xy[1])
                far_occ = occupied_mask & (pdist > self.probe_near_r)
                valid_costs[occupied_mask & ~far_occ] = occ_cost
                valid_costs[far_occ] = self.probe_far_cost
            else:
                valid_costs[occupied_mask] = occ_cost

            # Assign computed costs
            point_costs[valid_mask] = valid_costs

            # Reshape back to [K, T+1] and sum per trajectory
            point_costs_reshaped = point_costs.reshape(K, T_plus_1)
            total_costs = point_costs_reshaped.sum(axis=1)  # Shape: [K]

        # Convert to torch tensor and apply weight
        total_costs_tensor = torch.tensor(total_costs, device=self.device, dtype=self.dtype)
        return self.apply_weight(total_costs_tensor)

    def _rotate_footprint(self, footprint_points: list, poses: np.ndarray) -> np.ndarray:
        """
        Rotate and translate footprint for each pose in batch

        Args:
            footprint_points: List of (x, y) tuples in body frame
            poses: [N, 3] array of (x, y, theta) poses

        Returns:
            rotated_footprint: [N, num_vertices, 2] array of rotated footprint vertices in world frame
        """
        N = poses.shape[0]
        num_vertices = len(footprint_points)

        # Convert footprint to numpy array [num_vertices, 2]
        footprint_array = np.array(footprint_points, dtype=np.float32)  # [V, 2]

        # Extract poses
        x = poses[:, 0]      # [N]
        y = poses[:, 1]      # [N]
        theta = poses[:, 2]  # [N]

        # Compute rotation matrices for all poses
        cos_theta = np.cos(theta)  # [N]
        sin_theta = np.sin(theta)  # [N]

        # Broadcast footprint to all poses: [N, V, 2]
        footprint_batch = np.tile(footprint_array[np.newaxis, :, :], (N, 1, 1))  # [N, V, 2]

        # Apply rotation and translation vectorized
        # x' = x_robot + (x_footprint * cos(θ) - y_footprint * sin(θ))
        # y' = y_robot + (x_footprint * sin(θ) + y_footprint * cos(θ))

        fx = footprint_batch[:, :, 0]  # [N, V]
        fy = footprint_batch[:, :, 1]  # [N, V]

        # Rotated coordinates
        rotated_x = x[:, np.newaxis] + (fx * cos_theta[:, np.newaxis] - fy * sin_theta[:, np.newaxis])  # [N, V]
        rotated_y = y[:, np.newaxis] + (fx * sin_theta[:, np.newaxis] + fy * cos_theta[:, np.newaxis])  # [N, V]

        # Stack to [N, V, 2]
        rotated_footprint = np.stack([rotated_x, rotated_y], axis=2)  # [N, V, 2]

        return rotated_footprint

    def set_probe(self, active: bool):
        self.probe_active = bool(active)

    def set_costmap_info(self, costmap_info: dict):
        """
        Set costmap information for grid-based collision detection

        Args:
            costmap_info: Dictionary with costmap metadata
                - resolution: cell size in meters (e.g., 0.05)
                - origin_x, origin_y: map origin in world coordinates
                - width, height: grid dimensions in cells
                - data: 2D numpy array of shape (height, width) with values [0-100]
        """
        self.costmap_info = costmap_info

    def update_parameters(self, params: dict):
        """
        Update obstacle critic parameters dynamically

        Args:
            params: Dictionary with parameter updates
                - collision_cost: Cost for occupied cells
                - repulsion_factor: Scaling factor for inflation zone
                - occupied_cost_threshold: Threshold for occupied cells
                - inflation_zone_start: Threshold for inflation zone start
        """
        if 'collision_cost' in params:
            self.collision_cost = params['collision_cost']
        if 'repulsion_factor' in params:
            self.repulsion_factor = params['repulsion_factor']
        if 'occupied_cost_threshold' in params:
            self.occupied_cost_threshold = params['occupied_cost_threshold']
        if 'inflation_zone_start' in params:
            self.inflation_zone_start = params['inflation_zone_start']

        print(f"[ObstacleCritic] Parameters updated: "
              f"collision={self.collision_cost}, repulsion={self.repulsion_factor}, "
              f"thresholds=[{self.inflation_zone_start}, {self.occupied_cost_threshold}]")
