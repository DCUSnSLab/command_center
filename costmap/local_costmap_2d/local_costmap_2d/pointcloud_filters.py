"""PointCloud filtering utilities."""

import numpy as np
import math
from typing import List, Tuple
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


class PointCloudFilters:
    """Collection of point cloud filtering algorithms."""

    @staticmethod
    def remove_ground_plane(cloud: np.ndarray, distance_threshold: float = 0.1,
                           max_iterations: int = 1000) -> np.ndarray:
        """
        Remove ground plane using RANSAC.

        Args:
            cloud: Nx3 array of points (x, y, z)
            distance_threshold: RANSAC distance threshold
            max_iterations: Maximum RANSAC iterations

        Returns:
            Filtered point cloud without ground plane
        """
        if len(cloud) == 0:
            return cloud

        best_inliers = []
        best_num_inliers = 0

        # RANSAC algorithm
        for _ in range(max_iterations):
            # Randomly sample 3 points
            if len(cloud) < 3:
                break

            sample_indices = np.random.choice(len(cloud), 3, replace=False)
            p1, p2, p3 = cloud[sample_indices]

            # Calculate plane normal and d
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)

            if np.linalg.norm(normal) < 1e-6:
                continue

            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, p1)

            # Calculate distance of all points to plane
            distances = np.abs(np.dot(cloud, normal) + d)

            # Count inliers
            inliers = distances < distance_threshold
            num_inliers = np.sum(inliers)

            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers

        # Return points that are not inliers (non-ground points)
        if best_num_inliers > 0:
            return cloud[~best_inliers]
        return cloud

    @staticmethod
    def filter_by_height(cloud: np.ndarray, min_height: float,
                        max_height: float) -> np.ndarray:
        """
        Filter points by height (Z-axis).

        Args:
            cloud: Nx3 array of points
            min_height: Minimum Z value
            max_height: Maximum Z value

        Returns:
            Filtered point cloud
        """
        if len(cloud) == 0:
            return cloud

        mask = (cloud[:, 2] >= min_height) & (cloud[:, 2] <= max_height)
        return cloud[mask]

    @staticmethod
    def filter_by_range(cloud: np.ndarray, max_range: float,
                       min_range: float = 0.0) -> np.ndarray:
        """
        Filter points by distance from origin.

        Args:
            cloud: Nx3 array of points
            max_range: Maximum distance
            min_range: Minimum distance

        Returns:
            Filtered point cloud
        """
        if len(cloud) == 0:
            return cloud

        ranges = np.linalg.norm(cloud, axis=1)
        mask = (ranges >= min_range) & (ranges <= max_range)
        return cloud[mask]

    @staticmethod
    def voxel_grid_downsample(cloud: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        Downsample point cloud using voxel grid.

        Args:
            cloud: Nx3 array of points
            voxel_size: Size of voxel grid

        Returns:
            Downsampled point cloud
        """
        if len(cloud) == 0 or voxel_size <= 0:
            return cloud

        # Calculate voxel indices for each point
        voxel_indices = np.floor(cloud / voxel_size).astype(np.int32)

        # Create unique voxel identifier
        voxel_keys = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in voxel_keys:
                voxel_keys[key] = []
            voxel_keys[key].append(i)

        # Compute centroid for each voxel
        filtered_points = []
        for indices in voxel_keys.values():
            centroid = np.mean(cloud[indices], axis=0)
            filtered_points.append(centroid)

        return np.array(filtered_points) if filtered_points else np.array([]).reshape(0, 3)

    @staticmethod
    def remove_outliers(cloud: np.ndarray, mean_k: int = 50,
                       std_dev_mul: float = 1.0) -> np.ndarray:
        """
        Remove statistical outliers.

        Args:
            cloud: Nx3 array of points
            mean_k: Number of neighbors for statistics
            std_dev_mul: Standard deviation threshold multiplier

        Returns:
            Filtered point cloud
        """
        if len(cloud) < mean_k:
            return cloud

        # Build KD-tree
        tree = cKDTree(cloud)

        # Calculate mean distance to k nearest neighbors for each point
        mean_distances = []
        for point in cloud:
            distances, _ = tree.query(point, k=mean_k + 1)
            # Exclude the point itself (distance 0)
            mean_dist = np.mean(distances[1:])
            mean_distances.append(mean_dist)

        mean_distances = np.array(mean_distances)

        # Calculate threshold
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + std_dev_mul * global_std

        # Filter outliers
        mask = mean_distances <= threshold
        return cloud[mask]

    @staticmethod
    def euclidean_clustering(cloud: np.ndarray, min_cluster_size: int,
                            cluster_tolerance: float = 0.3,
                            max_cluster_size: int = 25000) -> np.ndarray:
        """
        Perform Euclidean clustering and filter small clusters.

        Args:
            cloud: Nx3 array of points
            min_cluster_size: Minimum points in cluster
            cluster_tolerance: Distance threshold for clustering
            max_cluster_size: Maximum points in cluster

        Returns:
            Filtered point cloud with only valid clusters
        """
        if len(cloud) == 0:
            return cloud

        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=cluster_tolerance, min_samples=min_cluster_size)
        labels = clustering.fit_predict(cloud)

        # Filter points based on cluster size
        filtered_points = []
        for label in set(labels):
            if label == -1:  # Noise points
                continue

            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)

            if min_cluster_size <= cluster_size <= max_cluster_size:
                filtered_points.append(cloud[cluster_mask])

        if filtered_points:
            return np.vstack(filtered_points)
        return np.array([]).reshape(0, 3)

    @staticmethod
    def filter_robot_footprint(cloud: np.ndarray,
                              footprint: List[Tuple[float, float]],
                              robot_x: float = 0.0,
                              robot_y: float = 0.0,
                              robot_yaw: float = 0.0) -> np.ndarray:
        """
        Remove points inside robot footprint.

        Args:
            cloud: Nx3 array of points
            footprint: List of (x, y) footprint vertices
            robot_x: Robot X position
            robot_y: Robot Y position
            robot_yaw: Robot yaw angle

        Returns:
            Filtered point cloud
        """
        if len(cloud) == 0 or not footprint:
            return cloud

        filtered_points = []
        for point in cloud:
            # Transform point to robot coordinate frame
            x = point[0] - robot_x
            y = point[1] - robot_y
            x_rot, y_rot = PointCloudFilters._transform_point(
                x, y, 0.0, 0.0, -robot_yaw)

            if not PointCloudFilters._is_point_in_polygon(x_rot, y_rot, footprint):
                filtered_points.append(point)

        return np.array(filtered_points) if filtered_points else np.array([]).reshape(0, 3)

    @staticmethod
    def filter_by_region(cloud: np.ndarray,
                        min_x: float, max_x: float,
                        min_y: float, max_y: float,
                        min_z: float, max_z: float) -> np.ndarray:
        """
        Filter points by region boundaries.

        Args:
            cloud: Nx3 array of points
            min_x, max_x: X boundaries
            min_y, max_y: Y boundaries
            min_z, max_z: Z boundaries

        Returns:
            Filtered point cloud
        """
        if len(cloud) == 0:
            return cloud

        mask = (cloud[:, 0] >= min_x) & (cloud[:, 0] <= max_x) & \
               (cloud[:, 1] >= min_y) & (cloud[:, 1] <= max_y) & \
               (cloud[:, 2] >= min_z) & (cloud[:, 2] <= max_z)

        return cloud[mask]

    @staticmethod
    def _is_point_in_polygon(x: float, y: float,
                            polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting."""
        inside = False
        n = len(polygon)

        j = n - 1
        for i in range(n):
            if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
               (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                    (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
                inside = not inside
            j = i

        return inside

    @staticmethod
    def _transform_point(x: float, y: float, trans_x: float,
                        trans_y: float, yaw: float) -> Tuple[float, float]:
        """Transform point by translation and rotation."""
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        temp_x = x * cos_yaw - y * sin_yaw + trans_x
        temp_y = x * sin_yaw + y * cos_yaw + trans_y

        return temp_x, temp_y
