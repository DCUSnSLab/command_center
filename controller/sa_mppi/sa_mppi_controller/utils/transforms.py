#!/usr/bin/env python3
"""
Transform utilities for SMPPI
Coordinate transformations and geometric calculations
"""

import torch
import numpy as np
import math
from typing import Tuple, List

from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path


class Transforms:
    """
    Coordinate transformations and geometric utilities
    """
    
    @staticmethod
    def quaternion_to_yaw(quat) -> float:
        """Convert quaternion to yaw angle"""
        return math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        )
    
    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
        """Convert yaw angle to quaternion (w, x, y, z)"""
        half_yaw = yaw * 0.5
        w = math.cos(half_yaw)
        x = 0.0
        y = 0.0
        z = math.sin(half_yaw)
        return w, x, y, z
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return math.atan2(math.sin(angle), math.cos(angle))
    
    @staticmethod
    def normalize_angle_tensor(angles: torch.Tensor) -> torch.Tensor:
        """Normalize angles tensor to [-pi, pi]"""
        return torch.atan2(torch.sin(angles), torch.cos(angles))
    
    @staticmethod
    def euclidean_distance(pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    @staticmethod
    def angle_difference(angle1: float, angle2: float) -> float:
        """Compute normalized angle difference"""
        diff = angle1 - angle2
        return Transforms.normalize_angle(diff)
    
    @staticmethod
    def pose_to_tensor(pose: PoseStamped, device: torch.device, 
                      dtype: torch.dtype) -> torch.Tensor:
        """Convert PoseStamped to tensor [x, y, yaw]"""
        x = pose.pose.position.x
        y = pose.pose.position.y
        yaw = Transforms.quaternion_to_yaw(pose.pose.orientation)
        
        return torch.tensor([x, y, yaw], device=device, dtype=dtype)
    
    @staticmethod
    def twist_to_tensor(twist: Twist, device: torch.device,
                       dtype: torch.dtype) -> torch.Tensor:
        """Convert Twist to tensor [v_x, w_z]"""
        v_x = twist.linear.x
        w_z = twist.angular.z
        
        return torch.tensor([v_x, w_z], device=device, dtype=dtype)
    
    @staticmethod
    def tensor_to_twist(control_tensor: torch.Tensor) -> Twist:
        """Convert control tensor to Twist message"""
        twist = Twist()
        twist.linear.x = float(control_tensor[0])
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(control_tensor[1])
        
        return twist
    
    @staticmethod
    def path_to_tensor(path: Path, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
        """Convert nav_msgs/Path to tensor"""
        if not path.poses:
            return torch.empty(0, 3, device=device, dtype=dtype)
        
        poses = []
        for pose_stamped in path.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            yaw = Transforms.quaternion_to_yaw(pose_stamped.pose.orientation)
            poses.append([x, y, yaw])
        
        return torch.tensor(poses, device=device, dtype=dtype)
    
    @staticmethod
    def global_to_local(global_point: Tuple[float, float], 
                       robot_pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Transform global point to robot local coordinates
        
        Args:
            global_point: (x, y) in global frame
            robot_pose: (x, y, yaw) robot pose in global frame
            
        Returns:
            local_point: (x, y) in robot local frame
        """
        gx, gy = global_point
        rx, ry, ryaw = robot_pose
        
        # Translate to robot origin
        dx = gx - rx
        dy = gy - ry
        
        # Rotate to robot frame
        cos_yaw = math.cos(-ryaw)
        sin_yaw = math.sin(-ryaw)
        
        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw
        
        return local_x, local_y
    
    @staticmethod
    def local_to_global(local_point: Tuple[float, float],
                       robot_pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Transform robot local point to global coordinates
        
        Args:
            local_point: (x, y) in robot local frame
            robot_pose: (x, y, yaw) robot pose in global frame
            
        Returns:
            global_point: (x, y) in global frame
        """
        lx, ly = local_point
        rx, ry, ryaw = robot_pose
        
        # Rotate to global frame
        cos_yaw = math.cos(ryaw)
        sin_yaw = math.sin(ryaw)
        
        dx = lx * cos_yaw - ly * sin_yaw
        dy = lx * sin_yaw + ly * cos_yaw
        
        # Translate to global origin
        global_x = dx + rx
        global_y = dy + ry
        
        return global_x, global_y
    
    @staticmethod
    def compute_path_curvature(path_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute curvature along path
        
        Args:
            path_tensor: [N, 3] path points (x, y, yaw)
            
        Returns:
            curvatures: [N-2] curvature values
        """
        if path_tensor.shape[0] < 3:
            return torch.empty(0, device=path_tensor.device, dtype=path_tensor.dtype)
        
        # Use three consecutive points to compute curvature
        p1 = path_tensor[:-2, :2]  # [N-2, 2]
        p2 = path_tensor[1:-1, :2] # [N-2, 2]
        p3 = path_tensor[2:, :2]   # [N-2, 2]
        
        # Vectors
        v1 = p2 - p1  # [N-2, 2]
        v2 = p3 - p2  # [N-2, 2]
        
        # Cross product for 2D (scalar)
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # [N-2]
        
        # Magnitudes
        v1_mag = torch.norm(v1, dim=1)  # [N-2]
        v2_mag = torch.norm(v2, dim=1)  # [N-2]
        
        # Curvature: k = |v1 x v2| / |v1|^3
        # Approximation for small segments
        curvatures = torch.abs(cross) / (v1_mag ** 2 + 1e-6)
        
        return curvatures
    
    @staticmethod
    def find_closest_point_on_path(path_tensor: torch.Tensor, 
                                  point: torch.Tensor) -> Tuple[int, float]:
        """
        Find closest point on path to given point
        
        Args:
            path_tensor: [N, 3] path points
            point: [2] query point (x, y)
            
        Returns:
            index: Index of closest path point
            distance: Distance to closest point
        """
        if path_tensor.shape[0] == 0:
            return 0, float('inf')
        
        # Compute distances to all path points
        distances = torch.norm(path_tensor[:, :2] - point, dim=1)
        
        # Find minimum
        min_index = torch.argmin(distances)
        min_distance = distances[min_index]
        
        return int(min_index), float(min_distance)

    def omega_cap_from_v(v: float, L: float, delta_max: float, ay_max: float, v_min: float=0.1) -> float:
        v = max(abs(v), v_min)
        omega_geom = (v / L) * math.tan(delta_max)
        omega_fric = ay_max / v
        return min(omega_geom, omega_fric)

    def delta_cap_from_v(v: float, L: float, delta_max: float, ay_max: float, v_min: float=0.1) -> float:
        v = max(abs(v), v_min)
        # tan(δ) ≤ L*ay_max / v^2  →  δ ≤ atan(L*ay_max / v^2)
        dyn = math.atan((L * ay_max) / (v * v))
        return min(delta_max, dyn)