#!/usr/bin/env python3
"""
Geometry utilities for SMPPI
Polygon operations, collision detection, and footprint handling
"""

import numpy as np
import torch
import math
from typing import List, Tuple, Optional


class GeometryUtils:
    """
    Geometry utilities for polygon-based collision detection
    """
    
    @staticmethod
    def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        Ray casting algorithm for point-in-polygon test
        
        Args:
            point: [2] point (x, y)
            polygon: [N, 2] polygon vertices
            
        Returns:
            bool: True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def point_to_polygon_distance(point: np.ndarray, polygon: np.ndarray) -> float:
        """
        Minimum distance from point to polygon edge
        
        Args:
            point: [2] point (x, y)
            polygon: [N, 2] polygon vertices
            
        Returns:
            float: minimum distance to polygon edge
        """
        min_distance = float('inf')
        
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            
            # Distance from point to line segment
            distance = GeometryUtils.point_to_line_distance(point, p1, p2)
            min_distance = min(min_distance, distance)
        
        # Check if point is inside polygon (negative distance)
        if GeometryUtils.point_in_polygon(point, polygon):
            return -min_distance
        else:
            return min_distance
    
    @staticmethod
    def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        Distance from point to line segment
        
        Args:
            point: [2] point
            line_start: [2] line start
            line_end: [2] line end
            
        Returns:
            float: distance to line segment
        """
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        # Vector from line_start to point
        point_vec = point - line_start
        
        # Project point onto line
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-10:  # Degenerate line
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        
        # Clamp to line segment
        proj_length = max(0, min(line_len, proj_length))
        
        # Closest point on line segment
        closest_point = line_start + proj_length * line_unitvec
        
        return np.linalg.norm(point - closest_point)
    
    @staticmethod
    def expand_polygon(polygon: np.ndarray, padding: float) -> np.ndarray:
        """
        Expand polygon outward by padding distance (simple offset)
        
        Args:
            polygon: [N, 2] polygon vertices (counterclockwise)
            padding: padding distance
            
        Returns:
            expanded_polygon: [N, 2] expanded polygon vertices
        """
        if len(polygon) < 3:
            return polygon
        
        expanded_vertices = []
        n = len(polygon)
        
        for i in range(n):
            # Get three consecutive vertices
            prev_vertex = polygon[(i - 1) % n]
            curr_vertex = polygon[i]
            next_vertex = polygon[(i + 1) % n]
            
            # Calculate edge vectors
            edge1 = curr_vertex - prev_vertex
            edge2 = next_vertex - curr_vertex
            
            # Normalize edge vectors
            edge1_norm = edge1 / (np.linalg.norm(edge1) + 1e-10)
            edge2_norm = edge2 / (np.linalg.norm(edge2) + 1e-10)
            
            # Calculate normal vectors (perpendicular to edges, pointing outward)
            normal1 = np.array([-edge1_norm[1], edge1_norm[0]])  # Rotate 90 degrees
            normal2 = np.array([-edge2_norm[1], edge2_norm[0]])
            
            # Calculate bisector (average of normals)
            bisector = normal1 + normal2
            bisector_norm = bisector / (np.linalg.norm(bisector) + 1e-10)
            
            # Calculate offset distance (accounting for angle)
            angle = np.arccos(np.clip(np.dot(normal1, normal2), -1, 1))
            offset_distance = padding / np.sin(angle / 2 + 1e-10)
            
            # Limit excessive offsets for sharp angles
            offset_distance = min(offset_distance, padding * 3)
            
            # Calculate expanded vertex
            expanded_vertex = curr_vertex + offset_distance * bisector_norm
            expanded_vertices.append(expanded_vertex)
        
        return np.array(expanded_vertices)
    
    @staticmethod
    def transform_polygon(polygon: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
        """
        Transform polygon to world coordinates
        
        Args:
            polygon: [N, 2] polygon in local coordinates
            pose: (x, y, yaw) transformation
            
        Returns:
            transformed_polygon: [N, 2] polygon in world coordinates
        """
        x, y, yaw = pose
        
        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        # Transform each vertex
        transformed_vertices = []
        for vertex in polygon:
            # Rotate
            rotated_vertex = rotation_matrix @ vertex
            # Translate
            transformed_vertex = rotated_vertex + np.array([x, y])
            transformed_vertices.append(transformed_vertex)
        
        return np.array(transformed_vertices)
    
    @staticmethod
    def footprint_to_polygon(footprint_flat: List[float]) -> np.ndarray:
        """
        Convert footprint from flat list to polygon array
        
        Args:
            footprint_flat: [x1, y1, x2, y2, x3, y3, x4, y4] format
            
        Returns:
            polygon: [N, 2] polygon vertices
        """
        if len(footprint_flat) % 2 != 0:
            raise ValueError("Footprint must have even number of coordinates")
        
        polygon = []
        for i in range(0, len(footprint_flat), 2):
            polygon.append([footprint_flat[i], footprint_flat[i + 1]])
        
        return np.array(polygon)
    
    @staticmethod
    def polygon_to_footprint(polygon: np.ndarray) -> List[float]:
        """
        Convert polygon array to flat footprint list
        
        Args:
            polygon: [N, 2] polygon vertices
            
        Returns:
            footprint_flat: [x1, y1, x2, y2, ...] format
        """
        footprint_flat = []
        for vertex in polygon:
            footprint_flat.extend([float(vertex[0]), float(vertex[1])])
        
        return footprint_flat
    
    @staticmethod
    def create_robot_footprint_at_pose(footprint: List[float], pose: Tuple[float, float, float], 
                                     padding: float = 0.0) -> np.ndarray:
        """
        Create robot footprint polygon at given pose with optional padding
        
        Args:
            footprint: [x1, y1, x2, y2, ...] base footprint
            pose: (x, y, yaw) robot pose
            padding: additional padding around footprint
            
        Returns:
            footprint_polygon: [N, 2] footprint at pose
        """
        # Convert to polygon
        base_polygon = GeometryUtils.footprint_to_polygon(footprint)
        
        # Apply padding if needed
        if padding > 0:
            base_polygon = GeometryUtils.expand_polygon(base_polygon, padding)
        
        # Transform to world coordinates
        world_polygon = GeometryUtils.transform_polygon(base_polygon, pose)
        
        return world_polygon


class TorchGeometryUtils:
    """
    GPU-accelerated geometry utilities for batch operations
    """
    
    @staticmethod
    def batch_point_to_polygon_distance(points: torch.Tensor, polygon: torch.Tensor, 
                                      device: torch.device) -> torch.Tensor:
        """
        Batch distance calculation from points to polygon
        
        Args:
            points: [K, T, 2] batch of trajectory points
            polygon: [N, 2] polygon vertices
            device: torch device
            
        Returns:
            distances: [K, T] distances to polygon
        """
        # This is a simplified implementation - in practice, you'd want 
        # a more optimized GPU kernel for polygon distance calculations
        
        # For now, use approximate circular distance
        # TODO: Implement proper polygon distance on GPU
        polygon_tensor = polygon.to(device)
        center = torch.mean(polygon_tensor, dim=0)  # Polygon centroid
        
        # Approximate with distance to centroid minus polygon "radius"
        max_radius = torch.max(torch.norm(polygon_tensor - center, dim=1))
        
        # Distance from points to polygon center
        distances_to_center = torch.norm(points - center, dim=-1)  # [K, T]
        
        # Approximate distance to polygon edge
        distances = distances_to_center - max_radius
        
        return distances