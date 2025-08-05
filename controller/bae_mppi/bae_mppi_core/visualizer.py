"""
RViz visualization for MPPI trajectories
"""
import numpy as np
import torch
from typing import List, Optional

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration


class MPPIVisualizer:
    """Visualizer for MPPI trajectories and paths"""
    
    def __init__(self, frame_id='odom'):
        """
        Initialize MPPI visualizer
        
        Args:
            frame_id (str): Reference frame for visualization
        """
        self.frame_id = frame_id
        self.marker_id = 0
    
    def create_trajectory_markers(self, trajectories, costs, num_best=30):
        """
        Create markers for best trajectories
        
        Args:
            trajectories (torch.Tensor): All trajectories (N x T x 3)
            costs (torch.Tensor): Costs for each trajectory (N,)
            num_best (int): Number of best trajectories to visualize
            
        Returns:
            MarkerArray: Trajectory markers
        """
        marker_array = MarkerArray()
        
        if trajectories is None or len(trajectories) == 0:
            return marker_array
        
        # Convert to numpy
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.detach().cpu().numpy()
        if isinstance(costs, torch.Tensor):
            costs = costs.detach().cpu().numpy()
        
        # Ensure dimensions match
        if len(trajectories) != len(costs):
            print(f"Warning: trajectory count ({len(trajectories)}) != cost count ({len(costs)})")
            return marker_array
        
        # Find best trajectories (safe indexing)
        num_valid = min(num_best, len(costs), len(trajectories))
        best_indices = np.argsort(costs)[:num_valid]
        
        # Create markers for best trajectories
        for i, traj_idx in enumerate(best_indices):
            # Double check bounds
            if traj_idx >= len(trajectories) or traj_idx < 0:
                continue
            trajectory = trajectories[traj_idx]  # (T x 3)
            
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp.sec = 0
            marker.header.stamp.nanosec = 0
            marker.ns = "mppi_trajectories"
            marker.id = self.marker_id
            self.marker_id += 1
            
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Set pose
            marker.pose.orientation.w = 1.0
            
            # Set scale
            marker.scale.x = 0.02  # Line width
            
            # Set color (blue with varying alpha)
            alpha = 0.3 + 0.7 * (num_best - i) / num_best  # Best trajectory is most opaque
            marker.color = ColorRGBA(r=0.0, g=0.4, b=1.0, a=alpha)
            
            # Set lifetime
            marker.lifetime = Duration(sec=0, nanosec=200000000)  # 0.2 seconds
            
            # Add trajectory points
            for point_idx in range(len(trajectory)):
                point = Point()
                point.x = float(trajectory[point_idx, 0])
                point.y = float(trajectory[point_idx, 1])
                point.z = 0.05  # Slightly above ground
                marker.points.append(point)
            
            marker_array.markers.append(marker)
        
        return marker_array
    
    def create_optimal_path_marker(self, optimal_trajectory):
        """
        Create marker for the optimal selected path
        
        Args:
            optimal_trajectory (torch.Tensor or np.ndarray): Optimal trajectory (T x 3)
            
        Returns:
            Marker: Optimal path marker
        """
        marker = Marker()
        
        if optimal_trajectory is None or len(optimal_trajectory) == 0:
            marker.action = Marker.DELETE
            return marker
        
        # Convert to numpy
        if isinstance(optimal_trajectory, torch.Tensor):
            optimal_trajectory = optimal_trajectory.detach().cpu().numpy()
        
        marker.header.frame_id = self.frame_id
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0
        marker.ns = "mppi_optimal_path"
        marker.id = 0
        
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Set pose
        marker.pose.orientation.w = 1.0
        
        # Set scale
        marker.scale.x = 0.05  # Thicker line for optimal path
        
        # Set color (bright red)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        
        # Set lifetime
        marker.lifetime = Duration(sec=1, nanosec=0)  # 1.0 seconds for better visibility
        
        # Add trajectory points
        for point_idx in range(len(optimal_trajectory)):
            point = Point()
            point.x = float(optimal_trajectory[point_idx, 0])
            point.y = float(optimal_trajectory[point_idx, 1])
            point.z = 0.1  # Above the blue trajectories
            marker.points.append(point)
        
        return marker
    
    def create_goal_marker(self, goal_pose):
        """
        Create marker for goal pose
        
        Args:
            goal_pose (list): Goal pose [x, y, theta]
            
        Returns:
            Marker: Goal marker
        """
        marker = Marker()
        
        if goal_pose is None:
            marker.action = Marker.DELETE
            return marker
        
        marker.header.frame_id = self.frame_id
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0
        marker.ns = "mppi_goal"
        marker.id = 0
        
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Set pose
        marker.pose.position.x = float(goal_pose[0])
        marker.pose.position.y = float(goal_pose[1])
        marker.pose.position.z = 0.2
        
        # Convert yaw to quaternion
        yaw = goal_pose[2]
        marker.pose.orientation.z = np.sin(yaw / 2.0)
        marker.pose.orientation.w = np.cos(yaw / 2.0)
        
        # Set scale
        marker.scale.x = 0.3  # Arrow length
        marker.scale.y = 0.05  # Arrow width
        marker.scale.z = 0.05  # Arrow height
        
        # Set color (green)
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        
        # Set lifetime
        marker.lifetime = Duration(sec=0, nanosec=500000000)  # 0.5 seconds
        
        return marker
    
    def create_obstacle_markers(self, obstacle_points):
        """
        Create markers for detected obstacles
        
        Args:
            obstacle_points (np.ndarray): Obstacle points (N x 2)
            
        Returns:
            MarkerArray: Obstacle markers
        """
        marker_array = MarkerArray()
        
        if obstacle_points is None or len(obstacle_points) == 0:
            return marker_array
        
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0
        marker.ns = "mppi_obstacles"
        marker.id = 0
        
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Set pose
        marker.pose.orientation.w = 1.0
        
        # Set scale
        marker.scale.x = 0.1  # Point size
        marker.scale.y = 0.1
        
        # Set color (yellow)
        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
        
        # Set lifetime
        marker.lifetime = Duration(sec=1, nanosec=0)  # 1.0 seconds for better visibility
        
        # Add obstacle points
        for point_data in obstacle_points:
            point = Point()
            point.x = float(point_data[0])
            point.y = float(point_data[1])
            point.z = 0.05
            marker.points.append(point)
        
        marker_array.markers.append(marker)
        return marker_array
    
    def reset_marker_id(self):
        """Reset marker ID counter"""
        self.marker_id = 0