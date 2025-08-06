"""
Cost functions for MPPI controller
"""
import torch
import numpy as np
from typing import List, Tuple


class ObstacleAvoidanceCost:
    """Cost function for obstacle avoidance using laser scan data with vehicle footprint consideration"""
    
    def __init__(self, safety_radius=0.3, max_range=5.0, penalty_weight=1000.0, exponential_factor=2.0, device='cpu'):
        """
        Initialize obstacle avoidance cost
        
        Args:
            safety_radius (float): Minimum safe distance from obstacles (m)
            max_range (float): Maximum sensor range to consider (m)
            penalty_weight (float): Penalty multiplier for close obstacles
            exponential_factor (float): Steepness of exponential penalty
            device (str): PyTorch device
        """
        self.safety_radius = safety_radius
        self.max_range = max_range
        self.penalty_weight = penalty_weight
        self.exponential_factor = exponential_factor
        self.device = device
        
        # Store raw laser scan data (more efficient)
        self.laser_ranges = None
        self.laser_angles = None
        self.robot_pose = None  # [x, y, theta] in world frame
        self.last_update_time = 0
    
    def update_obstacles(self, obstacle_points):
        """
        Update obstacle points from laser scan
        
        Args:
            obstacle_points (torch.Tensor): Obstacle points in robot frame (N x 2)
        """
        import time
        current_time = time.time()
        
        if obstacle_points is not None and len(obstacle_points) > 0:
            self.obstacle_points = torch.tensor(obstacle_points, 
                                              dtype=torch.float32, 
                                              device=self.device)
        else:
            self.obstacle_points = None
            
        self.last_update_time = current_time
    
    def update_laser_scan(self, laser_msg, robot_pose):
        """
        Update laser scan data directly (more efficient)
        
        Args:
            laser_msg: LaserScan ROS message
            robot_pose: Current robot pose [x, y, theta] in world frame
        """
        import time
        current_time = time.time()
        
        ranges = np.array(laser_msg.ranges)
        valid_mask = (ranges >= 0.1) & (ranges <= self.max_range) & np.isfinite(ranges)
        
        if np.any(valid_mask):
            valid_ranges = ranges[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            valid_angles = laser_msg.angle_min + valid_indices * laser_msg.angle_increment
            
            # Store as tensors
            self.laser_ranges = torch.tensor(valid_ranges, dtype=torch.float32, device=self.device)
            self.laser_angles = torch.tensor(valid_angles, dtype=torch.float32, device=self.device)
            self.robot_pose = torch.tensor(robot_pose, dtype=torch.float32, device=self.device)
            
        else:
            self.laser_ranges = None
            self.laser_angles = None
            
        self.last_update_time = current_time
    
    def check_footprint_collision(self, state, obstacle_points):
        """
        Fast approximate collision check using enlarged point-robot model
        
        Args:
            state (torch.Tensor): Robot states [x, y, theta] (K x 3)
            obstacle_points (torch.Tensor): Obstacle points in world frame (N x 2)
            
        Returns:
            torch.Tensor: Minimum distances for each state (K,)
        """
        batch_size = state.shape[0]
        
        if obstacle_points.size(0) == 0:
            return torch.full((batch_size,), float('inf'), device=self.device)
        
        # Fast approximation: Use enlarged point-robot model
        # Add half of vehicle diagonal as safety margin
        if self.footprint is not None:
            vehicle_radius = max(self.vehicle_length, self.vehicle_width) * 0.5
        else:
            vehicle_radius = 0.5  # Default 50cm radius
        
        # Calculate distances from robot center to all obstacles (vectorized)
        robot_pos = state[:, :2].unsqueeze(1)  # (K x 1 x 2)
        obs_pos = obstacle_points.unsqueeze(0)  # (1 x N x 2)
        distances = torch.norm(robot_pos - obs_pos, dim=2)  # (K x N)
        
        # Subtract vehicle radius to get approximate distance to footprint edge
        distances = distances - vehicle_radius
        distances = torch.clamp(distances, min=0.0)  # Ensure non-negative
        
        min_distances, _ = torch.min(distances, dim=1)  # (K,)
        return min_distances
    
    def _point_to_polygon_distance(self, points, polygon):
        """
        Calculate minimum distance from points to polygon edges
        
        Args:
            points (torch.Tensor): Points (N x 2)
            polygon (torch.Tensor): Polygon vertices (M x 2)
            
        Returns:
            torch.Tensor: Distances from each point to polygon (N,)
        """
        num_points = points.shape[0]
        num_vertices = polygon.shape[0]
        min_distances = torch.full((num_points,), float('inf'), device=self.device)
        
        # Check distance to each edge of the polygon
        for i in range(num_vertices):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % num_vertices]
            
            # Distance from points to line segment p1-p2
            edge_distances = self._point_to_line_segment_distance(points, p1, p2)
            min_distances = torch.min(min_distances, edge_distances)
        
        return min_distances
    
    def _point_to_line_segment_distance(self, points, p1, p2):
        """
        Calculate distance from points to line segment
        
        Args:
            points (torch.Tensor): Points (N x 2)
            p1, p2 (torch.Tensor): Line segment endpoints (2,)
            
        Returns:
            torch.Tensor: Distances (N,)
        """
        # Vector from p1 to p2
        line_vec = p2 - p1
        line_len_sq = torch.sum(line_vec * line_vec)
        
        if line_len_sq < 1e-8:
            # Degenerate case: p1 and p2 are the same
            return torch.norm(points - p1.unsqueeze(0), dim=1)
        
        # Vector from p1 to each point
        point_vecs = points - p1.unsqueeze(0)
        
        # Project points onto line
        t = torch.sum(point_vecs * line_vec.unsqueeze(0), dim=1) / line_len_sq
        t = torch.clamp(t, 0.0, 1.0)  # Clamp to line segment
        
        # Find closest point on line segment
        closest_points = p1.unsqueeze(0) + t.unsqueeze(1) * line_vec.unsqueeze(0)
        
        # Calculate distances
        distances = torch.norm(points - closest_points, dim=1)
        return distances

    def compute_laser_cost(self, state, action):
        """
        Compute obstacle cost directly from laser scan (vectorized, efficient)
        
        Args:
            state (torch.Tensor): Robot states [x, y, theta] (K x 3)
            action (torch.Tensor): Control actions [v, w] (K x 2)
            
        Returns:
            torch.Tensor: Obstacle costs (K,)
        """
        batch_size = state.shape[0]
        costs = torch.zeros(batch_size, device=self.device)
        
        if self.laser_ranges is None or self.laser_angles is None:
            return costs
        
        # Get global obstacle positions (from current robot pose)
        if self.robot_pose is None:
            return costs
            
        current_robot_x, current_robot_y, current_robot_theta = self.robot_pose[0], self.robot_pose[1], self.robot_pose[2]
        
        # Transform obstacles to global frame using current robot pose
        global_angles = self.laser_angles + current_robot_theta
        obs_x_global = current_robot_x + self.laser_ranges * torch.cos(global_angles)  # (N,)
        obs_y_global = current_robot_y + self.laser_ranges * torch.sin(global_angles)  # (N,)
        
        # Calculate distances from each candidate state to all obstacles
        candidate_positions = state[:, :2]  # (K x 2)
        obstacle_positions = torch.stack([obs_x_global, obs_y_global], dim=1)  # (N x 2)
        
        # Compute distances: (K x N) - distance from each candidate to each obstacle
        distances = torch.cdist(candidate_positions, obstacle_positions)
        
        # Find minimum distance for each state: (K,)
        min_distances, _ = torch.min(distances, dim=1)
        
        # Apply simpler, more focused obstacle penalty
        # Only penalize obstacles within safety zone (much smaller range)
        danger_zone = self.safety_radius * 1.5  # 1.2m instead of 2.4m
        
        # Smooth exponential penalty only for close obstacles
        close_mask = min_distances < danger_zone
        if torch.any(close_mask):
            # Smooth penalty that doesn't go to infinity
            penalty_factor = torch.clamp(danger_zone - min_distances[close_mask], 0, danger_zone) / danger_zone
            smooth_penalty = penalty_factor ** self.exponential_factor
            costs[close_mask] = smooth_penalty * self.penalty_weight
        
        # Debug information (every 50 calls)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter % 50 == 0:
            if len(min_distances) > 0:
                max_cost = torch.max(costs)
                min_dist = torch.min(min_distances)
                num_obstacles = len(self.laser_ranges) if self.laser_ranges is not None else 0
                
                # # 상세 디버그 정보
                # print(f"[OBSTACLE DEBUG] Min distance: {min_dist:.3f}m, Max cost: {max_cost:.1f}, "
                #       f"Obstacles: {num_obstacles}, Safety radius: {self.safety_radius:.2f}m")
                # print(f"[DEBUG] Robot pose: [{self.robot_pose[0]:.2f}, {self.robot_pose[1]:.2f}, {self.robot_pose[2]:.2f}]")
                # print(f"[DEBUG] First 3 candidates: {state[:3, :2]}")
                # print(f"[DEBUG] Laser min range: {torch.min(self.laser_ranges):.3f}m")
                # print(f"[DEBUG] Danger zone: {danger_zone:.3f}m")
                # print(f"[DEBUG] Close obstacles count: {torch.sum(close_mask)}")
                # if torch.sum(close_mask) > 0:
                #     print(f"[DEBUG] Close distances: {min_distances[close_mask][:3]}")  # 처음 3개
        
        return costs
    
    def __call__(self, state, action):
        """
        Compute obstacle avoidance cost
        
        Args:
            state (torch.Tensor): Robot states [x, y, theta] (K x 3)
            action (torch.Tensor): Control actions [v, w] (K x 2)
            
        Returns:
            torch.Tensor: Obstacle costs (K,) - Note: flattened output
        """
        import time
        current_time = time.time()
        
        # Use direct laser scan method if available (more efficient)
        if self.laser_ranges is not None and self.laser_angles is not None:
            return self.compute_laser_cost(state, action)
        
        # Fallback to obstacle points method
        batch_size = state.shape[0]
        costs = torch.zeros(batch_size, device=self.device)
        
        if not hasattr(self, 'obstacle_points') or self.obstacle_points is None or len(self.obstacle_points) == 0:
            return costs

        # Extract position
        robot_pos = state[:, :2]  # (K x 2)
        
        # Compute distance to all obstacles for all states
        # robot_pos: (K x 2), obstacle_points: (N x 2)
        # distances: (K x N)
        distances = torch.cdist(robot_pos, self.obstacle_points)
        
        # Find minimum distance to any obstacle for each state
        min_distances, _ = torch.min(distances, dim=1)  # (K,)
        
        # Apply exponential penalty for close obstacles
        penalty_mask = min_distances < self.safety_radius * 2.0
        costs[penalty_mask] = torch.exp(-min_distances[penalty_mask] / self.safety_radius) * 100.0
        
        # High penalty for collision
        collision_mask = min_distances < self.safety_radius
        costs[collision_mask] = 1000.0
        
        return costs


class GoalTrackingCost:
    """Cost function for tracking a goal position"""
    
    def __init__(self, goal_weight=1.0, angle_weight=0.5, device='cpu'):
        """
        Initialize goal tracking cost
        
        Args:
            goal_weight (float): Weight for position error
            angle_weight (float): Weight for orientation error
            device (str): PyTorch device
        """
        self.goal_weight = goal_weight
        self.angle_weight = angle_weight
        self.device = device
        self.goal_pose = None
    
    def set_goal(self, goal_pose):
        """
        Set goal pose [x, y, theta]
        
        Args:
            goal_pose (list): Goal pose [x, y, theta]
        """
        self.goal_pose = torch.tensor(goal_pose, dtype=torch.float32, device=self.device)
    
    def __call__(self, state, action):
        """
        Compute goal tracking cost
        
        Args:
            state (torch.Tensor): Robot states [x, y, theta] (K x 3)
            action (torch.Tensor): Control actions (K x 2)
            
        Returns:
            torch.Tensor: Goal tracking costs (K,)
        """
        batch_size = state.shape[0]
        costs = torch.zeros(batch_size, device=self.device)
        
        if self.goal_pose is None:
            return costs
        
        # Position error
        pos_error = torch.norm(state[:, :2] - self.goal_pose[:2], dim=1)
        
        # Angle error
        angle_diff = state[:, 2] - self.goal_pose[2]
        angle_error = torch.abs(torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)))
        
        # Simple goal tracking without forward reward (removed to prevent cliff effects)
        # The goal_weight provides sufficient attraction to the target
        costs = self.goal_weight * pos_error + self.angle_weight * angle_error
        
        return costs


class ControlEffortCost:
    """Cost function for penalizing control effort"""
    
    def __init__(self, linear_weight=0.1, angular_weight=0.1, device='cpu'):
        """
        Initialize control effort cost
        
        Args:
            linear_weight (float): Weight for linear velocity cost
            angular_weight (float): Weight for angular velocity cost
            device (str): PyTorch device
        """
        self.linear_weight = linear_weight
        self.angular_weight = angular_weight
        self.device = device
    
    def __call__(self, state, action):
        """
        Compute control effort cost with stopping penalty
        
        Args:
            state (torch.Tensor): Robot states (K x 3)
            action (torch.Tensor): Control actions [v, w] (K x 2)
            
        Returns:
            torch.Tensor: Control effort costs (K,)
        """
        linear_cost = self.linear_weight * torch.abs(action[:, 0])
        angular_cost = self.angular_weight * torch.abs(action[:, 1])
        
        # Remove stopping penalty - let obstacle avoidance handle it
        total_cost = linear_cost + angular_cost
        return total_cost


class MotionDirectionCost:
    """Cost function for controlling motion direction (forward/reverse preference)"""
    
    def __init__(self, allow_reverse=True, reverse_penalty_weight=10.0, 
                 min_forward_speed_preference=0.2, reverse_max_speed=-1.0, device='cpu'):
        """
        Initialize motion direction cost
        
        Args:
            allow_reverse (bool): Whether to allow reverse motion
            reverse_penalty_weight (float): Penalty weight for reverse motion
            min_forward_speed_preference (float): Speed below which forward motion is preferred
            reverse_max_speed (float): Maximum allowed reverse speed (negative value)
            device (str): PyTorch device
        """
        self.allow_reverse = allow_reverse
        self.reverse_penalty_weight = reverse_penalty_weight
        self.min_forward_speed_preference = min_forward_speed_preference
        self.reverse_max_speed = reverse_max_speed
        self.device = device
        
        # Status tracking for debugging
        self.last_reverse_count = 0
        self.last_forward_slow_count = 0
    
    def __call__(self, state, action):
        """
        Compute motion direction cost
        
        Args:
            state (torch.Tensor): Robot states (K x 3)
            action (torch.Tensor): Control actions [v, w/delta] (K x 2)
            
        Returns:
            torch.Tensor: Motion direction costs (K,)
        """
        velocity = action[:, 0]  # Extract linear velocity
        costs = torch.zeros_like(velocity, device=self.device)
        
        if not self.allow_reverse:
            # Complete prohibition of reverse motion
            backward_mask = velocity < 0.0
            costs[backward_mask] = 1000.0  # High penalty for reverse
            self.last_reverse_count = int(torch.sum(backward_mask))
        else:
            # Penalize reverse motion with configurable weight
            backward_mask = velocity < 0.0
            if torch.any(backward_mask):
                # Apply penalty proportional to reverse speed
                reverse_speeds = torch.abs(velocity[backward_mask])
                costs[backward_mask] = self.reverse_penalty_weight * reverse_speeds
            
            self.last_reverse_count = int(torch.sum(backward_mask))
            
            # Limit excessive reverse speed
            too_fast_reverse = velocity < self.reverse_max_speed
            if torch.any(too_fast_reverse):
                costs[too_fast_reverse] += 500.0  # High penalty for too fast reverse
            
            # Encourage forward motion at low speeds
            if self.min_forward_speed_preference > 0.0:
                slow_forward_mask = (velocity >= 0.0) & (velocity < self.min_forward_speed_preference)
                if torch.any(slow_forward_mask):
                    # Gentle encouragement for faster forward motion
                    forward_bonus = (self.min_forward_speed_preference - velocity[slow_forward_mask]) * 2.0
                    costs[slow_forward_mask] += forward_bonus
                
                self.last_forward_slow_count = int(torch.sum(slow_forward_mask))
        
        return costs
    
    def get_debug_info(self):
        """Get debugging information about motion constraints"""
        return {
            'allow_reverse': self.allow_reverse,
            'reverse_penalty_weight': self.reverse_penalty_weight,
            'last_reverse_samples': self.last_reverse_count,
            'last_slow_forward_samples': self.last_forward_slow_count
        }


class CombinedCostFunction:
    """Combined cost function that includes all individual costs"""
    
    def __init__(self, device='cpu', obstacle_params=None, motion_params=None):
        """
        Initialize combined cost function
        
        Args:
            device (str): PyTorch device
            obstacle_params (dict): Parameters for obstacle avoidance cost
            motion_params (dict): Parameters for motion direction cost
        """
        self.device = device
        
        # Initialize obstacle cost with custom parameters
        if obstacle_params:
            self.obstacle_cost = ObstacleAvoidanceCost(
                safety_radius=obstacle_params.get('safety_radius', 0.3),
                max_range=obstacle_params.get('max_range', 5.0),
                penalty_weight=obstacle_params.get('penalty_weight', 1000.0),
                exponential_factor=obstacle_params.get('exponential_factor', 2.0),
                device=device
            )
        else:
            self.obstacle_cost = ObstacleAvoidanceCost(device=device)
            
        self.goal_cost = GoalTrackingCost(device=device)
        self.control_cost = ControlEffortCost(device=device)
        
        # Initialize motion direction cost with custom parameters
        if motion_params:
            self.motion_cost = MotionDirectionCost(
                allow_reverse=motion_params.get('allow_reverse', True),
                reverse_penalty_weight=motion_params.get('reverse_penalty_weight', 10.0),
                min_forward_speed_preference=motion_params.get('min_forward_speed_preference', 0.2),
                reverse_max_speed=motion_params.get('reverse_max_speed', -1.0),
                device=device
            )
        else:
            self.motion_cost = MotionDirectionCost(device=device)
    
    def update_obstacles(self, obstacle_points):
        """Update obstacles for avoidance cost"""
        self.obstacle_cost.update_obstacles(obstacle_points)
    
    def update_laser_scan(self, laser_msg, robot_pose):
        """Update laser scan for efficient obstacle avoidance"""
        self.obstacle_cost.update_laser_scan(laser_msg, robot_pose)
    
    def set_goal(self, goal_pose):
        """Set goal for tracking cost"""
        self.goal_cost.set_goal(goal_pose)
    
    def __call__(self, state, action):
        """
        Compute total cost
        
        Args:
            state (torch.Tensor): Robot states (K x 3)
            action (torch.Tensor): Control actions (K x 2)
            
        Returns:
            torch.Tensor: Total costs (K,)
        """
        obstacle_cost = self.obstacle_cost(state, action)
        goal_cost = self.goal_cost(state, action)
        control_cost = self.control_cost(state, action)
        motion_cost = self.motion_cost(state, action)
        
        total_cost = obstacle_cost + goal_cost + control_cost + motion_cost
        
        return total_cost