#!/usr/bin/env python3
"""
Goal Tracking Critic for SMPPI
Tracks /goal_pose for navigation
"""

import torch
from typing import Optional, Any

from .base_critic import BaseCritic


class GoalCritic(BaseCritic):
    """
    Goal tracking critic for navigation
    """
    
    def __init__(self, params: dict):
        """Initialize goal critic"""
        super().__init__("GoalCritic", params)
        
        # Goal tracking parameters
        self.xy_goal_tolerance = params.get('xy_goal_tolerance', 0.25)
        self.yaw_goal_tolerance = params.get('yaw_goal_tolerance', 0.25)
        self.distance_scale = params.get('distance_scale', 1.0)
        self.angle_scale = params.get('angle_scale', 1.0)
        
        # Lookahead parameters
        self.lookahead_base_distance = params.get('lookahead_base_distance', 2.5)
        self.lookahead_velocity_factor = params.get('lookahead_velocity_factor', 1.2)
        self.lookahead_min_distance = params.get('lookahead_min_distance', 1.0)
        self.lookahead_max_distance = params.get('lookahead_max_distance', 6.0)
        
        # Multiple waypoints support
        self.multiple_waypoints = None
        self.use_multiple_waypoints = params.get('use_multiple_waypoints', True)
        
        print(f"[GoalCritic] xy_tolerance={self.xy_goal_tolerance}, yaw_tolerance={self.yaw_goal_tolerance}")
        print(f"[GoalCritic] Lookahead: base={self.lookahead_base_distance}m, vel_factor={self.lookahead_velocity_factor}s, range=[{self.lookahead_min_distance}-{self.lookahead_max_distance}]m")
        print(f"[GoalCritic] Multiple waypoints enabled: {self.use_multiple_waypoints}")
    
    def compute_cost(self, trajectories: torch.Tensor, controls: torch.Tensor,
                    robot_state: torch.Tensor, goal_state: Optional[torch.Tensor],
                    obstacles: Optional[Any]) -> torch.Tensor:
        """
        Args:
            trajectories: [K, T+1, 3] trajectory states (x, y, theta)
            controls: [K, T, 2] control sequences
            robot_state: [5] current robot state
            goal_state: [3] goal state (x, y, theta) or None
            obstacles: Not used for goal critic
            
        Returns:
            costs: [K] goal costs for each trajectory
        """
        if not self.enabled or goal_state is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)

        # Device/dtype alignment
        trajectories = trajectories.to(self.device, self.dtype)
        robot_state = robot_state.to(self.device, self.dtype)
        goal_state = goal_state.to(self.device, self.dtype)

        K, T_plus_1, _ = trajectories.shape
        current_pos = robot_state[:2]
        current_vel = torch.abs(robot_state[3])
        goal_pos = goal_state[:2]

        # Multiple waypoints lookahead calculation
        lookahead_point = self._compute_multiple_waypoints_lookahead(current_pos, current_vel, goal_pos)
        
        # Store lookahead for visualization
        self.last_lookahead_point = lookahead_point.clone()
        
        # Direction to lookahead point
        lookahead_vec = lookahead_point - current_pos
        lookahead_dist = torch.norm(lookahead_vec) + 1e-9
        target_direction = lookahead_vec / lookahead_dist
        target_yaw = torch.atan2(target_direction[1], target_direction[0])

        # Trajectory analysis
        traj_positions = trajectories[:, :, :2]  # [K, T+1, 2]
        traj_yaws = trajectories[:, :, 2]        # [K, T+1]

        # 1. Lookahead tracking cost - weighted average distance
        weights = torch.linspace(0.3, 1.0, steps=T_plus_1, device=self.device, dtype=self.dtype)
        weights = weights / weights.sum()
        
        distances_to_lookahead = torch.norm(
            traj_positions - lookahead_point.view(1, 1, 2), dim=2
        )  # [K, T+1]
        
        lookahead_cost = (distances_to_lookahead * weights.view(1, -1)).sum(dim=1)  # [K]

        # 2. Heading alignment cost
        final_yaws = traj_yaws[:, -1]  # [K]
        yaw_errors = self.normalize_angle(final_yaws - target_yaw)
        heading_cost = yaw_errors ** 2

        # 3. Path alignment cost
        if T_plus_1 > 1:
            trajectory_steps = traj_positions[:, 1:, :] - traj_positions[:, :-1, :]  # [K, T, 2]
            step_norms = torch.norm(trajectory_steps, dim=2, keepdim=True).clamp_min(1e-6)
            step_directions = trajectory_steps / step_norms  # [K, T, 2]
            
            # Alignment with target direction
            alignment = torch.sum(
                step_directions * target_direction.view(1, 1, 2), dim=2
            )  # [K, T]
            alignment_cost = (1.0 - alignment).mean(dim=1)  # [K]
        else:
            alignment_cost = torch.zeros(K, device=self.device, dtype=self.dtype)

        # Normalized costs
        lookahead_normalized = lookahead_cost / (self.xy_goal_tolerance + 1e-6)
        heading_normalized = torch.abs(yaw_errors) / (self.yaw_goal_tolerance + 1e-6)

        # Combined cost
        total_cost = (
            0.5 * lookahead_normalized +
            0.3 * heading_normalized +
            0.2 * alignment_cost
        )

        return self.apply_weight(total_cost)
    
    def _compute_nav2_lookahead(self, current_pos: torch.Tensor, current_vel: torch.Tensor, 
                               goal_pos: torch.Tensor) -> torch.Tensor:
        # Base lookahead distance (configurable)
        base_offset = self.lookahead_base_distance
        
        # Velocity-based additional lookahead (configurable factor)
        velocity_offset = self.lookahead_velocity_factor * current_vel
        
        # Total lookahead distance
        total_lookahead = base_offset + velocity_offset
        
        # Clamp lookahead distance (configurable limits)
        total_lookahead = torch.clamp(total_lookahead, self.lookahead_min_distance, self.lookahead_max_distance)
        
        # Direction to goal
        goal_vec = goal_pos - current_pos
        goal_distance = torch.norm(goal_vec) + 1e-9
        
        if (goal_distance <= total_lookahead).item():
            return goal_pos
        else:
            # Place lookahead point along the goal direction
            goal_direction = goal_vec / goal_distance
            lookahead_point = current_pos + goal_direction * total_lookahead
            return lookahead_point
    
    def get_lookahead_point(self) -> Optional[torch.Tensor]:
        """Get the current lookahead point for visualization"""
        return getattr(self, 'last_lookahead_point', None)
    
    def set_multiple_waypoints(self, waypoints_msg):
        """Set multiple waypoints for advanced lookahead"""
        self.multiple_waypoints = waypoints_msg
    
    def _compute_multiple_waypoints_lookahead(self, current_pos: torch.Tensor, current_vel: torch.Tensor, 
                                            goal_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute lookahead point using multiple waypoints for smooth path following
        Prevents speed reduction near intermediate goals by considering next waypoints
        """
        # If multiple waypoints are available and enabled, use them
        if self.use_multiple_waypoints and self.multiple_waypoints is not None:
            return self._compute_waypoint_chain_lookahead(current_pos, current_vel)
        else:
            # Fallback to single goal lookahead
            return self._compute_nav2_lookahead(current_pos, current_vel, goal_pos)
    
    def _compute_waypoint_chain_lookahead(self, current_pos: torch.Tensor, current_vel: torch.Tensor) -> torch.Tensor:
        """
        Compute lookahead using waypoint chain for continuous smooth motion
        """
        waypoints = self.multiple_waypoints
        
        # Current goal position
        current_goal_pos = torch.tensor([
            waypoints.current_goal.pose.position.x,
            waypoints.current_goal.pose.position.y
        ], device=self.device, dtype=self.dtype)
        
        # Calculate lookahead distance
        base_distance = self.lookahead_base_distance
        velocity_distance = self.lookahead_velocity_factor * current_vel
        total_lookahead = torch.clamp(
            base_distance + velocity_distance, 
            self.lookahead_min_distance, 
            self.lookahead_max_distance
        )
        
        # Distance to current goal
        distance_to_current = torch.norm(current_pos - current_goal_pos)
        
        # If current goal is far enough, use standard lookahead
        if (distance_to_current > total_lookahead).item():
            direction = (current_goal_pos - current_pos) / (distance_to_current + 1e-9)
            return current_pos + direction * total_lookahead
        
        # Current goal is close - use waypoint chain for extended lookahead
        if len(waypoints.next_waypoints) > 0:
            # Calculate extended lookahead through waypoint chain
            remaining_distance = total_lookahead - distance_to_current
            
            # Start from current goal, extend to next waypoints
            extended_point = self._extend_lookahead_through_waypoints(
                current_goal_pos, remaining_distance, waypoints.next_waypoints
            )
            return extended_point
        else:
            # No next waypoints - this is final goal, use direct approach
            if waypoints.is_final_waypoint:
                return current_goal_pos
            else:
                # Use current goal as lookahead
                return current_goal_pos
    
    def _extend_lookahead_through_waypoints(self, start_pos: torch.Tensor, remaining_distance: float, 
                                          next_waypoints: list) -> torch.Tensor:
        """
        Extend lookahead through a chain of waypoints
        """
        current_pos = start_pos
        remaining = remaining_distance
        
        for i, waypoint in enumerate(next_waypoints):
            next_pos = torch.tensor([
                waypoint.pose.position.x,
                waypoint.pose.position.y
            ], device=self.device, dtype=self.dtype)
            
            segment_vec = next_pos - current_pos
            segment_length = torch.norm(segment_vec)
            
            if (remaining <= segment_length).item():
                # Lookahead point is within this segment
                if segment_length > 1e-9:
                    direction = segment_vec / segment_length
                    return current_pos + direction * remaining
                else:
                    return current_pos
            else:
                # Move to next waypoint and continue
                remaining = (remaining - segment_length).item() if torch.is_tensor(remaining) else (remaining - segment_length)
                current_pos = next_pos
        
        # If we've exhausted all waypoints, return the last waypoint
        return current_pos
    
    def is_goal_reached(self, current_pose: torch.Tensor, goal_state: torch.Tensor) -> bool:
        """
        Check if goal is reached
        
        Args:
            current_pose: [3] current pose (x, y, theta)
            goal_state: [3] goal state (x, y, theta)
            
        Returns:
            reached: True if goal is reached
        """
        if goal_state is None:
            return False
        
        # Distance check
        distance = torch.sqrt(torch.sum((current_pose[:2] - goal_state[:2]) ** 2))
        
        # Angle check
        angle_diff = abs(self.normalize_angle(current_pose[2] - goal_state[2]))
        
        return (distance < self.xy_goal_tolerance and 
                angle_diff < self.yaw_goal_tolerance)
    
    def compute_progress_cost(self, trajectories: torch.Tensor, 
                             goal_state: torch.Tensor) -> torch.Tensor:
        """
        Compute progress-based cost (reward getting closer to goal)
        
        Args:
            trajectories: [K, T+1, 3] trajectory states
            goal_state: [3] goal state
            
        Returns:
            costs: [K] progress costs (negative for progress)
        """
        if goal_state is None:
            return torch.zeros(trajectories.shape[0], device=self.device, dtype=self.dtype)
        
        # Initial distances
        initial_distances = torch.sqrt(torch.sum(
            (trajectories[:, 0, :2] - goal_state[:2]) ** 2, dim=1
        ))
        
        # Final distances  
        final_distances = torch.sqrt(torch.sum(
            (trajectories[:, -1, :2] - goal_state[:2]) ** 2, dim=1
        ))
        
        # Progress (negative cost for getting closer)
        progress = initial_distances - final_distances
        
        return -progress  # Negative for reward
    
    def update_parameters(self, params: dict):
        """Update goal critic parameters"""
        if 'xy_goal_tolerance' in params:
            self.xy_goal_tolerance = params['xy_goal_tolerance']
        if 'yaw_goal_tolerance' in params:  
            self.yaw_goal_tolerance = params['yaw_goal_tolerance']
        if 'distance_scale' in params:
            self.distance_scale = params['distance_scale']
        if 'angle_scale' in params:
            self.angle_scale = params['angle_scale']
            
        print(f"[GoalCritic] Parameters updated")