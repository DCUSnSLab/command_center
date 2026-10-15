#!/usr/bin/env python3
"""
SMPPI Visualization Node
Handles all RViz visualization without impacting control performance
Subscribes to processed data, publishes markers and visualization
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import time

import numpy as np
from typing import Optional

# ROS2 messages
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from smppi.msg import OptimalPath, MPPIDebug, RolloutSamples
from command_center_interfaces.msg import MultipleWaypoints

# TF2 imports
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# Geometry utils
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from smppi_controller.utils.geometry import GeometryUtils
from smppi_controller.critics.obstacle_critic import cover_footprint

# BEV palette: normal marks stay grey/white, colour only for state
_WHITE = (0.92, 0.92, 0.92)
_AMBER = (1.0, 0.70, 0.10)      # clearance shortfall (allowed, costs)
_RED = (0.92, 0.25, 0.20)       # collision (clearance < collision_margin)


def _rgba(rgb, a):
    return ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=a)


class _VizRobotState:
    """(x, y, yaw) 홀더 — 구 MPPIState.state_vector 접근 호환"""
    __slots__ = ('state_vector',)

    def __init__(self, xyyaw):
        self.state_vector = xyyaw


class VisualizationNode(Node):
    """
    Dedicated visualization node
    Lightweight, non-critical visualization without performance impact
    """
    
    def __init__(self):
        super().__init__('smppi_visualization')
        
        # Initialize parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize TF
        self._init_tf()
        
        # Setup ROS2 interfaces
        self._setup_topics()
        
        # State variables
        self.optimal_path: Optional[OptimalPath] = None
        self.latest_goal: Optional[PoseStamped] = None
        self.multiple_waypoints: Optional[MultipleWaypoints] = None
        self.robot_state = None  # (x, y, yaw) — /odom 직접 변환
        self.lookahead_point: Optional[PoseStamped] = None
        self.debug: Optional[MPPIDebug] = None
        self.rollouts: Optional[RolloutSamples] = None
        
        # Visualization timer (runs at lower frequency)
        self.visualization_timer = self.create_timer(
            1.0 / self.visualization_frequency,
            self.visualization_callback
        )
        
        self.get_logger().info("smppi visualization ready")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        # Topic parameters
        self.declare_parameter('topics.input.optimal_path', '/mppi_optimal_path')
        self.declare_parameter('topics.input.goal_pose', '/subgoal')
        self.declare_parameter('topics.input.multiple_waypoints', '/multiple_waypoints')
        self.declare_parameter('topics.output.markers', '/smppi_visualization')
        self.declare_parameter('topics.output.debug', '/smppi/debug')          # main node outputs
        self.declare_parameter('topics.output.rollouts', '/smppi/rollouts')
        
        # Waypoint mode
        self.declare_parameter('waypoint_mode', 'multiple')
        
        # Visualization parameters
        self.declare_parameter('visualization_frequency', 20.0)
        self.declare_parameter('enable_obstacles', True)
        self.declare_parameter('enable_trajectory', True)
        self.declare_parameter('enable_goal', True)
        self.declare_parameter('enable_robot_footprint', True)
        
        # Vehicle footprint parameters
        self.declare_parameter('vehicle.footprint', [0.0, 0.0])
        self.declare_parameter('vehicle.footprint_padding', 0.0)

        # Obstacle critic geometry/costs and goal tests (same yaml as the main node)
        self.declare_parameter('costs.clearance.collision_margin', 0.08)
        self.declare_parameter('costs.clearance.footprint_circles', 3)
        self.declare_parameter('costs.collision_cost', 1.0e6)
        self.declare_parameter('optimizer.time_steps', 60)
        self.declare_parameter('goal_reached_threshold', 3.0)
        self.declare_parameter('passed_goal_max_distance', 8.0)
        
        # QoS
        self.declare_parameter('qos.reliable_depth', 5)
        
    
    def _load_parameters(self):
        """Load parameters from ROS2 parameter server"""
        # Topic names
        self.path_topic = self.get_parameter('topics.input.optimal_path').get_parameter_value().string_value
        self.goal_topic = self.get_parameter('topics.input.goal_pose').get_parameter_value().string_value
        self.multiple_waypoints_topic = self.get_parameter('topics.input.multiple_waypoints').get_parameter_value().string_value
        self.markers_topic = self.get_parameter('topics.output.markers').get_parameter_value().string_value
        
        # Waypoint mode
        self.waypoint_mode = self.get_parameter('waypoint_mode').get_parameter_value().string_value
        if self.waypoint_mode not in ['single', 'multiple']:
            self.get_logger().warn(f"Invalid waypoint_mode '{self.waypoint_mode}', defaulting to 'multiple'")
            self.waypoint_mode = 'multiple'
        
        # Visualization parameters
        self.visualization_frequency = self.get_parameter('visualization_frequency').get_parameter_value().double_value
        self.enable_obstacles = self.get_parameter('enable_obstacles').get_parameter_value().bool_value
        self.enable_trajectory = self.get_parameter('enable_trajectory').get_parameter_value().bool_value
        self.enable_goal = self.get_parameter('enable_goal').get_parameter_value().bool_value
        self.enable_robot_footprint = self.get_parameter('enable_robot_footprint').get_parameter_value().bool_value
        
        # Vehicle footprint parameters
        self.footprint = self.get_parameter('vehicle.footprint').get_parameter_value().double_array_value
        self.footprint_padding = self.get_parameter('vehicle.footprint_padding').get_parameter_value().double_value

        self.debug_topic = self.get_parameter('topics.output.debug').get_parameter_value().string_value
        self.rollouts_topic = self.get_parameter('topics.output.rollouts').get_parameter_value().string_value
        self.collision_margin = self.get_parameter('costs.clearance.collision_margin').get_parameter_value().double_value
        n_circles = self.get_parameter('costs.clearance.footprint_circles').get_parameter_value().integer_value
        self.circles, self.circle_radius = cover_footprint(list(self.footprint), n_circles) \
            if len(self.footprint) >= 6 else ([], 0.0)
        # a rollout costs at least collision_cost / T once it collides (graded severity floor)
        self.collision_floor = self.get_parameter('costs.collision_cost').get_parameter_value().double_value / \
            max(1, self.get_parameter('optimizer.time_steps').get_parameter_value().integer_value)
        self.goal_reached_threshold = self.get_parameter('goal_reached_threshold').get_parameter_value().double_value
        self.passed_goal_max_distance = self.get_parameter('passed_goal_max_distance').get_parameter_value().double_value
        
        # QoS parameters
        self.reliable_qos_depth = self.get_parameter('qos.reliable_depth').get_parameter_value().integer_value
    
    def _init_tf(self):
        """Initialize TF listener"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().debug("TF listener initialized for visualization")
    
    def _setup_topics(self):
        """Setup ROS2 topics"""
        # QoS profiles
        reliable_qos = QoSProfile(
            depth=self.reliable_qos_depth,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # Subscribers
        self.path_sub = self.create_subscription(
            OptimalPath, self.path_topic, self.path_callback, reliable_qos)
        # /odom 직구독 (구 MPPIState 릴레이 제거; 발행측 BEST_EFFORT)
        from rclpy.qos import QoSProfile as _QoS
        sensor_qos = _QoS(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.robot_state_sub = self.create_subscription(
            Odometry, '/odom', self.robot_state_callback, sensor_qos)
        self.lookahead_sub = self.create_subscription(
            PoseStamped, '/smppi_visualization/lookahead_point', self.lookahead_callback, reliable_qos)
        self.debug_sub = self.create_subscription(
            MPPIDebug, self.debug_topic, self.debug_callback, reliable_qos)
        self.rollouts_sub = self.create_subscription(
            RolloutSamples, self.rollouts_topic, self.rollouts_callback, reliable_qos)
        
        # Goal subscribers based on waypoint mode
        if self.waypoint_mode == 'single':
            self.goal_sub = self.create_subscription(
                PoseStamped, self.goal_topic, self.goal_callback, reliable_qos)
            self.get_logger().debug(f"Visualization single mode: subscribed to {self.goal_topic}")
        elif self.waypoint_mode == 'multiple':
            self.multiple_waypoints_sub = self.create_subscription(
                MultipleWaypoints, self.multiple_waypoints_topic, self.multiple_waypoints_callback, reliable_qos)
            self.get_logger().debug(f"Visualization multiple mode: subscribed to {self.multiple_waypoints_topic}")
        else:
            # Fallback - subscribe to both
            self.goal_sub = self.create_subscription(
                PoseStamped, self.goal_topic, self.goal_callback, reliable_qos)
            self.multiple_waypoints_sub = self.create_subscription(
                MultipleWaypoints, self.multiple_waypoints_topic, self.multiple_waypoints_callback, reliable_qos)
            self.get_logger().warn("Unknown waypoint mode for visualization, subscribing to both topics")
        
        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, self.markers_topic, reliable_qos)
        
        self.get_logger().debug(f"Visualization topics configured")
    
    
    def path_callback(self, msg: OptimalPath):
        """Receive optimal path"""
        self.optimal_path = msg
    
    def goal_callback(self, msg: PoseStamped):
        """Receive goal pose"""
        self.latest_goal = msg
    
    def multiple_waypoints_callback(self, msg: MultipleWaypoints):
        """Receive multiple waypoints"""
        self.multiple_waypoints = msg
        # Set latest goal from current goal for visualization compatibility
        self.latest_goal = msg.current_goal
    
    def robot_state_callback(self, msg: Odometry):
        """Receive /odom and keep (x, y, yaw) for markers"""
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.robot_state = _VizRobotState(
            (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw))
    
    def lookahead_callback(self, msg: PoseStamped):
        """Receive lookahead point"""
        self.lookahead_point = msg
    
    def debug_callback(self, msg: MPPIDebug):
        """Receive MPPI diagnostics"""
        self.debug = msg

    def rollouts_callback(self, msg: RolloutSamples):
        """Receive rollout samples"""
        self.rollouts = msg
    
    def visualization_callback(self):
        """Main visualization callback (runs at lower frequency)"""
        try:
            marker_array = MarkerArray()
            
            # Goal marker (목표점 진행방향 화살표) — BEV 정리로 비활성화
            # if self.enable_goal and self.latest_goal is not None:
            #     goal_marker = self.create_goal_marker()
            #     if goal_marker:
            #         marker_array.markers.append(goal_marker)
            
            # Optimal trajectory marker (coloured by clearance state when diagnostics arrive)
            if self.enable_trajectory and self.optimal_path is not None:
                traj_marker = self.create_trajectory_marker()
                if traj_marker:
                    marker_array.markers.append(traj_marker)

            # Sampled rollouts
            if self.rollouts is not None:
                marker_array.markers.extend(self.create_rollout_markers())
            
            # Lookahead point marker
            if self.lookahead_point is not None:
                lookahead_marker = self.create_lookahead_marker()
                if lookahead_marker:
                    marker_array.markers.append(lookahead_marker)
            
            # Multiple waypoints markers — BEV 정리로 비활성화
            # if self.waypoint_mode == 'multiple' and self.multiple_waypoints is not None:
            #     next_waypoint_markers = self.create_next_waypoints_markers()
            #     marker_array.markers.extend(next_waypoint_markers)

            # Goal reached radius + passed-plane (when smppi switches to the next waypoint)
            if self.latest_goal is not None:
                goal_region_marker = self.create_goal_region_marker()
                if goal_region_marker:
                    marker_array.markers.append(goal_region_marker)
            
            # Robot footprint (true outline) + the discs the obstacle critic actually checks
            if self.enable_robot_footprint and self.robot_state is not None and len(self.footprint) > 0:
                footprint_marker = self.create_footprint_marker()
                if footprint_marker:
                    marker_array.markers.append(footprint_marker)
                marker_array.markers.extend(self.create_footprint_circles_markers())

            # Cost breakdown / sampler state text next to the robot
            if self.debug is not None and self.robot_state is not None:
                text_marker = self.create_debug_text_marker()
                if text_marker:
                    marker_array.markers.append(text_marker)
            
            # Publish markers
            if len(marker_array.markers) > 0:
                self.marker_pub.publish(marker_array)
                
        except Exception as e:
            self.get_logger().warn(f"Visualization callback error: {str(e)}")
    
    def create_goal_marker(self) -> Optional[Marker]:
        """Create goal visualization marker"""
        if not self.latest_goal:
            return None
            
        goal_marker = Marker()
        goal_marker.header.frame_id = "odom"
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = "smppi_goal"
        goal_marker.id = 0
        goal_marker.type = Marker.ARROW
        goal_marker.action = Marker.ADD
        
        # Goal position and orientation
        goal_marker.pose = self.latest_goal.pose
        goal_marker.pose.position.z = 0.1
        
        # Arrow appearance
        goal_marker.scale.x = 0.8  # Arrow length
        goal_marker.scale.y = 0.1  # Arrow width
        goal_marker.scale.z = 0.1  # Arrow height
        goal_marker.color.r = 0.95
        goal_marker.color.g = 0.95
        goal_marker.color.b = 0.95
        goal_marker.color.a = 0.8
        
        return goal_marker
    
    def create_trajectory_marker(self) -> Optional[Marker]:
        """Create trajectory visualization marker"""
        if not self.optimal_path or not self.optimal_path.path_points:
            return None
            
        traj_marker = Marker()
        traj_marker.header.frame_id = "odom"
        traj_marker.header.stamp = self.get_clock().now().to_msg()
        traj_marker.ns = "smppi_trajectory"
        traj_marker.id = 1
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD
        
        # Trajectory points
        for pose_stamped in self.optimal_path.path_points:
            point = Point()
            point.x = pose_stamped.pose.position.x
            point.y = pose_stamped.pose.position.y
            point.z = 0.05
            traj_marker.points.append(point)
        
        # Trajectory appearance
        traj_marker.scale.x = 0.05  # Line width
        traj_marker.color.r = 1.0
        traj_marker.color.g = 1.0
        traj_marker.color.b = 1.0
        traj_marker.color.a = 1.0

        # Per-pose state from the diagnostics of the same plan: white = clear,
        # amber = closer than d_req(v), red = collision
        states = self._plan_states(len(traj_marker.points))
        if states is not None:
            traj_marker.colors = [_rgba((_WHITE, _AMBER, _RED)[k], 1.0) for k in states]
        
        return traj_marker

    def _plan_states(self, n_points: int) -> Optional[list]:
        """0 clear / 1 shortfall / 2 collision per plan pose, None without matching diagnostics."""
        d = self.debug
        if d is None or len(d.plan_clearance) != n_points or len(d.plan_required) != n_points - 1:
            return None
        clear, req = d.plan_clearance, d.plan_required
        states = [0]
        for i in range(1, n_points):
            states.append(2 if clear[i] < self.collision_margin else (1 if clear[i] < req[i - 1] else 0))
        return states

    def create_rollout_markers(self) -> list:
        """
        Sampled rollouts: grey by softmax weight, red when the rollout collides.
        One LINE_STRIP per rollout with a single colour keeps the message small
        (per-vertex colours on a LINE_LIST were ~300 KB per frame for 60 rollouts).
        Large sample counts fall back to one LINE_LIST in a few colour bins.
        """
        r = self.rollouts
        n, p = r.num_samples, r.num_poses
        if n <= 0 or p < 2 or len(r.xy) != n * p * 2:
            return []
        xy = np.asarray(r.xy, dtype=np.float32).reshape(n, p, 2)
        w = np.asarray(r.weight, dtype=np.float32)
        alpha = 0.12 + 0.78 * (w / max(float(w.max()), 1e-12))
        colliding = np.asarray(r.cost, dtype=np.float32) >= self.collision_floor
        stamp = self.get_clock().now().to_msg()

        def base(mid, mtype):
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = stamp
            m.ns = "rollouts"
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.scale.x = 0.012
            m.pose.orientation.w = 1.0
            m.lifetime.nanosec = 500000000     # refreshed at the visualization rate
            return m

        if n <= 200:
            xy = xy[:, ::2, :]                  # 0.2 s between vertices
            out = []
            for k in range(n):
                m = base(100 + k, Marker.LINE_STRIP)
                m.points = [Point(x=float(x), y=float(y), z=0.03) for x, y in xy[k]]
                m.color = _rgba(_RED, 0.6 * float(alpha[k])) if colliding[k] else _rgba(_WHITE, float(alpha[k]))
                out.append(m)
            return out
        # many samples: LINE_LIST per colour bin, vertex count bounded
        stride = max(1, int(math.ceil(n * p / 20000.0)))
        xy = xy[:, ::stride, :]
        bins = np.where(colliding, 0, 1 + np.minimum(3, (alpha * 4).astype(int)))
        out = []
        for b in range(5):
            sel = np.nonzero(bins == b)[0]
            if len(sel) == 0:
                continue
            m = base(100 + b, Marker.LINE_LIST)
            pts = []
            for k in sel:
                line = [Point(x=float(x), y=float(y), z=0.03) for x, y in xy[k]]
                for a, c in zip(line[:-1], line[1:]):
                    pts += (a, c)
            m.points = pts
            m.color = _rgba(_RED, 0.35) if b == 0 else _rgba(_WHITE, 0.15 + 0.2 * b)
            out.append(m)
        return out

    def create_footprint_marker(self) -> Optional[Marker]:
        """Create robot footprint visualization marker"""
        if not self.robot_state or len(self.footprint) == 0:
            return None
        
        try:
            # Extract robot pose from robot state
            robot_x, robot_y, robot_yaw = self.robot_state.state_vector
            robot_pose = (robot_x, robot_y, robot_yaw)
            
            # Create robot footprint at current pose
            # True outline (the obstacle critic does not use footprint_padding)
            robot_footprint = GeometryUtils.create_robot_footprint_at_pose(
                self.footprint, robot_pose, 0.0)
            
            # Create line strip marker for footprint boundary
            footprint_marker = Marker()
            footprint_marker.header.frame_id = "odom"
            footprint_marker.header.stamp = self.get_clock().now().to_msg()
            footprint_marker.ns = "robot_footprint"
            footprint_marker.id = 20
            footprint_marker.type = Marker.LINE_STRIP
            footprint_marker.action = Marker.ADD
            
            # Add footprint points to create closed polygon
            for vertex in robot_footprint:
                point = Point()
                point.x = float(vertex[0])
                point.y = float(vertex[1])
                point.z = 0.02  # Slightly above ground
                footprint_marker.points.append(point)
            
            # Close the polygon by adding first point at the end
            if len(footprint_marker.points) > 0:
                footprint_marker.points.append(footprint_marker.points[0])
            
            # Blue color for footprint boundary
            footprint_marker.scale.x = 0.05  # Line width
            footprint_marker.color.r = 0.9
            footprint_marker.color.g = 0.9
            footprint_marker.color.b = 0.9
            footprint_marker.color.a = 0.8
            
            return footprint_marker
            
        except Exception as e:
            self.get_logger().warn(f"Footprint marker creation error: {str(e)}")
            return None
    
    def create_lookahead_marker(self) -> Optional[Marker]:
        """Create lookahead point visualization marker with direction arrow"""
        if not self.lookahead_point:
            return None
        
        try:
            lookahead_marker = Marker()
            lookahead_marker.header.frame_id = "odom"
            lookahead_marker.header.stamp = self.get_clock().now().to_msg()
            lookahead_marker.ns = "lookahead"
            lookahead_marker.id = 30
            lookahead_marker.type = Marker.ARROW
            lookahead_marker.action = Marker.ADD
            
            # Position and orientation from lookahead pose
            lookahead_marker.pose.position.x = self.lookahead_point.pose.position.x
            lookahead_marker.pose.position.y = self.lookahead_point.pose.position.y
            lookahead_marker.pose.position.z = 0.2  # Above ground
            lookahead_marker.pose.orientation = self.lookahead_point.pose.orientation
            
            # Arrow scale - smaller than robot heading but visible
            lookahead_marker.scale.x = 0.8  # Arrow length
            lookahead_marker.scale.y = 0.12  # Arrow width
            lookahead_marker.scale.z = 0.12  # Arrow height
            
            # Bright green color for lookahead direction
            lookahead_marker.color.r = 0.55
            lookahead_marker.color.g = 0.55
            lookahead_marker.color.b = 0.55
            lookahead_marker.color.a = 0.9
            
            # Short lifetime for real-time updates
            lookahead_marker.lifetime.sec = 0
            lookahead_marker.lifetime.nanosec = 300000000  # 0.3 seconds
            
            return lookahead_marker
            
        except Exception as e:
            self.get_logger().warn(f"Lookahead marker creation error: {str(e)}")
            return None

    def create_next_waypoints_markers(self) -> list:
        """Create visualization markers for next waypoints"""
        markers = []
        
        try:
            if not self.multiple_waypoints or not self.multiple_waypoints.next_waypoints:
                return markers
            
            for i, waypoint in enumerate(self.multiple_waypoints.next_waypoints):
                marker = Marker()
                marker.header.frame_id = "odom"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "next_waypoints"
                marker.id = 40 + i  # Start from ID 40
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                
                # Position and orientation from waypoint pose
                marker.pose.position.x = waypoint.pose.position.x
                marker.pose.position.y = waypoint.pose.position.y
                marker.pose.position.z = 0.15  # Above ground
                marker.pose.orientation = waypoint.pose.orientation  # Use waypoint's orientation
                
                # Arrow scale - smaller than current goal and lookahead
                marker.scale.x = 0.6  # Arrow length
                marker.scale.y = 0.08  # Arrow width
                marker.scale.z = 0.08  # Arrow height
                
                # Color - blue with decreasing intensity for distant waypoints
                alpha = 0.8 - (i * 0.15)  # Fade out for distant waypoints
                marker.color.r = 0.45
                marker.color.g = 0.45
                marker.color.b = 0.45
                marker.color.a = max(alpha, 0.4)
                
                # Lifetime
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = 500000000  # 0.5 seconds
                
                markers.append(marker)
                
            return markers
                
        except Exception as e:
            self.get_logger().warn(f"Next waypoints marker creation error: {str(e)}")
            return []

    def _circle_points(self, cx: float, cy: float, radius: float, segments: int = 16) -> list:
        """Closed circle as LINE_STRIP vertices"""
        return [Point(x=cx + radius * math.cos(2 * math.pi * k / segments),
                      y=cy + radius * math.sin(2 * math.pi * k / segments), z=0.03) for k in range(segments + 1)]

    def create_footprint_circles_markers(self) -> list:
        """Discs the obstacle critic checks: at the robot and along the plan every 1 s"""
        if not self.circles:
            return []
        poses = [(self.robot_state.state_vector, 0)]
        if self.optimal_path is not None and self.optimal_path.path_points:
            pts = self.optimal_path.path_points
            states = self._plan_states(len(pts))
            for i in range(10, len(pts), 10):
                q = pts[i].pose
                yaw = 2.0 * math.atan2(q.orientation.z, q.orientation.w)
                poses.append(((q.position.x, q.position.y, yaw), states[i] if states else 0))
        stamp = self.get_clock().now().to_msg()
        out = []
        for j, ((x, y, yaw), state) in enumerate(poses):
            cs, sn = math.cos(yaw), math.sin(yaw)
            for i, (bx, by) in enumerate(self.circles):
                m = Marker()
                m.header.frame_id = "odom"
                m.header.stamp = stamp
                m.ns = "footprint_circles"
                m.id = 200 + j * len(self.circles) + i
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.02
                m.pose.orientation.w = 1.0
                m.lifetime.nanosec = 500000000
                m.color = _rgba((_WHITE, _AMBER, _RED)[state], 0.9 if j == 0 else 0.45)
                m.points = self._circle_points(x + bx * cs - by * sn, y + bx * sn + by * cs, self.circle_radius)
                out.append(m)
        return out

    def create_goal_region_marker(self) -> Optional[Marker]:
        """Goal reached radius, and the passed-plane through the goal normal to its heading"""
        g = self.latest_goal.pose
        gx, gy = g.position.x, g.position.y
        gyaw = 2.0 * math.atan2(g.orientation.z, g.orientation.w)
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_region"
        marker.id = 70
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.pose.orientation.w = 1.0
        marker.color = _rgba(_WHITE, 0.55)
        ring = self._circle_points(gx, gy, self.goal_reached_threshold, 48)
        pts = [q for a, b in zip(ring[:-1], ring[1:]) for q in (a, b)]
        final = self.multiple_waypoints is not None and self.multiple_waypoints.is_final_waypoint
        if self.passed_goal_max_distance > 0.0 and not final:
            half = max(self.passed_goal_max_distance, self.goal_reached_threshold)
            nx, ny = -math.sin(gyaw), math.cos(gyaw)
            pts += [Point(x=gx - half * nx, y=gy - half * ny, z=0.03), Point(x=gx + half * nx, y=gy + half * ny, z=0.03)]
            # entry direction: crossing the line along this tick counts as passed
            pts += [Point(x=gx, y=gy, z=0.03), Point(x=gx + 1.0 * math.cos(gyaw), y=gy + 1.0 * math.sin(gyaw), z=0.03)]
        marker.points = pts
        return marker

    def create_debug_text_marker(self) -> Optional[Marker]:
        """Cost breakdown and sampler state above the robot"""
        d = self.debug
        x, y, _ = self.robot_state.state_vector
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mppi_debug"
        marker.id = 80
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = float(x), float(y), 2.2
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.32
        marker.color = _rgba(_RED if d.collision_severity > 0 else (_AMBER if d.worst_shortfall > 0 else _WHITE), 0.95)
        hit = f"{d.first_hit_median:.1f}s" if d.first_hit_median >= 0 else "-"
        marker.text = (f"goal {d.goal_cost:.0f}  clear {d.obstacle_soft_cost:.0f}  col {d.collision_severity:.2f}  keep {d.keepout_severity:.2f}\n"
                       f"min clear {d.min_clearance:.2f} m @{d.min_clearance_time:.1f}s (need {d.required_at_min:.2f})\n"
                       f"rollouts collide {100 * d.collide_fraction:.0f}% (first hit {hit})  ESS {d.ess:.0f}/{d.num_rollouts}\n"
                       f"v {d.v_cmd:.2f}  {d.compute_ms:.1f} ms  costmap {d.costmap_age:.2f} s")
        return marker


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = VisualizationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()