#!/usr/bin/env python3
"""
Visualization Node for bae_mppi
Handles RViz visualization of MPPI results
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np

# ROS2 messages
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped as PathPose
from std_msgs.msg import Header

# Custom messages
from bae_mppi.msg import OptimalPath, ProcessedObstacles, HighCostPath

# Local modules
from bae_mppi_core.visualizer import MPPIVisualizer


class VisualizationNode(Node):
    """Node for MPPI visualization"""
    
    def __init__(self):
        super().__init__('mppi_visualization')
        
        # Topic parameters
        self.declare_parameter('topics.input.optimal_path', 'optimal_path')
        self.declare_parameter('topics.input.processed_obstacles', 'obstacles')
        self.declare_parameter('topics.input.goal_pose', '/goal_pose')
        self.declare_parameter('topics.output.optimal_path_marker', '/mppi_optimal_path')
        self.declare_parameter('topics.output.best_paths_markers', '/mppi_best_paths')
        self.declare_parameter('topics.output.nav_path', '/mppi_nav_path')
        self.declare_parameter('topics.output.goal_marker', '/mppi_goal')
        self.declare_parameter('topics.output.obstacle_markers', '/mppi_obstacles')
        
        # Parameters
        self.declare_parameter('viz_frequency', 5.0)
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('enable_path_viz', True)
        self.declare_parameter('enable_obstacle_viz', True)
        self.declare_parameter('enable_goal_viz', True)
        self.declare_parameter('enable_best_paths', True)
        
        viz_frequency = self.get_parameter('viz_frequency').get_parameter_value().double_value
        self.enable_visualization = self.get_parameter('enable_visualization').get_parameter_value().bool_value
        self.enable_path_viz = self.get_parameter('enable_path_viz').get_parameter_value().bool_value
        self.enable_obstacle_viz = self.get_parameter('enable_obstacle_viz').get_parameter_value().bool_value
        self.enable_goal_viz = self.get_parameter('enable_goal_viz').get_parameter_value().bool_value
        self.enable_best_paths = self.get_parameter('enable_best_paths').get_parameter_value().bool_value
        
        # Visualizer
        self.visualizer = MPPIVisualizer(frame_id='odom')
        
        # Get topic names
        path_topic = self.get_parameter('topics.input.optimal_path').get_parameter_value().string_value
        obstacles_topic = self.get_parameter('topics.input.processed_obstacles').get_parameter_value().string_value
        goal_topic = self.get_parameter('topics.input.goal_pose').get_parameter_value().string_value
        path_marker_topic = self.get_parameter('topics.output.optimal_path_marker').get_parameter_value().string_value
        best_paths_topic = self.get_parameter('topics.output.best_paths_markers').get_parameter_value().string_value
        nav_path_topic = self.get_parameter('topics.output.nav_path').get_parameter_value().string_value
        goal_marker_topic = self.get_parameter('topics.output.goal_marker').get_parameter_value().string_value
        obstacle_markers_topic = self.get_parameter('topics.output.obstacle_markers').get_parameter_value().string_value
        
        # State variables
        self.latest_path = None
        self.latest_goal = None
        self.latest_obstacles = None
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )
        
        # Subscribers
        self.path_sub = self.create_subscription(
            OptimalPath, path_topic, self.path_callback, reliable_qos)
        self.goal_sub = self.create_subscription(
            PoseStamped, goal_topic, self.goal_callback, reliable_qos)
        self.obstacles_sub = self.create_subscription(
            ProcessedObstacles, obstacles_topic, self.obstacles_callback, reliable_qos)
        
        # Publishers
        self.optimal_path_pub = self.create_publisher(
            Marker, path_marker_topic, reliable_qos)
        self.goal_marker_pub = self.create_publisher(
            Marker, goal_marker_topic, reliable_qos)
        self.obstacle_markers_pub = self.create_publisher(
            MarkerArray, obstacle_markers_topic, reliable_qos)
        self.nav_path_pub = self.create_publisher(
            Path, nav_path_topic, reliable_qos)
        self.best_paths_pub = self.create_publisher(
            MarkerArray, best_paths_topic, reliable_qos)
        
        # Visualization timer
        self.viz_timer = self.create_timer(
            1.0 / viz_frequency, self.visualization_callback)
        
        self.get_logger().info('Visualization Node initialized')
    
    def path_callback(self, msg: OptimalPath):
        """Receive optimal path from MPPI core"""
        self.latest_path = msg
        # self.get_logger().info(f'Received optimal path with {len(msg.path_points)} points, {len(msg.high_cost_paths)} best paths')
    
    def goal_callback(self, msg: PoseStamped):
        """Receive goal pose"""
        self.latest_goal = msg
    
    def obstacles_callback(self, msg: ProcessedObstacles):
        """Receive obstacle information"""
        self.latest_obstacles = msg
    
    def visualization_callback(self):
        """Main visualization loop"""
        if not self.enable_visualization:
            return
            
        try:
            # Publish optimal path
            if self.enable_path_viz and self.latest_path is not None:
                self.get_logger().debug(f'Publishing optimal path with {len(self.latest_path.path_points)} points')
                self.publish_optimal_path()
                self.publish_nav_path()
            else:
                if self.latest_path is None:
                    self.get_logger().debug('No path data available for visualization')
        except Exception as e:
            self.get_logger().warn(f'Path visualization failed: {str(e)}')
            
        try:
            # Publish goal marker
            if self.enable_goal_viz and self.latest_goal is not None:
                self.publish_goal_marker()
        except Exception as e:
            self.get_logger().warn(f'Goal visualization failed: {str(e)}')
            
        try:
            # Publish obstacle markers
            if self.enable_obstacle_viz and self.latest_obstacles is not None:
                self.publish_obstacle_markers()
        except Exception as e:
            self.get_logger().warn(f'Obstacle visualization failed: {str(e)}')
            
        try:
            # Publish best paths
            if self.enable_best_paths and self.latest_path is not None:
                self.get_logger().debug(f'Publishing {len(self.latest_path.high_cost_paths)} best paths')
                self.publish_best_paths()
        except Exception as e:
            self.get_logger().warn(f'Best paths visualization failed: {str(e)}')
    
    def publish_optimal_path(self):
        """Publish optimal path as line marker"""
        if not self.latest_path or len(self.latest_path.path_points) == 0:
            return
        
        # Convert to numpy array for visualizer
        path_points = []
        for point in self.latest_path.path_points:
            path_points.append([point.x, point.y, 0.0])
        
        if len(path_points) < 2:
            return
        
        path_array = np.array(path_points)
        
        # Create line marker
        marker = Marker()
        marker.header = self.latest_path.header
        marker.ns = 'optimal_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Set line properties
        marker.scale.x = 0.05  # Line width
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Add points
        from geometry_msgs.msg import Point
        for point in path_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)
        
        self.optimal_path_pub.publish(marker)
    
    def publish_nav_path(self):
        """Publish optimal path as nav_msgs/Path"""
        if not self.latest_path or len(self.latest_path.path_points) == 0:
            return
        
        path_msg = Path()
        path_msg.header = self.latest_path.header
        
        for point in self.latest_path.path_points:
            pose_stamped = PathPose()
            pose_stamped.header = self.latest_path.header
            pose_stamped.pose.position.x = point.x
            pose_stamped.pose.position.y = point.y
            pose_stamped.pose.position.z = point.z
            pose_stamped.pose.orientation.w = 1.0  # Default orientation
            path_msg.poses.append(pose_stamped)
        
        self.nav_path_pub.publish(path_msg)
    
    def publish_goal_marker(self):
        """Publish goal marker"""
        if not hasattr(self.latest_goal, 'pose'):
            return
            
        pose = self.latest_goal.pose
        if not hasattr(pose, 'position'):
            return
            
        x = pose.position.x
        y = pose.position.y
        
        # Convert quaternion to yaw
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        goal_pose = [x, y, yaw]
        goal_marker = self.visualizer.create_goal_marker(goal_pose)
        goal_marker.header.stamp = self.latest_goal.header.stamp
        
        self.goal_marker_pub.publish(goal_marker)
    
    def publish_obstacle_markers(self):
        """Publish obstacle markers"""
        if not self.latest_obstacles or len(self.latest_obstacles.obstacle_points) == 0:
            return
        
        markers = MarkerArray()
        
        for i, point in enumerate(self.latest_obstacles.obstacle_points):
            marker = Marker()
            marker.header = self.latest_obstacles.header
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = point.x
            marker.pose.position.y = point.y
            marker.pose.position.z = point.z
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker.lifetime.sec = 1  # 1 second lifetime
            
            markers.markers.append(marker)
        
        self.obstacle_markers_pub.publish(markers)
    
    def publish_best_paths(self):
        """Publish best paths (lowest cost) as line markers"""
        if not self.latest_path or len(self.latest_path.high_cost_paths) == 0:
            return
        
        markers = MarkerArray()
        
        # Get cost range for normalization
        all_costs = [path.path_cost for path in self.latest_path.high_cost_paths]
        max_cost = max(all_costs) if all_costs else 1.0
        min_cost = min(all_costs) if all_costs else 0.0
        
        for i, best_path in enumerate(self.latest_path.high_cost_paths):
            if len(best_path.path_points) < 2:
                continue
                
            marker = Marker()
            marker.header = self.latest_path.header
            marker.ns = 'best_paths'
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Set line properties - blue with varying alpha based on cost (lower cost = more opaque)
            marker.scale.x = 0.03  # Slightly thicker than obstacles, thinner than main optimal path
            marker.color.r = 0.2
            marker.color.g = 0.6
            marker.color.b = 1.0
            # Use cost to determine transparency (lower cost = more opaque for best paths)
            if max_cost > min_cost:
                normalized_cost = (best_path.path_cost - min_cost) / (max_cost - min_cost)
                marker.color.a = 0.8 - 0.4 * normalized_cost  # Alpha from 0.4 to 0.8 (inverted)
            else:
                marker.color.a = 0.6
            
            # Add points to marker
            for point in best_path.path_points:
                marker.points.append(point)
            
            marker.lifetime.sec = 1  # 1 second lifetime
            markers.markers.append(marker)
        
        self.best_paths_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    
    node = VisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()