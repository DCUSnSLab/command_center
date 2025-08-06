#!/usr/bin/env python3
"""
Sequential Global Path Planner Node
Reads JSON map file and publishes sequential path to behavior_planner
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import json
import os
from typing import List, Dict, Any, Optional

# ROS2 messages
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path

# Custom messages
from command_center_interfaces.msg import PlannedPath
from gmserver.msg import MapData, MapNode, MapLink, GpsInfo, UtmInfo


class SequentialPlannerNode(Node):
    """Sequential Global Path Planner - reads JSON and publishes path"""
    
    def __init__(self):
        super().__init__('sequential_planner')
        
        # Parameters
        self.declare_parameter('map_file', 'mando_full_map.json')
        self.declare_parameter('gps_ref_latitude', 37.23965631)  # First waypoint as reference
        self.declare_parameter('gps_ref_longitude', 126.7736361)
        self.declare_parameter('gps_ref_utm_easting', 302516.9182)
        self.declare_parameter('gps_ref_utm_northing', 4123781.294)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('loop_path', False)
        self.declare_parameter('publish_frequency', 1.0)  # Hz
        
        # Get parameters
        self.map_file = self.get_parameter('map_file').get_parameter_value().string_value
        self.gps_ref_lat = self.get_parameter('gps_ref_latitude').get_parameter_value().double_value
        self.gps_ref_lon = self.get_parameter('gps_ref_longitude').get_parameter_value().double_value
        self.gps_ref_utm_easting = self.get_parameter('gps_ref_utm_easting').get_parameter_value().double_value
        self.gps_ref_utm_northing = self.get_parameter('gps_ref_utm_northing').get_parameter_value().double_value
        self.auto_start = self.get_parameter('auto_start').get_parameter_value().bool_value
        self.loop_path = self.get_parameter('loop_path').get_parameter_value().bool_value
        self.publish_freq = self.get_parameter('publish_frequency').get_parameter_value().double_value
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Publishers
        self.path_pub = self.create_publisher(
            PlannedPath, '/planned_path_detailed', reliable_qos)
        self.nav_path_pub = self.create_publisher(
            Path, '/sequential_path_nav', reliable_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/sequential_path_markers', reliable_qos)
        self.status_pub = self.create_publisher(
            String, '/sequential_planner_status', reliable_qos)
        
        # Services (for future expansion)
        # self.start_service = self.create_service(...)
        
        # Data storage
        self.nodes_data = {}  # {node_id: node_data}
        self.links_data = []  # [link_data, ...]
        self.ordered_nodes = []  # Sequential order of nodes
        self.is_loaded = False
        self.path_published = False  # Track if path has been published
        
        # Timer for publishing
        self.publish_timer = self.create_timer(
            1.0 / self.publish_freq, self.publish_callback)
        
        # Load map data
        self.load_map_file()
        
        if self.auto_start and self.is_loaded:
            self.publish_status("Sequential planner started - auto publishing path")
        
        self.get_logger().info(f'Sequential Planner Node initialized')
        self.get_logger().info(f'Map file: {self.map_file}')
        self.get_logger().info(f'Loaded {len(self.nodes_data)} nodes, {len(self.links_data)} links')
    
    def load_map_file(self) -> bool:
        """Load JSON map file and extract nodes/links"""
        try:
            # Get package share directory path
            try:
                from ament_index_python.packages import get_package_share_directory
                package_share = get_package_share_directory('sequential_global_planner')
            except:
                # Fallback to source directory for development
                package_share = '/home/d2-521-30/repo/command_center_ws/src/command_center/Global_path_planner/sequential_global_planner'
            
            map_path = os.path.join(package_share, 'maps', self.map_file)
            
            with open(map_path, 'r', encoding='utf-8') as file:
                map_data = json.load(file)
            
            # Store nodes as dictionary for fast lookup
            for node in map_data.get('Node', []):
                self.nodes_data[node['ID']] = node
            
            # Store links
            self.links_data = map_data.get('Link', [])
            
            # Create ordered node sequence from links
            self.create_sequential_order()
            
            self.is_loaded = True
            self.get_logger().info(f'Successfully loaded map: {len(self.nodes_data)} nodes, {len(self.links_data)} links')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Failed to load map file {self.map_file}: {str(e)}')
            self.is_loaded = False
            return False
    
    def create_sequential_order(self) -> None:
        """Create ordered sequence of nodes following the links"""
        if not self.links_data:
            # If no links, use nodes in ID order
            self.ordered_nodes = sorted(self.nodes_data.keys())
            return
        
        # Build adjacency list from links
        adjacency = {}
        for link in self.links_data:
            from_node = link['FromNodeID']
            to_node = link['ToNodeID']
            
            if from_node not in adjacency:
                adjacency[from_node] = []
            adjacency[from_node].append(to_node)
        
        # Find starting node (node with no incoming links)
        incoming = set()
        for link in self.links_data:
            incoming.add(link['ToNodeID'])
        
        start_nodes = [node_id for node_id in self.nodes_data.keys() if node_id not in incoming]
        
        if start_nodes:
            start_node = start_nodes[0]  # Use first available start node
        else:
            start_node = list(self.nodes_data.keys())[0]  # Fallback to first node
        
        # Follow the chain of links
        self.ordered_nodes = []
        current = start_node
        visited = set()
        
        while current and current not in visited:
            self.ordered_nodes.append(current)
            visited.add(current)
            
            # Find next node
            next_nodes = adjacency.get(current, [])
            current = next_nodes[0] if next_nodes else None
        
        self.get_logger().info(f'Created sequential order: {len(self.ordered_nodes)} nodes')
        if len(self.ordered_nodes) > 0:
            self.get_logger().info(f'Path: {self.ordered_nodes[0]} -> ... -> {self.ordered_nodes[-1]}')
    
    def create_planned_path_message(self) -> PlannedPath:
        """Create PlannedPath message compatible with behavior_planner"""
        planned_path = PlannedPath()
        planned_path.header = Header()
        planned_path.header.stamp = self.get_clock().now().to_msg()
        planned_path.header.frame_id = 'map'
        
        # Path metadata
        planned_path.path_id = "sequential_path"
        planned_path.start_node_id = self.ordered_nodes[0] if self.ordered_nodes else ""
        planned_path.goal_node_id = self.ordered_nodes[-1] if self.ordered_nodes else ""
        planned_path.total_distance = 0.0  # Can calculate if needed
        planned_path.total_time = 0.0      # Can calculate if needed
        
        # Create MapData with nodes and links
        map_data = MapData()
        
        # Convert nodes to MapNode messages
        for node_id in self.ordered_nodes:
            node_data = self.nodes_data[node_id]
            
            map_node = MapNode()
            map_node.id = node_id
            map_node.admin_code = node_data.get('AdminCode', '110')
            map_node.node_type = node_data.get('NodeType', 1)
            map_node.its_node_id = node_data.get('ITSNodeID', f'ITS_{node_id}')
            map_node.maker = node_data.get('Maker', '한국도로공사')
            map_node.update_date = node_data.get('UpdateDate', '20250418')
            map_node.version = node_data.get('Version', '2021')
            map_node.remark = node_data.get('Remark', '')
            map_node.hist_type = node_data.get('HistType', '02A')
            map_node.hist_remark = node_data.get('HistRemark', '')
            
            # GPS info
            map_node.gps_info.lat = node_data['GpsInfo']['Lat']
            map_node.gps_info.longitude = node_data['GpsInfo']['Long']
            map_node.gps_info.alt = node_data['GpsInfo']['Alt']
            
            # UTM info - convert to odom frame (subtract reference)
            map_node.utm_info.easting = node_data['UtmInfo']['Easting'] - self.gps_ref_utm_easting
            map_node.utm_info.northing = node_data['UtmInfo']['Northing'] - self.gps_ref_utm_northing
            map_node.utm_info.zone = node_data['UtmInfo']['Zone']
            
            map_data.nodes.append(map_node)
        
        # Add relevant links
        for link_data in self.links_data:
            from_id = link_data['FromNodeID']
            to_id = link_data['ToNodeID']
            
            # Only include links that are part of our sequential path
            if from_id in self.ordered_nodes and to_id in self.ordered_nodes:
                map_link = MapLink()
                map_link.id = link_data.get('ID', '')
                map_link.admin_code = link_data.get('AdminCode', '110')
                map_link.road_rank = link_data.get('RoadRank', 1)
                map_link.road_type = link_data.get('RoadType', 1)
                map_link.road_no = link_data.get('RoadNo', '20')
                map_link.link_type = link_data.get('LinkType', 3)
                map_link.lane_no = link_data.get('LaneNo', 2)
                map_link.from_node_id = from_id
                map_link.to_node_id = to_id
                map_link.length = link_data.get('Length', 0.1)
                
                map_data.links.append(map_link)
        
        # Loop closure if enabled
        if self.loop_path and len(map_data.nodes) > 1:
            # Add first node at the end
            map_data.nodes.append(map_data.nodes[0])
        
        planned_path.path_data = map_data
        return planned_path
    
    def create_nav_path_message(self) -> Path:
        """Create nav_msgs/Path for RViz visualization"""
        nav_path = Path()
        nav_path.header = Header()
        nav_path.header.stamp = self.get_clock().now().to_msg()
        nav_path.header.frame_id = 'odom'
        
        for node_id in self.ordered_nodes:
            node_data = self.nodes_data[node_id]
            
            pose = PoseStamped()
            pose.header = nav_path.header
            
            # Convert UTM to odom frame
            pose.pose.position.x = node_data['UtmInfo']['Easting'] - self.gps_ref_utm_easting
            pose.pose.position.y = node_data['UtmInfo']['Northing'] - self.gps_ref_utm_northing
            pose.pose.position.z = 0.0
            
            # Set orientation (pointing to next node)
            pose.pose.orientation.w = 1.0  # No rotation for now
            
            nav_path.poses.append(pose)
        
        if self.loop_path and len(nav_path.poses) > 1:
            nav_path.poses.append(nav_path.poses[0])
        
        return nav_path
    
    def create_visualization_markers(self) -> MarkerArray:
        """Create visualization markers for nodes and links"""
        marker_array = MarkerArray()
        
        # Node markers (spheres)
        for i, node_id in enumerate(self.ordered_nodes):
            node_data = self.nodes_data[node_id]
            
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'sequential_nodes'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = node_data['UtmInfo']['Easting'] - self.gps_ref_utm_easting
            marker.pose.position.y = node_data['UtmInfo']['Northing'] - self.gps_ref_utm_northing
            marker.pose.position.z = 0.0
            
            # Orientation
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = 0.2
            marker.scale.y = 0.2  
            marker.scale.z = 0.2
            
            # Color - gradient from green to red
            ratio = float(i) / max(len(self.ordered_nodes) - 1, 1)
            marker.color.r = ratio
            marker.color.g = 1.0 - ratio
            marker.color.b = 0.2
            marker.color.a = 0.8
            
            marker.lifetime.sec = 0  # Persistent
            marker_array.markers.append(marker)
        
        # Link markers (lines)
        for i in range(len(self.ordered_nodes) - 1):
            from_node_id = self.ordered_nodes[i]
            to_node_id = self.ordered_nodes[i + 1]
            
            from_node = self.nodes_data[from_node_id]
            to_node = self.nodes_data[to_node_id]
            
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'sequential_links'
            marker.id = i + 1000  # Offset to avoid ID collision
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Start point
            start_point = Point()
            start_point.x = from_node['UtmInfo']['Easting'] - self.gps_ref_utm_easting
            start_point.y = from_node['UtmInfo']['Northing'] - self.gps_ref_utm_northing
            start_point.z = 0.0
            
            # End point  
            end_point = Point()
            end_point.x = to_node['UtmInfo']['Easting'] - self.gps_ref_utm_easting
            end_point.y = to_node['UtmInfo']['Northing'] - self.gps_ref_utm_northing
            end_point.z = 0.0
            
            marker.points = [start_point, end_point]
            
            # Scale
            marker.scale.x = 0.05  # Arrow shaft diameter
            marker.scale.y = 0.1   # Arrow head diameter
            marker.scale.z = 0.0   # Not used for arrows
            
            # Color - blue arrows
            marker.color.r = 0.0
            marker.color.g = 0.4
            marker.color.b = 1.0
            marker.color.a = 0.7
            
            marker.lifetime.sec = 0  # Persistent
            marker_array.markers.append(marker)
        
        # Loop closure if enabled
        if self.loop_path and len(self.ordered_nodes) > 1:
            from_node = self.nodes_data[self.ordered_nodes[-1]]
            to_node = self.nodes_data[self.ordered_nodes[0]]
            
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'sequential_links'
            marker.id = 2000  # Loop marker
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            start_point = Point()
            start_point.x = from_node['UtmInfo']['Easting'] - self.gps_ref_utm_easting
            start_point.y = from_node['UtmInfo']['Northing'] - self.gps_ref_utm_northing
            start_point.z = 0.0
            
            end_point = Point()
            end_point.x = to_node['UtmInfo']['Easting'] - self.gps_ref_utm_easting
            end_point.y = to_node['UtmInfo']['Northing'] - self.gps_ref_utm_northing
            end_point.z = 0.0
            
            marker.points = [start_point, end_point]
            
            marker.scale.x = 0.08  # Thicker for loop
            marker.scale.y = 0.15
            marker.scale.z = 0.0
            
            # Color - red for loop
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)
        
        return marker_array
    
    def publish_callback(self) -> None:
        """Timer callback to publish path and visualization"""
        if not self.is_loaded or not self.auto_start:
            return
        
        # Publish PlannedPath only once
        if not self.path_published:
            planned_path = self.create_planned_path_message()
            self.path_pub.publish(planned_path)
            self.path_published = True
            self.publish_status("Published sequential path once to behavior_planner")
        
        # Continue publishing visualization for RViz
        nav_path = self.create_nav_path_message()
        self.nav_path_pub.publish(nav_path)
        
        # Publish visualization markers
        markers = self.create_visualization_markers()
        self.marker_pub.publish(markers)
    
    def publish_status(self, message: str) -> None:
        """Publish status message"""
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)
        self.get_logger().info(f'Status: {message}')


def main(args=None):
    rclpy.init(args=args)
    
    node = SequentialPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()