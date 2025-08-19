#!/usr/bin/env python3
"""
Sequential Global Path Planner Node
Reads JSON map file and publishes sequential path to behavior_planner
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import rcl_interfaces.srv

import os
from typing import List, Dict, Any, Optional

# Import utilities
from utils.map_loader import MapLoader
from utils.visualization import PathVisualizer

# ROS2 messages
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from sensor_msgs.msg import NavSatFix

# Custom messages
from command_center_interfaces.msg import PlannedPath
from gmserver.msg import MapData, MapNode, MapLink, GpsInfo, UtmInfo


class SequentialPlannerNode(Node):
    """Sequential Global Path Planner - reads JSON and publishes path"""
    
    def __init__(self):
        super().__init__('sequential_planner')
        
        # Parameters
        self.declare_parameter('map_file', 'mando_full_map.json')
        self.declare_parameter('auto_start', True)
        self.declare_parameter('loop_path', False)
        self.declare_parameter('publish_frequency', 1.0)  # Hz
        
        # GPS subscription parameter
        self.declare_parameter('gps_topic', '/gps/fix')
        
        # Get parameters
        self.map_file = self.get_parameter('map_file').get_parameter_value().string_value
        self.auto_start = self.get_parameter('auto_start').get_parameter_value().bool_value
        self.loop_path = self.get_parameter('loop_path').get_parameter_value().bool_value
        self.publish_freq = self.get_parameter('publish_frequency').get_parameter_value().double_value
        self.gps_topic = self.get_parameter('gps_topic').get_parameter_value().string_value
        
        # Map origin - will be set from localization map_origin topic
        self.map_origin_utm_easting = 0.0
        self.map_origin_utm_northing = 0.0
        self.map_origin_set = False
        
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
        
        # Localization node name for parameter access
        self.localization_node_name = 'tiny_localization_node'
        
        # Services (for future expansion)
        # self.start_service = self.create_service(...)
        
        # Data storage
        self.nodes_data = {}  # {node_id: node_data}
        self.links_data = []  # [link_data, ...]
        self.ordered_nodes = []  # Sequential order of nodes
        self.is_loaded = False
        self.path_published = False  # Track if path has been published
        
        # Initialize map loader
        self.map_loader = MapLoader(logger=self.get_logger())
        
        # Timer for publishing
        self.publish_timer = self.create_timer(
            1.0 / self.publish_freq, self.publish_callback)
        
        # Timer to check for map origin parameters
        self.param_check_timer = self.create_timer(1.0, self.check_map_origin_params)
        
        # Load map data
        self.load_map_file()
        
        if self.auto_start and self.is_loaded:
            self.publish_status("Sequential planner started - auto publishing path")
            
        # Try to get parameters immediately on startup
        self.check_map_origin_params()
        
        self.get_logger().info(f'Sequential Planner Node initialized')
        self.get_logger().info(f'Map file: {self.map_file}')
        self.get_logger().info(f'Loaded {len(self.nodes_data)} nodes, {len(self.links_data)} links')
    
    def load_map_file(self) -> bool:
        """Load JSON map file and extract nodes/links"""
        # Get full path to map file
        map_path = self.map_loader.get_map_path('sequential_global_planner', self.map_file)
        
        # Load map data using map loader
        self.nodes_data, self.links_data, success = self.map_loader.load_map_file(map_path)
        
        if success:
            # Create ordered node sequence from links
            self.create_sequential_order()
            self.is_loaded = True
        else:
            self.is_loaded = False
            
        return success
    
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
            
            # UTM info - use absolute coordinates in map frame
            map_node.utm_info.easting = node_data['UtmInfo']['Easting']
            map_node.utm_info.northing = node_data['UtmInfo']['Northing'] 
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
            
            # Convert UTM to odom frame (subtract map origin)
            pose.pose.position.x = node_data['UtmInfo']['Easting'] - self.map_origin_utm_easting
            pose.pose.position.y = node_data['UtmInfo']['Northing'] - self.map_origin_utm_northing
            pose.pose.position.z = 0.0
            
            # Set orientation (pointing to next node)
            pose.pose.orientation.w = 1.0  # No rotation for now
            
            nav_path.poses.append(pose)
        
        if self.loop_path and len(nav_path.poses) > 1:
            nav_path.poses.append(nav_path.poses[0])
        
        return nav_path
    
    def create_visualization_markers(self) -> MarkerArray:
        """Create visualization markers for nodes and links"""
        # Use PathVisualizer to create all markers
        return PathVisualizer.create_marker_array(
            ordered_nodes=self.ordered_nodes,
            nodes_data=self.nodes_data,
            map_origin_utm_easting=self.map_origin_utm_easting,
            map_origin_utm_northing=self.map_origin_utm_northing,
            timestamp=self.get_clock().now().to_msg(),
            loop_path=self.loop_path,
            include_text=True,
            frame_id='odom'
        )
    
    def publish_callback(self) -> None:
        """Timer callback to publish path and visualization"""
        if not self.is_loaded or not self.auto_start or not self.map_origin_set:
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

    def check_map_origin_params(self) -> None:
        """Check if map origin parameters are available from localization node"""
        if not self.map_origin_set:
            try:
                # Create a parameter client for the localization node
                from rclpy.parameter import Parameter
                from rclpy.node import Node
                
                # Get parameters from the localization node using parameter client
                param_client = self.create_client(
                    rcl_interfaces.srv.GetParameters,
                    '/localization/tiny_localization_node/get_parameters'
                )
                
                if not param_client.wait_for_service(timeout_sec=0.1):
                    self.get_logger().debug('Localization node parameter service not available yet')
                    return
                
                # Request parameters
                request = rcl_interfaces.srv.GetParameters.Request()
                request.names = [
                    'map_origin.utm_easting',
                    'map_origin.utm_northing', 
                    'map_origin.utm_zone'
                ]
                
                future = param_client.call_async(request)
                
                # Use executor to wait for response with timeout
                import time
                start_time = time.time()
                while not future.done() and (time.time() - start_time) < 0.5:
                    rclpy.spin_once(self, timeout_sec=0.01)
                
                if future.done():
                    response = future.result()
                    if response and len(response.values) == 3:
                        # Extract parameter values
                        easting = response.values[0].double_value
                        northing = response.values[1].double_value
                        zone = response.values[2].integer_value
                        
                        # Validate values (check they're not default/zero)
                        if easting != 0.0 and northing != 0.0:
                            # Successfully got all parameters
                            self.map_origin_utm_easting = easting
                            self.map_origin_utm_northing = northing
                            self.map_origin_set = True
                            
                            # Cancel parameter check timer
                            self.param_check_timer.cancel()
                            
                            self.get_logger().info('Map origin received from localization parameters:')
                            self.get_logger().info(f'  UTM: ({self.map_origin_utm_easting:.4f}, {self.map_origin_utm_northing:.4f}) Zone {zone}')
                            
                            # Now that we have map origin, start publishing if auto_start is enabled
                            if self.auto_start and self.is_loaded and not self.path_published:
                                self.publish_status("Map origin set - starting path publication")
                            return
                        else:
                            self.get_logger().debug('Map origin parameters are still at default values')
                else:
                    self.get_logger().debug('Timeout waiting for parameter response')
                        
            except Exception as e:
                self.get_logger().debug(f'Map origin parameters not ready: {e}')
    
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