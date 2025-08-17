"""
Visualization utility for sequential global planner
Handles creation of RViz markers for path visualization
"""

from typing import List, Dict, Any, Optional
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time


class PathVisualizer:
    """Utility class for creating RViz visualization markers"""
    
    @staticmethod
    def create_node_markers(ordered_nodes: List[str], 
                          nodes_data: Dict[str, Any],
                          map_origin_utm_easting: float,
                          map_origin_utm_northing: float,
                          timestamp: Time,
                          frame_id: str = 'odom') -> List[Marker]:
        """
        Create sphere markers for nodes
        
        Args:
            ordered_nodes: List of node IDs in sequential order
            nodes_data: Dictionary of node data
            map_origin_utm_easting: UTM easting origin
            map_origin_utm_northing: UTM northing origin
            timestamp: ROS timestamp for markers
            frame_id: Coordinate frame ID
            
        Returns:
            List of node markers
        """
        markers = []
        
        for i, node_id in enumerate(ordered_nodes):
            node_data = nodes_data[node_id]
            
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = 'sequential_nodes'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = node_data['UtmInfo']['Easting'] - map_origin_utm_easting
            marker.pose.position.y = node_data['UtmInfo']['Northing'] - map_origin_utm_northing
            marker.pose.position.z = 0.0
            
            # Orientation
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            
            # Color - gradient from green to red
            ratio = float(i) / max(len(ordered_nodes) - 1, 1)
            marker.color.r = ratio
            marker.color.g = 1.0 - ratio
            marker.color.b = 0.2
            marker.color.a = 0.8
            
            marker.lifetime.sec = 0  # Persistent
            markers.append(marker)
            
        return markers
    
    @staticmethod
    def create_link_markers(ordered_nodes: List[str],
                          nodes_data: Dict[str, Any],
                          map_origin_utm_easting: float,
                          map_origin_utm_northing: float,
                          timestamp: Time,
                          loop_path: bool = False,
                          frame_id: str = 'odom') -> List[Marker]:
        """
        Create arrow markers for links between nodes
        
        Args:
            ordered_nodes: List of node IDs in sequential order
            nodes_data: Dictionary of node data
            map_origin_utm_easting: UTM easting origin
            map_origin_utm_northing: UTM northing origin
            timestamp: ROS timestamp for markers
            loop_path: Whether to create a loop closure link
            frame_id: Coordinate frame ID
            
        Returns:
            List of link markers
        """
        markers = []
        
        # Regular sequential links
        for i in range(len(ordered_nodes) - 1):
            from_node_id = ordered_nodes[i]
            to_node_id = ordered_nodes[i + 1]
            
            from_node = nodes_data[from_node_id]
            to_node = nodes_data[to_node_id]
            
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = 'sequential_links'
            marker.id = i + 1000  # Offset to avoid ID collision
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Start point
            start_point = Point()
            start_point.x = from_node['UtmInfo']['Easting'] - map_origin_utm_easting
            start_point.y = from_node['UtmInfo']['Northing'] - map_origin_utm_northing
            start_point.z = 0.0
            
            # End point
            end_point = Point()
            end_point.x = to_node['UtmInfo']['Easting'] - map_origin_utm_easting
            end_point.y = to_node['UtmInfo']['Northing'] - map_origin_utm_northing
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
            markers.append(marker)
        
        # Loop closure if enabled
        if loop_path and len(ordered_nodes) > 1:
            from_node = nodes_data[ordered_nodes[-1]]
            to_node = nodes_data[ordered_nodes[0]]
            
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = 'sequential_links'
            marker.id = 2000  # Loop marker
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            start_point = Point()
            start_point.x = from_node['UtmInfo']['Easting'] - map_origin_utm_easting
            start_point.y = from_node['UtmInfo']['Northing'] - map_origin_utm_northing
            start_point.z = 0.0
            
            end_point = Point()
            end_point.x = to_node['UtmInfo']['Easting'] - map_origin_utm_easting
            end_point.y = to_node['UtmInfo']['Northing'] - map_origin_utm_northing
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
            markers.append(marker)
            
        return markers
    
    @staticmethod
    def create_text_markers(ordered_nodes: List[str],
                          nodes_data: Dict[str, Any],
                          map_origin_utm_easting: float,
                          map_origin_utm_northing: float,
                          timestamp: Time,
                          frame_id: str = 'odom') -> List[Marker]:
        """
        Create text markers for node IDs
        
        Args:
            ordered_nodes: List of node IDs in sequential order
            nodes_data: Dictionary of node data
            map_origin_utm_easting: UTM easting origin
            map_origin_utm_northing: UTM northing origin
            timestamp: ROS timestamp for markers
            frame_id: Coordinate frame ID
            
        Returns:
            List of text markers
        """
        markers = []
        
        for i, node_id in enumerate(ordered_nodes):
            node_data = nodes_data[node_id]
            
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = 'sequential_node_labels'
            marker.id = i + 3000  # Offset for text markers
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            # Position (slightly above the node)
            marker.pose.position.x = node_data['UtmInfo']['Easting'] - map_origin_utm_easting
            marker.pose.position.y = node_data['UtmInfo']['Northing'] - map_origin_utm_northing
            marker.pose.position.z = 0.5
            
            # Text content
            marker.text = f"{i+1}: {node_id}"
            
            # Scale
            marker.scale.z = 0.3  # Text height
            
            # Color - white text
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            
            marker.lifetime.sec = 0  # Persistent
            markers.append(marker)
            
        return markers
    
    @staticmethod
    def create_marker_array(ordered_nodes: List[str],
                          nodes_data: Dict[str, Any],
                          map_origin_utm_easting: float,
                          map_origin_utm_northing: float,
                          timestamp: Time,
                          loop_path: bool = False,
                          include_text: bool = True,
                          frame_id: str = 'odom') -> MarkerArray:
        """
        Create complete marker array with all visualization elements
        
        Args:
            ordered_nodes: List of node IDs in sequential order
            nodes_data: Dictionary of node data
            map_origin_utm_easting: UTM easting origin
            map_origin_utm_northing: UTM northing origin
            timestamp: ROS timestamp for markers
            loop_path: Whether to create a loop closure link
            include_text: Whether to include text labels
            frame_id: Coordinate frame ID
            
        Returns:
            MarkerArray containing all markers
        """
        marker_array = MarkerArray()
        
        # Add node markers
        node_markers = PathVisualizer.create_node_markers(
            ordered_nodes, nodes_data, map_origin_utm_easting,
            map_origin_utm_northing, timestamp, frame_id
        )
        marker_array.markers.extend(node_markers)
        
        # Add link markers
        link_markers = PathVisualizer.create_link_markers(
            ordered_nodes, nodes_data, map_origin_utm_easting,
            map_origin_utm_northing, timestamp, loop_path, frame_id
        )
        marker_array.markers.extend(link_markers)
        
        # Add text markers if requested
        if include_text:
            text_markers = PathVisualizer.create_text_markers(
                ordered_nodes, nodes_data, map_origin_utm_easting,
                map_origin_utm_northing, timestamp, frame_id
            )
            marker_array.markers.extend(text_markers)
        
        return marker_array