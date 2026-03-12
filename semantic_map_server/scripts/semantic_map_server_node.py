#!/usr/bin/env python3
"""
Semantic Map Server Node
Loads semantic map JSON and publishes as ROS2 messages + visualization markers
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import json
from pathlib import Path

from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point32, Polygon, Point
from visualization_msgs.msg import Marker, MarkerArray
from command_center_interfaces.msg import SemanticArea, SemanticMap


class SemanticMapServerNode(Node):
    """Semantic Map Server - publishes area-based HD map"""

    # Area type colors (RGBA)
    AREA_COLORS = {
        'drivable': (0.30, 0.69, 0.31, 0.5),   # Green
        'crosswalk': (1.00, 0.60, 0.00, 0.6),  # Orange
        'sidewalk': (0.62, 0.62, 0.62, 0.5),   # Gray
        'no_entry': (0.96, 0.26, 0.21, 0.6),   # Red
        'plaza': (0.13, 0.59, 0.95, 0.5),      # Blue
    }

    def __init__(self):
        super().__init__('semantic_map_server')

        # Parameters
        self.declare_parameter('map_file', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('use_utm', False)
        self.declare_parameter('utm_zone', 52)
        self.declare_parameter('utm_band', 'N')

        self.map_file = self.get_parameter('map_file').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.use_utm = self.get_parameter('use_utm').value
        self.utm_zone = self.get_parameter('utm_zone').value
        self.utm_band = self.get_parameter('utm_band').value

        # Latched QoS for map data
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Publishers
        self.map_pub = self.create_publisher(
            SemanticMap, '/semantic_map', latched_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/semantic_map_markers', latched_qos)

        # Load and publish map
        self.map_data = None
        self.semantic_map_msg = None
        self.marker_array_msg = None

        if self.map_file:
            self.load_map(self.map_file)
        else:
            self.get_logger().warn('No map file specified. Use map_file parameter.')

        # Periodic publisher (for late subscribers)
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_map)

        self.get_logger().info(f'Semantic Map Server started')
        self.get_logger().info(f'  Map file: {self.map_file}')
        self.get_logger().info(f'  Frame ID: {self.frame_id}')

    def load_map(self, filepath: str):
        """Load semantic map from JSON file"""
        try:
            path = Path(filepath)
            if not path.exists():
                self.get_logger().error(f'Map file not found: {filepath}')
                return False

            with open(path, 'r', encoding='utf-8') as f:
                self.map_data = json.load(f)

            self.get_logger().info(f'Loaded map: {self.map_data.get("name", "unknown")}')
            self.get_logger().info(f'  Areas: {len(self.map_data.get("areas", []))}')

            # Convert to ROS messages
            self.semantic_map_msg = self.create_semantic_map_msg()
            self.marker_array_msg = self.create_marker_array()

            # Publish immediately
            self.publish_map()

            return True

        except Exception as e:
            self.get_logger().error(f'Failed to load map: {e}')
            return False

    def gps_to_local(self, lon: float, lat: float) -> tuple:
        """
        Convert GPS (lon, lat) to local coordinates
        If use_utm=True, converts to UTM. Otherwise uses simple meter offset.
        """
        if self.use_utm:
            try:
                import utm
                easting, northing, _, _ = utm.from_latlon(lat, lon)
                return easting, northing
            except ImportError:
                self.get_logger().warn('utm package not found, using simple conversion')

        # Simple conversion: degrees to approximate meters
        # Reference point: first area's first point
        if not hasattr(self, '_ref_lon'):
            areas = self.map_data.get('areas', [])
            if areas and areas[0].get('polygon'):
                self._ref_lon = areas[0]['polygon'][0][0]
                self._ref_lat = areas[0]['polygon'][0][1]
            else:
                self._ref_lon, self._ref_lat = lon, lat

        # Approximate conversion (works for small areas)
        import math
        lat_m = (lat - self._ref_lat) * 111320.0
        lon_m = (lon - self._ref_lon) * 111320.0 * math.cos(math.radians(self._ref_lat))

        return lon_m, lat_m

    def create_semantic_map_msg(self) -> SemanticMap:
        """Create SemanticMap message from loaded data"""
        msg = SemanticMap()
        msg.header = Header()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = self.map_data.get('name', 'unknown')
        msg.version = self.map_data.get('version', '1.0')
        msg.coordinate_system = self.map_data.get('coordinate_system', {}).get('type', 'WGS84')

        for area_data in self.map_data.get('areas', []):
            area_msg = SemanticArea()
            area_msg.id = area_data.get('id', '')
            area_msg.type = area_data.get('type', 'drivable')

            # Convert polygon
            polygon = Polygon()
            for coord in area_data.get('polygon', []):
                lon, lat = coord[0], coord[1]
                x, y = self.gps_to_local(lon, lat)
                point = Point32()
                point.x = float(x)
                point.y = float(y)
                point.z = 0.0
                polygon.points.append(point)
            area_msg.polygon = polygon

            # Properties
            props = area_data.get('properties', {})
            area_msg.speed_limit = float(props.get('speed_limit', 0.0))
            area_msg.priority = int(props.get('priority', 1))
            area_msg.traversable = bool(props.get('traversable', True))
            area_msg.cost_weight = float(props.get('cost_weight', 1.0))
            area_msg.remark = str(props.get('remark', ''))

            msg.areas.append(area_msg)

        return msg

    def create_marker_array(self) -> MarkerArray:
        """Create visualization MarkerArray from loaded data"""
        marker_array = MarkerArray()

        for idx, area_data in enumerate(self.map_data.get('areas', [])):
            # Polygon marker
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'semantic_areas'
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # Get color based on area type
            area_type = area_data.get('type', 'drivable')
            r, g, b, a = self.AREA_COLORS.get(area_type, (0.5, 0.5, 0.5, 0.5))

            marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            marker.scale.x = 0.3  # Line width

            # Add polygon points
            polygon = area_data.get('polygon', [])
            for coord in polygon:
                lon, lat = coord[0], coord[1]
                x, y = self.gps_to_local(lon, lat)
                point = Point()
                point.x = float(x)
                point.y = float(y)
                point.z = 0.1
                marker.points.append(point)

            # Close the polygon
            if polygon:
                lon, lat = polygon[0][0], polygon[0][1]
                x, y = self.gps_to_local(lon, lat)
                marker.points.append(Point(x=float(x), y=float(y), z=0.1))

            marker_array.markers.append(marker)

            # Filled polygon (TRIANGLE_LIST)
            fill_marker = Marker()
            fill_marker.header.frame_id = self.frame_id
            fill_marker.header.stamp = self.get_clock().now().to_msg()
            fill_marker.ns = 'semantic_areas_fill'
            fill_marker.id = idx
            fill_marker.type = Marker.TRIANGLE_LIST
            fill_marker.action = Marker.ADD
            fill_marker.color = ColorRGBA(r=r, g=g, b=b, a=a)
            fill_marker.scale.x = 1.0
            fill_marker.scale.y = 1.0
            fill_marker.scale.z = 1.0

            # Triangulate polygon (simple fan triangulation)
            if len(polygon) >= 3:
                # Convert all points first
                local_points = []
                for coord in polygon:
                    lon, lat = coord[0], coord[1]
                    x, y = self.gps_to_local(lon, lat)
                    local_points.append((x, y))

                # Fan triangulation from first point
                for i in range(1, len(local_points) - 1):
                    p0 = local_points[0]
                    p1 = local_points[i]
                    p2 = local_points[i + 1]

                    fill_marker.points.append(Point(x=float(p0[0]), y=float(p0[1]), z=0.05))
                    fill_marker.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=0.05))
                    fill_marker.points.append(Point(x=float(p2[0]), y=float(p2[1]), z=0.05))

            marker_array.markers.append(fill_marker)

            # Text label
            text_marker = Marker()
            text_marker.header.frame_id = self.frame_id
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = 'semantic_labels'
            text_marker.id = idx
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            # Calculate centroid for label position
            if polygon:
                cx = sum(c[0] for c in polygon) / len(polygon)
                cy = sum(c[1] for c in polygon) / len(polygon)
                lx, ly = self.gps_to_local(cx, cy)
                text_marker.pose.position.x = float(lx)
                text_marker.pose.position.y = float(ly)
                text_marker.pose.position.z = 0.5

            text_marker.text = area_data.get('id', '')
            text_marker.scale.z = 0.5
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

            marker_array.markers.append(text_marker)

        return marker_array

    def publish_map(self):
        """Publish map messages"""
        if self.semantic_map_msg:
            # Update timestamp
            self.semantic_map_msg.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.publish(self.semantic_map_msg)

        if self.marker_array_msg:
            # Update timestamps
            for marker in self.marker_array_msg.markers:
                marker.header.stamp = self.get_clock().now().to_msg()
            self.marker_pub.publish(self.marker_array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapServerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
