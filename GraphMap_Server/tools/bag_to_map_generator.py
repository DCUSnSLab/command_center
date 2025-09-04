#!/usr/bin/env python3

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Any
import math

try:
    import rclpy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import pyproj
except ImportError as e:
    print(f"Required dependencies not found: {e}")
    print("Install with: pip install pyproj")
    sys.exit(1)


class BagToMapGenerator:
    def __init__(self, min_distance=1.0, start_time=None, duration=None, node_prefix="N"):
        self.gps_points = []
        self.utm_transformer = None
        self.min_distance = min_distance  # Minimum distance in meters
        self.start_time = start_time  # Start time in seconds from bag start
        self.duration = duration  # Duration in seconds
        self.node_prefix = node_prefix  # Node ID prefix (e.g., "N", "Na", "Nb")
        
    def setup_utm_transformer(self, lat: float, lon: float):
        """Setup UTM transformer based on first GPS coordinate"""
        utm_zone = int((lon + 180) / 6) + 1
        utm_crs = f"EPSG:{32600 + utm_zone}"  # Northern hemisphere
        if lat < 0:
            utm_crs = f"EPSG:{32700 + utm_zone}"  # Southern hemisphere
        
        self.utm_transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        return utm_zone
    
    def gps_to_utm(self, lat: float, lon: float) -> tuple:
        """Convert GPS coordinates to UTM"""
        if self.utm_transformer is None:
            zone = self.setup_utm_transformer(lat, lon)
            zone_str = f"{zone}N" if lat >= 0 else f"{zone}S"
        else:
            zone_str = "52N"  # Default for Seoul area
            
        easting, northing = self.utm_transformer.transform(lon, lat)
        return easting, northing, zone_str
    
    def read_bag_file(self, bag_path: str, gps_topic: str):
        """Read GPS data from ROS2 bag file"""
        bag_dir = Path(bag_path)
        
        # Find .db3 file in the bag directory
        db_files = list(bag_dir.glob("*.db3"))
        if not db_files:
            raise FileNotFoundError(f"No .db3 files found in bag directory: {bag_path}")
        
        # Use the first .db3 file found
        db_path = db_files[0]
        print(f"Using database file: {db_path}")
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get topic info
        cursor.execute("SELECT id FROM topics WHERE name = ?", (gps_topic,))
        topic_result = cursor.fetchone()
        if not topic_result:
            raise ValueError(f"Topic '{gps_topic}' not found in bag file")
        
        topic_id = topic_result[0]
        
        # Get message type
        cursor.execute("SELECT type FROM topics WHERE id = ?", (topic_id,))
        msg_type = cursor.fetchone()[0]
        
        # Get start timestamp for time filtering
        cursor.execute("SELECT MIN(timestamp) FROM messages WHERE topic_id = ?", (topic_id,))
        bag_start_time = cursor.fetchone()[0]
        
        # Calculate time window
        start_timestamp = bag_start_time
        end_timestamp = None
        
        if self.start_time is not None:
            start_timestamp = bag_start_time + int(self.start_time * 1e9)  # Convert to nanoseconds
            
        if self.duration is not None:
            # If start_time is not specified, start from bag beginning
            if self.start_time is None:
                end_timestamp = bag_start_time + int(self.duration * 1e9)
            else:
                end_timestamp = start_timestamp + int(self.duration * 1e9)
        
        # Build query with time filtering
        query = "SELECT data, timestamp FROM messages WHERE topic_id = ?"
        params = [topic_id]
        
        if self.start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_timestamp)
            
        if end_timestamp is not None:
            query += " AND timestamp <= ?"
            params.append(end_timestamp)
            
        query += " ORDER BY timestamp"
        
        # Print time filtering info
        start_time_str = f"{self.start_time}s" if self.start_time is not None else "beginning"
        duration_str = f"{self.duration}s" if self.duration is not None else "end"
        print(f"Time filtering: start={start_time_str}, duration={duration_str}")
        if end_timestamp is not None:
            print(f"Timestamp range: {start_timestamp} to {end_timestamp} ({(end_timestamp - start_timestamp) / 1e9:.3f}s)")
        
        # Get messages
        cursor.execute(query, params)
        
        message_class = get_message(msg_type)
        total_messages = 0
        filtered_messages = 0
        
        for row in cursor.fetchall():
            total_messages += 1
            msg = deserialize_message(row[0], message_class)
            
            # Extract GPS coordinates based on message type
            lat, lon = self.extract_gps_from_message(msg)
            if lat is not None and lon is not None:
                filtered_messages += 1
                # Filter points based on minimum distance
                if self.should_add_point(lat, lon):
                    self.gps_points.append((lat, lon))
        
        conn.close()
        print(f"Processed {total_messages} messages, {filtered_messages} valid GPS points")
        print(f"Generated {len(self.gps_points)} GPS points after distance filtering")
    
    def should_add_point(self, lat: float, lon: float) -> bool:
        """Check if GPS point should be added based on minimum distance"""
        if not self.gps_points:
            return True  # Always add first point
        
        # Get last added point
        last_lat, last_lon = self.gps_points[-1]
        
        # Calculate distance using UTM coordinates for accuracy
        if self.utm_transformer is None:
            self.setup_utm_transformer(lat, lon)
        
        # Convert both points to UTM
        easting1, northing1, _ = self.gps_to_utm(last_lat, last_lon)
        easting2, northing2, _ = self.gps_to_utm(lat, lon)
        
        # Calculate Euclidean distance in meters
        distance = math.sqrt((easting2 - easting1)**2 + (northing2 - northing1)**2)
        
        return distance >= self.min_distance
    
    def extract_gps_from_message(self, msg) -> tuple:
        """Extract latitude and longitude from ROS message"""
        # Handle different GPS message types
        if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
            return msg.latitude, msg.longitude
        elif hasattr(msg, 'lat') and hasattr(msg, 'lon'):
            return msg.lat, msg.lon
        elif hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
            # For PoseStamped or similar
            return msg.pose.position.x, msg.pose.position.y
        elif hasattr(msg, 'position'):
            return msg.position.x, msg.position.y
        else:
            print(f"Unknown GPS message format: {type(msg)}")
            return None, None
    
    def generate_nodes_from_gps(self) -> List[Dict[str, Any]]:
        """Generate map nodes from GPS points"""
        if not self.gps_points:
            raise ValueError("No GPS points available")
        
        nodes = []
        for i, (lat, lon) in enumerate(self.gps_points):
            easting, northing, zone = self.gps_to_utm(lat, lon)
            
            node_id = f"{self.node_prefix}{i:03d}"
            node = {
                "ID": node_id,
                "AdminCode": "110",
                "NodeType": 1,
                "ITSNodeID": f"ITS_{node_id}",
                "Maker": "한국도로공사",
                "UpdateDate": "20250822",
                "Version": "2021",
                "Remark": f"Generated node {i} (prefix: {self.node_prefix})",
                "HistType": "02A",
                "HistRemark": "자동 생성",
                "GpsInfo": {
                    "Lat": round(lat, 6),
                    "Long": round(lon, 6),
                    "Alt": 0.0
                },
                "UtmInfo": {
                    "Easting": round(easting, 2),
                    "Northing": round(northing, 2),
                    "Zone": zone
                }
            }
            nodes.append(node)
        
        return nodes
    
    def generate_links_from_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate links between consecutive nodes"""
        links = []
        
        for i in range(len(nodes) - 1):
            from_node = nodes[i]
            to_node = nodes[i + 1]
            
            # Calculate distance
            from_utm = from_node["UtmInfo"]
            to_utm = to_node["UtmInfo"]
            distance = math.sqrt(
                (to_utm["Easting"] - from_utm["Easting"]) ** 2 +
                (to_utm["Northing"] - from_utm["Northing"]) ** 2
            ) / 1000.0  # Convert to km
            
            link = {
                "ID": f"L{i:08d}",
                "AdminCode": "110",
                "RoadRank": 1,
                "RoadType": 1,
                "RoadNo": "20",
                "LinkType": 3,
                "LaneNo": 2,
                "R_LinkID": f"R_{i:04d}",
                "L_LinkID": f"L_{i:04d}",
                "FromNodeID": from_node["ID"],
                "ToNodeID": to_node["ID"],
                "SectionID": f"SECTION_{i:02d}",
                "Length": round(distance, 3),
                "ITSLinkID": f"ITS_L{i:08d}",
                "Maker": "한국도로공사",
                "UpdateDate": "20250822",
                "Version": "2021",
                "Remark": f"Generated link {i}",
                "HistType": "02A",
                "HistRemark": "자동 생성"
            }
            links.append(link)
        
        return links
    
    def generate_map_json(self, output_path: str):
        """Generate complete map JSON file"""
        nodes = self.generate_nodes_from_gps()
        links = self.generate_links_from_nodes(nodes)
        
        map_data = {
            "Node": nodes,
            "Link": links
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(map_data, f, indent=4, ensure_ascii=False)
        
        print(f"Generated map with {len(nodes)} nodes and {len(links)} links")
        print(f"Map saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate map JSON from ROS2 bag file')
    parser.add_argument('bag_path', help='Path to ROS2 bag directory')
    parser.add_argument('gps_topic', help='GPS topic name (e.g., /gps/fix)')
    parser.add_argument('-o', '--output', default='generated_map.json', 
                       help='Output JSON file path (default: generated_map.json)')
    parser.add_argument('-d', '--distance', type=float, default=3.0,
                       help='Minimum distance between GPS points in meters (default: 1.0)')
    parser.add_argument('-s', '--start-time', type=float, default=None,
                       help='Start time in seconds from bag start (default: from beginning)')
    parser.add_argument('-t', '--duration', type=float, default=None,
                       help='Duration in seconds to extract (default: until end)')
    parser.add_argument('-p', '--prefix', type=str, default='N',
                       help='Node ID prefix (default: N, e.g., Na, Nb, Route1)')
    
    args = parser.parse_args()
    
    try:
        generator = BagToMapGenerator(
            min_distance=args.distance,
            start_time=args.start_time,
            duration=args.duration,
            node_prefix=args.prefix
        )
        generator.read_bag_file(args.bag_path, args.gps_topic)
        generator.generate_map_json(args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()