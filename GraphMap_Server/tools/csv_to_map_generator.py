#!/usr/bin/env python3

import argparse
import json
import csv
import sys
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import pyproj
except ImportError as e:
    print(f"Required dependencies not found: {e}")
    print("Install with: pip install pyproj")
    sys.exit(1)


class CSVToMapGenerator:
    def __init__(self, min_distance=1.0, filter_option=None):
        self.gps_points = []
        self.utm_transformer = None
        self.min_distance = min_distance  # Minimum distance in meters
        self.filter_option = filter_option  # Filter by option value
        
    def setup_utm_transformer(self, lat: float, lon: float):
        """Setup UTM transformer based on first GPS coordinate"""
        utm_zone = int((lon + 180) / 6) + 1
        utm_crs = f"EPSG:{32600 + utm_zone}"  # Northern hemisphere
        if lat < 0:
            utm_crs = f"EPSG:{32700 + utm_zone}"  # Southern hemisphere
        
        self.utm_transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        return utm_zone
    
    def gps_to_utm(self, lat: float, lon: float) -> Tuple[float, float, str]:
        """Convert GPS coordinates to UTM"""
        if self.utm_transformer is None:
            zone = self.setup_utm_transformer(lat, lon)
            zone_str = f"{zone}N" if lat >= 0 else f"{zone}S"
        else:
            zone_str = "52N"  # Default for Seoul area
            
        easting, northing = self.utm_transformer.transform(lon, lat)
        return easting, northing, zone_str
    
    def read_csv_file(self, csv_path: str):
        """Read GPS data from CSV file"""
        csv_file = Path(csv_path)
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"Reading CSV file: {csv_path}")
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check required columns
            required_cols = ['llatitude', 'longitude']
            if not all(col in reader.fieldnames for col in required_cols):
                # Try alternative column names
                alt_cols = {'lat': 'llatitude', 'lon': 'longitude', 
                           'latitude': 'llatitude', 'lng': 'longitude'}
                
                for row in reader:
                    for alt_col, std_col in alt_cols.items():
                        if alt_col in row and std_col not in row:
                            row[std_col] = row[alt_col]
                    break
                
                # Reset file pointer
                f.seek(0)
                reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 2):  # Start from 2 (header is line 1)
                try:
                    # Extract GPS coordinates
                    lat = float(row.get('llatitude', row.get('latitude', row.get('lat', 0))))
                    lon = float(row.get('longitude', row.get('lng', row.get('lon', 0))))
                    
                    if lat == 0 or lon == 0:
                        print(f"Warning: Invalid coordinates at row {row_num}")
                        continue
                    
                    # Check option filter if specified
                    if self.filter_option is not None:
                        option_val = row.get('option', '0')
                        try:
                            option_val = int(option_val)
                            if option_val != self.filter_option:
                                continue
                        except ValueError:
                            continue
                    
                    # Filter points based on minimum distance
                    if self.should_add_point(lat, lon):
                        # Store additional data if available
                        extra_data = {
                            'seq': row.get('seq', ''),
                            'option': row.get('option', '0')
                        }
                        self.gps_points.append((lat, lon, extra_data))
                        
                except ValueError as e:
                    print(f"Warning: Invalid data at row {row_num}: {e}")
                    continue
        
        print(f"Read {len(self.gps_points)} GPS points from CSV file")
        
        if self.filter_option is not None:
            print(f"Filtered by option={self.filter_option}")
    
    def should_add_point(self, lat: float, lon: float) -> bool:
        """Check if GPS point should be added based on minimum distance"""
        if not self.gps_points:
            return True  # Always add first point
        
        # Get last added point
        last_lat, last_lon, _ = self.gps_points[-1]
        
        # Calculate distance using UTM coordinates for accuracy
        if self.utm_transformer is None:
            self.setup_utm_transformer(lat, lon)
        
        # Convert both points to UTM
        easting1, northing1, _ = self.gps_to_utm(last_lat, last_lon)
        easting2, northing2, _ = self.gps_to_utm(lat, lon)
        
        # Calculate Euclidean distance in meters
        distance = math.sqrt((easting2 - easting1)**2 + (northing2 - northing1)**2)
        
        return distance >= self.min_distance
    
    def generate_nodes_from_gps(self) -> List[Dict[str, Any]]:
        """Generate map nodes from GPS points"""
        if not self.gps_points:
            raise ValueError("No GPS points available")
        
        nodes = []
        for i, (lat, lon, extra_data) in enumerate(self.gps_points):
            easting, northing, zone = self.gps_to_utm(lat, lon)
            
            # Use sequence number from CSV if available, otherwise use index
            seq_id = extra_data.get('seq', f'{i:06d}')
            
            node = {
                "ID": f"N{seq_id}",
                "AdminCode": "110",
                "NodeType": 1,
                "ITSNodeID": f"ITS_N{seq_id}",
                "Maker": "자율주행연구소",
                "UpdateDate": "20250902",
                "Version": "2025",
                "Remark": f"CSV Generated node {i} (seq={seq_id}, option={extra_data.get('option', '0')})",
                "HistType": "02A",
                "HistRemark": "CSV 자동 생성",
                "GpsInfo": {
                    "Lat": round(lat, 6),
                    "Long": round(lon, 6),
                    "Alt": 0.0
                },
                "UtmInfo": {
                    "Easting": round(easting, 2),
                    "Northing": round(northing, 2),
                    "Zone": zone
                },
                "ExtraInfo": {
                    "OriginalSeq": seq_id,
                    "Option": extra_data.get('option', '0')
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
            
            # Determine road characteristics based on option values
            from_option = int(from_node["ExtraInfo"]["Option"])
            to_option = int(to_node["ExtraInfo"]["Option"])
            
            # Set link type based on option values
            if from_option == 10 or to_option == 10:
                link_type = 2  # Special section (curve, intersection)
                road_rank = 2
            elif from_option > 0 or to_option > 0:
                link_type = 1  # Important section
                road_rank = 1
            else:
                link_type = 3  # Normal road
                road_rank = 1
            
            link = {
                "ID": f"L{i:08d}",
                "AdminCode": "110",
                "RoadRank": road_rank,
                "RoadType": 1,
                "RoadNo": "CSV_ROUTE",
                "LinkType": link_type,
                "LaneNo": 2,
                "R_LinkID": f"R_{i:04d}",
                "L_LinkID": f"L_{i:04d}",
                "FromNodeID": from_node["ID"],
                "ToNodeID": to_node["ID"],
                "SectionID": f"CSV_SECTION_{i:02d}",
                "Length": round(distance, 3),
                "ITSLinkID": f"ITS_L{i:08d}",
                "Maker": "자율주행연구소",
                "UpdateDate": "20250902",
                "Version": "2025",
                "Remark": f"CSV Generated link {i} (type={link_type})",
                "HistType": "02A",
                "HistRemark": "CSV 자동 생성"
            }
            links.append(link)
        
        return links
    
    def generate_statistics(self, nodes: List[Dict[str, Any]], links: List[Dict[str, Any]]):
        """Generate statistics about the generated map"""
        stats = {
            "total_nodes": len(nodes),
            "total_links": len(links),
            "total_distance_km": sum(link["Length"] for link in links),
            "option_distribution": {},
            "link_type_distribution": {}
        }
        
        # Count option distribution
        for node in nodes:
            option = node["ExtraInfo"]["Option"]
            stats["option_distribution"][option] = stats["option_distribution"].get(option, 0) + 1
        
        # Count link type distribution
        for link in links:
            link_type = link["LinkType"]
            stats["link_type_distribution"][link_type] = stats["link_type_distribution"].get(link_type, 0) + 1
        
        return stats
    
    def generate_map_json(self, output_path: str):
        """Generate complete map JSON file"""
        nodes = self.generate_nodes_from_gps()
        links = self.generate_links_from_nodes(nodes)
        stats = self.generate_statistics(nodes, links)
        
        map_data = {
            "Node": nodes,
            "Link": links,
            "Statistics": stats
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(map_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n=== Map Generation Complete ===")
        print(f"Generated map with {len(nodes)} nodes and {len(links)} links")
        print(f"Total distance: {stats['total_distance_km']:.3f} km")
        print(f"Option distribution: {stats['option_distribution']}")
        print(f"Link type distribution: {stats['link_type_distribution']}")
        print(f"Map saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate map JSON from CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 csv_to_map_generator.py mando.csv -o mando_map.json
  
  # Filter by distance
  python3 csv_to_map_generator.py mando.csv -o filtered_map.json -d 5.0
  
  # Filter by option value (e.g., only special sections)
  python3 csv_to_map_generator.py mando.csv -o special_map.json --filter-option 10
  
  # Combine filters
  python3 csv_to_map_generator.py mando.csv -o combined_map.json -d 2.0 --filter-option 0

Expected CSV format:
  seq,llatitude,longitude,llatitude_utm,longitude_utm,option
  135396,37.23965631,126.7736361,302516.9182,4123781.294,0
        """
    )
    
    parser.add_argument('csv_path', help='Path to CSV file with GPS data')
    parser.add_argument('-o', '--output', default='csv_generated_map.json', 
                       help='Output JSON file path (default: csv_generated_map.json)')
    parser.add_argument('-d', '--distance', type=float, default=1.0,
                       help='Minimum distance between GPS points in meters (default: 1.0)')
    parser.add_argument('--filter-option', type=int, metavar='N',
                       help='Filter points by option value (e.g., 0, 10)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics without generating map')
    
    args = parser.parse_args()
    
    try:
        generator = CSVToMapGenerator(
            min_distance=args.distance,
            filter_option=args.filter_option
        )
        
        generator.read_csv_file(args.csv_path)
        
        if args.stats_only:
            # Just show statistics
            nodes = generator.generate_nodes_from_gps()
            links = generator.generate_links_from_nodes(nodes)
            stats = generator.generate_statistics(nodes, links)
            
            print(f"\n=== CSV Analysis Results ===")
            print(f"Total points: {stats['total_nodes']}")
            print(f"Total distance: {stats['total_distance_km']:.3f} km")
            print(f"Option distribution: {stats['option_distribution']}")
        else:
            generator.generate_map_json(args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()