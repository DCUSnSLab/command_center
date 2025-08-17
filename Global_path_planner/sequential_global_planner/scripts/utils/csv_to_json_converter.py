#!/usr/bin/env python3
"""
CSV to JSON Converter for Sequential Global Path Planner
Converts mando.csv format to 3x3_map.json compatible format
"""

import csv
import json
import math
from typing import List, Dict, Any


class CSVToJSONConverter:
    """Converts CSV waypoint data to JSON map format"""
    
    def __init__(self):
        self.nodes = []
        self.links = []
        self.node_counter = 0
        
    def read_csv(self, csv_file_path: str) -> List[Dict]:
        """Read CSV file and return list of waypoint dictionaries"""
        waypoints = []
        
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                waypoint = {
                    'seq': int(row['seq']),
                    'latitude': float(row['llatitude']),
                    'longitude': float(row['longitude']),
                    'utm_easting': float(row['llatitude_utm']),
                    'utm_northing': float(row['longitude_utm']),
                    'option': int(row['option'])
                }
                waypoints.append(waypoint)
                
        print(f"Read {len(waypoints)} waypoints from CSV")
        return waypoints
    
    def create_nodes_from_waypoints(self, waypoints: List[Dict], 
                                  node_spacing: int = 10) -> None:
        """Create nodes from waypoints with specified spacing"""
        
        # Sample waypoints at regular intervals to avoid too many nodes
        sampled_waypoints = waypoints[::node_spacing]  # Take every Nth waypoint
        
        for i, wp in enumerate(sampled_waypoints):
            node_id = f"N{i:04d}"  # N0000, N0001, etc.
            
            node = {
                "ID": node_id,
                "AdminCode": "110",
                "NodeType": 1,
                "ITSNodeID": f"ITS_{node_id}",
                "Maker": "한국도로공사",
                "UpdateDate": "20250418",
                "Version": "2021",
                "Remark": f"Sequential waypoint {i} (seq={wp['seq']})",
                "HistType": "02A",
                "HistRemark": "순차 경로 노드",
                "GpsInfo": {
                    "Lat": wp['latitude'],
                    "Long": wp['longitude'],
                    "Alt": 0.0
                },
                "UtmInfo": {
                    "Easting": wp['utm_easting'],
                    "Northing": wp['utm_northing'],
                    "Zone": "52N"
                }
            }
            
            self.nodes.append(node)
            
        print(f"Created {len(self.nodes)} nodes from waypoints")
    
    def create_links_between_nodes(self) -> None:
        """Create sequential links between adjacent nodes"""
        
        for i in range(len(self.nodes) - 1):
            from_node = self.nodes[i]
            to_node = self.nodes[i + 1]
            
            # Calculate distance between nodes
            dx = to_node["UtmInfo"]["Easting"] - from_node["UtmInfo"]["Easting"]
            dy = to_node["UtmInfo"]["Northing"] - from_node["UtmInfo"]["Northing"]
            distance = math.sqrt(dx*dx + dy*dy)
            
            link_id = f"L{i:08d}"  # L00000000, L00000001, etc.
            
            link = {
                "ID": link_id,
                "AdminCode": "110",
                "RoadRank": 1,
                "RoadType": 1,
                "RoadNo": f"{20 + (i % 10)}",  # Road numbers 20-29
                "LinkType": 3,
                "LaneNo": 2,
                "R_LinkID": f"R_{i:04d}",
                "L_LinkID": f"L_{i:04d}",
                "FromNodeID": from_node["ID"],
                "ToNodeID": to_node["ID"],
                "SectionID": f"SEQ_SECTION_{i:02d}",
                "Length": round(distance, 2),
                "ITSLinkID": f"ITS_{link_id}",
                "Maker": "한국도로공사",
                "UpdateDate": "20250418",
                "Version": "2021",
                "Remark": f"순차 연결 {from_node['ID']}-{to_node['ID']}",
                "HistType": "02A",
                "HistRemark": "순차 경로 링크"
            }
            
            self.links.append(link)
            
        print(f"Created {len(self.links)} links between nodes")
    
    def export_to_json(self, output_file_path: str) -> None:
        """Export nodes and links to JSON file in 3x3_map.json format"""
        
        json_data = {
            "Node": self.nodes,
            "Link": self.links
        }
        
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(json_data, file, indent=4, ensure_ascii=False)
            
        print(f"Exported JSON to {output_file_path}")
        print(f"Total nodes: {len(self.nodes)}, Total links: {len(self.links)}")
    
    def convert_csv_to_json(self, csv_file: str, json_file: str, 
                           node_spacing: int = 10) -> None:
        """Complete conversion process from CSV to JSON"""
        print(f"Converting {csv_file} to {json_file} with node spacing {node_spacing}")
        
        # Read CSV waypoints
        waypoints = self.read_csv(csv_file)
        
        # Create nodes with spacing to reduce total count
        self.create_nodes_from_waypoints(waypoints, node_spacing)
        
        # Create sequential links
        self.create_links_between_nodes()
        
        # Export to JSON
        self.export_to_json(json_file)
        
        print("Conversion completed successfully!")


def main():
    """Main conversion script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CSV waypoints to JSON map format')
    parser.add_argument('csv_file', help='Input CSV file path')
    parser.add_argument('json_file', help='Output JSON file path')
    parser.add_argument('--spacing', type=int, default=10, 
                       help='Node spacing (take every Nth waypoint, default: 10)')
    
    args = parser.parse_args()
    
    converter = CSVToJSONConverter()
    converter.convert_csv_to_json(args.csv_file, args.json_file, args.spacing)


if __name__ == '__main__':
    main()