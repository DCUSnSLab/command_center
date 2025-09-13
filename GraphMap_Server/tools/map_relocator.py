#!/usr/bin/env python3
"""
Map Relocator Tool
맵 좌표 이동 및 회전 변환 도구

기존 맵의 첫 번째 노드를 기준점으로 전체 맵을 새로운 위치로 이동시키고,
선택적으로 회전 변환을 적용합니다.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    import pyproj
except ImportError as e:
    print(f"Required dependencies not found: {e}")
    print("Install with: pip install pyproj")
    sys.exit(1)


class MapRelocator:
    def __init__(self, rotation_angle=0.0):
        """
        Initialize map relocator
        
        Args:
            rotation_angle: Rotation angle in degrees (clockwise positive)
        """
        self.rotation_angle = math.radians(rotation_angle)  # Convert to radians
        self.original_data = None
        self.relocated_data = None
        
    def load_map(self, map_path: str) -> Dict[str, Any]:
        """Load map JSON file"""
        map_file = Path(map_path)
        
        if not map_file.exists():
            raise FileNotFoundError(f"Map file not found: {map_path}")
        
        print(f"Loading map file: {map_path}")
        
        with open(map_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'Node' not in data:
            raise ValueError("Invalid map format: 'Node' section not found")
        
        print(f"Loaded {len(data['Node'])} nodes")
        if 'Link' in data:
            print(f"Loaded {len(data['Link'])} links")
        
        self.original_data = data
        return data
    
    def setup_utm_transformer(self, lat: float, lon: float):
        """Setup UTM transformer based on GPS coordinate"""
        utm_zone = int((lon + 180) / 6) + 1
        utm_crs = f"EPSG:{32600 + utm_zone}"  # Northern hemisphere
        if lat < 0:
            utm_crs = f"EPSG:{32700 + utm_zone}"  # Southern hemisphere
        
        transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        zone_str = f"{utm_zone}N" if lat >= 0 else f"{utm_zone}S"
        return transformer, zone_str
    
    def gps_to_utm(self, lat: float, lon: float, transformer) -> Tuple[float, float]:
        """Convert GPS coordinates to UTM"""
        easting, northing = transformer.transform(lon, lat)
        return easting, northing
    
    def rotate_point(self, x: float, y: float, cx: float, cy: float, angle: float) -> Tuple[float, float]:
        """
        Rotate point (x, y) around center (cx, cy) by angle (radians)
        
        Args:
            x, y: Point coordinates
            cx, cy: Center of rotation
            angle: Rotation angle in radians (clockwise positive)
        
        Returns:
            Rotated coordinates (x', y')
        """
        # Translate to origin
        dx = x - cx
        dy = y - cy
        
        # Rotate (clockwise rotation)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        new_x = dx * cos_a + dy * sin_a
        new_y = -dx * sin_a + dy * cos_a
        
        # Translate back
        return new_x + cx, new_y + cy
    
    def relocate_map(self, target_lat: float, target_lon: float) -> Dict[str, Any]:
        """
        Relocate entire map to new position
        
        Args:
            target_lat: Target latitude for first node
            target_lon: Target longitude for first node
        
        Returns:
            Relocated map data
        """
        if not self.original_data:
            raise ValueError("No map data loaded")
        
        nodes = self.original_data['Node']
        if not nodes:
            raise ValueError("No nodes found in map data")
        
        # Get first node as reference point
        first_node = nodes[0]
        ref_lat = first_node['GpsInfo']['Lat']
        ref_lon = first_node['GpsInfo']['Long']
        
        print(f"Reference node: {first_node['ID']} at ({ref_lat:.6f}, {ref_lon:.6f})")
        print(f"Moving to: ({target_lat:.6f}, {target_lon:.6f})")
        
        if abs(self.rotation_angle) > 0:
            print(f"Applying rotation: {math.degrees(self.rotation_angle):.1f} degrees")
        
        # Setup UTM transformer for target location
        utm_transformer, utm_zone = self.setup_utm_transformer(target_lat, target_lon)
        
        # Create copy of original data
        relocated_data = json.loads(json.dumps(self.original_data))
        
        # Calculate rotation center (first node position in relative coordinates)
        rotation_center_lat = target_lat
        rotation_center_lon = target_lon
        
        # Process each node
        updated_nodes = 0
        for i, node in enumerate(relocated_data['Node']):
            try:
                # Get original coordinates
                orig_lat = node['GpsInfo']['Lat']
                orig_lon = node['GpsInfo']['Long']
                
                # Calculate relative position from reference node
                lat_offset = orig_lat - ref_lat
                lon_offset = orig_lon - ref_lon
                
                # Apply rotation if specified
                if abs(self.rotation_angle) > 0:
                    # Rotate the offset around origin
                    lat_offset, lon_offset = self.rotate_point(
                        lat_offset, lon_offset, 0, 0, self.rotation_angle
                    )
                
                # Calculate new absolute position
                new_lat = target_lat + lat_offset
                new_lon = target_lon + lon_offset
                
                # Convert to UTM
                new_easting, new_northing = self.gps_to_utm(new_lat, new_lon, utm_transformer)
                
                # Update node data
                node['GpsInfo']['Lat'] = round(new_lat, 8)
                node['GpsInfo']['Long'] = round(new_lon, 8)
                node['UtmInfo']['Easting'] = round(new_easting, 2)
                node['UtmInfo']['Northing'] = round(new_northing, 2)
                node['UtmInfo']['Zone'] = utm_zone
                
                updated_nodes += 1
                
                if updated_nodes % 100 == 0:
                    print(f"Processed {updated_nodes} nodes...")
                    
            except Exception as e:
                print(f"Warning: Failed to process node {node.get('ID', i)}: {e}")
                continue
        
        print(f"Successfully relocated {updated_nodes} nodes")
        
        self.relocated_data = relocated_data
        return relocated_data
    
    def save_map(self, output_path: str):
        """Save relocated map to file"""
        if not self.relocated_data:
            raise ValueError("No relocated data to save")
        
        output_file = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.relocated_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved relocated map to: {output_path}")
    
    def preview_changes(self, target_lat: float, target_lon: float, num_preview=5):
        """Preview coordinate changes for first few nodes"""
        if not self.original_data:
            raise ValueError("No map data loaded")
        
        nodes = self.original_data['Node'][:num_preview]
        ref_lat = self.original_data['Node'][0]['GpsInfo']['Lat']
        ref_lon = self.original_data['Node'][0]['GpsInfo']['Long']
        
        print(f"\n=== Preview Changes ===")
        print(f"Reference: ({ref_lat:.6f}, {ref_lon:.6f}) → ({target_lat:.6f}, {target_lon:.6f})")
        if abs(self.rotation_angle) > 0:
            print(f"Rotation: {math.degrees(self.rotation_angle):.1f} degrees")
        print()
        
        for node in nodes:
            orig_lat = node['GpsInfo']['Lat']
            orig_lon = node['GpsInfo']['Long']
            
            # Calculate relative position
            lat_offset = orig_lat - ref_lat
            lon_offset = orig_lon - ref_lon
            
            # Apply rotation if specified
            if abs(self.rotation_angle) > 0:
                lat_offset, lon_offset = self.rotate_point(
                    lat_offset, lon_offset, 0, 0, self.rotation_angle
                )
            
            # Calculate new position
            new_lat = target_lat + lat_offset
            new_lon = target_lon + lon_offset
            
            print(f"{node['ID']}: ({orig_lat:.6f}, {orig_lon:.6f}) → ({new_lat:.6f}, {new_lon:.6f})")
    
    def calculate_map_bounds(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate map bounds"""
        if 'Node' not in data or not data['Node']:
            return {}
        
        lats = [node['GpsInfo']['Lat'] for node in data['Node']]
        lons = [node['GpsInfo']['Long'] for node in data['Node']]
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
            'center_lat': (min(lats) + max(lats)) / 2,
            'center_lon': (min(lons) + max(lons)) / 2
        }


def main():
    parser = argparse.ArgumentParser(
        description='Relocate map to new coordinates with optional rotation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic relocation
  python3 map_relocator.py mando_test_2.0.json -lat 37.5665 -lon 126.9780 -o gangnam_map.json
  
  # With rotation (45 degrees clockwise)
  python3 map_relocator.py mando_test_2.0.json -lat 35.1595 -lon 129.0756 -r 45.0 -o busan_map.json
  
  # Preview only
  python3 map_relocator.py mando_test_2.0.json -lat 37.5665 -lon 126.9780 --preview-only
        """
    )
    
    parser.add_argument('input_map', help='Input map JSON file')
    parser.add_argument('-lat', '--latitude', type=float, required=True,
                       help='Target latitude for first node')
    parser.add_argument('-lon', '--longitude', type=float, required=True,
                       help='Target longitude for first node')
    parser.add_argument('-r', '--rotation', type=float, default=0.0,
                       help='Rotation angle in degrees (clockwise positive, default: 0.0)')
    parser.add_argument('-o', '--output', default='relocated_map.json',
                       help='Output JSON file (default: relocated_map.json)')
    parser.add_argument('--preview-only', action='store_true',
                       help='Only show preview without saving')
    parser.add_argument('--preview-count', type=int, default=5,
                       help='Number of nodes to show in preview (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Validate coordinates
        if not (-90 <= args.latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not (-180 <= args.longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")
        
        # Initialize relocator
        relocator = MapRelocator(rotation_angle=args.rotation)
        
        # Load original map
        relocator.load_map(args.input_map)
        
        # Show original bounds
        original_bounds = relocator.calculate_map_bounds(relocator.original_data)
        if original_bounds:
            print(f"\nOriginal map bounds:")
            print(f"  Latitude: {original_bounds['min_lat']:.6f} to {original_bounds['max_lat']:.6f}")
            print(f"  Longitude: {original_bounds['min_lon']:.6f} to {original_bounds['max_lon']:.6f}")
            print(f"  Center: ({original_bounds['center_lat']:.6f}, {original_bounds['center_lon']:.6f})")
        
        # Preview changes
        relocator.preview_changes(args.latitude, args.longitude, args.preview_count)
        
        if args.preview_only:
            print("\nPreview mode - no files were modified")
            return
        
        # Relocate map
        print(f"\nRelocating map...")
        relocated_data = relocator.relocate_map(args.latitude, args.longitude)
        
        # Show new bounds
        new_bounds = relocator.calculate_map_bounds(relocated_data)
        if new_bounds:
            print(f"\nNew map bounds:")
            print(f"  Latitude: {new_bounds['min_lat']:.6f} to {new_bounds['max_lat']:.6f}")
            print(f"  Longitude: {new_bounds['min_lon']:.6f} to {new_bounds['max_lon']:.6f}")
            print(f"  Center: ({new_bounds['center_lat']:.6f}, {new_bounds['center_lon']:.6f})")
        
        # Save relocated map
        relocator.save_map(args.output)
        
        print(f"\n✓ Map relocation completed successfully!")
        print(f"✓ Input: {args.input_map}")
        print(f"✓ Output: {args.output}")
        if abs(args.rotation) > 0:
            print(f"✓ Rotation: {args.rotation:.1f} degrees")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()