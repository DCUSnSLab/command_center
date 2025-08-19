"""
Map loading utility for sequential global planner
Handles JSON map file parsing and data extraction
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional


class MapLoader:
    """Utility class for loading and parsing map JSON files"""
    
    def __init__(self, logger=None):
        """
        Initialize MapLoader
        
        Args:
            logger: ROS logger instance (optional)
        """
        self.logger = logger
        self.nodes_data = {}  # {node_id: node_data}
        self.links_data = []  # [link_data, ...]
        
    def load_map_file(self, file_path: str) -> Tuple[Dict, List, bool]:
        """
        Load JSON map file and extract nodes/links
        
        Args:
            file_path: Full path to the JSON map file
            
        Returns:
            Tuple of (nodes_dict, links_list, success_bool)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                map_data = json.load(file)
            
            # Clear existing data
            self.nodes_data.clear()
            self.links_data.clear()
            
            # Store nodes as dictionary for fast lookup
            nodes = map_data.get('Node', [])
            for node in nodes:
                self.nodes_data[node['ID']] = node
            
            # Store links
            self.links_data = map_data.get('Link', [])
            
            if self.logger:
                self.logger.info(
                    f'Successfully loaded map: {len(self.nodes_data)} nodes, '
                    f'{len(self.links_data)} links'
                )
            
            return self.nodes_data, self.links_data, True
            
        except FileNotFoundError:
            error_msg = f'Map file not found: {file_path}'
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f'ERROR: {error_msg}')
            return {}, [], False
            
        except json.JSONDecodeError as e:
            error_msg = f'Invalid JSON in map file: {str(e)}'
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f'ERROR: {error_msg}')
            return {}, [], False
            
        except Exception as e:
            error_msg = f'Failed to load map file: {str(e)}'
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f'ERROR: {error_msg}')
            return {}, [], False
    
    def get_map_path(self, package_name: str, map_filename: str) -> str:
        """
        Get full path to map file in ROS package
        
        Args:
            package_name: Name of the ROS package
            map_filename: Name of the map file
            
        Returns:
            Full path to the map file
        """
        try:
            from ament_index_python.packages import get_package_share_directory
            package_share = get_package_share_directory(package_name)
            return os.path.join(package_share, 'maps', map_filename)
        except ImportError:
            # Fallback for non-ROS environment
            return os.path.join('maps', map_filename)
        except Exception as e:
            if self.logger:
                self.logger.error(f'Failed to get package path: {str(e)}')
            return os.path.join('maps', map_filename)