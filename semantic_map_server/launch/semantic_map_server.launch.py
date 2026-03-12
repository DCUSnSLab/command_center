#!/usr/bin/env python3
"""
Launch file for Semantic Map Server
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('semantic_map_server')

    # Default map file
    default_map = os.path.join(pkg_dir, 'maps', 'DCU_seg_map.json')

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'map_file',
            default_value=default_map,
            description='Path to semantic map JSON file'
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='map',
            description='TF frame ID for the map'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='1.0',
            description='Map publish rate in Hz'
        ),
        DeclareLaunchArgument(
            'use_utm',
            default_value='false',
            description='Use UTM coordinate conversion'
        ),

        # Semantic Map Server Node
        Node(
            package='semantic_map_server',
            executable='semantic_map_server_node.py',
            name='semantic_map_server',
            output='screen',
            parameters=[{
                'map_file': LaunchConfiguration('map_file'),
                'frame_id': LaunchConfiguration('frame_id'),
                'publish_rate': LaunchConfiguration('publish_rate'),
                'use_utm': LaunchConfiguration('use_utm'),
            }]
        ),
    ])
