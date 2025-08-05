#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Declare launch argument for map file path
    map_file_arg = DeclareLaunchArgument(
        'map_file_path',
        default_value='/home/d2-521-30/repo/local_ws/src/GraphMap_Server/maps/3x3_map.json',
        description='Path to the JSON map file'
    )
    
    # Global path planner node
    global_path_planner_node = Node(
        package='scv_global_planner',
        executable='path_planner_node',
        name='global_path_planner_node',
        output='screen',
        parameters=[{
            'map_file_path': LaunchConfiguration('map_file_path')
        }]
    )
    
    return LaunchDescription([
        map_file_arg,
        global_path_planner_node,
    ])
