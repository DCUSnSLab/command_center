#!/usr/bin/env python3
"""
Launch file for local costmap with manual lifecycle management
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Get package directory
    pkg_share = get_package_share_directory('local_costmap_2d')
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(pkg_share, 'config', 'local_costmap.yaml'),
        description='Path to config file'
    )
    
    # Create lifecycle node (but don't auto-configure)
    local_costmap_node = Node(
        package='local_costmap_2d',
        executable='local_costmap_node',
        name='local_costmap',
        namespace='',
        parameters=[LaunchConfiguration('config_file')],
        output='screen'
    )
    
    return LaunchDescription([
        config_file_arg,
        local_costmap_node
    ])