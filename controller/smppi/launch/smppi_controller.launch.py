#!/usr/bin/env python3
"""
SMPPI Controller Launch File
Unified Nav2-based MPPI with SMPPI enhancements
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for SMPPI controller"""
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time if true'
    )
    
    enable_visualization_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='Enable RViz visualization'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='smppi_params.yaml',
        description='Configuration file name'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='smppi',
        description='Node namespace'
    )
    
    # Get configuration
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_visualization = LaunchConfiguration('enable_visualization')
    config_file = LaunchConfiguration('config_file')
    namespace = LaunchConfiguration('namespace')
    
    # Get package directory and config path
    smppi_dir = get_package_share_directory('smppi')
    default_config_path = os.path.join(smppi_dir, 'config', 'smppi_params.yaml')
    
    # Main SMPPI Controller Node
    smppi_controller_node = Node(
        package='smppi',
        executable='smppi_controller_node.py',
        name='smppi_controller',
        namespace=namespace,
        parameters=[
            default_config_path,
            {
                'use_sim_time': use_sim_time,
                'enable_visualization': enable_visualization
            }
        ],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    return LaunchDescription([
        # Launch arguments
        use_sim_time_arg,
        enable_visualization_arg,
        config_file_arg,
        namespace_arg,
        
        # Group all nodes
        GroupAction([
            smppi_controller_node,
        ])
    ])