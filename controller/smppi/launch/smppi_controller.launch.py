#!/usr/bin/env python3
"""
SMPPI Controller Launch File
Modular 3-node architecture for optimal performance:
- Costmap Processing Node: Costmap-based obstacle processing (replaces sensor node)
- MPPI Main Node: Core optimization and control with grid-based collision detection
- Visualization Node: RViz markers and visualization
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
        default_value='false',
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
    
    # Costmap Processing Node (replaces sensor_processor_node)
    costmap_processor_node = Node(
        package='smppi',
        executable='costmap_processor_node.py',
        name='costmap_processor',
        namespace=namespace,
        parameters=[
            default_config_path,
            {
                'use_sim_time': use_sim_time,
            }
        ],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    # Corridor Keepout Node (sidewalk keep-in; /costmap -> /costmap_keepout)
    corridor_keepout_node = Node(
        package='smppi',
        executable='corridor_keepout_node.py',
        name='corridor_keepout',
        namespace=namespace,
        parameters=[
            default_config_path,
            {
                'use_sim_time': use_sim_time,
            }
        ],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info']
    )

    # MPPI Main Controller Node
    mppi_main_node = Node(
        package='smppi',
        executable='mppi_main_node.py',
        name='mppi_main_controller',
        namespace=namespace,
        parameters=[
            default_config_path,
            {
                'use_sim_time': use_sim_time,
            }
        ],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    # Visualization Node (conditional)
    visualization_node = Node(
        package='smppi',
        executable='visualization_node.py',
        name='visualization',
        namespace=namespace,
        condition=IfCondition(enable_visualization),
        parameters=[
            default_config_path,
            {
                'use_sim_time': use_sim_time,
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
            costmap_processor_node,
            corridor_keepout_node,
            mppi_main_node,
            visualization_node,
        ])
    ])