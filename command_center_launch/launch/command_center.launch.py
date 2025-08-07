#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directories
    bae_mppi_launch_dir = os.path.join(get_package_share_directory('bae_mppi'), 'launch')
    behavior_planner_launch_dir = os.path.join(get_package_share_directory('simple_behavior_planner'), 'launch')
    sequential_planner_launch_dir = os.path.join(get_package_share_directory('sequential_global_planner'), 'launch')
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'map_file',
            default_value='mando_full_map.json',
            description='JSON map file for sequential planner'
        ),
        
        DeclareLaunchArgument(
            'current_position_topic',
            default_value='/odom',
            description='Current position topic for behavior planner'
        ),
        
        # 1. MPPI Controller - 가장 먼저 시작 (distributed launch)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bae_mppi_launch_dir, 'mppi_distributed.launch.py')
            )
        ),
        
        # 2. Simple Behavior Planner - 2초 후 시작
        TimerAction(
            period=2.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(behavior_planner_launch_dir, 'simple_behavior_planner.launch.py')
                    ),
                    launch_arguments={
                        'current_position_topic': LaunchConfiguration('current_position_topic'),
                        'goal_tolerance': '1.0'
                    }.items()
                )
            ]
        ),
        
        # 3. Sequential Global Planner - 4초 후 시작
        TimerAction(
            period=4.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(sequential_planner_launch_dir, 'sequential_planner.launch.py')
                    ),
                    launch_arguments={
                        'map_file': LaunchConfiguration('map_file'),
                        'auto_start': 'true',
                        'loop_path': 'false',
                        'publish_frequency': '1.0'
                    }.items()
                )
            ]
        )
    ])