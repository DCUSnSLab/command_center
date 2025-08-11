#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    map_file = PathJoinSubstitution([
                FindPackageShare('sequential_global_planner'),
                'maps',
                'm.json'
            ])

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'map_file',
            default_value='m.json',
            description='JSON map file to load'
        ),
        
        DeclareLaunchArgument(
            'auto_start',
            default_value='true',
            description='Auto start publishing path'
        ),
        
        DeclareLaunchArgument(
            'loop_path',
            default_value='false',
            description='Create loop by connecting last to first node'
        ),
        
        DeclareLaunchArgument(
            'publish_frequency',
            default_value='1.0',
            description='Path publishing frequency in Hz'
        ),
        
        DeclareLaunchArgument(
            'gps_topic',
            default_value='/gps/fix',
            description='GPS topic name for reference point'
        ),
        
        # Sequential planner node
        Node(
            package='sequential_global_planner',
            executable='sequential_planner_node.py',
            name='sequential_planner',
            output='screen',
            parameters=[{
                'map_file': LaunchConfiguration('map_file'),
                'auto_start': LaunchConfiguration('auto_start'),
                'loop_path': LaunchConfiguration('loop_path'),
                'publish_frequency': LaunchConfiguration('publish_frequency'),
                'gps_topic': LaunchConfiguration('gps_topic'),
            }],
            remappings=[
                ('/planned_path_detailed', '/planned_path_detailed'),
                ('/sequential_path_nav', '/sequential_path_nav'),
                ('/sequential_path_markers', '/sequential_path_markers'),
            ]
        )
    ])