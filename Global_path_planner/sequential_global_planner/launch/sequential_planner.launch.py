#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false') # 시뮬레이션 환경인 경우 true, 밖이면 false

    map_file = PathJoinSubstitution([
                FindPackageShare('sequential_global_planner'),
                'maps',
                '20250807_v3.json'
            ])

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'map_file',
            default_value=map_file,
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

        DeclareLaunchArgument(
            'explicit_endpoints',
            default_value='false',
            description='Pin start/goal node IDs into the path. Off by default '
                        'so the behavior planner starts from the node nearest '
                        'the robot instead of the route head.'
        ),

        DeclareLaunchArgument(
            'goal_node', default_value='',
            description='Route to this node at startup (arbitrary-goal mode). '
                        'Empty = publish the whole ordered route.'),
        DeclareLaunchArgument(
            'start_node', default_value='',
            description='With goal_node: route FROM this node and pin it as '
                        'the start. Empty = nearest node to the robot.'),
        DeclareLaunchArgument(
            'map_origin_source', default_value='datum',
            description="'datum' (fixed UTM below; works for maps whose node0 "
                        "is not the datum) or 'first_node' (legacy)."),
        DeclareLaunchArgument(
            'datum_utm_easting', default_value='482232.7',
            description='UTM easting of the localization datum'),
        DeclareLaunchArgument(
            'datum_utm_northing', default_value='3974384.47',
            description='UTM northing of the localization datum'),

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
                'explicit_endpoints': ParameterValue(
                    LaunchConfiguration('explicit_endpoints'), value_type=bool),
                'initial_goal_node': ParameterValue(
                    LaunchConfiguration('goal_node'), value_type=str),
                'initial_start_node': ParameterValue(
                    LaunchConfiguration('start_node'), value_type=str),
                'map_origin_source': ParameterValue(
                    LaunchConfiguration('map_origin_source'), value_type=str),
                'datum_utm_easting': ParameterValue(
                    LaunchConfiguration('datum_utm_easting'), value_type=float),
                'datum_utm_northing': ParameterValue(
                    LaunchConfiguration('datum_utm_northing'), value_type=float),
                'use_sim_time': use_sim_time
            }],
            remappings=[
                ('/planned_path_detailed', '/planned_path_detailed'),
                ('/sequential_path_nav', '/sequential_path_nav'),
                ('/sequential_path_markers', '/sequential_path_markers'),
            ]
        )
    ])