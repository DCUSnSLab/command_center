#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Launch arguments
    current_position_topic_arg = DeclareLaunchArgument(
        'current_position_topic',
        default_value='/odom',
        description='Current vehicle position topic (Odometry)'
    )
    
    planned_path_topic_arg = DeclareLaunchArgument(
        'planned_path_topic',
        default_value='/planned_path_detailed',
        description='Planned path topic from global planner'
    )
    
    perception_topic_arg = DeclareLaunchArgument(
        'perception_topic',
        default_value='/perception',
        description='Perception results topic'
    )
    
    goal_status_topic_arg = DeclareLaunchArgument(
        'goal_status_topic',
        default_value='/goal_status',
        description='Goal achievement status topic'
    )
    
    subgoal_topic_arg = DeclareLaunchArgument(
        'subgoal_topic',
        default_value='/subgoal',
        description='Subgoal publishing topic'
    )
    
    emergency_stop_topic_arg = DeclareLaunchArgument(
        'emergency_stop_topic',
        default_value='/emergency_stop',
        description='Emergency stop command topic'
    )
    
    lookahead_distance_arg = DeclareLaunchArgument(
        'lookahead_distance',
        default_value='10.0',
        description='Lookahead distance for subgoal selection (meters)'
    )
    
    goal_tolerance_arg = DeclareLaunchArgument(
        'goal_tolerance',
        default_value='2.0',
        description='Goal tolerance distance (meters)'
    )
    
    # Simple Behavior Planner Node
    simple_behavior_planner_node = Node(
        package='simple_behavior_planner',
        executable='simple_behavior_planner_node.py',
        name='simple_behavior_planner',
        output='screen',
        parameters=[{
            'current_position_topic': LaunchConfiguration('current_position_topic'),
            'planned_path_topic': LaunchConfiguration('planned_path_topic'),
            'perception_topic': LaunchConfiguration('perception_topic'),
            'goal_status_topic': LaunchConfiguration('goal_status_topic'),
            'subgoal_topic': LaunchConfiguration('subgoal_topic'),
            'emergency_stop_topic': LaunchConfiguration('emergency_stop_topic'),
            'lookahead_distance': LaunchConfiguration('lookahead_distance'),
            'goal_tolerance': LaunchConfiguration('goal_tolerance'),
        }],
        remappings=[
            # Add topic remappings if needed
        ]
    )
    
    return LaunchDescription([
        current_position_topic_arg,
        planned_path_topic_arg,
        perception_topic_arg,
        goal_status_topic_arg,
        subgoal_topic_arg,
        emergency_stop_topic_arg,
        lookahead_distance_arg,
        goal_tolerance_arg,
        simple_behavior_planner_node,
    ])