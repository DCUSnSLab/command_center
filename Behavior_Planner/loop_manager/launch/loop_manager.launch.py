#!/usr/bin/env python3
# 순찰 루프 매니저.
#   ros2 launch loop_manager loop_manager.launch.py \
#       loop_goals:='[N0042, N0053]' use_sim_time:=true
# 전제: map_provider + scv_system 이 떠 있고, 그래프에 폐루프 링크가 있을 것.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('loop_goals', default_value='[N0042, N0053]',
                              description='순환할 노드 ID 목록 (인접 노드 연속 배치 금지)'),
        DeclareLaunchArgument('min_leg_time', default_value='15.0'),
        DeclareLaunchArgument('max_laps', default_value='0',
                              description='0 = 무한 반복'),

        Node(
            package='loop_manager',
            executable='loop_manager_node.py',
            name='loop_manager',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'loop_goals': LaunchConfiguration('loop_goals'),
                'min_leg_time': LaunchConfiguration('min_leg_time'),
                'max_laps': LaunchConfiguration('max_laps'),
            }],
        ),
    ])
