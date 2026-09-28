#!/usr/bin/env python3
"""scv_bt_planner 런치.

  mode:=active  기존 BP 토픽으로 발행 (simple_behavior_planner 를 대체)
  mode:=shadow  /bt/ 접두 토픽으로만 발행 — simple BP 와 같은 주행에서 결정만 비교

field_drive.launch.py 의 simple_behavior_planner include 와 같은 인자 이름을 쓴다
(current_position_topic, planned_path_topic, probe_enabled 등) — A/B 시 한 줄 교체.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('scv_bt_planner')
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='shadow',
                              description="active | shadow"),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('current_position_topic', default_value='/odometry/global'),
        DeclareLaunchArgument('planned_path_topic', default_value='/planned_path_detailed'),
        DeclareLaunchArgument('map_file_path', default_value='',
                              description='graph.json (Zone 조회). 비우면 전 노드 sidewalk'),
        DeclareLaunchArgument('tree_file',
                              default_value=os.path.join(share, 'behavior_trees', 'scv_behavior.xml')),
        DeclareLaunchArgument('profiles_file',
                              default_value=os.path.join(share, 'config', 'behavior_profiles.yaml')),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(share, 'config', 'bt_planner_params.yaml')),
        DeclareLaunchArgument('probe_enabled', default_value='false'),
        DeclareLaunchArgument('groot2_port', default_value='0',
                              description='>0 이면 Groot2 실시간 감시 포트 (기본 1667)'),
        Node(
            package='scv_bt_planner',
            executable='bt_planner_node',
            name='bt_planner',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'mode': LaunchConfiguration('mode'),
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'current_position_topic': LaunchConfiguration('current_position_topic'),
                    'planned_path_topic': LaunchConfiguration('planned_path_topic'),
                    'map_file_path': LaunchConfiguration('map_file_path'),
                    'tree_file': LaunchConfiguration('tree_file'),
                    'profiles_file': LaunchConfiguration('profiles_file'),
                    'probe.enabled': LaunchConfiguration('probe_enabled'),
                    'groot2_port': LaunchConfiguration('groot2_port'),
                },
            ],
        ),
    ])
