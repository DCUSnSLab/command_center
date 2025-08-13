#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true') # 시뮬레이션 환경인 경우 true, 밖이면 false
    
    # Map service node
    map_service_node = Node(
        package='gmserver',
        executable='map_service_node',
        name='map_service_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )
    
    return LaunchDescription([
        map_service_node,
    ])