#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')  # 시뮬레이션 환경인 경우 true, 밖이면 false

    # Global path planner node
    # 그래프/좌표계는 map_provider 토픽(/map_provider_node/graph, /map_provider_node/utm)에서 수신하므로
    # map_file_path 파라미터는 더 이상 사용하지 않는다.
    global_path_planner_node = Node(
        package='scv_global_planner',
        executable='path_planner_node',
        name='global_path_planner_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    return LaunchDescription([
        global_path_planner_node,
    ])
