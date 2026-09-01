#!/usr/bin/env python3
# 전역 경로 플래너.
#
# 그래프·좌표계는 map_provider 토픽(/map_provider_node/graph, /map_provider_node/utm)에서
# 받는다. map_file_path 파라미터는 더 이상 쓰지 않는다.
#
# 모드 두 가지:
#   route_mode:=astar     (기본) /goal_pose 까지 최단 경로
#   route_mode:=sequence  route_nodes 를 적은 순서대로 — 최단이 아니어도 그 길로 간다
#
# 예) 순찰 루프
#   ros2 launch scv_global_planner path_planner.launch.py \
#       route_mode:=sequence route_nodes:="N0394,N0426,NX0011,NX0020" route_loop:=true
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    route_mode = LaunchConfiguration('route_mode')
    route_nodes = LaunchConfiguration('route_nodes')
    route_loop = LaunchConfiguration('route_loop')
    route_fill_gaps = LaunchConfiguration('route_fill_gaps')
    route_tool = LaunchConfiguration('route_tool')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='시뮬레이션이면 true'),
        DeclareLaunchArgument('route_mode', default_value='astar',
                              choices=['astar', 'sequence'],
                              description='astar=최단 / sequence=지정 경로'),
        # 쉼표 구분 문자열. 빈 배열은 ROS 파라미터 타입 추론이 안 돼 노드가 죽으므로
        # 배열이 아니라 문자열로 받는다.
        DeclareLaunchArgument('route_nodes', default_value='',
                              description='sequence 모드에서 따라갈 노드 ID (쉼표 구분)'),
        DeclareLaunchArgument('route_loop', default_value='false',
                              description='마지막 노드에서 첫 노드로 닫아 순환'),
        DeclareLaunchArgument('route_fill_gaps', default_value='true',
                              description='이웃 웨이포인트가 직접 링크로 안 이어지면 A* 로 채움'),
        # 화면이 있을 때만 기본으로 켠다 — 로봇처럼 DISPLAY 가 없는 곳에서 GUI 를
        # 띄우면 Qt 가 알아보기 힘든 오류로 죽는다. 필요하면 명시적으로 켤 수 있다.
        DeclareLaunchArgument('route_tool',
                              default_value='true' if os.environ.get('DISPLAY') else 'false',
                              choices=['true', 'false'],
                              description='경로 지정 PyQt 툴 동시 실행 (기본: DISPLAY 있으면 on)'),

        Node(
            package='scv_global_planner',
            executable='path_planner_node',
            name='global_path_planner_node',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'route_mode': route_mode,
                'route_nodes': route_nodes,
                'route_loop': route_loop,
                'route_fill_gaps': route_fill_gaps,
            }]
        ),

        # 경로 지정 툴. 노드를 눌러 RouteRequest 를 발행한다.
        # 창을 닫아도 런치 전체는 살아 있다 (플래너는 계속 돈다).
        Node(
            package='scv_global_planner',
            executable='route_tool',
            name='route_tool',
            output='screen',
            condition=IfCondition(route_tool),
            parameters=[{'use_sim_time': use_sim_time}],
        ),
    ])
