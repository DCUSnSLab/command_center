#!/usr/bin/env python3
# command_center 알고리즘 전용 런치 (localization/map 은 별도 기동).
#
#   포함:  scv_global_planner → local_costmap → simple_behavior_planner
#          → waypoint_manage → controller(smppi)
#   전제(별도 런치로 먼저 띄울 것):
#     - map_provider     : /map_provider_node/graph, /map_provider_node/utm(datum), PCD 경로 토픽
#     - scv_localization : TF map→odom, odom→base_link, /odom  (REP-105)
#       예)  ros2 launch map_provider map_provider.launch.py bundle_dir:=~/scv_maps/d2_library_909_node
#            ros2 launch scv_localization scv_localization.launch.py use_sim_time:=<...>
#
#   datum 단일소스 = /map_provider_node/utm. planner 는 그래프/​datum 토픽 + TF map→base_link 로 동작.
#
#   waypoint 계층 (use_waypoint_manage, 기본 true):
#     true  : BP(waypoint_mode=external) → /target_waypoints → waypoint_manage → /multiple_waypoints
#             (localization 보정을 가드 거쳐 초 단위 전파; waypoint_manage/README 참조)
#     false : 레거시 — BP 가 직접 /multiple_waypoints 발행 (노드당 1회 스냅샷)
#     ※ 이중 발행 방지를 위해 waypoint_mode 는 이 플래그와 연동됨. 개별 오버라이드 금지.
#
#   예)  ros2 launch command_center_launch scv_system.launch.py use_sim_time:=false controller:=sa_mppi
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory


def _inc(pkg, rel, args=None, condition=None):
    src = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory(pkg), 'launch', rel))
    return IncludeLaunchDescription(
        src, launch_arguments=(args or {}).items(), condition=condition)


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_visualization = LaunchConfiguration('enable_visualization')
    use_waypoint_manage = LaunchConfiguration('use_waypoint_manage')
    # waypoint_manage 사용 시 BP 는 의미 목표(/target_waypoints)만 발행 (이중 발행 방지)
    waypoint_mode = PythonExpression(
        ["'external' if '", use_waypoint_manage, "' == 'true' else 'multiple'"])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('enable_visualization', default_value='true'),
        DeclareLaunchArgument('use_waypoint_manage', default_value='false',
                              choices=['true', 'false']),

        # global planner (map_provider /graph + datum 소비, TF map→base_link 로 현재위치)
        _inc('scv_global_planner', 'path_planner.launch.py',
             {'use_sim_time': use_sim_time}),

        # local costmap (odom 프레임; controller가 /costmap 소비하므로 먼저)
        TimerAction(period=1.0, actions=[
            _inc('local_costmap', 'costmap.launch.py',
                 {'use_sim_time': use_sim_time})]),

        # behavior planner (planner 이후)
        TimerAction(period=2.0, actions=[
            _inc('simple_behavior_planner', 'simple_behavior_planner.launch.py',
                 {'use_sim_time': use_sim_time,
                  'current_position_topic': '/odom',
                  'planned_path_topic': '/planned_path_detailed',
                  'goal_tolerance': '1.0',
                  'waypoint_mode': waypoint_mode})]),

        # waypoint 계층: 의미 목표 -> (graph/datum/TF/가드) -> odom waypoint
        TimerAction(period=2.0, actions=[
            _inc('waypoint_manage', 'waypoint_manage.launch.py',
                 {'use_sim_time': use_sim_time},
                 condition=IfCondition(use_waypoint_manage))]),

        # controller (smppi)
        TimerAction(period=3.0, actions=[
            _inc('smppi', 'smppi_controller.launch.py',
                 {'use_sim_time': use_sim_time,
                  'enable_visualization': enable_visualization})]),
    ])
