#!/usr/bin/env python3
# command_center 알고리즘 전용 런치 (localization/map 은 별도 기동).
#
#   포함:  scv_global_planner → local_costmap → simple_behavior_planner → controller(smppi)
#   전제(별도 런치로 먼저 띄울 것):
#     - map_provider     : /map_provider_node/graph, /map_provider_node/utm(datum), PCD 경로 토픽
#     - scv_localization : TF map→odom, odom→base_link, /odom  (REP-105)
#       예)  ros2 launch map_provider map_provider.launch.py bundle_dir:=~/scv_maps/d2_library_909_node
#            ros2 launch scv_localization scv_localization.launch.py use_sim_time:=<...>
#
#   datum 단일소스 = /map_provider_node/utm. planner 는 그래프/​datum 토픽 + TF map→base_link 로 동작.
#
#   예)  ros2 launch command_center_launch scv_system.launch.py use_sim_time:=false controller:=sa_mppi
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def _inc(pkg, rel, args=None):
    src = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory(pkg), 'launch', rel))
    return IncludeLaunchDescription(src, launch_arguments=(args or {}).items())


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_visualization = LaunchConfiguration('enable_visualization')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('enable_visualization', default_value='true'),

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
                  'goal_tolerance': '1.0'})]),

        # controller (smppi)
        TimerAction(period=3.0, actions=[
            _inc('smppi', 'smppi_controller.launch.py',
                 {'use_sim_time': use_sim_time,
                  'enable_visualization': enable_visualization})]),
    ])
