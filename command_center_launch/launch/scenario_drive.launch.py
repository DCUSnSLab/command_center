#!/usr/bin/env python3
"""시나리오 주행 런치 — field_drive 전체 스택 + GPS 게이트 웨이포인트 러너.

사용:
  ros2 launch command_center_launch scenario_drive.launch.py scenario:=1
  ros2 launch command_center_launch scenario_drive.launch.py scenario:=2 controller:=mpc
  ros2 launch command_center_launch scenario_drive.launch.py waypoints:=N0426,NX0004

시나리오 (2026-08-25 지정):
  1: N0426 → NX0004 → N0447 → N0415 → N0402
  2: N0414 → NX0006 → NX0002 → NX0016 → NX0022 → NX0031
  3: NX0029 → NX0017 → NX0010 → NX0001

동작: sequential planner 를 대기 모드(auto_start=false)로 띄워 게이트 통과
전에는 경로가 없다(수동으로 auto 를 올려도 출발하지 않음). scenario_runner 가
GPS 안정화 게이트(평균 175 s 대기, σ<=0.15 안정화 / 횡보 감지 시 출발)를
통과하면 첫 웨이포인트를 /goal_node_id 로 발행하고, 도달할 때마다 다음
웨이포인트로 넘어간다. 진행 상황은 /scenario/status 로 방송.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

SCENARIOS = {
    '1': 'N0426,NX0004,N0447,N0415,N0402',
    '2': 'N0414,NX0006,NX0002,NX0016,NX0022,NX0031',
    '3': 'NX0029,NX0017,NX0010,NX0001',
}
DEFAULT_MAP = '/home/scv/MAP/DCU_0819/graph.json'


def setup(context):
    scenario = LaunchConfiguration('scenario').perform(context)
    waypoints = LaunchConfiguration('waypoints').perform(context)
    if not waypoints:
        if scenario not in SCENARIOS:
            raise RuntimeError(f'scenario 는 {list(SCENARIOS)} 중 하나거나 waypoints 를 직접 지정')
        waypoints = SCENARIOS[scenario]
    map_file = LaunchConfiguration('map_file_path').perform(context)

    field_drive = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            FindPackageShare('command_center_launch').perform(context),
            'launch', 'field_drive.launch.py')),
        launch_arguments={
            'map_file_path': map_file,
            'controller': LaunchConfiguration('controller'),
            'scan_yaw_init': LaunchConfiguration('scan_yaw_init'),
            'curb_method': LaunchConfiguration('curb_method'),
            'probe': LaunchConfiguration('probe'),
            'pcd_loc': LaunchConfiguration('pcd_loc'),
            'bundle_dir': LaunchConfiguration('bundle_dir'),
            'corridor_half_width': LaunchConfiguration('corridor_half_width'),
            # 게이트 핵심: 전체 체인 자동 발행 금지 — goal 수신 대기 상태로 기동
            'sequential_auto_start': 'false',
            'goal_node': '',
        }.items())

    runner = Node(
        package='sequential_global_planner',
        executable='scenario_runner.py',
        name='scenario_runner',
        output='screen',
        parameters=[{
            'waypoints': waypoints,
            'map_file': map_file,
            'datum_utm_easting': 482232.7,
            'datum_utm_northing': 3974384.47,
            'sigma_ok': float(LaunchConfiguration('sigma_ok').perform(context)),
            'min_wait': float(LaunchConfiguration('min_wait').perform(context)),
            'plateau_window': 30.0,
            'plateau_eps': 0.03,
            'max_wait': float(LaunchConfiguration('max_wait').perform(context)),
            'arrive_radius': float(LaunchConfiguration('arrive_radius').perform(context)),
        }])
    return [field_drive, runner]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='1',
                              description='1|2|3 (waypoints 직접 지정 시 무시)'),
        DeclareLaunchArgument('waypoints', default_value='',
                              description='쉼표 구분 노드열 (지정 시 scenario 무시)'),
        DeclareLaunchArgument('map_file_path', default_value=DEFAULT_MAP),
        DeclareLaunchArgument('controller', default_value='mppi'),
        DeclareLaunchArgument('scan_yaw_init', default_value='1'),
        DeclareLaunchArgument('curb_method', default_value='below_grade',
                              description='below_grade|ring'),
        DeclareLaunchArgument('probe', default_value='false'),
        DeclareLaunchArgument('pcd_loc', default_value='false'),
        DeclareLaunchArgument('bundle_dir',
                              default_value='/home/scv/MAP/DCU_0819_clean'),
        # 8/25 교훈: 미전달 시 smppi 기본 1.4 m — 반드시 운용값 전달
        DeclareLaunchArgument('corridor_half_width', default_value='7.0'),
        # GPS 게이트 (분석 근거: 31 bag, σ<=0.15 달성 16본 평균 175 s)
        DeclareLaunchArgument('sigma_ok', default_value='0.15'),
        DeclareLaunchArgument('min_wait', default_value='175.0'),
        DeclareLaunchArgument('max_wait', default_value='420.0'),
        DeclareLaunchArgument('arrive_radius', default_value='0.6'),
        OpaqueFunction(function=setup),
    ])
