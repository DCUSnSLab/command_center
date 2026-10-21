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
#   behavior_planner (기본 simple):
#     simple : simple_behavior_planner (현행)
#     bt     : scv_bt_planner 가 명령 (simple 미기동)
#     shadow : simple 이 명령하고 scv_bt_planner 는 /bt/* 로 결정만 방송 (무위험 비교)
#     bt/shadow 는 scv_bt_planner 빌드 필요(BehaviorTree.CPP v4). bt_log_file:=auto 로 틱 단위 전이 기록.
#
#   record_bag (기본 false): 판단 근거용 bag 기록 — 결정 입력·출력 토픽 + 라이다 + 전방 컬러 카메라(compressed).
#     bag_dir:='' 이면 ~/.ros/bags/scv_<일시>, 경로를 주면 그 디렉토리. BT 판단과 센서를 같은 시간축에 두려면
#     behavior_planner:=bt|shadow bt_log_file:=auto 와 함께 켠다. 카메라 depth·d435 3대는 제외(이미지 릴레이만으로
#     57 MB/s 가 나와 디스크가 금방 찬다 — Pod 10/16 실측). 끝낼 때 ctrl-c 가 bag 을 닫는다.
#
#   예)  ros2 launch command_center_launch scv_system.launch.py use_sim_time:=false controller:=sa_mppi
#        ros2 launch command_center_launch scv_system.launch.py behavior_planner:=shadow
#        ros2 launch command_center_launch scv_system.launch.py behavior_planner:=shadow bt_log_file:=auto record_bag:=true
import os
from launch import LaunchDescription
import datetime
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (LaunchConfiguration, PythonExpression,
                                  EnvironmentVariable, PathJoinSubstitution)
from launch_ros.substitutions import FindPackageShare
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
    route_tool = LaunchConfiguration('route_tool')
    behavior_planner = LaunchConfiguration('behavior_planner')
    # waypoint_manage 사용 시 BP 는 의미 목표(/target_waypoints)만 발행 (이중 발행 방지)
    waypoint_mode = PythonExpression(
        ["'external' if '", use_waypoint_manage, "' == 'true' else 'multiple'"])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('enable_visualization', default_value='true'),
        DeclareLaunchArgument('use_waypoint_manage', default_value='false',
                              choices=['true', 'false']),
        DeclareLaunchArgument('behavior_planner', default_value='simple',
                              choices=['simple', 'bt', 'shadow'],
                              description="simple(현행) | bt(scv_bt_planner 명령) | shadow(simple 명령 + BT 결정 방송)"),
        DeclareLaunchArgument('bt_log_file', default_value='',
                              description="BT 틱 단위 전이 기록(.btlog): '' 끔 | 'auto' | 경로 (bt/shadow 에서만)"),
        DeclareLaunchArgument('record_bag', default_value='false', choices=['true', 'false'],
                              description='판단 근거용 bag 기록(결정 토픽+라이다+전방 카메라 compressed)'),
        DeclareLaunchArgument('bag_dir', default_value=os.path.expanduser(
            datetime.datetime.now().strftime('~/.ros/bags/scv_%Y%m%d_%H%M%S')),
            description='record_bag 출력 디렉토리 (기본 ~/.ros/bags/scv_<일시>)'),
        # 경로 지정 GUI. path_planner.launch.py 가 DISPLAY 유무로 기본값을 정하므로
        # 여기서는 빈 값을 넘겨 그 판단을 그대로 쓴다 ('true'/'false' 로 덮어쓸 수 있다).
        DeclareLaunchArgument('route_tool', default_value=EnvironmentVariable(
            'SCV_ROUTE_TOOL', default_value='true' if os.environ.get('DISPLAY') else 'false'),
            description='경로 지정 PyQt 툴 동시 실행'),

        # global planner (map_provider /graph + datum 소비, TF map→base_link 로 현재위치)
        _inc('scv_global_planner', 'path_planner.launch.py',
             {'use_sim_time': use_sim_time, 'route_tool': route_tool}),

        # local costmap (odom 프레임; controller가 /costmap 소비하므로 먼저)
        TimerAction(period=1.0, actions=[
            _inc('local_costmap', 'costmap.launch.py',
                 {'use_sim_time': use_sim_time})]),

        # behavior planner (planner 이후). simple 은 simple|shadow 에서, BT 는 bt|shadow 에서 기동.
        TimerAction(period=2.0, actions=[
            _inc('simple_behavior_planner', 'simple_behavior_planner.launch.py',
                 {'use_sim_time': use_sim_time,
                  'current_position_topic': '/odom',
                  'planned_path_topic': '/planned_path_detailed',
                  'goal_tolerance': '1.0',
                  'waypoint_mode': waypoint_mode},
                 condition=IfCondition(PythonExpression(
                     ["'", behavior_planner, "' in ('simple', 'shadow')"]))),
            # scv_bt_planner 는 지연 치환 include — 패키지가 없는 워크스페이스에서도
            # behavior_planner:=simple(기본) 이면 런치가 깨지지 않는다.
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PathJoinSubstitution([
                    FindPackageShare('scv_bt_planner'), 'launch', 'bt_planner.launch.py'])),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'mode': PythonExpression(
                        ["'active' if '", behavior_planner, "' == 'bt' else 'shadow'"]),
                    'current_position_topic': '/odom',
                    'planned_path_topic': '/planned_path_detailed',
                    'waypoint_mode': waypoint_mode,
                    'bt_log_file': LaunchConfiguration('bt_log_file'),
                }.items(),
                condition=IfCondition(PythonExpression(
                    ["'", behavior_planner, "' in ('bt', 'shadow')"]))),
        ]),

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

        # 판단 근거 bag — BT/simple 의 입력(odom·경로·goal_status·hunter_status·costmap)과 출력(waypoints·MPPIParams·
        # behavior_status·assist), 제어(cmd_vel·final_cmd), 센서(velodyne·GPS·IMU·2D scan·전방 컬러 카메라 compressed).
        # 정규식 토픽 선택: 없는 토픽은 그냥 안 잡힌다. 스택이 다 뜬 뒤 시작(4 s).
        TimerAction(period=4.0, actions=[
            ExecuteProcess(
                cmd=['ros2', 'bag', 'record', '-o', LaunchConfiguration('bag_dir'),
                     '-e', '^/(velodyne_points|velodyne_points_curb|costmap|odom|odometry/.*|ublox_gps_node/fix|'
                           'ublox_gps_node/fix_velocity|vectornav/imu|vectornav/pose|front/scan|rear/scan|'
                           'hunter_status|hunter/velocity|current_speed|current_steer_angle|cmd_vel|final_cmd|'
                           'vehicle/.*|remote/cmd_mode|tf|tf_static|planned_path.*|goal_status|goal_node_id|'
                           'sub_goal|multiple_waypoints|target_waypoints|mppi_update_params|mppi_optimal_path|'
                           'emergency_stop|stop_flag|pause_command|behavior_status|blocked_assist_request|'
                           'bt/.*|behavior/.*|map_provider_node/utm|map_anchor/.*|'
                           'camera/camera/color/image_raw/compressed|camera/camera/color/camera_info)$'],
                output='screen', name='scv_bag_record',
                condition=IfCondition(LaunchConfiguration('record_bag')))]),
    ])
