"""Field driving bring-up: the FULL autonomy stack for real outdoor runs.

Phased startup (hardware first, control last) of everything the sidewalk
mission needs, including the curb-safety layers:

  t= 0s  sensors        bring_up/sensors_start (vectornav, velodyne VLP32C,
                        realsense, sllidar) + hunter_start (base, teleop mux,
                        description/TF)  [+ gnss_start when with_gnss]
  t= 3s  map service    gmserver (map_file_path argument)
  t= 5s  localization   robot_localization dual-EKF + FAST-LIO + navsat
                        (odom->base_link & map->odom TF, /odom;
                        replaces tiny_localization — see scv_dual_ekf.yaml)
  t= 7s  route          sequential_global_planner (whole ordered route, no
                        start/goal pinned) or scv_global_planner A* when
                        route_source:=graph
  t= 9s  curb safety    pcd_ground_filter/curb_costmap
                        (L1 below-grade curb detection -> local_costmap on
                        /velodyne_points_curb)
  t=12s  controller     smppi_controller (corridor keepout -> /costmap_keepout,
                        costmap processor, MPPI with hard-lethal +
                        lateral-bias critics)
  t=15s  behavior       simple_behavior_planner (waypoints, BLOCKED_WAIT
                        stop-and-wait escalation)

Node-death policy: the perception chain is inline (curb node feeds the
costmap feeds the controller), so safety-critical nodes are started with
respawn in their own launch files; local_costmap keeps publishing its LAST
grid when input stops — check the curb node log if the costmap looks frozen.

Field checklist before launch:
  * curb_params.yaml plane_c bounds must bracket the CURRENT sensor mount
    height (measured 0.73-0.94 m across 2026 recordings).
  * smppi_params.yaml corridor_half_width (1.4 m) vs the actual sidewalk;
    gate_node_types must match the map's crossing node types (maps authored
    so far use node_type=1 only -- no gates will open).
  * hunter_teleop_mux keeps manual override priority over /cmd_vel.

Route endpoints are deliberately NOT named. sequential_global_planner
publishes the whole ordered route with start_node_id/goal_node_id empty, and
simple_behavior_planner then joins at the node nearest the robot rather than
driving back to the route head. Pass route_source:=graph (or
explicit_endpoints:=true on the planner) for the old named-endpoint behaviour.

can0 must be up before launching; use field_bringup.sh, which raises it and
then calls this file.

Usage:
  ros2 launch command_center_launch field_drive.launch.py
  ros2 launch command_center_launch field_drive.launch.py \
      map_file_path:=/path/to/map.json with_gnss:=false enable_visualization:=true
  ros2 launch command_center_launch field_drive.launch.py \
      route_source:=graph corridor_half_width:=7.0 max_slew_mps:=0
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.substitutions import FindPackageShare


def _include(pkg, launch_file, launch_arguments=None, condition=None):
    kwargs = {}
    if launch_arguments is not None:
        kwargs['launch_arguments'] = launch_arguments.items()
    if condition is not None:
        kwargs['condition'] = condition
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(FindPackageShare(pkg).find(pkg), 'launch', launch_file)
        ]),
        **kwargs,
    )


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    map_file_path = LaunchConfiguration('map_file_path')
    with_sensors = LaunchConfiguration('with_sensors', default='true')
    with_gnss = LaunchConfiguration('with_gnss', default='true')
    enable_visualization = LaunchConfiguration('enable_visualization', default='false')
    route_source = LaunchConfiguration('route_source', default='sequential')
    with_fastlio = LaunchConfiguration('with_fastlio', default='true')
    max_slew_mps = LaunchConfiguration('max_slew_mps', default='0.5')
    cov_ref_m2 = LaunchConfiguration('cov_ref_m2', default='1.0')
    corridor_half_width = LaunchConfiguration('corridor_half_width', default='1.4')

    # Map must contain UtmInfo (F2: UtmInfo-less maps yield zeroed
    # waypoints) and its node[0] must equal the navsat datum in
    # scv_dual_ekf.yaml. record_20260630_141709_map_d1 satisfies both.
    default_map = PathJoinSubstitution([
        FindPackageShare('gmserver'), 'maps', 'record_20260630_141709_map_d1.json'
    ])

    sim_args = {'use_sim_time': use_sim_time}

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation clock'),
        DeclareLaunchArgument(
            'map_file_path', default_value=default_map,
            description='Graph map JSON for gmserver / global planner'),
        DeclareLaunchArgument(
            'with_sensors', default_value='true',
            description='Start sensor + vehicle-base drivers '
                        '(false when they already run elsewhere)'),
        DeclareLaunchArgument(
            'with_gnss', default_value='true',
            description='Start ublox + NTRIP RTK stack'),
        DeclareLaunchArgument(
            'enable_visualization', default_value='false',
            description='SMPPI RViz marker node (leave off for field runs)'),
        DeclareLaunchArgument(
            'route_source', default_value='sequential',
            description="'sequential' publishes the whole ordered route with no "
                        "start/goal pinned (drive on from wherever you are); "
                        "'graph' uses gmserver + A* between named endpoints."),
        DeclareLaunchArgument(
            'goal_node', default_value='',
            description='Scenario goal: route to this node at startup '
                        '(nearest-node start unless start_node is set). '
                        'Empty = follow the whole route to its end.'),
        DeclareLaunchArgument(
            'start_node', default_value='',
            description='Scenario start: with goal_node, route FROM this node '
                        'and pin it (vehicle is expected to be placed there). '
                        'Empty = start from the node nearest the robot.'),
        DeclareLaunchArgument(
            'with_fastlio', default_value='true',
            description='LiDAR-inertial odometry source for the odom EKF'),
        DeclareLaunchArgument(
            'lio_source', default_value='fastlio',
            description='LIO 구현 선택: fastlio(FAST-LIO2, 기본/현장검증) | '
                        'fasterlio(Faster-LIO, iVox) | rko(RKO-LIO). '
                        '셋 다 /odometry/fast_lio 로 발행하므로 하류 무영향.'),
        DeclareLaunchArgument(
            'max_slew_mps', default_value='0.5',
            description='Absolute rate ceiling on map-anchor motion. 0 disables '
                        '(the 2026-07-30 failure mode: the anchor slid at 5 m/s, '
                        'four times the vehicle top speed).'),
        DeclareLaunchArgument(
            'cov_ref_m2', default_value='1.0',
            description='Reference GPS covariance for inverse-variance anchor gain'),
        DeclareLaunchArgument(
            'corridor_half_width', default_value='1.4',
            description='SMPPI keepout corridor half-width (m). Widen only when '
                        'the route is known to run off the mapped sidewalk.'),
        # 선언 없이 LaunchConfiguration(default=) 만으로도 동작하지만 그러면
        # `ros2 launch --show-args` 에 뜨지 않아 롤백 스위치가 있는지 현장에서
        # 알 수 없다. 게이트를 끄는 인자는 반드시 목록에 보여야 한다.
        DeclareLaunchArgument(
            'anchor_yaw_autocal', default_value='true',
            description='map_anchor 온라인 yaw 보정 (2026-08-04 반대주행 대책). '
                        'false 로 두면 anchor_yaw_offset 시드만 쓴다 — 회귀 '
                        'A/B 와 챔버 결함주입 전용. 필드에서는 켠 채로 둘 것.'),

        # --- t=0: hardware -------------------------------------------------
        _include('bring_up', 'sensors_start.launch.py',
                 condition=IfCondition(with_sensors)),
        _include('bring_up', 'hunter_start.launch.py',
                 condition=IfCondition(with_sensors)),
        _include('bring_up', 'gnss_start.launch.py',
                 condition=IfCondition(with_gnss)),

        # --- t=3: map service (serves /load_map, used by replan requests) ---
        TimerAction(period=3.0, actions=[
            _include('gmserver', 'map_service.launch.py', sim_args),
        ]),

        # --- t=5: localization (FAST-LIO + dual-EKF + navsat) --------------
        # Owns odom->base_link AND map->odom TF; datum = graph-map node[0].
        # The anchor rate/covariance limits are the 2026-07-30 field fix and
        # must be passed through, not left to the include's own defaults.
        TimerAction(period=5.0, actions=[
            _include('robot_localization', 'scv_dual_ekf.launch.py', {
                'use_sim_time': use_sim_time,
                'with_fastlio': with_fastlio,
                'map_anchor_pcd': '0',
                'max_slew_mps': max_slew_mps,
                'cov_ref_m2': cov_ref_m2,
                'lio_source': LaunchConfiguration('lio_source'),
                # yaw 자동 보정(2026-08-04 반대주행 대책) 롤백 스위치 —
                # field_bringup.sh anchor_yaw_autocal:=false 한 줄로 끈다
                'anchor_yaw_autocal': LaunchConfiguration(
                    'anchor_yaw_autocal', default='true'),
            }),
        ]),

        # --- t=7: route source ---------------------------------------------
        # Default: publish the entire ordered route with start/goal left blank,
        # so the behavior planner joins at the node nearest the robot. The A*
        # planner is kept for runs that really do name two endpoints.
        TimerAction(period=7.0, actions=[
            _include('sequential_global_planner', 'sequential_planner.launch.py', {
                'use_sim_time': use_sim_time,
                'map_file': map_file_path,
                'explicit_endpoints': 'false',
                'goal_node': LaunchConfiguration('goal_node', default=''),
                'start_node': LaunchConfiguration('start_node', default=''),
            }, condition=IfCondition(
                PythonExpression(["'", route_source, "' == 'sequential'"]))),
            _include('scv_global_planner', 'path_planner.launch.py', {
                'use_sim_time': use_sim_time,
                'map_file_path': map_file_path,
            }, condition=IfCondition(
                PythonExpression(["'", route_source, "' == 'graph'"]))),
        ]),

        # --- t=9: curb-safety perception (L1) ------------------------------
        # curb_detection_node (/velodyne_points -> /velodyne_points_curb with
        # below-grade walls) + local_costmap consuming the augmented cloud.
        TimerAction(period=9.0, actions=[
            _include('pcd_ground_filter', 'curb_costmap.launch.py', sim_args),
        ]),

        # --- t=12: SMPPI controller stack (L2 + L4) -------------------------
        # corridor_keepout (/costmap -> /costmap_keepout), costmap processor,
        # MPPI main (hard-lethal ObstacleCritic + LateralBiasCritic).
        TimerAction(period=12.0, actions=[
            _include('smppi', 'smppi_controller.launch.py', {
                'use_sim_time': use_sim_time,
                'enable_visualization': enable_visualization,
                'corridor_half_width': corridor_half_width,
            }),
        ]),

        # --- t=15: behavior planner (L3) ------------------------------------
        # /odometry/global, not the package default /odom: the graph nodes are
        # map-frame, and /odom drifts away from map by however much the anchor
        # has corrected. Matching against /odom picks the wrong nearest node.
        TimerAction(period=15.0, actions=[
            _include('simple_behavior_planner',
                     'simple_behavior_planner.launch.py', {
                         'use_sim_time': use_sim_time,
                         'current_position_topic': '/odometry/global',
                     }),
        ]),
    ])
