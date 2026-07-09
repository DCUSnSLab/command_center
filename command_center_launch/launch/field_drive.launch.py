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
  t= 7s  global plan    scv_global_planner
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

Usage:
  ros2 launch command_center_launch field_drive.launch.py
  ros2 launch command_center_launch field_drive.launch.py \
      map_file_path:=/path/to/map.json with_gnss:=false enable_visualization:=true
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
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

        # --- t=0: hardware -------------------------------------------------
        _include('bring_up', 'sensors_start.launch.py',
                 condition=IfCondition(with_sensors)),
        _include('bring_up', 'hunter_start.launch.py',
                 condition=IfCondition(with_sensors)),
        _include('bring_up', 'gnss_start.launch.py',
                 condition=IfCondition(with_gnss)),

        # --- t=3: map service (serves /load_map) ---------------------------
        TimerAction(period=3.0, actions=[
            _include('gmserver', 'map_service.launch.py', sim_args),
        ]),

        # --- t=5: localization (FAST-LIO + dual-EKF + navsat) --------------
        # Owns odom->base_link AND map->odom TF; datum = graph-map node[0].
        TimerAction(period=5.0, actions=[
            _include('robot_localization', 'scv_dual_ekf.launch.py', sim_args),
        ]),

        # --- t=7: global planner (calls /load_map with map_file_path) ------
        TimerAction(period=7.0, actions=[
            _include('scv_global_planner', 'path_planner.launch.py', {
                'use_sim_time': use_sim_time,
                'map_file_path': map_file_path,
            }),
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
            }),
        ]),

        # --- t=15: behavior planner (L3) ------------------------------------
        TimerAction(period=15.0, actions=[
            _include('simple_behavior_planner',
                     'simple_behavior_planner.launch.py', sim_args),
        ]),
    ])
