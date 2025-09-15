import os
import time
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare use_sim_time parameter
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    map_file = PathJoinSubstitution([
                    FindPackageShare('gmserver'),
                    'maps',
                    'mando2.json'
                ])

    return LaunchDescription([
        # Launch 1: Map service (immediate)
        DeclareLaunchArgument(
            'map_file_path',
            default_value=map_file,
            description='Path to the JSON map file'
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(
                    FindPackageShare('gmserver').find('gmserver'),
                    'launch',
                    'map_service.launch.py'
                )
            ]),
            launch_arguments={'use_sim_time': use_sim_time}.items()
        ),
        
        # Launch 2: Global planner (after 3 seconds)
        TimerAction(
            period=2.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        os.path.join(
                            FindPackageShare('scv_global_planner').find('scv_global_planner'),
                            'launch',
                            'path_planner.launch.py'
                        )
                    ]),
                    launch_arguments={'use_sim_time': use_sim_time}.items()
                )
            ]
        ),
        
        # # Launch 3: MPPI controller (after 6 seconds)
        # TimerAction(
        #     period=4.0,
        #     actions=[
        #         IncludeLaunchDescription(
        #             PythonLaunchDescriptionSource([
        #                 os.path.join(
        #                     FindPackageShare('bae_mppi').find('bae_mppi'),
        #                     'launch',
        #                     'mppi_distributed.launch.py'
        #                 )
        #             ]),
        #             launch_arguments={'use_sim_time': use_sim_time}.items()
        #         )
        #     ]
        # ),
        
        # Launch 4: Gazebo simulation (after 9 seconds)
        # TimerAction(
        #     period=9.0,
        #     actions=[
        #         IncludeLaunchDescription(
        #             PythonLaunchDescriptionSource([
        #                 os.path.join(
        #                     FindPackageShare('scv_robot_gazebo').find('scv_robot_gazebo'),
        #                     'launch',
        #                     'hunter_test.launch.py'
        #                 )
        #             ]),
        #             launch_arguments={'use_sim_time': use_sim_time}.items()
        #         )
        #     ]
        # ),
        
        # Launch 5: New Localization (after 12 seconds)
        TimerAction(
            period=4.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        os.path.join(
                            FindPackageShare('robot_localization').find('robot_localization'),
                            'launch',
                            'scv_dual_ekf_navsat_nomap.launch.py'
                        )
                    ]),
                    launch_arguments={'use_sim_time': use_sim_time}.items()
                )
            ]
        ),

        # pointcloud to laserscan
        TimerAction(
            period=6.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        os.path.join(
                            FindPackageShare('pointcloud_to_laserscan').find('pointcloud_to_laserscan'),
                            'launch',
                            'velodyne_to_scan.launch.py'
                        )
                    ]),
                    launch_arguments={'use_sim_time': use_sim_time}.items()
                )
            ]
        )
    ])