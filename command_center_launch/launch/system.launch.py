import os
import time
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare



def generate_launch_description():
    # Declare use_sim_time parameter
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    map_file = PathJoinSubstitution([
                    FindPackageShare('gmserver'),
                    'maps',
                    'pal_right.json'
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
        
        # Launch 5: Localization (after 12 seconds)
        TimerAction(
            period=6.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        os.path.join(
                            FindPackageShare('tiny_localization').find('tiny_localization'),
                            'launch',
                            'tiny_localization.launch.py'
                        )
                    ]),
                    launch_arguments={'use_sim_time': use_sim_time}.items()
                )
            ]
        ),
        
        # Launch 6: Behavior planner (after 15 seconds)
        TimerAction(
            period=8.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        os.path.join(
                            FindPackageShare('simple_behavior_planner').find('simple_behavior_planner'),
                            'launch',
                            'simple_behavior_planner.launch.py'
                        )
                    ]),
                    launch_arguments={'use_sim_time': use_sim_time}.items()
                )
            ]
        )
    ])
