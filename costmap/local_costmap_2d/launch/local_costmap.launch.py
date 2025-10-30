import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directory
    pkg_dir = FindPackageShare(package='local_costmap_2d').find('local_costmap_2d')

    # Parameters file path
    default_params_file = PathJoinSubstitution([
        FindPackageShare('local_costmap_2d'),
        'config',
        'costmap_params.yaml'
    ])

    # Launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to costmap parameters file'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level (debug, info, warn, error)'
    )

    # Node configuration
    local_costmap_node = Node(
        package='local_costmap_2d',
        executable='local_costmap_node',
        name='local_costmap',
        namespace='',
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        params_file_arg,
        use_sim_time_arg,
        log_level_arg,
        local_costmap_node
    ])
