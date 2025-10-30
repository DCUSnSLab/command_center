# file: costmap.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('simple_costmap_2d')

    # Path to config file
    default_config_file = os.path.join(pkg_share, 'config', 'costmap_params.yaml')

    # ----- Launch args -----
    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file = LaunchConfiguration('config_file')

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'
        ),

        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='Path to costmap configuration YAML file'
        ),

        # Costmap node
        Node(
            package='simple_costmap_2d',
            executable='costmap_node',
            name='costmap_node',
            output='screen',
            parameters=[
                config_file,
                {'use_sim_time': use_sim_time}
            ],
        ),
    ])
