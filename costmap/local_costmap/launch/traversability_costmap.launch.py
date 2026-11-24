"""/costmap from the traversability grid (replaces costmap.launch.py's accumulating node).

traversability_grid (SCV_Perception) must run separately:
    ros2 launch traversability_grid traversability_grid.launch.py use_sim_time:=<...>
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('local_costmap'), 'config', 'traversability_costmap.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation clock if true'),
        DeclareLaunchArgument('config_file', default_value=default_config,
                              description='Path to the parameter YAML file'),
        Node(
            package='local_costmap',
            executable='traversability_costmap_node',
            name='traversability_costmap_node',
            output='screen',
            parameters=[LaunchConfiguration('config_file'),
                        {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
    ])
