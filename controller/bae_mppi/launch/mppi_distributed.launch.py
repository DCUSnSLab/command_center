#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch distributed MPPI nodes"""

    use_sim_time = LaunchConfiguration('use_sim_time', default='true') # 시뮬레이션 환경인 경우 true, 밖이면 false
    # Get package directory
    bae_mppi_dir = get_package_share_directory('bae_mppi')
    config_file = os.path.join(bae_mppi_dir, 'config', 'mppi_params.yaml')
    topic_config_file = os.path.join(bae_mppi_dir, 'config', 'topic_config.yaml')
    
    return LaunchDescription([ 
        # Sensor Processing Node
        Node(
            package='bae_mppi',
            executable='sensor_processor_node.py',
            name='sensor_processor',
            namespace='bae_mppi',
            parameters=[config_file, topic_config_file, {'use_sim_time': use_sim_time}],
            output='screen'
        ),
        
        # Core MPPI Controller Node
        Node(
            package='bae_mppi',
            executable='mppi_core_node.py',
            name='mppi_core',
            namespace='bae_mppi',
            parameters=[config_file, topic_config_file, {'use_sim_time': use_sim_time}],
            output='screen'
        ),
        
        # Visualization Node
        Node(
            package='bae_mppi',
            executable='visualization_node.py',
            name='mppi_visualization',
            namespace='bae_mppi',
            parameters=[config_file, topic_config_file, {'use_sim_time': use_sim_time}],
            output='screen'
        ),
        
        # Steering Validation Node (enabled for Ackermann model)
        Node(
            package='bae_mppi',
            executable='steering_validation_node.py',
            name='steering_validation',
            namespace='bae_mppi',
            parameters=[config_file, topic_config_file, {'use_sim_time': use_sim_time}],
            output='screen'
        ),
    ])