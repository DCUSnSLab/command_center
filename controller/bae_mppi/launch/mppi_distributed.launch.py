#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch distributed MPPI nodes"""
    
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
            parameters=[config_file, topic_config_file],
            output='screen'
        ),
        
        # Core MPPI Controller Node
        Node(
            package='bae_mppi',
            executable='mppi_core_node.py',
            name='mppi_core',
            namespace='bae_mppi',
            parameters=[config_file, topic_config_file],
            output='screen'
        ),
        
        # Visualization Node
        Node(
            package='bae_mppi',
            executable='visualization_node.py',
            name='mppi_visualization',
            namespace='bae_mppi',
            parameters=[config_file, topic_config_file],
            output='screen'
        ),
        
        # Steering Validation Node (enabled for Ackermann model)
        Node(
            package='bae_mppi',
            executable='steering_validation_node.py',
            name='steering_validation',
            namespace='bae_mppi',
            parameters=[config_file, topic_config_file],
            output='screen'
        ),
    ])