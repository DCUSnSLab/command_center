#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("waypoint_manage")
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "config",
            default_value=PathJoinSubstitution([pkg, "config", "waypoint_manage.yaml"])),
        Node(
            package="waypoint_manage", executable="waypoint_manage_node",
            name="waypoint_manage", output="screen",
            parameters=[LaunchConfiguration("config"),
                        {"use_sim_time": LaunchConfiguration("use_sim_time")}],
        ),
    ])
