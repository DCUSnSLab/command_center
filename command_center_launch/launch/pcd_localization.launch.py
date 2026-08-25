#!/usr/bin/env python3
"""PCD 지도 정합 측위 파이프라인 (GPS 열화 대비 — 2026-08-25 통합).

~/SCV에서 분기해 온 스냅샷 패키지(scv_localization + map_provider + lio_sam
프런트엔드)로 /pcd/global_pose 를 발행한다. 융합은 기존 map_anchor 가 담당
(map_anchor_pcd:=1 로 소비, DCU datum -> 구 d2 map 프레임 오프셋은
map_anchor 에 이미 반영돼 있음: +45.518, +26.015).

사용 (field_drive 에서 pcd_loc:=true 로 포함):
  bundle_dir 기본값 = 클린맵 번들 (v4 23-bag, 8-bag + 실차 A/B 검증본)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    pkg_loc = FindPackageShare("scv_localization")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "bundle_dir", default_value="/home/scv/MAP/DCU_0819_clean",
            description="map_provider 번들 (기본: 동적객체 제거 클린맵)"),

        # 번들 배송 (PCD 경로 토픽 + datum)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("map_provider"), "launch",
                 "map_provider.launch.py"])),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "bundle_dir": LaunchConfiguration("bundle_dir"),
            }.items()),

        # LIO 프런트엔드 + 고정지도 정합 localizer -> /pcd/global_pose
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [pkg_loc, "launch", "temp",
                 "fixed_map_localization.launch.py"])),
            launch_arguments={
                "use_sim_time": use_sim_time,
                # 번들 토픽 수신 규약 — 실제 PCD 경로는 map_provider 가 배송
                "map_dir": "/tmp/nomap",
                "localizer_config": PathJoinSubstitution(
                    [pkg_loc, "config",
                     "fixed_map_localization_bundle.yaml"]),
                "require_initial_pose": "true",
                "global_pose_topic": "/pcd/global_pose",
                "map_to_odom_candidate_topic": "/pcd/map_to_odom",
                "processing_enable_topic": "/localization/pcd_enabled",
                "processing_enabled_at_start": "true",
                # TF 소유권은 기존 map_anchor 유지 — 후보만 발행
                "publish_map_to_odom": "false",
            }.items()),

        # GPS coarse prior (BBS 전역 초기화용) — datum 은 번들 토픽에서 수신
        Node(
            package="scv_localization", executable="gps_measurement_node",
            name="gps_measurement", output="screen",
            parameters=[PathJoinSubstitution(
                [pkg_loc, "config", "fgo_fusion.yaml"]),
                {"use_sim_time": use_sim_time}],
        ),
    ])
