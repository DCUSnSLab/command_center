#!/usr/bin/env python3
"""
Simple Behavior Planner Node (Refactored)
깔끔하고 모듈화된 새로운 구조
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from typing import Optional

# ROS2 messages
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Bool, String, Header, Int32

from command_center_interfaces.msg import (
    PlannedPath, ControllerGoalStatus, MultipleWaypoints,
    MPPIParams, PauseCommand, RequestReplan
)

# Local modules
from simple_behavior_planner.path_manager import PathManager
from simple_behavior_planner.waypoint_publisher import WaypointPublisher
from simple_behavior_planner.behavior_controller import BehaviorController
from simple_behavior_planner.safety_monitor import SafetyMonitor
from simple_behavior_planner.blocked_wait_monitor import BlockedWaitMonitor


class SimpleBehaviorPlannerNode(Node):
    """새로운 모듈화된 Simple Behavior Planner Node"""

    def __init__(self):
        super().__init__('simple_behavior_planner')

        # Declare parameters
        self._declare_parameters()
        self._load_parameters()

        # Initialize core modules
        self.path_manager = PathManager()
        self.waypoint_publisher = WaypointPublisher(self, self.waypoint_mode)
        self.behavior_controller = BehaviorController(
            self, self.behavior_config_path) if self.enable_behavior_control else None
        self.safety_monitor = SafetyMonitor(self)
        self.blocked_monitor = BlockedWaitMonitor(
            blocked_detect_sec=self.get_parameter('blocked.detect_sec').value,
            progress_eps=self.get_parameter('blocked.progress_eps').value,
            wait_timeout=self.get_parameter('blocked.wait_timeout').value,
            creep_timeout=self.get_parameter('blocked.creep_timeout').value,
            creep_speed=self.get_parameter('blocked.creep_speed').value,
            assist_repeat_sec=self.get_parameter('blocked.assist_repeat_sec').value)
        self.blocked_near_goal_hold_off = float(
            self.get_parameter('blocked.near_goal_hold_off').value)
        self.latest_goal_distance = None
        self.pause_until = 0.0

        # State variables
        self.current_pose: Optional[PoseStamped] = None
        self.subgoal_published = False
        self.emergency_stop_requested = False
        self.pause_signal_sent = False

        # Dynamic replanning state
        self.current_route_type = "A"  # Default to route A
        self.path_availability = True  # Assume path is available initially
        self.last_goal_node_id = ""   # Track last goal for replanning

        # One-time trigger flags
        self.path_query_sent = False  # Flag to prevent multiple queries for same node
        self.current_trigger_node_id = ""  # Track which node triggered the query
        self.replan_request_sent = False  # Flag to prevent multiple replan requests

        # Setup QoS profiles
        self._setup_qos_profiles()

        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()

        # Link modules with publishers
        self._link_module_publishers()

        # Main planning timer
        self.create_timer(0.05, self.planning_callback)  # 20Hz

        self.get_logger().info('Simple Behavior Planner Node (Refactored) initialized')
        self._log_configuration()

    def _declare_parameters(self):
        """파라미터 선언"""
        # Topic parameters
        self.declare_parameter('current_position_topic', '/odom')
        self.declare_parameter('planned_path_topic', '/planned_path_detailed')
        self.declare_parameter('goal_status_topic', '/goal_status')
        self.declare_parameter('subgoal_topic', '/subgoal')
        self.declare_parameter('multiple_waypoints_topic', '/multiple_waypoints')
        self.declare_parameter('emergency_stop_topic', '/emergency_stop')
        self.declare_parameter('stop_flag_topic', '/stop_flag')
        self.declare_parameter('traffic_light_topic', '/tl/state_id')

        # Dynamic replanning parameters
        self.declare_parameter('path_availability_topic', '/path_availability')
        self.declare_parameter('request_replan_topic', '/request_replan')
        self.declare_parameter('node_type_triggers', [12, 13])  # node types that trigger path check

        # Behavior parameters
        self.declare_parameter('pause_trigger_distance', 0.8)

        # Blocked-wait (회피 불가 시 정지·대기) parameters
        self.declare_parameter('blocked.detect_sec', 4.0)
        self.declare_parameter('blocked.progress_eps', 0.15)
        self.declare_parameter('blocked.wait_timeout', 12.0)
        self.declare_parameter('blocked.creep_timeout', 10.0)
        self.declare_parameter('blocked.creep_speed', 0.3)
        self.declare_parameter('blocked.assist_repeat_sec', 15.0)
        self.declare_parameter('blocked.near_goal_hold_off', 0.8)
        self.declare_parameter('waypoint_mode', 'multiple')
        self.declare_parameter('behavior_config_path', 'behavior_modifiers.yaml')
        self.declare_parameter('enable_behavior_control', True)

    def _load_parameters(self):
        """파라미터 로드"""
        self.current_position_topic = self.get_parameter('current_position_topic').value
        self.planned_path_topic = self.get_parameter('planned_path_topic').value
        self.goal_status_topic = self.get_parameter('goal_status_topic').value
        self.subgoal_topic = self.get_parameter('subgoal_topic').value
        self.multiple_waypoints_topic = self.get_parameter('multiple_waypoints_topic').value
        self.emergency_stop_topic = self.get_parameter('emergency_stop_topic').value
        self.stop_flag_topic = self.get_parameter('stop_flag_topic').value
        self.traffic_light_topic = self.get_parameter('traffic_light_topic').value

        # Dynamic replanning parameters
        self.path_availability_topic = self.get_parameter('path_availability_topic').value
        self.request_replan_topic = self.get_parameter('request_replan_topic').value
        self.node_type_triggers = self.get_parameter('node_type_triggers').value

        self.pause_trigger_distance = self.get_parameter('pause_trigger_distance').value
        self.waypoint_mode = self.get_parameter('waypoint_mode').value
        self.behavior_config_path = self.get_parameter('behavior_config_path').value
        self.enable_behavior_control = self.get_parameter('enable_behavior_control').value

        # Validate waypoint mode
        if self.waypoint_mode not in ['single', 'multiple']:
            self.get_logger().warn(f"Invalid waypoint_mode '{self.waypoint_mode}', defaulting to 'multiple'")
            self.waypoint_mode = 'multiple'

    def _setup_qos_profiles(self):
        """QoS 프로파일 설정"""
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

    def _setup_subscribers(self):
        """구독자 설정"""
        self.current_pose_sub = self.create_subscription(
            Odometry, self.current_position_topic,
            self.current_pose_callback, self.best_effort_qos)

        self.planned_path_sub = self.create_subscription(
            PlannedPath, self.planned_path_topic,
            self.planned_path_callback, self.reliable_qos)

        self.goal_status_sub = self.create_subscription(
            ControllerGoalStatus, self.goal_status_topic,
            self.goal_status_callback, self.reliable_qos)

        self.stop_flag_sub = self.create_subscription(
            Bool, self.stop_flag_topic,
            self.stop_flag_callback, self.reliable_qos)

        self.traffic_light_sub = self.create_subscription(
            Int32, self.traffic_light_topic,
            self.traffic_light_callback, self.reliable_qos)

        # Dynamic replanning subscribers
        self.path_availability_sub = self.create_subscription(
            Bool, self.path_availability_topic,
            self.path_availability_callback, self.reliable_qos)

    def _setup_publishers(self):
        """발행자 설정"""
        self.emergency_stop_pub = self.create_publisher(
            Bool, self.emergency_stop_topic, self.reliable_qos)

        self.pause_command_pub = self.create_publisher(
            PauseCommand, '/pause_command', self.reliable_qos)

        # Waypoint publishers based on mode
        if self.waypoint_mode in ['single', 'multiple']:
            if self.waypoint_mode == 'single' or self.waypoint_mode == 'multiple':
                self.subgoal_pub = self.create_publisher(
                    PoseStamped, self.subgoal_topic, self.reliable_qos)

            if self.waypoint_mode == 'multiple':
                self.multiple_waypoints_pub = self.create_publisher(
                    MultipleWaypoints, self.multiple_waypoints_topic, self.reliable_qos)

        # Behavior control publisher
        if self.enable_behavior_control:
            self.mppi_param_pub = self.create_publisher(
                MPPIParams, '/mppi_update_params', self.reliable_qos)

        # Dynamic replanning publisher
        self.request_replan_pub = self.create_publisher(
            RequestReplan, self.request_replan_topic, self.reliable_qos)

        # Blocked-wait status / assist publishers
        self.behavior_status_pub = self.create_publisher(
            String, '/behavior_status', self.reliable_qos)
        self.assist_request_pub = self.create_publisher(
            String, '/blocked_assist_request', self.reliable_qos)

    def _link_module_publishers(self):
        """모듈에 발행자 연결"""
        # Waypoint publisher
        single_pub = getattr(self, 'subgoal_pub', None)
        multiple_pub = getattr(self, 'multiple_waypoints_pub', None)
        self.waypoint_publisher.set_publishers(single_pub, multiple_pub)

        # Behavior controller
        if self.behavior_controller:
            mppi_pub = getattr(self, 'mppi_param_pub', None)
            self.behavior_controller.set_mppi_publisher(mppi_pub)

        # Safety monitor
        self.safety_monitor.set_pause_publisher(self.pause_command_pub)

    def _log_configuration(self):
        """설정 로깅"""
        self.get_logger().info(f'Waypoint mode: {self.waypoint_mode}')
        self.get_logger().info(f'Behavior control: {"enabled" if self.enable_behavior_control else "disabled"}')
        self.get_logger().info(f'Planned path topic: {self.planned_path_topic}')
        self.get_logger().info(f'Goal status topic: {self.goal_status_topic}')
        self.get_logger().info(f'Path availability topic: {self.path_availability_topic}')
        self.get_logger().info(f'Request replan topic: {self.request_replan_topic}')
        self.get_logger().info(f'Node type triggers: {self.node_type_triggers}')
        self.get_logger().info(f'Initial route type: {self.current_route_type}')

        # 수동/자율 모드 전이 로그 (2026-08-03 필드 분석에서 추가).
        # 사후 분석 때 "이 구간이 수동인가 자율인가"를 데이터로 답할 수 없어
        # GPS 급이동·우회 구간 해석이 모호했다. mux 의 /vehicle/mux_status 를
        # 구독해 전이만 INFO 로 남긴다. 메시지 패키지가 없는 환경(시뮬 등)
        # 에서는 조용히 생략 — 하드 의존을 만들지 않는다.
        try:
            from teleop_rover_msgs.msg import MuxStatus
            self._last_mux = None

            def _mux_cb(m):
                key = (m.mode, m.cmd_source, m.nav_active)
                if key != self._last_mux:
                    self.get_logger().info(
                        f'[MODE] mux mode={m.mode} source={m.cmd_source} '
                        f'nav_active={m.nav_active} teleop={m.teleop_active}')
                    self._last_mux = key
            self.create_subscription(MuxStatus, '/vehicle/mux_status',
                                     _mux_cb, 10)
        except ImportError:
            self.get_logger().info('teleop_rover_msgs 없음 — mux 모드 로그 생략')

        # 베이스 제어 모드 인지 (2026-08-03 S2 분석에서 추가).
        # 하드웨어 RC 는 mux 를 거치지 않고 CAN 레벨에서 베이스를 잡는다 —
        # 그 동안 자율 파이프라인의 명령은 무시되는데 스택은 그걸 모르고
        # "내 명령으로 차가 안 움직인다 = 차단"으로 해석해 수동 주행 중
        # ASSIST 를 3회나 헛발동했다(bag 124 실측). /hunter_status 의
        # control_mode(1=CAN 명령 청취=자율, 3=RC, 0=대기)를 구독해
        # 경로 합류 규칙 (2026-08-05 필드에서 최근접 규칙의 결함이 드러났다 —
        # path_manager.align_to_position docstring 참조).
        self.declare_parameter('join_nearest', False)
        self.declare_parameter('join_max_approach_m', 40.0)
        # 수동 주행 중 합류 노드 갱신 + 자율 전환 시 재선택 (2026-08-06 필드)
        self.declare_parameter('realign_on_engage', True)
        self.declare_parameter('realign_manual_move_m', 2.0)
        self._join_nearest = self.get_parameter('join_nearest').value
        self._join_max_approach = self.get_parameter('join_max_approach_m').value
        self._realign_on_engage = self.get_parameter('realign_on_engage').value
        self._realign_move_m = self.get_parameter('realign_manual_move_m').value
        self._align_pose = None      # 마지막 합류 계산 시점의 위치

        # 합류 후보까지의 접근 직선 통행성 검사 (2026-08-11 챔버).
        # 원본 /costmap 을 쓴다 — /costmap_keepout 은 경로에서 만든 코리도를
        # 얹은 것이라, 그걸로 합류를 고르면 순환 논리가 된다.
        self.declare_parameter('join_check_approach', True)
        self.declare_parameter('join_costmap_topic', '/costmap')
        self.declare_parameter('join_clear_half_width_m', 0.45)
        self.declare_parameter('join_clear_lethal', 90)
        self._join_check_approach = self.get_parameter('join_check_approach').value
        self._join_clear_half_w = self.get_parameter('join_clear_half_width_m').value
        self._join_clear_lethal = self.get_parameter('join_clear_lethal').value
        self._costmap = None
        if self._join_check_approach:
            self.create_subscription(
                OccupancyGrid, self.get_parameter('join_costmap_topic').value,
                lambda m: setattr(self, '_costmap', m), 1)

        # 베이스가 우리 명령을 듣지 않는 동안 차단 에스컬레이션을 보류한다.
        # hunter_msgs 가 없는 환경(챔버 시뮬)은 조용히 생략 — 항상 청취로
        # 간주해 기존 거동 보존.
        self._hunter_mode = None
        try:
            from hunter_msgs.msg import HunterStatus

            def _hs_cb(m):
                if m.control_mode != self._hunter_mode:
                    names = {0: '대기', 1: 'CAN(자율)', 3: 'RC(수동)'}
                    self.get_logger().info(
                        f'[MODE] hunter control_mode -> {m.control_mode} '
                        f'({names.get(m.control_mode, "?")})'
                        + ('' if m.control_mode == 1
                           else ' — 차단 에스컬레이션 보류'))
                    prev_mode = self._hunter_mode
                    self._hunter_mode = m.control_mode
                    # 수동->자율 전환 순간 합류 노드를 다시 고른다.
                    #
                    # 합류 노드는 경로 수신 시점에 한 번 정해지는데, 운용
                    # 절차가 자율 전환 전 RC 전진(yaw 게이트)을 요구하므로
                    # 그 사이 차량이 20 m 씩 움직인다 — 2026-08-06 필드:
                    # 기동 위치 (47.2,44.2)에서 고른 N0440 이 RC 이동 후
                    # (32.3,59.7)에서는 뒤쪽 22 m 에 있었고, 자율 전환 즉시
                    # 목표(북서)가 아니라 그 노드(남동)로 18.8 m 갔다.
                    # 절차와 알고리즘이 서로를 무력화하던 구멍이다.
                    if (prev_mode is not None and m.control_mode == 1
                            and self._realign_on_engage
                            and getattr(self, '_unpinned_route', False)):
                        self.get_logger().info(
                            '[MODE] 자율 전환 — 현재 위치로 합류 노드 재선택')
                        self._pending_align = True
            self.create_subscription(HunterStatus, '/hunter_status',
                                     _hs_cb, 10)
        except ImportError:
            self.get_logger().info('hunter_msgs 없음 — control_mode 게이트 생략')

    def _base_listening(self) -> bool:
        """베이스가 자율(CAN) 명령을 듣고 있는가. 미상(None)=참으로 간주."""
        return self._hunter_mode is None or self._hunter_mode == 1

    # ===== Callback Methods =====

    def current_pose_callback(self, msg: Odometry):
        """현재 위치 콜백"""
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose

        # 순간이동 감지 → 재정렬. 최초 정렬은 위치추정이 수렴하기 전에 일어날
        # 수 있다: map_anchor 부트스트랩은 (0,0)=datum=B000 이므로, 경로가
        # 포즈보다 먼저 도착하면 정렬이 항상 B000 을 고르고 첫 GPS 스냅과
        # 동시에 '도달 처리'까지 된다(bag 재생으로 실측: 정렬 0.2 s 뒤 3.9 m
        # 스냅). 차량 최고속은 1.3 m/s 라 연속 포즈 간 점프가 임계를 넘으면
        # 주행이 아니라 앵커 스냅/FAST-LIO 재초기화다 — 그때만 다시 정렬한다.
        prev = self.current_pose
        self.current_pose = pose_stamped
        if prev is not None and getattr(self, '_unpinned_route', False):
            dt = ((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                  - (prev.header.stamp.sec + prev.header.stamp.nanosec * 1e-9))
            dx = pose_stamped.pose.position.x - prev.pose.position.x
            dy = pose_stamped.pose.position.y - prev.pose.position.y
            jump = (dx * dx + dy * dy) ** 0.5
            if jump > max(2.0, 5.0 * max(dt, 0.0)):
                self.get_logger().warn(
                    f'pose teleport {jump:.1f} m (dt={dt:.2f}s) — '
                    're-aligning start node to converged position')
                self._pending_align = True

        # RC 수동 주행 중에는 합류 노드를 계속 따라오게 한다. 수동 주행은
        # 사실상 '차량 재배치'이므로, 자율 전환 시점의 위치가 반영돼야 한다.
        # 순간이동 감지(위 블록)는 느린 주행을 잡지 못한다 — 20 m 를 20 초에
        # 걸쳐 가면 프레임당 1 m 라 임계에 걸리지 않는다.
        if (self._hunter_mode == 3 and self._realign_on_engage
                and getattr(self, '_unpinned_route', False)):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            last = self._align_pose
            if last is None or math.hypot(px - last[0], py - last[1]) > self._realign_move_m:
                self._pending_align = True

        if getattr(self, '_pending_align', False) and self._align_start_to_pose():
            self._pending_align = False
            self._align_pose = (pose_stamped.pose.position.x,
                                pose_stamped.pose.position.y)
            # 정렬로 목표가 바뀌었으니 웨이포인트를 다시 내보낸다. 이걸 리셋하지
            # 않으면 부트스트랩 때 낸 옛 목표(B000)가 제어기에 남아 있는다.
            self.subgoal_published = False

    def _align_start_to_pose(self) -> bool:
        """경로 합류 노드를 골라 시작 목표로 정렬.

        기본은 비용 기반(접근거리 + 잔여 경로거리 최소). join_nearest:=true 로
        예전 최근접 규칙으로 되돌릴 수 있다 — 회귀 비교용.
        """
        if self.current_pose is None or not self.path_manager.path_nodes:
            return False
        px = self.current_pose.pose.position.x
        py = self.current_pose.pose.position.y
        # cap=0 이면 어떤 후보도 상한을 통과하지 못해 최근접 폴백으로 떨어진다
        # — 그게 곧 예전 규칙이라 별도 분기를 두지 않는다.
        cap = 0.0 if self._join_nearest else self._join_max_approach
        clear = self._approach_clear if self._join_check_approach else None
        idx = self.path_manager.align_to_position(
            px, py, max_approach_m=cap, approach_clear=clear)
        node = self.path_manager.get_current_target_node()
        rule = 'nearest' if self._join_nearest else 'min(approach+remaining)'
        if clear is not None:
            rule += '+approach_clear'
        self.get_logger().info(
            f'start node unspecified -> join idx {idx} '
            f'({node["id"] if node else "?"}) by {rule}')
        return True

    def _approach_clear(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """(x0,y0)->(x1,y1) map 프레임 직선이 코스트맵에서 통행 가능한가.

        합류란 곧 "여기서 저 노드까지 경로 밖을 직진한다"는 뜻이다. 그 직선이
        막혀 있으면 그 노드는 합류점이 될 수 없다 — 2026-08-11 챔버에서 이
        검사가 없어 차량이 매핑된 경로를 통째로 건너뛰고 지름길을 시도하다
        장애물에 갇혔다(3/3 미도달). 경로는 바로 그 구역을 우회하려고 그려진
        것이었다.

        코스트맵이나 TF 가 아직 없으면 **참을 반환한다**(fail-open). 여기서
        거짓을 내면 합류 자체가 막혀 차량이 기동 직후 멎는다 — 정보 부족을
        장애물로 취급하면 안 된다.
        """
        grid = self._costmap
        if grid is None:
            return True
        try:
            tr = self.waypoint_publisher.tf_buffer.lookup_transform(
                grid.header.frame_id, 'map', rclpy.time.Time())
        except Exception:
            return True
        tx = tr.transform.translation.x
        ty = tr.transform.translation.y
        q = tr.transform.rotation
        th = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(th), math.sin(th)

        def to_grid(mx, my):
            gx = c * mx - s * my + tx
            gy = s * mx + c * my + ty
            res = grid.info.resolution
            col = int((gx - grid.info.origin.position.x) / res)
            row = int((gy - grid.info.origin.position.y) / res)
            return col, row

        w, h = grid.info.width, grid.info.height
        d = math.hypot(x1 - x0, y1 - y0)
        if d <= 1e-6:
            return True
        # 로봇 반폭만큼 좌우로도 훑는다 — 중심선만 보면 폭이 있는 차량이
        # 스쳐 지나갈 수 없는 틈을 통행 가능으로 판정한다.
        nx, ny = -(y1 - y0) / d, (x1 - x0) / d
        steps = max(2, int(d / max(grid.info.resolution, 0.05)))
        for k in range(steps + 1):
            t = k / steps
            bx, by = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            for off in (-self._join_clear_half_w, 0.0, self._join_clear_half_w):
                col, row = to_grid(bx + nx * off, by + ny * off)
                if not (0 <= col < w and 0 <= row < h):
                    continue          # 코스트맵 밖은 미지 — 막힌 것으로 보지 않는다
                if grid.data[row * w + col] >= self._join_clear_lethal:
                    return False
        return True

    def planned_path_callback(self, msg: PlannedPath):
        """경로 계획 콜백"""
        self.path_manager.update_path(msg)
        self.subgoal_published = False

        # 시작 노드 미지정(start_node_id == '') 운용: 경로의 첫 노드가 아니라
        # 현재 위치의 최근접 노드부터 추종 시작 (2026-07-30 필드 운용 변경).
        # 명시된 start_node_id가 있으면 종전대로 첫 노드부터.
        self._unpinned_route = not msg.start_node_id
        if not msg.start_node_id:
            if not self._align_start_to_pose():
                self._pending_align = True

        # Update last goal node ID for replanning
        self._update_last_goal_node_id(msg)

        # Update route type based on received path (if it was a replan response)
        if self.replan_request_sent:
            # Determine route type from path_id or start_node_id
            new_route_type = self._determine_route_type_from_path(msg)
            if new_route_type and new_route_type != self.current_route_type:
                old_route = self.current_route_type
                self.current_route_type = new_route_type
                self.get_logger().info(f'Route type updated: {old_route} -> {self.current_route_type}')

        # Log path info
        path_info = self.path_manager.get_path_info()
        node_types = self.path_manager.get_node_types()

        self.get_logger().info(f'Received path with {path_info["total_nodes"]} nodes')
        self.get_logger().info(f'Path ID: {path_info["path_id"]}')
        self.get_logger().info(f'Current route type: {self.current_route_type}')
        self.get_logger().info(f'Node types: {list(set(node_types))}')

        # Update initial behavior
        if self.behavior_controller and self.path_manager.path_nodes:
            first_node = self.path_manager.get_current_target_node()
            if first_node:
                first_node_type = first_node.get('node_type', 1)
                self.behavior_controller.update_behavior(first_node_type)
                self.safety_monitor.update_behavior_type(first_node_type)

    def goal_status_callback(self, msg: ControllerGoalStatus):
        """목표 상태 콜백"""
        self.latest_goal_distance = msg.distance_to_goal
        current_target = self.path_manager.get_current_target_node()
        if not current_target or msg.goal_id != current_target['id']:
            return

        # Check for pause trigger (before goal reached)
        self._check_pause_trigger(msg)

        if msg.goal_reached and msg.status_code == 1:  # SUCCEEDED
            self._handle_goal_success(msg)
        elif msg.status_code == 2:  # FAILED
            self._handle_goal_failure(msg)
        elif msg.status_code == 3:  # ABORTED
            self._handle_goal_abort(msg)

    def stop_flag_callback(self, msg: Bool):
        """장애물 감지 플래그 콜백"""
        self.safety_monitor.update_stop_flag(msg.data)
        self.get_logger().info(f'Stop flag received: {msg.data}')

    def traffic_light_callback(self, msg: Int32):
        """신호등 상태 콜백"""
        self.safety_monitor.update_traffic_light_state(msg.data)
        self.get_logger().debug(f'Traffic light state received: {msg.data}')

    def path_availability_callback(self, msg: Bool):
        """경로 가용성 콜백 - 인지 모듈로부터 받는 응답"""
        # Only process if we're currently at a trigger node and haven't sent replan request yet
        if not self.path_query_sent or self.replan_request_sent:
            return

        self.path_availability = msg.data
        self.get_logger().info(f'Path availability received: {self.path_availability} for node {self.current_trigger_node_id}')

        # Only trigger replanning if path is NOT available (false)
        if not self.path_availability:
            self.get_logger().info(f'Current route {self.current_route_type} is NOT available, triggering replanning...')
            # Get current trigger node type
            current_target = self.path_manager.get_current_target_node()
            trigger_node_type = current_target.get('node_type') if current_target else None
            self._trigger_replanning(trigger_node_type)
        else:
            self.get_logger().info(f'Current route {self.current_route_type} is available, continuing with existing path')
            # Mark as processed but don't trigger replanning
            self.replan_request_sent = True  # Prevent further processing for this trigger node

    # ===== Planning Logic =====

    def planning_callback(self):
        """메인 계획 루프 (10Hz)"""
        if not self._is_ready_for_planning():
            return

        if self.emergency_stop_requested:
            self._publish_emergency_stop()
            return

        # Safety monitoring
        self.safety_monitor.check_safety_conditions()

        # Check for node type trigger (dynamic replanning)
        self._check_node_type_trigger()

        # Behavior control
        self._update_behavior_if_needed()

        # Blocked-wait monitoring (회피 불가/보행자 차단 시 정지·대기·서행 에스컬레이션)
        self._tick_blocked_monitor()

        # Waypoint publishing
        #
        # 앵커 이동 시 재발행: 웨이포인트의 odom 좌표는 발행 순간의 map→odom
        # 변환으로 고정되는데, 앵커(특히 RTK 부재의 EMA 추종)는 계속 움직인다.
        # 재발행 없이는 목표의 '물리' 위치가 앵커 이동량만큼 틀어진다 —
        # 2026-08-03 필드 실측: 기동 후 정차 대기 중 map 거리 14.6 m 목표가
        # 제어기에는 35.4 m 로 보였다(발행 후 앵커 ~20 m 이동).
        try:
            tr = self.waypoint_publisher.tf_buffer.lookup_transform(
                'odom', 'map', rclpy.time.Time())
            t = tr.transform.translation
            q = tr.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            cur = (t.x, t.y, yaw)
            last = getattr(self, '_wp_tf_at_publish', None)
            if self.subgoal_published and last is not None:
                dx = cur[0] - last[0]
                dy = cur[1] - last[1]
                # 회전도 감시: yaw 자동 보정(map_anchor autocal)이 주행 중
                # th 를 돌리면 병진은 그대로여도 waypoint 의 odom 투영이
                # 통째로 회전한다. 5도 초과면 재발행.
                dyaw = math.atan2(math.sin(yaw - last[2]),
                                  math.cos(yaw - last[2])) \
                    if len(last) > 2 else 0.0
                if (dx * dx + dy * dy) ** 0.5 > 0.5 or abs(dyaw) > 0.087:
                    self.get_logger().info(
                        f'map→odom anchor moved '
                        f'{(dx * dx + dy * dy) ** 0.5:.2f} m / '
                        f'{math.degrees(dyaw):+.1f} deg since last '
                        'waypoint publish — refreshing waypoints')
                    self.subgoal_published = False
            self._wp_tf_now = cur
        except Exception:
            pass
        if not self.subgoal_published:
            self._publish_waypoints()
            self._wp_tf_at_publish = getattr(self, '_wp_tf_now', None)

    def _is_ready_for_planning(self) -> bool:
        """계획 준비 상태 확인"""
        return (self.path_manager.is_path_following and
                self.current_pose is not None and
                self.path_manager.path_nodes)

    def _update_behavior_if_needed(self):
        """필요시 행동 업데이트"""
        if not self.behavior_controller:
            return

        current_target = self.path_manager.get_current_target_node()
        if current_target:
            current_node_type = current_target.get('node_type', 1)
            if self.behavior_controller.update_behavior(current_node_type):
                self.safety_monitor.update_behavior_type(current_node_type)

    def _publish_waypoints(self):
        """웨이포인트 발행"""
        current_target = self.path_manager.get_current_target_node()
        if not current_target:
            return

        next_nodes = self.path_manager.get_next_nodes()
        path_info = self.path_manager.get_path_info()

        self.waypoint_publisher.publish_waypoints(current_target, next_nodes, path_info)
        self.subgoal_published = True

        self.get_logger().info(f'Published waypoints: current={current_target["id"]}, '
                             f'next_count={len(next_nodes)}')

    def _tick_blocked_monitor(self):
        """Blocked-wait 상태기계 틱"""
        xy = None
        if self.current_pose is not None:
            xy = (self.current_pose.pose.position.x,
                  self.current_pose.pose.position.y)

        should_be_moving = (
            self.path_manager.is_path_following
            and not self.emergency_stop_requested
            and time.time() >= self.pause_until
            and not self.safety_monitor.get_safety_status()['should_pause']
            # 베이스가 RC/대기 모드면 우리 명령이 무시되는 중 — "안 움직임"은
            # 차단이 아니라 권한 부재다. 에스컬레이션(CREEP/ASSIST) 보류.
            and self._base_listening()
            and (self.latest_goal_distance is None
                 or self.latest_goal_distance > self.blocked_near_goal_hold_off))

        actions = self.blocked_monitor.update(time.time(), xy, should_be_moving)
        for action in actions:
            self._handle_blocked_action(action)

    def _handle_blocked_action(self, action: str):
        """Blocked-wait 상태기계 액션 처리"""
        if action.startswith('announce:'):
            state = action.split(':', 1)[1]
            msg = String()
            msg.data = state
            self.behavior_status_pub.publish(msg)
            if state == 'NORMAL':
                self.get_logger().info('[BLOCKED] cleared -> NORMAL')
            else:
                self.get_logger().warn(f'[BLOCKED] state -> {state}')
        elif action == 'creep_on':
            if self.behavior_controller:
                self.behavior_controller.apply_creep(
                    self.blocked_monitor.creep_speed)
        elif action == 'creep_off':
            if self.behavior_controller:
                self.behavior_controller.reapply_current_behavior()
        elif action == 'assist':
            msg = String()
            current = self.path_manager.get_current_target_node()
            msg.data = (f"blocked at node {current['id'] if current else '?'}: "
                        f"no in-corridor avoidance path; operator attention requested")
            self.assist_request_pub.publish(msg)
            self.get_logger().error(f'[BLOCKED] assist requested: {msg.data}')

    # ===== Goal Status Handlers =====

    def _check_pause_trigger(self, msg: ControllerGoalStatus):
        """Pause 트리거 확인"""
        current_target = self.path_manager.get_current_target_node()
        if not current_target or self.pause_signal_sent:
            return

        if msg.distance_to_goal <= self.pause_trigger_distance:
            node_type = current_target.get('node_type', 1)

            if node_type in [7, 8]:  # Pause nodes
                pause_duration = 2.0 if node_type == 7 else 4.0
                self._send_pause_command(pause_duration, current_target['id'], f"Node type {node_type} pause")
                self.pause_signal_sent = True
                self.pause_until = time.time() + pause_duration + 1.0

                self.get_logger().info(f"Pause command sent: {pause_duration}s for node {msg.goal_id}")

    def _handle_goal_success(self, msg: ControllerGoalStatus):
        """목표 성공 처리"""
        self.path_manager.mark_goal_completed(msg.goal_id)
        self.pause_signal_sent = False
        self.subgoal_published = False

        if self.path_manager.advance_to_next_node():
            next_target = self.path_manager.get_current_target_node()
            if next_target:
                self.get_logger().info(f'Advanced to next node: {next_target["id"]} '
                                     f'({self.path_manager.current_target_index + 1}/'
                                     f'{len(self.path_manager.path_nodes)})')
        else:
            self.get_logger().info('Path following completed!')

    def _handle_goal_failure(self, msg: ControllerGoalStatus):
        """목표 실패 처리"""
        self.get_logger().warn(f'Goal {msg.goal_id} failed! Distance: {msg.distance_to_goal:.3f}m')
        self.subgoal_published = False

    def _handle_goal_abort(self, msg: ControllerGoalStatus):
        """목표 중단 처리"""
        self.get_logger().warn(f'Goal {msg.goal_id} aborted! Distance: {msg.distance_to_goal:.3f}m')
        self.emergency_stop_requested = True

    # ===== Utility Methods =====

    def _send_pause_command(self, duration: float, node_id: str, reason: str):
        """Pause command 전송"""
        try:
            pause_msg = PauseCommand()
            pause_msg.header = Header()
            pause_msg.header.stamp = self.get_clock().now().to_msg()
            pause_msg.header.frame_id = 'behavior_planner'
            pause_msg.pause_duration = duration
            pause_msg.node_id = node_id
            pause_msg.reason = reason

            self.pause_command_pub.publish(pause_msg)
            self.get_logger().info(f"Pause command sent: {reason}")

        except Exception as e:
            self.get_logger().error(f"Failed to send pause command: {e}")

    def _publish_emergency_stop(self):
        """긴급 정지 발행"""
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)
        self.get_logger().warn('Emergency stop published!')

    # ===== Dynamic Replanning Logic =====

    def _check_node_type_trigger(self):
        """Node type 트리거 확인 - 경로 변경 지점에서 호출"""
        current_target = self.path_manager.get_current_target_node()
        if not current_target:
            return

        current_node_type = current_target.get('node_type')
        current_node_id = current_target['id']

        # Check if current node type matches any trigger (12 or 13)
        if current_node_type in self.node_type_triggers:
            # Check if we already processed this specific node
            if self.current_trigger_node_id == current_node_id and self.path_query_sent:
                # Already processed this node, skip
                return

            # New trigger node detected
            self.current_trigger_node_id = current_node_id
            self.path_query_sent = True
            self.replan_request_sent = False  # Reset replan flag for new trigger

            self.get_logger().info(f'Node type trigger detected: {current_node_type} at node {current_node_id}')
            self.get_logger().info(f'Querying perception system for route availability...')
            # Here we would query perception system, but for now we assume it responds via path_availability_callback
        else:
            # Not a trigger node, reset flags when moving to different node type
            if self.current_trigger_node_id != current_node_id:
                self.path_query_sent = False
                self.current_trigger_node_id = ""
                self.replan_request_sent = False

    def _trigger_replanning(self, trigger_node_type=None):
        """재계획 트리거 - 경로가 사용 불가능할 때 호출"""
        # Prevent multiple replan requests for same trigger
        if self.replan_request_sent:
            self.get_logger().debug('Replan request already sent for this trigger node')
            return

        if not self.path_manager.path_nodes:
            self.get_logger().warn('No path available for replanning')
            return

        # Determine route type based on trigger node type
        if trigger_node_type == 12:
            new_route_type = "T"
        elif trigger_node_type == 13:
            new_route_type = "P"
        else:
            # Fallback for unknown node types
            new_route_type = "B"
            self.get_logger().warn(f'Unknown trigger node type: {trigger_node_type}, using fallback route type B')

        # Get current position in path for start_node_id
        current_target = self.path_manager.get_current_target_node()
        start_node_id = current_target['id'] if current_target else ""

        # Use the last goal from current path
        goal_node_id = self.last_goal_node_id or self._get_final_node_id()

        self._send_replan_request(start_node_id, goal_node_id, new_route_type)

    def _send_replan_request(self, start_node_id: str, goal_node_id: str, route_type: str):
        """재계획 요청 메시지 전송"""
        try:
            replan_msg = RequestReplan()
            replan_msg.header = Header()
            replan_msg.header.stamp = self.get_clock().now().to_msg()
            replan_msg.header.frame_id = 'behavior_planner'

            replan_msg.start_node_id = start_node_id
            replan_msg.goal_node_id = goal_node_id
            replan_msg.route_type = route_type

            self.request_replan_pub.publish(replan_msg)

            # Mark replan request as sent
            self.replan_request_sent = True

            # NOTE: Don't update current_route_type here - wait until new path is received
            # self.current_route_type will be updated when planned_path_callback receives new path

            self.get_logger().info(f'Replan request sent: route_type={route_type}, '
                                 f'start: {start_node_id}, goal: {goal_node_id}')
            self.get_logger().info(f'Waiting for new planned path from Global Planner...')

        except Exception as e:
            self.get_logger().error(f'Failed to send replan request: {e}')

    def _get_final_node_id(self) -> str:
        """현재 경로의 마지막 노드 ID 반환"""
        if self.path_manager.path_nodes:
            return self.path_manager.path_nodes[-2]['id']
        return ""

    def _update_last_goal_node_id(self, path_msg: PlannedPath):
        """마지막 목표 노드 ID 업데이트"""
        self.last_goal_node_id = path_msg.goal_node_id

    def _determine_route_type_from_path(self, path_msg: PlannedPath) -> str:
        """수신된 경로로부터 route type 결정"""
        # Path ID나 start_node_id를 기반으로 route type 판단
        path_id = path_msg.path_id.lower()
        start_node_id = path_msg.start_node_id

        # Path ID에서 route type 추출 시도
        if 'route_a' in path_id or '_a_' in path_id or path_id.endswith('_a'):
            return "A"
        elif 'route_b' in path_id or '_b_' in path_id or path_id.endswith('_b'):
            return "B"

        # 현재와 다른 타입으로 추정 (toggle)
        return "B" if self.current_route_type == "A" else "A"


def main(args=None):
    rclpy.init(args=args)

    try:
        node = SimpleBehaviorPlannerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()