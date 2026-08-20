#!/usr/bin/env python3
"""waypoint_manage 노드 — waypoint 계층 (기계 계층).

역할 (BP/BT 로부터 waypoint 배관을 분리):
  입력  TargetWaypoints (의미 목표: 노드 ID + 행동 파라미터)  <- BP / (미래) BT
        /map_provider_node/graph, /utm (latched)             <- 좌표 해석
        /fgo/status                                          <- localization 건강 신호
        TF map->odom                                         <- 프레임 변환 (스냅샷)
  출력  MultipleWaypoints (odom 프레임)                       -> MPPI (기존 규약 그대로)

핵심 동작:
  - 새 타깃(노드 전환/재계획): 즉시 변환·발행 (기존 BP 래치와 동일한 반응성)
  - 같은 타깃 유지 중: update_rate 마다 최신 map->odom 으로 재변환하되
    UpdateGuard(상태 게이트/데드밴드/새너티/슬루)를 통과한 보정만 반영
    -> localization 보정이 노드 단위가 아니라 초 단위로, 그러나 안전하게 전파됨
  - heartbeat: 마지막 waypoint 를 주기 재발행 (스탬프 갱신)
"""

import math
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from command_center_interfaces.msg import MultipleWaypoints, PlannedPath, TargetWaypoints
from map_interfaces.msg import GraphLayer, UtmLayer

import tf2_ros
import tf2_geometry_msgs.tf2_geometry_msgs as tf2_geometry_msgs

from .graph_store import GraphStore
from .update_guard import GuardDecision, GuardParams, UpdateGuard
from .waypoint_generator import MapWaypoint, WaypointGenerator


class WaypointManageNode(Node):

    def __init__(self):
        super().__init__('waypoint_manage')

        # ---- parameters ----
        self.target_topic = self.declare_parameter(
            'topics.target_waypoints', '/target_waypoints').value
        self.output_topic = self.declare_parameter(
            'topics.multiple_waypoints', '/multiple_waypoints').value
        self.graph_topic = self.declare_parameter(
            'topics.graph', '/map_provider_node/graph').value
        self.utm_topic = self.declare_parameter(
            'topics.utm', '/map_provider_node/utm').value
        self.status_topic = self.declare_parameter(
            'topics.localization_status', '/fgo/status').value
        self.planned_path_topic = self.declare_parameter(
            'topics.planned_path', '/planned_path_detailed').value
        self.map_frame = self.declare_parameter('frames.map', 'map').value
        self.odom_frame = self.declare_parameter('frames.odom', 'odom').value
        self.update_rate = self.declare_parameter('update_rate_hz', 5.0).value
        self.heartbeat_rate = self.declare_parameter('heartbeat_rate_hz', 2.0).value

        guard = GuardParams()
        guard.deadband_m = self.declare_parameter('guard.deadband_m', 0.3).value
        guard.sanity_m = self.declare_parameter('guard.sanity_m', 3.0).value
        guard.slew_rate_mps = self.declare_parameter('guard.slew_rate_mps', 0.5).value
        guard.freeze_when_not_ok = self.declare_parameter(
            'guard.freeze_when_not_ok', True).value
        self.guard = UpdateGuard(guard)

        self.graph_store = GraphStore()
        self.generator = WaypointGenerator()

        # ---- state ----
        self.target: Optional[TargetWaypoints] = None
        self.localization_ok = True          # status 토픽 미존재 시(구 스택) 항상 OK 로 동작
        self.last_published: Optional[MultipleWaypoints] = None
        self.last_published_target_id: str = ''
        self.last_update_t: Optional[float] = None

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- I/O ----
        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        reliable = QoSProfile(depth=10)
        reliable.reliability = ReliabilityPolicy.RELIABLE

        self.create_subscription(
            TargetWaypoints, self.target_topic, self._target_callback, reliable)
        # 경로 노드 캐시: planner 의 임시 노드(GPS_START 등)는 그래프에 없어
        # planned_path 에서 좌표를 얻는다 (2026-08-04 시뮬 테스트에서 skip 확인)
        self.create_subscription(
            PlannedPath, self.planned_path_topic, self._planned_path_callback, reliable)
        self.create_subscription(
            GraphLayer, self.graph_topic, self._graph_callback, latched)
        self.create_subscription(
            UtmLayer, self.utm_topic, self._utm_callback, latched)
        self.create_subscription(
            String, self.status_topic, self._status_callback, latched)

        self.waypoints_pub = self.create_publisher(
            MultipleWaypoints, self.output_topic, reliable)

        self.create_timer(1.0 / self.update_rate, self._update_loop)

        self.get_logger().info(
            f'waypoint_manage: {self.target_topic} -> {self.output_topic} '
            f'(update {self.update_rate}Hz, guard deadband={guard.deadband_m} '
            f'sanity={guard.sanity_m} slew={guard.slew_rate_mps}m/s)')

    # ================= callbacks =================

    def _graph_callback(self, msg: GraphLayer):
        n = self.graph_store.set_graph(msg)
        self.get_logger().info(f'graph received: {n} nodes')

    def _utm_callback(self, msg: UtmLayer):
        self.graph_store.set_datum(msg)

    def _status_callback(self, msg: String):
        ok = msg.data == 'OK'
        if ok != self.localization_ok:
            self.get_logger().warn(f'localization status: {msg.data}')
        self.localization_ok = ok

    def _planned_path_callback(self, msg: PlannedPath):
        n = self.graph_store.set_path_nodes(msg.path_data.nodes)
        self.get_logger().info(f'planned path nodes cached: {n} (path_id={msg.path_id})')

    def _target_callback(self, msg: TargetWaypoints):
        if self.target is None or msg.current_node_id != self.target.current_node_id:
            self.get_logger().info(
                f'target received: {msg.current_node_id} (next={list(msg.next_node_ids)})')
        self.target = msg
        # 새 타깃은 update 루프를 기다리지 않고 즉시 처리 (노드 전환 반응성 유지)
        self._update_loop()

    # ================= core =================

    def _update_loop(self):
        if self.target is None or not self.graph_store.ready():
            return

        now = self.get_clock().now()
        dt = 1.0 / self.update_rate
        if self.last_update_t is not None:
            dt = max((now.nanoseconds - self.last_update_t) * 1e-9, 0.0)
        self.last_update_t = now.nanoseconds

        # 1) 노드 ID -> map 좌표 -> (재계산 모듈) -> map waypoints
        ids = [self.target.current_node_id] + list(self.target.next_node_ids)
        map_wps = self.graph_store.resolve_many(ids)
        if not map_wps or map_wps[0].node_id != self.target.current_node_id:
            self.get_logger().warn(
                f'target node {self.target.current_node_id} not in graph — skip',
                throttle_duration_sec=5.0)
            return
        map_wps = self.generator.generate(self.target.recalc_mode, map_wps)

        # 2) 최신 map->odom 스냅샷으로 후보 좌표 생성
        candidate = self._transform_to_odom(map_wps, now)
        if candidate is None:
            return  # TF 미가용 -> 잘못된 좌표를 내보내지 않음 (기존 규약 유지)

        # 3) 가드 판정 (current goal 기준)
        same_target = self.target.current_node_id == self.last_published_target_id
        prev_pos = None
        if self.last_published is not None and same_target:
            p = self.last_published.current_goal.pose.position
            prev_pos = (p.x, p.y)
        cand_pos = (candidate.current_goal.pose.position.x,
                    candidate.current_goal.pose.position.y)

        result = self.guard.decide(
            prev_pos, cand_pos, same_target, self.localization_ok, dt)

        if result.decision == GuardDecision.PUBLISH_NEW:
            out = candidate
        elif result.decision == GuardDecision.SLEW:
            ratio = self.guard.correction_ratio(result, prev_pos, cand_pos)
            out = self._blend(self.last_published, candidate, ratio)
        else:
            # HOLD / FROZEN / REJECT: 기존 좌표 유지 (heartbeat 재발행)
            if result.decision == GuardDecision.REJECT:
                self.get_logger().warn(
                    f'waypoint update rejected: {result.note}',
                    throttle_duration_sec=2.0)
            out = self.last_published
            if out is None:
                return
            out.header.stamp = now.to_msg()
            self._heartbeat_publish(out, now)
            return

        out.header.stamp = now.to_msg()
        self.waypoints_pub.publish(out)
        if self.last_published_target_id != self.target.current_node_id:
            self.get_logger().info(
                f'publishing current={out.current_goal.header.frame_id} '
                f'(decision={result.decision.value}, target={self.target.current_node_id})')
        self.last_published = out
        self.last_published_target_id = self.target.current_node_id

    def _heartbeat_publish(self, msg: MultipleWaypoints, now):
        # heartbeat_rate 로 다운샘플 (update 루프는 5Hz, 재발행은 2Hz 기본)
        period = 1.0 / max(self.heartbeat_rate, 0.1)
        t = now.nanoseconds * 1e-9
        if not hasattr(self, '_last_heartbeat_t') or t - self._last_heartbeat_t >= period:
            self.waypoints_pub.publish(msg)
            self._last_heartbeat_t = t

    # ================= helpers =================

    def _transform_to_odom(self, map_wps: List[MapWaypoint], now) -> Optional[MultipleWaypoints]:
        """map waypoints -> MultipleWaypoints(odom). TF 미가용 시 None."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame, self.map_frame, rclpy.time.Time())
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f'map->odom TF 미가용, waypoint 스킵: {e}', throttle_duration_sec=5.0)
            return None

        def to_odom_pose(w: MapWaypoint) -> PoseStamped:
            ps = PoseStamped()
            ps.header.stamp = now.to_msg()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = w.x
            ps.pose.position.y = w.y
            ps.pose.position.z = w.z
            ps.pose.orientation.z = math.sin(w.yaw / 2.0)
            ps.pose.orientation.w = math.cos(w.yaw / 2.0)
            odom_ps = tf2_geometry_msgs.do_transform_pose_stamped(ps, transform)
            # 기존 규약: frame_id 필드에 노드 ID 를 실음 (컨트롤러/시각화 호환)
            odom_ps.header.frame_id = w.node_id
            return odom_ps

        msg = MultipleWaypoints()
        msg.header.frame_id = self.odom_frame
        msg.current_goal = to_odom_pose(map_wps[0])
        msg.current_goal_node_type = map_wps[0].node_type
        msg.current_goal_reverse_heading = self._is_reverse(map_wps[0].node_type)
        for w in map_wps[1:]:
            msg.next_waypoints.append(to_odom_pose(w))
            msg.next_waypoints_node_types.append(w.node_type)
            msg.next_waypoints_reverse_heading.append(self._is_reverse(w.node_type))
        msg.path_id = self.target.path_id
        msg.current_waypoint_index = self.target.current_waypoint_index
        msg.total_waypoints = self.target.total_waypoints
        msg.is_final_waypoint = self.target.is_final_waypoint
        return msg

    def _blend(self, prev: MultipleWaypoints, candidate: MultipleWaypoints,
               ratio: float) -> MultipleWaypoints:
        """SLEW: prev -> candidate 로 ratio 만큼 이동한 좌표 (전 waypoint 동일 비율).
        보정은 map->odom 변환 차이라 모든 waypoint 가 같은 벡터를 공유한다."""
        import copy
        out = copy.deepcopy(candidate)

        def lerp(a: PoseStamped, b: PoseStamped) -> PoseStamped:
            r = copy.deepcopy(b)
            r.pose.position.x = a.pose.position.x + (b.pose.position.x - a.pose.position.x) * ratio
            r.pose.position.y = a.pose.position.y + (b.pose.position.y - a.pose.position.y) * ratio
            return r

        out.current_goal = lerp(prev.current_goal, candidate.current_goal)
        n = min(len(prev.next_waypoints), len(candidate.next_waypoints))
        for i in range(n):
            out.next_waypoints[i] = lerp(prev.next_waypoints[i], candidate.next_waypoints[i])
        return out

    @staticmethod
    def _is_reverse(node_type: int) -> bool:
        # 기존 waypoint_publisher 규약: 후진 행동 타입 (2, 4)
        return node_type in (2, 4)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointManageNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
