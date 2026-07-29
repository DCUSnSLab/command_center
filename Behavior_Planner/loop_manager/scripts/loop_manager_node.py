#!/usr/bin/env python3
"""
Loop Manager (임시 순찰 노드)

loop_goals 노드 ID 목록을 순환하며 /goal_pose 를 발행하고, /goal_status 의
goal_reached 로 leg 완료를 감지해 다음 goal 로 넘어간다. 목록 끝에 도달하면
처음으로 돌아가 무한 반복.

전제:
  - 그래프가 방향 사이클을 이룰 것 (scvmap_d2_0723 은 N0053->N0032 폐루프
    링크 추가됨). 없으면 복귀 leg 에서 A* 가 실패한다.
  - loop_goals 의 인접 원소는 서로 충분히 떨어져 있을 것 (권장: 루프 반대편
    중간 노드 + 종점, 예: [N0042, N0053]). 붙어 있는 두 노드를 번갈아 쏘면
    planner 의 시작 최근접 노드 선택이 모호해져 0m 경로가 나올 수 있다.

안전장치:
  - min_leg_time 이전의 goal_reached 는 오판정/즉시도달로 간주하고 무시
  - goal 발행 후 planned_path 가 관측될 때까지 goal_retry_period 마다 재발행
    (planner 의 datum/graph 미초기화로 goal 이 유실되는 기동 타이밍 대응)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from map_interfaces.msg import GraphLayer, UtmLayer
from command_center_interfaces.msg import ControllerGoalStatus


def latched_qos(depth=1):
    qos = QoSProfile(depth=depth)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class LoopManagerNode(Node):
    def __init__(self):
        super().__init__('loop_manager')

        self.declare_parameter('loop_goals', ['N0042', 'N0053'])
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('goal_status_topic', '/goal_status')
        self.declare_parameter('planned_path_topic', '/planned_path')
        self.declare_parameter('graph_topic', '/map_provider_node/graph')
        self.declare_parameter('utm_topic', '/map_provider_node/utm')
        self.declare_parameter('min_leg_time', 15.0)      # 이 시간 내 reached 무시 [s]
        self.declare_parameter('goal_retry_period', 5.0)  # 채택 전 재발행 주기 [s]
        self.declare_parameter('start_delay', 3.0)        # 준비 완료 후 첫 goal 지연 [s]
        self.declare_parameter('autostart', True)
        self.declare_parameter('max_laps', 0)             # 0 = 무한

        self.loop_goals = list(self.get_parameter('loop_goals').value)
        self.min_leg_time = float(self.get_parameter('min_leg_time').value)
        self.goal_retry_period = float(self.get_parameter('goal_retry_period').value)
        self.start_delay = float(self.get_parameter('start_delay').value)
        self.autostart = bool(self.get_parameter('autostart').value)
        self.max_laps = int(self.get_parameter('max_laps').value)

        if len(self.loop_goals) < 1:
            self.get_logger().error('loop_goals is empty')
            raise SystemExit(1)

        # 그래프/datum (latched)
        self.node_utm = {}          # id -> (easting, northing)
        self.datum = None           # (origin_easting, origin_northing)
        self.create_subscription(
            GraphLayer, self.get_parameter('graph_topic').value,
            self._graph_cb, latched_qos())
        self.create_subscription(
            UtmLayer, self.get_parameter('utm_topic').value,
            self._utm_cb, latched_qos())

        self.create_subscription(
            ControllerGoalStatus, self.get_parameter('goal_status_topic').value,
            self._status_cb, 10)
        self.create_subscription(
            Path, self.get_parameter('planned_path_topic').value,
            self._path_cb, 10)

        self.goal_pub = self.create_publisher(
            PoseStamped, self.get_parameter('goal_topic').value, 10)

        # 상태
        self.started = False
        self.ready_since = None     # 그래프+datum 준비된 시각
        self.goal_idx = 0
        self.leg_sent_time = None
        self.leg_adopted = False    # goal 발행 후 planned_path 관측 여부
        self.lap_count = 0

        self.timer = self.create_timer(1.0, self._tick)
        self.get_logger().info(
            f'Loop manager: goals={self.loop_goals}, min_leg_time={self.min_leg_time}s, '
            f'max_laps={"inf" if self.max_laps <= 0 else self.max_laps}')

    # ------------------------------------------------------------------
    def _graph_cb(self, msg: GraphLayer):
        self.node_utm = {n.id: (n.easting, n.northing) for n in msg.nodes}
        missing = [g for g in self.loop_goals if g not in self.node_utm]
        if missing:
            self.get_logger().error(f'loop_goals not in graph: {missing}')

    def _utm_cb(self, msg: UtmLayer):
        self.datum = (msg.origin_easting, msg.origin_northing)

    def _path_cb(self, msg: Path):
        # 우리가 goal 을 보낸 뒤 도착한 경로는 채택으로 간주 (임시 구현으로 충분)
        if self.leg_sent_time is not None and not self.leg_adopted:
            self.leg_adopted = True
            self.get_logger().info(
                f'Leg adopted: planned path received for {self._current_goal_id()} '
                f'({len(msg.poses)} poses)')

    def _status_cb(self, msg: ControllerGoalStatus):
        if not self.started or self.leg_sent_time is None:
            return
        if msg.goal_id != self._current_goal_id():
            return
        if not (msg.goal_reached and msg.status_code == 1):
            return

        elapsed = (self.get_clock().now() - self.leg_sent_time).nanoseconds * 1e-9
        if elapsed < self.min_leg_time:
            self.get_logger().warn(
                f'{msg.goal_id} reached in {elapsed:.1f}s < min_leg_time '
                f'{self.min_leg_time}s -> ignored (false/instant reach)',
                throttle_duration_sec=5.0)
            return

        self.get_logger().info(
            f'Leg complete: {msg.goal_id} (elapsed {elapsed:.1f}s, '
            f'dist {msg.distance_to_goal:.2f}m)')
        self._advance()

    # ------------------------------------------------------------------
    def _current_goal_id(self) -> str:
        return self.loop_goals[self.goal_idx]

    def _advance(self):
        if self.goal_idx == len(self.loop_goals) - 1:
            self.lap_count += 1
            self.get_logger().info(f'=== Lap {self.lap_count} complete ===')
            if self.max_laps > 0 and self.lap_count >= self.max_laps:
                self.get_logger().info('max_laps reached, stopping loop')
                self.started = False
                self.leg_sent_time = None
                return
        self.goal_idx = (self.goal_idx + 1) % len(self.loop_goals)
        self._publish_goal()

    def _publish_goal(self):
        goal_id = self._current_goal_id()
        utm = self.node_utm.get(goal_id)
        if utm is None or self.datum is None:
            self.get_logger().error(f'cannot publish goal {goal_id}: graph/datum missing')
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = utm[0] - self.datum[0]
        msg.pose.position.y = utm[1] - self.datum[1]
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

        self.leg_sent_time = self.get_clock().now()
        self.leg_adopted = False
        self.get_logger().info(
            f'Goal published: {goal_id} at map({msg.pose.position.x:.1f}, '
            f'{msg.pose.position.y:.1f})  [leg {self.goal_idx + 1}/{len(self.loop_goals)}]')

    # ------------------------------------------------------------------
    def _tick(self):
        ready = bool(self.node_utm) and self.datum is not None
        if not ready:
            return
        if self.ready_since is None:
            self.ready_since = self.get_clock().now()
            self.get_logger().info('graph + datum ready')

        now = self.get_clock().now()

        # 시작
        if not self.started:
            if not self.autostart:
                return
            if (now - self.ready_since).nanoseconds * 1e-9 < self.start_delay:
                return
            self.started = True
            self._publish_goal()
            return

        # goal 유실 대응: 채택 전이면 주기 재발행
        if self.leg_sent_time is not None and not self.leg_adopted:
            since = (now - self.leg_sent_time).nanoseconds * 1e-9
            if since > self.goal_retry_period:
                self.get_logger().warn(
                    f'no planned path for {self._current_goal_id()} after '
                    f'{since:.0f}s -> republishing goal')
                self._publish_goal()


def main(args=None):
    rclpy.init(args=args)
    node = LoopManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
