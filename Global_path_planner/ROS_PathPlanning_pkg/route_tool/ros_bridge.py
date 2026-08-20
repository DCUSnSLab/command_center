"""ROS 쪽 배선 — rclpy 는 이 파일에만 있다.

별도 스레드에서 spin 하고 결과를 Qt 시그널로 넘긴다. GUI 스레드에서 rclpy 를 돌리면
콜백이 화면을 멈추고, 반대로 콜백에서 위젯을 직접 만지면 크래시한다.
"""
import math
import threading

from PyQt6.QtCore import QObject, pyqtSignal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from map_interfaces.msg import GraphLayer, UtmLayer
from command_center_interfaces.msg import RouteRequest, RouteStatus


def latched(depth=1):
    q = QoSProfile(depth=depth)
    q.durability = DurabilityPolicy.TRANSIENT_LOCAL
    q.reliability = ReliabilityPolicy.RELIABLE
    return q


class RosBridge(QObject):
    graphReceived = pyqtSignal(dict, list)      # nodes, links
    datumReceived = pyqtSignal(float, float, int, bool)
    robotMoved = pyqtSignal(float, float, float)
    pathReceived = pyqtSignal(list)
    statusReceived = pyqtSignal(str, bool, str, int)   # id, accepted, reason, count

    def __init__(self, parent=None):
        super().__init__(parent)
        rclpy.init()
        self.node = Node("route_tool")
        self._datum = None

        self.node.create_subscription(GraphLayer, "/map_provider_node/graph",
                                      self._on_graph, latched())
        self.node.create_subscription(UtmLayer, "/map_provider_node/utm",
                                      self._on_utm, latched())
        self.node.create_subscription(Path, "/planned_path", self._on_path, 10)
        self.node.create_subscription(RouteStatus, "/route_status",
                                      self._on_status, latched(10))
        # 요청은 latched 로 낸다 — 플래너가 재시작해도 지금 경로를 다시 받는다.
        self.pub = self.node.create_publisher(RouteRequest, "/route_request", latched())

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.node.create_timer(0.3, self._poll_tf)

        # spin 은 start() 에서 시작한다. 생성자에서 바로 돌리면 latched 메시지가
        # 창이 시그널을 연결하기 **전에** 도착해 그대로 사라진다.
        self._stop = False
        self._thread = None
        self._last_graph = None

    def start(self):
        """구독자들이 연결을 마친 뒤 호출한다."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while rclpy.ok() and not self._stop:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    # ------------------------------------------------------------ 수신
    def _on_graph(self, msg):
        self._last_graph = msg
        if self._datum is None:
            # datum 이 아직이면 절대 UTM 을 그릴 수 없다. 다음 수신 때 다시 온다.
            return   # datum 이 오면 _on_utm 이 다시 내보낸다
        self.graphReceived.emit(*self._convert(msg))

    def _convert(self, msg):
        e, n = self._datum[0], self._datum[1]
        nodes = {nd.id: (nd.easting - e, nd.northing - n, nd.source or "slam")
                 for nd in msg.nodes}
        links = [(lk.from_node_id, lk.to_node_id, bool(lk.bidirectional))
                 for lk in msg.links]
        return nodes, links

    def _on_utm(self, msg):
        self._datum = (msg.origin_easting, msg.origin_northing,
                       int(msg.utm_zone), bool(msg.northern))
        self.datumReceived.emit(*self._datum)
        # datum 이 늦게 와도 이미 받아둔 그래프를 바로 살린다.
        if self._last_graph is not None:
            self.graphReceived.emit(*self._convert(self._last_graph))

    def _on_path(self, msg):
        self.pathReceived.emit([(p.pose.position.x, p.pose.position.y) for p in msg.poses])

    def _on_status(self, msg):
        self.statusReceived.emit(msg.request_id, msg.accepted, msg.reason, msg.node_count)

    def _poll_tf(self):
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.robotMoved.emit(tf.transform.translation.x, tf.transform.translation.y, yaw)

    # ------------------------------------------------------------ 송신
    def publish_route(self, request_id, via_nodes, goal_xy=None,
                      loop=False, fill_gaps=True):
        msg = RouteRequest()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.request_id = request_id
        msg.via_nodes = list(via_nodes)
        msg.loop = bool(loop)
        msg.fill_gaps = bool(fill_gaps)
        if goal_xy is not None:
            g = PoseStamped()
            g.header = msg.header
            g.pose.position.x = float(goal_xy[0])
            g.pose.position.y = float(goal_xy[1])
            g.pose.orientation.w = 1.0
            msg.goal = g
            msg.use_goal = True
        else:
            msg.use_goal = False
        self.pub.publish(msg)

    def shutdown(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        try:
            self.node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
