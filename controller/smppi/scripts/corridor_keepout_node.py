#!/usr/bin/env python3
"""
Corridor Keepout Node for SMPPI

Sits between local_costmap (/costmap) and the MPPI controller. Everything
outside a keep-in corridor around the planned route is overwritten as LETHAL,
so obstacle avoidance can never leave the sidewalk (e.g. drop over the curb
onto the road) even if perception misses the curb itself. This is the map-side
second line of defence next to the curb_detection_node (perception-side).

Corridor construction (all in the costmap frame, usually 'odom'):
  - polyline = [robot position] + current_goal + next_waypoints
    (from /multiple_waypoints, already transformed to odom by the
    behavior planner)
  - corridor  = polyline dilated by corridor_half_width
  - gates     = circles of gate_radius around waypoints whose node_type is in
    gate_node_types (e.g. signalised crossings) — inside a gate the keepout is
    NOT applied, so legal road crossings stay drivable.

Output cell = max(input cell, keepout), i.e. real obstacles are preserved.
If no waypoints have been received yet the costmap passes through unchanged
(publishes a warning throttled) — safe default for bring-up.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from command_center_interfaces.msg import MultipleWaypoints

LETHAL = 100


class CorridorKeepoutNode(Node):
    def __init__(self):
        super().__init__('corridor_keepout_node')

        self.declare_parameter('input_costmap_topic', '/costmap')
        self.declare_parameter('output_costmap_topic', '/costmap_keepout')
        self.declare_parameter('waypoints_topic', '/multiple_waypoints')
        self.declare_parameter('odom_topic', '/odom')

        self.declare_parameter('corridor_half_width', 1.4)   # m from route centreline
        # Real graph maps space nodes only 1.0-2.1 m apart, so the 1+3 lookahead
        # waypoints reach barely ~4 m -- the same as the MPPI horizon. Extend the
        # corridor past the last waypoint along its heading so normal driving is
        # never throttled by the corridor's far end.
        self.declare_parameter('forward_extension', 8.0)
        self.declare_parameter('gate_node_types', [10])      # crossing node types
        self.declare_parameter('gate_radius', 3.0)           # m opened around gate nodes
        self.declare_parameter('keepout_cost', LETHAL)
        self.declare_parameter('passthrough_without_route', True)

        gp = self.get_parameter
        self.in_topic = gp('input_costmap_topic').value
        self.out_topic = gp('output_costmap_topic').value
        self.wp_topic = gp('waypoints_topic').value
        self.odom_topic = gp('odom_topic').value
        self.half_width = float(gp('corridor_half_width').value)
        self.forward_ext = float(gp('forward_extension').value)
        self.gate_types = set(int(v) for v in gp('gate_node_types').value)
        self.gate_radius = float(gp('gate_radius').value)
        self.keepout_cost = int(gp('keepout_cost').value)
        self.passthrough = bool(gp('passthrough_without_route').value)

        self.route_pts = None      # [N,2] polyline in costmap frame
        self.gate_pts = None       # [M,2] gate centres
        self.robot_xy = None

        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        best_effort = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(OccupancyGrid, self.in_topic, self.costmap_cb, reliable)
        self.create_subscription(MultipleWaypoints, self.wp_topic, self.waypoints_cb, reliable)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, best_effort)
        self.pub = self.create_publisher(OccupancyGrid, self.out_topic, reliable)

        self.get_logger().info(
            f"Corridor keepout ready: {self.in_topic} -> {self.out_topic}, "
            f"half_width={self.half_width} m, fwd_ext={self.forward_ext} m, "
            f"gates={sorted(self.gate_types)} r={self.gate_radius} m")

    # ---------- inputs ----------

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.robot_xy = (float(p.x), float(p.y))

    def waypoints_cb(self, msg: MultipleWaypoints):
        pts = [(msg.current_goal.pose.position.x, msg.current_goal.pose.position.y)]
        types = [int(msg.current_goal_node_type)]
        for wp, nt in zip(msg.next_waypoints, msg.next_waypoints_node_types):
            pts.append((wp.pose.position.x, wp.pose.position.y))
            types.append(int(nt))
        pts_arr = np.array(pts, dtype=np.float64)
        # extend past the final waypoint along the last segment direction
        if self.forward_ext > 0.0 and len(pts_arr) >= 2:
            d = pts_arr[-1] - pts_arr[-2]
            n = np.linalg.norm(d)
            if n > 1e-6:
                pts_arr = np.vstack([pts_arr, pts_arr[-1] + d / n * self.forward_ext])
        self.route_pts = pts_arr
        gates = [pts[i] for i in range(len(pts)) if types[i] in self.gate_types]
        self.gate_pts = np.array(gates, dtype=np.float64) if gates else None

    # ---------- core ----------

    def costmap_cb(self, msg: OccupancyGrid):
        if self.route_pts is None:
            if self.passthrough:
                self.pub.publish(msg)
                self.get_logger().warn(
                    'No route yet — passing costmap through WITHOUT keepout',
                    throttle_duration_sec=5.0)
            return

        h, w = msg.info.height, msg.info.width
        res = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y

        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)

        # corridor polyline: robot position first (so the corridor always
        # covers the vehicle), then the lookahead waypoints
        pts = self.route_pts
        if self.robot_xy is not None:
            pts = np.vstack([np.array(self.robot_xy)[None, :], pts])

        # world -> grid pixels (col=x, row=y)
        px = ((pts[:, 0] - ox) / res).round().astype(np.int32)
        py = ((pts[:, 1] - oy) / res).round().astype(np.int32)
        poly = np.stack([px, py], axis=1).reshape(-1, 1, 2)

        corridor = np.zeros((h, w), dtype=np.uint8)
        thickness = max(1, int(round(2.0 * self.half_width / res)))
        cv2.polylines(corridor, [poly], isClosed=False, color=1,
                      thickness=thickness, lineType=cv2.LINE_8)
        # single-waypoint degenerate case: polylines may draw nothing
        if corridor.max() == 0:
            for cx, cy in zip(px, py):
                cv2.circle(corridor, (int(cx), int(cy)),
                           max(1, thickness // 2), 1, -1)

        # open gates (legal crossings)
        if self.gate_pts is not None:
            gr = max(1, int(round(self.gate_radius / res)))
            for gx_w, gy_w in self.gate_pts:
                gx = int(round((gx_w - ox) / res))
                gy = int(round((gy_w - oy) / res))
                cv2.circle(corridor, (gx, gy), gr, 1, -1)

        out = grid.copy()
        outside = corridor == 0
        out[outside] = np.maximum(out[outside], np.int8(self.keepout_cost))

        out_msg = OccupancyGrid()
        out_msg.header = msg.header
        out_msg.info = msg.info
        out_msg.data = out.reshape(-1).tolist()
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CorridorKeepoutNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
