#!/usr/bin/env python3
"""
Sequential Global Path Planner Node
Reads JSON map file and publishes sequential path to behavior_planner
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import rcl_interfaces.srv

import os
import math
from typing import List, Dict, Any, Optional

# Import utilities
from utils.map_loader import MapLoader
from utils.visualization import PathVisualizer

# ROS2 messages
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from sensor_msgs.msg import NavSatFix

# Custom messages
from command_center_interfaces.msg import PlannedPath
from gmserver.msg import MapData, MapNode, MapLink, GpsInfo, UtmInfo


class SequentialPlannerNode(Node):
    """Sequential Global Path Planner - reads JSON and publishes path"""
    
    def __init__(self):
        super().__init__('sequential_planner')
        
        # Parameters
        self.declare_parameter('map_file', 'mando_full_map.json')
        self.declare_parameter('auto_start', True)
        self.declare_parameter('loop_path', False)
        self.declare_parameter('publish_frequency', 1.0)  # Hz
        
        # GPS subscription parameter
        self.declare_parameter('gps_topic', '/gps/fix')

        # 시작/종료 노드를 경로에 박아 넣을지. 기본은 False — 즉 빈 문자열을 보낸다.
        # 그러면 behavior planner 가 경로의 첫 노드가 아니라 **로봇에 가장 가까운
        # 노드**부터 따라간다(simple_behavior_planner_node.py:282). 경로 위 아무
        # 지점에서나 출발할 수 있어야 하는데, 첫 노드를 강제하면 멀리 있는 시작점
        # 으로 되돌아가려 한다.
        self.declare_parameter('explicit_endpoints', False)

        # 'first_node'          — 경로 첫 노드의 UtmInfo 를 map 원점으로 (기본)
        # 'tiny_localization'   — 구 tiny_localization 노드에 파라미터 질의 (레거시)
        self.declare_parameter('map_origin_source', 'first_node')

        # Get parameters
        self.map_file = self.get_parameter('map_file').get_parameter_value().string_value
        self.auto_start = self.get_parameter('auto_start').get_parameter_value().bool_value
        self.loop_path = self.get_parameter('loop_path').get_parameter_value().bool_value
        self.publish_freq = self.get_parameter('publish_frequency').get_parameter_value().double_value
        self.gps_topic = self.get_parameter('gps_topic').get_parameter_value().string_value
        self.explicit_endpoints = self.get_parameter(
            'explicit_endpoints').get_parameter_value().bool_value
        self.origin_source = self.get_parameter(
            'map_origin_source').get_parameter_value().string_value
        
        # Map origin - will be set from localization map_origin topic
        self.map_origin_utm_easting = 0.0
        self.map_origin_utm_northing = 0.0
        self.map_origin_set = False
        self._max_sub_seen = 0

        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # 경로는 딱 한 번 발행되므로 반드시 latched(TRANSIENT_LOCAL) 여야 한다.
        # 기동 순서상 플래너(t=7s)가 소비자(smppi t=12s, behavior t=15s)보다
        # 먼저 뜨고 t~9s 에 발행해버리는데, VOLATILE 이면 그 뒤에 구독한 쪽은
        # 경로를 영영 못 받는다 — corridor_keepout 이 "No route yet" 을 무한
        # 반복하며 keepout 없이 코스트맵을 통과시키던 원인.
        # TRANSIENT_LOCAL 제공은 VOLATILE 요청과도 호환되므로 구독자 수정 불필요.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Publishers
        self.path_pub = self.create_publisher(
            PlannedPath, '/planned_path_detailed', latched_qos)
        self.nav_path_pub = self.create_publisher(
            Path, '/sequential_path_nav', reliable_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/sequential_path_markers', reliable_qos)
        self.status_pub = self.create_publisher(
            String, '/sequential_planner_status', reliable_qos)
        
        # Localization node name for parameter access
        self.localization_node_name = 'tiny_localization_node'
        
        # Services (for future expansion)
        # self.start_service = self.create_service(...)
        
        # Data storage
        self.nodes_data = {}  # {node_id: node_data}
        self.links_data = []  # [link_data, ...]
        self.ordered_nodes = []  # Sequential order of nodes
        self.is_loaded = False
        self.path_published = False  # Track if path has been published
        
        # Initialize map loader
        self.map_loader = MapLoader(logger=self.get_logger())
        
        # Timer for publishing
        self.publish_timer = self.create_timer(
            1.0 / self.publish_freq, self.publish_callback)
        
        # Timer to check for map origin parameters
        self.param_check_timer = self.create_timer(1.0, self.check_map_origin_params)
        
        # Load map data
        self.load_map_file()
        
        if self.auto_start and self.is_loaded:
            self.publish_status("Sequential planner started - auto publishing path")
            
        # Try to get parameters immediately on startup
        self.check_map_origin_params()
        
        self.get_logger().info(f'Sequential Planner Node initialized')
        self.get_logger().info(f'Map file: {self.map_file}')
        self.get_logger().info(f'Loaded {len(self.nodes_data)} nodes, {len(self.links_data)} links')
    
    def load_map_file(self) -> bool:
        """Load JSON map file and extract nodes/links"""
        # Get full path to map file
        map_path = self.map_loader.get_map_path('sequential_global_planner', self.map_file)
        
        # Load map data using map loader
        self.nodes_data, self.links_data, success = self.map_loader.load_map_file(map_path)
        
        if success:
            # Create ordered node sequence from links
            self.create_sequential_order()
            self.is_loaded = True
        else:
            self.is_loaded = False
            
        return success
    
    def create_sequential_order(self) -> None:
        """Create ordered sequence of nodes following the links"""
        if not self.links_data:
            # If no links, use nodes in ID order
            self.ordered_nodes = sorted(self.nodes_data.keys())
            return
        
        # Build adjacency list from links
        adjacency = {}
        for link in self.links_data:
            from_node = link['FromNodeID']
            to_node = link['ToNodeID']
            
            if from_node not in adjacency:
                adjacency[from_node] = []
            adjacency[from_node].append(to_node)
        
        # Find starting node (node with no incoming links)
        incoming = set()
        for link in self.links_data:
            incoming.add(link['ToNodeID'])
        
        start_nodes = [node_id for node_id in self.nodes_data.keys() if node_id not in incoming]
        
        if start_nodes:
            start_node = start_nodes[0]  # Use first available start node
        else:
            start_node = list(self.nodes_data.keys())[0]  # Fallback to first node
        
        # Follow the chain of links
        self.ordered_nodes = []
        current = start_node
        visited = set()
        
        while current and current not in visited:
            self.ordered_nodes.append(current)
            visited.add(current)
            
            # Find next node
            next_nodes = adjacency.get(current, [])
            current = next_nodes[0] if next_nodes else None
        
        self.get_logger().info(f'Created sequential order: {len(self.ordered_nodes)} nodes')
        if len(self.ordered_nodes) > 0:
            self.get_logger().info(f'Path: {self.ordered_nodes[0]} -> ... -> {self.ordered_nodes[-1]}')
    
    def calculate_heading(self, from_easting: float, from_northing: float, 
                         to_easting: float, to_northing: float) -> float:
        """Calculate heading between two UTM points in degrees (0-360)"""
        dx = to_easting - from_easting
        dy = to_northing - from_northing
        
        # Calculate heading in radians
        heading_rad = math.atan2(dy, dx)
        
        # Convert to degrees and normalize to 0-360
        heading_deg = math.degrees(heading_rad)
        if heading_deg < 0:
            heading_deg += 360.0
            
        return heading_deg
    
    def calculate_node_headings(self, map_data: MapData) -> None:
        """Calculate headings for nodes with heading = 0.0 based on sequential order"""
        for i, node in enumerate(map_data.nodes):
            if abs(node.heading) < 1e-6:  # heading is approximately 0.0
                if i > 0:
                    # Calculate heading from previous node
                    prev_node = map_data.nodes[i-1]
                    calculated_heading = self.calculate_heading(
                        prev_node.utm_info.easting, prev_node.utm_info.northing,
                        node.utm_info.easting, node.utm_info.northing
                    )
                    node.heading = calculated_heading
                    
                    self.get_logger().debug(
                        f'Calculated heading for node {node.id}: {calculated_heading:.2f} degrees'
                    )
    
    def create_planned_path_message(self) -> PlannedPath:
        """Create PlannedPath message compatible with behavior_planner"""
        planned_path = PlannedPath()
        planned_path.header = Header()
        planned_path.header.stamp = self.get_clock().now().to_msg()
        planned_path.header.frame_id = 'map'
        
        # Path metadata
        planned_path.path_id = "sequential_path"
        if self.explicit_endpoints and self.ordered_nodes:
            planned_path.start_node_id = self.ordered_nodes[0]
            planned_path.goal_node_id = self.ordered_nodes[-1]
        else:
            planned_path.start_node_id = ""
            planned_path.goal_node_id = ""
        planned_path.total_distance = 0.0  # Can calculate if needed
        planned_path.total_time = 0.0      # Can calculate if needed
        
        # Create MapData with nodes and links
        map_data = MapData()
        
        # Convert nodes to MapNode messages
        for node_id in self.ordered_nodes:
            node_data = self.nodes_data[node_id]
            
            map_node = MapNode()
            map_node.id = node_id
            map_node.admin_code = node_data.get('AdminCode', '110')
            map_node.node_type = node_data.get('NodeType', 1)
            map_node.its_node_id = node_data.get('ITSNodeID', f'ITS_{node_id}')
            map_node.maker = node_data.get('Maker', '한국도로공사')
            map_node.update_date = node_data.get('UpdateDate', '20250418')
            map_node.version = node_data.get('Version', '2021')
            map_node.remark = node_data.get('Remark', '')
            map_node.hist_type = node_data.get('HistType', '02A')
            map_node.hist_remark = node_data.get('HistRemark', '')
            # Handle heading field - use value if present, default to 0.0 if missing
            if 'Heading' in node_data:
                map_node.heading = node_data['Heading']
            else:
                map_node.heading = 0.0
                self.get_logger().debug(f"Node {node_id} has no Heading key, using default 0.0")
            
            # GPS info
            map_node.gps_info.lat = node_data['GpsInfo']['Lat']
            map_node.gps_info.longitude = node_data['GpsInfo']['Long']
            map_node.gps_info.alt = node_data['GpsInfo']['Alt']
            
            # PlannedPath 의 utm_info 는 **map 원점 상대 좌표**다. 절대 UTM 이
            # 아니다 — A* 플래너도 발행 직전에 node[0] UTM 을 빼고
            # (path_planner_node.cpp:1361), 소비자인 behavior planner 는 이 값을
            # 그대로 map 프레임 x/y 로 써서 /odometry/global 과 거리 비교를 한다
            # (path_manager.py:33,52).
            #
            # 여기서만 절대 UTM 을 넣고 있었다. 그러면 48만 대 −35 를 비교하게
            # 되어 "최근접 노드"가 기하와 무관해진다 — 실측: 로봇을 B060 위에
            # 놓았는데 28.6 m 떨어진 B032 를 골랐다. 같은 파일의
            # create_nav_path_message 는 이미 원점을 빼고 있어 한 파일 안에서
            # 두 메시지가 서로 다른 프레임을 쓰고 있었다.
            map_node.utm_info.easting = (
                node_data['UtmInfo']['Easting'] - self.map_origin_utm_easting)
            map_node.utm_info.northing = (
                node_data['UtmInfo']['Northing'] - self.map_origin_utm_northing)
            map_node.utm_info.zone = node_data['UtmInfo']['Zone']
            
            map_data.nodes.append(map_node)
        
        # Add relevant links
        for link_data in self.links_data:
            from_id = link_data['FromNodeID']
            to_id = link_data['ToNodeID']
            
            # Only include links that are part of our sequential path
            if from_id in self.ordered_nodes and to_id in self.ordered_nodes:
                map_link = MapLink()
                map_link.id = link_data.get('ID', '')
                map_link.admin_code = link_data.get('AdminCode', '110')
                map_link.road_rank = link_data.get('RoadRank', 1)
                map_link.road_type = link_data.get('RoadType', 1)
                map_link.road_no = link_data.get('RoadNo', '20')
                map_link.link_type = link_data.get('LinkType', 3)
                map_link.lane_no = link_data.get('LaneNo', 2)
                map_link.from_node_id = from_id
                map_link.to_node_id = to_id
                map_link.length = link_data.get('Length', 0.1)
                
                map_data.links.append(map_link)
        
        # Calculate headings for nodes with heading = 0.0
        self.calculate_node_headings(map_data)
        
        # Loop closure if enabled
        if self.loop_path and len(map_data.nodes) > 1:
            # Add first node at the end
            map_data.nodes.append(map_data.nodes[0])
        
        planned_path.path_data = map_data
        return planned_path
    
    def create_nav_path_message(self) -> Path:
        """Create nav_msgs/Path for RViz visualization"""
        nav_path = Path()
        nav_path.header = Header()
        nav_path.header.stamp = self.get_clock().now().to_msg()
        nav_path.header.frame_id = 'odom'
        
        for node_id in self.ordered_nodes:
            node_data = self.nodes_data[node_id]
            
            pose = PoseStamped()
            pose.header = nav_path.header
            
            # Convert UTM to odom frame (subtract map origin)
            pose.pose.position.x = node_data['UtmInfo']['Easting'] - self.map_origin_utm_easting
            pose.pose.position.y = node_data['UtmInfo']['Northing'] - self.map_origin_utm_northing
            pose.pose.position.z = 0.0
            
            # Set orientation (pointing to next node)
            pose.pose.orientation.w = 1.0  # No rotation for now
            
            nav_path.poses.append(pose)
        
        if self.loop_path and len(nav_path.poses) > 1:
            nav_path.poses.append(nav_path.poses[0])
        
        return nav_path
    
    def create_visualization_markers(self) -> MarkerArray:
        """Create visualization markers for nodes and links"""
        # Use PathVisualizer to create all markers
        return PathVisualizer.create_marker_array(
            ordered_nodes=self.ordered_nodes,
            nodes_data=self.nodes_data,
            map_origin_utm_easting=self.map_origin_utm_easting,
            map_origin_utm_northing=self.map_origin_utm_northing,
            timestamp=self.get_clock().now().to_msg(),
            loop_path=self.loop_path,
            include_text=True,
            frame_id='odom'
        )
    
    def publish_callback(self) -> None:
        """Timer callback to publish path and visualization"""
        if not self.is_loaded or not self.auto_start or not self.map_origin_set:
            return
        # 한 번만 발행하되, **구독자가 새로 붙으면 다시 발행한다.**
        #
        # latched(TRANSIENT_LOCAL) 만으로는 부족하다: QoS 호환성과 이력 전달은
        # 별개라, VOLATILE 로 구독하는 쪽(behavior planner)은 자기가 붙기 전에
        # 발행된 샘플을 받지 못한다. 기동 순서상 플래너(t=7s)가 소비자
        # (behavior t=15s)보다 먼저 발행하므로, 고치지 않으면 경로가 아무에게도
        # 도달하지 않는다. 구독자를 TRANSIENT_LOCAL 로 바꾸는 방법은 쓸 수 없다 —
        # A* 플래너는 VOLATILE 로 발행해서 그쪽과 연결이 아예 끊긴다.
        #
        # 매 주기 재발행하지 않는 이유: 콜백이 최근접 노드 재정렬과 subgoal
        # 재발행을 유발하므로 주행 중 계속 때리면 진행을 방해한다.
        n_sub = self.path_pub.get_subscription_count()
        if not self.path_published or n_sub > self._max_sub_seen:
            planned_path = self.create_planned_path_message()
            self.path_pub.publish(planned_path)
            first = not self.path_published
            self.path_published = True
            self._max_sub_seen = max(self._max_sub_seen, n_sub)
            self.publish_status(
                f"Published sequential path to {n_sub} subscriber(s)"
                f"{' (first)' if first else ' (new subscriber joined)'}")

        # Continue publishing visualization for RViz
        nav_path = self.create_nav_path_message()
        self.nav_path_pub.publish(nav_path)
        
        # Publish visualization markers
        markers = self.create_visualization_markers()
        self.marker_pub.publish(markers)

    def set_origin_from_first_node(self) -> bool:
        """Take the map frame origin from the route's first node.

        The old path asked /localization/tiny_localization_node for the origin,
        but that node is gone — robot_localization replaced it — so the query
        never succeeded and publish_callback's map_origin_set gate stayed shut
        forever. The planner loaded the map, reported success, and silently
        published nothing.

        node[0] is the right answer anyway, not a workaround: the navsat datum
        in scv_dual_ekf.yaml is *defined* as the graph map's node[0], so the
        map frame's origin already is this point. Reading it back from the map
        keeps one source of truth instead of a constant copied into two files.
        """
        if not self.ordered_nodes:
            return False
        n = self.nodes_data.get(self.ordered_nodes[0], {})
        utm = n.get('UtmInfo') or {}
        if 'Easting' not in utm or 'Northing' not in utm:
            self.get_logger().warn(
                'first node has no UtmInfo — cannot set map origin')
            return False
        self.map_origin_utm_easting = float(utm['Easting'])
        self.map_origin_utm_northing = float(utm['Northing'])
        self.map_origin_set = True
        self.get_logger().info(
            f'map origin from first node {self.ordered_nodes[0]}: '
            f'({self.map_origin_utm_easting:.2f}, '
            f'{self.map_origin_utm_northing:.2f})')
        return True

    def check_map_origin_params(self) -> None:
        """Check if map origin parameters are available from localization node"""
        if not self.map_origin_set and self.origin_source == 'first_node':
            if self.set_origin_from_first_node():
                return
        if not self.map_origin_set and self.origin_source == 'tiny_localization':
            try:
                # Create a parameter client for the localization node
                from rclpy.parameter import Parameter
                from rclpy.node import Node
                
                # Get parameters from the localization node using parameter client
                param_client = self.create_client(
                    rcl_interfaces.srv.GetParameters,
                    '/localization/tiny_localization_node/get_parameters'
                )
                
                if not param_client.wait_for_service(timeout_sec=0.1):
                    self.get_logger().debug('Localization node parameter service not available yet')
                    return
                
                # Request parameters
                request = rcl_interfaces.srv.GetParameters.Request()
                request.names = [
                    'map_origin.utm_easting',
                    'map_origin.utm_northing', 
                    'map_origin.utm_zone'
                ]
                
                future = param_client.call_async(request)
                
                # Use executor to wait for response with timeout
                import time
                start_time = time.time()
                while not future.done() and (time.time() - start_time) < 0.5:
                    rclpy.spin_once(self, timeout_sec=0.01)
                
                if future.done():
                    response = future.result()
                    if response and len(response.values) == 3:
                        # Extract parameter values
                        easting = response.values[0].double_value
                        northing = response.values[1].double_value
                        zone = response.values[2].integer_value
                        
                        # Validate values (check they're not default/zero)
                        if easting != 0.0 and northing != 0.0:
                            # Successfully got all parameters
                            self.map_origin_utm_easting = easting
                            self.map_origin_utm_northing = northing
                            self.map_origin_set = True
                            
                            # Cancel parameter check timer
                            self.param_check_timer.cancel()
                            
                            self.get_logger().info('Map origin received from localization parameters:')
                            self.get_logger().info(f'  UTM: ({self.map_origin_utm_easting:.4f}, {self.map_origin_utm_northing:.4f}) Zone {zone}')
                            
                            # Now that we have map origin, start publishing if auto_start is enabled
                            if self.auto_start and self.is_loaded and not self.path_published:
                                self.publish_status("Map origin set - starting path publication")
                            return
                        else:
                            self.get_logger().debug('Map origin parameters are still at default values')
                else:
                    self.get_logger().debug('Timeout waiting for parameter response')
                        
            except Exception as e:
                self.get_logger().debug(f'Map origin parameters not ready: {e}')
    
    def publish_status(self, message: str) -> None:
        """Publish status message"""
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)
        self.get_logger().info(f'Status: {message}')


def main(args=None):
    rclpy.init(args=args)
    
    node = SequentialPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()