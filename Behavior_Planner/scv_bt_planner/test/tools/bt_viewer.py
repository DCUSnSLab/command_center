#!/usr/bin/env python3
"""SCV BehaviorTree 실시간 뷰어 — BT.CPP 4.x Groot2Publisher(ZMQ) 에 붙어 트리와 '마지막 틱의 경로'를 그린다.

왜 Groot2 앱이 아닌가: Groot2 는 독점 바이너리이고 실시간 모니터 일부가 유료이며, 차량·Pod 에서 내려받기도
막혀 있다(10/02 S3 403). 프로토콜은 BT.CPP 소스(loggers/groot2_protocol.h, groot2_publisher.cpp)에 공개돼 있다.

왜 STATUS 폴링이 아니라 전이 기록인가: scv_behavior.xml 은 20 Hz 결정 트리라 모든 노드가 같은 틱 안에서
끝나고, 틱이 끝나면 루트가 resetStatus 한다. STATUS 버퍼에는 '10+직전 상태'만 남아 이번 틱에 실제로 거친
노드와 몇 틱 전에 거친 노드(예: Fallback 의 뒤쪽 분기)를 구분할 수 없다. TOGGLE_RECORDING('r', "start") 뒤
GET_TRANSITIONS('t') 로 전이를 받아 루트 RUNNING→완료 구간 하나를 '마지막 완료 틱'으로 잘라 칠한다.

정보 패널은 DDS 참여자를 새로 만들지 않도록 ROS 구독 대신 가장 최근 bt.log 꼬리를 읽는다
(Pod 에서 런 중 새 참여자가 간헐적으로 디스커버리되지 않는 문제를 피한다).

지도 패널(--design 지정 시 두 번째 창): 챔버 설계 JSON 의 월드(보도 띠·장애물)와 경로 그래프 위에 차량 위치·자세·궤적,
현재 목표 노드, 지나온 노드를 그린다. 차량 위치는 Gazebo 자체 전송(`gz topic -e /gazebo/<world>/pose/info -u`)에서
읽는다 — ROS/DDS 를 거치지 않는다. 목표 노드는 bt.log 꼬리에서 읽는다(simple 런이면 behavior.log 로 대체).

사용:
  bt_viewer.py [--host 127.0.0.1] [--port 1667] [--log-glob '~/scv_sim/bt_ab/**/bt.log'] [--geometry 540x1000+1380+29]
               [--design chamber.design.json --map-geometry 1380x500+0+529]
BT 쪽: bt_planner_node --ros-args -p groot2_port:=1667   (서버 1667, 퍼블리셔 1668)
"""
import argparse
import glob
import os
import random
import re
import struct
import time
import xml.etree.ElementTree as ET

import zmq
from PyQt5 import QtCore, QtGui, QtWidgets

PROTOCOL = 2
IDLE, RUNNING, SUCCESS, FAILURE, SKIPPED = 0, 1, 2, 3, 4
STATUS_NAME = {IDLE: 'IDLE', RUNNING: 'RUNNING', SUCCESS: 'SUCCESS', FAILURE: 'FAILURE', SKIPPED: 'SKIPPED'}
HIDDEN_ATTRS = {'_uid', '_fullpath', 'name', 'ID', '_autoremap', '_skipIf', '_successIf', '_failureIf',
                '_while', '_onSuccess', '_onFailure', '_onHalted', '_post'}

# 팔레트 — Gazebo 의 밝은 회색 화면 옆에서 구분되도록 어두운 패널
C = {
    'bg': '#1c2024', 'panel': '#24292e', 'ink': '#e6e8ea', 'muted': '#8b949e', 'line': '#3a4047',
    'idle_box': '#2d333b', SUCCESS: '#2f9e57', FAILURE: '#d4473f', RUNNING: '#e5a23a', SKIPPED: '#6e7681',
    'accent': '#58a6ff',
}


# ---------------------------------------------------------------- Groot2 클라이언트
class Groot2Client:
    def __init__(self, host, port, timeout_ms=400):
        self.ctx = zmq.Context.instance()
        self.addr = f'tcp://{host}:{port}'
        self.timeout = timeout_ms
        self.sock = None
        self._open()

    def _open(self):
        if self.sock is not None:
            self.sock.close(linger=0)
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, self.timeout)
        s.setsockopt(zmq.SNDTIMEO, self.timeout)
        s.connect(self.addr)
        self.sock = s

    def request(self, rtype, *parts):
        """(tree_id, [body...]) 또는 None. 타임아웃이면 REQ 소켓 상태가 꼬이므로 새로 연다."""
        header = struct.pack('<BBI', PROTOCOL, ord(rtype), random.getrandbits(32))
        try:
            self.sock.send_multipart([header] + [p.encode() if isinstance(p, str) else p for p in parts])
            rep = self.sock.recv_multipart()
        except zmq.ZMQError:
            self._open()
            return None
        if not rep or rep[0] == b'error' or len(rep[0]) < 22:
            return None
        return rep[0][6:22], rep[1:]


# ---------------------------------------------------------------- 트리 모델
class Row:
    __slots__ = ('uid', 'depth', 'tag', 'label', 'detail', 'last_child', 'ancestors_last')

    def __init__(self, uid, depth, tag, label, detail, last_child, ancestors_last):
        self.uid, self.depth, self.tag, self.label, self.detail = uid, depth, tag, label, detail
        self.last_child, self.ancestors_last = last_child, ancestors_last


def parse_tree(xml_text):
    root = ET.fromstring(xml_text)
    trees = {e.get('ID'): e for e in root if e.tag == 'BehaviorTree'}
    main_id = root.get('main_tree_to_execute') or next(iter(trees))
    rows = []

    def walk(elem, depth, last, anc):
        uid = elem.get('_uid')
        name = elem.get('name')
        label = elem.tag if not name or name == elem.tag else f'{name}'
        kind = '' if not name or name == elem.tag else elem.tag
        ports = [f'{k}={v}' for k, v in elem.attrib.items() if k not in HIDDEN_ATTRS and not k.startswith('_')]
        detail = '  '.join(([kind] if kind else []) + ports)
        if uid is not None:
            rows.append(Row(int(uid), depth, elem.tag, label, detail, last, tuple(anc)))
        kids = list(elem)
        if elem.tag == 'SubTree' and elem.get('ID') in trees:
            kids = list(trees[elem.get('ID')])
        for i, k in enumerate(kids):
            walk(k, depth + 1, i == len(kids) - 1, anc + [last])

    top = list(trees[main_id])
    for i, e in enumerate(top):
        walk(e, 0, i == len(top) - 1, [])
    return rows


# ---------------------------------------------------------------- bt.log 꼬리 → 정보 패널
LOG_PATTERNS = [
    ('mode', re.compile(r'\[MODE\] hunter control_mode -> (\d+) \((.*?)\)(?:\s+—.*)?$')),
    ('target', re.compile(r'advanced to (\S+) \((\d+)/(\d+)\)')),
    ('join', re.compile(r'join idx (\d+) \((\S+)\)')),
    ('behavior', re.compile(r'\[BEHAVIOR\] (.*)')),
    ('blocked', re.compile(r'\[BLOCKED\] (state -> \w+|cleared -> \w+)')),
    ('resync', re.compile(r'\[shadow\] target resync (\S+ -> \S+)')),
    ('done', re.compile(r'Path following completed!')),
]
HAZ_SHIFT_RE = re.compile(r'\[HAZARD\] shift (\S+) by .*-> at \(([-\d.]+), ([-\d.]+)\)')
HAZ_RESTORE_RE = re.compile(r'\[HAZARD\] restore (\S+)')
HAZ_SKIP_RE = re.compile(r'\[HAZARD\] skip (\S+):')
EVENT_RE = re.compile(r'\[(?:INFO|WARN|ERROR)\] \[(\d+\.\d+)\] \[bt_planner\]: (.*)')
EVENT_KEEP = re.compile(r'MODE\]|advanced to|BEHAVIOR\]|BLOCKED\]|\[shadow\]|\[HAZARD\]|join idx|completed|assist')


class LogTail:
    def __init__(self, pattern):
        self.pattern = os.path.expanduser(pattern)
        self.path = None
        self.checked = 0.0

    def read(self):
        now = time.monotonic()
        if self.path is None or now - self.checked > 3.0:
            self.checked = now
            files = glob.glob(self.pattern, recursive=True)
            if files:
                self.path = max(files, key=os.path.getmtime)
        if not self.path:
            return None
        try:
            with open(self.path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 96 * 1024))
                text = f.read().decode('utf-8', 'replace')
        except OSError:
            return None
        info = {'run': os.path.basename(os.path.dirname(self.path)).replace('logs_', ''), 'events': []}
        for line in text.splitlines():
            for key, rx in LOG_PATTERNS:
                m = rx.search(line)
                if m:
                    info[key] = m.groups() if m.groups() else True
            m = EVENT_RE.search(line)
            if m and EVENT_KEEP.search(m.group(2)):
                info['events'].append((float(m.group(1)), m.group(2)))
        info['events'] = info['events'][-7:]
        # 위험 지대 경유점 재배치: 노드 → 재배치된 절대 좌표(UTM — 챔버는 datum 0 이라 월드 좌표와 같다)
        shifted, skipped = {}, []
        for line in text.splitlines():
            m = HAZ_SHIFT_RE.search(line)
            if m:
                shifted[m.group(1)] = (float(m.group(2)), float(m.group(3)))
                continue
            m = HAZ_RESTORE_RE.search(line)
            if m:
                shifted.pop(m.group(1), None)
                continue
            m = HAZ_SKIP_RE.search(line)
            if m and m.group(1) not in skipped:
                skipped.append(m.group(1))
        info['shifted'], info['skipped'] = shifted, skipped
        return info


# ---------------------------------------------------------------- 위젯
class BtView(QtWidgets.QWidget):
    ROW_H = 20

    def __init__(self, args):
        super().__init__()
        self.client = Groot2Client(args.host, args.port)
        self.port = args.port
        self.log = LogTail(args.log_glob)
        self.rows, self.tree_id, self.root_uid = [], None, None
        self.last_tick = {}            # uid -> status (마지막 완료 틱에서 거친 노드만)
        self.pending = []
        self.tick_times = []           # 완료 틱 타임스탬프(us) — 틱 주기 표시
        self.connected = False
        self.info = None
        self.last_try = 0.0
        self.setWindowTitle('SCV BT Monitor')
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        self.font_main = QtGui.QFont('DejaVu Sans', 10)
        self.font_small = QtGui.QFont('DejaVu Sans', 8)
        self.font_mono = QtGui.QFont('DejaVu Sans Mono', 8)
        self.font_title = QtGui.QFont('DejaVu Sans', 12, QtGui.QFont.Bold)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)          # 10 Hz — 전이 버퍼(1000개, 틱당 ~40개) 넘침 방지
        self.log_timer = QtCore.QTimer(self)
        self.log_timer.timeout.connect(self.poll_log)
        self.log_timer.start(500)

    # ---- 통신
    def connect_tree(self):
        r = self.client.request('T')
        if r is None:
            return False
        tree_id, body = r
        try:
            self.rows = parse_tree(body[0].decode())
        except (ET.ParseError, IndexError, StopIteration):
            return False
        self.tree_id = tree_id
        self.root_uid = self.rows[0].uid if self.rows else None
        self.last_tick, self.pending, self.tick_times = {}, [], []
        return self.client.request('r', 'start') is not None

    def poll(self):
        if not self.connected:
            if time.monotonic() - self.last_try < 1.0:
                return
            self.last_try = time.monotonic()
            self.connected = self.connect_tree()
            self.update()
            return
        r = self.client.request('t')
        if r is None:
            self.connected = False
            self.update()
            return
        tree_id, body = r
        if tree_id != self.tree_id:          # 다른 프로세스(다음 런) — 트리를 다시 받는다
            self.connected = False
            return
        data = body[0] if body else b''
        for off in range(0, len(data) - len(data) % 9, 9):
            ts = int.from_bytes(data[off:off + 6], 'little')
            uid, st = struct.unpack_from('<HB', data, off + 6)
            self.pending.append((ts, uid, st))
        self.cut_tick()
        self.update()

    def cut_tick(self):
        """pending 에서 마지막 '루트 RUNNING → 루트 완료' 구간을 찾아 last_tick 으로 만든다."""
        p, root = self.pending, self.root_uid
        end = None
        for i in range(len(p) - 1, -1, -1):
            if p[i][1] == root and p[i][2] in (SUCCESS, FAILURE):
                end = i
                break
        if end is None:
            self.pending = p[-2000:]
            return
        start = None
        for i in range(end - 1, -1, -1):
            if p[i][1] == root and p[i][2] == RUNNING:
                start = i
                break
        if start is None:
            self.pending = p[end + 1:]
            return
        tick = {}
        for ts, uid, st in p[start:end + 1]:
            if st != IDLE:
                tick[uid] = st
        self.last_tick = tick
        for ts, uid, st in p[:end + 1]:
            if uid == root and st in (SUCCESS, FAILURE):
                self.tick_times.append(ts)
        self.tick_times = self.tick_times[-40:]
        self.pending = p[end + 1:]

    def poll_log(self):
        self.info = self.log.read()

    # ---- 그리기
    def paintEvent(self, _):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        qp.fillRect(0, 0, w, h, QtGui.QColor(C['bg']))
        y = self.draw_header(qp, w)
        info_h = 180
        y = self.draw_tree(qp, w, y + 6, h - info_h - 8)
        self.draw_info(qp, w, h - info_h, info_h)
        qp.end()

    def draw_header(self, qp, w):
        qp.setPen(QtGui.QColor(C['ink']))
        qp.setFont(self.font_title)
        qp.drawText(12, 24, 'SCV BehaviorTree')
        qp.setFont(self.font_small)
        if self.connected:
            hz = ''
            if len(self.tick_times) >= 2:
                span = (self.tick_times[-1] - self.tick_times[0]) / 1e6
                if span > 0:
                    hz = f'  ·  {(len(self.tick_times) - 1) / span:.0f} Hz'
            qp.setPen(QtGui.QColor(C[SUCCESS]))
            qp.drawText(12, 42, f'● 연결 :{self.port}{hz}  ·  마지막 완료 틱의 경로')
        else:
            qp.setPen(QtGui.QColor(C['muted']))
            qp.drawText(12, 42, f'○ BT 대기 중 (:{self.port}) — simple 런이거나 기동 전')
        # 범례
        x = w - 12
        qp.setFont(self.font_small)
        for st in (FAILURE, SUCCESS):
            name = STATUS_NAME[st]
            tw = qp.fontMetrics().horizontalAdvance(name)
            x -= tw
            qp.setPen(QtGui.QColor(C['muted']))
            qp.drawText(x, 24, name)
            x -= 14
            qp.fillRect(x, 15, 10, 10, QtGui.QColor(C[st]))
            x -= 10
        return 50

    def draw_tree(self, qp, w, y0, bottom):
        indent, rh = 16, self.ROW_H
        line_pen = QtGui.QPen(QtGui.QColor(C['line']), 1)
        for i, r in enumerate(self.rows):
            y = y0 + i * rh
            if y + rh > bottom:
                qp.setPen(QtGui.QColor(C['muted']))
                qp.drawText(12, bottom, f'… {len(self.rows) - i} 노드 생략')
                break
            x = 12 + r.depth * indent
            # 연결선
            qp.setPen(line_pen)
            for d, anc_last in enumerate(r.ancestors_last[1:], start=0):
                if not anc_last:
                    lx = 12 + d * indent + 6
                    qp.drawLine(lx, y, lx, y + rh)
            if r.depth > 0:
                lx = x - indent + 6
                qp.drawLine(lx, y, lx, y + (rh // 2 if r.last_child else rh))
                qp.drawLine(lx, y + rh // 2, x - 2, y + rh // 2)
            st = self.last_tick.get(r.uid)
            box_w = w - x - 12
            box = QtCore.QRectF(x, y + 2, box_w, rh - 4)
            if st is None:
                qp.setPen(QtCore.Qt.NoPen)
                qp.setBrush(QtGui.QColor(C['idle_box']))
                qp.drawRoundedRect(box, 3, 3)
                ink = QtGui.QColor(C['muted'])
            else:
                col = QtGui.QColor(C.get(st, C['idle_box']))
                qp.setPen(QtCore.Qt.NoPen)
                qp.setBrush(col.darker(260))
                qp.drawRoundedRect(box, 3, 3)
                qp.setBrush(col)
                qp.drawRoundedRect(QtCore.QRectF(x, y + 2, 5, rh - 4), 2, 2)
                ink = QtGui.QColor(C['ink'])
            qp.setPen(ink)
            qp.setFont(self.font_main)
            qp.drawText(QtCore.QRectF(x + 10, y, box_w - 14, rh), QtCore.Qt.AlignVCenter, r.label)
            if r.detail:
                lw = qp.fontMetrics().horizontalAdvance(r.label)
                qp.setFont(self.font_small)
                qp.setPen(QtGui.QColor(C['muted']))
                qp.drawText(QtCore.QRectF(x + 18 + lw, y, box_w - lw - 24, rh),
                            QtCore.Qt.AlignVCenter, r.detail)
        if not self.rows:
            qp.setPen(QtGui.QColor(C['muted']))
            qp.setFont(self.font_main)
            qp.drawText(12, y0 + 20, '트리 미수신')
        return y0

    def draw_info(self, qp, w, y0, hgt):
        qp.fillRect(0, y0, w, hgt, QtGui.QColor(C['panel']))
        qp.setPen(QtGui.QColor(C['line']))
        qp.drawLine(0, y0, w, y0)
        inf = self.info or {}
        rows = [
            ('런', inf.get('run', '—')),
            ('제어 모드', ' '.join(inf['mode']) if inf.get('mode') else '—'),
            ('목표 노드', f"{inf['target'][0]}  ({inf['target'][1]}/{inf['target'][2]})" if inf.get('target')
             else (f"합류 {inf['join'][1]}" if inf.get('join') else '—')),
            ('행동', (inf['behavior'][0][:46] if inf.get('behavior') else '—')
             + ('   ✓ 완주' if inf.get('done') else '')),
            ('차단', inf['blocked'][0] if inf.get('blocked') else 'NORMAL'),
        ]
        y = y0 + 18
        for k, v in rows:
            qp.setFont(self.font_small)
            qp.setPen(QtGui.QColor(C['muted']))
            qp.drawText(12, y, k)
            qp.setFont(self.font_main)
            qp.setPen(QtGui.QColor(C['ink']))
            qp.drawText(80, y, str(v))
            y += 18
        qp.setFont(self.font_mono)
        qp.setPen(QtGui.QColor(C['muted']))
        for ts, msg in inf.get('events', [])[-4:]:
            y += 1
            qp.drawText(QtCore.QRectF(12, y - 10, w - 24, 14), QtCore.Qt.AlignVCenter,
                        f'{time.strftime("%H:%M:%S", time.localtime(ts))}  {msg[:80]}')
            y += 13


# ---------------------------------------------------------------- 지도 패널
MAP_C = {
    'road': '#3b4148', 'sidewalk': '#c9ccd1', 'obstacle': '#8a5a35', 'route': '#2f81f7', 'node': '#ffffff',
    'passed': '#2f9e57', 'target': '#e5a23a', 'trail': '#d4473f', 'vehicle_auto': '#2f81f7',
    'vehicle_rc': '#e5a23a', 'vehicle_idle': '#8b949e', 'ink': '#e6e8ea', 'muted': '#8b949e', 'panel': '#1c2024',
    'pit': '#111111',
}
POSE_RE = re.compile(r'pose \{ name: "scv" id: \d+ position \{ x: (\S+) y: (\S+) z: (\S+) \} '
                     r'orientation \{ x: (\S+) y: (\S+) z: (\S+) w: (\S+) \}')
TIME_RE = re.compile(r'^time \{ sec: (\d+) nsec: (\d+) \}')


def load_world(path):
    """월드 SDF → (보도 rect 목록, 장애물 rect 목록, 월드 이름). rect = (x, y, sx, sy, yaw)."""
    root = ET.parse(path).getroot()
    world = root.find('world')
    sidewalks, obstacles, pits = [], [], []
    for m in world.findall('model'):
        name = m.get('name', '')
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        size = m.find('.//collision/geometry/box/size')
        if size is None:
            continue
        sx, sy, _ = (float(v) for v in size.text.split())
        rect = (pose[0], pose[1], sx, sy, pose[5] if len(pose) > 5 else 0.0)
        if name.startswith('sw'):
            sidewalks.append(rect)
        elif name.startswith('obs') or name.startswith('hz'):
            obstacles.append(rect)
        elif name.startswith('pit'):
            pits.append(rect)
    return sidewalks, obstacles, pits, world.get('name', 'default')


def load_route(path):
    g = __import__('json').load(open(path))
    pos = {n['ID']: (n['UtmInfo']['Easting'], n['UtmInfo']['Northing']) for n in g['Node']}
    nxt = {l['FromNodeID']: l['ToNodeID'] for l in g['Link']}
    starts = set(nxt) - set(nxt.values())
    order = [next(iter(starts))] if starts else [g['Node'][0]['ID']]
    while order[-1] in nxt and nxt[order[-1]] not in order:
        order.append(nxt[order[-1]])
    return [(nid, *pos[nid]) for nid in order]


class MapView(QtWidgets.QWidget):
    def __init__(self, design_path, bt_view):
        super().__init__()
        d = __import__('json').load(open(os.path.expanduser(design_path)))
        self.sidewalks, self.obstacles, self.pits, self.world = load_world(d['world'])
        self.scn = __import__('json').load(open(d['scenario'])) if d.get('scenario') else None
        self.scn_state = {h['id']: dict(status='대기', clear=None) for h in (self.scn or {}).get('hazards', [])}
        self.scn_failed = False
        self.route = load_route(d['graph'])
        sp = d.get('spawn') or {}
        self.spawn = (sp['x'], sp['y']) if 'x' in sp else None
        self.bt = bt_view
        self.pose = None             # (x, y, yaw)
        self.sim_t = None
        self.trail = []              # (t, x, y)
        self.speed = 0.0
        self.buf = b''
        self.setWindowTitle('SCV Route Map')
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        pts = [(x, y) for _, x, y in self.route] + ([self.spawn] if self.spawn else [])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = 3.5
        self.bounds = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
        if self.scn:      # 보도 폭 전체가 보이도록
            sw = self.scn.get('sidewalk', {})
            ext = [self.course_to_world(u, v) for u in (0.0,) for v in (sw.get('y_left', 0) + 1.5, sw.get('y_right', 0) - 1.5)]
            ys2 = ys + [e[1] for e in ext]
            self.bounds = (self.bounds[0], self.bounds[1], min(ys2) - 1.0, max(ys2) + 1.0)
        self.proc = None
        self.start_stream()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)

    # ---- 위치 스트림 (Gazebo transport)
    def start_stream(self):
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyRead.connect(self.on_data)
        self.proc.start('gz', ['topic', '-e', f'/gazebo/{self.world}/pose/info', '-u'])
        self.stream_started = time.monotonic()

    def on_data(self):
        self.buf += bytes(self.proc.readAll())
        if b'\n' not in self.buf:
            return
        lines = self.buf.split(b'\n')
        self.buf = lines[-1]
        for raw in reversed(lines[:-1]):
            line = raw.decode('utf-8', 'replace')
            m = POSE_RE.search(line)
            if not m:
                continue
            x, y, z, qx, qy, qz, qw = map(float, m.groups())
            yaw = __import__('math').atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            self.update_scenario(x, y, z, yaw)
            tm = TIME_RE.search(line)
            t = int(tm.group(1)) + int(tm.group(2)) * 1e-9 if tm else None
            self.pose, self.sim_t = (x, y, yaw), t
            if t is not None:
                if not self.trail or t - self.trail[-1][0] >= 0.25:
                    if self.trail and t > self.trail[-1][0]:
                        lt, lx, ly = self.trail[-1]
                        self.speed = __import__('math').hypot(x - lx, y - ly) / (t - lt)
                    if self.trail and t < self.trail[-1][0]:      # 시뮬 재시작
                        self.trail = []
                    self.trail.append((t, x, y))
                    self.trail = self.trail[-2400:]
            break

    def tick(self):
        if self.proc.state() == QtCore.QProcess.NotRunning and time.monotonic() - self.stream_started > 2.0:
            self.start_stream()
        self.update()

    # ---- 연속 위험 시나리오 실시간 추정 (판정 정본은 gz_judge --scenario)
    def update_scenario(self, x, y, z, yaw):
        if not self.scn:
            return
        math = __import__('math')
        x, y, yaw = self.to_course(x, y, yaw)
        fp = self.scn.get('footprint', {})
        L, W = fp.get('length', 0.98), fp.get('width', 0.74)
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [(x + u * c - v * s, y + u * s + v * c)
               for u in (-L / 2, 0, L / 2) for v in (-W / 2, 0, W / 2)]
        for h in sorted(self.scn['hazards'], key=lambda h: h['start_x']):
            st = self.scn_state[h['id']]
            if st['status'] in ('통과', '실패: 추락', '실패: 충돌', '미시험'):
                continue
            if self.scn_failed:
                st['status'] = '미시험'
                continue
            if x < h['start_x']:
                continue
            st['status'] = '진행 중'
            items = h.get('items') or [dict(type=h['type'], rect=h['rect'])]
            for it in items:
                r = it['rect']
                d = min(math.hypot(max(r[0] - px, 0, px - r[1]), max(r[2] - py, 0, py - r[3])) for px, py in pts)
                pen = max((min(px - r[0], r[1] - px, py - r[2], r[3] - py)
                           if r[0] <= px <= r[1] and r[2] <= py <= r[3] else 0.0) for px, py in pts)
                st['clear'] = d if st['clear'] is None else min(st['clear'], d)
                if it['type'] == 'pit' and (pen > self.scn.get('pit_fail_penetration', 0.15)
                                            or (z < -0.10 and d < 0.8)):
                    st['status'], self.scn_failed = '실패: 추락', True
                elif it['type'] == 'obstacle' and d <= 0.0:
                    st['status'], self.scn_failed = '실패: 충돌', True
                if self.scn_failed:
                    break
            if st['status'] == '진행 중' and x >= h['pass_x']:
                st['status'] = '통과'

    def _frame(self):
        fr = (self.scn or {}).get('frame', {})
        return fr.get('x0', 0.0), fr.get('y0', 0.0), __import__('math').radians(fr.get('heading_deg', 0.0))

    def to_course(self, x, y, yaw):
        math = __import__('math')
        x0, y0, h = self._frame()
        dx, dy = x - x0, y - y0
        return dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h), yaw - h

    def course_to_world(self, u, v):
        math = __import__('math')
        x0, y0, h = self._frame()
        return x0 + u * math.cos(h) - v * math.sin(h), y0 + u * math.sin(h) + v * math.cos(h)

    def draw_scenario(self, qp):
        if not self.scn:
            return
        col = {'대기': MAP_C['muted'], '진행 중': MAP_C['target'], '통과': MAP_C['passed'],
               '실패: 추락': '#d4473f', '실패: 충돌': '#d4473f', '미시험': MAP_C['muted']}
        hz = sorted(self.scn['hazards'], key=lambda h: h['start_x'])
        w, rh = 330, 22
        x0, y0 = self.width() - w - 40, 10
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor(28, 32, 36, 225))
        qp.drawRoundedRect(QtCore.QRectF(x0, y0, w, 26 + rh * len(hz)), 5, 5)
        qp.setFont(QtGui.QFont('DejaVu Sans', 8))
        qp.setPen(QtGui.QColor(MAP_C['muted']))
        qp.drawText(int(x0 + 10), int(y0 + 16), '시나리오 (실시간 추정 · 정본은 판정기)')
        for i, h in enumerate(hz):
            st = self.scn_state[h['id']]
            y = y0 + 26 + i * rh
            qp.setBrush(QtGui.QColor(col.get(st['status'], MAP_C['muted'])))
            qp.setPen(QtCore.Qt.NoPen)
            qp.drawEllipse(QtCore.QPointF(x0 + 16, y + 9), 5, 5)
            qp.setFont(QtGui.QFont('DejaVu Sans', 9, QtGui.QFont.Bold))
            qp.setPen(QtGui.QColor(MAP_C['ink']))
            qp.drawText(int(x0 + 28), int(y + 13), f"{h['id']} {h.get('name', h['id'])}")
            qp.setFont(QtGui.QFont('DejaVu Sans', 9))
            qp.setPen(QtGui.QColor(col.get(st['status'], MAP_C['muted'])))
            extra = f"  이격 {st['clear']:.2f} m" if st['clear'] is not None else ''
            qp.drawText(int(x0 + 175), int(y + 13), st['status'] + extra)
        # 구간 경계(진입·통과선)
        sw = self.scn.get('sidewalk', {})
        vl, vr = sw.get('y_left', 3.0), sw.get('y_right', -3.0)
        for h in hz:
            for uu, dash in ((h['start_x'], QtCore.Qt.DotLine), (h['pass_x'], QtCore.Qt.DashLine)):
                qp.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 110), 1, dash))
                qp.drawLine(self.to_px(*self.course_to_world(uu, vr - 1.0)),
                            self.to_px(*self.course_to_world(uu, vl + 1.0)))
            rs = [it['rect'] for it in (h.get('items') or [h])]
            cu = (min(r[0] for r in rs) + max(r[1] for r in rs)) / 2
            p = self.to_px(*self.course_to_world(cu, vl + 0.6))
            self.label(qp, p + QtCore.QPointF(-8, 0), h['id'], '#0d1117', bold=True, halo=True)

    # ---- 좌표
    def setup_xf(self):
        x0, x1, y0, y1 = self.bounds
        w, h = self.width(), self.height()
        self.s = min(w / (x1 - x0), h / (y1 - y0))
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2

    def to_px(self, x, y):
        return QtCore.QPointF(self.width() / 2 + (x - self.cx) * self.s, self.height() / 2 - (y - self.cy) * self.s)

    def draw_rect(self, qp, r, color):
        x, y, sx, sy, yaw = r
        qp.save()
        qp.translate(self.to_px(x, y))
        qp.rotate(-__import__('math').degrees(yaw))
        qp.fillRect(QtCore.QRectF(-sx * self.s / 2, -sy * self.s / 2, sx * self.s, sy * self.s), color)
        qp.restore()

    def target_index(self):
        inf = self.bt.info or {}
        ids = [nid for nid, _, _ in self.route]
        if inf.get('target') and inf['target'][0] in ids:
            return ids.index(inf['target'][0])
        if inf.get('join') and inf['join'][1] in ids:
            return ids.index(inf['join'][1])
        return None

    # ---- 그리기
    def paintEvent(self, _):
        math = __import__('math')
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setup_xf()
        qp.fillRect(self.rect(), QtGui.QColor(MAP_C['road']))
        qp.setRenderHint(QtGui.QPainter.Antialiasing, False)   # 0.5 m 띠 사이 이음선 방지
        for r in self.sidewalks:
            self.draw_rect(qp, r, QtGui.QColor(MAP_C['sidewalk']))
        for r in self.obstacles:
            self.draw_rect(qp, r, QtGui.QColor(MAP_C['obstacle']))
        for r in self.pits:
            self.draw_rect(qp, r, QtGui.QColor(MAP_C['pit']))
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        # 1 m 격자
        qp.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 28), 1))
        x0, x1, y0, y1 = self.bounds
        span_x = self.width() / self.s / 2
        for gx in range(int(self.cx - span_x) - 1, int(self.cx + span_x) + 2):
            qp.drawLine(self.to_px(gx, y0 - 50), self.to_px(gx, y1 + 50))
        for gy in range(int(y0) - 1, int(y1) + 2):
            qp.drawLine(self.to_px(self.cx - span_x - 1, gy), self.to_px(self.cx + span_x + 1, gy))

        tgt = self.target_index()
        done = bool((self.bt.info or {}).get('done'))
        # 경로
        pts = [self.to_px(x, y) for _, x, y in self.route]
        qp.setPen(QtGui.QPen(QtGui.QColor(MAP_C['route']), 3, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        for a, b in zip(pts, pts[1:]):
            qp.drawLine(a, b)
        # 궤적
        if len(self.trail) > 1:
            path = QtGui.QPainterPath(self.to_px(self.trail[0][1], self.trail[0][2]))
            for _, x, y in self.trail[1:]:
                path.lineTo(self.to_px(x, y))
            qp.setPen(QtGui.QPen(QtGui.QColor(MAP_C['trail']), 2))
            qp.setBrush(QtCore.Qt.NoBrush)
            qp.drawPath(path)
        # 출발점
        if self.spawn:
            p = self.to_px(*self.spawn)
            qp.setPen(QtGui.QPen(QtGui.QColor(MAP_C['muted']), 2))
            qp.drawLine(p + QtCore.QPointF(-6, -6), p + QtCore.QPointF(6, 6))
            qp.drawLine(p + QtCore.QPointF(-6, 6), p + QtCore.QPointF(6, -6))
            self.label(qp, p + QtCore.QPointF(9, 14), 'spawn', MAP_C['muted'])
        # 노드
        for i, (nid, x, y) in enumerate(self.route):
            p = pts[i]
            if tgt is not None and (i < tgt or done):
                fill = MAP_C['passed']
            else:
                fill = MAP_C['node']
            qp.setPen(QtGui.QPen(QtGui.QColor('#0d1117'), 1.5))
            qp.setBrush(QtGui.QColor(fill))
            qp.drawEllipse(p, 7, 7)
            if tgt is not None and i == tgt and not done:
                qp.setPen(QtGui.QPen(QtGui.QColor(MAP_C['target']), 3))
                qp.setBrush(QtCore.Qt.NoBrush)
                qp.drawEllipse(p, 13, 13)
            fm = QtGui.QFontMetrics(QtGui.QFont('DejaVu Sans', 9, QtGui.QFont.Bold))
            self.label(qp, p + QtCore.QPointF(-fm.horizontalAdvance(nid) / 2, -11), nid, '#0d1117', bold=True, halo=True)
        # 위험 지대 재배치 경유점(주황 마름모) · 건너뛴 노드(X)
        inf = self.bt.info or {}
        ids = {nid: i for i, (nid, _, _) in enumerate(self.route)}
        for nid, (sx, sy) in (inf.get('shifted') or {}).items():
            if nid not in ids:
                continue
            q = self.to_px(sx, sy)
            qp.setPen(QtGui.QPen(QtGui.QColor(MAP_C['target']), 1.5, QtCore.Qt.DashLine))
            qp.drawLine(pts[ids[nid]], q)
            qp.setPen(QtGui.QPen(QtGui.QColor('#0d1117'), 1.2))
            qp.setBrush(QtGui.QColor(MAP_C['target']))
            qp.drawPolygon(QtGui.QPolygonF([q + QtCore.QPointF(0, -7), q + QtCore.QPointF(7, 0),
                                            q + QtCore.QPointF(0, 7), q + QtCore.QPointF(-7, 0)]))
        for nid in inf.get('skipped') or []:
            if nid in ids:
                q = pts[ids[nid]]
                qp.setPen(QtGui.QPen(QtGui.QColor('#d4473f'), 2.5))
                qp.drawLine(q + QtCore.QPointF(-6, -6), q + QtCore.QPointF(6, 6))
                qp.drawLine(q + QtCore.QPointF(-6, 6), q + QtCore.QPointF(6, -6))
        # 목표 연결선 + 차량
        if self.pose:
            x, y, yaw = self.pose
            vp = self.to_px(x, y)
            if tgt is not None and not done:
                pen = QtGui.QPen(QtGui.QColor(MAP_C['target']), 2, QtCore.Qt.DashLine)
                qp.setPen(pen)
                sh = (inf.get('shifted') or {}).get(self.route[tgt][0])
                qp.drawLine(vp, self.to_px(*sh) if sh else pts[tgt])
            mode = ((self.bt.info or {}).get('mode') or ('0',))[0]
            col = MAP_C['vehicle_auto'] if mode == '1' else MAP_C['vehicle_rc'] if mode == '3' else MAP_C['vehicle_idle']
            L, W = 0.98, 0.74          # Hunter 2.0 외곽(대략)
            qp.save()
            qp.translate(vp)
            qp.rotate(-math.degrees(yaw))
            body = QtCore.QRectF(-L / 2 * self.s, -W / 2 * self.s, L * self.s, W * self.s)
            qp.setPen(QtGui.QPen(QtGui.QColor('#0d1117'), 1.5))
            qp.setBrush(QtGui.QColor(col))
            qp.drawRoundedRect(body, 3, 3)
            tri = QtGui.QPolygonF([QtCore.QPointF(L / 2 * self.s + 9, 0), QtCore.QPointF(L / 2 * self.s - 2, -6),
                                   QtCore.QPointF(L / 2 * self.s - 2, 6)])
            qp.setBrush(QtGui.QColor('#ffffff'))
            qp.drawPolygon(tri)
            qp.restore()
        self.draw_scenario(qp)
        self.draw_overlay(qp, tgt, done)
        qp.end()

    def label(self, qp, p, text, color, bold=False, halo=False):
        f = QtGui.QFont('DejaVu Sans', 9, QtGui.QFont.Bold if bold else QtGui.QFont.Normal)
        qp.setFont(f)
        if halo:
            path = QtGui.QPainterPath()
            path.addText(p, f, text)
            qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 3))
            qp.setBrush(QtCore.Qt.NoBrush)
            qp.drawPath(path)
        qp.setPen(QtGui.QColor(color))
        qp.drawText(p, text)

    def draw_overlay(self, qp, tgt, done):
        math = __import__('math')
        inf = self.bt.info or {}
        mode = inf.get('mode')
        lines = [('시뮬 시각', f'{self.sim_t:.1f} s' if self.sim_t is not None else '—'),
                 ('제어 모드', ' '.join(mode) if mode else '—')]
        if self.pose:
            x, y, yaw = self.pose
            lines.append(('위치', f'({x:+.2f}, {y:+.2f})  {math.degrees(yaw):+.0f}°'))
            lines.append(('속도', f'{self.speed:.2f} m/s'))
            # 경로 횡오차(최근접 링크까지 거리)
            best = None
            for (_, ax, ay), (_, bx, by) in zip(self.route, self.route[1:]):
                dx, dy = bx - ax, by - ay
                u = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy or 1)))
                d = math.hypot(x - (ax + u * dx), y - (ay + u * dy))
                best = d if best is None else min(best, d)
            lines.append(('경로 이탈', f'{best:.2f} m' if best is not None else '—'))
            if tgt is not None and not done:
                nid, tx, ty = self.route[tgt]
                rem = math.hypot(tx - x, ty - y) + sum(
                    math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(self.route[tgt:], self.route[tgt + 1:]))
                lines.append(('목표', f'{nid}  {math.hypot(tx - x, ty - y):.1f} m  ·  잔여 {rem:.1f} m'))
            elif done:
                lines.append(('목표', '완주'))
        else:
            lines.append(('위치', 'Gazebo pose 대기'))
        w = 268
        h = 14 + 17 * len(lines)
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor(28, 32, 36, 225))
        qp.drawRoundedRect(QtCore.QRectF(10, 10, w, h), 5, 5)
        y = 28
        for k, v in lines:
            qp.setFont(QtGui.QFont('DejaVu Sans', 8))
            qp.setPen(QtGui.QColor(MAP_C['muted']))
            qp.drawText(20, y, k)
            qp.setFont(QtGui.QFont('DejaVu Sans', 9))
            qp.setPen(QtGui.QColor(MAP_C['ink']))
            qp.drawText(90, y, v)
            y += 17
        # 범례 + 축척
        leg = [(MAP_C['route'], '경로'), (MAP_C['trail'], '궤적'), (MAP_C['target'], '목표 노드'),
               (MAP_C['passed'], '지나온 노드'), (MAP_C['sidewalk'], '보도'), (MAP_C['obstacle'], '장애물'),
               (MAP_C['pit'], '구덩이'), (MAP_C['target'], '◆ 재배치 경유점')]
        qp.setFont(QtGui.QFont('DejaVu Sans', 8))
        lx = self.width() - 10
        ly = self.height() - 12
        for col, name in reversed(leg):
            tw = qp.fontMetrics().horizontalAdvance(name)
            lx -= tw
            qp.setPen(QtGui.QColor('#0d1117'))
            qp.drawText(lx, ly, name)
            lx -= 14
            qp.fillRect(QtCore.QRectF(lx, ly - 9, 10, 10), QtGui.QColor(col))
            lx -= 12
        p0 = QtCore.QPointF(14, self.height() - 16)
        qp.setPen(QtGui.QPen(QtGui.QColor('#0d1117'), 2))
        qp.drawLine(p0, p0 + QtCore.QPointF(5 * self.s, 0))
        qp.drawText(p0 + QtCore.QPointF(5 * self.s + 6, 4), '5 m')
        n = QtCore.QPointF(self.width() - 24, 26)
        qp.drawLine(n + QtCore.QPointF(0, 10), n + QtCore.QPointF(0, -10))
        qp.drawText(n + QtCore.QPointF(-4, -14), 'N')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=1667)
    ap.add_argument('--log-glob', default='~/scv_sim/bt_ab/**/bt.log')
    ap.add_argument('--geometry', default='540x1000+1380+29')
    ap.add_argument('--design', default='', help='챔버 설계 JSON — 주면 지도 창을 띄운다')
    ap.add_argument('--map-geometry', default='1380x500+0+529')
    args = ap.parse_args()
    app = QtWidgets.QApplication([])

    def place(win, geo):
        m = re.match(r'(\d+)x(\d+)\+(\d+)\+(\d+)', geo)
        if m:
            gw, gh, gx, gy = map(int, m.groups())
            win.setGeometry(gx, gy, gw, gh)
        win.show()

    v = BtView(args)
    place(v, args.geometry)
    if args.design:
        mv = MapView(args.design, v)
        place(mv, args.map_geometry)
    app.exec_()


if __name__ == '__main__':
    main()
