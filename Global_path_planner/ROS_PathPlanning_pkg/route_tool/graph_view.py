"""그래프 지도 캔버스 — 노드를 눌러 경로를 지시한다.

map_editor 의 BEV 캔버스에서 필요한 것만 가져왔다. 저쪽은 편집 도구라 폴리곤·점군
래스터·타일 진척까지 들고 있는데, 여기서는 노드를 보고 고르는 것이 전부다.

좌표: 그래프는 절대 UTM 으로 오고, 화면에는 datum 상대(map 프레임)로 그린다.
"""
import math

from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (QPainter, QPen, QBrush, QColor, QFont, QTransform)
from PyQt6.QtWidgets import QWidget

from . import basemap as BM
from . import theme as T


class GraphView(QWidget):
    """노드/링크를 그리고 클릭을 노드 ID 로 돌려준다."""

    nodePicked = pyqtSignal(str)
    statusMessage = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(480, 360)

        self.nodes = {}          # id -> (x, y, src)   map 프레임 미터
        self.links = []          # (from_id, to_id, bidirectional)
        self.queue = []          # 선택된 노드 ID (순서 유지)
        self.robot = None        # (x, y, yaw)
        self.path = []           # [(x, y)] 로봇이 되돌려준 경로
        self.preview = []        # [(x, y)] 발행 전 예상 경로
        self.bad_legs = []       # [(id_a, id_b)] 도달 불가 구간

        self.cx = self.cy = 0.0
        self.scale = 2.0
        self._drag = None
        self._fitted = False

        self.basemap = "off"
        self.basemap_opacity = 0.75
        self._bm = None
        self._proj = None        # (datum_e, datum_n, zone) -> 위경도 변환용

    # ------------------------------------------------------------- 데이터
    def set_graph(self, nodes, links):
        self.nodes = nodes
        self.links = links
        if nodes and not self._fitted:
            self.fit()
            self._fitted = True
        self.update()

    def set_datum(self, easting, northing, zone, northern):
        self._proj = (easting, northing, zone, northern)
        self.update()

    def set_robot(self, pose):
        self.robot = pose
        self.update()

    def set_path(self, pts):
        self.path = pts
        self.update()

    def set_preview(self, pts, bad_legs=None):
        self.preview = pts
        self.bad_legs = bad_legs or []
        self.update()

    def set_queue(self, ids):
        self.queue = list(ids)
        self.update()

    def fit(self):
        if not self.nodes:
            return
        xs = [p[0] for p in self.nodes.values()]
        ys = [p[1] for p in self.nodes.values()]
        w = max(xs) - min(xs) + 40.0
        h = max(ys) - min(ys) + 40.0
        self.cx = (min(xs) + max(xs)) / 2.0
        self.cy = (min(ys) + max(ys)) / 2.0
        self.scale = max(0.05, min(self.width() / w, self.height() / h))
        self.update()

    # ------------------------------------------------------------- 좌표
    def to_screen(self, x, y):
        return QPointF((x - self.cx) * self.scale + self.width() / 2.0,
                       -(y - self.cy) * self.scale + self.height() / 2.0)

    def to_map(self, px, py):
        return ((px - self.width() / 2.0) / self.scale + self.cx,
                -(py - self.height() / 2.0) / self.scale + self.cy)

    def hit(self, px, py, tol=12.0):
        best, bd = None, tol
        for nid, (x, y, _s) in self.nodes.items():
            s = self.to_screen(x, y)
            d = math.hypot(s.x() - px, s.y() - py)
            if d < bd:
                best, bd = nid, d
        return best

    # ------------------------------------------------------------- 입력
    def mousePressEvent(self, ev):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        px, py = ev.position().x(), ev.position().y()
        if ev.button() == Qt.MouseButton.LeftButton:
            nid = self.hit(px, py)
            if nid:
                self.nodePicked.emit(nid)
                return
        self._drag = (px, py)

    def mouseMoveEvent(self, ev):
        px, py = ev.position().x(), ev.position().y()
        if self._drag:
            ax, ay = self._drag
            self.cx -= (px - ax) / self.scale
            self.cy += (py - ay) / self.scale
            self._drag = (px, py)
            self.update()
            return
        nid = self.hit(px, py)
        self.setCursor(Qt.CursorShape.PointingHandCursor if nid
                       else Qt.CursorShape.ArrowCursor)
        if nid:
            self.statusMessage.emit(nid)

    def mouseReleaseEvent(self, _ev):
        self._drag = None

    def wheelEvent(self, ev):
        mx, my = self.to_map(ev.position().x(), ev.position().y())
        f = 1.18 if ev.angleDelta().y() > 0 else 1 / 1.18
        self.scale = max(0.02, min(200.0, self.scale * f))
        nx, ny = self.to_map(ev.position().x(), ev.position().y())
        self.cx += mx - nx
        self.cy += my - ny
        self.update()

    # ------------------------------------------------------------- 배경
    def set_basemap(self, provider):
        if provider not in BM.PROVIDERS:
            return
        self.basemap = provider
        if provider != "off" and self._bm is None:
            from PyQt6.QtCore import QTimer
            self._bm = BM.BasemapLoader(self)
            self._bm_timer = QTimer(self)
            self._bm_timer.setSingleShot(True)
            self._bm_timer.setInterval(90)
            self._bm_timer.timeout.connect(self.update)
            self._bm.tileReady.connect(self._bm_timer.start)
        self.update()

    def _latlon(self, x, y):
        import utm as U
        e, n, zone, northern = self._proj
        return U.to_latlon(e + x, n + y, int(zone), northern=bool(northern))

    def _local(self, lat, lon):
        import utm as U
        e, n, zone, _nh = self._proj
        ee, nn, _z, _l = U.from_latlon(lat, lon, force_zone_number=int(zone))
        return ee - e, nn - n

    def _paint_basemap(self, p):
        if self.basemap == "off" or not self._proj or not self.nodes:
            return
        try:
            x0, y0 = self.to_map(0, self.height())
            x1, y1 = self.to_map(self.width(), 0)
            corners = [self._latlon(a, b) for a, b in
                       ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        except Exception:
            return
        lats = [c[0] for c in corners]
        lons = [c[1] for c in corners]
        z = BM.pick_zoom(self.scale, sum(lats) / 4.0, self.basemap)
        tx0, ty0 = BM.deg2tile(max(lats), min(lons), z)
        tx1, ty1 = BM.deg2tile(min(lats), max(lons), z)
        ix0, iy0 = int(math.floor(tx0)), int(math.floor(ty0))
        ix1, iy1 = int(math.floor(tx1)), int(math.floor(ty1))
        n = 2 ** z
        if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > 400:
            return
        p.save()
        p.setOpacity(self.basemap_opacity)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                if not (0 <= ix < n and 0 <= iy < n):
                    continue
                img = self._bm.get(self.basemap, z, ix, iy)
                if img is None or img.isNull():
                    continue
                nw = self._tile_pt(ix, iy, z)
                ne = self._tile_pt(ix + 1, iy, z)
                sw = self._tile_pt(ix, iy + 1, z)
                t = QTransform()
                t.setMatrix(ne.x() - nw.x(), ne.y() - nw.y(), 0,
                            sw.x() - nw.x(), sw.y() - nw.y(), 0,
                            nw.x(), nw.y(), 1)
                p.setWorldTransform(t)
                p.drawImage(QRectF(0, 0, 1, 1), img,
                            QRectF(0, 0, img.width(), img.height()))
        p.restore()

    def _tile_pt(self, xt, yt, z):
        lat, lon = BM.tile2deg(xt, yt, z)
        mx, my = self._local(lat, lon)
        return self.to_screen(mx, my)

    # ------------------------------------------------------------- 그리기
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), T.BG)
        if not self.nodes:
            p.setPen(QPen(T.TEXT_FAINT))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "그래프를 기다리는 중 … (/map_provider_node/graph)")
            return
        self._paint_basemap(p)
        self._paint_links(p)
        self._paint_preview(p)
        self._paint_path(p)
        self._paint_nodes(p)
        self._paint_robot(p)
        self._paint_hud(p)

    def _paint_links(self, p):
        pts = []
        for a, b, _bi in self.links:
            na, nb = self.nodes.get(a), self.nodes.get(b)
            if not na or not nb:
                continue
            pts.append((self.to_screen(na[0], na[1]), self.to_screen(nb[0], nb[1])))
        if not pts:
            return
        # 밝은 배경에서도 선이 살도록 어두운 테두리를 먼저
        p.setPen(QPen(T.GRAPH_HALO, 3.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for a, b in pts:
            p.drawLine(a, b)
        p.setPen(QPen(T.GRAPH_SLAM_LO, 1.4))
        for a, b in pts:
            p.drawLine(a, b)

    def _poly(self, p, pts, color, width, dashed=False):
        if len(pts) < 2:
            return
        pen = QPen(color, width, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        scr = [self.to_screen(x, y) for x, y in pts]
        for i in range(1, len(scr)):
            p.drawLine(scr[i - 1], scr[i])

    def _paint_preview(self, p):
        self._poly(p, self.preview, T.PREVIEW, 2.6, dashed=True)
        for a, b in self.bad_legs:
            na, nb = self.nodes.get(a), self.nodes.get(b)
            if na and nb:
                p.setPen(QPen(T.ERROR, 3.0, Qt.PenStyle.DashLine))
                p.drawLine(self.to_screen(na[0], na[1]), self.to_screen(nb[0], nb[1]))

    def _paint_path(self, p):
        self._poly(p, self.path, T.PATH, 3.2)

    def _paint_nodes(self, p):
        r = max(2.5, min(7.0, self.scale * 0.8))
        d = r * 2
        qpos = {nid: i + 1 for i, nid in enumerate(self.queue)}
        for nid, (x, y, src) in self.nodes.items():
            s = self.to_screen(x, y)
            if not (-30 <= s.x() <= self.width() + 30 and -30 <= s.y() <= self.height() + 30):
                continue
            in_q = nid in qpos
            col = T.QUEUE if in_q else (T.GRAPH_GPS if src == "gps" else T.GRAPH_SLAM_HI)
            hr = r + 1.3
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(T.GRAPH_HALO))
            p.drawEllipse(QRectF(s.x() - hr, s.y() - hr, hr * 2, hr * 2))
            p.setPen(QPen(col, 1.4))
            p.setBrush(QBrush(col.darker(125)))
            if src == "gps":
                p.drawRect(QRectF(s.x() - r, s.y() - r, d, d))
            else:
                p.drawEllipse(QRectF(s.x() - r, s.y() - r, d, d))
            if in_q:
                p.setPen(QPen(T.QUEUE_TEXT))
                f = QFont(); f.setPointSizeF(max(7.0, r * 1.3)); f.setBold(True)
                p.setFont(f)
                p.drawText(QRectF(s.x() - r, s.y() - r, d, d),
                           Qt.AlignmentFlag.AlignCenter, str(qpos[nid]))

    def _paint_robot(self, p):
        if not self.robot:
            return
        x, y, yaw = self.robot
        s = self.to_screen(x, y)
        p.setPen(QPen(T.GRAPH_HALO, 3.0))
        p.setBrush(QBrush(T.ROBOT))
        rr = 7.0
        p.drawEllipse(QRectF(s.x() - rr, s.y() - rr, rr * 2, rr * 2))
        p.setPen(QPen(T.ROBOT, 2.5))
        p.drawLine(s, QPointF(s.x() + 18 * math.cos(yaw), s.y() - 18 * math.sin(yaw)))

    def _paint_hud(self, p):
        p.setPen(QPen(T.TEXT_FAINT))
        f = QFont(); f.setPointSize(9); p.setFont(f)
        bar_m = 50.0 if self.scale > 1 else 200.0
        w = bar_m * self.scale
        y = self.height() - 18
        p.drawLine(QPointF(14, y), QPointF(14 + w, y))
        p.drawText(QPointF(14, y - 5), f"{bar_m:.0f} m")
