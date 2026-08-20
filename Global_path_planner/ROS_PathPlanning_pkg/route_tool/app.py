"""경로 지시 툴 — 노드를 눌러 로봇에 경로를 보낸다.

두 가지 방식이 있는데 **모드를 고르는 게 아니라 무엇을 보내느냐의 차이**다.
플래너도 같은 규칙으로 읽는다 (RouteRequest.msg 참고).

  목표 하나   : 노드 하나를 누르면 바로 발행 -> 플래너가 최단 경로로 간다
  경로 지정   : 노드를 차례로 눌러 큐에 쌓고 발행 -> 그 순서대로 간다

나중에 웹 서버가 같은 RouteRequest 를 내려보내면 로봇 쪽은 바뀌지 않는다.
이 툴은 그 메시지의 생산자 중 하나일 뿐이다.
"""
import sys
import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QListWidget,
                             QListWidgetItem, QCheckBox, QComboBox, QFrame,
                             QStatusBar, QButtonGroup, QSlider, QMessageBox)

from . import theme as T
from . import basemap as BM
from .graph_view import GraphView

MODE_GOAL = "goal"
MODE_ROUTE = "route"


def _panel(title):
    f = QFrame(); f.setObjectName("panel")
    v = QVBoxLayout(f); v.setContentsMargins(11, 9, 11, 11); v.setSpacing(7)
    if title:
        lb = QLabel(title); lb.setStyleSheet("font-weight:600;")
        v.addWidget(lb)
    return f, v


class RouteToolWindow(QMainWindow):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.mode = MODE_GOAL
        self.queue = []
        self.setWindowTitle("SCV 경로 지시")
        self.resize(1280, 840)
        self.setStyleSheet(T.QSS)

        self.view = GraphView()
        self.view.nodePicked.connect(self._on_pick)

        root = QWidget(); h = QHBoxLayout(root)
        h.setContentsMargins(10, 10, 10, 10); h.setSpacing(10)
        h.addWidget(self.view, 1)
        h.addWidget(self._side(), 0)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.view.statusMessage.connect(lambda s: self.statusBar().showMessage(s, 1500))

        bridge.graphReceived.connect(self._on_graph)
        bridge.datumReceived.connect(self.view.set_datum)
        bridge.robotMoved.connect(lambda x, y, t: self.view.set_robot((x, y, t)))
        bridge.pathReceived.connect(self.view.set_path)
        bridge.statusReceived.connect(self._on_status)

    # ------------------------------------------------------------ UI
    def _side(self):
        box = QWidget(); box.setFixedWidth(320)
        v = QVBoxLayout(box); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(10)

        f, fv = _panel("보내는 것")
        self.grp = QButtonGroup(self)
        row = QHBoxLayout()
        for key, label in ((MODE_GOAL, "목표 하나"), (MODE_ROUTE, "경로 지정")):
            b = QPushButton(label); b.setCheckable(True)
            b.clicked.connect(lambda _c, k=key: self._set_mode(k))
            self.grp.addButton(b); row.addWidget(b)
            if key == MODE_GOAL:
                b.setChecked(True)
        fv.addLayout(row)
        self.hint = QLabel(""); self.hint.setObjectName("faint"); self.hint.setWordWrap(True)
        fv.addWidget(self.hint)
        v.addWidget(f)

        f2, f2v = _panel("경로 큐")
        self.list = QListWidget()
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.model().rowsMoved.connect(self._reorder)
        f2v.addWidget(self.list, 1)
        r = QHBoxLayout()
        self.btn_del = QPushButton("선택 삭제"); self.btn_del.clicked.connect(self._del)
        self.btn_clr = QPushButton("비우기"); self.btn_clr.clicked.connect(self._clear)
        r.addWidget(self.btn_del); r.addWidget(self.btn_clr)
        f2v.addLayout(r)
        self.chk_loop = QCheckBox("루프 (마지막 → 처음)")
        self.chk_fill = QCheckBox("간격 자동 채우기"); self.chk_fill.setChecked(True)
        self.chk_fill.setToolTip("이웃한 두 노드가 직접 링크로 안 이어져 있으면 "
                                 "그 구간만 최단 경로로 채웁니다.")
        for c in (self.chk_loop, self.chk_fill):
            c.stateChanged.connect(lambda _s: self._refresh_preview())
            f2v.addWidget(c)
        v.addWidget(f2, 1)

        f3, f3v = _panel("위성 배경")
        cb = QComboBox()
        for k in BM.ORDER:
            cb.addItem("배경 꺼짐" if k == "off" else BM.PROVIDERS[k]["label"], k)
        cb.currentIndexChanged.connect(
            lambda i: self.view.set_basemap(cb.itemData(i)))
        f3v.addWidget(cb)
        sl = QSlider(Qt.Orientation.Horizontal); sl.setRange(10, 100); sl.setValue(75)
        sl.valueChanged.connect(lambda x: (setattr(self.view, "basemap_opacity", x / 100.0),
                                           self.view.update()))
        f3v.addWidget(sl)
        v.addWidget(f3)

        self.btn_pub = QPushButton("발행"); self.btn_pub.setObjectName("primary")
        self.btn_pub.clicked.connect(self._publish)
        v.addWidget(self.btn_pub)
        self.result = QLabel("대기 중"); self.result.setObjectName("dim")
        self.result.setWordWrap(True)
        v.addWidget(self.result)

        self._set_mode(MODE_GOAL)
        return box

    def _set_mode(self, key):
        self.mode = key
        route = (key == MODE_ROUTE)
        self.hint.setText(
            "노드를 누르면 그 자리로 갑니다. 플래너가 최단 경로를 찾습니다."
            if not route else
            "노드를 차례로 눌러 큐에 담고 발행하세요. 최단이 아니어도 이 순서로 갑니다.")
        for w in (self.list, self.btn_del, self.btn_clr, self.chk_loop, self.chk_fill):
            w.setEnabled(route)
        self.btn_pub.setEnabled(route)   # 목표 하나는 클릭 즉시 발행
        if not route:
            self._clear()

    # ------------------------------------------------------------ 큐
    def _on_pick(self, nid):
        if self.mode == MODE_GOAL:
            n = self.view.nodes.get(nid)
            if not n:
                return
            self._send(f"goal_{int(time.time() * 1000)}", [], (n[0], n[1]))
            self.statusBar().showMessage(f"목표 발행 — {nid}", 4000)
            return
        if self.queue and self.queue[-1] == nid:
            return                      # 같은 노드 연속 클릭은 무시
        self.queue.append(nid)
        self._sync_list()

    def _sync_list(self):
        self.list.clear()
        for i, nid in enumerate(self.queue):
            self.list.addItem(QListWidgetItem(f"{i + 1}.  {nid}"))
        self.view.set_queue(self.queue)
        self._refresh_preview()

    def _reorder(self, *_a):
        self.queue = [self.list.item(i).text().split('.', 1)[1].strip()
                      for i in range(self.list.count())]
        self._sync_list()

    def _del(self):
        for it in self.list.selectedItems():
            nid = it.text().split('.', 1)[1].strip()
            if nid in self.queue:
                self.queue.remove(nid)
        self._sync_list()

    def _clear(self):
        self.queue = []
        self.list.clear()
        self.view.set_queue([])
        self.view.set_preview([], [])

    # ------------------------------------------------------------ 미리보기
    def _refresh_preview(self):
        """발행 전에 이어지는지 여기서 확인한다.

        그래프를 이미 갖고 있으니 로봇에 보내고 실패 로그를 뒤질 이유가 없다.
        끊긴 구간은 지도에 빨간 점선으로 바로 보인다.
        """
        if len(self.queue) < 2:
            self.view.set_preview([], [])
            return
        way = list(self.queue)
        if self.chk_loop.isChecked():
            way.append(way[0])
        adj = {}
        for a, b, bi in self.view.links:
            adj.setdefault(a, set()).add(b)
            if bi:
                adj.setdefault(b, set()).add(a)
        pts, bad = [], []
        for i, nid in enumerate(way):
            n = self.view.nodes.get(nid)
            if n:
                pts.append((n[0], n[1]))
            if i == 0:
                continue
            a, b = way[i - 1], nid
            if b in adj.get(a, ()):          # 직접 링크
                continue
            if not self.chk_fill.isChecked() or not self._reachable(adj, a, b):
                bad.append((a, b))
        self.view.set_preview(pts, bad)
        if bad:
            self.result.setText(f"이어지지 않는 구간 {len(bad)}개 — 발행 전에 확인하세요")
            self.result.setStyleSheet(f"color:{T.ERROR.name()};")
        else:
            self.result.setText(f"구간 {max(0, len(way) - 1)}개 — 발행 가능")
            self.result.setStyleSheet(f"color:{T.TEXT_DIM.name()};")

    @staticmethod
    def _reachable(adj, a, b, limit=4000):
        seen, stack = {a}, [a]
        while stack and len(seen) < limit:
            x = stack.pop()
            if x == b:
                return True
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return b in seen

    # ------------------------------------------------------------ 발행
    def _publish(self):
        if not self.queue:
            QMessageBox.information(self, "발행", "큐가 비어 있습니다.")
            return
        self._send(f"route_{int(time.time() * 1000)}", self.queue, None)

    def _send(self, rid, via, goal_xy):
        self.bridge.publish_route(rid, via, goal_xy,
                                  loop=self.chk_loop.isChecked(),
                                  fill_gaps=self.chk_fill.isChecked())
        self.result.setText(f"발행 — {rid} (응답 대기)")
        self.result.setStyleSheet(f"color:{T.TEXT_DIM.name()};")

    def _on_graph(self, nodes, links):
        self.view.set_graph(nodes, links)
        self.statusBar().showMessage(
            f"그래프 수신 — 노드 {len(nodes)} 링크 {len(links)}", 5000)
        self._refresh_preview()

    def _on_status(self, rid, accepted, reason, count):
        if accepted:
            self.result.setText(f"수락 — {rid}: 노드 {count}개")
            self.result.setStyleSheet(f"color:{T.PATH.name()};")
        else:
            self.result.setText(f"거부 — {rid}: {reason}")
            self.result.setStyleSheet(f"color:{T.ERROR.name()};")

    def closeEvent(self, ev):
        self.bridge.shutdown()
        super().closeEvent(ev)


def main(argv=None):
    app = QApplication(sys.argv if argv is None else argv)
    app.setStyleSheet(T.QSS)
    from .ros_bridge import RosBridge
    bridge = RosBridge()
    w = RouteToolWindow(bridge)
    w.show()
    bridge.start()          # 시그널 연결이 끝난 뒤에 수신 시작
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
