"""색 — SCV_MapManager/map_editor 와 같은 팔레트.

두 도구를 나란히 놓고 볼 때 같은 지도가 같은 색으로 보여야 한다.
그래프는 분홍 계열: 의미 클래스(청록·노랑·보라)와 위성영상(초록·갈색)에서
모두 떨어져 있어 어디서든 읽힌다.
"""
from PyQt6.QtGui import QColor

BG          = QColor("#0b0d11")
BG_PANEL    = QColor("#12151b")
BORDER      = QColor("#232936")
TEXT        = QColor("#e6eaf0")
TEXT_DIM    = QColor("#a3adbc")
TEXT_FAINT  = QColor("#6d7787")
ACCENT      = QColor("#c8ff4d")

GRAPH_GPS     = QColor("#ff2d95")   # 사각형 — GPS 유래
GRAPH_SLAM_HI = QColor("#ff85c2")   # 원 — SLAM
GRAPH_SLAM_LO = QColor("#b8577f")
GRAPH_HALO    = QColor(11, 13, 17, 200)

QUEUE      = QColor("#c8ff4d")      # 큐에 담긴 노드
QUEUE_TEXT = QColor("#0b0d11")
ROBOT      = QColor("#3dd6ff")
PATH       = QColor("#26d07c")      # 로봇이 되돌려준 실제 경로
PREVIEW    = QColor("#ffd166")      # 발행 전 예상 경로
ERROR      = QColor("#ff5d5d")

QSS = f"""
QWidget {{ background:{BG.name()}; color:{TEXT.name()};
           font-family:'Noto Sans KR','DejaVu Sans'; font-size:12px; }}
QFrame#panel {{ background:{BG_PANEL.name()}; border:1px solid {BORDER.name()};
                border-radius:8px; }}
QPushButton {{ background:{BG_PANEL.name()}; border:1px solid {BORDER.name()};
               border-radius:6px; padding:6px 12px; }}
QPushButton:hover {{ border-color:{TEXT_FAINT.name()}; }}
QPushButton#primary {{ background:{ACCENT.name()}; color:#0b0d11; font-weight:600;
                       border:none; }}
QPushButton:disabled {{ color:{TEXT_FAINT.name()}; }}
QListWidget {{ background:{BG.name()}; border:1px solid {BORDER.name()};
               border-radius:6px; }}
QLabel#dim {{ color:{TEXT_DIM.name()}; }}
QLabel#faint {{ color:{TEXT_FAINT.name()}; font-size:11px; }}
"""
