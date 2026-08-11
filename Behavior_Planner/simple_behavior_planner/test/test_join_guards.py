#!/usr/bin/env python3
"""합류 규칙 상한 두 종 — 서로 반대 방향의 실패를 각각 막는지.

두 실측 사례가 정반대를 요구한다:

  (F) 2026-08-06 필드  경로 옆을 20 m RC 주행 후 자율 전환. 노드에 닿지
      않아 진행 인덱스는 0. **앞쪽으로 따라잡아야** 한다 — 못 하면 경로
      머리로 되돌아가는 그날 오전의 18.8 m 역주행이 재현된다.

  (C) 2026-08-11 챔버  경로에서 5~6 m 벗어난 곳에서 전환. 최근접이 이미
      경로 끝이라 합류가 목표 노드로 잡히고, 차량이 매핑된 경로를 버리고
      지름길을 시도하다 그 직선 위 장애물에 막혔다(3/3 미도달).
      **건너뛰지 말아야** 한다.

이 파일의 목적은 "하나의 상한으로 둘 다 되는가"를 코드로 못박는 것이다.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from simple_behavior_planner.path_manager import PathManager  # noqa: E402

P = F = 0


def check(name, ok, detail=''):
    global P, F
    P, F = (P + 1, F) if ok else (P, F + 1)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def mk(nodes):
    pm = PathManager()
    pm.path_nodes = [{'id': i, 'x': x, 'y': y} for i, x, y in nodes]
    return pm


# --- (F) 필드: 경로 옆 20 m 이동 후 따라잡기 -----------------------------
# d2_unha N0438->N026 구간을 단순화 — 1.8 m 간격 직선, 차량은 그 옆 2 m.
FIELD = [(f'N{i}', 36.1 + 1.3 * i, 35.5 + 1.3 * i) for i in range(20)]
V_START = (35.0, 34.5)      # 경로 머리 옆
V_ENGAGE = (48.0, 47.5)     # 20 m 진행 후, 경로에서 ~2 m 옆

pm = mk(FIELD)
i_start = pm.align_to_position(*V_START)
i_engage = pm.align_to_position(*V_ENGAGE)
check('(F) 이동 후 앞쪽으로 따라잡음', i_engage > i_start + 3,
      f'(idx {i_start} -> {i_engage})')

pm2 = mk(FIELD)
pm2.align_to_position(*V_START)          # 진행 인덱스가 여기서 굳는다
i_bound = pm2.align_to_position(*V_ENGAGE, max_skip_from_current=2)
check('(F) 현재기준 상한 2 를 켜면 따라잡기가 막힌다 — 그날 오전 재현',
      i_bound <= i_start + 2, f'(idx {i_bound}, 상한 없으면 {i_engage})')

# --- (C) 챔버: 경로를 건너뛰는 지름길 -----------------------------------
CHAMBER = [('P0', 0, 1), ('P1', 0, 3), ('P2', 0, 5), ('P3', 3, 5)]
V_CH = (6.0, 1.0)

pm3 = mk(CHAMBER)
i_ch = pm3.align_to_position(*V_CH)
check('(C) 상한 없으면 경로 끝(P3)으로 건너뛴다 — 챔버 실패 재현',
      i_ch == 3, f'(idx {i_ch} = {CHAMBER[i_ch][0]})')

pm4 = mk(CHAMBER)
i_ch2 = pm4.align_to_position(*V_CH, max_skip_from_current=2)
check('(C) 현재기준 상한 2 로 건너뛰기 차단', i_ch2 <= 2,
      f'(idx {i_ch2} = {CHAMBER[i_ch2][0]})')


# --- approach_clear: 접근 직선 통행성 -----------------------------------
# 챔버 실측 장애물 3개. 로봇 반폭 + 여유 0.45 m 로 부풀린다.
OBS = [(4.75, 2.25), (4.75, 2.75), (5.25, 2.25)]


def clear(x0, y0, x1, y1, margin=0.45, step=0.25):
    d = math.hypot(x1 - x0, y1 - y0)
    for k in range(int(d / step) + 1):
        t = (k * step) / d if d > 0 else 0.0
        px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        for ox, oy in OBS:
            if abs(px - ox) < 0.25 + margin and abs(py - oy) < 0.25 + margin:
                return False
    return True


check('(C) 접근 직선 판정: (6,1)->P3(3,5) 은 막힘', not clear(6, 1, 3, 5))
check('(C) 접근 직선 판정: (6,1)->P0(0,1) 은 통행 가능', clear(6, 1, 0, 1))

pm5 = mk(CHAMBER)
i_ch3 = pm5.align_to_position(*V_CH, approach_clear=clear)
check('(C) approach_clear 로 P3 제외 -> 경로를 유지', i_ch3 < 3,
      f'(idx {i_ch3} = {CHAMBER[i_ch3][0]})')

# 필드 쪽은 approach_clear 를 켜도 따라잡기가 유지돼야 한다(장애물 없음)
pm6 = mk(FIELD)
pm6.align_to_position(*V_START)
i_f3 = pm6.align_to_position(*V_ENGAGE, approach_clear=lambda *a: True)
check('(F) approach_clear 켜도 따라잡기 유지', i_f3 == i_engage,
      f'(idx {i_f3})')

# 전부 막히면 최근접으로 폴백 — 합류 불가로 멎지 않는다
pm7 = mk(CHAMBER)
i_blk = pm7.align_to_position(*V_CH, approach_clear=lambda *a: False)
check('전 후보 차단 시 최근접 폴백', i_blk == 3, f'(idx {i_blk})')

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
