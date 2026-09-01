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

# 옛 거동(상한 전부 없음)이 그날의 실패를 만든다는 증거를 남긴다.
pm3 = mk(CHAMBER)
i_ch = pm3.align_to_position(*V_CH, limit_to_passed=False)
check('(C) 상한 없으면 경로 끝(P3)으로 건너뛴다 — 챔버 실패 재현',
      i_ch == 3, f'(idx {i_ch} = {CHAMBER[i_ch][0]})')

# 기본값에서는 재현되지 않아야 한다. 실제 챔버처럼 RC 주행 이력을 넣는다 —
# 게이트는 '이동 후 따라잡기'를 대상으로 하므로 이력이 있어야 작동한다.
def ch_with_history():
    pm = mk(CHAMBER)
    for k in range(13):              # (0,1) -> (6,1) RC 주행
        pm.note_position(0.5 * k, 1.0)
    return pm

i_ch_def = ch_with_history().align_to_position(*V_CH)
check('(C) 기본값에서는 건너뛰지 않는다', i_ch_def < 3,
      f'(idx {i_ch_def} = {CHAMBER[i_ch_def][0]})')

pm4 = mk(CHAMBER)
i_ch2 = pm4.align_to_position(*V_CH, max_skip_from_current=2, limit_to_passed=False)
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

# 전부 막혀도 합류 불가로 멎지 않는다. 다만 되돌아갈 곳은 '최근접'이 아니라
# 통과 게이트 안이다 — 최근접으로 떨어지면 방금 막혀서 뺀 노드로 돌아간다.
i_blk = ch_with_history().align_to_position(*V_CH, approach_clear=lambda *a: False)
check('전 후보 차단 시에도 합류점은 정해진다(게이트 안)', i_blk <= 2,
      f'(idx {i_blk})')

pm7b = mk(CHAMBER)
i_blk2 = pm7b.align_to_position(*V_CH, approach_clear=lambda *a: False,
                                limit_to_passed=False)
check('게이트를 끄면 종전대로 최근접 폴백', i_blk2 == 3, f'(idx {i_blk2})')


# --- 통과 이력 기준 (limit_to_passed) -----------------------------------
# "가까운가" 가 아니라 "지나왔는가" 로 가른다. 위 두 사례가 기하만으로는
# 구분되지 않는 것을 이력이 갈라 준다.

# (F) 필드: RC 로 경로 옆을 훑고 지나갔다 -> 지나온 노드까지 따라잡기 허용
pm8 = mk(FIELD)
pm8.align_to_position(*V_START, limit_to_passed=True)
# RC 주행을 표본으로 재현 (경로에서 ~1.1 m 옆을 따라 이동)
for k in range(21):
    t = k / 20
    pm8.note_position(V_START[0] + (V_ENGAGE[0] - V_START[0]) * t,
                      V_START[1] + (V_ENGAGE[1] - V_START[1]) * t)
i_f8 = pm8.align_to_position(*V_ENGAGE, limit_to_passed=True)
check('(F) 곁을 지나온 노드까지는 따라잡기 허용', i_f8 >= i_start + 3,
      f'(idx {i_f8}, 지나온 최대 {pm8.max_passed_index})')

# (C) 챔버: 경로에서 6 m 옆을 지나갔을 뿐 P1/P2 곁에 간 적이 없다
pm9 = mk(CHAMBER)
for k in range(13):                      # (0,1) -> (6,1) RC 주행
    pm9.note_position(0.5 * k, 1.0)
i_c9 = pm9.align_to_position(*V_CH, limit_to_passed=True)
# P1(0,3) 은 RC 선(y=1)에서 정확히 2.0 m — pass_radius 경계라 '지나옴'으로
# 잡힌다. 규칙은 지나온 다음 칸까지 허용하므로 P2 까지가 상한이다. 중요한
# 것은 **경로 끝 P3 로 건너뛰지 않는 것**이고, P2 로의 접근 직선은
# approach_clear 가 다시 거른다(둘이 겹쳐 막는다).
check('(C) 지나온 적 없는 경로 끝으로는 건너뛰지 않음', i_c9 <= 2,
      f'(idx {i_c9} = {CHAMBER[i_c9][0]}, 지나온 최대 {pm9.max_passed_index})')

pm9b = mk(CHAMBER)
for k in range(13):
    pm9b.note_position(0.5 * k, 1.0)
i_c9b = pm9b.align_to_position(*V_CH, limit_to_passed=True, approach_clear=clear)
check('(C) 이력+접근검사 함께면 경로 유지', i_c9b <= 1,
      f'(idx {i_c9b} = {CHAMBER[i_c9b][0]})')

# 통과 이력이 아예 없으면 게이트하지 않는다 — 2026-08-05 최초 합류 보호.
# 그날 차량은 경로 옆에 놓인 채 기동했고(경로 머리는 뒤쪽) 앞쪽 합류가
# 맞았다. 여기까지 게이트하면 그 수정이 되살아나 무효가 된다.
pm10 = mk(CHAMBER)
i_c10 = pm10.align_to_position(20.0, 20.0, limit_to_passed=True)
check('이력 없으면 게이트하지 않음 (8/5 최초 합류 보호)',
      i_c10 == 3 and pm10.max_passed_index == -1, f'(idx {i_c10})')

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
