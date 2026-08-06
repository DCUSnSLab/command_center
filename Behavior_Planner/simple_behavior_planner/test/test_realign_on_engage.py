#!/usr/bin/env python3
"""자율 전환 시 합류 노드 재선택 — 2026-08-06 필드 기하 재현.

그날 실패: 기동 시점에 합류 노드 N0440(경로 idx 2)을 고른 뒤, yaw 게이트
절차대로 RC 로 (32.3, 59.7)까지 이동했는데 합류 노드가 갱신되지 않아
자율 전환 즉시 뒤쪽 노드(N0440, 남동 -72.8도)로 18.8 m 갔다. 목표 N026
은 북서 +139.4도였다.

이 테스트는 PathManager 수준에서 "기동 시점 선택 vs 전환 시점 재선택"이
서로 다른 노드를 고르는지, 그리고 재선택 결과가 목표 방향과 일치하는지
확인한다. (노드 전체를 띄우지 않고 선택 로직만 본다 — 재현에 필요한 건
그것뿐이고, hunter_msgs 가 없는 개발 PC 에서도 돌아간다.)
"""
import math
import sys

sys.path.insert(0, '/home/ppub/scv_ws/src/command_center/Behavior_Planner/'
                   'simple_behavior_planner')
from simple_behavior_planner.path_manager import PathManager  # noqa: E402

P = F = 0


def check(name, ok, detail=''):
    global P, F
    P, F = (P + 1, F) if ok else (P, F + 1)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# d2_unha 경로 N0438 → N026 중 해당 구간 (map, datum 상대 m)
ROUTE = [
    ('N0438', 36.12, 35.51), ('N0439', 37.50, 36.96), ('N0440', 38.87, 38.41),
    ('N0441', 40.25, 39.86), ('N0442', 41.62, 41.31), ('N0443', 43.00, 42.77),
    ('N0444', 43.73, 44.97), ('N000', 43.55, 46.60), ('N001', 42.43, 48.43),
    ('N002', 40.90, 49.77), ('N003', 39.28, 51.21), ('N004', 37.75, 52.54),
    ('N005', 36.31, 53.88), ('N006', 34.78, 55.21), ('N007', 33.25, 56.55),
    ('N008', 31.72, 57.88), ('N009', 30.09, 59.10), ('N010', 28.56, 60.44),
    ('N011', 27.12, 61.77), ('N012', 25.41, 62.88), ('N013', 23.34, 63.33),
    ('N014', 21.62, 64.45), ('N015', 20.64, 66.33), ('N016', 20.10, 68.33),
    ('N017', 19.29, 70.22), ('N018', 18.03, 71.77), ('N019', 16.59, 73.11),
    ('N020', 15.06, 74.55), ('N021', 13.53, 75.89), ('N022', 12.00, 77.11),
    ('N023', 10.46, 78.44), ('N024', 8.93, 79.78), ('N025', 7.40, 81.11),
    ('N026', 5.87, 82.33),
]
# 기동 시점 위치는 로깅 전이라 정확한 값이 없다. 다만 그때 플래너가
# 고른 합류 노드가 N0440(idx 2)이었으므로, 그 선택이 재현되는 지점을
# 역산해 쓴다. 전환 위치는 로거 첫 표본(실측)이다.
START = (40.5, 36.5)      # 기동 시점 (N0440 을 고르는 위치 — 역산)
ENGAGE = (32.3, 59.7)     # yaw 게이트(RC 20 m) 후 자율 전환 위치 (실측)
GOAL = (5.87, 82.33)


def mk():
    pm = PathManager()
    pm.path_nodes = [{'id': i, 'x': x, 'y': y} for i, x, y in ROUTE]
    return pm


def bearing(frm, to):
    return math.degrees(math.atan2(to[1] - frm[1], to[0] - frm[0]))


def diff(a, b):
    return abs((a - b + 180) % 360 - 180)


pm = mk()
i_start = pm.align_to_position(*START)
n_start = ROUTE[i_start]
pm2 = mk()
i_engage = pm2.align_to_position(*ENGAGE)
n_engage = ROUTE[i_engage]

print("기동 시점 합류: %s (%.1f, %.1f)" % n_start)
print("전환 시점 합류: %s (%.1f, %.1f)" % n_engage)

# 기동 위치가 로깅되지 않아 노드를 정확히 특정할 수는 없다. 그날 실제
# 선택은 idx 2(N0440)였으므로 '경로 앞머리 부근'인지만 확인한다.
check('기동 시점 선택이 경로 앞머리 (idx<=3, 그날 실제 2)', i_start <= 3,
      f'(idx {i_start}, {n_start[0]})')

# 그날의 실패: 기동 시점 노드를 전환 위치에서 쓰면 목표 반대로 간다
br_stale = bearing(ENGAGE, (n_start[1], n_start[2]))
br_goal = bearing(ENGAGE, GOAL)
check('갱신 안 하면 목표와 90도 넘게 어긋남 (그날 재현)',
      diff(br_stale, br_goal) > 90,
      f'(합류 {br_stale:+.0f}° vs 목표 {br_goal:+.0f}°, 차 {diff(br_stale, br_goal):.0f}°)')

# 수정 후: 전환 시점에 재선택하면 목표 방향으로 향한다
# 합류 노드가 3 m 안이면 제어기는 곧 다음 노드로 넘어가므로 그쪽 방향을 본다
# (코앞 노드의 방위각은 잡음이라 판정에 쓸 수 없다)
tgt = n_engage
if math.dist(ENGAGE, (n_engage[1], n_engage[2])) < 3.0 and i_engage + 1 < len(ROUTE):
    tgt = ROUTE[i_engage + 1]
br_fresh = bearing(ENGAGE, (tgt[1], tgt[2]))
check('재선택하면 진행 방향이 목표와 45도 이내', diff(br_fresh, br_goal) < 45,
      f'(향하는 곳 {tgt[0]} {br_fresh:+.0f}° vs 목표 {br_goal:+.0f}°, 차 {diff(br_fresh, br_goal):.0f}°)')

check('재선택 노드가 경로상 더 앞쪽', i_engage > i_start,
      f'(idx {i_engage} > {i_start})')

# 재선택 후 목표까지 총비용이 줄어드는가
def total(i, frm):
    d = math.dist(frm, (ROUTE[i][1], ROUTE[i][2]))
    rem = sum(math.dist((ROUTE[k][1], ROUTE[k][2]), (ROUTE[k+1][1], ROUTE[k+1][2]))
              for k in range(i, len(ROUTE) - 1))
    return d + rem


t_stale, t_fresh = total(i_start, ENGAGE), total(i_engage, ENGAGE)
check('총 주행거리 감소', t_fresh < t_stale - 5,
      f'({t_fresh:.1f} m vs 갱신없음 {t_stale:.1f} m)')

# 전환 위치가 기동 위치와 같으면 재선택해도 결과가 같아야 (불필요한 변경 방지)
pm3 = mk()
check('이동이 없으면 선택 불변', pm3.align_to_position(*START) == i_start)

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
