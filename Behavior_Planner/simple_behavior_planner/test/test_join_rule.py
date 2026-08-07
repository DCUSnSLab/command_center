#!/usr/bin/env python3
"""경로 합류 규칙 단위 테스트 (2026-08-05 필드 기하 재현)."""
import math
import sys

import os
# 절대경로를 박으면 개발 PC 밖에서 죽는다 — 차량(/home/scv/SCV_park)에서
# 이 테스트가 ModuleNotFoundError 로 실행조차 안 됐다. 패키지 루트는
# 이 파일 기준 한 단계 위다.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from simple_behavior_planner.path_manager import PathManager  # noqa: E402

P = F = 0


def check(name, ok, detail=''):
    global P, F
    P, F = (P + 1, F) if ok else (P, F + 1)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def mk(coords):
    pm = PathManager()
    pm.path_nodes = [{'id': 'N%02d' % i, 'x': x, 'y': y}
                     for i, (x, y) in enumerate(coords)]
    return pm


# 2026-08-05 실측 기하: 경로는 N0020(36.9,33.2) -> N0009(-2.3,71.9) 북서 방향,
# 차량은 (55.2,40.8) 즉 경로 꼬리 쪽 동편. 최근접은 꼬리 N0020(19.8 m)이지만
# 그쪽으로 가면 목표에서 멀어진다.
FIELD = [(36.9, 33.2), (33.0, 36.5), (29.1, 39.7), (25.6, 43.3), (21.8, 46.6),
         (17.4, 49.0), (13.4, 52.2), (10.0, 55.9), (6.7, 59.7), (3.1, 63.3),
         (0.2, 67.5), (-2.3, 71.9)]
VX, VY = 55.2, 40.8

pm = mk(FIELD)
idx_cost = pm.align_to_position(VX, VY)
pm2 = mk(FIELD)
idx_near = pm2.align_to_position(VX, VY, max_approach_m=0.0)   # 폴백=최근접


def total(pm_, i):
    d = math.hypot(pm_.path_nodes[i]['x'] - VX, pm_.path_nodes[i]['y'] - VY)
    rem = sum(math.dist((pm_.path_nodes[k]['x'], pm_.path_nodes[k]['y']),
                        (pm_.path_nodes[k + 1]['x'], pm_.path_nodes[k + 1]['y']))
              for k in range(i, len(pm_.path_nodes) - 1))
    return d + rem


check('최근접 규칙은 경로 꼬리(N00)를 고른다', idx_near == 0,
      f'(idx {idx_near})')
check('비용 규칙은 더 앞쪽 노드를 고른다', idx_cost > idx_near,
      f'(idx {idx_cost} vs {idx_near})')
check('비용 규칙 총거리가 더 짧다',
      total(pm, idx_cost) < total(pm2, idx_near) - 1.0,
      f'({total(pm, idx_cost):.1f} m vs {total(pm2, idx_near):.1f} m)')

# 합류 노드가 목표 방향(북서)에 있어야 한다 — 8/5 처럼 반대로 틀지 않도록
n = pm.path_nodes[idx_cost]
br = math.degrees(math.atan2(n['y'] - VY, n['x'] - VX))
goal = pm.path_nodes[-1]
gbr = math.degrees(math.atan2(goal['y'] - VY, goal['x'] - VX))
diff = abs((br - gbr + 180) % 360 - 180)
check('합류 방향이 목표 방향과 90도 이내', diff < 90, f'({diff:.0f}도 차)')

# 접근거리 상한: 상한을 낮추면 가까운 노드로 물러난다
pm3 = mk(FIELD)
i3 = pm3.align_to_position(VX, VY, max_approach_m=30.0)
d3 = math.hypot(pm3.path_nodes[i3]['x'] - VX, pm3.path_nodes[i3]['y'] - VY)
check('접근거리 상한 준수', d3 <= 30.0 + 1e-6, f'({d3:.1f} m)')
check('상한이 크면 더 멀리 합류', i3 <= idx_cost, f'({i3} <= {idx_cost})')

# 가중치가 없으면(1.0) 경로를 상시 가로지른다 — 회귀 감시
pm_w1 = mk(FIELD)
i_w1 = pm_w1.align_to_position(35.0, 34.0, max_approach_m=40.0,
                               approach_weight=1.0, max_skip_nodes=99)
check('무가중(1.0)은 경로 머리 옆에서도 앞쪽으로 건너뜀 (그래서 쓰지 않음)',
      i_w1 > 2, f'(idx {i_w1})')

# 경로 머리 옆에서는 크게 건너뛰지 않아야 한다. 바로 다음 노드로 합류하는
# 것은 정상이다 — 이미 지나친 노드로 되돌아가지 않는다는 뜻이므로.
pm4 = mk(FIELD)
i4 = pm4.align_to_position(35.0, 34.0)
pm5 = mk(FIELD)
i5 = pm5.align_to_position(35.0, 34.0, max_approach_m=0.0)
check('경로 머리 옆에서는 1노드 이내 합류', i4 <= i5 + 1, f'({i4} vs 최근접 {i5})')

# 경로 중간에 있을 때 뒤로 돌아가지 않는다
pm6 = mk(FIELD)
i6 = pm6.align_to_position(13.0, 53.0)          # N06 근처
check('경로 중간에서는 그 지점부터 합류', i6 >= 5, f'(idx {i6})')

# 전부 상한 초과 시 최근접 폴백 (합류 불가 방지)
pm7 = mk(FIELD)
i7 = pm7.align_to_position(500.0, 500.0, max_approach_m=10.0)
near = min(range(len(FIELD)),
           key=lambda k: (FIELD[k][0]-500.0)**2 + (FIELD[k][1]-500.0)**2)
check('상한 전부 초과 시 최근접 폴백', i7 == near, f'(idx {i7}, 최근접 {near})')

# 노드 건너뛰기 상한: 직선 경로에서 목표 끝까지 가로지르지 않아야 한다
STRAIGHT = [(3.0*i, 0.0) for i in range(8)]      # 챔버 그래프와 동일 형태
pm8 = mk(STRAIGHT)
i8 = pm8.align_to_position(10.0, 8.0)            # 경로 옆 8 m
pm9 = mk(STRAIGHT)
i9 = pm9.align_to_position(10.0, 8.0, max_skip_nodes=99)
check('직선 경로에서 건너뛰기 제한됨', i8 <= 5 and i8 < i9,
      f'(제한 {i8} vs 무제한 {i9})')

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
