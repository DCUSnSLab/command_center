#!/usr/bin/env python3
"""챔버 A/B 결과 요약: <out_dir>/{simple,bt}N.json (gz_judge 산출) + shadow1 로그 비교."""
import glob
import json
import os
import re
import sys

out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, '*.json'))):
    if f.endswith('.traj.json'):
        continue
    name = os.path.basename(f)[:-5]
    if name.startswith('shadow'):
        continue
    d = json.load(open(f))
    if not isinstance(d, dict):
        continue
    # 도달 시각: 궤적 사이드카(t,x,y,z)에서 목표 반경(REACH_D) 최초 진입 시각
    reach_t = None
    tj = f + '.traj.json'
    if os.path.exists(tj):
        import math
        goal = json.load(open(os.environ.get('SCV_DESIGN', os.path.expanduser('~/SCV/scv_sim/chamber/chamber_20260803.design.json'))))['route']['goal']
        for tt, x, y, z in json.load(open(tj)):
            if math.hypot(x - goal[0], y - goal[1]) < 1.5:
                reach_t = tt; break
    d['reach_t'] = reach_t
    arm = re.sub(r'\d+$', '', name)
    rows.append((arm, name, d.get('verdict'), d.get('distance'), d.get('min_goal_dist'), d.get('reach_t'),
                 d.get('n_violations'), d.get('blocked_seen'), d.get('min_groundtruth_z'), d.get('behavior_transitions')))
print(f"{'arm':7} {'run':9} {'verdict':14} {'dist':>7} {'min_goal':>9} {'reach_t':>8} {'viol':>5} {'blocked':>8} {'min_z':>7}  transitions")
for r in rows:
    print(f"{r[0]:7} {r[1]:9} {str(r[2]):14} {r[3]!s:>7} {r[4]!s:>9} {r[5]!s:>8} {r[6]!s:>5} {r[7]!s:>8} {r[8]!s:>7}  {[b for _, b in (r[9] or [])]}")
for arm in ('simple', 'bt'):
    rs = [r for r in rows if r[0] == arm]
    if not rs:
        continue
    reached = sum(1 for r in rs if r[2] == 'PASS_REACHED')
    mg = [r[4] for r in rs if r[4] is not None]
    di = [r[3] for r in rs if r[3] is not None]
    rt = [r[5] for r in rs if r[5] is not None]
    print(f"{arm}: PASS_REACHED {reached}/{len(rs)}, min_goal_dist mean {sum(mg)/len(mg):.2f}, distance mean {sum(di)/len(di):.2f}, "
          f"reach_t mean {sum(rt)/len(rt):.1f}s (n={len(rt)})" if mg and rt else f"{arm}: n/a")

# shadow: simple 의 목표 노드 전이 vs BT 의 /bt/behavior 트레이스 (로그에서 추출)
sh = os.path.join(out, 'logs_shadow1')
if os.path.isdir(sh):
    beh = open(os.path.join(sh, 'behavior.log'), errors='replace').read() if os.path.exists(os.path.join(sh, 'behavior.log')) else ''
    bt = open(os.path.join(sh, 'bt.log'), errors='replace').read() if os.path.exists(os.path.join(sh, 'bt.log')) else ''
    s_join = re.findall(r'join idx (\d+) \(([A-Za-z0-9]+)\)', beh)
    b_join = re.findall(r'join idx (\d+) \(([A-Za-z0-9]+)\)', bt)
    s_adv = re.findall(r'Advanced to next node: ([A-Za-z0-9]+)', beh)
    if not s_adv:
        # simple 은 advance 를 debug 로만 찍는다 — 도달 보고를 무시한 로그의 target= 열(목표 변경 순서)로 복원
        for tgt in re.findall(r'!= target=([A-Za-z0-9]+)', beh):
            if not s_adv or s_adv[-1] != tgt:
                s_adv.append(tgt)
    b_adv = re.findall(r'advanced to ([A-Za-z0-9]+)', bt)
    print('\nshadow1: join simple', s_join, ' bt', b_join)
    print('shadow1: advance simple', s_adv, '\n         advance bt    ', b_adv)
    print('shadow1: BT blocked states', re.findall(r'\[BLOCKED\] state -> (\w+)', bt),
          ' simple', re.findall(r'\[BLOCKED\] state -> (\w+)', beh))
