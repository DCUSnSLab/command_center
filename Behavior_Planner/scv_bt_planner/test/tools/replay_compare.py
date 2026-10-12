#!/usr/bin/env python3
"""bag 재생 shadow 비교 기록기.

같은 입력(재생 bag)을 받는 simple_behavior_planner(active)와 scv_bt_planner(shadow)의 출력을
사이드바이사이드로 기록하고, 종료 시 JSON 과 요약을 낸다.

  simple : /multiple_waypoints, /mppi_params, /behavior_status, /pause_command
  bt     : /bt/multiple_waypoints, /bt/mppi_params, /bt/behavior_status, /bt/pause_command, /bt/behavior
  녹화본 : /rec/behavior_status (bag 의 /behavior_status 를 리맵)

지표:
  - target_agreement : 두 플래너의 현재 목표 노드 ID 가 같은 시간 비율 (둘 다 목표가 있는 구간 기준)
  - switch_delta     : 같은 노드로 넘어간 시각 차 (bt - simple) 통계
  - params_agreement : MPPIParams current_behavior_type 일치 비율 (발행 시점 기준 최근값)
  - escalation       : BLOCKED/CREEP/ASSIST 전이 시각 비교 (simple vs bt vs 녹화본)
사용: replay_compare.py <out.json> <duration_s>
"""
import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from command_center_interfaces.msg import MPPIParams, MultipleWaypoints, PauseCommand


class Rec(Node):
    def __init__(self, out, duration):
        super().__init__('replay_compare')
        self.out, self.duration = out, duration
        self.t0 = time.time()
        self.ev = {'simple': [], 'bt': [], 'rec': []}
        self.cur = {'simple': None, 'bt': None}
        self.ptype = {'simple': None, 'bt': None}
        self.agree_t = 0.0
        self.both_t = 0.0
        self.ptype_agree = 0.0
        self.ptype_both = 0.0
        self.last_tick = None
        for side, p in (('simple', ''), ('bt', '/bt')):
            self.create_subscription(MultipleWaypoints, p + '/multiple_waypoints',
                                     lambda m, s=side: self.wp_cb(s, m), 10)
            self.create_subscription(MPPIParams, p + '/mppi_update_params',
                                     lambda m, s=side: self.params_cb(s, m), 10)
            self.create_subscription(String, p + '/behavior_status',
                                     lambda m, s=side: self.log(s, 'status', m.data), 10)
            self.create_subscription(PauseCommand, p + '/pause_command',
                                     lambda m, s=side: self.log(s, 'pause', f'{m.node_id} {m.pause_duration:.1f}s {m.reason}'), 10)
        self.create_subscription(String, '/bt/behavior', lambda m: self.log('bt', 'behavior', m.data), 10)
        self.create_subscription(String, '/rec/behavior_status', lambda m: self.log('rec', 'status', m.data), 10)
        self.create_timer(0.1, self.tick)

    def t(self):
        return round(time.time() - self.t0, 2)

    def log(self, side, kind, text):
        self.ev[side].append((self.t(), kind, text))

    def wp_cb(self, side, m):
        gid = m.current_goal.header.frame_id
        if gid != self.cur[side]:
            self.cur[side] = gid
            self.log(side, 'target', f'{gid} idx={m.current_waypoint_index}/{m.total_waypoints} final={m.is_final_waypoint}')

    def params_cb(self, side, m):
        self.ptype[side] = m.current_behavior_type
        self.log(side, 'params', f'type={m.current_behavior_type} max_v={m.max_linear_velocity:.2f} '
                                 f'thr={m.goal_reached_threshold:.2f} desc={m.current_behavior_desc}')

    def tick(self):
        now = time.time()
        if self.last_tick is not None:
            dt = now - self.last_tick
            if self.cur['simple'] and self.cur['bt']:
                self.both_t += dt
                if self.cur['simple'] == self.cur['bt']:
                    self.agree_t += dt
            if self.ptype['simple'] is not None and self.ptype['bt'] is not None:
                self.ptype_both += dt
                if self.ptype['simple'] == self.ptype['bt']:
                    self.ptype_agree += dt
        self.last_tick = now
        if now - self.t0 >= self.duration:
            self.finish()
            raise SystemExit

    def finish(self):
        def targets(side):
            return [(t, x.split()[0]) for t, k, x in self.ev[side] if k == 'target']
        ts, tb = targets('simple'), targets('bt')
        deltas = []
        for t1, g in ts:
            for t2, g2 in tb:
                if g2 == g:
                    deltas.append(round(t2 - t1, 2))
                    break
        esc = {s: [(t, x) for t, k, x in self.ev[s] if k == 'status'] for s in ('simple', 'bt', 'rec')}
        summary = dict(
            duration=self.t(),
            simple_targets=[g for _, g in ts], bt_targets=[g for _, g in tb],
            target_sequence_equal=[g for _, g in ts] == [g for _, g in tb],
            target_agreement=round(self.agree_t / self.both_t, 4) if self.both_t else None,
            both_have_target_s=round(self.both_t, 1),
            switch_delta_bt_minus_simple=dict(
                n=len(deltas), mean=round(sum(deltas) / len(deltas), 2) if deltas else None,
                max_abs=round(max(abs(d) for d in deltas), 2) if deltas else None, values=deltas),
            params_type_agreement=round(self.ptype_agree / self.ptype_both, 4) if self.ptype_both else None,
            n_params={s: sum(1 for _, k, _ in self.ev[s] if k == 'params') for s in ('simple', 'bt')},
            n_pause={s: sum(1 for _, k, _ in self.ev[s] if k == 'pause') for s in ('simple', 'bt')},
            escalation=esc,
        )
        json.dump(dict(summary=summary, events=self.ev), open(self.out, 'w'), ensure_ascii=False, indent=1)
        print(json.dumps(summary, ensure_ascii=False, indent=1))


def main():
    out, dur = sys.argv[1], float(sys.argv[2])
    rclpy.init()
    n = Rec(out, dur)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    try:
        n.finish()
    except Exception:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
