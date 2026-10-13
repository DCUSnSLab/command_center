#!/usr/bin/env python3
"""BT.CPP FileLogger2(.btlog) → CSV / 요약.

용량: 결정 트리는 틱마다 전 노드가 IDLE↔상태 를 오가므로 20 Hz 에서 ~4 MB/10 min (Pod 실측 322 s 1.3 MB).

형식(bt_file_logger_v2.cpp): 'BTCPP4-FileLogger2' | uint8 version | int32 xml_len | xml(_uid 포함) | int64 first_ts_usec
| 전이 9바이트 × N = (uint48 t_usec 상대, uint16 node_uid, uint8 status)
status: 0 IDLE, 1 RUNNING, 2 SUCCESS, 3 FAILURE, 4 SKIPPED

사용:
  btlog_to_csv.py run.btlog [-o run.csv] [--summary] [--ticks]
    --summary : 노드별 SUCCESS/FAILURE 횟수
    --ticks   : 루트 노드 전이를 기준으로 틱마다 SUCCESS 였던 노드 경로를 한 줄로
"""
import argparse, csv, struct, sys
import xml.etree.ElementTree as ET

STATUS = {0: 'IDLE', 1: 'RUNNING', 2: 'SUCCESS', 3: 'FAILURE', 4: 'SKIPPED'}
MAGIC = b'BTCPP4-FileLogger2'


def read_btlog(path):
    b = open(path, 'rb').read()
    if not b.startswith(MAGIC):
        sys.exit(f'{path}: FileLogger2 파일이 아님')
    i = len(MAGIC)
    version = b[i]; i += 1          # 프로토콜 버전 1바이트 (4.10: 1)
    (n,) = struct.unpack_from('<i', b, i); i += 4
    xml = b[i:i + n].decode('utf-8'); i += n
    (first_ts,) = struct.unpack_from('<q', b, i); i += 8
    trans = []
    while i + 9 <= len(b):
        t = int.from_bytes(b[i:i + 6], 'little'); uid = int.from_bytes(b[i + 6:i + 8], 'little'); st = b[i + 8]
        trans.append((t, uid, st)); i += 9
    return xml, first_ts, trans


def node_table(xml):
    """uid → (이름, 태그, 주요 속성)."""
    root = ET.fromstring(xml)
    table = {}
    for el in root.iter():
        uid = el.get('_uid')
        if uid is None:
            continue
        attrs = {k: v for k, v in el.attrib.items() if not k.startswith('_') and k != 'name'}
        label = el.get('name') or el.tag
        table[int(uid)] = (label, el.tag, attrs)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('btlog'); ap.add_argument('-o', '--out'); ap.add_argument('--summary', action='store_true')
    ap.add_argument('--ticks', action='store_true')
    a = ap.parse_args()
    xml, first_ts, trans = read_btlog(a.btlog)
    tab = node_table(xml)
    out = a.out or a.btlog.rsplit('.', 1)[0] + '.csv'
    t0 = trans[0][0] if trans else 0
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_rel_s', 't_usec_since_first', 'node_uid', 'node', 'type', 'status', 'attrs'])
        for t, uid, st in trans:
            label, tag, attrs = tab.get(uid, ('?', '?', {}))
            w.writerow([f'{(t - t0) / 1e6:.3f}', t, uid, label, tag, STATUS.get(st, st),
                        ' '.join(f'{k}={v}' for k, v in attrs.items())])
    print(f'{out}: 전이 {len(trans)}건, 노드 {len(tab)}개, 첫 시각(epoch us) {first_ts}, 길이 {(trans[-1][0]-t0)/1e6:.1f} s' if trans else f'{out}: 전이 없음')
    if a.summary:
        cnt = {}
        for _, uid, st in trans:
            if st in (2, 3):
                cnt.setdefault(uid, [0, 0])[0 if st == 2 else 1] += 1
        print(f'{"node":34s} {"type":20s} {"SUCCESS":>8s} {"FAILURE":>8s}')
        for uid, (s_, f_) in sorted(cnt.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
            label, tag, attrs = tab.get(uid, ('?', '?', {}))
            if attrs:   # 같은 이름의 노드(ZoneIs·IsTurn·Drive·Sequence)를 포트 값으로 구분
                label = f"{label}[{' '.join(f'{k}={v}' for k, v in attrs.items())}]"
            print(f'{label:34s} {tag:20s} {s_:8d} {f_:8d}')
    if a.ticks:
        # 루트(가장 작은 uid) 가 IDLE→(SUCCESS|FAILURE) 로 끝나는 지점을 틱 경계로 본다
        root_uid = min(tab) if tab else None
        cur, prev_sig = [], None
        for t, uid, st in trans:
            if st == 2:
                cur.append(tab.get(uid, ('?',))[0])
            if uid == root_uid and st in (2, 3):
                sig = (STATUS[st], tuple(cur))
                if sig != prev_sig:
                    print(f'{(t - t0) / 1e6:9.3f}s root={STATUS[st]:7s} SUCCESS 경로: ' + ' > '.join(cur))
                    prev_sig = sig
                cur = []


if __name__ == '__main__':
    main()
