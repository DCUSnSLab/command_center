# SCV 필드 테스트 절차 (2026-08-06 판)

8/4 실외 전 시나리오 반대주행의 근본 원인(VectorNav yaw가 자기 기준 없이
전원 시점부터 자이로 적분 → 세션마다 임의 오프셋: 8/3 +62°, 8/4 +170°)을
반영한 운용 절차다. 지금까지 반영된 변경:

| 변경 | 내용 | 상태 |
|---|---|---|
| yaw 자동 보정 (autocal) | map_anchor가 주행 중 [GPS 실이동 방위 − IMU yaw]를 추정해 자가 교정. **자율 전환 전 RC 전진 보정 주행이 필수 게이트** | 배포 완료 (bag·챔버 검증) |
| 자기 앵커 복원 (mag cal) | VectorNav HSI 보정 + ABSOLUTE 모드 전환으로 절대 방위 복원 | **8/5 수행 완료** (NV 저장, 8/6 전원 재투입 재현성 확인). 재수행 절차는 §5 |
| 지도 존재 확인 | `field_bringup.sh` 가 `map_file_path` 부재 시 중단 — 8/5 무단 대체로 목표가 13 m 어긋난 사고 대책 | 배포 완료 |
| 경로 합류 규칙 | 최근접 → 비용 기반(접근×1.5 + 잔여). 챔버 A/B 검증 | 배포 완료 |
| 합류 노드 재선택 | **자율 전환 순간**과 RC 수동 주행 2 m 마다 다시 고른다. yaw 게이트(RC 전진)가 합류 노드를 낡게 만들던 구멍 — 8/6 오전 실패 → 오후 재현 성공(68.8 m → 16.1 m) | 배포 완료 |
| 카메라 제외 기록 | `field_bringup.sh --no-camera` — 실측 4.8 GB/분의 90 % 이상이 카메라. 실내→실외 장거리 이동 기록에 사용 | 배포 완료 |

---

## 0. 사전 조건

- 개활지(RTK fixed 잡히는 곳) 포함 경로. RTK는 8/4에 status 2(σ≈2cm) 재확인됨.
- 접속: 필드 중 `scv-field`(점프 경유), 랩 복귀 후 직결. 테더링 경로에서는
  텍스트만 — bag 업로드는 랩 유선에서.
- 차량 파일 위치: 절차 스크립트 `/home/scv/field_magcal/`,
  기동 정본 `~/SCV_park/src/command_center/command_center_launch/scripts/`.

## 1. 공통 기동

```bash
cd ~/SCV_park/src/command_center/command_center_launch/scripts
./field_bringup.sh --record \
  map_file_path:=/home/scv/MAP/D2/d2_unha.json \
  route_source:=sequential goal_node:=N026        # (예: 8/4 주차장 구석 시나리오)
```

- `--record` 는 스택 안정 후 bag 기록 시작. `/map_anchor/yaw_corr` 포함(8/5 추가).
- 실내→실외 이동처럼 긴 구간을 기록할 때는 `--no-camera` 를 함께 준다
  (카메라 원본이 bag 용량의 90 % 이상이다 — 30분 이동 145 GB → 10 GB 남짓).
- `map_file_path` 의 파일이 없으면 기동이 **중단**된다. 다른 지도로 대체하지
  말 것 — 목표 노드가 달라진다(8/5 사고).
- 정지는 반드시 `./teardown.sh` (조상-안전 kill). 기록만 끊을 때는 bag record
  프로세스만 kill.

## 2. ★ yaw 보정 게이트 (매 기동마다, 자율 전환 전 필수)

기동 직후의 map yaw는 신뢰할 수 없다(전원 시점 기준 임의값). 보정 없이 자율
전환하면 8/4처럼 앵커 처닝 교착(제자리 지그재그)에 빠지고, 그 상태에선 보정
주행 자체가 생기지 않는다는 것까지 챔버에서 재현·확인됐다.

1. **RC 수동으로 전진 ~10 m** (RTK fixed면 ~5 m면 충분, 직선일 필요 없음 —
   완만한 곡선 가능, 후진은 카운트 안 됨).
2. 보정 발동 확인 (둘 중 하나):
   ```bash
   ros2 topic echo /map_anchor/yaw_corr          # 값이 발행되기 시작하면 측정 중
   grep "yaw autocal" ~/field_*/field_drive.log | tail -3
   ```
   `yaw autocal ENGAGED: th rotated ...` 워닝이 뜨면 게이트 통과.
3. ENGAGED 확인 후 자율 전환.

주의: 자기 앵커 복원(4절)이 성공적으로 끝난 뒤의 세션부터는 이 게이트를
축소(확인만)할 수 있으나, **검증 주행 2~3회 전까지는 유지**한다.

## 3. 자율 주행 및 모니터링

- 목표 전송(기동 인자로 안 준 경우):
  `ros2 topic pub -r 2 -t 4 /goal_node_id std_msgs/String "{data: 'N026'}"`
  (`--once`는 디스커버리 유실 이력 있음 — `-r 2 -t N` 사용)
- 모니터링 포인트:
  - `/map_anchor/mode` — GPS_RTK/GPS_SUSPECT/PCD 전환
  - `/map_anchor/yaw_corr` — 보정값 안정성(주행 중 수 도 이내 미세 조정이 정상)
  - `/hunter_status.control_mode` — 1=CAN(자율)/3=RC. **RC 링크 플래핑**(8/4에
    33회/60초 관측) 시 자율 창이 수 초로 쪼개진다 — 조종기 안테나/거리 주의
  - behavior 로그의 `[MODE]`, `[BLOCKED]`, TURNAROUND 발동 여부

## 4. ★ 자기 앵커 복원 (이번 필드 특별 작업 — 개활지에서 1회)

RTK 잡히는 개활지에서, 다른 시나리오와 독립적으로 수행한다.

**2026-08-05 1회 수행 완료** — 아래는 그때 실제로 통한 경로다. 드라이버의
`mag_cal` 액션은 **믿지 말 것**: 수렴 판정이 "값이 변하지 않음"이라 정지
상태면 21표본(11초)에서 무조건 성공 처리되고, **검증 없이 USEONBOARD +
NV 영구저장**까지 해버린다(0/항등 보정이 그대로 박힐 수 있다). 액션은
ABSOLUTE 전환용으로만 쓰고, HSI 는 직접 구동한다.

```bash
cd /home/scv/field_magcal

# (1) 스택/드라이버 정지 상태 — 보정 전 레지스터 기록
./magcal_run.sh pre

# (2) 드라이버(또는 스택) 기동 후 — ABSOLUTE 전환 (액션의 유일한 쓸모)
./magcal_run.sh cal        # 11초 만에 끝나고 결과가 0 이어도 정상 진행

# (3) HSI 직접 구동 — RC 저속(≤0.5 m/s) 원/8자 주행하며 관찰
python3 hsi_run.py 240     # reg47 이 '비항등!' 로 바뀌고 값이 멎으면 수렴

# (4) 적용 + 영구 저장
python3 hsi_apply.py       # reg44=0,3 / reg35 ABSOLUTE / VNWNV

# (5) 드라이버 정지 후 3판정 (ABSOLUTE / USEONBOARD / 비항등)
./magcal_run.sh post

# (6) 스택 재기동 후 검증 주행 → bag 으로 정밀 판정
python3 heading_eval2.py /home/scv/field_<날짜>/bag
```

**판정 (heading_eval2, 반드시 엔코더 부호로 전/후진 게이트)**:
- 전진 구간 오프셋의 잔차 RMS < 15° 면 세션 내 상수로 본다 → autocal 흡수 가능.
- **후진 구간은 방위가 정확히 180° 반전**한다. 게이트하지 않으면 "방향 의존
  오차"로 오판한다(8/5에 실제로 한 번 오판했다).
- 라이브 `heading_check.py` 는 참고용이다 — 짧은 변위·비RTK·후진 혼입으로
  ±100° 씩 출렁인다. 판정은 bag 으로 한다.

**8/5 실측 결과와 남은 한계** (다음 수행 시 기대치 조정용):
- 세션 내부는 안정적(잔차 RMS 4.1~4.3°)이나 **세션 간 span 23.6°**,
  같은 5 m 격자에서도 30분간 **11~21° 배회**했다. "±5° 상수"에는 못 미친다.
- ABSOLUTE 는 새 실패 모드를 들여온다 — 국소 자기 왜곡이 헤딩에 즉시
  주입된다(정지 중 −30° 슬루 관측). **2절 yaw 게이트를 축소하지 말 것.**
- 소프트아이언 미보정(reg47 비대각 전부 0). 더 느리고 넓은 원 주행으로
  재추정 여지가 있다.
- **미검증**: 전원 순환 후 같은 절대 방위로 복귀하는가. 다음 필드에서
  동일 지점·동일 헤딩으로 전원 껐다 켜고 재측정할 것 — 이게 진짜 합격선이다.
- 롤백: 드라이버 정지 후 시리얼로 reg35 headingMode=1 재기록(RELATIVE 복귀).
  기존 autocal 경로가 그대로 유효하다.

## 4-B. LiDAR-관성 오도메트리(LIO) 교체 — 선택 실험

세 구현이 **같은 계약**(`/odometry/fast_lio` 로 odom 프레임 Odometry 발행,
odom→base_link TF 발행 금지 — EKF 소유)을 지키도록 배선돼 있어 인자 하나로
바꾼다. 상위 스택(앵커·행동·제어)은 `/odom` 만 보므로 무영향.

```bash
./field_bringup.sh --record lio_source:=fastlio     # 기본(현장 검증됨)
./field_bringup.sh --record lio_source:=fasterlio   # Faster-LIO (iVox)
./field_bringup.sh --record lio_source:=rko         # RKO-LIO (IMU 느슨결합)
```

| 구현 | 특징 | 주의 |
|---|---|---|
| FAST-LIO2 | iEKF + ikd-Tree, 현장 이력 다수 | 기본값 — 실주행 시나리오는 이걸로 |
| Faster-LIO | 같은 iEKF, iVox 로 근방탐색 가속 | velodyne time 단위 `time_scale: 1.0`(초) 확인 |
| RKO-LIO | 센서별 모델링 없음, IMU 느슨결합 | 온라인 모드에서 처리가 밀리면 IMU 큐 넘침→영구 락아웃 이력. CPU 여유 확인 필수 |

교체 후에는 **반드시 2절 yaw 게이트부터 다시** 수행한다(오도메트리 소스가
바뀌면 앵커의 th 추정 입력이 바뀐다).

## 5. 성공 판정 기준 요약

| 항목 | 기준 |
|---|---|
| yaw 게이트 | RC 전진 ≤10 m 내 ENGAGED 로그 |
| 자율 주행 | 목표 방향으로 접근(거리 단조 감소), TURNAROUND 오발 없음 |
| mag cal | post 3판정 OK + heading_check 오프셋 상수(±5° 이내 변동) |
| 위치 품질 | RTK fixed 구간에서 /odometry/global 이 실궤적과 일치 |

## 6. 이상 대응

| 증상 | 조치 |
|---|---|
| 목표 반대/사선 주행 | 즉시 RC 전환(하드웨어 우선권). ENGAGED 이전 자율 전환 여부 확인 — 2절 게이트 재수행 |
| 제자리 지그재그 + waypoint 재발행 반복 | yaw 미보정 서명. RC로 빼내서 전진 보정 주행 |
| autocal 의심(보정값 이상) | `field_bringup.sh anchor_yaw_autocal:=false` 로 재기동(롤백 스위치) 후 증상 비교 |
| RC 플래핑 (control_mode 3↔0↔1) | 조종기 거리/안테나 확인. 자율 창이 수 초면 주행 판정 불가 |
| GPS 열화(cov 급증) | 앵커가 EMA 감속·슬루 제한으로 버팀. /map_anchor/mode 관찰 |

## 7. 종료 및 복귀

1. 로깅만 종료: bag record 프로세스 kill (스택 유지 시).
2. 전체 종료: `./teardown.sh`.
3. 랩 복귀(유선) 후: bag 업로드 → 아카이브 바이트 대조 검증 → 차량 사본 삭제.
4. 세션 리포트: yaw_corr 수렴값(=그 세션의 오프셋), mag cal 전후 비교,
   RTK 비율을 기록해 두면 다음 분석이 빨라진다.
