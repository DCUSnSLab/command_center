# SCV 실차 테스트 절차서 — 연석 안전 스택 + 신규 위치추정

기준: 2026-07-09 코드 상태
(연석 4계층 방어 + robot_localization dual-EKF/FAST-LIO 전환 완료 시점)

- 실행: `ros2 launch command_center_launch field_drive.launch.py`
- 브랜치: SCV_Perception / command_center `feature/curb-safety`,
  robot_localization `feature/scv-localization`
- 사전 검증 이력: 개루프 30-bag 회귀, 실노드 e2e, 폐루프 33-월드(침범 0),
  위치추정 bag 재생 검증 — **미검증 잔여 = 실차 폐루프뿐**

---

## 0. 출발 전 (연구실)

| # | 항목 | 명령/기준 |
|---|---|---|
| 0-1 | 브랜치·빌드 확인 | 세 저장소가 위 브랜치인지: `git -C src/perception branch --show-current` 등. 재빌드 시 사본 캐시 주의: `rm -rf build/<pkg> install/<pkg>` 후 빌드 |
| 0-2 | 스크립트 실행권한 | `--symlink-install` 사용 시 `chmod +x src/**/scripts/*.py` (미적용 시 "No executable found") |
| 0-3 | 지도-datum 일치 | 기본 지도 `record_20260630_141709_map_d1.json`. **다른 지도 사용 시** ① UtmInfo 필드 존재 확인(없으면 waypoint 전부 0) ② `scv_dual_ekf.yaml`의 `datum:`을 그 지도 node[0]의 Lat/Long으로 변경 |
| 0-4 | NTRIP 계정/망 | ntrip_client 설정 및 통신망(SIM/테더링) 확인 — 2026-07-14 실측: 국토지리정보원 VRS 접속 성공(계정 `nevlife12` 유효). **기준 bag은 전 구간 RTK 미고정 상태로 주행했음** |
| 0-5 | 안테나 (AKA900 RTK로 교체됨) | URDF `gnss_antenna` = base_link 기준 (+0.02, 0, +0.795) — 2026-07-14 실기기 TF 실측 일치 확인. **안테나 마운트를 물리적으로 옮겼다면 tf_mounts.xacro 갱신 + 2-2 회전 시험 필수** |
| 0-6 | CAN (can0) | 부팅 후 can0는 DOWN이 기본. hunter_base.launch가 자동으로 올림(sudoers.d/scv-can0 필요, 2026-07-14 설치됨). 수동 확인: `ip -br link \| grep can0` → UP. **2026-07-14 필드런은 can0 미기동 → hunter_base 즉사(SIGABRT) → 구동·휠속도 전무가 1차 실패 원인** |
| 0-7 | 조종기 | teleop 조종기 배터리/페어링 — mux 수동 개입이 유일한 즉시 개입 수단 |
| 0-8 | 시작 위치 | **차량을 지도 경로 위(node 인접)에 두고 시작.** 사용 지도 경로는 공학관 순환 루프임(동쪽 운동장 인도 아님 — 2026-07-14 위치 착오 재발 방지) |

## 1. 현장 기동 (자율주행 전, 정지 상태)

`field_drive.launch.py` 실행 후 순서대로:

| # | 확인 | 명령 | 합격 기준 |
|---|---|---|---|
| 1-1 | 노드 생존 | `ros2 node list` | 위치추정 5종(fastlio, wheel_odom_adapter, ekf×2, navsat) + curb/costmap/corridor/mppi/behavior 존재, "process has died" 없음 |
| 1-2 | 센서 스트림 | `ros2 topic hz /velodyne_points /vectornav/imu /ublox_gps_node/fix /hunter/velocity` | 10 / ~100 / ~10 / ~50 Hz |
| 1-2b | 신규 2D LiDAR (RPLIDAR C1 전·후방) | `ros2 topic hz /front/scan /rear/scan` | 각 ~10 Hz. **현재 안전 스택은 이 스캔을 소비하지 않음**(코스트맵 융합은 후속 작업) — 검출 실패 시에도 주행 가능하나 기록 권장 |
| 1-2c | velodyne 링크 | `ip -br a \| grep enp7s0` | `UP` — 2026-07-14 실기기 점검에서 DOWN(전원/케이블 분리) 상태였음. 미연결 시 C3가 전면 lethal을 발행해 출발 자체가 차단됨(정상 페일세이프) |
| 1-3 | **RTK Fix** | `ros2 topic echo /ublox_gps_node/fix --once` | `status.status: 2` 또는 covariance 대각 < 0.01 (1σ<10cm). **미달 시 자율주행 보류** — NTRIP부터 해결 |
| 1-4 | TF 트리 | `ros2 run tf2_tools view_frames` | map→odom→base_link 단선 연결, base_link→gnss_antenna는 **URDF 1개만** (launch 폴백 off 확인) |
| 1-5 | 위치추정 초기화 | `ros2 topic echo /odom --once` | **정지 상태에서 즉시** 출력 (구 tiny와 달리 주행 불필요). `/odometry/global`도 확인 |
| 1-5b | **GPS 게이트 통과** | gps_fix_gate 로그 | `first sane fix passed` 확인. 콜드스타트 쓰레기 좌표(→2026-07-14 EKF −168 km 발산 원인)는 게이트가 자동 차단하지만, `fix DROPPED`가 1분 이상 지속되면 수신기 재부팅 |
| 1-5c | 전역 위치 정합 | `/odometry/global`과 `/odometry/gps` 비교 | 두 좌표 차이 < 5 m (발산 잔재 없음) |
| 1-6 | 연석 검출 | curb 노드 로그 | `plane a=…, c=−0.7~−0.95` 범위, "passthrough" 경고 지속되면 전방 개활 방향으로 차량 회전 |
| 1-7 | 코스트맵 | RViz: `/costmap_keepout` | **차도 영역이 lethal(적색)**, 인도 회랑만 free. 회랑이 경로 따라 형성되는지 |

## 2. 캘리브레이션 (첫 방문 시 1회)

**2-1. navsat yaw_offset**: 개활지에서 로봇을 **정동(east)** 방향으로 정렬(원거리 지형지물/지도 기준) 후
`ros2 topic echo /vectornav/imu --field orientation` → yaw 환산값이 0이 아니면 그 차이를 `scv_dual_ekf.yaml`의 `yaw_offset`에 입력.
검증: 직선 10m 수동 주행 → `/odometry/global` 궤적이 실제 진행 방향과 일치하는지.

**2-2. (선택) 안테나 오프셋 확인**: RTK fix 상태에서 제자리 360° 회전 → `/odometry/gps` 위치가 원을 그리면 URDF 오프셋 오차 (반지름=오차). URDF값 (+0.02, 0, 0.795)이 이미 검증되어 있어 확인용.

## 3. 수동 주행 검증 (mux 수동 모드)

| # | 시나리오 | 확인 |
|---|---|---|
| 3-1 | 직선 20m 왕복 | `/odom` 매끈(점프 없음), RViz에서 costmap이 차량 따라 롤링, 연석 lethal 유지 |
| 3-2 | 제자리 회전 | TF/odometry 안정, FAST-LIO 발산 없음 |
| 3-3 | **mux 개입 테스트** | 자율 모드 전환 준비 상태에서 조종기 입력이 항상 우선하는지 — **이후 모든 단계의 전제조건** |

## 4. 자율주행 — 점진 시나리오 (각 단계 합격 후 다음으로)

안전요원 1인 조종기 파지, 1인 차량 측방 동행. 최초엔 `max_linear_velocity`를 0.5로 제한 권장 (`smppi_params.yaml`).

| 단계 | 시나리오 | 합격 기준 | 관찰 토픽 |
|---|---|---|---|
| A | 개활 직선 10m | 목표 도달, 경로이탈 <0.5m | `/odom`, `/goal_status` |
| B | 연석 인접 구간 직진 | **연석 방향 접근 없음** (폐루프에서 ±1cm였음), 차도 셀 lethal 상시 | RViz costmap |
| C | 인도 위 정적 장애물(박스) — 회랑 내 회피 여유 있게 배치 | 회랑 안에서 회피, **연석 쪽 이탈 금지** | `/mppi_optimal_path` |
| D | **회랑 전폭 차단** (안전요원 2인이 길 막기) | 정지 → `BLOCKED_WAIT` → (12s) `CREEP` → 비켜주면 `NORMAL` 복귀·재주행 | `/behavior_status` |
| E | D에서 계속 차단 유지 | (10s creep 후) `ASSIST` 발행 → **연석 침범 없이 대기 지속** | `/blocked_assist_request` |
| F | GPS 열화 구간(수목 아래) 통과 | FAST-LIO 주도로 경로 유지, `/odometry/global` 점프 후 자연 수렴 | 위치추정 토픽들 |

## 5. 즉시 중단 기준 (조종기 개입 → 수동 회수)

- 차량이 **연석 30cm 이내** 접근 또는 차도 방향 조향 지속
- mppi 로그 `ALL trajectories lethal` 1초 이상 반복 (비상정지 가드 작동 중 — 원인 파악 전 재개 금지)
- `/odom` 1m 이상 점프, TF 끊김(RViz 프레임 적색), 노드 반복 재시작(respawn 루프)
- RTK 상실 + F 시나리오 외 구간에서 경로이탈 >1m

## 6. 기록 (전 세션 상시)

```bash
ros2 bag record /velodyne_points /vectornav/imu /ublox_gps_node/fix /hunter/velocity \
  /odom /odometry/global /odometry/gps /odometry/fast_lio /gps/fix_gated /tf /tf_static \
  /front/scan /rear/scan \
  /costmap /costmap_keepout /velodyne_points_curb /cmd_vel /behavior_status \
  /blocked_assist_request /planned_path_detailed /multiple_waypoints /goal_status \
  /camera/camera/color/image_raw
```
종료 후 BagArchive(203.250.35.87:31447) 업로드 → 회귀 하니스로 재검증 가능.

## 7. 알려진 특성 / 튜닝 후보 (이상 아님)

- BLOCKED↔NORMAL 채터링 가능 (경계 진동) — 알림만 영향, 필요 시 `blocked.progress_eps` 상향
- MPPI가 장애물 앞에서 완전정지 대신 저속 크리프 → BLOCKED 감지가 수 초 지연될 수 있음
- 게이트(횡단) 기능은 지도에 node_type 10이 없어 현재 비활성 — 횡단 노선 제작 시 지정
- gps_manager 미사용 중 — RTK 두절 드리프트 >3m 관측 시 도입 검토
