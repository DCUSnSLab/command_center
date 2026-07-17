# SCV 실차 필드 테스트 Handoff

**대상**: 다른 장치(맥북 등)에서 이 작업을 이어받는 세션.
**목표**: 2차 실차 테스트 수행 + 로그 회수/분석. 이 문서만으로 접속부터 분석까지 가능하도록 작성.
**작성**: 2026-07-17 (직전 세션: ppub 연구실 PC).

---

## 1. 차량 접속 (Tailscale)

### 맥북 최초 1회
```bash
brew install --cask tailscale     # 또는 https://tailscale.com/download
# Tailscale 앱 실행 → 계정 junhp12345@ 로 로그인 (차량과 동일 계정)
tailscale status                  # scv-vehicle 이 목록에 보이면 성공
```
`macbookpro`(100.94.103.28)는 이미 이 tailnet에 등록돼 있어 로그인만 하면 됩니다.

### 접속
```bash
ssh scv@100.102.222.82            # MagicDNS: ssh scv@scv-vehicle
```
- 계정 `scv` / 비밀번호는 기존 장치 비밀번호 (팀 공유값, 여기 기재 안 함).
- 차량 tailscaled는 systemd enable — 전원만 켜면 자동 참여.
- **실외(테더링)에서는 추가 설정 불필요.** 차량 default 라우트가 LTE라 그대로 붙습니다.

### tailnet 노드
| 노드 | 주소 | 비고 |
|---|---|---|
| scv-vehicle (차량) | **100.102.222.82** | 필드 테스트 대상 |
| ppub-lab (연구실 PC) | 100.74.53.67 | 이전 세션 환경, 유선 |
| macbookpro | 100.94.103.28 | 이번 세션 |

### 접속 안 될 때
1. 차량 화면에서 `tailscale status` — `Logged out`이면 `sudo tailscale up --hostname scv-vehicle` 후 링크 승인.
2. **교내망 유선 상태라면** Fortinet이 `controlplane.tailscale.com`을 차단하므로 tailscale이 죽습니다.
   그 경우 교내망 IP로 직접 접속: `ssh scv@203.250.35.76`.
   (유선 default를 유지한 채 tailscale을 쓰려면 컨트롤플레인만 테더링으로 우회 — 휘발성:
   `for i in 101 102 103 104; do sudo ip route replace 192.200.0.$i via <테더GW> dev <테더IF>; done`)
3. 차량에 소스 기반 정책 라우팅 설치됨(`/etc/NetworkManager/dispatcher.d/90-scv-policy-routing`) —
   유선+테더링 동시 활성 시에도 교내망 인바운드 유지됨(2026-07-16 실측). 없으면 테더링 켜는 순간
   교내망 SSH가 끊깁니다(응답이 LTE로 새어 통신사에서 폐기).

---

## 2. LTE 데이터 수칙 (테더링 사용 중이므로 필수)

- **필터링은 차량 쪽에서**: `ssh scv@scv-vehicle '... | grep ... | tail -20'` — 원본 로그가 LTE를 건너지 않게.
- **금지**: 이미지/포인트클라우드 `ros2 topic echo`, bag 전송, 대용량 파일 복사.
- **bag은 현장에서 기록만** → 연구실 유선 복귀 후 업로드(§5).
- 진단은 텍스트 스냅샷(§4)으로 — 1회 4 KB~수백 KB.
- 이 세션(맥북)에서 Claude Code를 돌리면 **API 트래픽도 테더링을 탑니다**(턴당 수백 KB~).
  긴 디버깅·분석은 가능하면 연구실 복귀 후 유선에서.

---

## 3. 실차 테스트 실행

### 워크스페이스
차량의 **`~/SCV_park`** (원본 `~/SCV` 아님). 재빌드 시 `rm -rf build/<pkg> install/<pkg>` 후
`colcon build --packages-select <pkg> --symlink-install`, scripts/*.py 실행권한 확인.

### 시작 전 체크 (상세: 같은 디렉토리 `field_test_procedure.md`)
| 항목 | 확인 |
|---|---|
| **시작 위치** | 차량을 **지도 경로 위**에 둘 것. 사용 지도(`record_20260630_141709_map_d1.json`)의 경로는 **공학관 순환 루프** — 동쪽 운동장 인도가 아님. 2026-07-14 실패의 부차 원인 |
| velodyne 링크 | `ip -br a \| grep enp7s0` → UP. DOWN이면 C3가 전면 lethal 발행해 출발 차단(정상 동작) |
| can0 | `ip -br link \| grep can0` — DOWN이어도 hunter_base launch가 자동으로 올림 |
| 조종기 | teleop mux가 유일한 즉시 개입 수단. 배터리/페어링 확인 |

### 실행
```bash
source ~/SCV_park/install/setup.bash
ros2 launch command_center_launch field_drive.launch.py
```

### 기동 후 확인 (하나라도 미달 시 자율주행 보류)
```bash
ros2 node list                                    # 위치추정 5종 + curb/costmap/corridor/mppi/behavior
ros2 topic hz /velodyne_points /vectornav/imu /ublox_gps_node/fix /hunter/velocity
                                                  # 10 / 100 / 10 / ~50 Hz — velocity가 0이면 hunter_base 확인
ros2 topic echo /ublox_gps_node/fix --once        # status: 2 (RTK fixed) 아니면 NTRIP부터 해결
ros2 topic echo /odom --once                      # 정지 상태에서 즉시 나와야 함
```
gps_fix_gate 로그에 `first sane fix passed`가 떠야 합니다. `fix DROPPED`가 1분 이상 지속되면 수신기 재부팅.

### 시나리오 (앞 단계 통과 후 진행, 항상 조종기 대기)
| # | 내용 | 기대 (시뮬 예측치) |
|---|---|---|
| A | 개활 직선 주행 | 정상 추종 |
| B | 경로 위 6~7 m 앞 장애물 | **장애물 1.5~1.7 m 앞 인지 정지** (Gazebo: 5 m 차량 앞 3.34~3.38 m) |
| C | 장애물 유지 | 정지 4 s → `BLOCKED_WAIT` → 12 s → `CREEP`(0.3 m/s) → 10 s → `ASSIST`, 치우면 `NORMAL` 복귀. `ros2 topic echo /behavior_status` |
| D | **연석 옆에서 회피 유도 (핵심)** | 연석 쪽으로 **절대 나가지 않음**, 회피 불가 시 C 시퀀스. RViz `/costmap_keepout`에서 차도가 lethal인지 병행 확인 |
| E | (선택) 저속 주행 중 `pkill -9 -f curb_detection_node` | **0.8 m 내 정지** (Gazebo 실증 0.82 m). respawn으로 자동 복귀 |
| F | **GPS 열화 구간(수목 아래) 통과 — 마지막 미검증 항목** | FAST-LIO 주도로 경로 유지, `/odometry/global` 점프 후 자연 수렴 |

### 즉시 수동 개입
- 연석 방향 회피 시작 / RTK 상실 상태에서 경로이탈 >1 m / BLOCKED인데 계속 미는 거동

### 기록
```bash
ros2 bag record /velodyne_points /vectornav/imu /ublox_gps_node/fix /hunter/velocity \
  /odom /odometry/global /odometry/gps /odometry/fast_lio /gps/fix_gated /tf /tf_static \
  /front/scan /rear/scan /costmap /costmap_keepout /velodyne_points_curb /cmd_vel \
  /behavior_status /blocked_assist_request /planned_path_detailed /multiple_waypoints \
  /goal_status /camera/camera/color/image_raw \
  -o ~/bags/field_$(date +%Y%m%d_%H%M%S)
```
시나리오마다 bag을 끊어 주면 분석이 쉬움.

---

## 4. 진단 스냅샷 (문제 발생 시 반드시)

```bash
bash ~/SCV_park/src/command_center/command_center_launch/scripts/collect_field_logs.sh
# → ~/field_logs/scv_diag_<timestamp>.tar.gz (텍스트 ~4 KB~수백 KB, LTE 안전)
```
**스택이 떠 있는 상태에서 실행**하면 ROS 그래프(토픽 Hz/TF/GPS/위치추정 정합)까지 채워집니다.
차량을 움직이는 동작은 전혀 없습니다. 회수:
```bash
scp scv@100.102.222.82:~/field_logs/scv_diag_*.tar.gz .
```
담기는 것: 코드 신원(브랜치+미커밋 변경) / 오늘 launch 로그의 `process has died`+중복제거 에러 /
can0·velodyne 링크 실측 / 라이브 토픽·TF·GPS / bag 인벤토리 / 시스템·USB·dmesg.

---

## 5. 복귀 후 (연구실 유선)

```bash
bash ~/SCV_park/src/command_center/command_center_launch/scripts/upload_bags.sh   # 오늘 bag 자동 선택
# 또는 upload_bags.sh ~/bags/field_2026...
```
BagArchive `http://203.250.35.87:31447` 에 업로드 + 재색인. **LTE에서 실행 금지** (bag은 GB 단위).
업로드 후 시뮬 예측치와 정량 비교(정지거리·에스컬레이션 타이밍·연석 침범 0)가 다음 분석 과제.

---

## 6. 현재까지의 진행상황

### 완료 — 소프트웨어 검증 사다리 전체
| 항목 | 결과 |
|---|---|
| 4계층 연석 방어 (L1 연석검출 / L2 회랑 keepout / L3 BLOCKED_WAIT / L4 MPPI 보강) | 개루프 30-bag 회귀 통과 |
| 치명 결함 C1(무발행 기아)·C2(전궤적 lethal 폭주)·C3(코스트맵 신선도) | 전부 수정. C3는 Gazebo 고장주입 실증(인지 사망 → 0.82 m 내 정지) |
| Frozen-world 폐루프 33월드 | 침범 0 |
| Gazebo 물리 폐루프 33월드 (인지 인루프) | 침범 0·추락 0, 위치추정 RMS 중앙값 2.2 cm |
| tiny_localization → robot_localization dual-EKF + FAST-LIO | bag 검증 완료, datum = 지도 node[0] (35.91361, 128.80308) |
| 리포트 | https://claude.ai/code/artifact/edfb1737-efa0-48a3-bcf0-3e6df9cc2d0f |

### 2026-07-14 1차 실차 시도 — 실패, 원인 2건 모두 수정·실차 검증 완료
1. **hunter_base SIGABRT** — can0가 DOWN이면 ugv_sdk가 0.2초 만에 abort. 구동·휠속도 전무.
   → launch에 can0 자동 브링업(sudoers.d/scv-can0) + 2 s 지연 + respawn. 실차 검증: can0 down →
   launch → `/hunter/velocity` 50 Hz.
2. **맵 EKF −168 km 발산** — 수신기 콜드스타트 쓰레기 좌표(110 km 밖, status 0인데 공분산은 정상)가
   무거부 설계의 맵 EKF에 첫 관측으로 유입 + 휠속도 부재로 속도 상태 폭주.
   → `gps_fix_gate.py`: datum 5 km 밖/비정상 fix를 navsat 앞에서 차단. 필드 bag 재생 A/B —
   게이트 없음 121 km 발산 / 있음 60/60 차단 후 정상 수렴. 실기기 스모크 통과.
- 당일 bag 3개 BagArchive 업로드 완료 (id 77 / 79 / 81).

### 확인 필요 (팀의 미커밋 변경, 제 작업 아님)
- `costmap_params.yaml`: **`min_obstacle_height` 0.5 → 0.15** — 연석 단차(0.26 m)보다 낮아
  **연석이 일반 장애물로도 잡힘**. 안전 방향이나 좁은 인도에서 BLOCKED 빈발 가능. 의도 확인 필요.
- `system.launch.py` 기본 맵 = `1_5_map hard_relocated.json` (UtmInfo 없음 → 구 스택에서 waypoint 붕괴).
  `field_drive.launch.py`는 올바른 맵 사용하므로 실차 절차엔 영향 없음.

### 저장소/브랜치
| 저장소 | 브랜치 | 최근 커밋 |
|---|---|---|
| DCUSnSLab/SCV_Perception | feature/curb-safety | c702620 |
| DCUSnSLab/command_center | feature/curb-safety | 07b105f |
| nevlife/robot_localization | feature/scv-localization | 3de0d4f |
| DCUSnSLab/hunter_ros2 | feature/field-hardening | bc6e8de |

푸시 주의: ppub의 gh 토큰(junhp1234)이 2026-07-16 만료 → 로컬 푸시 불가, 차량(gh=nevlife) 경유로
푸시해 옴. 맥북에서 푸시하려면 해당 장치 gh 인증 필요. nevlife/robot_localization은 junhp1234
계정으로 403이므로 차량에서 푸시.

### 남은 일
1. **2차 실차 테스트** (§3) — 특히 시나리오 D(연석 비침범)와 **F(GPS 열화, 마지막 미검증 항목)**
2. bag 회수 → 유선에서 업로드 → 시뮬 예측치와 정량 비교
3. 선택: 전·후방 RPLIDAR C1 스캔의 local_costmap 융합 (설계 미착수)
4. 선택: 3개 저장소 PR 생성 (사용자 승인 대기)
