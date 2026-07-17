# SCV 원격 작업 Handoff (2026-07-16)

새 환경(개인 맥북 + 테더링)에서 작업을 재개하기 위한 접속 가이드와 진행상황 요약.

## 1. Tailscale 연결 가이드

### 배경 (왜 이 구성인가)
- 교내망은 Fortinet 방화벽이 `controlplane.tailscale.com`을 TLS 가로채기로 **차단** →
  교내 유선에 물린 기기(ppub 포함)는 tailscale 로그인 불가.
- 차량이 USB 테더링(LTE)이면 차단 없음. 맥북도 테더링/외부망이면 없음.
- 따라서 tailnet 구성원은 **차량(scv) + 맥북** 두 대. ppub은 교내망에 있는 동안 참여 불가.
- 주의: 차량에서 테더링을 켜면 기본 라우트가 usb0으로 넘어가 **교내망 IP(203.250.35.76)
  경로가 끊긴다** (2026-07-16 실측). 테더링 중 원격 접근은 tailscale이 유일 경로.

### 구성 완료 상태 (2026-07-16 검증)
tailnet 계정: junhp12345@ (개인 계정, 기존 macbookpro·iphone 등록돼 있음)

| 노드 | tailscale 주소 | 상태 |
|---|---|---|
| scv-vehicle (차량) | **100.102.222.82** | 등록·온라인 검증 완료 |
| ppub-lab (연구실 PC) | 100.74.53.67 | 등록·온라인 검증 완료 |
| macbookpro | 100.94.103.28 | 기등록 — 맥북에서 Tailscale 앱 로그인만 하면 사용 가능 |

접속: `ssh scv@100.102.222.82` (또는 MagicDNS: `ssh scv@scv-vehicle`).
계정/비밀번호는 기존 장치 계정 그대로. ppub↔차량 tailnet SSH 실측 완료(직결 2 ms).

### 이후 매번
차량 전원 → 테더링 켜기 → 맥북/ppub에서 `ssh scv@scv-vehicle`. 끝.
(tailscaled는 systemd enable — 자동 기동)

### 주의: 교내망 유선과 병행할 때 (연구실)
- 차량에 소스 기반 정책 라우팅 설치됨(`/etc/NetworkManager/dispatcher.d/90-scv-policy-routing`) —
  테더링과 유선을 동시에 켜도 교내망 IP(203.250.35.76) 인바운드가 유지됨 (2026-07-16 실측).
- 유선이 default인 상태에서 tailscale을 쓰려면 컨트롤플레인(192.200.0.101~104)만 테더링으로
  우회하는 host route가 필요함 (Fortinet 차단 회피). **이 우회 라우트는 휘발성** — 재부팅/테더링
  재연결 시 재추가 필요:
  `for i in 101 102 103 104; do sudo ip route replace 192.200.0.$i via <테더GW> dev <테더IF>; done`
- **실외(유선 없음)에서는 우회 불필요** — default가 LTE라서 그냥 동작.
- ppub은 iPhone 테더링을 컨트롤플레인 통로로만 사용하도록 설정됨
  (`ipv4/ipv6.never-default`, IPv6 비활성 — API 트래픽이 폰 데이터로 새는 것 차단).

## 2. 테더링 데이터 절약 수칙

- **필터는 차량 쪽에서**: `ssh scv-vehicle '... | grep ... | tail -5'` — 원본 로그가 LTE를 건너지 않게
- 이미지·포인트클라우드 echo 금지, bag 전송 금지 — **bag은 현장 기록만, 업로드는 연구실 유선 복귀 후**
- `ssh -C` (압축) 권장, 세션 재사용(ControlMaster) 권장
- Claude Code를 맥북에서 돌리면 **API 트래픽도 테더링을 탐** — 세션이 길어지면
  턴당 컨텍스트 전체가 업로드되므로(수백 KB/턴) 긴 디버깅은 유선 환경에서
- 유휴 tailscale 유지비용은 시간당 1~2 MB 미만 — 상시 연결 자체는 부담 없음

## 3. 진행상황 (2026-07-16 기준)

### 완료 — 소프트웨어 검증 사다리 전체
| 항목 | 상태 |
|---|---|
| 4계층 연석 방어 (L1 연석검출 / L2 회랑 keepout / L3 BLOCKED_WAIT / L4 MPPI 보강) | 구현 + 개루프 30-bag 회귀 통과 |
| 치명 결함 C1(무발행 기아)·C2(전궤적 lethal 폭주)·C3(코스트맵 신선도) | 전부 수정, C3는 Gazebo 고장주입으로 실증 (인지 사망 → 0.82 m 내 정지) |
| Frozen-world 폐루프 33월드 | 침범 0 |
| Gazebo 물리 폐루프 33월드 (인지 인루프) | 침범 0·추락 0, loc RMS 중앙값 2.2 cm |
| tiny_localization → robot_localization dual-EKF + FAST-LIO | bag 검증 완료, datum=지도 node[0] |
| 리포트 아티팩트 | https://claude.ai/code/artifact/edfb1737-efa0-48a3-bcf0-3e6df9cc2d0f |

### 2026-07-14 1차 실차 시도 — 실패, 원인 2건 모두 수정·검증 완료
1. **hunter_base SIGABRT** (can0 미기동): launch에 can0 자동 브링업 + respawn 추가
   (sudoers.d/scv-can0 설치됨). 실차 검증: can0 down → launch → /hunter/velocity 50 Hz.
2. **맵 EKF −168 km 발산** (GPS 콜드스타트 쓰레기 + 휠속도 부재): `gps_fix_gate.py` 신설 —
   datum 5 km 밖/비정상 fix를 navsat 앞에서 차단. 필드 bag 재생 A/B: 게이트 없으면
   121 km 발산, 있으면 60/60 차단 + 정상 수렴. 실기기 스모크 테스트 통과.
- 부차 요인: 시작 위치 착오 — **사용 지도(record_20260630_141709_map_d1.json)의 경로는
  공학관 순환 루프**임 (동쪽 운동장 인도 아님). 절차서 0-8 참고.
- 당일 bag 3개는 BagArchive(203.250.35.87:31447) 업로드 완료 (id 77/79/81).

### 다음 할 일 (우선순위순)
1. **2차 실차 테스트** — `field_test_procedure.md` (이 디렉토리) 절차대로.
   시작 위치를 지도 루프 위에 두고, 체크리스트 0-1~1-7 통과 후 시나리오 A~F.
   **시나리오 F(GPS 열화 구간)가 마지막 미검증 항목.**
2. 실차 bag 회수 → (유선에서) BagArchive 업로드 → 시뮬 예측치와 정량 비교
   (정지거리: 5 m 장애물 앞 ~3.4 m, BLOCKED 에스컬레이션 4 s/12 s/10 s)
3. 선택: 전·후방 RPLIDAR C1 스캔의 local_costmap 융합 (설계 필요, 미착수)
4. 선택: 3개 저장소 PR 생성 (사용자 승인 대기 상태)

### 저장소/브랜치
| 저장소 | 브랜치 | 최근 커밋 |
|---|---|---|
| DCUSnSLab/SCV_Perception | feature/curb-safety | c702620 |
| DCUSnSLab/command_center | feature/curb-safety | ef82a46 (docs) / 43b3e36 (C3) |
| nevlife/robot_localization | feature/scv-localization | 3de0d4f (GPS gate) |
| DCUSnSLab/hunter_ros2 | feature/field-hardening | bc6e8de (can0+respawn) |

작업 워크스페이스는 차량의 **~/SCV_park** (본 ~/SCV 아님).
재빌드 주의: 사본 특성상 `rm -rf build/<pkg> install/<pkg>` 후 빌드,
`--symlink-install` 시 scripts/*.py 실행권한 확인.
