# BT 플래너 챔버 A/B — V100 Pod 실험 환경 (2026-09-29 구축)

Pod: `ppub-claude/sim-dev-a-0` (d1-k8s, Tesla V100 32GB, CPU 12 / RAM 24Gi 한도, 홈 = Longhorn PVC 50Gi)
이미지: `harbor.cu.ac.kr/k8s_ros2/gazebo-desktop:stack2-20260901` (Ubuntu 22.04 · ROS 2 Humble · Gazebo 11.10 · selkies 데스크톱)

## 구성
| 경로 | 내용 | 출처 |
|---|---|---|
| `~/vehicle_ws/src` | 차량 작업본 기준 워크스페이스 + `scv_bt_planner`(feature/bt-planner, command_center 9ddec8e / 슈퍼 801c3235) | ppub `~/SCV/vehicle_ws` (.git 제외) |
| `~/vehicle_ws/install` | 챔버에 필요한 16 패키지 빌드 (`~/build_vehicle_ws.sh`, 로그 `build.log`) | Pod 빌드 |
| `~/scv_ws/tools` | scv_sim_tools 하니스 (65dd553 + SCV_VGL/SCV_SPAWN_Z/SCV_URDF 우선 패치) | ppub `~/scv_ws/tools` |
| `~/scv_sim/chamber/chamber_20260803.*` | 좌측 연석 곡선로 챔버(8노드 13.8 m, 장애물 445) | ppub |
| `~/bt_chamber/` | 이 환경의 진입점 (아래) | 신규 |

## 실행 (Pod 안)
```bash
~/bt_chamber/run_one.sh simple  ~/scv_sim/bt_ab/smoke_simple.json 300   # 단일 런
~/bt_chamber/run_one.sh bt      ~/scv_sim/bt_ab/smoke_bt.json 300
~/bt_chamber/run_one.sh shadow  ~/scv_sim/bt_ab/smoke_shadow.json 300
~/bt_chamber/run_ab.sh ~/scv_sim/bt_ab/run1 3 420                        # simple×3 + bt×3 + shadow×1, 보고서 자동
```
결과 json 옆 `logs_<이름>/` 에 gzserver·behavior·bt·mppi·rc_phase·judge 로그. 판정은 `verdict`
(PASS_REACHED 등) 과 연속 지표(min_goal_dist, distance, reach_t) 를 함께 본다.

## 환경 변수 (`env.sh`)
- `SCV_WS=~/vehicle_ws`, `SCV_DESIGN=chamber_20260803.pod.design.json`(경로만 Pod 로 교정), `SCV_ZONES=rtk_clean.yaml`
- `SCV_URDF=scv_sim_robot_chamber_gpu.urdf` + `SCV_VGL=1` — 라이다 gpu_ray 와 OGRE 렌더를 V100(EGL) 으로.
  xvfb 만 쓰면 llvmpipe 로 조용히 떨어진다(8/27 실측).
- `GAZEBO_MASTER_URI=http://127.0.0.1:11355` — 데스크톱용 Gazebo(11345) 와 분리.
- `FASTRTPS_DEFAULT_PROFILES_FILE=fastdds_udp_only.xml` — SHM 전송 끔. 공유 /dev/shm(2 GiB) 에 잔재를 남기지 않는다.
- 하니스 자체는 `ROS_DOMAIN_ID=96 ROS_LOCALHOST_ONLY=1` 로 고정한다(데스크톱 스택은 도메인 20).

## 주의
- 이 Pod 는 ppub_claude 의 Mando 용인 코스 작업(`~/mando_ws`, 도메인 20)과 공유한다. 하니스 `kill_stack` 은
  자기 노드명 패턴(gzserver 포함)으로 전체 kill 하므로, **데스크톱에 Gazebo 를 띄운 채 A/B 를 돌리지 말 것.**
- 챔버 kill_stack 은 `/dev/shm/fastrtps_*` 를 지운다 — 같은 Pod 의 다른 DDS 참가자(도메인 20)에 영향.
- CPU 한도 12 코어: 런 하나가 gzserver+torch MPPI+코스트맵으로 8~10 코어를 쓴다. 동시 2 런 금지.
- 이미지에 torch 가 없어 `~/.local`(pip --user, torch 2.5.1+cu121)에 의존한다. PVC 를 갈면 사라진다.
- 결과·로그는 Pod PVC 에 남는다. 보고서는 `chamber_ab_report.py` 출력을 ppub 로 가져와 작성.

## 10/01 추가: 실험 도메인·누수·뷰어
- `SCV_DOMAIN`(env.sh, 현재 97): 도메인 96 은 지난 런들이 누수한 `/map_provider_node/utm` latched pub 13개 + ros2 daemon 이
  살아 있어 늦게 뜨는 노드(hunter_status 3 / rc_watch / base_mux)가 디스커버리되지 않았다(view_bt: RC 0.21 m, PASS_BLOCKED).
  하니스는 종료 시 UTM_PID 를 정리하도록 고쳤다(a039e34). 96 의 잔재는 `ros2 topic pub`/`ros2-daemon`(environ ROS_DOMAIN_ID=96)만 골라 kill 하면 된다.
- 런마다 남는 것: xvfb-run 의 Xvfb+dbus 1벌, 목표 지연 서브셸(`SCV_GOAL_DELAYS=99999` sleep). DDS 참여자가 아니라 무해.
- 뷰어: `watch.sh start` → 데스크톱(:20)에 gzclient. gzclient 는 gzserver 가 바뀌어도 같은 URI(11355)에 재접속하므로 런 사이에 그대로 둬도 된다.
- 진단: 런 중 `~/bt_chamber/diag.sh`(노드 목록·groundtruth·hunter_status·rc_watch 샘플·RTF·CPU).

## 10/01 run2·run3 뒤 하네스 상태
- RC 단계: `rc_watch.py --drive 1.0 --settle 2` 1회 호출(시작·도달·정지 후 표본), hunter_status 는 `--mode-file $LOGD/hunter_mode` 단일 프로세스.
  런 중 새 DDS 참여자를 만들지 않는다 — 늦게 뜬 참여자가 간헐적으로 디스커버리되지 않던 플레이크(view_bt, run3 bt3 stall) 노출면 제거.
- BT shadow 모드 목표 재동기화(shadowResync) 반영 빌드(ba15bcf). 결과: run3 bt 3/3·shadow 1/1 PASS_REACHED.
- 결과: run2(simple 3/3, bt 2/3 — bt3 는 RC 정지 지연 3.6 m 뒤 급선회로 연석 추락, shadow 1/1), run3 (`run3/stall/` 은 전환 미도달 런).

## 10/02 BT 실시간 뷰어
- `bt_viewer.py`(저장소 scv_bt_planner/test/tools, command_center 4e5bbd8): Groot2Publisher(ZMQ 1667)에 붙어 마지막 완료 틱 경로를 칠한다.
  env.sh 의 `SCV_BT_ARGS="-p groot2_port:=1667"` 로 BT 쪽 서버를 켠다. pyzmq 는 ~/.local(pip --user).
- watch.sh 가 런마다 Gazebo 창을 왼쪽 1380 px 로 줄이고 오른쪽 540 px 에 뷰어를 띄워 함께 녹화한다. 카메라는 로봇 추적.
- 자가시험: `viewer_selftest.sh`(도메인 98 에 BT 노드만, xvfb 캡처).
- 10/02 경로 지도: 뷰어에 `--design` 을 주면 왼쪽 아래(1380x500+0+529)에 경로 지도 창. Gazebo 는 왼쪽 위 1380x500.
  차량 위치는 `gz topic -e /gazebo/scv_log_world/pose/info -u`(Gazebo transport). 런 시작은 `view_run.sh <bp> <이름>`
  (exec 명령줄에 뷰어 이름이 들어가면 watch.sh stop 의 pkill 이 exec 셸을 죽인다).

> 저장소 사본(2026-10-02): Pod `~/bt_chamber` 의 스크립트를 그대로 옮긴 것이다. 경로는 Pod(/home/ubuntu) 기준이며,
> 사용할 때는 이 디렉터리 내용을 Pod `~/bt_chamber` 로 복사하고 `bt_viewer.py` 는 `../../bt_viewer.py` 를 함께 둔다.
