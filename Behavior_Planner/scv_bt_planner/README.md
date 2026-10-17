# scv_bt_planner

SCV 행동 플래너의 BehaviorTree.CPP(v4) 신축본. `simple_behavior_planner` 와 **같은 입출력 계약** 위에서
"다음 행동"만 BT 로 판단한다. 설계 문서: claude.ai/artifact/BFLHvg8jxnqx1Kv1hnXuBC (2026-09-28).

## 구성

| 파일 | 역할 |
|---|---|
| `behavior_trees/scv_behavior.xml` | 트리 (Groot2 편집 가능). 모든 노드가 즉시 SUCCESS/FAILURE 를 돌려주는 결정 트리 |
| `config/behavior_profiles.yaml` | node_type 프로필(현행 behavior_modifiers 와 동일 값) + BT 오버레이(구역·회전) |
| `config/bt_planner_params.yaml` | 노드 파라미터 (이름은 simple_behavior_planner 와 동일) |
| `src/path_manager.cpp` | 합류·통과 이력 (원본 path_manager.py 이식, gtest 로 동등성 검증) |
| `src/blocked_wait_monitor.cpp` | NORMAL→BLOCKED_WAIT→CREEP→ASSIST (원본 이식) |
| `src/zone_table.cpp` | graph.json `Node.Zone` 조회 (없으면 sidewalk) |
| `src/profiles.cpp` | smppi baseline × node 프로필 × 오버레이 → MPPIParams |
| `src/bt_nodes.cpp` | 조건/액션/데코레이터 + Context |
| `src/bt_planner_node.cpp` | ROS 노드 (콜백 이식, 틱 루프, 발행) |

## 모드

- `mode:=shadow` (기본): `/bt/` 접두 토픽으로만 발행. simple BP 와 같은 주행에서 결정만 비교. 명령 없음.
- `mode:=active`: 기존 토픽으로 발행 → simple BP 를 대체 (둘을 동시에 active 로 띄우지 말 것).
- 모드와 무관하게 `/bt/behavior` 에 `행동|에스컬레이션|목표노드|zone=..|type=..` 를 변경 시마다 방송한다.

## 지도 확장

graph.json Node 에 `"Zone": "sidewalk"|"road"|"crosswalk"|"shared_road"|"gps_denied"` 를 넣는다.
없으면 sidewalk → simple BP 와 같은 결정(동등성 모드). `map_file_path` 로 파일을 준다.

## 빌드·시험

의존: `behaviortree_cpp`(v4) — `sudo apt install ros-humble-behaviortree-cpp`(4.10) 또는 SCV 슈퍼프로젝트의
`src/third_party/BehaviorTree.CPP` 서브모듈(4.10.0) 소스 빌드. `hunter_msgs` 는 SCV 슈퍼프로젝트(`src/vehicle/hunter_ros2`).
둘 중 하나라도 없으면 CMake 가 경고만 내고 이 패키지를 건너뛴다(`behavior_planner:=simple` 은 영향 없음).

```
colcon build --packages-up-to scv_bt_planner --cmake-args -DBTCPP_EXAMPLES=OFF -DBTCPP_BUILD_TOOLS=OFF
colcon test --packages-select scv_bt_planner && colcon test-result --verbose
```

gtest 는 ROS 없이 돈다: `test_path_manager`(합류 규칙 동등성) · `test_blocked_wait_monitor` · `test_zone_and_profiles` ·
`test_hazard_waypoints` · `test_behavior_tree`(실제 `scv_behavior.xml` 을 로드해 결정 순서·포트 값·에스컬레이션을 틱 단위로
확인, 마지막 케이스가 틱 비용 `[BENCH] tick avg … us, heap allocs/tick …` 를 찍는다 — 2026-10-17 x86 Release 기준 약 2 µs·0 회).
`bt_nodes.cpp` 는 `scv_bt_planner_bt` 라이브러리로 분리돼 노드와 이 시험이 같은 코드를 링크한다.

### 틱 비용
- 노드 안의 파생값(`zone()`·`turnDegAtTarget()`)은 목표 노드별 틱 캐시에서 읽고, XML 리터럴 포트는 첫 틱에 한 번만 파싱한다
  (`readPortOnce`). 틱마다 바뀌는 값은 포트가 아니라 `Context` 로 넘길 것.
- 노드 로그 `[TICK] n ticks / s: avg, max, >period` (`tick_stats_sec`, 기본 60 s, 0 = 끔). 한 틱이 주기(1/`tick_hz`)를 넘기면 즉시 경고.

## 미이식 / 보류

- probe(탐침 전진)의 far_wall 코스트맵 분류 — `probe.enabled` 를 켜도 경고만.
- node_type 12/13 재계획 트리거(path_availability, 인지 의존) — 이번 범위 밖.
- 후방 목표 자동 후진(`reverse.allow_behind_goal`) — 판정 미구현(초기 OFF 권장).
- traffic_light(node_type 10) 안전 정지 — 인지 미사용 방침에 따라 미이식.

## 판단 로깅

- `/bt/behavior`(모드 무관): 행동·에스컬레이션·목표 노드·구역·node_type 이 바뀔 때마다 한 줄. `/behavior_status`, `/blocked_assist_request`,
  `/pause_command`, `/mppi_update_params`, `/multiple_waypoints` 와 함께 bag 으로 남기면 결정과 제어기 입력이 전부 남는다.
- 노드 로그: 합류 선택, `advanced to`, `[BEHAVIOR]`, `[BLOCKED]`, `[HAZARD]`, `[MODE]`, pause 사유.
- **틱 단위 전이 기록** `bt_log_file`(BT.CPP FileLogger2, `.btlog`): `''` 끔(기본) | `auto` → `~/.ros/btlog/bt_<시각>.btlog` | 경로.
  field_drive 에서는 `behavior_planner:=bt bt_log_file:=auto`. 전이당 9바이트, 결정 트리는 틱마다 전 노드가 리셋되므로 20 Hz 에서 약 4 MB/10 min(Pod 실측 322 s 1.3 MB).
  변환·요약: `test/tools/btlog_to_csv.py run.btlog --summary --ticks` (CSV: t_rel_s, 노드, 유형, 상태, 포트 속성;
  `--ticks` 는 틱마다 SUCCESS 였던 노드 경로). Groot2 앱에서도 그대로 재생된다.

## 실시간 BT 뷰어 (`test/tools/bt_viewer.py`)
`groot2_port` 파라미터를 켜면(`-p groot2_port:=1667`) BT.CPP Groot2Publisher(ZMQ 1667/1668)가 열린다.
`bt_viewer.py --port 1667` 은 Groot2 앱 없이 같은 프로토콜로 트리를 받아 **마지막 완료 틱에서 실제로 거친 노드**를
SUCCESS/FAILURE 색으로 그린다(전이 기록 TOGGLE_RECORDING/GET_TRANSITIONS 사용 — 결정 트리는 틱마다 resetStatus 되므로
STATUS 폴링으로는 이번 틱 경로를 구분할 수 없다). 하단 패널은 최신 bt.log 꼬리(제어 모드·목표 노드·행동·차단 상태)를
읽어 ROS 참여자를 만들지 않는다. 의존성: PyQt5, pyzmq.
`--design <챔버 설계 JSON>` 을 주면 두 번째 창(경로 지도)을 띄운다: 월드의 보도 띠·장애물, 경로 그래프 노드·링크,
지나온 노드·현재 목표 노드, 차량 위치·자세·궤적, 속도·경로 이탈·잔여 거리. 차량 위치는 Gazebo 자체 전송
(`gz topic -e /gazebo/<world>/pose/info -u`)에서 읽어 ROS/DDS 를 거치지 않는다.
