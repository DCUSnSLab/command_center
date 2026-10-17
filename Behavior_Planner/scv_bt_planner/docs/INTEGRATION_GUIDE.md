# scv_bt_planner 통합 가이드 (팀원용)

2026-10-15 기준, main 9cb6889 이후. 질문은 PR #23·#24·#25 스레드 또는 Park 에게.

BT 행동 플래너는 **기존 simple_behavior_planner 와 같은 입출력 계약** 위에서 "다음 행동" 결정만 BehaviorTree.CPP v4 트리로 바꾼 것입니다.
**기본값은 전부 현행 유지**(simple BP 기동, BT 미기동)라 아무것도 하지 않으면 동작이 바뀌지 않습니다.

## 1. 한눈에

| 항목 | 내용 |
|---|---|
| 패키지 | `Behavior_Planner/scv_bt_planner` (C++17, ament_cmake) |
| 외부 의존 | `behaviortree_cpp` 4.x (`sudo apt install ros-humble-behaviortree-cpp`, 4.10 확인) · `hunter_msgs`(SCV 슈퍼프로젝트 `src/vehicle/hunter_ros2`) |
| 의존 없을 때 | CMake 가 경고 후 **패키지를 건너뜀** — 나머지 빌드·기본 런치 영향 없음 |
| 켜는 방법 | 런치 인자 `behavior_planner:=simple|bt|shadow` (`scv_system.launch.py`, `field_drive.launch.py`) |
| 입력 | `/planned_path_detailed`(PlannedPath), 위치(`/odom` 또는 `/odometry/global`), `/goal_status`, `/hunter_status`, `/emergency_stop`, `/stop_flag`, `/map_provider_node/utm`(datum), `/costmap`(합류 통행성) |
| 출력(active) | `/multiple_waypoints` 또는 `/target_waypoints`, `/mppi_update_params`, `/pause_command`, `/behavior_status`, `/blocked_assist_request`, `/behavior/localization_hint` — simple BP 와 동일 토픽 |
| 출력(shadow) | 위 토픽에 `/bt/` 접두. 명령 없음 |
| 항상 | `/bt/behavior` — 결정이 바뀔 때마다 `행동|에스컬레이션|목표노드|zone=..|type=..` |
| 기록 | `bt_log_file:=auto` → `~/.ros/btlog/bt_<시각>.btlog` (틱 단위 전이, `test/tools/btlog_to_csv.py` 로 CSV) |

## 2. 설치·빌드

```
sudo apt install ros-humble-behaviortree-cpp        # 4.10.0, 한 번만
cd ~/SCV && colcon build --packages-select scv_bt_planner command_center_launch
```
- 차량은 플레인 빌드(심볼릭 아님)라 런치·XML·yaml 을 고쳐도 **재빌드해야 반영**됩니다.
- 소스 빌드가 필요하면 `src/third_party/BehaviorTree.CPP`(4.10.0 서브모듈, SCV 슈퍼프로젝트 `feature/bt-planner`) 를 쓰고 `--cmake-args -DBTCPP_EXAMPLES=OFF -DBTCPP_BUILD_TOOLS=OFF`.
- v3(`behaviortree_cpp_v3`)와 패키지명이 달라 공존합니다.

## 3. 운용 순서 (권장)

1. **shadow 로 먼저** — simple 이 주행하고 BT 는 결정만 방송. 실차 위험 0.
   ```
   ros2 launch command_center_launch scv_system.launch.py behavior_planner:=shadow bt_log_file:=auto
   ros2 bag record /bt/behavior /behavior_status /bt/behavior_status /bt/mppi_update_params /mppi_update_params \
                   /bt/multiple_waypoints /multiple_waypoints /goal_status /odometry/global /hunter_status
   ```
   `/bt/behavior_status` 와 `/behavior_status`, `/bt/multiple_waypoints` 와 `/multiple_waypoints` 가 같은 시각에 같은 값이면 동등.
   (bag 137 재생 기준: 전이 34건 순서 동일, 시각 차 ≤0.07 s, 목표 일치 99.7 %.)
2. **bt 로 전환** — `behavior_planner:=bt`. simple 은 뜨지 않고 BT 가 기존 토픽으로 명령.
3. 문제 시 인자만 빼면 즉시 현행으로 복귀(코드 변경 불필요).

`field_drive.launch.py`(개인 운용 런치)도 같은 인자를 받습니다.

## 4. 결정 구조 (수정할 때 알아야 할 것)

- 트리: `behavior_trees/scv_behavior.xml`. **모든 노드가 즉시 SUCCESS/FAILURE** 를 돌려주는 결정 트리이고 20 Hz 마다 루트부터 다시 평가합니다. RUNNING 을 쓰지 않으므로 상태(에스컬레이션 타이머·pause 발동·발행 이력)는 `Context`(`bt_nodes.hpp`)에 둡니다.
- 순서: SafetyGuards(비상정지→Stop) → MissionReady → SafetyPause → AvoidHazardWaypoints(기본 비활성) → ZoneEntryRules(횡단보도 진입 pause, ForceSuccess) → BlockedEscalation[SelectBehavior] → RefreshWaypoints.
- SelectBehavior(Fallback): NeedsReverse → IsTurn 좌/우 → ZoneIs crosswalk/shared_road/gps_denied/road → 기본 Drive(sidewalk).
- 그림: `docs/pres/graph/bt_tree_part1_top.png`(상위), `bt_tree_part2_select.png`(행동 선택) — 노드별 근거·전이 조건 포함. Graphviz 소스 `.dot` 동봉.
- 행동 프로필: `config/behavior_profiles.yaml` = `node_profiles`(node_type → 현행 behavior_modifiers 와 **동일 값**) × `overlays`(구역·회전). MPPIParams = smppi baseline × node_profile × overlay. `Drive(sidewalk)` 는 overlay 없음 = simple 과 같은 결정.
- 구역(Zone): graph.json 의 `Node.Zone`(`sidewalk|road|crosswalk|shared_road|gps_denied`). 없으면 sidewalk. **scv_system 은 그래프를 map_provider 토픽으로 받으므로 파일 조회가 비어 전 노드 sidewalk** — 구역 행동을 쓰려면 번들의 graph.json 경로를 `map_file_path` 로 넘기는 연결이 필요(미구현 항목 1).

## 5. 확장 방법

| 하고 싶은 것 | 손댈 곳 |
|---|---|
| 새 조건/행동 노드 | `bt_nodes.hpp/.cpp` 에 클래스 추가 → `registerScvNodes()` 등록 → XML 에 배치. 즉시 SUCCESS/FAILURE 만 반환할 것 |
| 구역별 속도·가중치 | `behavior_profiles.yaml` `overlays` 값 (코드 변경 없음, 재빌드만) |
| node_type 프로필 | `behavior_profiles.yaml` `node_profiles` — simple 의 behavior_modifiers 와 같이 유지해야 동등성 보존 |
| 에스컬레이션 시간 | `config/bt_planner_params.yaml` `blocked.*` (detect 4 s → wait 12 s → creep 10 s → assist 15 s 반복) |
| 회전 판정 | `turn.min_deg`(35) `turn.window_m`(6) |
| 트리 편집 | Groot2 로 XML 편집 가능(노드 모델은 `registerScvNodes` 의 포트 정의) |

변경 뒤 확인 순서: `colcon test --packages-select scv_bt_planner`(gtest 47건 — `test_behavior_tree` 가 실제 XML 을 로드해 결정 순서·포트 값·에스컬레이션을 틱 단위로 확인하고 틱 비용 `[BENCH]` 를 찍는다) → `test/tools/shadow_smoke.py` → bag 재생 `test/tools/run_replay_compare.sh` → 챔버(Gazebo) `test/tools/chamber/`.

## 6. 검증 도구

| 도구 | 용도 |
|---|---|
| `test/tools/btlog_to_csv.py run.btlog --summary --ticks` | 전이 기록 → CSV, 노드별 SUCCESS/FAILURE, 틱별 경로 |
| `test/tools/bt_viewer.py --port 1667 [--design …]` | 실시간 트리 색칠·지도(`groot2_port:=1667` 필요, PyQt5·pyzmq) |
| `test/tools/scvsys_modes.sh <install/setup.bash> [domain]` | scv_system 3모드 기동 가용성 점검 |
| `test/tools/replay_compare.py`, `run_replay_compare.sh` | bag 재생 simple vs BT 결정 비교 |
| `test/tools/chamber/` | Gazebo 챔버 A/B 하니스(V100 Pod 스크립트 포함) |
| 노드 로그 `[TICK]` | 틱 비용 통계(평균·최대·주기 초과 횟수), `tick_stats_sec`(기본 60 s, 0 = 끔). 한 틱이 주기(50 ms)를 넘기면 즉시 경고 |

## 7. 알려진 제약·미구현

1. Zone 조회가 파일 기반 — map_provider 토픽 연동 미구현(4절).
2. 인지 입력 미사용: 장애물·구덩이는 코스트맵을 통해 SMPPI 가 피하고, BT 는 막힘 에스컬레이션만 담당. 위험 코스 통과 여부는 perception 품질에 좌우됨(10/08 Pod: 차량 perception 으로는 simple·bt 모두 S1 구덩이 실패).
3. 미이식: probe far_wall 분류, node_type 12/13 재계획, 후방 목표 자동 후진(`reverse.allow_behind_goal` 기본 false), traffic_light 정지.
4. `AvoidHazardWaypoints` 는 기본 비활성(`hazard.enable=false`) — 챔버에서 hold 교착이 있어 켜지 말 것.
5. simple BP 의 `join_frame_fix`(기본 false)는 합류 프레임 결함 수정 스위치 — 팀 결정 전까지 끔.

## 8. 함정 (실제로 겪은 것)

- BT XML 에 새 노드를 넣고 C++ 등록 없이 올리면 `Node not recognized` 로 노드가 죽는다 — XML 과 C++ 는 함께 빌드·배포.
- 차량 git 전역 신원이 backgroundmin 이라 차량에서 커밋하면 그 이름이 찍힌다 — `git -c user.name=…` 로 명시.
- `ros2 run` 래퍼를 kill 하면 자식 `bt_planner_node` 가 살아남는다 — `pkill -x bt_planner_node`.
- 스모크에서 위치를 순간이동시키면 teleport 재정렬·4 s 정체 감지가 발동(정상) — 점진 이동.
- `.btlog` 는 20 Hz 에서 약 4 MB/10 min. 장시간 주행이면 `bt_log_file` 경로를 큰 디스크로.

## 9. 이력

- 9/28 신축 → 10/03 `BT_implement` 태그 → PR #23 main 병합(10/12) → PR #24 전이 기록(10/13) → PR #25 scv_system 스위치(10/13). 설계 원칙·검증 수치는 `README.md`, `docs/PRESENTATION_GUIDE.md`, `docs/pres/numbers.md`.
