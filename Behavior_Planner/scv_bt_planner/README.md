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

```
colcon build --packages-up-to scv_bt_planner --cmake-args -DBTCPP_EXAMPLES=OFF -DBTCPP_UNIT_TESTS=OFF -DBTCPP_BUILD_TOOLS=OFF
colcon test --packages-select scv_bt_planner && colcon test-result --verbose
```

## 미이식 / 보류

- probe(탐침 전진)의 far_wall 코스트맵 분류 — `probe.enabled` 를 켜도 경고만.
- node_type 12/13 재계획 트리거(path_availability, 인지 의존) — 이번 범위 밖.
- 후방 목표 자동 후진(`reverse.allow_behind_goal`) — 판정 미구현(초기 OFF 권장).
- traffic_light(node_type 10) 안전 정지 — 인지 미사용 방침에 따라 미이식.
