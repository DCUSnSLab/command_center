# waypoint_manage — waypoint 계층 (기계 계층)

> 2026-08-03. BP(→ 미래 Behavior Tree)에서 waypoint 배관을 분리한 노드.
> 판단(어느 노드로 갈지/어떤 행동일지)은 BP/BT, 실행(좌표 해석·프레임 변환·
> localization 보정 전파·재계산)은 여기.

```
[BP / (미래)BT] ──TargetWaypoints(의미 목표: 노드ID+행동파라미터)──► [waypoint_manage] ──MultipleWaypoints(odom)──► MPPI
                                                                      ▲ graph/utm(latched), /fgo/status, TF map→odom
```

## 왜 분리했나

- **BT 전환 대비**: BP↔waypoint_manage 사이 계약(TargetWaypoints)이 안정되면
  BP 를 BT 로 교체해도 이 계층 무변경. 역도 성립.
- **localization 보정 전파**: 기존 BP 래치는 map→odom 을 노드당 1회만 샘플링
  → 보정 반영이 노드 간격(~5m)만큼 지연. 여기서는 같은 타깃도 5Hz 로 재변환하되
  **가드**(아래)로 FGO 오류를 걸러서 반영. MPPI 는 여전히 odom 폐쇄 세계 유지.
- **FGO 가 틀렸을 때의 방어선**: FGO 내부 게이트(1선) → **waypoint 가드(2선)** →
  MPPI odom 격리(3선).

## 가드 (update_guard.py, ROS 비의존)

| 단계 | 조건 | 동작 |
|---|---|---|
| 새 타깃 | 노드 ID 변경 | 즉시 발행 (기존 반응성 유지) |
| 상태 게이트 | /fgo/status != OK | 동결 (freeze_when_not_ok) |
| 데드밴드 | Δ < 0.3m | 무시 |
| 새너티 | Δ > 3.0m | 거부 + 경고 (localization 점프 의심) |
| 슬루 | 그 외 | ≤0.5m/s 로 목표 이동 (스텝→램프) |

## 모듈 구성

| 파일 | 역할 | ROS 의존 |
|---|---|---|
| `waypoint_manage_node.py` | I/O·TF·조립 | O |
| `graph_store.py` | graph/datum 보관, 노드ID→map 좌표 | X |
| `update_guard.py` | 가드 판정 | X |
| `waypoint_generator.py` | 재계산/생성 (passthrough / densify·offset TODO) | X |

재계산의 **결정**(모드 선택)은 BT 가 `TargetWaypoints.recalc_mode` 로 내려보내고,
여기서는 **실행**만 한다.

## 사용 (마이그레이션)

```bash
# 1) 병행 검증: BP 기존 발행 유지 + waypoint_manage 는 별도 토픽으로 비교
ros2 launch waypoint_manage waypoint_manage.launch.py   # topics.multiple_waypoints 를 임시로 바꿔 비교

# 2) 전환: BP 를 external 모드로 (내부 odom waypoint 발행 중단)
ros2 run simple_behavior_planner simple_behavior_planner_node --ros-args -p waypoint_mode:=external
ros2 launch waypoint_manage waypoint_manage.launch.py
```

## simple_behavior_planner 에서 빼야 할 것 (전환 완료 후 삭제 체크리스트)

전환(waypoint_mode:=external 상시화) 확인 후:

- [ ] `simple_behavior_planner/waypoint_publisher.py` 모듈 전체 (좌표 변환·TF·datum 로직이 여기로 이관됨)
- [ ] BP 노드의 `WaypointPublisher` 생성/연결 (`__init__`, `_link_module_publishers`)
- [ ] BP 노드의 `subgoal_pub`/`multiple_waypoints_pub` publisher 및 `waypoint_mode` single/multiple 분기
- [ ] BP 노드의 TF buffer/listener (waypoint 변환용이었다면)
- [ ] `_publish_waypoints` 를 `_publish_target_waypoints` 호출만 남기고 축소
- 유지: `subgoal_published` 래치(노드 전환 이벤트 관리), path_manager, goal_status 진행 관리
  — 이것들은 "판단"이라 BP/BT 소관

## 현재 BP 에 이미 넣어둔 것 (additive, 기존 동작 무영향)

- `/target_waypoints` (TargetWaypoints) 발행 — 항상
- `waypoint_mode: external` 지원 — 내부 odom waypoint 발행 스킵

## TODO

- [ ] update_guard 단위테스트
- [ ] waypoint_generator densify/offset 구현 + BT recalc_mode 규약 확정
- [ ] graph heading 규약(진북 CW vs ENU CCW) 확정 (`graph_store.resolve` 참조)
- [ ] /fgo/status DEGRADED 지속 시 감속 요청 (behavior 파라미터 연계) — 2선 방어 강화
- [ ] goal_reached_threshold/speed_limit 파라미터의 컨트롤러 전달 경로 정리
