# BP/BT -> waypoint_manage: "의미 수준" 목표.
# 좌표가 아니라 그래프 노드 ID + 행동 파라미터만 전달한다.
# 좌표 해석(graph/datum)과 프레임 변환(map->odom), 가드/재계산은 waypoint_manage 담당.
# 이 계약이 안정적이면 BP 를 Behavior Tree 로 교체해도 waypoint 계층은 무변경.

std_msgs/Header header

string path_id
string current_node_id            # 현재 목표 노드 (graph 노드 ID)
string[] next_node_ids            # lookahead 노드들 (보통 1~3개)

# 경로 진행 정보 (MultipleWaypoints 로 전달용)
int32 current_waypoint_index
int32 total_waypoints
bool is_final_waypoint

# 행동 파라미터 (결정은 BP/BT, 실행은 waypoint_manage)
float32 speed_limit               # m/s, 0 = 기본값 사용
float32 goal_reached_threshold    # m,   0 = 기본값 사용
string recalc_mode                # ""(passthrough) | "densify" | "offset" ... (waypoint_generator 모드)
