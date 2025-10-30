# RequestReplan.msg
# 재계획 요청 메시지

std_msgs/Header header

string start_node_id    # B경로의 시작점
string goal_node_id     # 마지막으로 사용하던 경로의 최종 지점
string route_type       # "A" or "B"