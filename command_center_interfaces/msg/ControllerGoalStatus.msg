# ControllerGoalStatus - Controller에서 목표 도달 상태를 알리는 메시지

# 헤더 정보
std_msgs/Header header

# Goal 상태 정보
bool goal_reached           # 목표 도달 여부
string goal_id              # 목표 노드 ID  
float64 distance_to_goal    # 목표까지 거리 (m)
int32 status_code           # 상태 코드
    # 0: PENDING - 진행 중
    # 1: SUCCEEDED - 도착
    # 2: FAILED - 실패
    # 3: ABORTED - 중단