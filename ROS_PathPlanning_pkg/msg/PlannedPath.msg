# PlannedPath - 계획된 경로의 노드와 링크 정보
# 기존 GraphMap 구조를 재활용하여 경로상의 노드와 링크들을 포함

# 헤더 정보
std_msgs/Header header

# 경로 메타데이터
string path_id
string start_node_id
string goal_node_id
float64 total_distance  # 총 경로 거리 (미터)
float64 total_time      # 예상 총 이동 시간 (초)

# 경로를 구성하는 노드와 링크 (기존 MapData 구조 재활용)
gmserver/MapData path_data