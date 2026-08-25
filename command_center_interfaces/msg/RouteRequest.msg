# 운영자/서버 -> global planner: "이 경로로 가라".
#
# 모드 파라미터가 따로 없다. **요청 내용이 곧 방식**이다 —
#   via_nodes 가 비어 있으면  : goal 까지 최단 경로 (A*)
#   via_nodes 가 있으면       : 그 순서대로 (최단이 아니어도 그 길로)
# 운영자가 고른 모드와 실제 데이터가 어긋나는 상태를 아예 만들지 않기 위해서다.
#
# QoS 는 transient_local(latched) 로 쓴다. 플래너가 재시작해도 지금 유효한
# 경로를 다시 받아야 하기 때문이다. map_provider 의 graph/utm 과 같은 이유.
#
# 지금은 PyQt 툴이 발행하고, 나중에 웹 서버가 같은 메시지를 내려보내면
# 로봇 쪽은 아무것도 바뀌지 않는다.

std_msgs/Header header

string request_id                 # 응답(RouteStatus)·로그 대조용. 비면 플래너가 생성
string[] via_nodes                # 거쳐갈 그래프 노드 ID (순서 그대로). 비면 최단 경로
geometry_msgs/PoseStamped goal    # via_nodes 가 비었을 때의 목적지 (map 프레임)
bool use_goal                     # goal 필드가 유효한가 (빈 Pose 와 원점을 구분)

bool loop                         # 마지막 노드에서 첫 노드로 닫아 순환
bool fill_gaps                    # 이웃 via_node 가 직접 링크로 안 이어지면 A* 로 채움
