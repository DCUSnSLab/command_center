# global planner -> 운영자/서버: 요청을 받았는가, 못 받았으면 왜.
#
# 지금까지 실패는 **로그에만** 남아서, 경로가 안 나오는 이유를 보려면 터미널을
# 뒤져야 했다. 발행자가 알아야 고칠 수 있다.

std_msgs/Header header

string request_id
bool accepted                     # 경로를 만들어 발행했는가
string reason                     # 거부/실패 사유 (사람이 읽는 문장)
uint32 node_count                 # 만들어진 경로의 노드 수
string mode                       # 실제로 쓴 방식: "astar" | "sequence"
