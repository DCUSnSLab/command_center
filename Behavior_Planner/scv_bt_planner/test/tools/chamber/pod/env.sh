# V100 Pod 용 BT 챔버 A/B 환경 — source 해서 쓴다
export SCV_WS=$HOME/vehicle_ws
export SCV_DESIGN=${SCV_DESIGN:-$HOME/bt_chamber/chamber_20260803.pod.design.json}
export SCV_ZONES=$HOME/scv_ws/tools/gazebo/zones/rtk_clean.yaml
export SCV_URDF=$HOME/scv_ws/tools/gazebo/scv_sim_robot_chamber_gpu.urdf   # gpu_ray 라이다
export SCV_VGL=1 VGL_DISPLAY=egl                                           # OGRE 렌더 → V100(EGL)
export GAZEBO_MASTER_URI=http://127.0.0.1:11355                            # 데스크톱용 11345 와 분리
export PATH=$HOME/.local/bin:$PATH                                          # torch(cu121) 는 ~/.local
export RCUTILS_LOGGING_BUFFERED_STREAM=0
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/bt_chamber/fastdds_udp_only.xml   # SHM 전송 비활성(UDP 만) — 공유 /dev/shm 잔재 없음
export SCV_DOMAIN=97   # 96 은 누수된 utm latched pub 참여자 13개가 점유해 디스커버리 불능(10/01) — 새 도메인
export SCV_BT_ARGS="-p groot2_port:=1667"   # BT 뷰어(bt_viewer.py)용 Groot2 ZMQ 서버(1667/1668). simple 런에선 BT 미기동
