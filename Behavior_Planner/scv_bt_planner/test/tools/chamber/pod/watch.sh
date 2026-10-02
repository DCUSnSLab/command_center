#!/bin/bash
# 실험 gzserver(GAZEBO_MASTER_URI 11355)에 데스크톱(:20) gzclient 를 붙이고, 런마다 화면을 mp4 로 녹화한다.
#   watch.sh start   런(gzserver)마다 gzclient 를 새로 띄운다 — 10/01: 서버가 죽은 뒤 남은 gzclient 는 재접속 후 응답없음(97 % CPU 공회전)
#   watch.sh stop
#   녹화: ~/bt_chamber/rec/<런 시작 epoch>.mp4 (x11grab :20, 5 fps, Gazebo 창 영역). SCV_REC=0 이면 녹화 생략.
source /opt/ros/humble/setup.bash
export DISPLAY=:20 VGL_DISPLAY=egl GAZEBO_MASTER_URI=http://127.0.0.1:11355
DESIGN=$(bash -c 'source ~/bt_chamber/env.sh >/dev/null 2>&1; echo "$SCV_DESIGN"')
REC=~/bt_chamber/rec; mkdir -p "$REC"
up() { ss -ltn | grep -q ":11355 "; }
case "${1:-start}" in
  start)
    pgrep -f "watch\.sh loop$" >/dev/null && { echo "이미 실행 중"; exit 0; }
    setsid nohup bash "$HOME/bt_chamber/watch.sh" loop > ~/bt_chamber/watch.log 2>&1 < /dev/null &
    echo "뷰어 루프 시작 — 런마다 Gazebo 창이 새로 뜨고 $REC 에 녹화된다" ;;
  loop)
    while true; do
      until up; do sleep 2; done
      t0=$(date +%s); echo "[$t0] gzserver up → gzclient"
      sleep 4
      vglrun -d egl gzclient > /dev/null 2>&1 & GC=$!
      # 창이 뜨면 Gazebo 를 왼쪽 위(1380x500)로 줄이고, 오른쪽 540 px 에 BT 뷰어·왼쪽 아래에 경로 지도를 띄운 뒤 녹화 시작
      for _ in $(seq 1 30); do xdotool search --name "^Gazebo$" >/dev/null 2>&1 && break; sleep 1; done
      for w in $(xdotool search --name "^Gazebo$" 2>/dev/null); do xdotool windowsize $w 1380 500 windowmove $w 0 29 2>/dev/null; done
      python3 ~/bt_chamber/bt_viewer.py --port 1667 --geometry 540x1000+1380+29 --design "$(cat ~/bt_chamber/current_design 2>/dev/null || echo "$DESIGN")" --map-geometry 1380x500+0+529 > ~/bt_chamber/bt_viewer.log 2>&1 & BV=$!
      # 로봇(scv) 스폰 뒤 GUI 카메라를 로봇 추적으로 (gazebo 11 에는 gz model -l 이 없다 — -m scv -i 로 존재 확인) — 기본 시점은 멀어서 로봇이 점으로 보인다(10/01 녹화 확인)
      ( for _ in $(seq 1 40); do gz model -m scv -i >/dev/null 2>&1 && break; sleep 2; done
        gz camera -c gzclient_camera -f scv >/dev/null 2>&1 ) &
      FF=""
      if [ "${SCV_REC:-1}" = "1" ]; then
        ffmpeg -y -loglevel error -f x11grab -framerate 5 -video_size 1920x1000 -i :20+0,29 \
          -vf scale=1280:-2 -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p "$REC/$t0.mp4" < /dev/null & FF=$!
        echo "[$t0] 녹화 시작 pid $FF"
      fi
      while up; do sleep 2; done
      echo "[$(date +%s)] gzserver down → gzclient/녹화 종료"
      [ -n "$FF" ] && kill -INT $FF 2>/dev/null
      kill $GC $BV 2>/dev/null; sleep 2; kill -9 $GC $BV 2>/dev/null
      wait $FF 2>/dev/null
      sleep 2
    done ;;
  stop)
    pkill -f "watch\.sh loop$"; pkill -x gzclient; pkill -f "bt_chamber/bt_viewer\.py"; pkill -INT -x ffmpeg; echo "뷰어 중지" ;;
esac
