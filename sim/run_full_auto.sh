#!/usr/bin/env bash
# 전체 구성을 한 번에 기동 (kiosk_4walls 월드, 비전 기반 자동 접근 bringup_level=4 FULL):
#   1) PX4 SITL + 월드 + XRCE Agent + gcs_keepalive  (run_kiosk_4walls.sh 재사용)
#   2) PX4/카메라 토픽 대기
#   3) ros_gz_bridge (image, camera_info -> /camera/image_raw, /camera/camera_info)
#   4) aruco_pnp_node
#   5) approach_control_node  bringup_level=4 (arm+offboard+이륙+탐색+접근까지 자동)
#
# 주의: 5)는 실제로 arm하고 비행한다. 레벨 0~3 단계 검증을 마친 뒤에만 쓸 것.
# Ctrl+C 로 노드/브리지를 정리한다. PX4/Agent/keepalive는 기본적으로 유지하고
# STOP_INFRA=1 이면 함께 종료한다.
#
# 환경변수: TARGET_WALL(동) BRINGUP_LEVEL(4) STOP_INFRA(0) WAIT_TIMEOUT(120)
set -euo pipefail

TARGET_WALL="${TARGET_WALL:-동}"
BRINGUP_LEVEL="${BRINGUP_LEVEL:-4}"
STOP_INFRA="${STOP_INFRA:-0}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"
WORLD=kiosk_4walls

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=~/kiosk_drone_ws/logs
mkdir -p "${LOG_DIR}"

# ROS 환경 (set -u 와 충돌하므로 잠시 해제)
set +u
source /opt/ros/jazzy/setup.bash
source ~/kiosk_drone_ws/install/setup.bash
set -u
export GZ_CONFIG_PATH=/usr/share/gz:${GZ_CONFIG_PATH:-}

# 이 스크립트가 띄우는 노드/브리지 프로세스 패턴. `ros2 run`은 래퍼(python) 밑에 실제
# 노드를 자식으로 띄우는데, 비대화형 셸의 백그라운드(&) 프로세스는 SIGINT가 SIG_IGN으로
# 상속돼 래퍼 PID에 kill -INT를 보내도 무시되고, 래퍼만 KILL하면 실제 노드가 고아로
# 남는다. 실측: 이전 실행의 aruco_pnp_node(target_wall=남)가 고아로 살아남아 같은
# /target/pose에 계속 발행 -> 이후 TARGET_WALL을 뭘 주든 남쪽 벽으로 접근했음
# (docs/PROGRESS.md 참고). 그래서 PID가 아니라 패턴으로 기동 전/종료 시 모두 정리한다.
NODE_PATTERNS=(
    "kiosk_vision/lib/kiosk_vision/aruco_pnp_node"
    "kiosk_vision/lib/kiosk_vision/approach_control_node"
    "ros2 run kiosk_vision"
    "ros_gz_bridge/parameter_bridge"
    "ros2 run ros_gz_bridge parameter_bridge"
    "tail -f ${LOG_DIR}/approach_control.log"
)
kill_nodes() {  # kill_nodes <signal>
    local p
    for p in "${NODE_PATTERNS[@]}"; do
        pkill "-$1" -f "${p}" 2>/dev/null || true
    done
}
stop_stale_nodes() {
    kill_nodes INT
    local t=0
    while [ "${t}" -lt 5 ] && pgrep -f "kiosk_vision/lib/kiosk_vision/|ros_gz_bridge/parameter_bridge" >/dev/null; do
        sleep 1; t=$((t + 1))
    done
    kill_nodes KILL
}

PIDS=()
cleanup() {
    trap - INT TERM EXIT
    echo; echo "== 정리 =="
    stop_stale_nodes
    if [ "${STOP_INFRA}" = "1" ]; then
        pkill -f "gcs_keepalive.py" 2>/dev/null || true
        pkill -f "MicroXRCEAgent udp4 -p 8888" 2>/dev/null || true
        pkill -f "build/px4_sitl_default/bin/px4" 2>/dev/null || true
        pkill -f "gz sim" 2>/dev/null || true
        echo "PX4/Agent/keepalive/gz 종료"
    else
        echo "PX4/Agent/keepalive는 유지 (같이 끄려면 STOP_INFRA=1)"
    fi
}
trap cleanup INT TERM EXIT

wait_for() {  # wait_for <설명> <명령...>
    local what="$1"; shift
    local t=0
    printf '대기: %s ' "${what}"
    until "$@" >/dev/null 2>&1; do
        sleep 2; t=$((t + 2)); printf '.'
        if [ "${t}" -ge "${WAIT_TIMEOUT}" ]; then
            echo; echo "시간 초과(${WAIT_TIMEOUT}s): ${what}" >&2
            exit 1
        fi
    done
    echo " OK"
}

echo "== 0) 이전 실행의 잔여 노드 정리 =="
if pgrep -af "kiosk_vision/lib/kiosk_vision/|ros_gz_bridge/parameter_bridge"; then
    echo "  ^ 잔여 노드 발견 — 종료"
    stop_stale_nodes
fi

echo "== 1) PX4 + 월드 + XRCE Agent + gcs_keepalive =="
if pgrep -f "build/px4_sitl_default/bin/px4" >/dev/null; then
    echo "PX4 이미 실행 중 — 재사용 (월드가 ${WORLD}인지 확인 필요)"
    pgrep -f "MicroXRCEAgent udp4 -p 8888" >/dev/null || {
        setsid bash -c "MicroXRCEAgent udp4 -p 8888 > '${LOG_DIR}/agent.log' 2>&1" </dev/null >/dev/null 2>&1 &
        disown
    }
else
    "${SIM_DIR}/run_kiosk_4walls.sh"
fi

echo "== 2) 대기 =="
wait_for "gz 카메라 토픽" bash -c "gz topic -l | grep -q 'IMX214/image\$'"
GZ_IMG=$(gz topic -l | grep 'IMX214/image$' | head -n1)
GZ_INFO=$(gz topic -l | grep 'IMX214/camera_info$' | head -n1)
[ -n "${GZ_IMG}" ] && [ -n "${GZ_INFO}" ] || { echo "카메라 토픽 이름 확인 실패" >&2; exit 1; }
echo "  image=${GZ_IMG}"
echo "  info =${GZ_INFO}"
wait_for "/fmu/out 토픽 (XRCE 연결)" bash -c "ros2 topic list | grep -q '^/fmu/out/vehicle_local_position'"

echo "== 3) ros_gz_bridge =="
ros2 run ros_gz_bridge parameter_bridge \
    "${GZ_IMG}@sensor_msgs/msg/Image[gz.msgs.Image" \
    "${GZ_INFO}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
    --ros-args -r "${GZ_IMG}:=/camera/image_raw" -r "${GZ_INFO}:=/camera/camera_info" \
    > "${LOG_DIR}/camera_bridge.log" 2>&1 &
PIDS+=($!)
wait_for "/camera/image_raw 프레임 수신" bash -c "timeout 5 ros2 topic echo --once /camera/image_raw --no-arr"

echo "== 4) aruco_pnp_node (target_wall=${TARGET_WALL}) =="
ros2 run kiosk_vision aruco_pnp_node --ros-args -p "target_wall:=${TARGET_WALL}" \
    > "${LOG_DIR}/aruco.log" 2>&1 &
PIDS+=($!)
sleep 3

echo "== 5) approach_control_node (bringup_level=${BRINGUP_LEVEL}) =="
ros2 run kiosk_vision approach_control_node --ros-args -p "bringup_level:=${BRINGUP_LEVEL}" \
    -p "target_wall:=${TARGET_WALL}" \
    > "${LOG_DIR}/approach_control.log" 2>&1 &
PIDS+=($!)

echo "== 기동 완료 — Ctrl+C 로 종료 =="
echo "로그: ${LOG_DIR}/{px4_sitl_4walls,agent,gcs_keepalive,camera_bridge,aruco,approach_control}.log"
tail -f "${LOG_DIR}/approach_control.log" &
PIDS+=($!)
# 어느 노드든 죽으면 알리고 종료
wait -n "${PIDS[@]}" || true
echo "노드 하나가 종료됨 — 로그 확인" >&2
