#!/usr/bin/env bash
# kiosk_4walls 월드(원점 사방 3m 4벽 + 16 ArUco마커)로 인프라 기동:
#   1) repo의 world/model을 PX4-Autopilot Tools 경로로 동기화 (sim/kiosk.sdf와 동일 패턴)
#   2) PX4 SITL (빌드된 바이너리 직접 실행 — make 금지, sim/README.md 참고)
#   3) Micro XRCE-DDS Agent (px4_msgs 브리지, docs/SETUP.md 참고)
#   4) gcs_keepalive.py — MAVSDK로 GCS 하트비트 유지(없으면 PX4가 arm 거부) +
#      MPC_XY_VEL_MAX/CRUISE를 approach_control_node의 속도캡(0.3m/s)에 맞춤
# ros_gz_bridge(image, camera_info)와 노드(aruco_pnp_node / approach_control_node)는
# 이 스크립트가 아니라 별도로 띄운다 (카메라 토픽 이름이 스폰된 모델 인스턴스에
# 의존하고, bringup_level 단계별로 노드를 재시작해야 하기 때문).
set -euo pipefail

export GZ_CONFIG_PATH=/usr/share/gz:${GZ_CONFIG_PATH:-}

REPO_SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR=~/PX4-Autopilot
PX4_WORLDS="${PX4_DIR}/Tools/simulation/gz/worlds"
PX4_MODELS="${PX4_DIR}/Tools/simulation/gz/models"
LOG_DIR=~/kiosk_drone_ws/logs
mkdir -p "${LOG_DIR}"

echo "== 1) world/model 동기화 =="
cp "${REPO_SIM_DIR}/worlds/kiosk_4walls.sdf" "${PX4_WORLDS}/kiosk_4walls.sdf"
for wall in kiosk_wall kiosk_wall_east kiosk_wall_south kiosk_wall_west; do
    mkdir -p "${PX4_MODELS}/${wall}"
    cp "${REPO_SIM_DIR}/${wall}/model.sdf" "${REPO_SIM_DIR}/${wall}/model.config" "${PX4_MODELS}/${wall}/"
    cp "${REPO_SIM_DIR}/${wall}/${wall}_marker.png" "${PX4_MODELS}/${wall}/"
done
echo "동기화 완료: ${PX4_WORLDS}/kiosk_4walls.sdf, ${PX4_MODELS}/kiosk_wall{,_east,_south,_west}/"

if ! command -v gz >/dev/null 2>&1; then
    echo "경고: gz 명령을 찾을 수 없음. gz sim(Gazebo) 설치 및 PATH를 확인할 것." >&2
elif ! gz sim --versions >/dev/null 2>&1; then
    echo "경고: gz sim이 인식되지 않음. GZ_CONFIG_PATH=${GZ_CONFIG_PATH} 로도 sim8.yaml을 찾지 못하고 있음." >&2
fi

echo "== 2) PX4 SITL (world=kiosk_4walls) =="
PX4_BIN="${PX4_DIR}/build/px4_sitl_default/bin/px4"
if [ ! -x "${PX4_BIN}" ]; then
    echo "먼저 빌드하세요: cd ~/PX4-Autopilot && make px4_sitl gz_x500_depth" >&2
    exit 1
fi
cd "${PX4_DIR}"
setsid bash -c "GZ_CONFIG_PATH='${GZ_CONFIG_PATH}' PX4_GZ_WORLD=kiosk_4walls PX4_SIM_MODEL=gz_x500_depth script -qec '${PX4_BIN}' '${LOG_DIR}/px4_sitl_4walls.log'" \
    < /dev/null > /dev/null 2>&1 &
disown
echo "PX4 SITL 백그라운드 기동, 로그: ${LOG_DIR}/px4_sitl_4walls.log"

echo "== 3) Micro XRCE-DDS Agent =="
if ! pgrep -f "MicroXRCEAgent udp4 -p 8888" > /dev/null; then
    setsid bash -c "MicroXRCEAgent udp4 -p 8888 > '${LOG_DIR}/agent.log' 2>&1" \
        < /dev/null > /dev/null 2>&1 &
    disown
    echo "Agent 백그라운드 기동, 로그: ${LOG_DIR}/agent.log"
else
    echo "Agent 이미 실행 중 — 재사용"
fi

echo "== 4) GCS keepalive + 속도 파라미터 =="
VENV_PY=~/kiosk_drone_ws/venv/bin/python3
if [ -x "${VENV_PY}" ]; then
    if ! pgrep -f "gcs_keepalive.py" > /dev/null; then
        setsid "${VENV_PY}" "${REPO_SIM_DIR}/gcs_keepalive.py" > "${LOG_DIR}/gcs_keepalive.log" 2>&1 \
            < /dev/null &
        disown
        echo "gcs_keepalive 백그라운드 기동, 로그: ${LOG_DIR}/gcs_keepalive.log"
    else
        echo "gcs_keepalive 이미 실행 중 — 재사용"
    fi
else
    echo "경고: ${VENV_PY} 없음 — GCS 연결 없이는 arm이 거부됨(docs/troubleshooting.md 참고). venv에 mavsdk 설치 필요." >&2
fi

echo "== 완료 =="
echo "gz sim / px4 부팅에는 수십 초 걸릴 수 있음. 확인:"
echo "  tail -f ${LOG_DIR}/px4_sitl_4walls.log"
echo "  gz topic -l | grep -i camera   # 카메라 토픽 이름 확인 후 ros_gz_bridge 기동"
echo "  source /opt/ros/jazzy/setup.bash && source ~/kiosk_drone_ws/install/setup.bash && ros2 topic list | grep fmu"
