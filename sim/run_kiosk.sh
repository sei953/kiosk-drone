#!/usr/bin/env bash
# kiosk 월드로 PX4 SITL + gz_x500_depth 실행.
# make 대신 빌드된 바이너리를 직접 실행한다 (sim/README.md 참고:
# make는 ninja 컴파일 잡이 메모리를 순식간에 먹는 문제가 있음).
set -euo pipefail

export GZ_CONFIG_PATH=/usr/share/gz:${GZ_CONFIG_PATH:-}

if ! command -v gz >/dev/null 2>&1; then
    echo "경고: gz 명령을 찾을 수 없음. gz sim(Gazebo) 설치 및 PATH를 확인할 것." >&2
elif ! gz sim --versions >/dev/null 2>&1; then
    echo "경고: gz sim이 인식되지 않음. GZ_CONFIG_PATH=${GZ_CONFIG_PATH} 로도 sim8.yaml을 찾지 못하고 있음." >&2
fi

cd ~/PX4-Autopilot

PX4_BIN=build/px4_sitl_default/bin/px4
if [ ! -x "${PX4_BIN}" ]; then
    echo "먼저 빌드하세요: cd ~/PX4-Autopilot && make px4_sitl gz_x500_depth" >&2
    exit 1
fi

PX4_GZ_WORLD=kiosk PX4_SIM_MODEL=gz_x500_depth "${PX4_BIN}"
