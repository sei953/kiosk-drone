#!/usr/bin/env python3
"""PX4 SITL은 GCS(지상국) 연결이 없으면 arm을 거부한다(health_and_arming_checks의
"No connection to the GCS" 체크, NAV_DLL_ACT>0일 때). 순수 ROS2/px4_msgs
오프보드(px4_ros_com, uXRCE-DDS)만 쓰면 MAVLink 하트비트가 전혀 없어서 이 체크에
걸린다 -- MAVSDK로 udp:14540(Onboard 링크)에 연결만 해둬도 "GCS 연결"로 인정돼
arm이 풀린다.

겸사겸사 approach_control_node의 속도캡(max_speed=0.3m/s)이 실제로 관철되도록
PX4의 MPC_XY_VEL_MAX/MPC_XY_CRUISE도 여기서 같이 낮춰둔다 (기본값 5~12m/s면
우리 쪽 위치 setpoint를 훨씬 빠르게 쫓아가려 해서 접근 속도가 우리 의도보다
빨라진다).

사용: SITL + MicroXRCEAgent가 떠 있는 상태에서 백그라운드로 띄워두고 그대로 둔다.
    source ~/kiosk_drone_ws/venv/bin/activate
    python3 sim/gcs_keepalive.py &
"""
import asyncio

from mavsdk import System

SPEED_PARAMS = {
    "MPC_XY_VEL_MAX": 0.3,
    "MPC_XY_CRUISE": 0.3,
}


async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("GCS keepalive connected")
            break

    for name, value in SPEED_PARAMS.items():
        try:
            await drone.param.set_param_float(name, value)
            print(f"set {name} = {value}")
        except Exception as e:  # noqa: BLE001 - SITL 파라미터 이름이 버전마다 조금씩 다를 수 있음
            print(f"failed to set {name}: {e}")

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run())
