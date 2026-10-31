#!/usr/bin/env python3
"""Generic MAVSDK flight pattern for verifying a vision ROS2 node live against
PX4 SITL -- arm, hold, then step through WAYPOINTS, then land. Doesn't touch
ROS2/the vision node at all; it only needs *some* real translation to happen
so a baseline-dependent node (mono_multiview_node) has something to compute
from, or so a side-by-side comparison has motion to compare against.

Meant to run ALONGSIDE whichever node(s) you're verifying (aruco_pnp_node,
mono_multiview_node, approach_control_node, ...), launched separately --
this script only flies the drone and prints its own progress; watch the
node's own log/topics in another terminal for the actual verification.

Consolidates what had been copy-pasted ad hoc into /tmp scripts for each
verification round this session (lateral-step sweeps, mockup-wall checks,
side-by-side PnP/mono comparisons) -- same shape every time, just the
WAYPOINTS list differed.

Usage:
  cd ~/kiosk-drone && source venv/bin/activate
  python control/fly_vision_node_verify.py
"""

import asyncio

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

ALT = -1.5  # NED z for 1.5m altitude

# (north, east, yaw_deg, label, hold_sec). First entry after takeoff is the
# approach position; edit this list per what you're verifying -- e.g. for a
# markerless-mockup check, point yaw at whichever wall the mockup replaced;
# for a side-by-side PnP/mono comparison, a few small lateral steps is enough
# to see both nodes react to the same motion.
WAYPOINTS = [
    (0.3, 0.0, 0.0, "approach", 10),
    (0.3, 0.6, 0.0, "lateral step 1", 10),
    (0.3, -0.2, 0.0, "lateral step 2", 10),
]

# health.is_global_position_ok/is_home_position_ok can take much longer than
# a few seconds to flip true on a PX4 SITL instance that's had several
# MAVSDK connections come and go already (observed up to ~20s on an instance
# otherwise behaving normally) -- wait generously, but don't block forever if
# it never resolves; the underlying EKF2/local position is usually fine
# regardless (px4_msgs' own xy_valid/z_valid flags are a better check for
# anything consuming VehicleLocalPosition directly).
HEALTH_WAIT_SAMPLES = 60


async def goto(drone, north, east, yaw, label, hold_sec):
    print(f"-- {label}")
    await drone.offboard.set_position_ned(PositionNedYaw(north, east, ALT, yaw))
    await asyncio.sleep(hold_sec)


async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            break

    print("Waiting for global position estimate...")
    n = 0
    async for health in drone.telemetry.health():
        n += 1
        if health.is_global_position_ok and health.is_home_position_ok:
            print(f"-- Position estimate OK (after {n} samples)")
            break
        if n >= HEALTH_WAIT_SAMPLES:
            print(f"-- Health check timed out after {n} samples, proceeding anyway")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Priming setpoint stream")
    for _ in range(10):
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(0.1)

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Offboard start failed: {error._result.result}")
        await drone.action.disarm()
        return

    await goto(drone, 0.0, 0.0, 0.0, "Takeoff to 1.5m", 6)

    for north, east, yaw, label, hold_sec in WAYPOINTS:
        await goto(drone, north, east, yaw, label, hold_sec)

    print("-- Returning to origin")
    await goto(drone, 0.0, 0.0, 0.0, "Return to origin", 6)

    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Offboard stop failed: {error._result.result}")

    print("-- Landing")
    await drone.action.land()
    async for state in drone.telemetry.landed_state():
        if state.name == "ON_GROUND":
            print("-- Landed")
            break
    print("-- Done")


if __name__ == "__main__":
    asyncio.run(run())
