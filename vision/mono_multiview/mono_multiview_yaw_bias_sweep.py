#!/usr/bin/env python3
"""Does the ~0.27m lateral bias scale with forward distance (heading/yaw
error) or stay flat regardless of distance (some other fixed cause)?

mono_multiview_lateral_bias_sweep.py found a ~0.27m lateral gap between
geom_lateral (PX4 position + known wall/spawn pose) and BOTH aruco_pnp-style
PnP and our own mono/homography estimate -- at a fixed forward distance
(~2.7m). Camera mounting offset (SDF chain sums to exactly 0 laterally),
marker-layout asymmetry (generate_marker.py is symmetric), intrinsics
principal point (cx=960.0, dead center), and wall/spawn world pose (checked
live via `gz topic -t /world/kiosk/pose/info`) were all ruled out as the
source -- none of them explain a bias that shows up equally in two
independent methods (PnP corners vs ORB+homography) but not in geom_lateral.

The remaining candidate: PX4 holds a few degrees of residual heading error
off true world-north while "hovering at yaw=0" (wait_until_stable only checks
position+velocity convergence, never attitude). A constant heading offset
converts to a lateral error that SCALES with forward distance (lateral_bias
~= forward * sin(heading_err)), unlike a translational-offset bug, which
would stay flat in meters regardless of distance.

This script isolates that: east held at 0 throughout, forward distance swept
via NORTH_SWEEP, ONE frame captured per distance (PnP recovers absolute
scale from the known marker size alone -- no second frame/baseline needed,
unlike the Essential/Homography path). If lateral_pnp/forward_geom (the
implied heading-error angle) stays roughly CONSTANT across distances, that
confirms a heading bias. If lateral_pnp itself stays roughly constant in
meters instead, the cause is still open and is NOT a heading bias.

TEMPORARY diagnostic script, not production.
"""

import asyncio

import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

from mono_multiview_capture_and_match import (
    ALT, SETTLE_SEC, capture_frame, goto,
)
from mono_multiview_pose_estimate import capture_camera_info

WALL_WORLD_Y = 3.0  # kiosk.sdf: <pose>0 3 1.5 0 0 0</pose>, confirmed live via
                     # `gz topic -t /world/kiosk/pose/info` during the lateral sweep

# Copied inline from aruco_pnp_node.py (pulls in rclpy -> can't import
# directly outside a ROS2-sourced env). Must stay byte-identical to that
# file's ARUCO_DICT/MARKER_LAYOUT/MARKER_SIZE/marker_corners_3d.
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE = 0.150
MARKER_LAYOUT = {
    0: (-0.375, 0.375), 1: (0.375, 0.375),
    2: (0.375, -0.375), 3: (-0.375, -0.375),
}


def marker_corners_3d(cx, cy, s):
    h = s / 2.0
    return [(cx - h, cy + h, 0.0), (cx + h, cy + h, 0.0),
            (cx + h, cy - h, 0.0), (cx - h, cy - h, 0.0)]


def pnp_lateral(img, K, D, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    obj_pts, img_pts = [], []
    for mid, cn in zip(ids.flatten(), corners):
        mid = int(mid)
        if mid not in MARKER_LAYOUT:
            continue
        cx, cy = MARKER_LAYOUT[mid]
        obj_pts += marker_corners_3d(cx, cy, MARKER_SIZE)
        img_pts += list(cn.reshape(4, 2))
    if len(obj_pts) < 4:
        return None
    obj_pts = np.array(obj_pts, dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    return float(tvec.flatten()[0])


# Spans the full-wall-visible range confirmed by mono_multiview_position_sweep.py
# (north<=0.8 saw 0 keypoints -- too far; north approaching ~2.0+ starts cropping
# the wall). Kept inside that window so every shot gets a clean 4-marker PnP.
NORTH_SWEEP = [0.3, 0.8, 1.3, 1.8]


async def get_position_ned(drone):
    async for pv in drone.telemetry.position_velocity_ned():
        return pv.position


async def run(K, D, detector):
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Position estimate OK")
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

    await goto(drone, 0.0, 0.0, 0.0, "Takeoff to 1.5m", SETTLE_SEC)
    await goto(drone, 0.0, 0.0, 0.0, "Rotate to yaw=0.0 (facing wall)", SETTLE_SEC)

    loop = asyncio.get_running_loop()
    samples = []  # (north, frame, pos)
    for north in NORTH_SWEEP:
        await goto(drone, north, 0.0, 0.0, f"North {north}m (east held at 0)", SETTLE_SEC)
        frame = await loop.run_in_executor(None, capture_frame, f"yaw_{north:.1f}")
        pos = await get_position_ned(drone)
        print(f"  north={north:.1f}m  actual: north={pos.north_m:.3f}m east={pos.east_m:.3f}m")
        samples.append((north, frame, pos))

    print("-- Returning to origin")
    await goto(drone, 0.0, 0.0, 0.0, "Return to origin", SETTLE_SEC)

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

    print("\n=== YAW-BIAS SWEEP RESULTS (east held at 0 throughout) ===")
    print(f"{'north':>6} {'geom_fwd':>9} {'geom_lat':>9} {'pnp_lat':>8} {'implied_deg':>12}")
    angles = []
    for north, frame, pos in samples:
        geom_forward = WALL_WORLD_Y - pos.north_m
        geom_lateral = 0.0 - pos.east_m  # wall world_x=0, see mono_multiview_pose_estimate.py
        lat = pnp_lateral(frame, K, D, detector)
        if lat is None:
            print(f"{north:6.1f}  PnP FAILED (markers not detected)")
            continue
        implied_deg = float(np.degrees(np.arctan2(lat - geom_lateral, geom_forward)))
        angles.append(implied_deg)
        print(f"{north:6.1f} {geom_forward:9.3f} {geom_lateral:+9.3f} {lat:+8.3f} {implied_deg:+12.2f}")

    if len(angles) >= 2:
        print(f"\nimplied heading-error angle: mean={np.mean(angles):+.2f}deg "
              f"std={np.std(angles):.2f}deg (over {len(angles)} points)")
        print("Read this as: if implied_deg stays roughly constant across forward "
              "distances (low std), the bias is a fixed heading/yaw error (scales "
              "with distance). If (lat - geom_lateral) itself stays constant in "
              "meters instead while implied_deg drifts a lot, it's a fixed "
              "translational offset, not heading -- look elsewhere.")


def main():
    print("-- Getting camera intrinsics")
    K = capture_camera_info()
    D = np.zeros((5,), dtype=np.float64)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    asyncio.run(run(K, D, detector))


if __name__ == "__main__":
    main()
