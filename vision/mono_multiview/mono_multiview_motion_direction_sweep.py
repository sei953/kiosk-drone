#!/usr/bin/env python3
"""Separates two candidate causes of mixed-motion instability (reviewer's
question 2): does error track TRAJECTORY DIRECTION, or does direction only
matter because it changes the matched-point COUNT (forward motion shrinks
the two frames' shared FOV, so fewer points can match at all)?

Earlier ad-hoc testing (see mono_multiview_capture_and_match.py's
BASELINE_EAST comment) found a forward+east combined baseline gave a great
Essential-matrix result but a badly wrong Homography one for the SAME pair --
suggestive of an instability, but a single anecdotal data point can't tell
whether that was about the direction mix itself or just happened to land at
a lower inlier count.

This sweeps several (BASELINE_NORTH, BASELINE_EAST) combinations spanning
pure-lateral, pure-forward, and 45-degree-mixed motion, at more than one
magnitude each, in one flight. For every trial it logs the RANSAC inlier
count alongside each method's rotation error against PX4's own IMU attitude
(quaternion_angle_deg -- the same real-sensor ground truth select_pose()
already uses, so this reuses an established metric instead of inventing a
new one). Plotting error against inlier count and checking whether points
cluster by direction (not just by count) answers the question: if trials
with similar inlier counts but different directions still show clearly
different error, direction matters independently of point count; if error
tracks inlier count regardless of direction, direction only mattered via its
effect on point count.

TEMPORARY diagnostic script, not production.
"""

import asyncio

import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

from mono_multiview_capture_and_match import (
    ALT, APPROACH_NORTH, SETTLE_SEC, capture_frame, detect_and_match, goto,
)
from mono_multiview_pose_estimate import (
    estimate_pose, estimate_pose_homography, quaternion_angle_deg,
    rotation_angle_deg,
)

# (label, baseline_north, baseline_east) -- spans pure-lateral, pure-forward,
# and mixed motion, each at two magnitudes, so direction and point-count
# effects can be told apart instead of covarying.
TRIALS = [
    ("lateral_0.6", 0.0, 0.6),
    ("lateral_1.0", 0.0, 1.0),
    ("forward_0.3", 0.3, 0.0),
    ("forward_0.6", 0.6, 0.0),
    ("mixed_0.3_0.3", 0.3, 0.3),
    ("mixed_0.6_0.3", 0.6, 0.3),
]


async def get_attitude_quaternion(drone):
    async for q in drone.telemetry.attitude_quaternion():
        return q


async def run(K):
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
    trial_data = []  # (label, bn, be, frame_a, quat_a, frame_b, quat_b)
    for label, bn, be in TRIALS:
        await goto(drone, APPROACH_NORTH, 0.0, 0.0, f"[{label}] approach", SETTLE_SEC)
        frame_a = await loop.run_in_executor(None, capture_frame, f"dir_{label}_a")
        quat_a = await get_attitude_quaternion(drone)

        await goto(drone, APPROACH_NORTH + bn, be, 0.0,
                   f"[{label}] baseline step (north+{bn}, east+{be})", SETTLE_SEC)
        frame_b = await loop.run_in_executor(None, capture_frame, f"dir_{label}_b")
        quat_b = await get_attitude_quaternion(drone)

        trial_data.append((label, bn, be, frame_a, quat_a, frame_b, quat_b))

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

    print("\n=== MOTION-DIRECTION SWEEP RESULTS ===")
    print(f"{'label':>15} {'inliers':>8} {'true_rot':>9} {'e_err':>7} {'h_err':>7}")
    for label, bn, be, frame_a, quat_a, frame_b, quat_b in trial_data:
        true_cam_rot_deg = quaternion_angle_deg(quat_a, quat_b)
        try:
            _, pts_a, pts_b = detect_and_match(frame_a, frame_b)
        except SystemExit as e:
            print(f"{label:>15}  match FAILED: {e}")
            continue
        inlier_count = len(pts_a)

        e_err = None
        try:
            R, t_unit, pts3d_unit, inlier_2d_a = estimate_pose(pts_a, pts_b, K)
            e_cam_rot = rotation_angle_deg(R)
            e_err = abs(e_cam_rot - true_cam_rot_deg)
        except SystemExit as e:
            print(f"  [{label}] Essential FAILED: {e}")

        h_err = None
        try:
            R_h, t_h_internal, n_h, inlier_a_h, h_inliers = estimate_pose_homography(pts_a, pts_b, K)
            h_cam_rot = rotation_angle_deg(R_h)
            h_err = abs(h_cam_rot - true_cam_rot_deg)
        except SystemExit as e:
            print(f"  [{label}] Homography FAILED: {e}")

        e_str = f"{e_err:.2f}" if e_err is not None else "n/a"
        h_str = f"{h_err:.2f}" if h_err is not None else "n/a"
        print(f"{label:>15} {inlier_count:8d} {true_cam_rot_deg:9.2f} {e_str:>7} {h_str:>7}")

    print("\nRead this as: sort by inliers. If error tracks inlier count "
          "regardless of label (direction), the earlier mixed-motion "
          "instability was mediated by point count, not direction per se. "
          "If trials with SIMILAR inlier counts but different labels still "
          "show clearly different error, direction matters independently.")


def main():
    from mono_multiview_pose_estimate import capture_camera_info
    print("-- Getting camera intrinsics")
    K = capture_camera_info()
    asyncio.run(run(K))


if __name__ == "__main__":
    main()
