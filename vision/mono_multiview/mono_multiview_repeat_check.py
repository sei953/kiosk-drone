#!/usr/bin/env python3
"""Reproducibility check for the mixed_0.3_0.3 anomaly found by
mono_multiview_motion_direction_sweep.py: that trial (baseline north+0.3,
east+0.3) gave a Homography rotation error of 5.76deg, a clear outlier next
to every other trial (0.30-1.46deg) despite a mid-range inlier count (169) --
not the lowest of the six. Before concluding this baseline direction is
genuinely a bad geometry for Homography decomposition, rule out that it was
a one-off RANSAC/matching-noise fluke: repeat the SAME (north+0.3, east+0.3)
baseline 3 times. A control repeat of lateral_1.0 (which was clean, h_err
0.30deg) is included once, to confirm repeats of a KNOWN-good baseline also
stay clean and the sim/pipeline isn't just noisy in general today.

If mixed_0.3_0.3 stays consistently bad across repeats, direction/baseline
geometry is a real cause, independent of the earlier point-count question.
If it varies wildly run to run, the single 5.76deg reading was noise, not a
geometry effect.

TEMPORARY diagnostic script, not production.
"""

import asyncio

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

from mono_multiview_capture_and_match import (
    ALT, APPROACH_NORTH, SETTLE_SEC, capture_frame, detect_and_match, goto,
)
from mono_multiview_pose_estimate import (
    estimate_pose, estimate_pose_homography, quaternion_angle_deg,
    rotation_angle_deg,
)

TRIALS = [
    ("mixed_0.3_0.3_rep1", 0.3, 0.3),
    ("mixed_0.3_0.3_rep2", 0.3, 0.3),
    ("mixed_0.3_0.3_rep3", 0.3, 0.3),
    ("lateral_1.0_control", 0.0, 1.0),
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
    trial_data = []
    for label, bn, be in TRIALS:
        await goto(drone, APPROACH_NORTH, 0.0, 0.0, f"[{label}] approach", SETTLE_SEC)
        frame_a = await loop.run_in_executor(None, capture_frame, f"rep_{label}_a")
        quat_a = await get_attitude_quaternion(drone)

        await goto(drone, APPROACH_NORTH + bn, be, 0.0,
                   f"[{label}] baseline step (north+{bn}, east+{be})", SETTLE_SEC)
        frame_b = await loop.run_in_executor(None, capture_frame, f"rep_{label}_b")
        quat_b = await get_attitude_quaternion(drone)

        trial_data.append((label, frame_a, quat_a, frame_b, quat_b))

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

    print("\n=== REPRODUCIBILITY CHECK RESULTS ===")
    print(f"{'label':>20} {'inliers':>8} {'true_rot':>9} {'e_err':>7} {'h_err':>7}")
    for label, frame_a, quat_a, frame_b, quat_b in trial_data:
        true_cam_rot_deg = quaternion_angle_deg(quat_a, quat_b)
        try:
            _, pts_a, pts_b = detect_and_match(frame_a, frame_b)
        except SystemExit as e:
            print(f"{label:>20}  match FAILED: {e}")
            continue
        inlier_count = len(pts_a)

        e_err = None
        try:
            R, t_unit, pts3d_unit, inlier_2d_a = estimate_pose(pts_a, pts_b, K)
            e_err = abs(rotation_angle_deg(R) - true_cam_rot_deg)
        except SystemExit as e:
            print(f"  [{label}] Essential FAILED: {e}")

        h_err = None
        try:
            R_h, t_h_internal, n_h, inlier_a_h, h_inliers = estimate_pose_homography(pts_a, pts_b, K)
            h_err = abs(rotation_angle_deg(R_h) - true_cam_rot_deg)
        except SystemExit as e:
            print(f"  [{label}] Homography FAILED: {e}")

        e_str = f"{e_err:.2f}" if e_err is not None else "n/a"
        h_str = f"{h_err:.2f}" if h_err is not None else "n/a"
        print(f"{label:>20} {inlier_count:8d} {true_cam_rot_deg:9.2f} {e_str:>7} {h_str:>7}")

    print("\nRead this as: if the 3 mixed_0.3_0.3 repeats cluster together "
          "(all high or all low h_err), the earlier 5.76deg reading was a "
          "real, repeatable geometry effect. If they scatter widely, it was "
          "matching/RANSAC noise, not a direction-specific instability.")


def main():
    from mono_multiview_pose_estimate import capture_camera_info
    print("-- Getting camera intrinsics")
    K = capture_camera_info()
    asyncio.run(run(K))


if __name__ == "__main__":
    main()
