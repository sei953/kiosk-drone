#!/usr/bin/env python3
"""Lateral-bias diagnostic: is the ~0.3m lateral error a constant offset, or a
camera<->NED axis/sign mismatch?

bbox-center (see bbox_center_lateral_vertical) already removed the
match-density bias from using the reconstructed-point centroid. A residual
~0.3m lateral error remained after that fix, and there are two very different
explanations that look identical in a single static reading:

  1. Constant offset (e.g. a calibration/mounting-frame bias) -- estimated
     lateral would be geom_lateral + constant_error for any true position.
  2. Axis/sign mismatch (e.g. camera-X wired to the wrong body axis, or with
     the wrong sign) -- estimated lateral would NOT track geom_lateral's sign
     as the true position moves side to side.

A single frame pair can't distinguish these. This script commands frame A to
a KNOWN, swept east offset (EAST_SWEEP) with the wall still fully in frame,
and compares the estimated lateral against geom_lateral (PX4 position +
known wall pose -- no vision involved, see mono_multiview_pose_estimate.py's
run()) at each one. If (est - geom) stays roughly constant across the sweep,
it's case 1 (fixable with a single offset correction). If it changes sign or
doesn't move with geom_lateral at all, it's case 2 (the camera->NED mapping
itself needs fixing, not a calibration tweak).

RESULT (first run): case 2-shaped, but not a mapping bug -- a ground-truth
sign bug. est_lateral tracked -geom_lateral (not +geom_lateral) almost
exactly. Both aruco_pnp_node's tvec[0] and bbox_center_lateral_vertical()
report the TARGET's position in the camera's own frame (positive = wall to
the camera's right); geom_lateral was computed as the CAMERA's own east
offset (pos_a.east_m). Those are geometric negatives of each other when the
wall sits at east=0 (kiosk.sdf), not two measurements of the same quantity --
moving the camera east makes the wall appear to the camera's left. Fixed
below by comparing against wall_east - camera_east, the same pattern
geom_forward already used (3.0 - pos_a.north_m). Re-run this after any future
lateral change to confirm est tracks +geom_lateral now.

TEMPORARY diagnostic script, not production.
"""

import asyncio

import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

from mono_multiview_capture_and_match import (
    ALT, APPROACH_NORTH, BASELINE_EAST, BASELINE_NORTH, SETTLE_SEC,
    capture_frame, detect_and_match, goto,
)
from mono_multiview_pose_estimate import (
    SPREAD_MIN_M, bbox_center_lateral_vertical, capture_camera_info,
    estimate_pose, estimate_pose_homography, fit_plane, homography_to_metric,
    rotation_angle_deg,
)

WALL_WORLD_X = 0.0  # kiosk.sdf: <pose>0 3 1.5 0 0 0</pose>
WALL_WORLD_Y = 3.0  # kiosk.sdf: <pose>0 3 1.5 0 0 0</pose>

# Centered on 0 so a constant offset shows up as a same-sign shift at every
# point, and a sign/axis bug shows up as est_lateral not crossing zero where
# geom_lateral does. +-0.3m matches the magnitude of the bias being chased.
EAST_SWEEP = [-0.3, 0.0, 0.3]


async def get_position_ned(drone):
    async for pv in drone.telemetry.position_velocity_ned():
        return pv.position


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
    pairs = []  # (east0, frame_a, pos_a, frame_b, pos_b)
    for east0 in EAST_SWEEP:
        await goto(drone, APPROACH_NORTH, east0, 0.0,
                   f"East {east0:+.1f}m (frame A, approach)", SETTLE_SEC)
        frame_a = await loop.run_in_executor(None, capture_frame, f"lat_{east0:+.1f}_a")
        pos_a = await get_position_ned(drone)

        baseline_target_n = APPROACH_NORTH + BASELINE_NORTH
        baseline_target_e = east0 + BASELINE_EAST
        await goto(drone, baseline_target_n, baseline_target_e, 0.0,
                   f"East {baseline_target_e:+.1f}m (frame B, baseline step)", SETTLE_SEC)
        frame_b = await loop.run_in_executor(None, capture_frame, f"lat_{east0:+.1f}_b")
        pos_b = await get_position_ned(drone)

        print(f"  east0={east0:+.1f}m  actual_a.east={pos_a.east_m:.3f}m  "
              f"actual_b.east={pos_b.east_m:.3f}m")
        pairs.append((east0, frame_a, pos_a, frame_b, pos_b))

        # Return to the approach line before the next sweep point so every
        # pair starts from the same north/alt and only east differs.
        await goto(drone, APPROACH_NORTH, 0.0, 0.0, "Reset to east=0", SETTLE_SEC)

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

    print("\n=== LATERAL BIAS SWEEP RESULTS ===")
    print(f"{'east0':>7} {'geom_lat':>9} {'est_lat[E]':>11} {'err[E]':>8} "
          f"{'est_lat[H]':>11} {'err[H]':>8}")
    for east0, frame_a, pos_a, frame_b, pos_b in pairs:
        # wall_east - camera_east (see RESULT note above), not pos_a.east_m alone.
        geom_lateral = WALL_WORLD_X - pos_a.east_m
        real_baseline = float(np.linalg.norm([
            pos_b.north_m - pos_a.north_m,
            pos_b.east_m - pos_a.east_m,
            pos_b.down_m - pos_a.down_m,
        ]))
        try:
            _, pts_a, pts_b = detect_and_match(frame_a, frame_b)
        except SystemExit as e:
            print(f"{east0:+7.1f}  match FAILED: {e}")
            continue

        lat_e = None
        try:
            R, t_unit, pts3d_unit, inlier_2d_a = estimate_pose(pts_a, pts_b, K)
            pts3d_m = pts3d_unit * real_baseline
            normal, centroid, spread = fit_plane(pts3d_m)
            forward = float(-np.dot(normal, centroid))
            lat_e, _ = bbox_center_lateral_vertical(inlier_2d_a, K, normal, forward)
        except SystemExit as e:
            print(f"  [{east0:+.1f}] Essential FAILED: {e}")

        lat_h = None
        try:
            R_h, t_h_internal, n_h, inlier_a_h, _ = estimate_pose_homography(pts_a, pts_b, K)
            _, lat_h, _, _, _, _ = homography_to_metric(
                R_h, t_h_internal, n_h, inlier_a_h, K, real_baseline)
        except SystemExit as e:
            print(f"  [{east0:+.1f}] Homography FAILED: {e}")

        e_str = f"{lat_e:+.3f}" if lat_e is not None else "n/a"
        h_str = f"{lat_h:+.3f}" if lat_h is not None else "n/a"
        err_e = f"{lat_e - geom_lateral:+.3f}" if lat_e is not None else "n/a"
        err_h = f"{lat_h - geom_lateral:+.3f}" if lat_h is not None else "n/a"
        print(f"{east0:+7.1f} {geom_lateral:+9.3f} {e_str:>11} {err_e:>8} "
              f"{h_str:>11} {err_h:>8}")

    print("\nRead this as: if (est - geom) stays roughly constant sign+magnitude "
          "across the east0 sweep, it's a constant offset (calibration fix). "
          "If est_lat doesn't track geom_lateral's sign/scale as east0 sweeps "
          "through zero, it's a camera<->NED axis or sign mismatch (mapping fix).")


def main():
    print("-- Getting camera intrinsics")
    K = capture_camera_info()
    asyncio.run(run(K))


if __name__ == "__main__":
    main()
