#!/usr/bin/env python3
"""Steps 5-8 of the mono-multiview 이동 구간 plan, built on top of
mono_multiview_capture_and_match.py's proven capture+ORB+match+RANSAC
(steps 1-4):

  5. Essential Matrix -> relative pose (rotation + unit-scale translation)
  6. Triangulation -> 3D points, still in unit-baseline scale
  7. Scale correction -> PX4's own position estimate gives the real
     frame-A-to-frame-B displacement; recoverPose's translation is already
     unit-length, so that real displacement IS the scale factor.
  8. Time sync -> simplified here to "read telemetry immediately after each
     capture" (good enough for this two-shot test); the production ROS2 node
     will need a real image-timestamp <-> position-timestamp association
     (e.g. message_filters), not this shortcut.

Prints a final forward/lateral distance estimate (meters, camera-A frame) for
the matched points -- the same shape of output as aruco_pnp_node's PoseStamped,
so it can be cross-checked against PnP/PX4 ground truth.

TEMPORARY validation script, not the production node.
"""

import asyncio
import threading
import time

import cv2
import numpy as np
from gz.msgs10.camera_info_pb2 import CameraInfo
import gz.transport13 as gz_transport
# mavsdk is imported lazily inside run() below, not here: compute_mono_pose()
# (this file's actual reusable payload) is imported by mono_multiview_node.py,
# a ROS2 node with no mavsdk dependency (px4_msgs covers telemetry there), and
# `from ... import compute_mono_pose` still executes every top-level import in
# this file -- a module-level mavsdk import would make the node fail to import
# in an environment that correctly doesn't have mavsdk installed.

from mono_multiview_capture_and_match import (
    ALT, APPROACH_NORTH, BASELINE_EAST, BASELINE_NORTH, SETTLE_SEC,
    capture_frame, detect_and_match, goto,
)

CAMERA_INFO_TOPIC = "/world/kiosk/model/x500_depth_0/link/camera_link/sensor/IMX214/camera_info"


def capture_camera_info():
    """Blocking single-message grab of camera intrinsics via gz-transport."""
    info = {}
    got = threading.Event()

    def cb(msg: CameraInfo):
        if got.is_set():
            return
        info["K"] = np.array(msg.intrinsics.k, dtype=np.float64).reshape(3, 3)
        got.set()

    node = gz_transport.Node()
    node.subscribe(CameraInfo, CAMERA_INFO_TOPIC, cb)
    if not got.wait(timeout=10):
        raise SystemExit("failed to get camera_info")
    return info["K"]


async def get_position_ned(drone):
    async for pv in drone.telemetry.position_velocity_ned():
        return pv.position


async def get_attitude_quaternion(drone):
    async for q in drone.telemetry.attitude_quaternion():
        return q


def quaternion_angle_deg(q_a, q_b):
    """Angle between two attitudes from PX4's own IMU-based estimate -- this
    is real sensor data (not simulation-only, unlike the wall-world-pose
    ground truth elsewhere in this file), so it carries over to production.

    We don't need the exact camera-to-body mounting transform to use this as
    a ground truth for cam_rot: rotation ANGLE (unlike axis) is invariant
    under conjugation by any fixed rotation, so the body's rotation angle
    between frame A and B equals the camera's rotation angle between the same
    two frames regardless of how the camera is mounted on the body.
    """
    w_a, x_a, y_a, z_a = q_a.w, q_a.x, q_a.y, q_a.z
    w_b, x_b, y_b, z_b = q_b.w, q_b.x, q_b.y, q_b.z
    # relative quaternion q_rel = q_a^-1 * q_b (unit quaternions: inverse == conjugate)
    w_rel = w_a * w_b + x_a * x_b + y_a * y_b + z_a * z_b
    cos_half_angle = np.clip(abs(w_rel), -1.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(cos_half_angle)))


# Below this bounding-box diagonal, matched points are more likely clustered
# on a single ~0.15m ArUco marker (diagonal ~0.21m) than spread across the
# wall -- confirmed by inspecting mono_frame_a/b.png for a run whose plane-fit
# yaw was off by ~20deg from aruco_pnp ground truth: frame B (closer to the
# wall, narrower FOV coverage) only shared its top two markers with frame A,
# so only points from that shared region could match at all. Two markers
# ~0.75m apart give a diagonal well above this; a single marker's corners do
# not, so this threshold separates the two cases without needing to know how
# many markers were involved.
SPREAD_MIN_M = 0.4


def fit_plane(points):
    """Least-squares plane fit (SVD) to the scaled 3D points -- the kiosk wall
    is assumed planar. Returns (normal, centroid, spread). normal is oriented
    to face back toward the camera (negative Z), matching aruco_pnp_node's
    board-normal convention (R @ [0,0,1] in its wall-local frame) so the yaw
    formula below is directly comparable to its yaw_err_from_R. That formula
    is atan2(-normal[0], -normal[2]), NOT atan2(normal[0], -normal[2]) --
    upstream found the un-negated form gives the wrong sign via a real
    closed-loop check (wall_geometry.py's yaw_err_from_R docstring,
    docs/PROGRESS.md 2026-09-19); this file used to copy that same wrong
    convention before being fixed to match. spread is the matched points'
    bounding-box diagonal [m]: the plane's normal (hence the yaw angle) is
    only well-constrained if the points span a meaningful area, not a single
    small cluster -- see SPREAD_MIN_M."""
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]
    if normal[2] > 0:
        normal = -normal
    spread = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return normal, centroid, spread


def bbox_center_lateral_vertical(pts_2d_a, K, normal_facing, forward):
    """lateral/vertical via the 2D bounding-box CENTER of the matched points
    in frame A, back-projected onto the fitted plane -- instead of the
    density-weighted mean/centroid of the reconstructed 3D points.

    The centroid approach is biased by which points happened to match: if
    more corners matched on the right marker than the left one, the mean
    drags toward the right even though the wall's actual visible extent was
    roughly symmetric. The bounding-box center only depends on the min/max
    extent of the matched region, not how densely points fill it, so it's a
    better proxy for "where does the observed region sit relative to the
    camera" -- same fix in spirit as using spread (extent) rather than count
    for the confidence check.

    normal_facing must already be oriented toward the camera (negative Z,
    this file's convention), and forward is the already-computed
    perpendicular camera-to-plane distance in meters -- both fit_plane() and
    the homography path already produce these.
    """
    px_center = (pts_2d_a.min(axis=0) + pts_2d_a.max(axis=0)) / 2.0
    ray = cv2.undistortPoints(px_center.reshape(1, 1, 2).astype(np.float64), K, None).ravel()
    ray_dir = np.array([ray[0], ray[1], 1.0])
    denom = float(normal_facing @ ray_dir)
    if abs(denom) < 1e-9:
        return None, None
    s = -forward / denom
    point = s * ray_dir
    return float(point[0]), float(point[1])


def rotation_angle_deg(R):
    """Angle of the rotation R represents, via the trace formula. Sanity check
    only: we commanded yaw=0.0 throughout, so this should be close to 0 -- it
    is NOT the wall-relative angle (that comes from the plane normal), just a
    check that the drone held the commanded attitude between frame A and B."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def cheirality_vote(R, t, inlier_a, inlier_b, K):
    """How many matched points end up in front of BOTH cameras for one
    candidate (R, t). This is exactly what recoverPose uses internally to
    pick the winner among the 4 candidates from decomposeEssentialMat, but it
    only ever reports the winner -- diagnosing why Essential Matrix
    occasionally picks the wrong one needs seeing all 4 vote counts."""
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P0, P1, inlier_a.T, inlier_b.T)
    pts3d_a = (pts4d[:3] / pts4d[3]).T
    depth_a = pts3d_a[:, 2]
    pts3d_b = (R @ pts3d_a.T + t.reshape(3, 1)).T
    depth_b = pts3d_b[:, 2]
    return int(((depth_a > 0) & (depth_b > 0)).sum())


def diagnose_essential_candidates(E, inlier_a, inlier_b, K):
    """Manually replicate recoverPose's own disambiguation to see the vote
    margin between the winning candidate and the runner-up -- a close margin
    would mean small amounts of noise (which matched points survived RANSAC,
    etc.) can flip which candidate wins between otherwise-identical runs."""
    R1, R2, t = cv2.decomposeEssentialMat(E)
    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    votes = [cheirality_vote(R, tt, inlier_a, inlier_b, K) for R, tt in candidates]
    ranked = sorted(votes, reverse=True)
    margin_pct = 100 * (ranked[0] - ranked[1]) / max(ranked[0], 1)
    print(f"Essential candidate cheirality votes (of {len(inlier_a)}): {votes} "
          f"-- winner leads runner-up by {margin_pct:.0f}%")
    return votes, margin_pct


def estimate_pose(pts_a, pts_b, K):
    """Steps 5-6: Essential Matrix -> pose -> triangulated points (unit scale)."""
    E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC,
                                    prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        raise SystemExit(f"findEssentialMat failed (E shape: {None if E is None else E.shape})")

    inlier_a = pts_a[mask.ravel() == 1]
    inlier_b = pts_b[mask.ravel() == 1]
    print(f"Essential matrix inliers: {len(inlier_a)}/{len(pts_a)}")
    if len(inlier_a) < 5:
        raise SystemExit("too few essential-matrix inliers to recover pose")

    diagnose_essential_candidates(E, inlier_a, inlier_b, K)

    n_front, R, t, pose_mask = cv2.recoverPose(E, inlier_a, inlier_b, K)
    print(f"recoverPose: {n_front} points in front of both cameras "
          f"(t is unit-length: this is the mono scale-ambiguity step 7 fixes)")
    # n_front IS len(keep) below; checked here (not after triangulating) since
    # fit_plane's SVD needs at least 3 points and crashes outright (uncaught
    # ValueError, not SystemExit) on an empty array -- hit live via
    # mono_multiview_node.py on a real flight: a close/ambiguous cheirality
    # vote (same failure mode as the lateral_0.6 Essential outlier in
    # motion_direction_sweep.py) left pose_mask all-False.
    if n_front < 3:
        raise SystemExit(f"too few points passed cheirality check ({n_front})")

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t])
    pts4d = cv2.triangulatePoints(P0, P1, inlier_a.T, inlier_b.T)
    pts3d_unit = (pts4d[:3] / pts4d[3]).T  # Nx3, camera-A frame, unit baseline

    keep = pose_mask.ravel() > 0
    return R, t, pts3d_unit[keep], inlier_a[keep]


def estimate_pose_homography(pts_a, pts_b, K):
    """Alternative to estimate_pose(), appropriate when the matched points lie
    on a single plane -- our kiosk wall, always, in this sim. Essential Matrix
    decomposition is known to be poorly conditioned for purely planar point
    sets (a real risk here, confirmed by the distance sweep: some pairs came
    back with a negative estimated distance -- a nonsensical result). Unlike
    estimate_pose(), this solves for the plane's normal directly from the 2D
    correspondences instead of triangulating noisy 3D points and fitting a
    plane to them afterward, so the wall-relative angle no longer depends on a
    second, separately-noisy estimation step.

    Returns (R, forward, lateral, vertical, yaw_deg, spread, inlier_count).
    forward/lateral/vertical/spread are scaled directly to meters here (unlike
    estimate_pose(), which returns unit-scale points for run() to scale) since
    the plane-relative point reconstruction needs the scale factor internally.
    """
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 3.0)
    if H is None:
        raise SystemExit("findHomography failed")
    inlier_count = int(mask.sum())
    print(f"Homography inliers: {inlier_count}/{len(pts_a)}")
    if inlier_count < 8:
        raise SystemExit("too few homography inliers")

    _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

    pts_a_norm = cv2.undistortPoints(pts_a.reshape(-1, 1, 2).astype(np.float32), K, None)
    pts_b_norm = cv2.undistortPoints(pts_b.reshape(-1, 1, 2).astype(np.float32), K, None)

    possible = cv2.filterHomographyDecompByVisibleRefpoints(
        Rs, Ns, pts_a_norm, pts_b_norm, pointsMask=mask)
    if possible is None or len(possible) == 0:
        raise SystemExit("no valid homography decomposition after filtering")
    idx = int(possible.ravel()[0])
    R = Rs[idx]
    t_internal = Ts[idx].ravel()   # translation / plane-depth (unscaled)
    n = Ns[idx].ravel()            # plane normal, camera-A frame, same internal scale

    inlier_a = pts_a[mask.ravel() == 1]
    return R, t_internal, n, inlier_a, inlier_count


def homography_to_metric(R, t_internal, n, inlier_a, K, real_baseline):
    """Scale the homography decomposition to meters using PX4's real baseline
    (step 7, same idea as estimate_pose()'s scale correction), then reconstruct
    each inlier point's 3D position by intersecting its camera ray with the
    known plane (n . X = 1 in the decomposition's internal units) rather than
    triangulating -- this leans on the planar assumption instead of fighting
    it, which is the whole point of using homography here."""
    scale = real_baseline / np.linalg.norm(t_internal)

    rays = cv2.undistortPoints(inlier_a.reshape(-1, 1, 2).astype(np.float64), K, None).reshape(-1, 2)
    rays = np.hstack([rays, np.ones((len(rays), 1))])  # Nx3, camera-A frame, unnormalized ray directions

    denom = rays @ n
    valid = np.abs(denom) > 1e-6
    s = np.full(len(rays), np.nan)
    s[valid] = 1.0 / denom[valid]
    pts3d_internal = s[:, None] * rays  # points on the plane, internal scale

    pts3d_m = pts3d_internal[valid] * scale
    if len(pts3d_m) < 3:
        # Same class of bug as estimate_pose()'s n_front guard: an empty/tiny
        # array here crashes outright (uncaught ValueError) a few lines down
        # at pts3d_m.max/min(axis=0) and .mean(axis=0), instead of failing
        # cleanly like every other degenerate case in this file.
        raise SystemExit(f"too few valid homography points on the plane ({len(pts3d_m)})")

    n_facing = -n if n[2] > 0 else n
    forward = float(-np.dot(n_facing, pts3d_m.mean(axis=0)))
    lateral, vertical = bbox_center_lateral_vertical(inlier_a[valid], K, n_facing, forward)
    # atan2(-nx, -nz), not atan2(nx, -nz): see the module-level note on
    # fit_plane() below -- this mirrors wall_geometry.py's yaw_err_from_R,
    # fixed there against a real closed-loop sign check (docs/PROGRESS.md,
    # 2026-09-19). The un-negated form is the OLD, wrong convention this file
    # used to copy.
    yaw_deg = float(np.degrees(np.arctan2(-n_facing[0], -n_facing[2])))
    spread = float(np.linalg.norm(pts3d_m.max(axis=0) - pts3d_m.min(axis=0)))
    return forward, lateral, vertical, yaw_deg, spread, n_facing


# Homography-primary, Essential-fallback, not a per-frame popularity contest
# between the two: this project's wall is always planar (kiosk wall), and
# mono_multiview_motion_direction_sweep.py + mono_multiview_repeat_check.py
# (vision/mono_multiview/README.md) found Homography stayed accurate
# (0.30-1.46deg rotation error) across a 3x range of matched-point counts and
# every pure-lateral/pure-forward direction tried, with exactly one
# direction-specific outlier (north+0.3/east+0.3, reproducible across 3
# repeats: 4.08-5.71deg) -- still bounded and far below Essential's own
# failure mode, a close cheirality-vote tie that produced a 12.26deg miss on
# a HIGH-inlier-count pure-lateral trial, unrelated to point count. Since
# Essential is occasionally either much better or much worse than Homography
# on the same frame pair with no simple predictor, picking "whichever is
# closer to PX4's attitude this frame" (the previous logic) just lets
# Essential win sometimes on noise. Defaulting to Homography and only
# falling back when it fails its OWN sanity check is more predictable.
HOMOGRAPHY_SANITY_THRESHOLD_DEG = 15.0  # generous vs the observed 0.3-5.7deg
                                          # range above; only trips for a
                                          # genuinely bad decomposition, not
                                          # normal noise.


def select_pose(essential, homography, true_cam_rot_deg):
    """Homography-primary: use it unless its own self-reported cam_rot
    disagrees badly with PX4's real IMU-based attitude estimate
    (quaternion_angle_deg -- real sensor data, not a simulation-only
    shortcut, so this carries over to production). That disagreement is a
    sanity check on Homography, not a contest against Essential -- see the
    module-level comment above for why treating the two as equal candidates
    picked per-frame is the wrong model for this planar-scene project.
    """
    agree_deg = rotation_angle_deg(homography["R"] @ essential["R"].T)
    e_err = abs(essential["cam_rot"] - true_cam_rot_deg)
    h_err = abs(homography["cam_rot"] - true_cam_rot_deg)
    if h_err > HOMOGRAPHY_SANITY_THRESHOLD_DEG:
        chosen, result = "Essential (Homography failed sanity check)", essential
    else:
        chosen, result = "Homography", homography
    return chosen, result, agree_deg, e_err, h_err


def compute_mono_pose(frame_a, frame_b, quat_a, quat_b, real_baseline, K, verbose=False):
    """Steps 1-9 (match -> Essential+Homography -> select) as a single call.

    Factored out of run() so mono_multiview_node.py's image callback (and any
    test harness) calls the exact same, already-validated pipeline instead of
    duplicating it -- see vision/mono_multiview/README.md for the diagnostic
    scripts that validated this sequence (sign fix, Homography-primary
    select_pose, etc.).

    Returns (chosen, result, essential_result, homography_result, extra):
      - result is the selected dict (forward/lateral/vertical/yaw_deg/spread/
        cam_rot/method), or None if matching or BOTH pose methods failed.
      - extra is {'agree_deg', 'e_err', 'h_err'} when both methods produced a
        result (so select_pose ran), else None (Essential-only fallback, or
        total failure).
    """
    true_cam_rot_deg = quaternion_angle_deg(quat_a, quat_b)
    log = print if verbose else (lambda *a, **k: None)

    log("-- Steps 1-4: matching frame A/B")
    try:
        _, pts_a, pts_b = detect_and_match(frame_a, frame_b)
    except SystemExit as e:
        log(f"-- matching FAILED: {e}")
        return None, None, None, None, None

    log("-- Steps 5-6: Essential Matrix + triangulation (unit scale)")
    essential_result = None
    try:
        R, t_unit, pts3d_unit, inlier_2d_a = estimate_pose(pts_a, pts_b, K)
        log("-- Step 7: scale correction using PX4 baseline")
        pts3d_m = pts3d_unit * real_baseline
        normal, centroid, spread = fit_plane(pts3d_m)
        forward = float(-np.dot(normal, centroid))
        lateral, vertical = bbox_center_lateral_vertical(inlier_2d_a, K, normal, forward)
        # See fit_plane()'s docstring: must be -normal[0], matching
        # wall_geometry.py's corrected yaw_err_from_R convention.
        yaw_deg = float(np.degrees(np.arctan2(-normal[0], -normal[2])))
        cam_rotation_deg = rotation_angle_deg(R)
        confident = spread >= SPREAD_MIN_M
        log(f"-- {len(pts3d_m)} scaled 3D points, spread={spread:.3f}m "
            f"({'CONFIDENT' if confident else f'LOW CONFIDENCE (< {SPREAD_MIN_M}m)'})")
        log(f"-- ESTIMATED [Essential] forward={forward:.3f}m lateral={lateral:.3f}m "
            f"vertical={vertical:.3f}m yaw={yaw_deg:+.1f}deg CONFIDENT={confident}")
        essential_result = {"method": "Essential", "R": R, "normal": normal, "forward": forward,
                             "lateral": lateral, "vertical": vertical, "yaw_deg": yaw_deg,
                             "spread": spread, "cam_rot": cam_rotation_deg}
    except SystemExit as e:
        log(f"-- Essential pose FAILED: {e}")

    log("-- Alternative: Homography-based pose (planar-scene-specific)")
    homography_result = None
    try:
        R_h, t_h_internal, n_h, inlier_a_h, h_inliers = estimate_pose_homography(pts_a, pts_b, K)
        h_forward, h_lateral, h_vertical, h_yaw_deg, h_spread, h_normal = homography_to_metric(
            R_h, t_h_internal, n_h, inlier_a_h, K, real_baseline)
        h_cam_rotation_deg = rotation_angle_deg(R_h)
        log(f"-- ESTIMATED [Homography] forward={h_forward:.3f}m lateral={h_lateral:.3f}m "
            f"vertical={h_vertical:.3f}m yaw={h_yaw_deg:+.1f}deg spread={h_spread:.3f}m "
            f"cam_rot={h_cam_rotation_deg:.1f}deg")
        homography_result = {"method": "Homography", "R": R_h, "normal": h_normal, "forward": h_forward,
                              "lateral": h_lateral, "vertical": h_vertical, "yaw_deg": h_yaw_deg,
                              "spread": h_spread, "cam_rot": h_cam_rotation_deg}
    except SystemExit as e:
        log(f"-- Homography pose FAILED: {e}")

    if essential_result is None and homography_result is None:
        return None, None, None, None, None
    if homography_result is None:
        return "Essential (Homography unavailable)", essential_result, essential_result, None, None
    if essential_result is None:
        return "Homography (Essential unavailable)", homography_result, None, homography_result, None

    chosen, result, agree_deg, e_err, h_err = select_pose(
        essential_result, homography_result, true_cam_rot_deg)
    extra = {"agree_deg": agree_deg, "e_err": e_err, "h_err": h_err}
    return chosen, result, essential_result, homography_result, extra


async def run(K):
    from mavsdk import System
    from mavsdk.offboard import OffboardError, PositionNedYaw
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
    await goto(drone, APPROACH_NORTH, 0.0, 0.0, f"North {APPROACH_NORTH}m (approach)", SETTLE_SEC)

    loop = asyncio.get_running_loop()
    print("-- Capturing frame A + position")
    frame_a = await loop.run_in_executor(None, capture_frame, "a")
    pos_a = await get_position_ned(drone)
    quat_a = await get_attitude_quaternion(drone)
    t_a = time.time()
    print(f"-- frame A wall-clock time: {t_a:.3f}  north={pos_a.north_m:.3f}m")

    baseline_target_n = APPROACH_NORTH + BASELINE_NORTH
    baseline_target_e = BASELINE_EAST
    await goto(drone, baseline_target_n, baseline_target_e, 0.0,
               f"North {baseline_target_n}m East {baseline_target_e}m (baseline step)", SETTLE_SEC)

    print("-- Capturing frame B + position")
    frame_b = await loop.run_in_executor(None, capture_frame, "b")
    pos_b = await get_position_ned(drone)
    quat_b = await get_attitude_quaternion(drone)
    t_b = time.time()
    print(f"-- frame B wall-clock time: {t_b:.3f}  north={pos_b.north_m:.3f}m")

    true_cam_rot_deg = quaternion_angle_deg(quat_a, quat_b)
    print(f"-- PX4-measured attitude change (ground truth, real IMU data): {true_cam_rot_deg:.1f}deg "
          f"between A and B")

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

    real_baseline = float(np.linalg.norm([
        pos_b.north_m - pos_a.north_m,
        pos_b.east_m - pos_a.east_m,
        pos_b.down_m - pos_a.down_m,
    ]))
    print(f"-- PX4-measured baseline (ground truth): {real_baseline:.3f}m "
          f"(commanded: {BASELINE_NORTH:.3f}m)")

    # Geometric ground truth for frame A, independent of BOTH vision pipelines
    # (ours and aruco_pnp's): the wall's world pose is fixed at (0, 3, 1.5) in
    # kiosk.sdf, and PX4's local NED origin is set at the spawn point (no pose
    # override is passed when px4-rc.gzsim spawns the model), which is world
    # (0, 0, ~). So world_y ~= north_m and world_x ~= east_m, and the
    # wall-facing distance/lateral follow directly from PX4's own GPS/IMU
    # position estimate -- no camera involved at all, so it stays valid even
    # at ranges where both PnP and our own reconstruction may be degrading.
    #
    # lateral here must be wall_east - camera_east, NOT camera_east alone:
    # both aruco_pnp_node's tvec[0] and our own bbox_center_lateral_vertical()
    # report the TARGET's position in the camera's own frame (positive =
    # wall is to the camera's right), which is the geometric NEGATIVE of the
    # camera's own east offset when the wall sits at east=0 -- moving the
    # camera east makes the wall appear to its left, not further right. A
    # sweep across known east offsets (mono_multiview_lateral_bias_sweep.py)
    # confirmed the estimate tracks -pos_a.east_m, not +pos_a.east_m: this
    # line was comparing against the wrong sign of ground truth, not a bug in
    # the estimate itself. geom_forward already got this right (3.0 is the
    # wall's own north coordinate, subtracted from camera_north the same way).
    geom_forward = 3.0 - pos_a.north_m
    geom_lateral = 0.0 - pos_a.east_m  # wall world_x=0 (kiosk.sdf) minus camera east
    print(f"-- GEOMETRIC ground truth (from PX4 position + known wall pose, no vision): "
          f"forward={geom_forward:.3f}m lateral={geom_lateral:.3f}m")

    chosen, result, essential_result, homography_result, extra = compute_mono_pose(
        frame_a, frame_b, quat_a, quat_b, real_baseline, K, verbose=True)
    if result is None:
        print("-- Both Essential and Homography FAILED, no pose this pair")
        return
    if extra is not None:
        print(f"-- R agreement between methods: {extra['agree_deg']:.1f}deg apart")
        print(f"-- vs PX4 attitude ({true_cam_rot_deg:.1f}deg): "
              f"Essential off by {extra['e_err']:.1f}deg, Homography off by {extra['h_err']:.1f}deg")
    print(f"-- SELECTED [{chosen}] forward={result['forward']:.3f}m lateral={result['lateral']:.3f}m "
          f"vertical={result['vertical']:.3f}m yaw={result['yaw_deg']:+.1f}deg")


def main():
    print("-- Getting camera intrinsics")
    K = capture_camera_info()
    print(f"K=\n{K}")
    asyncio.run(run(K))


if __name__ == "__main__":
    main()
