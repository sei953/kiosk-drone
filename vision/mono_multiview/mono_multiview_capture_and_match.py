#!/usr/bin/env python3
"""Two-frame monocular baseline test for the 이동 구간 (movement zone) 방향 B.

Captures frame A, moves the drone forward a fixed step, captures frame B,
then runs ORB detection + BFMatcher + RANSAC (Fundamental matrix) to check
whether the matches are clean enough to build Essential-matrix pose recovery
and triangulation on top of (steps 5-8 of the mono-multiview plan).

TEMPORARY validation script, not the production node — see fly_to_marker_and_capture.py
for the pattern this will move to once matching is proven reliable: a persistent
ROS2 node subscribing to the camera continuously, like aruco_pnp_node.py, rather
than a one-shot script. BASELINE_NORTH is hardcoded for now; production will read
the real frame-to-frame displacement from PX4 vehicle_local_position for scale
correction (step 7) instead.
"""

import asyncio
import threading
from pathlib import Path

import cv2
import numpy as np
from gz.msgs10.image_pb2 import Image
import gz.transport13 as gz_transport
# mavsdk (System/OffboardError/PositionNedYaw) is imported lazily inside
# goto()/run() below, not here: this module's detect_and_match()/capture_frame()
# are reused by mono_multiview_node.py (a ROS2 node with no mavsdk dependency,
# now that px4_msgs covers telemetry), and `from ... import detect_and_match`
# still executes every top-level import in this file -- a module-level mavsdk
# import would make the node fail to import in an environment that correctly
# doesn't have mavsdk installed.

RGB_TOPIC = "/world/kiosk/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALT = -1.5           # NED z for 1.5m altitude
# Chosen to keep the whole wall (all 4 markers) in frame for BOTH captures.
# north=1.4m was already borderline -- visually confirmed frame A (1.47m) has
# all 4 markers but frame B (2.04m, closer) only has the top 2: the bottom
# pair fell outside the FOV as the wall filled more of the frame. That shrunk
# the two frames' shared visible area to a single small cluster, which is what
# made the plane-fit yaw unreliable (see SPREAD_MIN_M). Staying farther back
# (1.0m/1.3m) keeps both frames comfortably inside the "full wall visible"
# range from the position sweep (1.0m: 334-349kp, well above the 0kp cutoff at
# <=0.8m) without ever crossing into the crops-the-wall territory near 2.0m.
APPROACH_NORTH = 0.3  # wall is at world y=3 (kiosk.sdf), spawn ~= world (0,0),
                      # so this is ~2.7m from the wall -- testing whether the
                      # true "far" end of the movement zone is visible now
                      # that the stability-wait fix is in (earlier low-north
                      # readings showing 0 keypoints predate that fix and may
                      # have just caught the vehicle mid-transit, not a real
                      # visibility limit).
BASELINE_NORTH = 0.0  # frame A -> frame B forward step [m], hardcoded for this test
                      # (mostly-forward, small-lateral ratio test: keeping the
                      # total baseline magnitude close to the ~0.6-0.7m that
                      # worked great pure-lateral, but shifting the ratio
                      # toward forward to see whether degradation tracks the
                      # forward:lateral ratio or just total point count/overlap)
BASELINE_EAST = 0.6  # frame A -> frame B sideways step [m]; pure-forward motion
                      # gives weak parallax for points near the image center
                      # (moving toward a point doesn't shift it across the
                      # frame much) -- mono_multiview_pose_estimate.py's yaw
                      # has been noisy across every distance tried so far, and
                      # this has only ever been tested with BASELINE_EAST=0.
                      # A sideways component should condition triangulation/
                      # homography decomposition better for the same total
                      # baseline length. A forward+east combined run got a
                      # great Essential result (yaw -4.9deg) but a badly wrong
                      # Homography one (-50deg, cam_rot disagreed with
                      # Essential's for the same pair) -- isolating pure east
                      # motion here to see whether that's about the sideways
                      # component specifically, or the wall's 2cm real
                      # thickness (kiosk_wall/model.sdf) breaking the
                      # single-plane assumption once viewed at an angle.
SETTLE_SEC = 10  # max wait for wait_until_stable(); PX4's offboard position
                 # controller in this sim regularly takes longer than 4s to
                 # converge on a new setpoint (observed repeatedly), so 4s was
                 # timing out on almost every waypoint


def capture_frame(tag):
    """Blocking single-frame grab via gz-transport (same pattern as capture_and_detect.py)."""
    frame = {}
    got = threading.Event()

    def cb(msg: Image):
        if got.is_set():
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape((msg.height, msg.width, 3))  # RGB_INT8
        frame["img"] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        got.set()

    node = gz_transport.Node()
    node.subscribe(Image, RGB_TOPIC, cb)
    if not got.wait(timeout=10):
        raise SystemExit(f"failed to capture frame {tag}")

    img = frame["img"]
    cv2.imwrite(str(OUT_DIR / f"mono_frame_{tag}.png"), img)
    print(f"saved {OUT_DIR / f'mono_frame_{tag}.png'}  shape={img.shape}")
    return img


def detect_and_match(img_a, img_b):
    orb = cv2.ORB_create(nfeatures=2000)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    print(f"keypoints: A={len(kp_a)} B={len(kp_b)}")

    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        raise SystemExit("too few keypoints to match")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_a, des_b), key=lambda m: m.distance)
    print(f"raw matches: {len(matches)}")

    if len(matches) < 8:
        raise SystemExit("too few matches for RANSAC (need >= 8)")

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])

    _, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.0, 0.99)
    inlier_count = int(mask.sum()) if mask is not None else 0
    print(f"RANSAC inliers: {inlier_count}/{len(matches)} "
          f"({100 * inlier_count / len(matches):.1f}%)")

    inlier_matches = [m for m, keep in zip(matches, mask.ravel()) if keep]
    vis = cv2.drawMatches(img_a, kp_a, img_b, kp_b, inlier_matches[:60], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(OUT_DIR / "mono_matches.png"), vis)
    print(f"saved {OUT_DIR / 'mono_matches.png'}")

    print(f"MATCH_SUCCESS={inlier_count >= 20}")
    inlier_pts_a = pts_a[mask.ravel() == 1]
    inlier_pts_b = pts_b[mask.ravel() == 1]
    return inlier_matches, inlier_pts_a, inlier_pts_b


async def wait_until_stable(drone, target_north, target_east, target_down,
                             pos_tolerance=0.1, vel_threshold=0.1, timeout=8.0):
    """Poll position+velocity until the vehicle has both ARRIVED at the target
    and STOPPED, or timeout.

    Velocity alone is not enough: it reads near-zero both when the vehicle has
    settled at the target AND in the instant right after a new setpoint is
    issued, before the controller has started accelerating toward it. A first
    version that checked only velocity got fooled by the second case and
    captured frames at north=0.01m while the commanded target was 1.4m/2.0m --
    the vehicle had barely left its previous position. Checking position error
    too rules that out.
    """
    start = asyncio.get_event_loop().time()
    async for pv in drone.telemetry.position_velocity_ned():
        pos_err = ((pv.position.north_m - target_north) ** 2
                   + (pv.position.east_m - target_east) ** 2
                   + (pv.position.down_m - target_down) ** 2) ** 0.5
        speed = (pv.velocity.north_m_s ** 2 + pv.velocity.east_m_s ** 2
                 + pv.velocity.down_m_s ** 2) ** 0.5
        if pos_err < pos_tolerance and speed < vel_threshold:
            return
        if asyncio.get_event_loop().time() - start > timeout:
            print(f"  !! wait_until_stable timed out after {timeout}s "
                  f"(pos_err={pos_err:.3f}m, speed={speed:.3f}m/s), proceeding anyway")
            return


async def goto(drone, north, east, yaw, label, hold):
    from mavsdk.offboard import PositionNedYaw
    print(f"-- {label}")
    await drone.offboard.set_position_ned(PositionNedYaw(north, east, ALT, yaw))
    await asyncio.sleep(1.0)  # let the controller start responding before polling stability
    await wait_until_stable(drone, north, east, ALT, timeout=hold)


async def run():
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
    await goto(drone, APPROACH_NORTH, 0.0, 0.0, f"North {APPROACH_NORTH}m (approach, wall in frame)", SETTLE_SEC)

    print("-- Capturing frame A")
    loop = asyncio.get_running_loop()
    frame_a = await loop.run_in_executor(None, capture_frame, "a")

    baseline_target = APPROACH_NORTH + BASELINE_NORTH
    await goto(drone, baseline_target, 0.0, 0.0, f"North {baseline_target}m (baseline step)", SETTLE_SEC)

    print("-- Capturing frame B")
    frame_b = await loop.run_in_executor(None, capture_frame, "b")

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

    print("-- Matching frame A/B")
    detect_and_match(frame_a, frame_b)  # returns (matches, pts_a, pts_b); see mono_multiview_pose_estimate.py


if __name__ == "__main__":
    asyncio.run(run())
