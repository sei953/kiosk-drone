#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""위치계산부 + 판단부: 검출된 마커를 벽별(ID 그룹)로 묶어 각 벽을 통합
solvePnP(IPPE)로 풀고, target_wall 파라미터가 지정한 벽이 보이면 그 벽의
pose만 /target/pose 로 발행한다 (판단부: 타겟 벽이 안 보이면 계속 탐색)."""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge

from kiosk_vision.wall_geometry import (
    MARKER_SIZE, CORNER_LAYOUT, WALL_ID_BASE,
    marker_id_to_wall, marker_corners_3d, rotmat_to_quat, yaw_err_from_R, wrap_deg_diff,
)

ARUCO_DICT = cv2.aruco.DICT_4X4_50
REPROJ_ERR_WARN = 2.0
# 4(벽 전부) -> 3: flip 방어는 이제 solvePnPGeneric 시간적 일관성(a5f025c)이 소스에서
# 담당하므로, 마커 수 게이트의 원래 목적(flip 방지)은 이미 달성됨. 3마커도 넓은
# baseline(코너 배치)이라 pose가 충분히 안정적 -- 미세 자세 흔들림으로 마커 1개가
# 프레임 경계를 넘나들 때마다 락이 깨지는 것을 줄이기 위해 완화.
REQUIRED_MARKERS = 3

# 연속성(점프) 게이트 (b): 직전 채택 pose 대비 이만큼 넘게 튀면 solvePnP의 평면
# pose ambiguity(거울 해)로 간주하고 버린다 — reproj_err가 낮아도 버림.
JUMP_GATE_YAW_DEG = 30.0
JUMP_GATE_LATERAL_M = 0.5
JUMP_GATE_RESET_S = 2.0  # 이만큼 공백이 있었으면 재포착으로 보고 게이트 없이 수용


class ArucoPnPNode(Node):
    def __init__(self):
        super().__init__('aruco_pnp_node')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('pose_topic', '/target/pose')
        self.declare_parameter('visible_topic', '/target/visible')
        self.declare_parameter('target_wall', '동')  # 북/동/남/서
        # 지상 정적 진단(부호 확인 등)에서 임시로 낮춰 쓸 수 있게 파라미터화.
        # 기본값은 REQUIRED_MARKERS(3) 그대로.
        self.declare_parameter('required_markers', REQUIRED_MARKERS)
        self.required_markers = int(self.get_parameter('required_markers').value)
        img_topic = self.get_parameter('image_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        visible_topic = self.get_parameter('visible_topic').value
        self.target_wall = self.get_parameter('target_wall').value
        if self.target_wall not in WALL_ID_BASE:
            raise ValueError(f"target_wall='{self.target_wall}' invalid, must be one of {list(WALL_ID_BASE)}")

        self.bridge = CvBridge()
        self.K = None
        self.D = None
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
        else:
            self.detector = None
            self.aruco_params = cv2.aruco.DetectorParameters_create()  # OpenCV 4.6 폴백

        self.create_subscription(CameraInfo, info_topic, self.on_camera_info, 10)
        self.create_subscription(Image, img_topic, self.on_image, qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.visible_pub = self.create_publisher(Bool, visible_topic, 10)
        self.latest_pose = None
        self.last_accepted = None  # {'yaw_err','lateral','time'} — 점프 게이트 기준점
        self.get_logger().info(
            f'aruco_pnp_node start img={img_topic} info={info_topic} target_wall={self.target_wall}')

    def on_camera_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64)
        if self.D.size == 0:
            self.D = np.zeros((5,), dtype=np.float64)

    def _detect(self, gray):
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids

    def _group_by_wall(self, ids, corners):
        """검출 마커를 ID로 벽별 그룹핑 -> {wall_name: (obj_pts, img_pts)}"""
        groups = {}
        for mid, cn in zip(ids, corners):
            mid = int(mid)
            wall_name, corner_off = marker_id_to_wall(mid)
            if wall_name is None:
                continue
            cx, cy = CORNER_LAYOUT[corner_off]
            obj_pts, img_pts = groups.setdefault(wall_name, ([], []))
            obj_pts += marker_corners_3d(cx, cy, MARKER_SIZE)
            img_pts += list(cn.reshape(4, 2))
        return groups

    def _last_accepted_age(self):
        """last_accepted가 없거나 너무 낡았으면 None (재포착 취급 기준 공용)."""
        if self.last_accepted is None:
            return None
        age = (self.get_clock().now() - self.last_accepted['time']).nanoseconds * 1e-9
        return age if age <= JUMP_GATE_RESET_S else None

    def _solve_wall(self, wall_name, obj_pts, img_pts):
        if len(obj_pts) < 4:
            return None
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)
        # solvePnP(단일해) 대신 solvePnPGeneric 사용: 평면 마커(IPPE)는 근본적으로
        # 두 개의 유사-타당 거울 해를 반환할 수 있다 -- 그중 하나를 reprojection
        # error만으로 고르면 애매한 프레임에서 flip이 난다. 직전 채택 pose(같은 벽)와
        # yaw가 가장 가까운 해를 골라 시간적 일관성으로 flip을 소스에서 제거한다.
        n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj_pts, img_pts, self.K, self.D,
                                                          flags=cv2.SOLVEPNP_IPPE)
        if n_sol < 1:
            self.get_logger().warn(f'[{wall_name}] solvePnP failed')
            return None

        candidates = []
        for rvec, tvec in zip(rvecs, tvecs):
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.D)
            reproj_err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))
            tv = tvec.flatten()
            R, _ = cv2.Rodrigues(rvec)
            candidates.append(dict(
                wall=wall_name, forward=float(tv[2]), lateral=float(tv[0]), vertical=float(tv[1]),
                dist=float(np.linalg.norm(tv)), yaw_err=yaw_err_from_R(R), reproj_err=reproj_err,
                n_markers=len(obj_pts) // 4, R=R, tv=tv,
            ))

        if len(candidates) == 1:
            return candidates[0]

        ref = self.last_accepted if (wall_name == self.target_wall and self._last_accepted_age() is not None) else None
        if ref is not None:
            return min(candidates, key=lambda c: abs(wrap_deg_diff(c['yaw_err'], ref['yaw_err'])))
        return min(candidates, key=lambda c: c['reproj_err'])  # 기준점 없음(첫 검출 등) -> 기본값

    def _passes_jump_gate(self, candidate):
        """(b) 연속성 게이트: temporal disambiguation을 뚫고 나온 잔여 튐을 잡는 2차 방어선."""
        if self._last_accepted_age() is None:
            return True
        dyaw = abs(wrap_deg_diff(candidate['yaw_err'], self.last_accepted['yaw_err']))
        dlat = abs(candidate['lateral'] - self.last_accepted['lateral'])
        return dyaw <= JUMP_GATE_YAW_DEG and dlat <= JUMP_GATE_LATERAL_M

    def on_image(self, msg):
        if self.K is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect(gray)
        if ids is None:
            self.visible_pub.publish(Bool(data=False))
            return
        ids = ids.flatten()

        groups = self._group_by_wall(ids, corners)
        results = {}
        for wall_name, (obj_pts, img_pts) in groups.items():
            r = self._solve_wall(wall_name, obj_pts, img_pts)
            if r is not None:
                results[wall_name] = r

        if results:
            seen = ', '.join(f"{w}({r['n_markers']}mk,{r['reproj_err']:.1f}px)" for w, r in results.items())
            self.get_logger().info(f'[walls seen] {seen}')

        target = results.get(self.target_wall)

        # (a) 마커 수 게이트: 벽 하나(4마커, 16점) 전부 보일 때만 pose 후보로 인정.
        if target is not None and target['n_markers'] < self.required_markers:
            self.get_logger().info(
                f"[gate] target={self.target_wall} {target['n_markers']}mk < {self.required_markers}mk -> 보류(마지막 pose 유지)",
                throttle_duration_sec=1.0)
            target = None

        # (b) 점프 게이트: 마커 수 조건은 통과했지만 직전 채택값 대비 비물리적으로 튀면 버림.
        if target is not None and not self._passes_jump_gate(target):
            self.get_logger().warn(
                f"[gate] target={self.target_wall} yaw/lateral 점프 감지(flip 의심) -> 이 프레임 버림 "
                f"(yaw {target['yaw_err']:+.1f}deg, lat {target['lateral']:+.3f}m)")
            target = None

        self.visible_pub.publish(Bool(data=bool(target)))
        if target is None:
            return

        self.latest_pose = target
        self.last_accepted = dict(yaw_err=target['yaw_err'], lateral=target['lateral'],
                                   time=self.get_clock().now())
        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = f'wall_{self.target_wall}'
        ps.pose.position.x = target['lateral']
        ps.pose.position.y = target['vertical']
        ps.pose.position.z = target['forward']
        qx, qy, qz, qw = rotmat_to_quat(target['R'])
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pose_pub.publish(ps)

        # 주의: get_logger().warn/.info를 변수에 담아뒀다 호출하면("lvl = ... ; lvl(msg)")
        # 같은 콜사이트에서 severity가 달라져 rclpy가 ValueError("Logger severity cannot
        # be changed between calls")를 던지며 노드가 죽는다 (실측: reproj_err가 처음
        # 2.0px를 넘은 순간 크래시, docs/PROGRESS.md 참고). if/else로 분기해 각각 고정
        # severity로 직접 호출한다.
        msg = (f"[PnP target={self.target_wall}] {target['n_markers']}mk | fwd {target['forward']:.3f}m "
               f"| lat {target['lateral']:+.3f} | yaw {target['yaw_err']:+.1f}deg | reproj {target['reproj_err']:.2f}px")
        if target['reproj_err'] > REPROJ_ERR_WARN:
            self.get_logger().warn(msg)
        else:
            self.get_logger().info(msg)

    def get_pose(self):
        return self.latest_pose


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPnPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
