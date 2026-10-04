#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""get_pose() 소스를 PnP(aruco_pnp_node)에서 단안 다시점(mono multiview)으로
교체하는 ROS2 노드. SCRUM-25(마커 폐루프)에 SCRUM-22(단안 다시점)를 통합하는
작업의 vision 쪽 절반 -- approach_control_node.py는 /target/pose +
/target/visible을 aruco_pnp_node와 동일한 규약으로 받으므로 그대로 재사용한다:

  - PoseStamped.pose.position: x=lateral, y=vertical, z=forward (카메라 프레임)
  - PoseStamped.pose.orientation: 벽 normal이 담긴 쿼터니언. approach_control_node가
    여기서 yaw_err_from_R로 yaw를 역산하므로, scalar yaw_deg만 보내는 게 아니라
    normal을 담은 쿼터니언을 실제로 채워야 한다 (_normal_to_quat 참고).
  - PoseStamped.header.frame_id = f'wall_{target_wall}' -- 이게 안 맞으면
    approach_control_node가 조용히 버린다 (on_target_pose의 frame_id 체크).
  - /target/visible(Bool): pose가 안 나온 프레임에도 매번 발행 (aruco_pnp_node와
    동일 패턴) -- approach_control_node의 target_lost()는 "마지막 pose 수신
    시각"을 보되, 어쨌든 두 노드가 같은 모양으로 행동해야 한다.

aruco_pnp_node와 달리 이 노드는 절대 스케일을 마커 한 변 길이가 아니라 "두
프레임 사이 실제 이동거리(PX4 로컬 위치 델타)"로 얻는다 (mono_multiview
계획의 step 7). approach_control_node.py가 이미 px4_msgs로 PX4 텔레메트리를
받고 있으므로, 이 노드도 같은 방식(VehicleLocalPosition/VehicleAttitude 구독)을
쓴다 -- 예전 버전은 px4_msgs가 없다고 보고 MAVSDK 스레드를 따로 띄웠는데,
이제 그 인프라가 있으니 중복 연결을 없앤다.

PROBE(좌우로 움직여서 baseline 확보)는 판단부의 몫이다 -- 이 노드는 스스로
움직이지 않고, "마지막 keyframe 대비 충분히 이동했다"고 판단되면 그때부터
pose를 내기 시작한다. 이동이 부족하면 pose 없이 visible=False만 발행하므로,
판단부는 "pose가 안 들어온다"를 "PROBE가 필요하다"는 신호로 쓸 수 있다.

벽 판별(target_wall)은 aruco_pnp_node처럼 마커 ID로 어느 벽인지 식별하는 게
아니라, 설정된 target_wall을 그냥 그대로 믿고 찍는다 -- ORB 매칭은 마커 ID
개념이 없어 "지금 보고 있는 게 어느 벽인지" 스스로 판별할 수 없다. 여러 벽을
오가는 SEARCH 단계에서는 이 노드의 pose를 신뢰하지 말 것.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

from kiosk_vision.wall_geometry import WALL_ID_BASE, rotmat_to_quat

# vision/mono_multiview/*는 아직 별도 패키지가 아니라 검증 스크립트 모음이라
# ament_python으로 설치되지 않는다. 이미 검증된 파이프라인(compute_mono_pose,
# detect_and_match)을 복제하지 않으려고 경로만 추가해서 그대로 가져다 쓴다 --
# mono_multiview가 정식 패키지로 승격되면 이 sys.path 트릭은 정리해야 한다.
_MONO_MULTIVIEW_DIR = str(Path(__file__).resolve().parents[2] / "mono_multiview")
if _MONO_MULTIVIEW_DIR not in sys.path:
    sys.path.insert(0, _MONO_MULTIVIEW_DIR)

from mono_multiview_capture_and_match import detect_and_match  # noqa: E402
from mono_multiview_pose_estimate import compute_mono_pose  # noqa: E402

MIN_BASELINE_M = 0.15  # 이만큼 실제로 움직여야 삼각측량/homography가 의미 있는
                        # parallax를 얻는다 (mono_multiview 검증 스크립트들이
                        # 쓴 0.3~1.0m 범위보다 낮게 잡아 keyframe을 더 자주
                        # 갱신하되, 노이즈만 잡는 수준의 미세 이동은 걸러낸다).


class _Quat:
    """mono_multiview_pose_estimate.quaternion_angle_deg가 기대하는 .w/.x/.y/.z
    속성 모양으로 px4_msgs VehicleAttitude.q(단순 [w,x,y,z] 배열)를 감싼다 --
    그 함수는 어떤 텔레메트리 소스를 쓰든 상관없게 그대로 둔다."""
    def __init__(self, q):
        self.w, self.x, self.y, self.z = float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _normal_to_quat(normal):
    """평면 normal(카메라 프레임, 카메라 쪽을 향하도록 이미 정규화됨) 하나로부터
    쿼터니언을 만든다. approach_control_node의 yaw_err_from_R(R)은 R의 3번째
    컬럼(R @ [0,0,1])만 읽으므로, 그 컬럼이 normal이기만 하면 나머지 두 축은
    임의의 직교 기저로 채워도 yaw_err 역산 결과는 동일하다 -- aruco_pnp_node의
    solvePnP R처럼 벽의 실제 in-plane 회전(roll)까지 복원할 필요가 없다."""
    n = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(ref, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)
    R = np.column_stack([x_axis, y_axis, n])
    return rotmat_to_quat(R)


_PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class MonoMultiviewNode(Node):
    def __init__(self):
        super().__init__('mono_multiview_node')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('pose_topic', '/target/pose')
        self.declare_parameter('visible_topic', '/target/visible')
        self.declare_parameter('target_wall', '동')  # 북/동/남/서 -- approach_control_node와 맞출 것
        self.declare_parameter('min_baseline_m', MIN_BASELINE_M)

        img_topic = self.get_parameter('image_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        visible_topic = self.get_parameter('visible_topic').value
        self.target_wall = self.get_parameter('target_wall').value
        if self.target_wall not in WALL_ID_BASE:
            raise ValueError(f"target_wall='{self.target_wall}' invalid, must be one of {list(WALL_ID_BASE)}")
        self.frame_id = f'wall_{self.target_wall}'
        self.min_baseline_m = float(self.get_parameter('min_baseline_m').value)

        self.bridge = CvBridge()
        self.K = None
        self.latest_ned = None   # (north_m, east_m, down_m) from VehicleLocalPosition
        self.latest_quat = None  # _Quat from VehicleAttitude

        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
                                  self.on_local_position, _PX4_QOS)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude',
                                  self.on_attitude, _PX4_QOS)

        self.keyframe = None  # {'frame', 'ned', 'quat'}

        self.create_subscription(CameraInfo, info_topic, self.on_camera_info, 10)
        self.create_subscription(Image, img_topic, self.on_image, qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.visible_pub = self.create_publisher(Bool, visible_topic, 10)
        self.get_logger().info(
            f'mono_multiview_node start img={img_topic} info={info_topic} '
            f'target_wall={self.target_wall} min_baseline={self.min_baseline_m}m')

    def on_local_position(self, msg):
        if msg.xy_valid and msg.z_valid:
            self.latest_ned = (msg.x, msg.y, msg.z)

    def on_attitude(self, msg):
        self.latest_quat = _Quat(msg.q)

    def on_camera_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def on_image(self, msg):
        if self.K is None or self.latest_ned is None or self.latest_quat is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        ned = self.latest_ned
        quat = self.latest_quat

        if self.keyframe is None:
            self.keyframe = {'frame': frame, 'ned': ned, 'quat': quat}
            self.visible_pub.publish(Bool(data=False))
            return

        kf_ned = self.keyframe['ned']
        baseline = float(np.linalg.norm([
            ned[0] - kf_ned[0], ned[1] - kf_ned[1], ned[2] - kf_ned[2],
        ]))
        if baseline < self.min_baseline_m:
            # PROBE(움직여서 baseline 확보)는 판단부의 결정이다 -- 여기서는
            # 그냥 기다린다. keyframe을 안 갈아치우는 게 중요: 매 프레임마다
            # keyframe을 현재 프레임으로 덮으면 baseline이 영원히 0 근처에
            # 머물러 절대 pose를 못 낸다.
            self.visible_pub.publish(Bool(data=False))
            return

        try:
            _, pts_a, pts_b = detect_and_match(self.keyframe['frame'], frame)
        except SystemExit as e:
            # Refresh the keyframe even on failure, not just on success: if
            # the ORIGINAL keyframe was bad (e.g. captured right at takeoff
            # before the wall was in view -- zero ORB keypoints, found live
            # on the markerless mockup wall test), retrying it against every
            # later frame fails forever and pose never recovers for the rest
            # of the flight, even once good frames start arriving. A stale
            # keyframe is strictly worse than a fresh one: worst case, a
            # fresh keyframe just costs one more baseline-accumulation cycle.
            self.get_logger().warn(f'matching failed, refreshing keyframe: {e}')
            self.keyframe = {'frame': frame, 'ned': ned, 'quat': quat}
            self.visible_pub.publish(Bool(data=False))
            return

        try:
            chosen, result, _, _, extra = compute_mono_pose(
                self.keyframe['frame'], frame, self.keyframe['quat'], quat,
                baseline, self.K, verbose=False)
        except Exception as e:
            # compute_mono_pose/estimate_pose*는 "실패"를 SystemExit로 신호하지만,
            # 실제 비행에서 cheirality 체크 후 포인트가 0개 남는 등 거기서 못 잡은
            # 엣지케이스가 ValueError 등으로 새어나온 적이 있다 (n_front 가드 추가로
            # 그 특정 케이스는 고쳤지만, 한 프레임의 예외가 노드 전체를 죽이게
            # 두지 않는다 -- 다음 프레임에서 복구될 수 있는 일시적 문제여야 한다).
            self.get_logger().error(f'compute_mono_pose crashed unexpectedly: {e}')
            self.keyframe = {'frame': frame, 'ned': ned, 'quat': quat}
            self.visible_pub.publish(Bool(data=False))
            return

        # 다음 keyframe은 지금 이 프레임으로 갱신한다 -- 성공/실패 여부와
        # 무관하게: 실패해도 옛 keyframe을 계속 붙들고 있으면 baseline이
        # 계속 커져 다음 시도의 parallax 조건은 더 유리해지지만, 그만큼
        # FOV 겹침은 줄어든다 (motion_direction_sweep에서 확인한 트레이드
        # 오프). 매 프레임 슬라이딩이 이 트레이드오프에서 더 안전한 기본값.
        self.keyframe = {'frame': frame, 'ned': ned, 'quat': quat}

        if result is None:
            self.get_logger().warn('Essential and Homography both failed for this pair')
            self.visible_pub.publish(Bool(data=False))
            return

        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = self.frame_id
        # aruco_pnp_node.py와 동일 규약: x=lateral(camera-X), y=vertical
        # (camera-Y), z=forward(camera-Z).
        ps.pose.position.x = float(result['lateral'])
        ps.pose.position.y = float(result['vertical'])
        ps.pose.position.z = float(result['forward'])
        qx, qy, qz, qw = _normal_to_quat(result['normal'])
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pose_pub.publish(ps)
        self.visible_pub.publish(Bool(data=True))

        extra_str = f" e_err={extra['e_err']:.1f} h_err={extra['h_err']:.1f}" if extra else ""
        self.get_logger().info(
            f'[{chosen}] baseline={baseline:.2f}m fwd={result["forward"]:.2f}m '
            f'lat={result["lateral"]:+.2f}m yaw={result["yaw_deg"]:+.1f}deg{extra_str}')


def main(args=None):
    rclpy.init(args=args)
    node = MonoMultiviewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
