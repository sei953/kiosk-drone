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

시간 동기화(mono_multiview 계획의 step 8): 이미지 콜백이 실행되는 시점의
"최신" 텔레메트리를 그냥 쓰면, 프레임 전달 지연(WSL 렌더링, 최대 ~2초 공백
관측됨 -- approach_control_node.py 참고)만큼 어긋난 위치/자세로 baseline을
계산하게 된다. 그런데 이미지 header.stamp는 Gazebo 시뮬레이션 시각(초 단위,
0부터 시작)이고 px4_msgs의 timestamp 필드는 실제 wall-clock epoch(마이크로초)
라서 -- 직접 비교가 불가능한 서로 다른 시계다 (실측 확인: 같은 순간에
header.stamp=25.3, px4 timestamp=1791173357.5). 그래서 최초 프레임에서 한 번
"sim_time + offset = wall_time" 오프셋을 보정해두고, 이후 모든 프레임은 그
오프셋으로 추정한 실제 촬영 시각에 가장 가까운 텔레메트리 샘플을 버퍼에서
찾아 쓴다 (_nearest_sample). offset 보정은 lockstep의 real_time_factor=1.0을
전제하므로, 시뮬레이션이 그 비율에서 크게 벗어나면(렌더링 과부하 등) 오차가
누적될 수 있다 -- 짧은 비행(검증 스크립트 기준 수십 초~수 분)에서는 충분.
"""

import sys
from collections import deque
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

_TELEMETRY_BUFFER_LEN = 50  # 텔레메트리는 ned/attitude 합쳐 초당 수십~수백 Hz라
                             # 50개면 보통 초 단위 이하 구간을 커버한다.


def _nearest_sample(buffer, target_wall_time):
    """buffer: [(wall_time_sec, value), ...] (시간순 아닐 수 있음, deque라
    오래된 것부터). target_wall_time에 가장 가까운 value를 반환, 비어있으면
    None. buffer 길이가 작아(<=50) 선형 탐색으로 충분하다."""
    if not buffer:
        return None
    return min(buffer, key=lambda item: abs(item[0] - target_wall_time))[1]


def _wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


POSE_HISTORY_LEN = 5
# motion_direction_sweep.py/실비행 테스트에서 반복 관측된 이상치 폭(수십~80도대)과
# 정상 샘플끼리의 흔들림(수 도대)을 가르는 문턱값. 너무 taight하면 정상적인 빠른
# 재정렬까지 이상치로 버리고, 너무 느슨하면 이상치가 그냥 통과한다.
OUTLIER_YAW_DEG = 25.0
# yaw만 보던 첫 버전은 실비행에서 구멍이 발견됨: yaw는 멀쩡한데 forward가
# 127m로 튀는 샘플(거의 특이점에 가까운 스케일 계산, 예: 삼각측량 포인트가
# 카메라와 거의 같은 방향으로 늘어서는 경우)이 그대로 통과함. forward/lateral도
# 최근 중앙값 대비 거리로 같이 검사.
OUTLIER_DIST_M = 3.0  # 지금까지 모든 검증에서 forward/lateral이 이 이상 한 번에
                       # 튄 적 없음 (정상 범위는 수 m 이내) -- 거리가 이보다
                       # 더 벌어지면 샘플 자체가 깨진 것으로 본다.
OUTLIER_RESET_STREAK = 3  # 이만큼 연속 거부되면 "진짜 바뀐 것"으로 보고 히스토리 리셋


class _PoseFilter:
    """최근 POSE_HISTORY_LEN개 pose 추정치의 성분별 중앙값을 내는 temporal
    filter (SCRUM-22 8단계/temporal filtering) -- Essential/Homography 둘 다
    단일 프레임 추정이 가끔 크게 튀는 이상치를 내는 걸 여러 번 확인했다
    (motion_direction_sweep.py, 실비행 테스트). 평균은 이상치 1개에도 전체가
    끌려가지만 중앙값은 덜 휘둘린다. 추가로, 최근 중앙값에서 OUTLIER_YAW_DEG
    또는 OUTLIER_DIST_M 이상 벗어난 새 샘플은 아예 히스토리에 넣지 않는다 --
    yaw만 보던 초기 버전은 실비행에서 구멍이 드러남(yaw는 정상인데 forward가
    127m로 튀는 샘플이 그냥 통과), 그래서 forward/lateral도 같이 검사. 다만
    OUTLIER_RESET_STREAK번 연속 거부되면 드론이 실제로 빠르게 재정렬했다고 보고
    히스토리를 리셋한다(안 그러면 진짜 자세 변화를 영구히 이상치로 오인할 수 있음).

    approach_control_node.py도 자체 EMA(ema_alpha)로 스무딩하지만, 그건
    "들어온 값"을 스무딩하는 거라 이 필터가 입력 자체의 이상치를 걸러내는 것과는
    다른 층위 -- 여기서 걸러진 덜 noisy한 입력이 그 EMA의 수렴을 더 빠르게 한다."""

    def __init__(self):
        self._history = deque(maxlen=POSE_HISTORY_LEN)
        self._reject_streak = 0

    def update(self, result):
        """result: compute_mono_pose가 고른 dict (forward/lateral/vertical/
        yaw_deg/normal 포함). 반환: (스무딩된 dict 또는 히스토리가 아직 없으면
        None, 이번 샘플이 이상치로 거부됐는지 bool)."""
        is_outlier = False
        if self._history:
            median_yaw = float(np.median([h['yaw_deg'] for h in self._history]))
            median_fwd = float(np.median([h['forward'] for h in self._history]))
            median_lat = float(np.median([h['lateral'] for h in self._history]))
            yaw_jump = abs(_wrap_deg(result['yaw_deg'] - median_yaw))
            dist_jump = max(abs(result['forward'] - median_fwd), abs(result['lateral'] - median_lat))
            if yaw_jump > OUTLIER_YAW_DEG or dist_jump > OUTLIER_DIST_M:
                is_outlier = True
                self._reject_streak += 1
                if self._reject_streak >= OUTLIER_RESET_STREAK:
                    self._history.clear()
                    is_outlier = False  # 리셋 후 이 샘플은 새 히스토리의 시작으로 받아들임

        if not is_outlier:
            self._reject_streak = 0
            self._history.append(result)

        return self._smoothed(), is_outlier

    def _smoothed(self):
        if not self._history:
            return None
        normal = np.median(np.stack([h['normal'] for h in self._history]), axis=0)
        normal = normal / np.linalg.norm(normal)
        return {
            'forward': float(np.median([h['forward'] for h in self._history])),
            'lateral': float(np.median([h['lateral'] for h in self._history])),
            'vertical': float(np.median([h['vertical'] for h in self._history])),
            'yaw_deg': float(np.median([h['yaw_deg'] for h in self._history])),
            'normal': normal,
        }


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
        # (wall_time_sec, value) 샘플 버퍼 -- on_image에서 이미지의 추정 촬영
        # 시각에 가장 가까운 샘플을 찾는다 (_nearest_sample, 모듈 docstring의
        # "시간 동기화" 참고). ned=(north_m,east_m,down_m), quat=_Quat.
        self._ned_buffer = deque(maxlen=_TELEMETRY_BUFFER_LEN)
        self._quat_buffer = deque(maxlen=_TELEMETRY_BUFFER_LEN)
        self._sim_to_wall_offset = None  # 최초 프레임에서 한 번만 보정
        self._pose_filter = _PoseFilter()

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
            wall_t = msg.timestamp * 1e-6  # px4_msgs timestamp: wall-clock epoch, us
            self._ned_buffer.append((wall_t, (msg.x, msg.y, msg.z)))

    def on_attitude(self, msg):
        wall_t = msg.timestamp * 1e-6
        self._quat_buffer.append((wall_t, _Quat(msg.q)))

    def on_camera_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def on_image(self, msg):
        if self.K is None:
            return

        image_sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now_wall = self.get_clock().now().nanoseconds * 1e-9
        if self._sim_to_wall_offset is None:
            # 최초 프레임을 기준점으로 삼아 sim_time -> wall_time 오프셋을 한 번
            # 보정한다 (모듈 docstring의 "시간 동기화" 참고). 그 뒤로는 이
            # 오프셋으로 매 프레임의 추정 촬영 시각(wall-clock)을 구한다.
            self._sim_to_wall_offset = now_wall - image_sim_time
        estimated_capture_wall_time = image_sim_time + self._sim_to_wall_offset

        ned = _nearest_sample(self._ned_buffer, estimated_capture_wall_time)
        quat = _nearest_sample(self._quat_buffer, estimated_capture_wall_time)
        if ned is None or quat is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

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
            if extra is not None:
                # select_pose ran (both methods produced a candidate) but
                # neither cleared the ground-truth sanity check -- see
                # select_pose's docstring for why this is reported as no
                # pose rather than publishing whichever was "less bad".
                self.get_logger().warn(
                    f'pose rejected: both Essential and Homography failed sanity check '
                    f'(e_err={extra["e_err"]:.1f} h_err={extra["h_err"]:.1f}) -- treating as not visible')
            else:
                self.get_logger().warn('Essential and Homography both failed for this pair')
            self.visible_pub.publish(Bool(data=False))
            return

        smoothed, is_outlier = self._pose_filter.update(result)
        if is_outlier:
            self.get_logger().warn(
                f'pose outlier rejected: yaw={result["yaw_deg"]:+.1f}deg far from recent '
                f'median -- publishing last smoothed estimate instead', throttle_duration_sec=1.0)
        if smoothed is None:
            # 필터 히스토리가 아직 없다 (이번이 첫 샘플이면서 동시에 이상치일
            # 리는 없지만, 방어적으로) -- 낼 pose가 없으니 invisible.
            self.visible_pub.publish(Bool(data=False))
            return

        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = self.frame_id
        # aruco_pnp_node.py와 동일 규약: x=lateral(camera-X), y=vertical
        # (camera-Y), z=forward(camera-Z). raw result가 아니라 smoothed를
        # 발행한다 -- 단일 프레임 이상치가 그대로 제어 루프까지 가지 않도록.
        ps.pose.position.x = float(smoothed['lateral'])
        ps.pose.position.y = float(smoothed['vertical'])
        ps.pose.position.z = float(smoothed['forward'])
        qx, qy, qz, qw = _normal_to_quat(smoothed['normal'])
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pose_pub.publish(ps)
        self.visible_pub.publish(Bool(data=True))

        extra_str = f" e_err={extra['e_err']:.1f} h_err={extra['h_err']:.1f}" if extra else ""
        raw_str = (f" (raw fwd={result['forward']:.2f} lat={result['lateral']:+.2f} "
                   f"yaw={result['yaw_deg']:+.1f}deg)" if not is_outlier else "")
        self.get_logger().info(
            f'[{chosen}] baseline={baseline:.2f}m fwd={smoothed["forward"]:.2f}m '
            f'lat={smoothed["lateral"]:+.2f}m yaw={smoothed["yaw_deg"]:+.1f}deg{extra_str}{raw_str}')


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
