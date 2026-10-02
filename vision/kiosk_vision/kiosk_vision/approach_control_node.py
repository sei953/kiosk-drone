#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""제어부: SEARCH(제자리 yaw 회전 탐색) -> APPROACH(P제어 정렬·접근) -> HOLD
상태머신. px4_msgs 오프보드로 명령한다. APPROACH/HOLD에서 타겟을 놓치면 곧바로
SEARCH(전체 회전)로 튀지 않고 REACQUIRE(제자리 대기 -> 좁은 스윕 -> 그래도
안되면 SEARCH 승격)를 거친다 (_handle_lost 참고).

목표는 "위치"로 커맨드하고(NaN 아닌 position, velocity는 NaN) 감속·정지는 PX4
온보드 위치 컨트롤러에 맡긴다 — 외부에서 속도를 직접 만들면 우리 쪽 루프 주기에
그 매끄러움이 통째로 종속되는데, 이 루프는 비전 프레임 처리 부하 등으로 주기가
들쭉날쭉해질 수 있음(20Hz 목표가 실측 ~2Hz까지 저하된 적 있음, docs/PROGRESS.md).
접근 속도 상한은 PX4 파라미터(MPC_XY_VEL_MAX/MPC_XY_CRUISE, sim/gcs_keepalive.py가
설정)로 건다. 오프보드 setpoint 스트림(on_setpoint_timer)은 상태 판단 로직
(on_timer)과 분리된 별도 타이머 + MultiThreadedExecutor로 돌려서, 로직 쪽이
느려져도 PX4가 요구하는 안정적인 스트림(>2Hz)은 항상 유지한다.

안전 브링업 단계는 bringup_level 파라미터로 순서대로 검증한다:
  0 LOG_ONLY     : 오프보드 미발행/미arm. 변환된 목표 NED만 로그 (변환 검증).
  1 HOVER_HOLD   : arm+offboard, 제자리 유지만 (탐색/접근 비활성).
  2 YAW          : + 탐색 회전, 접근 시 yaw축만 서보 (위치 고정).
  3 YAW_LATERAL  : + lateral축도 서보 (forward는 고정, standoff 접근 없음).
  4 FULL         : forward 포함 전체 (표준 접근 시퀀스).

레벨과 무관하게 LOG_ONLY는 항상 레벨 4 기준으로 계산만 하고 로그만 남긴다
(변환 로직 자체를 검증하기 위함)."""
import math

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleLocalPosition, VehicleAttitude,
)
from geometry_msgs.msg import PoseStamped

from kiosk_vision.wall_geometry import WALL_ID_BASE, quat_to_rotmat, yaw_err_from_R, wrap_deg_diff

LOG_ONLY, HOVER_HOLD, YAW, YAW_LATERAL, FULL = range(5)
LEVEL_NAMES = ['LOG_ONLY', 'HOVER_HOLD', 'YAW', 'YAW_LATERAL', 'FULL']

CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
# 오프보드 setpoint 스트림 주기. PX4는 오프보드 유지에 >2Hz를 요구하는데, 상태 판단
# 로직(on_timer)은 비전 처리 부하로 그 아래까지 느려진 적이 있어 스트림만 떼어낸다.
SETPOINT_HZ = 20.0
SETPOINT_DT = 1.0 / SETPOINT_HZ
ARM_TICK = 10  # 예제(offboard_control.py)와 동일하게 setpoint 10회 스트리밍 후 arm

# 카메라 마운트고 vs 마커 중심고 (docs/PROGRESS.md 2026-09-22 '카메라 마운트고 정렬' 참고).
# takeoff_alt 기본값을 이 둘로부터 유도한다 — 비행 고도를 마커 중심고와 그대로 맞추면
# 카메라가 mount_offset만큼 위에서 내려다보게 되어, standoff 근접 시 마커 패턴 아래쪽
# 행이 화각 밖으로 잘려나간다(실측: 1.0m 거리에서 3마커 게이트 통과율 0%).
MARKER_CENTER_HEIGHT = 1.5  # m — 벽 마커 패턴 중심 높이 (sim/worlds/kiosk_4walls.sdf 벽 pose z)
CAMERA_MOUNT_OFFSET = 0.242  # m — base_link 위 카메라 장착고
# (PX4-Autopilot/Tools/simulation/gz/models/x500_depth/model.sdf CameraJoint pose z)


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ApproachControlNode(Node):
    def __init__(self):
        super().__init__('approach_control_node')
        self.declare_parameter('bringup_level', LOG_ONLY)
        self.declare_parameter('pose_topic', '/target/pose')
        # aruco_pnp_node의 target_wall과 같은 값을 줘야 한다. 이 벽(frame_id='wall_<벽>')이
        # 아닌 pose는 버린다 — 이전 실행에서 고아로 남은 aruco_pnp_node(다른 target_wall)가
        # 같은 토픽에 발행하면 엉뚱한 벽으로 접근하던 문제 방어 (docs/PROGRESS.md 참고).
        self.declare_parameter('target_wall', '동')  # 북/동/남/서
        # 0.6m -> 1.0m: 1m 벽 패널 기준으로도 접근 끝(정지 지점)에서 4마커가 화각에
        # 남아 aruco_pnp_node의 마커 수 게이트(REQUIRED_MARKERS=4)가 막판에 끊기지
        # 않도록 여유를 둠.
        self.declare_parameter('standoff', 1.0)
        self.declare_parameter('search_yaw_rate_deg', 20.0)
        # 목표점을 "비전 프레임마다 그 시점 위치 기준으로 한 번" 앵커링하는 구조로
        # 바뀌면서(불변 목표 + PX4 위치 컨트롤러가 가감속 담당) 부분 스텝(0.3)일 이유가 없어짐 ->
        # 1.0(전체 보정)이 기본. 필요시 낮춰서 프레임 간 미세보정을 더 완만하게 할 수 있음.
        self.declare_parameter('approach_gain', 1.0)
        self.declare_parameter('yaw_gain', 1.0)
        # solvePnP 거울해(flip) 사이 chatter 완화용 EMA 저역통과.
        self.declare_parameter('ema_alpha', 0.3)
        self.declare_parameter('tol_forward', 0.06)
        # 6cm -> 12cm: approach_align 실측 lateral 노이즈 바닥이 약 3~12cm라 6cm는
        # 노이즈보다 빡빡해서 HOLD<->APPROACH가 계속 토글됐음 (docs/PROGRESS.md).
        self.declare_parameter('tol_lateral', 0.12)
        self.declare_parameter('tol_yaw_deg', 3.0)  # 실측 yaw 진동폭(~±2deg)이 이미 여유 있어 유지
        # 실기 마운트나 마커 높이가 바뀌면 두 파라미터를 오버라이드 — takeoff_alt 기본값이
        # 따라간다(바로 아래).
        self.declare_parameter('marker_center_height', MARKER_CENTER_HEIGHT)
        self.declare_parameter('camera_mount_offset', CAMERA_MOUNT_OFFSET)
        default_takeoff_alt = (float(self.get_parameter('marker_center_height').value)
                                - float(self.get_parameter('camera_mount_offset').value))
        # 카메라 광축을 마커 중심고에 맞춰 근접 시 전체 마커 가시성 확보.
        self.declare_parameter('takeoff_alt', default_takeoff_alt)  # m (NED z = -takeoff_alt)
        # 실측 카메라 프레임 간격이 5Hz 스펙보다 불규칙(WSL 렌더링, 최대 약 2.0s 공백
        # 관측됨)해서 여유를 두고 3.0s로 설정 (미세 드롭이 REACQUIRE로 안 튀도록 소폭 상향).
        self.declare_parameter('target_lost_timeout', 3.0)  # s
        # REACQUIRE(재포착): 락을 막 놓쳤을 때 곧바로 반대 방향 전체 SEARCH로 튀지
        # 않기 위한 3단계 — ①grace(제자리 대기) ②좁은 스윕(마지막 방향 근방) ③그래도
        # 못 찾으면 전체 SEARCH로 승격.
        self.declare_parameter('reacquire_grace_s', 1.75)
        self.declare_parameter('reacquire_sweep_deg', 40.0)
        self.declare_parameter('reacquire_sweep_rate_deg', 9.0)
        self.declare_parameter('reacquire_sweep_max_s', 15.0)

        self.level = int(self.get_parameter('bringup_level').value)
        self.target_wall = self.get_parameter('target_wall').value
        if self.target_wall not in WALL_ID_BASE:
            raise ValueError(f"target_wall='{self.target_wall}' invalid, must be one of {list(WALL_ID_BASE)}")
        self.expected_frame_id = f'wall_{self.target_wall}'
        self.standoff = float(self.get_parameter('standoff').value)
        self.search_yaw_rate = math.radians(float(self.get_parameter('search_yaw_rate_deg').value))
        self.gain = float(self.get_parameter('approach_gain').value)
        self.yaw_gain = float(self.get_parameter('yaw_gain').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)
        self.tol_forward = float(self.get_parameter('tol_forward').value)
        self.tol_lateral = float(self.get_parameter('tol_lateral').value)
        self.tol_yaw = math.radians(float(self.get_parameter('tol_yaw_deg').value))
        self.marker_center_height = float(self.get_parameter('marker_center_height').value)
        self.camera_mount_offset = float(self.get_parameter('camera_mount_offset').value)
        self.takeoff_alt = float(self.get_parameter('takeoff_alt').value)
        self.target_lost_timeout = float(self.get_parameter('target_lost_timeout').value)
        self.reacquire_grace_s = float(self.get_parameter('reacquire_grace_s').value)
        self.reacquire_sweep = math.radians(float(self.get_parameter('reacquire_sweep_deg').value))
        self.reacquire_sweep_rate = math.radians(float(self.get_parameter('reacquire_sweep_rate_deg').value))
        self.reacquire_sweep_max_s = float(self.get_parameter('reacquire_sweep_max_s').value)

        # 모든 콜백(구독 + 두 타이머)을 하나의 재진입 가능 그룹에 묶고
        # MultiThreadedExecutor(main() 참고)로 돌려서, on_timer(상태판단 로직,
        # 비전 콜백 처리 등으로 느려질 수 있음)가 on_setpoint_timer(PX4 오프보드
        # 스트림)를 굶기지 않게 한다.
        cb_group = ReentrantCallbackGroup()

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_qos)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', px4_qos)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
                                  self.on_local_position, px4_qos, callback_group=cb_group)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude',
                                  self.on_attitude, px4_qos, callback_group=cb_group)

        pose_topic = self.get_parameter('pose_topic').value
        self.pose_topic = pose_topic
        self.create_subscription(PoseStamped, pose_topic, self.on_target_pose, 10, callback_group=cb_group)

        self.vlp = None
        self.att = None
        self.last_target_pose = None
        self.last_target_time = None
        self.ema_lateral = None
        self.ema_yaw_err_deg = None

        # 비전 프레임마다 한 번씩만 갱신되는 앵커 목표(그 사이엔 그대로 유지) —
        # "매 제어 틱마다 현재위치+오차로 재계산"하면 실제 속도로 움직일 때 오차 갱신이
        # 못 따라가서 목표가 계속 도망가듯 밀리는 문제가 있었음 (docs/PROGRESS.md).
        # (ned, yaw, forward_err, lateral_err, yaw_err)를 한 튜플로 묶어 통째로 교체한다 —
        # 비전 콜백과 on_timer가 다른 스레드에서 도니까, 필드를 따로 두면 앵커는 새 프레임
        # 건데 오차는 이전 프레임 값인 조합으로 HOLD 판정이 날 수 있다.
        self.anchor = None
        # on_timer가 만들고 on_setpoint_timer가 읽는 최신 목표 (n, e, z, yaw).
        self.cmd = None

        self.state = 'SEARCH'
        self.search_ned = None
        self.yaw_ref = None
        self.target_ned = None
        self.target_yaw = None
        self.offboard_setpoint_counter = 0

        self._reacquire_since = None
        self._reacquire_center_yaw = None
        self._reacquire_hold_ned = None
        self._reacquire_sweep_dir = 1.0
        self._last_tick_time = None

        self.create_timer(DT, self.on_timer, callback_group=cb_group)
        self.create_timer(SETPOINT_DT, self.on_setpoint_timer, callback_group=cb_group)
        self.get_logger().info(
            f'approach_control_node start target_wall={self.target_wall} '
            f'bringup_level={self.level}({LEVEL_NAMES[self.level]}) '
            f'standoff={self.standoff}m takeoff_alt={self.takeoff_alt}m '
            f'(marker_center_height={self.marker_center_height}m - camera_mount_offset={self.camera_mount_offset}m) '
            f'gain={self.gain} yaw_gain={self.yaw_gain} ema_alpha={self.ema_alpha}')

    # ---- subscriptions ----
    def on_local_position(self, msg):
        if msg.xy_valid and msg.z_valid:
            self.vlp = msg

    def on_attitude(self, msg):
        self.att = msg

    def on_target_pose(self, msg):
        if msg.header.frame_id != self.expected_frame_id:
            n_pub = self.count_publishers(self.pose_topic)
            self.get_logger().warn(
                f"[{self.pose_topic}] frame_id='{msg.header.frame_id}' != '{self.expected_frame_id}' -> 버림 "
                f"(publisher {n_pub}개 — 다른 target_wall의 aruco_pnp_node가 살아있는지 확인)",
                throttle_duration_sec=2.0)
            return
        self.last_target_pose = msg
        self.last_target_time = self.get_clock().now()

        raw_lateral = msg.pose.position.x
        raw_forward = msg.pose.position.z
        R = quat_to_rotmat(msg.pose.orientation.x, msg.pose.orientation.y,
                            msg.pose.orientation.z, msg.pose.orientation.w)
        raw_yaw_err_deg = yaw_err_from_R(R)
        if self.ema_lateral is None:
            self.ema_lateral = raw_lateral
            self.ema_yaw_err_deg = raw_yaw_err_deg
        else:
            a = self.ema_alpha
            self.ema_lateral = a * raw_lateral + (1.0 - a) * self.ema_lateral
            # 각도는 선형 평균하면 ±180deg 경계에서 깨짐 -> wrap된 델타로 갱신
            self.ema_yaw_err_deg += a * wrap_deg_diff(raw_yaw_err_deg, self.ema_yaw_err_deg)

        self._update_anchor(raw_forward)

    def _update_anchor(self, forward):
        """비전 프레임 도착 시점(pos_at_frame, yaw_at_frame) 기준으로 절대 목표를 한 번
        계산해 고정한다. 다음 프레임이 올 때까지 이 앵커를 그대로 유지 -> 기체는 고정된
        지점으로 날아가 속도캡에 따라 자연 감속·정지, 다음 프레임이 미세보정한다."""
        if self.vlp is None or self.att is None:
            return  # 아직 위치/자세를 모름 -> 이 프레임으로는 앵커링 불가, 다음 프레임 대기

        pos_n, pos_e = self.vlp.x, self.vlp.y
        yaw_at_frame = self.current_yaw()

        forward_err = forward - self.standoff
        lateral_err = self.ema_lateral
        yaw_err = math.radians(self.ema_yaw_err_deg)

        gate_level = FULL if self.level == LOG_ONLY else self.level
        active_forward = forward_err if gate_level >= FULL else 0.0
        active_lateral = lateral_err if gate_level >= YAW_LATERAL else 0.0

        dn = active_forward * math.cos(yaw_at_frame) - active_lateral * math.sin(yaw_at_frame)
        de = active_forward * math.sin(yaw_at_frame) + active_lateral * math.cos(yaw_at_frame)
        z = self.search_ned[2] if self.search_ned is not None else -self.takeoff_alt

        self.anchor = (
            (pos_n + self.gain * dn, pos_e + self.gain * de, z),
            wrap_pi(yaw_at_frame + self.yaw_gain * yaw_err),
            forward_err, lateral_err, yaw_err,
        )

    # ---- helpers ----
    def current_yaw(self):
        w, x, y, z = self.att.q  # Hamilton, FRD body -> NED, order (w,x,y,z)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def target_lost(self):
        # 프레임 단위 순간 미검출(self.target_visible=False)만으로 즉시 SEARCH로 튀지
        # 않도록, "마지막으로 pose를 받은 시각"만 기준으로 판단한다 (target_lost_timeout
        # 동안은 마지막 pose를 유지). 카메라 프레임 간격이 불규칙(WSL 렌더링)해서
        # self.target_visible을 직접 쓰면 수백ms 공백마다 SEARCH<->APPROACH가 토글됨.
        if self.last_target_time is None:
            return True
        age = (self.get_clock().now() - self.last_target_time).nanoseconds * 1e-9
        return age > self.target_lost_timeout

    # ---- main loop ----
    def on_timer(self):
        if self.vlp is None or self.att is None:
            return

        now = self.get_clock().now()
        # 콜백이 이론상 20Hz(DT)지만 WSL에서 시스템 부하로 실제 간격이 늘어날 수 있음
        # (실측: gz 렌더링/비전 처리로 인해 몇 배까지 지연됨, docs/PROGRESS.md 참고).
        # 고정 DT로 회전율(SEARCH/REACQUIRE 스윕)을 적분하면 실제로는 그보다 훨씬
        # 느리게 돌게 되므로 매 틱 실측 경과시간을 사용한다.
        if self._last_tick_time is None:
            dt = DT
        else:
            dt = max((now - self._last_tick_time).nanoseconds * 1e-9, 1e-3)
        self._last_tick_time = now

        cur_n, cur_e, cur_d = self.vlp.x, self.vlp.y, self.vlp.z
        cur_yaw = self.current_yaw()

        if self.search_ned is None:
            self.search_ned = (cur_n, cur_e, -self.takeoff_alt)
            self.yaw_ref = cur_yaw
            self.target_ned = self.search_ned
            self.target_yaw = self.yaw_ref

        if self.level == HOVER_HOLD:
            self.state = 'SEARCH'
            target_ned, target_yaw = self.search_ned, self.yaw_ref
        else:
            target_ned, target_yaw = self.run_state_machine(dt)

        self.target_ned, self.target_yaw = target_ned, target_yaw

        if self.level == LOG_ONLY:
            self.get_logger().info(
                f'[LOG_ONLY][{self.state}] cur_ned=({cur_n:.3f},{cur_e:.3f},{cur_d:.3f}) yaw={math.degrees(cur_yaw):+.1f}deg '
                f'-> anchor_ned=({target_ned[0]:.3f},{target_ned[1]:.3f},{target_ned[2]:.3f}) '
                f'anchor_yaw={math.degrees(target_yaw):+.1f}deg',
                throttle_duration_sec=0.5)
            return  # 오프보드 발행/arm 없음 — 드론은 움직이지 않는다

        # 실제 발행은 on_setpoint_timer가 전담한다 — 여기서는 목표 스냅샷만 통째로 교체
        # (튜플 하나를 바꾸므로 스트림 쪽이 반쯤 갱신된 목표를 읽는 일이 없다).
        self.cmd = (target_ned[0], target_ned[1], target_ned[2], target_yaw)

        self.get_logger().info(
            f'[{LEVEL_NAMES[self.level]}][{self.state}] cur_ned=({cur_n:.3f},{cur_e:.3f}) '
            f'target_ned=({target_ned[0]:.3f},{target_ned[1]:.3f},{target_ned[2]:.3f}) '
            f'target_yaw={math.degrees(target_yaw):+.1f}deg dt={dt:.3f}s', throttle_duration_sec=0.5)

    def on_setpoint_timer(self):
        """오프보드 스트림 전담(on_timer와 별도 타이머·스레드). 최신 목표 스냅샷을 그대로
        재발행하기만 하므로, 상태 판단 로직이 느려져도 스트림 주기는 흔들리지 않는다."""
        if self.level == LOG_ONLY:
            return
        cmd = self.cmd
        if cmd is None:
            return  # 아직 위치/자세 미수신 -> 발행할 목표 없음

        n, e, z, yaw = cmd
        self.publish_offboard_heartbeat()
        self.publish_trajectory_setpoint(n, e, z, yaw)

        if self.offboard_setpoint_counter == ARM_TICK:
            self.engage_offboard_mode()
            self.arm()
        if self.offboard_setpoint_counter <= ARM_TICK:
            self.offboard_setpoint_counter += 1

    def _handle_lost(self, dt):
        """타겟 유실 시 곧바로 전체 SEARCH로 튀지 않고 REACQUIRE 3단계를 거친다:
        ①grace(제자리 대기) ②마지막 방향 근방 좁은 스윕 ③그래도 못 찾으면 전체 SEARCH."""
        now = self.get_clock().now()

        if self.state in ('APPROACH', 'HOLD'):
            self.state = 'REACQUIRE'
            self._reacquire_since = now
            self._reacquire_center_yaw = self.target_yaw
            self._reacquire_hold_ned = self.target_ned
            self._reacquire_sweep_dir = 1.0
            self.yaw_ref = self.target_yaw
            self.get_logger().info('타겟 유실 -> REACQUIRE(제자리 대기) 진입')

        if self.state == 'REACQUIRE':
            elapsed = (now - self._reacquire_since).nanoseconds * 1e-9
            if elapsed <= self.reacquire_grace_s:
                return self._reacquire_hold_ned, self._reacquire_center_yaw  # ① grace: 제자리 대기

            if elapsed <= self.reacquire_grace_s + self.reacquire_sweep_max_s:
                # ② 마지막 방향 기준 좁은 스윕 (느린 속도로 왕복)
                step = self.reacquire_sweep_rate * dt * self._reacquire_sweep_dir
                candidate = wrap_pi(self.yaw_ref + step)
                offset = wrap_pi(candidate - self._reacquire_center_yaw)
                if offset > self.reacquire_sweep:
                    candidate = wrap_pi(self._reacquire_center_yaw + self.reacquire_sweep)
                    self._reacquire_sweep_dir = -1.0
                elif offset < -self.reacquire_sweep:
                    candidate = wrap_pi(self._reacquire_center_yaw - self.reacquire_sweep)
                    self._reacquire_sweep_dir = 1.0
                self.yaw_ref = candidate
                return self._reacquire_hold_ned, self.yaw_ref

            # ③ 좁은 스윕도 실패 -> 전체 SEARCH로 승격 (기존 동작, search_ned로 복귀)
            self.get_logger().info('REACQUIRE 실패(좁은 스윕 시간초과) -> 전체 SEARCH 회전으로 전환')
            self.state = 'SEARCH'
            self.yaw_ref = self.target_yaw

        self.yaw_ref = wrap_pi(self.yaw_ref + self.search_yaw_rate * dt)
        return self.search_ned, self.yaw_ref

    def run_state_machine(self, dt):
        anchor = self.anchor  # 비전 콜백이 다른 스레드에서 갈아끼우므로 한 번만 읽어 쓴다
        if self.target_lost() or anchor is None:
            return self._handle_lost(dt)

        # 앵커는 on_target_pose(_update_anchor)에서 비전 프레임 도착 시 이미 계산해
        # 고정해뒀다 — 여기서는 그걸 그대로 목표로 쓰고, 허용오차 판정만 한다.
        anchor_ned, anchor_yaw, forward_err, lateral_err, yaw_err = anchor
        gate_level = FULL if self.level == LOG_ONLY else self.level
        checks = [abs(yaw_err) < self.tol_yaw]
        if gate_level >= YAW_LATERAL:
            checks.append(abs(lateral_err) < self.tol_lateral)
        if gate_level >= FULL:
            checks.append(abs(forward_err) < self.tol_forward)

        self.state = 'HOLD' if all(checks) else 'APPROACH'
        return anchor_ned, anchor_yaw

    # ---- px4 command helpers (offboard_control.py 예제와 동일 패턴) ----
    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, n, e, z, yaw):
        """목표를 위치로 커맨드 — velocity는 NaN으로 두고 가감속은 PX4 위치 컨트롤러에
        맡긴다. 속도 상한은 MPC_XY_VEL_MAX/MPC_XY_CRUISE(sim/gcs_keepalive.py)."""
        msg = TrajectorySetpoint()
        msg.position = [float(n), float(e), float(z)]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.param3 = params.get('param3', 0.0)
        msg.param4 = params.get('param4', 0.0)
        msg.param5 = params.get('param5', 0.0)
        msg.param6 = params.get('param6', 0.0)
        msg.param7 = params.get('param7', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Switching to offboard mode')


def main(args=None):
    rclpy.init(args=args)
    node = ApproachControlNode()
    # 로직 타이머와 setpoint 스트림 타이머가 서로를 굶기지 않도록 멀티스레드로 돈다.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
