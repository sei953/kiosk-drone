#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""벽/마커 공통 상수 및 좌표 변환 유틸 (위치계산부 aruco_pnp_node.py, 제어부
approach_control_node.py가 공유). sim/generate_marker.py와 상수 일치 필수
(sim/README.md '마커 크기 상수 일치' 함정 참고)."""
import numpy as np

MARKER_SIZE = 0.150  # 마커 한 변 실제 길이 [m] (sim/generate_marker.py MARKER_MM=150)

# 벽별 ID 오프셋 (sim/worlds/kiosk_4walls.sdf 배치와 일치)
WALL_ID_BASE = {'북': 0, '동': 4, '남': 8, '서': 12}
WALL_NAME_BY_BASE = {v: k for k, v in WALL_ID_BASE.items()}

# 벽-로컬 프레임(중심=4마커 중앙, X오른쪽/Z위, 카메라가 벽을 정면으로 볼 때 기준)
# 코너 오프셋(id % 4) -> (cx, cy) [m]
CORNER_LAYOUT = {
    0: (-0.375, 0.375),   # top-left
    1: (0.375, 0.375),    # top-right
    2: (0.375, -0.375),   # bottom-right
    3: (-0.375, -0.375),  # bottom-left
}


def marker_id_to_wall(marker_id):
    """id(0-15) -> (wall_name, corner_offset) or (None, None)."""
    if not (0 <= marker_id < 16):
        return None, None
    base = (marker_id // 4) * 4
    return WALL_NAME_BY_BASE.get(base), marker_id % 4


def marker_corners_3d(cx, cy, s):
    h = s / 2.0
    return [(cx - h, cy + h, 0.0), (cx + h, cy + h, 0.0),
            (cx + h, cy - h, 0.0), (cx - h, cy - h, 0.0)]  # TL,TR,BR,BL


def rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return (x, y, z, w)


def quat_to_rotmat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def yaw_err_from_R(R):
    """벽 법선(로컬 +Z)을 카메라 프레임으로 회전 후 정면=0deg 부호로 yaw 오차[deg] 계산.
    부호는 "target_yaw = cur_yaw + gain*yaw_err가 정렬 방향인가"로 정적 진단 검증됨
    (2026-09-19, 동 벽 +15deg 오프셋 스폰: cur_yaw 80.4deg, 기준(정면) ~95deg,
    필요 보정 +14.7deg인데 atan2(nx,-nz)는 -15.7deg를 내놓아 부호가 반대였음 ->
    atan2(-nx,-nz)로 수정, docs/PROGRESS.md 참고). 기존 "atan2(nx,-nz)"는 SCRUM-23
    때 정지 상태에서 "정면=0"으로만 확인됐고 실제 폐루프 보정 방향으로는 검증된 적이
    없었음."""
    normal = R @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arctan2(-normal[0], -normal[2])))


def wrap_deg_diff(a, b):
    """a-b를 [-180,180] 범위로 wrap (각도 불연속 경계에서의 거짓 점프 방지)."""
    return (a - b + 180.0) % 360.0 - 180.0
