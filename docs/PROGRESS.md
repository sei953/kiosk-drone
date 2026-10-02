# 진행 상황

## 2026-09-08 — 시뮬 마커 4점 검출 성공

### 결과
- 드론이 좌표 이동 후 ArUco 마커 4개(ID 0~3) 전부 검출
- 검출 픽셀 좌표가 정의한 배치와 일치 (UV 좌우 반전 없음)

| ID | 위치 | 픽셀 좌표 |
|---|---|---|
| 0 | 좌상 | (213, 174) |
| 1 | 우상 | (423, 173) |
| 2 | 우하 | (422, 389) |
| 3 | 좌하 | (211, 381) |

- depth min = 1.59m (벽까지 거리)
- 좌우 x가 210↔422, 상하 y가 174↔385로 대칭 → 벽 정면에 거의 정렬됨

### 해결한 문제

**1. yaw 부호 착각**
`PositionNedYaw`의 yaw를 상대 회전량으로 착각해 90도를 넣었더니 벽을 등지고 촬영.
NED 절대 방위각이므로 북쪽은 0. `YAW_TO_MARKER = 0.0`으로 수정.
증상: depth 프레임에 수평선만, min=2.41m(지면 거리).

**2. 접근 거리 과다**
NORTH_TARGET=2.4일 때 벽까지 0.39m로 너무 가까워 마커가 시야 밖.
1.5 → ID2만 화면 경계에서 잘림 → 1.2로 조정해 4개 전부 확보.

**3. 마커 크기**
53mm는 캘리브레이션용 실측값이라 시뮬 확인에는 작음.
150mm 4개(네 모서리)로 변경. ID를 0~3으로 구분해 자세 판별 가능하게.

**4. Gazebo 카메라 메모리 누수**
WSL에서 GPU 가속 없이(llvmpipe) 카메라 센서 렌더링 시 프레임 버퍼 누적.
1920×1080/30Hz에서 초당 150MB → VM이 40초 만에 죽음.
640×480/5Hz로 낮춰 초당 4.5MB로 완화 (약 33배). 실기 D435i는 해당 없음.

### 아직 안 된 것
- solvePnP 미착수 → 거리·각도 숫자가 아직 없음
- **현재는 좌표 하드코딩 이동. 비전 이동 아님**
- 판단부, 제어 루프 연결 미착수

### 다음
1. solvePnP로 거리·각도 산출
2. `decide()` 규칙판
3. 제어 루프 연결 → 비전 이동 완성
4. 커밋 & 푸시

## 2026-09-18 — 4벽 탐색·타겟 정렬 접근 (SCRUM-25) — LOG_ONLY 검증

### 결과
- `sim/worlds/kiosk_4walls.sdf`: 원점 사방 3m에 4벽(북0-3/동4-7/남8-11/서12-15), 각 벽이
  원점을 향하도록 yaw 계산해서 배치 (북0°/동-90°/남180°/서90°, gz ENU 기준)
- `aruco_pnp_node.py` 확장: 검출 마커를 ID로 벽별 그룹핑 → 벽별 통합 solvePnP,
  `target_wall` 파라미터(기본 '동')로 타겟 벽만 `/target/pose`+`/target/visible` 발행
- `approach_control_node.py` 신설: px4_msgs 오프보드 SEARCH/APPROACH/HOLD 상태머신,
  `bringup_level` 0~4단계 게이팅, 속도캡(0.3m/s, 틱당 rate-limit)
- **LOG_ONLY(레벨0) 실행 결과** (기체는 스폰 위치 그대로, 오프보드 미발행/미arm):
  - 스폰 기본 yaw가 이미 동쪽(+X)을 향함 → 별도 회전 없이 동 벽(id 4-7) 검출됨
  - `aruco_pnp_node`: `fwd 2.924m | lat -0.003 | yaw +0.0deg | reproj 0.67px`
    (동 벽이 실제로 x=3m 근처, reproj<2px 기준 충족)
  - `approach_control_node`: APPROACH 상태에서 raw_target_ned(속도캡 전) ≈
    (N -0.01~-0.05, E +0.65~0.71, -1.5) — standoff(0.6m) 보정된 전진오차(gain 0.3)가
    거의 전부 +E(동쪽) 성분으로 변환됨 → FRD→NED 회전 변환이 동 벽 방향을 정확히 가리킴
  - clamped target은 실제 현재 위치 기준 0.015m/tick(=0.3m/s*50ms) 이내로만 이동 지시
    → 기체는 실측상 계속 원점 근방(±5cm)에 머묾, 실제로 움직이지 않음을 확인

### 알려진 이슈
- 타겟 가시성이 간헐적으로 튐(`타겟 미검출/유실 -> SEARCH 복귀`가 수백ms 간격으로 발생) —
  카메라 프레임 도착 간격이 5Hz 스펙보다 불규칙(WSL 렌더링) 한 것으로 보임.
  `target_lost_timeout`(현 1.0s) 조정 여지 있음. HOVER_HOLD 이후 단계에서 재확인 필요.
- PX4 기본 스폰 yaw가 이미 동쪽이라(gz ENU에서 body +X = world +X = East) LOG_ONLY
  검증에 별도 조치가 필요 없었음 — 다른 벽을 타겟으로 검증할 때는 `PX4_GZ_MODEL_POSE`로
  스폰 yaw를 돌려야 함 (예: 북쪽 확인 시 `PX4_GZ_MODEL_POSE="0,0,0,0,0,1.570796"`로 이미
  검증 — 이때는 북 벽 id=3이 잡힘, gz ENU 기준 yaw는 +X축에서 CCW로 측정됨에 주의).

### 다음 (사용자 확인 후 진행)
1. HOVER_HOLD(레벨1) — arm+offboard 제자리 유지만 확인
2. YAW(2) → YAW_LATERAL(3) → FULL(4) 순서로 단계적 검증
3. 전 구간 성공 시 커밋 & 푸시 & PR

## 2026-09-22 — 위치 setpoint 전환 + 카메라 마운트고 정렬 (SCRUM-25) — HOLD 도달

### 핵심 교훈: 카메라 마운트고 vs 마커고 정렬이 근접 마커 가시성을 좌우
`takeoff_alt`(비행 고도)를 벽 마커 패턴 중심고(1.5m)와 그대로 맞췄더니, OakD-Lite가
`base_link` 위 +0.242m에 장착돼 있어(x500_depth/model.sdf CameraJoint) 카메라가 항상
마커 중심보다 24cm 위에서 내려다봄. 원거리에서는 여유 FOV로 안 보였지만, standoff(1.0m)
근접 시 수직 FOV 반각(±0.515m@1m)에 비해 광축 오프셋(0.692m, 하단 마커 기준)이 커서
패턴 아래쪽 행이 화각 밖으로 잘려나감 — 실측 1.0m 거리에서 평균 1.34개/4마커,
`REQUIRED_MARKERS=3` 게이트 통과율 0%. `aruco_pnp_node`가 계속 "마커 부족 -> 보류"로
막판 pose를 못 내니 `approach_control_node`가 `target_lost_timeout`에 걸려 REACQUIRE로
튕기고, 좁은 스윕도 실패해 전체 SEARCH로 승격 — 반복 루프에 갇혀 HOLD를 못 감.
`takeoff_alt=marker_center_height-camera_mount_offset`(=1.26m)로 카메라 광축을 마커
중심고에 정렬하니 1.0m 거리 평균 마커수 3.43개, 게이트 통과 78.4%로 회복, 즉시 HOLD
도달·3분간 246틱 유지(SEARCH 승격 0회). **접근 standoff를 설계할 때는 항상 "카메라가
그 거리에서 마커 패턴 전체를 몇 도 각도로 내려다보는가"를 먼저 계산할 것** — 수평 위치
정렬(lateral/yaw)만으로는 안 잡히는, 수직 축 하나만의 문제였음.

### 위치 setpoint + 스트림 분리 (근본 fix, `approach_control_node.py`)
이전 커밋(`1cd0fe7`)은 docstring과 콜백 그룹만 들어가고 실제 발행은 여전히
`velocity_xy_toward`(XY 속도 커맨드)였음 — 로그상 `vel=(...)`이 매 틱 부호를 뒤집으며
앵커 주변을 맴돌았고(`-0.296`→`+0.284`→`+0.256`), 이게 접근 중 락이 자꾸 풀리던
근본 원인 중 하나로 확인됨. 이번에 마저 적용:
- `velocity_xy_toward` 제거. `publish_trajectory_setpoint`가 앵커 위치를 그대로
  `position`으로 발행(`velocity=[NaN,NaN,NaN]`), 감속·정지는 PX4 온보드 위치
  컨트롤러에 위임. 속도 상한은 `MPC_XY_VEL_MAX`/`MPC_XY_CRUISE`(`sim/gcs_keepalive.py`)로만.
- `on_setpoint_timer`(20Hz) 신설, `on_timer`(상태판단)와 분리. `on_timer`는 목표
  스냅샷(`self.cmd` 튜플)만 교체하고, 실제 발행·arm 카운터는 스트림 타이머가 전담.
  `main()`을 `MultiThreadedExecutor`로 전환해 로직 쪽이 비전 처리 부하로 느려져도
  PX4가 요구하는 안정적인 오프보드 스트림(>2Hz)이 끊기지 않게 함.
- 두 타이머가 다른 스레드에서 도니, 앵커를 `(ned, yaw, forward_err, lateral_err,
  yaw_err)` 한 튜플로 묶어 통째로 교체 — 필드를 따로 두면 앵커는 새 프레임인데
  오차는 이전 프레임 값인 조합으로 HOLD 오판정 가능.
- 검증: `cur_ned` E가 0.47→0.52→0.66→0.83→0.98로 단조 수렴(약 0.3m/s), 목표 도달 후
  ~5cm 이내 정지. 이전의 방향 반전 진동 사라짐.

### `takeoff_alt` 유도 (`marker_center_height` / `camera_mount_offset` 파라미터화)
매직넘버 1.5 대신 `marker_center_height`(기본 1.5, `kiosk_4walls.sdf` 벽 pose z)와
`camera_mount_offset`(기본 0.242, `x500_depth/model.sdf` CameraJoint pose z)을 ROS
파라미터로 노출, `takeoff_alt` 기본값을 `marker_center_height - camera_mount_offset`으로
유도. 실기 마운트나 마커 높이가 바뀌어도 두 파라미터만 오버라이드하면 됨.

### REACQUIRE 스윕(2번, 미착수) — 향후 로버스트니스 항목으로 보류
락 유실 시 REACQUIRE가 마지막 방향 반대로 스윕하며 카메라를 벽에서 돌려버릴 수 있다는
우려가 있었으나, 이번 재검증(3분/246 HOLD틱)에서는 순간 유실 4회 전부 grace 단계에서
복구됐고 전체 SEARCH 승격은 0회 — 카메라고 정렬 fix로 유실 자체가 크게 줄어 지금은
불필요. 다만 `_handle_lost`의 좁은 스윕이 마지막 벽 방향과 반대로 튈 수 있는 구조적
여지는 남아있으니, 유실이 잦은 환경(조명/거리 조건 악화 등)에서 재발하면 "REACQUIRE는
회전 없이 마지막 yaw·위치 유지, 스윕이 꼭 필요하면 마지막 방향 ± 좁은 범위로만 제한"
방향으로 손볼 것.

### 인프라
- `sim/gcs_keepalive.py` 신설: MAVSDK로 `udp:14540`에 연결해 GCS 하트비트 유지(PX4
  SITL이 GCS 연결 없이는 arm 거부, `NAV_DLL_ACT>0`) + `MPC_XY_VEL_MAX`/`MPC_XY_CRUISE`를
  0.3m/s로 설정해 접근 속도 상한을 PX4 파라미터 쪽에서 관철.
- `sim/run_kiosk_4walls.sh`에 keepalive 백그라운드 기동 단계 추가.
- `sim/run_full_auto.sh` 신설: 인프라+브리지+aruco+approach_control까지 한 번에
  기동하는 전체 자동 실행 스크립트(`bringup_level=4` 기본, 실제 arm+비행).

### 결과: HOLD 도달, 3단계(YAW/YAW_LATERAL/FULL) 재검증 불필요해짐
`bringup_level=4`(FULL)로 바로 검증 — 카메라고 정렬 후 APPROACH→HOLD 안정 도달,
3분 관찰 동안 HOLD 246틱/APPROACH 68틱/REACQUIRE 62틱(전부 grace 복구), 전체 SEARCH
재승격 0회. 벽까지 거리 평균 1.151m(standoff=1.0m 대비 표준 카메라 오프셋 이내).

### 다음
- 실기체 카메라 마운트고 확정되면 `camera_mount_offset` 실측치로 갱신
- REACQUIRE 반대 방향 스윕 이슈는 재발 시에만 대응 (위 항목 참고)

## 2026-09-24 — 4방향(4벽) HOLD 검증 완료 (SCRUM-25)
- `kiosk_4walls.sdf`의 북/동/남/서 4벽 각각에 대해 SEARCH → 타겟 ID 정렬 APPROACH → HOLD
  폐루프 도달 확인. 위치 setpoint + 20Hz 스트림 분리 + 카메라 마운트고 정렬
  (`takeoff_alt = marker_center_height - camera_mount_offset`) 조합이 방향에 무관하게 동작.
- 교훈 재확인: 근접 standoff에서 마커 가시성은 수평 정렬보다 **카메라 마운트고 vs 마커고**
  정렬이 좌우함 — 벽 방향이 바뀌어도 수직 축 조건이 같으면 결과가 동일하게 재현됨.
- `sim/*.sh` 실행 권한(+x)을 git 인덱스에 반영.

## 2026-09-29 — "TARGET_WALL 무관하게 항상 남쪽 벽 HOLD" 원인: 고아 aruco_pnp_node (SCRUM-25)
- 증상: `run_full_auto.sh`에 TARGET_WALL=서/북/동/남 무엇을 줘도 물리적 남쪽 벽 앞에서 HOLD.
  배너의 target_wall 값은 정상.
- 월드/매핑은 정상이었음: 4벽 텍스처를 직접 검출해보니 북 0-3 / 동 4-7 / 남 8-11 / 서 12-15로
  고유하고, `wall_geometry.WALL_ID_BASE`·`kiosk_4walls.sdf` 배치와 일치. `aruco_pnp_node`도
  타겟 벽 ID 그룹일 때만 `/target/pose` 발행(판단 로직 정상).
- 원인: 15:34 실행(target_wall=남)의 `aruco_pnp_node`가 고아로 살아남아(PPID=init) 이후 실행들과
  **같은 `/target/pose`에 계속 발행**. 실측: target_wall=북 실행의 aruco.log에 `PnP target=남`만
  36줄, 북 pose는 0건 → 제어부가 남 pose를 따라감.
- 고아가 생긴 이유: 비대화형 셸의 백그라운드(`&`) 프로세스는 SIGINT가 SIG_IGN으로 상속됨
  (`/proc/<pid>/status` SigIgn 비트로 확인). `ros2 run` 래퍼 PID에 `kill -INT`는 무시되고,
  래퍼만 죽으면 실제 노드는 남는다.
- 수정:
  - `run_full_auto.sh`: 기동 전(0단계)·종료 시 PID가 아니라 프로세스 패턴으로 kiosk_vision
    노드/브리지/tail을 INT→(5s)→KILL 정리. `approach_control_node`에도 `target_wall` 전달.
  - `approach_control_node`: `target_wall` 파라미터 추가, `frame_id != wall_<target_wall>`인
    pose는 버리고 경고(publisher 수 포함) — 다른 벽 pose가 섞여도 엉뚱한 벽으로 가지 않음.
- 검증: 매 방향 인프라 완전 재기동 후 `run_full_auto.sh`, HOLD 4회 연속(5s 간격) 확인, gz 기체 pose(ENU):
  서 (-1.87, 0.03) / 북 (0.05, 1.85) / 동 (1.87, 0.06) / 남 (-0.02, -1.87) — 전부 지정 벽
  (±3m) 앞 standoff 위치. 각 실행의 `PnP target=`은 전부 지정 벽, 제어부 frame_id 거부 0건,
  종료 후 잔여 노드 0개. 가드 단독 테스트(target=동에 wall_남 pose 주입)에서 거부 로그 확인.
