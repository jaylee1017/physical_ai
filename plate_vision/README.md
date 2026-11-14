# plate_vision

스마트 식기 수거 시스템의 비전 ROS2 패키지 (ament_python). 카메라 2대, 노드 2개.

| 노드 | 카메라 | 역할 |
|---|---|---|
| **`empty_count_node`** | **위에서 1대** | **최종 구조.** YOLO(empty_plate/food_plate) 추적으로 빈 그릇을 ID 당 한 번씩 세고, 3개가 되면 `/tray/ready` 로 주행로봇 호출. 학습: `python plate_vision/tools/train.py --data <data.yaml>`, 실행: `ros2 launch plate_vision empty_count.launch.py camera:=0`, Windows: `python run_local.py --count-cam 1` |
| **`top_vision_node`** | **위에서 1대** | **현재 구조.** 같은 화면에서 원판 위 접시 판별 + 트레이 영역(ROI) 파란 띠 세기. `blobs`(나란히 놓기) / `events`(겹쳐 쌓기) 방식 선택. 실행: `ros2 launch plate_vision top_vision.launch.py camera:=0`, Windows: `python run_local.py --top-cam 1` |
| `plate_state_node` | A: 회전 테이블 위 | YOLO(`models/best.pt`)로 접시 위치 검출 → 접시 안쪽 음식 픽셀 비율로 empty / food → 빈 접시 위치를 로봇팔에 전달 |
| `band_count_node` | B: 트레이 옆 | 쌓인 접시 옆면의 **파란 띠 줄 수**를 행 투영으로 셈 (`band_counter.py`). 띠 3개 **또는** 로봇팔 적재 3회 → 트레이 가득 참 |
| `red_count_node` | (B) | 이전 버전: 팀원 원본 로직(나타났다 사라지는 횟수)에 색만 빨간색. 비교용으로 남겨 둠, launch 에서는 안 씀 |

팀원 원본 코드는 `original/yellow_count_node.py` 에 수정 없이 보관.

## 띠 세기 원리 (band_counter.py)

1. **ROI**: 트레이 부분만 봄 (창에서 `o` 키 → 마우스로 드래그)
2. **HSV 마스크**: 파란색(H 100~130, S ≥ 120)만 흰색
3. **노이즈 정리**: 3x3 opening 으로 점 제거 → **가로 25x1 커널** closing 으로 띠의 좌우 끊김만 연결
   (세로 1px 커널이라 위아래 띠 사이 간격은 메워지지 않음 → 띠끼리 안 합쳐짐)
4. **행 투영**: 각 행의 파란 픽셀 수 ≥ max(가장 넓은 행 x 0.5, ROI 폭 x 0.15) 이면 '띠 행'
5. **구간 세기**: 연속된 띠 행 구간 1개 = 띠 1개 (3행보다 얇으면 노이즈). `band_height` 를 주면 붙은 띠는 높이로 나눠 계산
6. **안정화**: 같은 개수가 15프레임(약 0.5초) 유지돼야 확정 → 팔이 잠깐 가려도 흔들리지 않음

## 동작 흐름

```
plate_state_node ── /plate/empty_target ──▶ 로봇팔 (빈 접시 pick & place)
로봇팔           ── /arm/place_count   ──▶ band_count_node
band_count_node  ── /tray/ready = True ──▶ 이동형 로봇 (Tag 1 로 출발)
                                       └──▶ plate_state_node (회수 중지)
새 트레이 놓으면 ── /tray/reset ──▶ band_count_node 개수 0, /tray/ready = False ──▶ 회수 재개
```

## 토픽

| 토픽 | 타입 | 보내는 쪽 | 내용 |
|---|---|---|---|
| `/plate/detections` | `std_msgs/String` | plate_state_node | 매 프레임 접시 목록 JSON (`state`, `food_ratio`, `conf`, `cx`, `cy`, `box`) |
| `/plate/empty_target` | `geometry_msgs/PointStamped` | plate_state_node | 빈 접시가 연속 5프레임 보이면 중심 **픽셀** 좌표. 트레이 가득 찬 동안은 발행 안 함 |
| `/tray/band_count` | `std_msgs/Int32` | band_count_node | 확정된 띠 개수가 바뀔 때 |
| `/tray/ready` | `std_msgs/Bool` (transient_local) | band_count_node | 가득 차면 True → 이동형 로봇 호출 |
| `/arm/place_count` | `std_msgs/Int32` | **로봇팔 담당** | 트레이에 놓은 누적 개수 (`use_arm_count:=false` 로 끌 수 있음) |
| `/tray/reset` | `std_msgs/Empty` | (이동형 로봇 / 수동) | 새 트레이로 교체 후 초기화 |

`band_count_node` 는 `/cmd_vel` 을 보내지 않음 (원본은 END 시 정지 명령을 보냈지만, 가득 찬 순간은 이동형 로봇이 출발해야 하므로 충돌 위험).

## 설치 / 빌드 (Ubuntu + ROS2)

```bash
pip install ultralytics "numpy<2"
cd ~/ros2_ws
colcon build --packages-select plate_vision
source install/setup.bash
```

## 실행

```bash
v4l2-ctl --list-devices                      # 카메라 번호 확인
ros2 launch plate_vision plate_vision.launch.py plate_camera:=0 band_camera:=2

# 하나씩 실행
ros2 run plate_vision plate_state_node --ros-args -p camera_index:=0
ros2 run plate_vision band_count_node --ros-args -p camera_index:=2 -p roi:="[200, 100, 440, 400]"

# 새 트레이로 교체 후 초기화 (Band Count 창의 r 키와 같음)
ros2 topic pub --once /tray/reset std_msgs/msg/Empty
# 로봇팔 없이 적재 횟수 흉내
ros2 topic pub --once /arm/place_count std_msgs/msg/Int32 "{data: 3}"
```

## ROS2 없이 Windows 에서 테스트

`run_local.py` 는 두 노드 코드를 그대로 쓰고 ROS 부분만 가짜로 바꿔 실행한다 (토픽은 터미널에 출력).

```bash
python run_local.py --plate-cam 0 --band-cam 2
python run_local.py --only band --band-cam 2
python run_local.py --plate-src test_images --band-src demo   # 카메라 없이
python run_local.py --only plate --plate-cam 2
```

## plate_state_node 파라미터

| 이름 | 기본값 | 설명 |
|---|---|---|
| `camera_index` | 0 | /dev/videoN |
| `frame_width`, `frame_height` | 1280, 720 | 모델 학습 해상도와 같은 16:9 |
| `food_thresh` | 0.05 | 접시 안 음식 픽셀 비율이 이 이상이면 food |
| `sat_min` | 90 | 색 있는 음식으로 볼 최소 채도 |
| `white_min` | 240 | 하얀 덩어리 음식(계란초밥)으로 볼 최소 밝기, 256 이면 끔 |
| `stable_frames` | 5 | 빈 접시가 연속 N프레임 보여야 target 발행 |

## band_count_node 파라미터 (실험실에서 조정)

| 이름 | 기본값 | 설명 |
|---|---|---|
| `roi` | [0,0,0,0] | 트레이 영역 [x1,y1,x2,y2], 전부 0 이면 화면 전체. 창에서 `o` 키로 지정하면 값이 로그에 출력됨 |
| `lower_hsv`, `upper_hsv` | [100,120,60], [130,255,255] | 파란색 범위 |
| `close_width` | 25 | 띠 좌우 끊김 연결 폭(px). 반사로 끊긴 틈보다 크게 |
| `rel_fill` | 0.5 | 가장 넓은 행 대비 이 비율 이상이면 띠 행 |
| `min_width_ratio` | 0.15 | ROI 폭 대비 최소 띠 폭 (작은 파란 물체 제외) |
| `min_band_rows` | 3 | 이보다 얇은 구간은 노이즈 |
| `band_height` | 0 | 띠 1개 높이(px). 띠끼리 완전히 붙을 때만 설정 |
| `stable_frames` | 15 | 같은 개수가 이만큼 유지돼야 확정 |
| `target_count` | 3 | 가득 참 기준 |
| `use_arm_count` | true | `/arm/place_count` 와 OR 조건 |
