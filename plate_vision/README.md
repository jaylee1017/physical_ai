# plate_vision

스마트 식기 수거 시스템의 비전 ROS2 패키지 (ament_python). 카메라 2대, 노드 2개.

| 노드 | 카메라 | 역할 |
|---|---|---|
| `plate_state_node` | A: 회전 테이블 위 | YOLO(`models/best.pt`)로 접시 위치 검출 → 접시 안쪽 음식 픽셀 비율로 empty / food → 빈 접시 위치를 로봇팔에 전달 |
| `red_count_node` | B: 트레이 옆 | 팀원 YellowCountNode 원본에서 색만 빨간색으로 변경. 쌓인 접시의 빨간 띠 count 가 3 이 되면 END → 트레이 준비 완료 신호 |

`red_count_node.py` 는 팀원 원본 코드와 같고, 바뀐 곳은 `# [변경]`, `# [추가]` 주석으로 표시함
(색 범위, 카메라 번호 파라미터, `/tray/ready` 발행, `/tray/reset` 구독).

## 동작 흐름

```
plate_state_node ── /plate/empty_target ──▶ 로봇팔 (빈 접시 pick & place)
red_count_node   ── /tray/ready = True  ──▶ 이동형 로봇 (Tag 1 로 출발)
                                       └──▶ plate_state_node (회수 중지)
새 트레이 놓으면 ── /tray/reset ──▶ red_count_node count 0, /tray/ready = False ──▶ 회수 재개
```

## 토픽

| 토픽 | 타입 | 보내는 노드 | 내용 |
|---|---|---|---|
| `/plate/detections` | `std_msgs/String` | plate_state_node | 매 프레임 접시 목록 JSON (`state`, `food_ratio`, `conf`, `cx`, `cy`, `box`) |
| `/plate/empty_target` | `geometry_msgs/PointStamped` | plate_state_node | 빈 접시가 연속 5프레임 보이면 중심 **픽셀** 좌표. 트레이 가득 찬 동안은 발행 안 함 |
| `/tray/ready` | `std_msgs/Bool` (transient_local) | red_count_node | END 시 True → 이동형 로봇 호출 |
| `/tray/reset` | `std_msgs/Empty` | (이동형 로봇 / 수동) | 새 트레이로 교체 후 count 초기화 |
| `/cmd_vel` | `geometry_msgs/Twist` | red_count_node | END 동안 정지 명령 (팀원 원본 동작) |

## 설치 / 빌드 (Ubuntu + ROS2)

```bash
pip install ultralytics "numpy<2"
# 이 폴더(plate_vision)를 ~/ros2_ws/src/ 에 복사
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
ros2 run plate_vision red_count_node --ros-args -p camera_index:=2

# 새 트레이로 교체 후 초기화 (Red Detection 창에서 END 후 r 키와 같음)
ros2 topic pub --once /tray/reset std_msgs/msg/Empty
```

## ROS2 없이 Windows 에서 테스트

`run_local.py` 는 두 노드 코드를 그대로 쓰고 ROS 부분만 가짜로 바꿔 실행한다 (토픽은 터미널에 출력).

```bash
python run_local.py --plate-cam 0 --band-cam 2
python run_local.py --only band --band-cam 2
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

red_count_node 의 색 범위·MIN_AREA·TARGET_COUNT 는 원본처럼 코드 안의 값을 직접 수정.
