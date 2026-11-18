# physical_ai

팀 씨저스킹: 비전 기반 빈 접시 회수 및 자율주행 양팔로봇 스마트 식기 수거 시스템

이 저장소의 `plate_vision` 패키지는 **카메라 2대 + YOLO** 로 동작합니다.

| 카메라 | 보는 곳 | 노드 | 하는 일 |
|---|---|---|---|
| **A** | 회전 원판 (위에서) | `plate_detect_node` | **빈 그릇 / 음식 그릇 구분** |
| **B** | 빈 그릇을 쌓는 박스 | `band_count_node` | **파란 띠 개수 세기 → 3개면 주행로봇 호출** (`/tray/ready`) |

---

## 빠른 시작 (Ubuntu + ROS2)

ROS2(Humble 또는 Jazzy)가 설치되어 있다고 가정합니다.

**1. 받기** (`~/ros2_ws/src` 폴더가 아직 없을 때)
```bash
mkdir -p ~/ros2_ws && cd ~/ros2_ws
git clone https://github.com/jaylee1017/physical_ai.git src
```

**2. 파이썬 패키지 설치**
```bash
pip install ultralytics "numpy<2"
```
Ubuntu 24.04 에서 `externally-managed-environment` 오류가 나면 뒤에 `--break-system-packages` 를 붙이세요.

**3. 빌드**
```bash
cd ~/ros2_ws
colcon build --packages-select plate_vision
source install/setup.bash
```

**4. 카메라 번호 확인**
```bash
v4l2-ctl --list-devices
```
카메라마다 `/dev/video0`, `/dev/video2` 처럼 나옵니다. 카메라 하나에 번호가 2개 잡히면 보통 **앞 번호**가 실제 카메라입니다.

**5. 실행**

**카메라가 1대뿐일 때 (지금)**: 하나씩 따로 실행합니다. 카메라 번호는 4번에서 확인한 번호(보통 0)를 넣으세요.
```bash
# 카메라 A 역할: 원판 위 빈 그릇 / 음식 그릇 구분
ros2 run plate_vision plate_detect_node --ros-args -p camera_index:=0
```
```bash
# 카메라 B 역할: 박스 쪽 파란 띠 카운트 → 3개면 주행로봇 호출
ros2 run plate_vision band_count_node --ros-args -p camera_index:=0
```
`Ctrl+C`(또는 창에서 `q`)로 끄고, 카메라 방향을 바꾼 뒤 다른 노드를 실행하면 됩니다.

**카메라 2대일 때**: 한 번에 실행 (`plate_camera` = 카메라 A 번호, `box_camera` = 카메라 B 번호)
```bash
ros2 launch plate_vision plate_vision.launch.py plate_camera:=0 box_camera:=2
```

2대 모드에서는 창이 2개 뜹니다.
- **Camera A - Plates**: 빈 그릇은 초록 `EMPTY`, 음식 그릇은 주황 `FOOD`
- **Camera B - Bands**: 파란 띠는 보라색 박스, 위쪽에 `BAND 0 / 3`. 3 이 되면 `ROBOT CALLED`

창 단축키: `r` 띠 카운트 초기화(카메라 B 창), `q` 종료

**코드 업데이트 받기**
```bash
cd ~/ros2_ws/src && git pull
cd ~/ros2_ws && colcon build --packages-select plate_vision && source install/setup.bash
```

---

## 다른 팀원 코드와 연결하는 토픽

| 토픽 | 타입 | 보내는 쪽 → 받는 쪽 | 내용 |
|---|---|---|---|
| `/tray/ready` | `std_msgs/Bool` | 카메라 B → **주행로봇** | **True = 띠 3개 → 주행로봇 출발** |
| `/tray/reset` | `std_msgs/Empty` | **주행로봇** → 카메라 B | 박스를 가져간 뒤 보냄 (박스가 비워지면 다시 카운트) |
| `/tray/band_count` | `std_msgs/Int32` | 카메라 B → | 지금 확정된 띠 개수 |
| `/band/detections` | `std_msgs/String` | 카메라 B → | 띠 목록 JSON `[{"conf", "cx", "cy", "box"}]` |
| `/plate/detections` | `std_msgs/String` | 카메라 A → | 그릇 목록 JSON `[{"state": "empty"/"food", "conf", "cx", "cy", "box"}]` |
| `/plate/count` | `std_msgs/String` | 카메라 A → | 그릇 개수 JSON `{"empty": 1, "food": 1}` |

좌표(`cx`, `cy`, `box`)는 모두 **카메라 화면 픽셀** 기준입니다.

**터미널에서 확인 / 테스트**
```bash
ros2 topic echo /tray/ready                               # 주행로봇 호출 신호
ros2 topic echo /tray/band_count                          # 띠 개수
ros2 topic echo /plate/count                              # 빈 그릇 / 음식 그릇 개수
ros2 topic pub --once /tray/reset std_msgs/msg/Empty      # 박스 가져간 뒤 초기화
```

**주행로봇 쪽 예시 (Python)**
```python
from std_msgs.msg import Bool, Empty
from rclpy.qos import QoSProfile, DurabilityPolicy

# /tray/ready 는 transient_local 로 보내므로 받는 쪽도 맞춰야 늦게 켜져도 받음
qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
self.create_subscription(Bool, '/tray/ready', self.on_ready, qos)
self.reset_pub = self.create_publisher(Empty, '/tray/reset', 10)

def on_ready(self, msg):
    if msg.data:
        ...                                  # Tag 3 → Tag 1 이동, 박스 들기, Tag 2 에 내려놓기
        self.reset_pub.publish(Empty())      # 운반이 끝나면 초기화 신호
```

---

## 파라미터

**카메라 A (`plate_detect_node`)**
| 이름 | 기본값 | 설명 |
|---|---|---|
| `camera_index` | 0 | /dev/videoN |
| `conf` | 0.5 | YOLO 신뢰도 기준 |
| `show_window` | true | false 면 창 없이 실행 (모니터 없는 로봇 PC) |

**카메라 B (`band_count_node`)**
| 이름 | 기본값 | 설명 |
|---|---|---|
| `camera_index` | 2 | /dev/videoN |
| `target_count` | 3 | 띠가 이만큼 보이면 주행로봇 호출 |
| `stable_frames` | 15 | 같은 개수가 이만큼(약 0.5초) 유지돼야 확정 → 로봇팔이 잠깐 가려도 안 흔들림 |
| `conf` | 0.5 | YOLO 신뢰도 기준 |
| `show_window` | true | 창 표시 |

한 노드만 따로 실행하기:
```bash
ros2 run plate_vision plate_detect_node --ros-args -p camera_index:=0
ros2 run plate_vision band_count_node --ros-args -p camera_index:=2 -p target_count:=3
```

---

## 모델

- 파일: `plate_vision/models/best.pt` (YOLOv8n, 두 노드가 같은 모델 사용)
- 클래스: `band`(파란 띠), `empty_plate`(빈 그릇), `food_plate`(음식 그릇)
- 성능 (test): band mAP50 0.982 / empty_plate 0.995 / food_plate 0.995
- 원본 사진 307장 전수 검사: 그릇 판정 307/307, 띠 검출 307/307
- 지금 띠는 **회전 원판 위 그릇** 사진으로 학습됨. 박스에 쌓는 방식이 정해지면 박스 사진을 추가 학습해야 함

ROS 없이 파이썬에서 모델만 쓰기:
```python
from ultralytics import YOLO
model = YOLO('plate_vision/models/best.pt')
r = model(frame, conf=0.5)[0]                 # frame = OpenCV 이미지
for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
    print(r.names[int(cls)], box.tolist())    # band / empty_plate / food_plate
```

---

## Windows 에서 테스트 (ROS2 없이)

```bash
conda activate yolo_course
cd C:\Physical_AI\ros2_ws\src\plate_vision
python run_local.py --plate-cam 1 --box-cam 2                      # 카메라 2대
python run_local.py --only box --box-cam 1                         # 카메라 B 만
python run_local.py --plate-src test_images --box-src test_images  # 카메라 없이 사진으로
```

## 다시 학습할 때

Roboflow 에서 YOLOv8 형식으로 받아 압축을 푼 폴더 기준:
```bash
python plate_vision/tools/clean_labels.py --src <데이터셋폴더>             # 자동 라벨링 중복 박스 정리
python plate_vision/tools/train.py --data <데이터셋폴더>_clean/data.yaml    # 학습 → 평가 → models/best.pt 교체
```
사진 찍기: `python plate_vision/tools/capture_dataset.py --cam 1`

---

## 자주 나는 오류

| 오류 | 해결 |
|---|---|
| `카메라를 열 수 없습니다` | `v4l2-ctl --list-devices` 로 번호 확인. 권한 문제면 `sudo usermod -aG video $USER` 후 재로그인 |
| `No module named 'ultralytics'` | `python3 -m pip install ultralytics "numpy<2"` |
| `Package 'plate_vision' not found` | `source ~/ros2_ws/install/setup.bash` 를 안 함 |
| 화면이 가로줄로 깨짐 | USB 허브 말고 PC 에 직접 연결, 카메라 2대는 서로 다른 USB 포트에 |
| 코드를 고쳤는데 반영이 안 됨 | `colcon build --packages-select plate_vision` 다시 실행 |

## 폴더 구조

```
plate_vision/
  plate_vision/plate_detect_node.py   카메라 A: 빈 그릇 / 음식 그릇 구분
  plate_vision/band_count_node.py     카메라 B: 파란 띠 카운트 → 주행로봇 호출
  plate_vision/camera.py              카메라 열기 (공용)
  launch/plate_vision.launch.py       두 노드 한 번에 실행
  models/best.pt                      YOLO 모델
  tools/                              사진 찍기, 라벨 정리, 학습 스크립트
  run_local.py                        Windows 테스트용 (ROS2 없이)
  test_images/                        테스트용 사진
  original/                           팀원 원본 코드 보관 (참고용, 실행 안 함)
```
