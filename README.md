# physical_ai

팀 씨저스킹 — 비전 기반 빈 접시 회수 및 자율주행 양팔로봇 스마트 식기 수거 시스템 ROS2 패키지 모음

| 패키지 | 내용 |
|---|---|
| [`plate_vision`](plate_vision/README.md) | 회전 테이블 빈 접시 판별(YOLO + HSV), 트레이 빨간 띠 카운트 → 트레이 준비 완료 신호 |

## 사용법 (Ubuntu + ROS2)

이 저장소를 워크스페이스의 `src` 로 clone 하면 colcon 이 안의 패키지를 모두 찾는다.

```bash
mkdir -p ~/ros2_ws && cd ~/ros2_ws
git clone https://github.com/<계정>/physical_ai.git src
pip install ultralytics "numpy<2"
colcon build
source install/setup.bash
ros2 launch plate_vision plate_vision.launch.py plate_camera:=0 band_camera:=2
```
