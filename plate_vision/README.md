# plate_vision

카메라 2대 + YOLO 비전 패키지 (ament_python)

- **카메라 A** `plate_detect_node`: 회전 원판 위 빈 그릇 / 음식 그릇 구분
- **카메라 B** `band_count_node`: 박스 쪽 파란 띠 개수 → 3개면 `/tray/ready` 로 주행로봇 호출

```bash
ros2 launch plate_vision plate_vision.launch.py plate_camera:=0 box_camera:=2
```

설치, 실행, 토픽, 파라미터, 오류 해결은 **저장소 맨 위 [README](../README.md)** 를 보세요.
