"""
카메라 B (빈 그릇을 쌓는 박스 쪽): YOLO 로 파란 띠 검출 → 개수 세기 → 3개면 주행로봇 호출

  1) 매 프레임 YOLO(models/best.pt)의 band 박스 개수를 셈 (지금 화면에 보이는 띠 개수)
  2) 같은 개수가 stable_frames 동안 유지되면 '확정 개수' 로 인정 (로봇팔이 잠깐 가려도 흔들리지 않게)
  3) 확정 개수 ≥ target_count(3) → /tray/ready = True (주행로봇 출발)
  4) 주행로봇이 박스를 가져간 뒤 /tray/reset → 띠 개수가 target 아래로 내려가야 다시 호출 가능
     (박스를 아직 안 치웠는데 바로 또 호출되는 것 방지)

Publish
  /tray/band_count  std_msgs/Int32   확정 띠 개수가 바뀔 때
  /tray/ready       std_msgs/Bool    True = 띠 3개 → 주행로봇 출발 (늦게 켜진 노드도 마지막 값을 받게 transient_local)
  /band/detections  std_msgs/String  매 프레임 띠 목록 JSON [{"conf", "cx", "cy", "box"}, ...]
Subscribe
  /tray/reset       std_msgs/Empty   주행로봇이 박스를 가져간 뒤 보냄

창 단축키: [r] 초기화  [q] 종료
"""
import json
import os

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Empty, Int32, String
from ultralytics import YOLO

from plate_vision.camera import open_camera

BAND_COLOR = (255, 0, 255)


class BandCountNode(Node):

    def __init__(self):
        super().__init__('band_count_node')

        self.declare_parameter('camera_index', 2)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)
        self.declare_parameter('show_window', True)
        self.declare_parameter('weights', '')      # 비우면 패키지의 models/best.pt
        self.declare_parameter('conf', 0.5)        # YOLO 신뢰도 기준
        self.declare_parameter('target_count', 3)  # 띠가 이만큼 보이면 주행로봇 호출
        self.declare_parameter('stable_frames', 15)  # 같은 개수가 이만큼 유지돼야 확정 (약 0.5초)

        weights = self.p('weights') or os.path.join(
            get_package_share_directory('plate_vision'), 'models', 'best.pt')
        self.model = YOLO(weights)
        if 'band' not in self.model.names.values():
            self.get_logger().error(f'모델에 band 클래스가 없음: {list(self.model.names.values())}')

        cam = self.p('camera_index')
        self.cap, fmt = open_camera(cam, self.p('frame_width'), self.p('frame_height'))
        if self.cap is None:
            self.get_logger().error(f'/dev/video{cam} 카메라를 열 수 없습니다.')
            raise RuntimeError('Camera open failed')

        # 상태 (구독 콜백보다 먼저 초기화)
        self.target = self.p('target_count')
        self.count = 0          # 확정 띠 개수
        self.last_raw = -1      # 직전 프레임 띠 개수
        self.same = 0           # 같은 개수가 이어진 프레임 수
        self.full = False       # 주행로봇 호출함
        self.armed = True       # 호출 가능 상태 (reset 후 띠가 target 아래로 내려가야 다시 True)
        self.last_pub = -1

        self.count_pub = self.create_publisher(Int32, '/tray/band_count', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ready_pub = self.create_publisher(Bool, '/tray/ready', latched)
        self.band_pub = self.create_publisher(String, '/band/detections', 10)
        self.create_subscription(Empty, '/tray/reset', self.reset_callback, 10)

        self.ready_pub.publish(Bool(data=False))
        self.timer = self.create_timer(0.03, self.timer_callback)
        self.get_logger().info(f'[카메라 B] 파란 띠 카운트 START  /dev/video{cam} {fmt}  target={self.target}')

    def p(self, name):
        return self.get_parameter(name).value

    def reset_callback(self, msg):
        # 주행로봇이 박스를 가져간 뒤 호출
        self.full = False
        self.armed = False      # 띠가 target 아래로 내려간 걸 확인한 뒤에 다시 셀 수 있게
        self.ready_pub.publish(Bool(data=False))
        self.get_logger().info('RESET → 박스가 비워지면(띠 < target) 다시 카운트')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('카메라 frame을 읽지 못했습니다.')
            return

        r = self.model(frame, conf=self.p('conf'), verbose=False)[0]
        bands = [{'conf': round(c, 3), 'cx': round((b[0] + b[2]) / 2, 1), 'cy': round((b[1] + b[3]) / 2, 1),
                  'box': [round(v, 1) for v in b]}
                 for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
                 if r.names[int(k)] == 'band']
        self.band_pub.publish(String(data=json.dumps(bands)))

        # 같은 개수가 stable_frames 동안 이어져야 확정
        raw = len(bands)
        if raw == self.last_raw:
            self.same += 1
        else:
            self.last_raw, self.same = raw, 1
        if self.same >= self.p('stable_frames'):
            self.count = raw

        if self.count != self.last_pub:
            self.count_pub.publish(Int32(data=self.count))
            self.get_logger().info(f'띠 개수 {self.count}/{self.target}')
            self.last_pub = self.count

        if not self.armed and self.count < self.target:
            self.armed = True
            self.get_logger().info('박스 비워짐 확인 → 다시 카운트 시작')
        if self.armed and not self.full and self.count >= self.target:
            self.full = True
            self.ready_pub.publish(Bool(data=True))
            self.get_logger().info(f'==== 띠 {self.count}개 → 주행로봇 호출 (/tray/ready = True) ====')

        if self.p('show_window'):
            self.show(frame, bands, raw)

    def show(self, frame, bands, raw):
        view = frame.copy()
        for i, b in enumerate(bands, 1):
            x1, y1, x2, y2 = [int(v) for v in b['box']]
            cv2.rectangle(view, (x1, y1), (x2, y2), BAND_COLOR, 3)
            cv2.putText(view, f"band{i} {b['conf']:.2f}", (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, BAND_COLOR, 2)
        text = f'BAND {self.count} / {self.target}  (now {raw})' + ('   -> ROBOT CALLED' if self.full else '')
        cv2.putText(view, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(view, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if self.full else (255, 255, 255), 2)
        cv2.putText(view, '[r] reset  [q] quit', (15, view.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Camera B - Bands', view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.reset_callback(None)
        elif key == ord('q') and rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BandCountNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
