"""
회전 테이블 빈 접시 판별 노드 (카메라 A: 회전 테이블 위)

YOLO 로 접시 위치 검출 → 접시 안쪽 음식 픽셀 비율로 empty / food (plate_state.py)

Publish
  /plate/detections   std_msgs/String            매 프레임 접시 목록(JSON)
  /plate/empty_target geometry_msgs/PointStamped 빈 접시가 stable_frames 연속 보이면 중심 픽셀 좌표
                                                  (로봇팔 pick 목표)
Subscribe
  /tray/ready         std_msgs/Bool              red_count_node 가 보냄. True 동안은 empty_target 중지
                                                  (트레이가 가득 차서 로봇팔이 더 집으면 안 됨)
"""
import json
import os

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String
from ultralytics import YOLO

from plate_vision.plate_state import food_ratio

EMPTY_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)


class PlateStateNode(Node):

    def __init__(self):
        super().__init__('plate_state_node')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)
        self.declare_parameter('show_window', True)
        self.declare_parameter('weights', '')          # 비우면 패키지의 models/best.pt
        self.declare_parameter('conf', 0.5)            # 접시 검출 신뢰도
        self.declare_parameter('food_thresh', 0.03)    # 음식 픽셀 비율이 이 이상이면 food
        self.declare_parameter('sat_min', 90)
        self.declare_parameter('val_min', 60)
        self.declare_parameter('shrink', 0.85)
        self.declare_parameter('white_min', 256)       # 흰 덩어리 음식 검사. 흰 접시면 256(끔), 검은 접시면 240
        self.declare_parameter('ignore_hue', [100, 130])  # 이 색상(파란 띠)은 음식에서 제외
        self.declare_parameter('stable_frames', 5)     # 연속 N프레임 empty 여야 target 발행

        weights = self.p('weights') or os.path.join(
            get_package_share_directory('plate_vision'), 'models', 'best.pt')
        self.model = YOLO(weights)
        self.get_logger().info(f'YOLO weights = {weights}')

        cam = self.p('camera_index')
        self.cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f'/dev/video{cam} 카메라를 열 수 없습니다.')
            raise RuntimeError('Camera open failed')
        # 1280x720 비압축(YUY2)은 USB 2.0 대역폭을 넘어 화면이 가로줄로 깨짐 → MJPG(압축) 필수
        # 백엔드마다 적용 순서가 달라서(V4L2: 포맷 먼저, Windows DSHOW: 해상도 먼저) 앞뒤로 두 번 지정
        mjpg = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.p('frame_width'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.p('frame_height'))
        self.cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('latin1')
        self.get_logger().info(
            f'camera format = {fourcc} {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
            f'{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}'
            + ('' if fourcc == 'MJPG' else '  ← MJPG 아님: 화면이 깨지면 frame_width/height 를 640x480 으로'))

        self.tray_full = False
        self.empty_streak = 0
        self.show_mask = False

        self.det_pub = self.create_publisher(String, '/plate/detections', 10)
        self.target_pub = self.create_publisher(PointStamped, '/plate/empty_target', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/tray/ready', self.on_tray_ready, latched)

        self.timer = self.create_timer(0.03, self.timer_callback)
        self.get_logger().info(f'Plate state detector START (camera /dev/video{cam})')

    def p(self, name):
        return self.get_parameter(name).value

    def on_tray_ready(self, msg):
        if msg.data != self.tray_full:
            self.get_logger().info('트레이 가득 참 → 회수 중지' if msg.data else '새 트레이 → 회수 재개')
        self.tray_full = msg.data

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('카메라 frame을 읽지 못했습니다.')
            return

        plates = self.detect_plates(frame)
        self.det_pub.publish(String(data=json.dumps(
            [{k: v for k, v in pl.items() if k != '_mask'} for pl in plates])))

        # 빈 접시가 연속으로 보이고, 트레이가 아직 안 찼을 때만 로봇팔에 목표 전달
        empties = [pl for pl in plates if pl['state'] == 'empty']
        self.empty_streak = self.empty_streak + 1 if empties else 0
        if empties and self.empty_streak >= self.p('stable_frames') and not self.tray_full:
            target = max(empties, key=lambda pl: pl['conf'])
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'plate_camera'
            msg.point.x = float(target['cx'])
            msg.point.y = float(target['cy'])
            self.target_pub.publish(msg)

        if self.p('show_window'):
            self.show(frame, plates)

    def detect_plates(self, frame):
        result = self.model(frame, conf=self.p('conf'), verbose=False)[0]
        plates = []
        for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            ratio, mask = food_ratio(frame, box, self.p('shrink'), self.p('sat_min'), self.p('val_min'),
                                     self.p('white_min'), ignore_hue=self.p('ignore_hue'))
            plates.append({
                'state': 'food' if ratio >= self.p('food_thresh') else 'empty',
                'food_ratio': round(ratio, 4),
                'conf': round(conf, 3),
                'cx': round((box[0] + box[2]) / 2, 1),
                'cy': round((box[1] + box[3]) / 2, 1),
                'box': [round(v, 1) for v in box],
                '_mask': mask,
            })
        return plates

    def show(self, view, plates):
        for pl in plates:
            x1, y1, x2, y2 = [int(v) for v in pl['box']]
            color = FOOD_COLOR if pl['state'] == 'food' else EMPTY_COLOR
            if self.show_mask and pl['_mask'] is not None:
                region = view[y1:y1 + pl['_mask'].shape[0], x1:x1 + pl['_mask'].shape[1]]
                region[pl['_mask'] > 0] = (255, 0, 255)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 3)
            label = f"{pl['state'].upper()} {pl['food_ratio'] * 100:.1f}%"
            cv2.putText(view, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        status = 'TRAY FULL - PAUSED' if self.tray_full else f'empty streak: {self.empty_streak}'
        cv2.putText(view, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if self.tray_full else (255, 255, 255), 2)
        cv2.imshow('Plate State', view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            self.show_mask = not self.show_mask
        elif key == ord('q') and rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PlateStateNode()
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
