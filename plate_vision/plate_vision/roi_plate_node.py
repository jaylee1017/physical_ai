"""
ROI 안의 그릇 2개: 빈 그릇/음식 그릇(YOLO empty_plate, food_plate) + 띠(색) 인식 노드

Publish
  /plate/detections   std_msgs/String            ROI 안 그릇 목록(JSON): state, band, band_color, band_center, band_angle ...
  /plate/empty_target geometry_msgs/PointStamped ROI 안에 완전히 들어온 빈 그릇이 stable_frames 연속이면 중심 픽셀 좌표
Subscribe
  /tray/ready         std_msgs/Bool              True 동안은 empty_target 중지

창 단축키: [t] 마우스로 ROI 지정(파일에 저장)  [q] 종료
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

from plate_vision.roi_plates import RoiPlateJudge, draw


class RoiPlateNode(Node):

    def __init__(self):
        super().__init__('roi_plate_node')

        self.declare_parameter('camera_index', 1)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)
        self.declare_parameter('show_window', True)
        self.declare_parameter('weights', '')                  # 비우면 패키지의 models/best.pt
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('roi', [0, 0, 0, 0])            # 전부 0 이면 roi_file 에서 읽음
        self.declare_parameter('roi_file', 'plate_roi.json')
        self.declare_parameter('stable_frames', 5)

        weights = self.p('weights') or os.path.join(
            get_package_share_directory('plate_vision'), 'models', 'best.pt')
        model = YOLO(weights)
        missing = {'empty_plate', 'food_plate'} - set(model.names.values())
        if missing:
            self.get_logger().warning(f'모델에 클래스 {missing} 없음 (현재: {list(model.names.values())}) '
                                      '→ empty_plate/food_plate 로 학습한 모델이 필요')

        cam = self.p('camera_index')
        self.cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f'/dev/video{cam} 카메라를 열 수 없습니다.')
            raise RuntimeError('Camera open failed')
        # MJPG 가 아니면 1280x720 에서 USB 대역폭 초과로 화면이 깨짐. 백엔드마다 순서가 달라 앞뒤로 지정
        mjpg = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.p('frame_width'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.p('frame_height'))
        self.cap.set(cv2.CAP_PROP_FOURCC, mjpg)

        # 구독 콜백보다 먼저 초기화
        self.judge = RoiPlateJudge(model, roi=self.load_roi(), conf=self.p('conf'))
        self.paused = False
        self.empty_streak = 0
        self.frame = None

        self.det_pub = self.create_publisher(String, '/plate/detections', 10)
        self.target_pub = self.create_publisher(PointStamped, '/plate/empty_target', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/tray/ready', self.on_tray_ready, latched)

        self.timer = self.create_timer(0.03, self.timer_callback)
        self.get_logger().info(f'ROI plate START  camera=/dev/video{cam}  roi={self.judge.roi}  weights={weights}')
        if self.judge.roi is None:
            self.get_logger().warning('ROI 없음 → 창에서 t 키로 그릇 2개가 들어오는 영역을 지정하세요')

    def p(self, name):
        return self.get_parameter(name).value

    def load_roi(self):
        roi = self.p('roi')
        if any(roi):
            return tuple(roi)
        path = self.p('roi_file')
        if path and os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return tuple(json.load(f)['roi'])
        return None

    def save_roi(self, roi):
        with open(self.p('roi_file'), 'w', encoding='utf-8') as f:
            json.dump({'roi': list(roi)}, f)
        self.get_logger().info(f'ROI 저장: {list(roi)} → {os.path.abspath(self.p("roi_file"))}')

    def on_tray_ready(self, msg):
        if msg.data != self.paused:
            self.get_logger().info('트레이 가득 참 → 회수 중지' if msg.data else '회수 재개')
        self.paused = msg.data

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('카메라 frame을 읽지 못했습니다.')
            return
        self.frame = frame

        plates, band_mask = self.judge.judge(frame)
        self.det_pub.publish(String(data=json.dumps(
            [{k: v for k, v in pl.items() if not k.startswith('_')} for pl in plates])))

        # ROI 안에 완전히 들어온 빈 그릇이 연속으로 보일 때만 로봇팔에 전달
        empties = [pl for pl in plates if pl['full'] and pl['state'] == 'empty']
        self.empty_streak = self.empty_streak + 1 if empties else 0
        if empties and self.empty_streak >= self.p('stable_frames') and not self.paused:
            target = max(empties, key=lambda pl: pl['conf'])
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'top_camera'
            msg.point.x = float(target['cx'])
            msg.point.y = float(target['cy'])
            self.target_pub.publish(msg)

        if self.p('show_window'):
            view = draw(frame, plates, self.judge.roi)
            status = 'PAUSED (tray full)' if self.paused else f'empty streak {self.empty_streak}'
            cv2.putText(view, status + '   [t] ROI  [q] quit', (15, view.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('ROI Plates', view)
            cv2.imshow('Band Mask (ROI)', band_mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):
                x, y, w, h = cv2.selectROI('ROI Plates', self.frame, showCrosshair=False)
                if w > 0 and h > 0:
                    self.judge.roi = (x, y, x + w, y + h)
                    self.save_roi(self.judge.roi)
            elif key == ord('q') and rclpy.ok():
                rclpy.shutdown()

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoiPlateNode()
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
