"""
위에서 내려다보는 카메라 1대로 빈 접시 판별 + 트레이 파란 띠 세기

화면 = 회전 원판 + 트레이(ROI). 같은 프레임으로
  1) 접시 판별 (plate_state.py): 트레이 ROI 밖의 접시만 → 빈 접시면 로봇팔에 위치 전달
     (트레이 위 접시는 이미 회수한 접시라 다시 집으면 안 됨)
  2) 띠 세기 (top_band_counter.py): 트레이 ROI 안의 파란 띠만
     띠 개수 OR 로봇팔 적재 횟수 ≥ target_count → 트레이 가득 참

Publish
  /plate/detections   std_msgs/String            원판 위 접시 목록(JSON)
  /plate/empty_target geometry_msgs/PointStamped 빈 접시가 stable_frames 연속 보이면 중심 픽셀 좌표 (가득 차면 중지)
  /tray/band_count    std_msgs/Int32             확정 띠 개수가 바뀔 때
  /tray/ready         std_msgs/Bool              가득 차면 True (transient_local)
Subscribe
  /arm/place_count    std_msgs/Int32             로봇팔 적재 누적 개수
  /tray/reset         std_msgs/Empty             새 트레이 → 초기화

창 단축키: [t] 마우스로 트레이 영역 지정(파일에 저장)  [b] 세는 방식 blobs/events 전환
           [r] 초기화  [m] 음식 마스크  [q] 종료
"""
import json
import os

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Empty, Int32, String
from ultralytics import YOLO

from plate_vision.plate_state import food_ratio
from plate_vision.top_band_counter import TopBandCounter

EMPTY_COLOR = (0, 255, 0)
FOOD_COLOR = (0, 140, 255)
TRAY_COLOR = (0, 255, 255)


class TopVisionNode(Node):

    def __init__(self):
        super().__init__('top_vision_node')

        # ---------------- parameters ----------------
        self.declare_parameter('camera_index', 1)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)
        self.declare_parameter('show_window', True)
        # 접시 판별
        self.declare_parameter('weights', '')
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('food_thresh', 0.03)
        self.declare_parameter('sat_min', 90)
        self.declare_parameter('val_min', 60)
        self.declare_parameter('shrink', 0.85)
        self.declare_parameter('white_min', 256)
        self.declare_parameter('ignore_hue', [100, 130])
        self.declare_parameter('stable_frames', 5)
        # 띠 세기
        self.declare_parameter('tray_roi', [0, 0, 0, 0])        # 전부 0 이면 roi_file 에서 읽음
        self.declare_parameter('roi_file', 'plate_vision_roi.json')
        self.declare_parameter('band_mode', 'blobs')             # blobs(나란히) / events(겹쳐 쌓기)
        self.declare_parameter('lower_hsv', [100, 120, 60])
        self.declare_parameter('upper_hsv', [130, 255, 255])
        self.declare_parameter('min_area_ratio', 0.003)
        self.declare_parameter('band_stable_frames', 15)
        self.declare_parameter('reset_frames', 10)
        self.declare_parameter('target_count', 3)
        self.declare_parameter('use_arm_count', True)

        p = self.p

        weights = p('weights') or os.path.join(
            get_package_share_directory('plate_vision'), 'models', 'best.pt')
        self.model = YOLO(weights)

        # ---------------- camera ----------------
        cam = p('camera_index')
        self.cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f'/dev/video{cam} 카메라를 열 수 없습니다.')
            raise RuntimeError('Camera open failed')
        # MJPG 가 아니면 1280x720 에서 USB 대역폭 초과로 화면이 깨짐. 백엔드마다 순서가 달라 앞뒤로 지정
        mjpg = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, p('frame_width'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, p('frame_height'))
        self.cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('latin1')

        # ---------------- state (구독 콜백보다 먼저) ----------------
        self.band = TopBandCounter(
            roi=self.load_roi(), lower_hsv=p('lower_hsv'), upper_hsv=p('upper_hsv'),
            mode=p('band_mode'), min_area_ratio=p('min_area_ratio'),
            stable_frames=p('band_stable_frames'), reset_frames=p('reset_frames'))
        self.target = p('target_count')
        self.arm_count = 0
        self.full = False
        self.last_band = -1
        self.empty_streak = 0
        self.show_mask = False
        self.frame = None

        # ---------------- pub / sub ----------------
        self.det_pub = self.create_publisher(String, '/plate/detections', 10)
        self.target_pub = self.create_publisher(PointStamped, '/plate/empty_target', 10)
        self.count_pub = self.create_publisher(Int32, '/tray/band_count', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ready_pub = self.create_publisher(Bool, '/tray/ready', latched)
        self.create_subscription(Empty, '/tray/reset', self.reset_callback, 10)
        if p('use_arm_count'):
            self.create_subscription(Int32, '/arm/place_count', self.arm_callback, 10)

        self.ready_pub.publish(Bool(data=False))
        self.timer = self.create_timer(0.03, self.timer_callback)
        self.get_logger().info(
            f'Top vision START  camera=/dev/video{cam} {fourcc} '
            f'{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  '
            f'tray_roi={self.band.roi}  band_mode={self.band.mode}  target={self.target}')
        if self.band.roi is None:
            self.get_logger().warning('트레이 영역이 없음 → 창에서 t 키로 지정하세요 (지정 전에는 화면 전체에서 띠를 셈)')

    def p(self, name):
        return self.get_parameter(name).value

    # ---------------- 트레이 영역 저장/불러오기 ----------------
    def load_roi(self):
        roi = self.p('tray_roi')
        if any(roi):
            return tuple(roi)
        path = self.p('roi_file')
        if path and os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return tuple(json.load(f)['tray_roi'])
        return None

    def save_roi(self, roi):
        path = self.p('roi_file')
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'tray_roi': list(roi)}, f)
        self.get_logger().info(f'트레이 영역 저장: {list(roi)} → {os.path.abspath(path)}')

    # ---------------- callbacks ----------------
    def arm_callback(self, msg):
        self.arm_count = msg.data
        self.get_logger().info(f'ARM PLACE COUNT = {self.arm_count}')

    def reset_callback(self, msg):
        self.band.reset()
        self.arm_count = 0
        self.full = False
        self.last_band = -1
        self.empty_streak = 0
        self.ready_pub.publish(Bool(data=False))
        self.get_logger().info('RESET → 새 트레이 count 시작, 회수 재개')

    def in_tray(self, cx, cy):
        r = self.band.roi
        return r is not None and r[0] <= cx <= r[2] and r[1] <= cy <= r[3]

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('카메라 frame을 읽지 못했습니다.')
            return
        self.frame = frame

        # ---------- 1) 트레이 띠 세기 ----------
        band = self.band.update(frame)
        if band['count'] != self.last_band:
            self.count_pub.publish(Int32(data=band['count']))
            self.get_logger().info(f'BAND COUNT = {band["count"]} ({band["mode"]})')
            self.last_band = band['count']
        if not self.full and (band['count'] >= self.target or self.arm_count >= self.target):
            self.full = True
            self.ready_pub.publish(Bool(data=True))
            reason = 'band' if band['count'] >= self.target else 'arm'
            self.get_logger().info(f'==== TRAY FULL ({reason}: band={band["count"]}, arm={self.arm_count}) ====')

        # ---------- 2) 원판 위 접시 판별 (트레이 위 접시는 제외) ----------
        plates = self.detect_plates(frame)
        table = [pl for pl in plates if not pl['on_tray']]
        self.det_pub.publish(String(data=json.dumps(
            [{k: v for k, v in pl.items() if k != '_mask'} for pl in table])))

        empties = [pl for pl in table if pl['state'] == 'empty']
        self.empty_streak = self.empty_streak + 1 if empties else 0
        if empties and self.empty_streak >= self.p('stable_frames') and not self.full:
            target = max(empties, key=lambda pl: pl['conf'])
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'top_camera'
            msg.point.x = float(target['cx'])
            msg.point.y = float(target['cy'])
            self.target_pub.publish(msg)

        if self.p('show_window'):
            self.show(frame, plates, band)

    def detect_plates(self, frame):
        result = self.model(frame, conf=self.p('conf'), verbose=False)[0]
        plates = []
        for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            ratio, mask = food_ratio(frame, box, self.p('shrink'), self.p('sat_min'), self.p('val_min'),
                                     self.p('white_min'), ignore_hue=self.p('ignore_hue'))
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            plates.append({
                'state': 'food' if ratio >= self.p('food_thresh') else 'empty',
                'food_ratio': round(ratio, 4),
                'conf': round(conf, 3),
                'cx': round(cx, 1),
                'cy': round(cy, 1),
                'box': [round(v, 1) for v in box],
                'on_tray': self.in_tray(cx, cy),
                '_mask': mask,
            })
        return plates

    # ---------------- 화면 ----------------
    def show(self, frame, plates, band):
        view = frame.copy()
        r = self.band.roi
        if r is not None:
            cv2.rectangle(view, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), TRAY_COLOR, 2)
            cv2.putText(view, 'TRAY', (int(r[0]) + 5, int(r[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TRAY_COLOR, 2)
        for x1, y1, x2, y2 in band['blobs']:
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for pl in plates:
            x1, y1, x2, y2 = [int(v) for v in pl['box']]
            if pl['on_tray']:
                cv2.rectangle(view, (x1, y1), (x2, y2), (160, 160, 160), 1)
                continue
            color = FOOD_COLOR if pl['state'] == 'food' else EMPTY_COLOR
            if self.show_mask and pl['_mask'] is not None:
                view[y1:y1 + pl['_mask'].shape[0], x1:x1 + pl['_mask'].shape[1]][pl['_mask'] > 0] = (255, 0, 255)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 3)
            cv2.putText(view, f"{pl['state'].upper()} {pl['food_ratio'] * 100:.1f}%", (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        status = (f"BAND {band['count']}/{self.target} [{band['mode']}] now {band['now']}   "
                  f"arm {self.arm_count}   " + ('TRAY FULL - PAUSED' if self.full else f'empty streak {self.empty_streak}'))
        cv2.putText(view, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(view, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if self.full else (255, 255, 255), 2)
        cv2.putText(view, '[t] tray area  [b] blobs/events  [r] reset  [m] mask  [q] quit',
                    (15, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Top Vision', view)
        cv2.imshow('Tray Blue Mask', band['mask'])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            x, y, w, h = cv2.selectROI('Top Vision', self.frame, showCrosshair=False)
            if w > 0 and h > 0:
                self.band.roi = (x, y, x + w, y + h)
                self.band.reset()
                self.save_roi(self.band.roi)
        elif key == ord('b'):
            self.get_logger().info(f'band_mode → {self.band.toggle_mode()}')
            self.last_band = -1
        elif key == ord('r'):
            self.reset_callback(None)
        elif key == ord('m'):
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
    node = TopVisionNode()
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
