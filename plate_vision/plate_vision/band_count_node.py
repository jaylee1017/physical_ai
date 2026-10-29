"""
트레이 적재 개수 확인 노드 (카메라 B: 트레이 옆에서 쌓인 접시 옆면을 봄)

파란 띠 개수(band_counter.py, 행 투영 방식)  OR  로봇팔 적재 횟수
→ 둘 중 하나라도 target_count 에 도달하면 트레이 가득 참 → /tray/ready = True

Publish
  /tray/band_count   std_msgs/Int32  확정된 띠 개수가 바뀔 때
  /tray/ready        std_msgs/Bool   가득 차면 True (transient_local, 이동형 로봇 호출 / 접시 판별 중지)
Subscribe
  /arm/place_count   std_msgs/Int32  로봇팔이 트레이에 놓은 누적 개수 (로봇팔 담당이 발행)
  /tray/reset        std_msgs/Empty  새 트레이로 교체 → 개수 초기화

창 단축키: [o] 마우스로 ROI 지정  [r] 초기화  [q] 종료
"""
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Empty, Int32

from plate_vision.band_counter import BandCounter, draw_debug


class BandCountNode(Node):

    def __init__(self):
        super().__init__('band_count_node')

        # ---------------- parameters ----------------
        self.declare_parameter('camera_index', 2)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('show_window', True)
        self.declare_parameter('target_count', 3)
        # 띠 검출 (실험실에서 조정할 값들)
        self.declare_parameter('roi', [0, 0, 0, 0])            # [x1, y1, x2, y2], 전부 0 이면 화면 전체
        self.declare_parameter('lower_hsv', [100, 120, 60])     # 파란색
        self.declare_parameter('upper_hsv', [130, 255, 255])
        self.declare_parameter('close_width', 25)
        self.declare_parameter('rel_fill', 0.5)
        self.declare_parameter('min_width_ratio', 0.15)
        self.declare_parameter('min_band_rows', 3)
        self.declare_parameter('band_height', 0)
        self.declare_parameter('stable_frames', 15)
        # 로봇팔 적재 횟수와 OR 조건
        self.declare_parameter('use_arm_count', True)

        p = self.p
        roi = p('roi')
        self.counter = BandCounter(
            lower_hsv=p('lower_hsv'), upper_hsv=p('upper_hsv'),
            roi=roi if any(roi) else None,
            close_width=p('close_width'), rel_fill=p('rel_fill'),
            min_width_ratio=p('min_width_ratio'), min_band_rows=p('min_band_rows'),
            band_height=p('band_height'), stable_frames=p('stable_frames'))
        self.target = p('target_count')

        # ---------------- camera ----------------
        cam = p('camera_index')
        self.cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f'/dev/video{cam} 카메라를 열 수 없습니다.')
            raise RuntimeError('Camera open failed')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, p('frame_width'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, p('frame_height'))

        # ---------------- state (구독 콜백보다 먼저 초기화) ----------------
        self.arm_count = 0
        self.full = False
        self.last_published = -1
        self.frame = None

        # ---------------- pub / sub ----------------
        self.count_pub = self.create_publisher(Int32, '/tray/band_count', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ready_pub = self.create_publisher(Bool, '/tray/ready', latched)
        self.create_subscription(Empty, '/tray/reset', self.reset_callback, 10)
        if p('use_arm_count'):
            self.create_subscription(Int32, '/arm/place_count', self.arm_callback, 10)

        self.ready_pub.publish(Bool(data=False))
        self.timer = self.create_timer(0.03, self.timer_callback)
        self.get_logger().info(
            f'Band counter START  camera=/dev/video{cam}  target={self.target}  '
            f'roi={self.counter.roi or "full frame"}  arm_count={"ON" if p("use_arm_count") else "OFF"}')

    def p(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------
    def arm_callback(self, msg):
        self.arm_count = msg.data
        self.get_logger().info(f'ARM PLACE COUNT = {self.arm_count}')

    def reset_callback(self, msg):
        # 이동형 로봇이 트레이를 가져가고 새 트레이가 놓이면 호출
        self.counter.reset()
        self.arm_count = 0
        self.full = False
        self.last_published = -1
        self.ready_pub.publish(Bool(data=False))
        self.get_logger().info('RESET → 새 트레이 count 시작')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('카메라 frame을 읽지 못했습니다.')
            return
        self.frame = frame

        result = self.counter.update(frame)
        band = result['stable']

        if band != self.last_published:
            self.count_pub.publish(Int32(data=band))
            self.get_logger().info(f'BAND COUNT = {band}')
            self.last_published = band

        # 띠 개수 OR 로봇팔 적재 횟수
        if not self.full and (band >= self.target or self.arm_count >= self.target):
            self.full = True
            reason = 'band' if band >= self.target else 'arm'
            self.ready_pub.publish(Bool(data=True))
            self.get_logger().info('====================')
            self.get_logger().info(f'  TRAY FULL ({reason}: band={band}, arm={self.arm_count})')
            self.get_logger().info('====================')

        if self.p('show_window'):
            self.show(frame, result)

    # ------------------------------------------------------------
    def show(self, frame, result):
        view, debug = draw_debug(frame, result, self.counter, self.target,
                                 extra_text=f'arm place count: {self.arm_count}')
        cv2.imshow('Band Count', view)
        cv2.imshow('Band Mask | row projection', debug)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.reset_callback(None)
        elif key == ord('o'):
            self.select_roi()
        elif key == ord('q') and rclpy.ok():
            rclpy.shutdown()

    def select_roi(self):
        # 마우스로 트레이 영역을 드래그 → Enter. 출력된 값을 파라미터로 저장해 두면 됨
        x, y, w, h = cv2.selectROI('Band Count', self.frame, showCrosshair=False)
        if w > 0 and h > 0:
            self.counter.roi = (x, y, x + w, y + h)
            self.counter.reset()
            self.get_logger().info(f'ROI 설정: -p roi:="[{x}, {y}, {x + w}, {y + h}]"')

    def destroy_node(self):
        if self.cap.isOpened():
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
