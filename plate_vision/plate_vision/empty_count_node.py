"""
카메라 1대 + YOLO(empty_plate / food_plate / band)로 빈 그릇을 세고, 3개가 되면 주행로봇 호출

  1) 화면 전체를 YOLO 추적(model.track) → 그릇마다 고유 번호(ID)
     파란 띠(band)는 띠 중심이 들어 있는 그릇 중 중심이 가장 가까운 그릇에 배정
  2) 같은 ID 가 empty_plate 로 stable_frames 연속 보이면 '빈 그릇 1개' 로 카운트 (ID 당 한 번만)
     추적이 잠깐 끊겨 새 ID 가 붙어도, 직전에 센 그릇과 같은 자리면 다시 세지 않음
  3) 카운트가 target_count(3) 에 도달 → /tray/ready = True (주행로봇 출발, 로봇팔 적재 중지)
  4) 주행로봇이 그릇 3개를 가져가면 /tray/reset → 카운트 0 부터 다시

Publish
  /plate/detections   std_msgs/String            화면 속 그릇 목록(JSON): id, state, counted, band, band_box, cx, cy, conf, box
  /band/detections    std_msgs/String            화면 속 파란 띠 목록(JSON): box, cx, cy, conf, plate_id(어느 그릇의 띠인지)
  /plate/empty_target geometry_msgs/PointStamped 빈 그릇 중심 픽셀 좌표 (로봇팔 pick 목표, 가득 차면 중지)
  /tray/empty_count   std_msgs/Int32             누적 빈 그릇 개수가 바뀔 때
  /tray/ready         std_msgs/Bool              3개 도달 시 True (transient_local)
Subscribe
  /tray/reset         std_msgs/Empty             주행로봇이 가져간 뒤 → 카운트 초기화

창 단축키: [r] 카운트 초기화  [q] 종료
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

EMPTY_COLOR = (0, 255, 0)
FOOD_COLOR = (0, 140, 255)
COUNTED_COLOR = (255, 255, 0)
BAND_COLOR = (255, 0, 255)


class EmptyCountNode(Node):

    def __init__(self):
        super().__init__('empty_count_node')

        self.declare_parameter('camera_index', 1)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)
        self.declare_parameter('show_window', True)
        self.declare_parameter('weights', '')                 # 비우면 패키지의 models/best.pt
        self.declare_parameter('conf', 0.5)                   # YOLO 신뢰도 기준
        self.declare_parameter('stable_frames', 8)            # 빈 그릇으로 이만큼 연속 보여야 카운트 (약 0.3초)
        self.declare_parameter('target_count', 3)             # 이만큼 모이면 주행로봇 호출
        self.declare_parameter('same_dist', 80)               # 추적이 끊겨도 이 거리(px) 안이면 같은 그릇
        self.declare_parameter('lost_frames', 30)             # 그 판단을 몇 프레임 동안 유지할지 (약 1초)
        self.declare_parameter('require_band', False)         # True 면 띠가 보이는 빈 그릇만 카운트

        try:
            import lap  # noqa: F401  (YOLO 추적기 ByteTrack 이 필요로 함)
        except ImportError:
            self.get_logger().error('추적에 필요한 패키지 없음 → pip install lap')
            raise

        weights = self.p('weights') or os.path.join(
            get_package_share_directory('plate_vision'), 'models', 'best.pt')
        self.model = YOLO(weights)
        names = set(self.model.names.values())
        if not {'empty_plate', 'food_plate'} <= names:
            self.get_logger().warning(f'모델 클래스가 {sorted(names)} → empty_plate/food_plate 로 학습한 모델이 필요')

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

        # 상태 (구독 콜백보다 먼저 초기화)
        self.target = self.p('target_count')
        self.count = 0
        self.full = False
        self.frame_no = 0
        self.tracks = {}        # 추적 ID → {'empty_frames': n, 'counted': bool}
        self.recent = []        # 최근에 센 그릇 [(cx, cy, 마지막으로 본 프레임)] → 추적이 끊겨도 다시 안 셈
        self.frame = None

        self.det_pub = self.create_publisher(String, '/plate/detections', 10)
        self.band_pub = self.create_publisher(String, '/band/detections', 10)
        self.target_pub = self.create_publisher(PointStamped, '/plate/empty_target', 10)
        self.count_pub = self.create_publisher(Int32, '/tray/empty_count', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ready_pub = self.create_publisher(Bool, '/tray/ready', latched)
        self.create_subscription(Empty, '/tray/reset', self.reset_callback, 10)

        self.ready_pub.publish(Bool(data=False))
        self.timer = self.create_timer(0.03, self.timer_callback)
        self.get_logger().info(f'Empty count START  camera=/dev/video{cam}  target={self.target}')

    def p(self, name):
        return self.get_parameter(name).value

    # ---------------- 카운트 ----------------
    def reset_callback(self, msg):
        # 주행로봇이 쌓인 빈 그릇을 가져간 뒤 호출 → 처음부터 다시 셈
        self.count = 0
        self.full = False
        self.recent = []
        for t in self.tracks.values():          # 지금 보이는 그릇은 이미 센 것으로 두어 다시 세지 않음
            t['empty_frames'] = 0
        self.count_pub.publish(Int32(data=0))
        self.ready_pub.publish(Bool(data=False))
        self.get_logger().info('RESET → 빈 그릇 카운트 0')

    def already_counted_here(self, cx, cy):
        d = self.p('same_dist')
        return any((cx - x) ** 2 + (cy - y) ** 2 < d * d for x, y, _ in self.recent)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('카메라 frame을 읽지 못했습니다.')
            return
        self.frame = frame
        self.frame_no += 1

        # YOLO 추적 (persist=True: 프레임 사이에 같은 그릇은 같은 ID 유지)
        r = self.model.track(frame, persist=True, tracker='bytetrack.yaml',
                             conf=self.p('conf'), verbose=False)[0]

        plates, bands = [], []
        if r.boxes.id is not None:
            for b, c, k, tid in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                    r.boxes.cls.tolist(), r.boxes.id.int().tolist()):
                name = r.names[int(k)]
                item = dict(conf=round(c, 3), box=[round(v, 1) for v in b],
                            cx=round((b[0] + b[2]) / 2, 1), cy=round((b[1] + b[3]) / 2, 1))
                if name == 'band':
                    bands.append(dict(item, plate_id=None))
                elif name in ('empty_plate', 'food_plate'):
                    plates.append(dict(item, id=tid, state='empty' if name == 'empty_plate' else 'food',
                                       band=False, band_box=None))

        # 띠 → 그릇 배정: 띠 중심이 그릇 박스 안에 있고, 그릇 중심과 가장 가까운 그릇 (그릇마다 신뢰도 높은 띠 하나)
        for bd in sorted(bands, key=lambda x: -x['conf']):
            inside = [pl for pl in plates if pl['box'][0] <= bd['cx'] <= pl['box'][2]
                      and pl['box'][1] <= bd['cy'] <= pl['box'][3] and not pl['band']]
            if inside:
                pl = min(inside, key=lambda q: (q['cx'] - bd['cx']) ** 2 + (q['cy'] - bd['cy']) ** 2)
                pl['band'], pl['band_box'], bd['plate_id'] = True, bd['box'], pl['id']

        # 오래 안 보인 '최근에 센 그릇' 기록은 지움
        self.recent = [(x, y, f) for x, y, f in self.recent if self.frame_no - f <= self.p('lost_frames')]

        for pl in plates:
            t = self.tracks.setdefault(pl['id'], {'empty_frames': 0, 'counted': False})
            if not t['counted'] and self.already_counted_here(pl['cx'], pl['cy']):
                t['counted'] = True                     # 추적이 끊겨 새 ID 가 붙은 같은 그릇
            t['empty_frames'] = t['empty_frames'] + 1 if pl['state'] == 'empty' else 0

            band_ok = pl['band'] or not self.p('require_band')
            if pl['state'] == 'empty' and band_ok and not t['counted'] and not self.full \
                    and t['empty_frames'] >= self.p('stable_frames'):
                t['counted'] = True
                self.count += 1
                self.count_pub.publish(Int32(data=self.count))
                self.get_logger().info(f'빈 그릇 카운트 {self.count}/{self.target}  (ID {pl["id"]})')
                if self.count >= self.target:
                    # 같은 프레임에 그릇 2개가 동시에 확정돼도 target 을 넘기지 않도록 바로 멈춤
                    self.full = True
                    self.ready_pub.publish(Bool(data=True))
                    self.get_logger().info(f'==== 빈 그릇 {self.count}개 → 주행로봇 호출 (/tray/ready = True) ====')
            if t['counted']:
                self.recent = [(x, y, f) for x, y, f in self.recent
                               if (x - pl['cx']) ** 2 + (y - pl['cy']) ** 2 >= self.p('same_dist') ** 2]
                self.recent.append((pl['cx'], pl['cy'], self.frame_no))
            pl['counted'] = t['counted']

        self.det_pub.publish(String(data=json.dumps(plates)))
        self.band_pub.publish(String(data=json.dumps(bands)))

        # 로봇팔 목표: 가득 차지 않았을 때, 안정적으로 보이는 빈 그릇 중 신뢰도 가장 높은 것
        stable = [pl for pl in plates if pl['state'] == 'empty'
                  and self.tracks[pl['id']]['empty_frames'] >= self.p('stable_frames')]
        if stable and not self.full:
            tgt = max(stable, key=lambda pl: pl['conf'])
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'top_camera'
            msg.point.x, msg.point.y = float(tgt['cx']), float(tgt['cy'])
            self.target_pub.publish(msg)

        if self.p('show_window'):
            self.show(frame, plates, bands)

    # ---------------- 화면 ----------------
    def show(self, frame, plates, bands):
        view = frame.copy()
        for bd in bands:
            x1, y1, x2, y2 = [int(v) for v in bd['box']]
            cv2.rectangle(view, (x1, y1), (x2, y2), BAND_COLOR, 2)
            cv2.putText(view, f"band {bd['conf']:.2f}", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BAND_COLOR, 2)
        for pl in plates:
            x1, y1, x2, y2 = [int(v) for v in pl['box']]
            color = EMPTY_COLOR if pl['state'] == 'empty' else FOOD_COLOR
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 3)
            label = f"ID{pl['id']} {pl['state'].upper()} {pl['conf']:.2f}  band {'O' if pl['band'] else 'X'}" +                     ('  COUNTED' if pl['counted'] else '')
            cv2.putText(view, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(view, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        COUNTED_COLOR if pl['counted'] else color, 2)

        status = f'EMPTY COUNT {self.count} / {self.target}' + ('   -> ROBOT CALLED' if self.full else '')
        cv2.putText(view, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(view, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if self.full else (255, 255, 255), 2)
        cv2.putText(view, '[r] reset  [q] quit', (15, view.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Empty Plate Count', view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.reset_callback(None)
        elif key == ord('q') and rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EmptyCountNode()
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
