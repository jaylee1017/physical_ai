"""
ROS2 없이 내 PC(Windows)에서 두 노드를 그대로 실행해 보는 테스트용 스크립트

plate_state_node.py / band_count_node.py 코드는 한 줄도 바꾸지 않고,
rclpy·메시지 타입만 가짜(stub)로 바꿔 끼워서 실행한다.
  - 노드끼리 주고받는 토픽(/tray/ready 등)은 실제처럼 전달됨
  - 발행되는 메시지는 터미널에 출력
  - Windows 에서는 CAP_V4L2 대신 DSHOW 로 카메라를 연다

사용 예 (yolo_course 환경, 이 폴더에서):
    python run_local.py --plate-cam 0 --band-cam 2     # 카메라 2대
    python run_local.py --only band --band-cam 2       # 빨간 띠 노드만
    python run_local.py --only plate --plate-cam 2     # 빈 접시 노드만

  카메라 1대(위에서) 통합: 빈 접시 판별 + 트레이 파란 띠 세기
    python run_local.py --top-cam 1
    python run_local.py --top-src 영상.mp4

  카메라 없이 (사진 폴더 / 동영상 / 'demo' 를 카메라처럼 사용):
    python run_local.py --plate-src test_images --band-src demo
    python run_local.py --only plate --plate-src 접시영상.mp4

  Plate State 창: [m] 음식 마스크  [q] 종료
  Band Count 창: [o] 마우스로 ROI 지정  [r] count 초기화  [q] 종료
"""
import argparse
import glob
import os
import sys
import time
import types

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_running = True
_params = {}
_subs = {}       # topic -> [callback]
_latched = {}    # topic -> 마지막 메시지 (transient_local 흉내)


# ------------------------------------------------------------------ 메시지 stub
class _Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Vec:
    x = y = z = 0.0


class Twist:
    def __init__(self):
        self.linear, self.angular = _Vec(), _Vec()


class PointStamped:
    def __init__(self):
        self.header = _Msg(stamp=None, frame_id='')
        self.point = _Vec()


def _msg_type(name):
    return type(name, (_Msg,), {})


Bool, Empty, String, Int32 = (_msg_type(n) for n in ('Bool', 'Empty', 'String', 'Int32'))


# ------------------------------------------------------------------ rclpy stub
class _Publisher:
    def __init__(self, topic):
        self.topic, self.last_print = topic, None

    def publish(self, msg):
        _latched[self.topic] = msg
        for cb in _subs.get(self.topic, []):
            cb(msg)
        self._print(msg)

    def _print(self, msg):
        if self.topic == '/plate/detections':
            return  # 매 프레임이라 출력 생략
        if self.topic == '/plate/empty_target':
            xy = (msg.point.x, msg.point.y)
            # 같은 접시 좌표가 매 프레임 찍히지 않게, 위치가 크게 바뀔 때만 출력
            if self.last_print and abs(xy[0] - self.last_print[0]) + abs(xy[1] - self.last_print[1]) < 30:
                return
            self.last_print = xy
            print(f'[/plate/empty_target] x={xy[0]:.0f}, y={xy[1]:.0f}')
            return
        if self.topic == '/cmd_vel':
            if self.last_print:
                return  # END 동안 매 프레임 발행되므로 첫 번째만 출력
            self.last_print = True
            print('[/cmd_vel] 정지 (all 0)')
            return
        print(f'[{self.topic}] {getattr(msg, "data", "")}')


class _Logger:
    def __init__(self, name):
        self.name = name

    def info(self, s):
        print(f'[{self.name}] {s}')

    warning = error = info


class _Param:
    def __init__(self, value):
        self.value = value


class Node:
    def __init__(self, name):
        self._name, self._timers = name, []

    def create_publisher(self, msg_type, topic, qos):
        return _Publisher(topic)

    def create_subscription(self, msg_type, topic, cb, qos):
        _subs.setdefault(topic, []).append(cb)
        if topic in _latched:
            cb(_latched[topic])

    def create_timer(self, period, cb):
        self._timers.append(cb)
        return cb

    def get_logger(self):
        return _Logger(self._name)

    def declare_parameter(self, name, default):
        _params.setdefault((self._name, name), default)

    def get_parameter(self, name):
        return _Param(_params[(self._name, name)])

    def get_clock(self):
        return _Msg(now=lambda: _Msg(to_msg=lambda: None))

    def destroy_node(self):
        pass


def _shutdown():
    global _running
    _running = False


def _install_stubs():
    mods = {
        'rclpy': dict(init=lambda args=None: None, ok=lambda: _running, shutdown=_shutdown),
        'rclpy.node': dict(Node=Node),
        'rclpy.qos': dict(QoSProfile=lambda **kw: None,
                          DurabilityPolicy=types.SimpleNamespace(TRANSIENT_LOCAL=1)),
        'geometry_msgs': {}, 'geometry_msgs.msg': dict(Twist=Twist, PointStamped=PointStamped),
        'std_msgs': {}, 'std_msgs.msg': dict(Bool=Bool, Empty=Empty, String=String, Int32=Int32),
        'ament_index_python': {},
        'ament_index_python.packages': dict(get_package_share_directory=lambda pkg: HERE),
    }
    for name, attrs in mods.items():
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m

    # Windows 에는 V4L2 가 없으므로 DSHOW 로 열기
    if sys.platform.startswith('win'):
        real = cv2.VideoCapture

        def video_capture(index, backend=cv2.CAP_ANY):
            return real(index, cv2.CAP_DSHOW)   # 포맷·해상도는 노드가 설정

        cv2.VideoCapture = video_capture


# ------------------------------------------------------------------ 카메라 없이 테스트
class FakeCamera:
    """
    카메라 대신 쓰는 입력
      - 이미지 폴더 / 이미지 파일: 한 장을 hold 프레임 동안 보여 주고 다음 장으로 (끝나면 처음부터)
      - 동영상 파일: 끝나면 처음부터 반복
      - 'demo': 파란 띠 접시가 0 → 1 → 2 → 3 개로 쌓이는 장면 반복 (띠 카운트 테스트용)
    """

    def __init__(self, src, hold):
        self.hold, self.tick, self.video = hold, 0, None
        if src == 'demo':
            self.frames = [_demo_stack(n) for n in (0, 1, 2, 3)]
        elif os.path.isdir(src):
            files = sorted(f for f in glob.glob(os.path.join(src, '*'))
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
            self.frames = [cv2.imread(f) for f in files]
        elif src.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            self.frames = [cv2.imread(src)]
        else:
            self.video, self.frames = _real_capture(src), None
        if self.frames is not None and not self.frames:
            raise SystemExit(f'[오류] {src} 에 이미지가 없습니다.')

    def isOpened(self):
        return self.frames is not None or self.video.isOpened()

    def set(self, *args):
        return True

    def get(self, prop):
        return 0

    def read(self):
        time.sleep(0.03)  # 실제 카메라(약 30fps)처럼 속도 맞춤
        if self.video is not None:
            ok, frame = self.video.read()
            if not ok:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.video.read()
            return ok, frame
        frame = self.frames[(self.tick // self.hold) % len(self.frames)]
        self.tick += 1
        return True, frame.copy()

    def release(self):
        if self.video is not None:
            self.video.release()


def _demo_stack(n):
    """옆에서 본 트레이 위 접시 n 개 (각 접시 옆면에 파란 띠)"""
    img = np.full((480, 640, 3), (170, 170, 170), np.uint8)
    cv2.rectangle(img, (160, 390), (480, 410), (60, 60, 60), -1)                 # 트레이
    for i in range(n):
        y = 385 - i * 30
        cv2.rectangle(img, (190, y - 24), (450, y), (40, 40, 40), -1)            # 접시 옆면
        cv2.rectangle(img, (200, y - 18), (440, y - 8), (200, 90, 20), -1)       # 파란 띠
    cv2.putText(img, f'demo: {n} plates', (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


_real_capture = cv2.VideoCapture
_fake_sources = {}   # 가짜 카메라 번호 -> FakeCamera


def _install_fake_sources():
    current = cv2.VideoCapture  # (Windows 면 DSHOW 로 바꾼 버전)

    def video_capture(index, backend=cv2.CAP_ANY):
        if index in _fake_sources:
            return _fake_sources[index]
        return current(index, backend)

    cv2.VideoCapture = video_capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plate-cam', type=int, default=0, help='회전 테이블 카메라 번호')
    ap.add_argument('--band-cam', type=int, default=2, help='트레이 띠 카메라 번호')
    ap.add_argument('--plate-src', help='카메라 대신: 이미지 폴더 / 이미지 / 동영상 경로')
    ap.add_argument('--band-src', help="카메라 대신: 이미지 폴더 / 이미지 / 동영상 경로, 또는 'demo'")
    ap.add_argument('--hold', type=int, default=30, help='이미지 한 장을 보여 줄 프레임 수 (30 ≈ 1초)')
    ap.add_argument('--only', choices=['plate', 'band'], help='한 노드만 실행')
    ap.add_argument('--top-cam', type=int, help='카메라 1대(위에서) 통합 노드로 실행: 카메라 번호')
    ap.add_argument('--top-src', help='카메라 1대 통합 노드를 카메라 대신 이미지 폴더 / 동영상으로')
    ap.add_argument('--count-cam', type=int, help='빈 그릇 카운트 노드(YOLO empty/food)로 실행: 카메라 번호')
    ap.add_argument('--count-src', help='빈 그릇 카운트 노드를 카메라 대신 이미지 폴더 / 동영상으로')
    args = ap.parse_args()

    try:
        import torch
        print(f'[device] {"GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    except ImportError:
        pass

    _install_stubs()
    plate_cam, band_cam = args.plate_cam, args.band_cam
    if args.plate_src:
        plate_cam = -1
        _fake_sources[plate_cam] = FakeCamera(args.plate_src, args.hold)
    if args.band_src:
        band_cam = -2
        _fake_sources[band_cam] = FakeCamera(args.band_src, args.hold)
    if args.top_src:
        _fake_sources[-3] = FakeCamera(args.top_src, args.hold)
    if args.count_src:
        _fake_sources[-4] = FakeCamera(args.count_src, args.hold)
    _install_fake_sources()

    if args.count_cam is not None or args.count_src:
        # 빈 그릇 카운트 → 3개면 주행로봇 호출 (YOLO empty_plate / food_plate)
        from plate_vision.empty_count_node import EmptyCountNode
        _params[('empty_count_node', 'camera_index')] = -4 if args.count_src else args.count_cam
        nodes = [EmptyCountNode()]
    elif args.top_cam is not None or args.top_src:
        # 위에서 보는 카메라 1대로 빈 접시 판별 + 트레이 띠 세기
        from plate_vision.top_vision_node import TopVisionNode
        _params[('top_vision_node', 'camera_index')] = -3 if args.top_src else args.top_cam
        nodes = [TopVisionNode()]
    else:
        nodes = _two_camera_nodes(args, plate_cam, band_cam)

    try:
        while _running:
            for node in nodes:
                for cb in node._timers:
                    cb()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.destroy_node()


def _two_camera_nodes(args, plate_cam, band_cam):
    from plate_vision.plate_state_node import PlateStateNode
    from plate_vision.band_count_node import BandCountNode

    _params[('plate_state_node', 'camera_index')] = plate_cam
    _params[('band_count_node', 'camera_index')] = band_cam

    nodes = []
    if args.only != 'plate':
        nodes.append(BandCountNode())  # /tray/ready 를 먼저 발행하도록 먼저 생성
    if args.only != 'band':
        nodes.append(PlateStateNode())
    return nodes


if __name__ == '__main__':
    main()
