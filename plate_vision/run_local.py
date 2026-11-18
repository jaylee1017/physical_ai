"""
ROS2 없이 내 PC(Windows)에서 두 노드를 그대로 실행해 보는 테스트용 스크립트

plate_detect_node.py(카메라 A) / band_count_node.py(카메라 B) 코드는 그대로 쓰고,
rclpy·메시지 타입만 가짜(stub)로 바꿔 끼워서 실행한다.
  - 노드끼리 주고받는 토픽은 실제처럼 전달됨, 발행되는 메시지는 터미널에 출력
  - Windows 에서는 CAP_V4L2 대신 DSHOW 로 카메라를 연다

사용 예 (yolo_course 환경, 이 폴더에서):
    python run_local.py --plate-cam 1 --box-cam 2     # 카메라 2대
    python run_local.py --only plate --plate-cam 1    # 카메라 A 만 (빈 그릇/음식 그릇)
    python run_local.py --only box --box-cam 1        # 카메라 B 만 (파란 띠 카운트)

  카메라 없이 (사진 폴더 / 이미지 / 동영상을 카메라처럼 사용):
    python run_local.py --plate-src test_images --box-src test_images

  창 단축키: [r] 띠 카운트 초기화(카메라 B)  [q] 종료
"""
import argparse
import glob
import os
import sys
import time
import types

import cv2

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
        if self.topic in ('/plate/detections', '/band/detections', '/plate/count'):
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
    """

    def __init__(self, src, hold):
        self.hold, self.tick, self.video = hold, 0, None
        if os.path.isdir(src):
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
    ap.add_argument('--plate-cam', type=int, default=0, help='카메라 A 번호 (회전 원판, 빈 그릇/음식 그릇)')
    ap.add_argument('--box-cam', type=int, default=2, help='카메라 B 번호 (박스, 파란 띠 카운트)')
    ap.add_argument('--plate-src', help='카메라 A 대신: 이미지 폴더 / 이미지 / 동영상 경로')
    ap.add_argument('--box-src', help='카메라 B 대신: 이미지 폴더 / 이미지 / 동영상 경로')
    ap.add_argument('--hold', type=int, default=30, help='이미지 한 장을 보여 줄 프레임 수 (30 ≈ 1초)')
    ap.add_argument('--only', choices=['plate', 'box'], help='한 노드만 실행')
    args = ap.parse_args()

    try:
        import torch
        print(f'[device] {"GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    except ImportError:
        pass

    _install_stubs()
    plate_cam, box_cam = args.plate_cam, args.box_cam
    if args.plate_src:
        plate_cam = -1
        _fake_sources[plate_cam] = FakeCamera(args.plate_src, args.hold)
    if args.box_src:
        box_cam = -2
        _fake_sources[box_cam] = FakeCamera(args.box_src, args.hold)
    _install_fake_sources()

    from plate_vision.plate_detect_node import PlateDetectNode
    from plate_vision.band_count_node import BandCountNode
    _params[('plate_detect_node', 'camera_index')] = plate_cam
    _params[('band_count_node', 'camera_index')] = box_cam

    nodes = []
    if args.only != 'plate':
        nodes.append(BandCountNode())
    if args.only != 'box':
        nodes.append(PlateDetectNode())

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


if __name__ == '__main__':
    main()
