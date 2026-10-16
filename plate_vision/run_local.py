"""
ROS2 없이 내 PC(Windows)에서 두 노드를 그대로 실행해 보는 테스트용 스크립트

plate_state_node.py / red_count_node.py 코드는 한 줄도 바꾸지 않고,
rclpy·메시지 타입만 가짜(stub)로 바꿔 끼워서 실행한다.
  - 노드끼리 주고받는 토픽(/tray/ready 등)은 실제처럼 전달됨
  - 발행되는 메시지는 터미널에 출력
  - Windows 에서는 CAP_V4L2 대신 DSHOW 로 카메라를 연다

사용 예 (yolo_course 환경, 이 폴더에서):
    python run_local.py --plate-cam 0 --band-cam 2     # 카메라 2대
    python run_local.py --only band --band-cam 2       # 빨간 띠 노드만
    python run_local.py --only plate --plate-cam 2     # 빈 접시 노드만

  Plate State 창: [m] 음식 마스크  [q] 종료
  Red Detection 창: [r] count 초기화(END 후)  [q] 종료
"""
import argparse
import os
import sys
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
            cap = real(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            return cap

        cv2.VideoCapture = video_capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plate-cam', type=int, default=0, help='회전 테이블 카메라 번호')
    ap.add_argument('--band-cam', type=int, default=2, help='트레이 띠 카메라 번호')
    ap.add_argument('--only', choices=['plate', 'band'], help='한 노드만 실행')
    args = ap.parse_args()

    _install_stubs()
    from plate_vision.plate_state_node import PlateStateNode
    from plate_vision.red_count_node import RedCountNode

    _params[('plate_state_node', 'camera_index')] = args.plate_cam
    _params[('red_count_node', 'camera_index')] = args.band_cam

    nodes = []
    if args.only != 'plate':
        nodes.append(RedCountNode())   # /tray/ready 를 먼저 발행하도록 먼저 생성
    if args.only != 'band':
        nodes.append(PlateStateNode())

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
