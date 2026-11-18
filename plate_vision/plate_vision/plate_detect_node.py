"""
카메라 A (회전 원판을 위에서): 빈 그릇 / 음식 그릇 구분

YOLO(models/best.pt)의 empty_plate, food_plate 결과만 사용 (band 는 카메라 B 담당이라 무시)

Publish
  /plate/detections  std_msgs/String  매 프레임 그릇 목록 JSON
                     [{"state": "empty"|"food", "conf": 0.97, "cx": 640.0, "cy": 360.0, "box": [x1, y1, x2, y2]}, ...]
  /plate/count       std_msgs/String  매 프레임 개수 JSON  {"empty": 1, "food": 1}

창 단축키: [q] 종료
"""
import json
import os

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

from plate_vision.camera import open_camera

COLORS = {'empty': (0, 255, 0), 'food': (0, 140, 255)}


class PlateDetectNode(Node):

    def __init__(self):
        super().__init__('plate_detect_node')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)
        self.declare_parameter('show_window', True)
        self.declare_parameter('weights', '')      # 비우면 패키지의 models/best.pt
        self.declare_parameter('conf', 0.5)        # YOLO 신뢰도 기준

        weights = self.p('weights') or os.path.join(
            get_package_share_directory('plate_vision'), 'models', 'best.pt')
        self.model = YOLO(weights)

        cam = self.p('camera_index')
        self.cap, fmt = open_camera(cam, self.p('frame_width'), self.p('frame_height'))
        if self.cap is None:
            self.get_logger().error(f'/dev/video{cam} 카메라를 열 수 없습니다.')
            raise RuntimeError('Camera open failed')

        self.det_pub = self.create_publisher(String, '/plate/detections', 10)
        self.count_pub = self.create_publisher(String, '/plate/count', 10)
        self.timer = self.create_timer(0.03, self.timer_callback)
        self.get_logger().info(f'[카메라 A] 빈 그릇/음식 그릇 구분 START  /dev/video{cam} {fmt}')

    def p(self, name):
        return self.get_parameter(name).value

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('카메라 frame을 읽지 못했습니다.')
            return

        r = self.model(frame, conf=self.p('conf'), verbose=False)[0]
        plates = []
        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            name = r.names[int(k)]
            if name not in ('empty_plate', 'food_plate'):
                continue
            plates.append({'state': 'empty' if name == 'empty_plate' else 'food', 'conf': round(c, 3),
                           'cx': round((b[0] + b[2]) / 2, 1), 'cy': round((b[1] + b[3]) / 2, 1),
                           'box': [round(v, 1) for v in b]})

        count = {'empty': sum(p['state'] == 'empty' for p in plates),
                 'food': sum(p['state'] == 'food' for p in plates)}
        self.det_pub.publish(String(data=json.dumps(plates)))
        self.count_pub.publish(String(data=json.dumps(count)))

        if self.p('show_window'):
            view = frame.copy()
            for p in plates:
                x1, y1, x2, y2 = [int(v) for v in p['box']]
                cv2.rectangle(view, (x1, y1), (x2, y2), COLORS[p['state']], 3)
                cv2.putText(view, f"{p['state'].upper()} {p['conf']:.2f}", (x1, max(y1 - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[p['state']], 2)
            text = f"EMPTY {count['empty']}   FOOD {count['food']}"
            cv2.putText(view, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(view, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow('Camera A - Plates', view)
            if (cv2.waitKey(1) & 0xFF) == ord('q') and rclpy.ok():
                rclpy.shutdown()

    def destroy_node(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PlateDetectNode()
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
