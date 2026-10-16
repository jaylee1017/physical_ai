import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import Twist

# [추가] 이동형 로봇 호출 신호 / 새 트레이 초기화용 메시지
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Empty


# 팀원 YellowCountNode 원본에서 [변경]/[추가] 표시한 부분만 바뀜
class RedCountNode(Node):

    def __init__(self):
        super().__init__('red_count_node')

        # ---------------------------------------
        # /cmd_vel publisher
        # ---------------------------------------
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # ---------------------------------------
        # [추가] 트레이 준비 완료 신호 (END 시 True)
        # 이동형 로봇 / 접시 판별 노드가 늦게 켜져도 받도록 transient_local
        # ---------------------------------------
        self.ready_pub = self.create_publisher(
            Bool,
            '/tray/ready',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )
        self.ready_pub.publish(Bool(data=False))

        # ---------------------------------------
        # [추가] 새 트레이로 교체되면 count 초기화
        # ---------------------------------------
        self.create_subscription(
            Empty,
            '/tray/reset',
            self.reset_callback,
            10
        )

        # ---------------------------------------
        # Camera
        # /dev/video2 = camera index 2
        # [변경] launch 에서 바꿀 수 있게 파라미터로 (기본값은 원본과 같은 2)
        # ---------------------------------------
        self.declare_parameter('camera_index', 2)
        cam = self.get_parameter('camera_index').value

        self.cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error(f'video{cam} 카메라를 열 수 없습니다.')
            raise RuntimeError('Camera open failed')

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ---------------------------------------
        # [변경] Yellow -> Red HSV range
        # 빨간색은 H 값이 0 근처와 180 근처 양 끝에 있어서 범위 2개
        # ---------------------------------------
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

        # 너무 작은 빨간색은 무시
        self.MIN_AREA = 1500

        # 목표 count
        self.TARGET_COUNT = 3

        self.red_count = 0

        # 현재 빨간색을 이미 센 상태인지
        self.red_active = False

        # 빨간색이 사라졌다고 판단하기 위한 frame 수
        self.no_red_frames = 0
        self.RESET_FRAMES = 5

        # 종료 상태
        self.end_flag = False

        # 약 30Hz
        self.timer = self.create_timer(
            0.03,
            self.timer_callback
        )

        self.get_logger().info('Red detector START')
        self.get_logger().info(f'Camera = /dev/video{cam}')
        self.get_logger().info('Target Count = 3')


    # [추가] 이동형 로봇이 트레이를 가져가고 새 트레이가 놓이면 호출
    def reset_callback(self, msg):

        self.red_count = 0
        self.red_active = False
        self.no_red_frames = 0
        self.end_flag = False

        self.ready_pub.publish(Bool(data=False))

        self.get_logger().info('RESET → 새 트레이 count 시작')


    def timer_callback(self):

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning('카메라 frame을 읽지 못했습니다.')
            return

        # ---------------------------------------
        # 이미 END 상태
        # ---------------------------------------
        if self.end_flag:

            self.stop_robot()

            cv2.putText(
                frame,
                'END',
                (190, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 0, 255),
                8
            )

            cv2.putText(
                frame,
                f'COUNT : {self.red_count}',
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            cv2.imshow('Red Detection', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.stop_robot()
                rclpy.shutdown()

            # [추가] 창에서 r 키로도 초기화 (테스트용)
            elif key == ord('r'):
                self.reset_callback(None)

            return

        # ---------------------------------------
        # BGR -> HSV
        # ---------------------------------------
        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        # ---------------------------------------
        # [변경] Red mask (범위 2개 합치기)
        # ---------------------------------------
        mask1 = cv2.inRange(
            hsv,
            self.lower_red1,
            self.upper_red1
        )

        mask2 = cv2.inRange(
            hsv,
            self.lower_red2,
            self.upper_red2
        )

        mask = cv2.bitwise_or(mask1, mask2)

        # ---------------------------------------
        # Noise 제거
        # ---------------------------------------
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ---------------------------------------
        # contour 찾기
        # ---------------------------------------
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        red_detected = False

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > self.MIN_AREA:

                red_detected = True

                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    'RED',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        # ---------------------------------------
        # COUNT
        # ---------------------------------------
        if red_detected:

            self.no_red_frames = 0

            # 새 빨간색이 처음 들어왔을 때만 count
            if not self.red_active:

                self.red_count += 1
                self.red_active = True

                self.get_logger().info(
                    f'RED COUNT = {self.red_count}'
                )

        else:

            self.no_red_frames += 1

            # 빨간색이 일정 시간 사라지면
            # 다음 물체를 count할 준비
            if self.no_red_frames >= self.RESET_FRAMES:

                self.red_active = False

        # ---------------------------------------
        # COUNT 3 -> END
        # ---------------------------------------
        if self.red_count >= self.TARGET_COUNT:

            self.end_flag = True

            self.stop_robot()

            # [추가] 이동형 로봇에게 트레이 준비 완료 신호
            self.ready_pub.publish(Bool(data=True))

            self.get_logger().info('====================')
            self.get_logger().info('        END')
            self.get_logger().info('====================')

        # ---------------------------------------
        # 화면 표시
        # ---------------------------------------
        cv2.putText(
            frame,
            f'COUNT : {self.red_count}',
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        if self.end_flag:

            cv2.putText(
                frame,
                'END',
                (190, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 0, 255),
                8
            )

        cv2.imshow(
            'Red Detection',
            frame
        )

        cv2.imshow(
            'Red Mask',
            mask
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):

            self.stop_robot()

            if rclpy.ok():
                rclpy.shutdown()


    def stop_robot(self):

        msg = Twist()

        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        self.cmd_pub.publish(msg)


    def destroy_node(self):

        self.stop_robot()

        if self.cap.isOpened():
            self.cap.release()

        cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = RedCountNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.stop_robot()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
