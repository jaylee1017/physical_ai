import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import Twist


class YellowCountNode(Node):

    def __init__(self):
        super().__init__('yellow_count_node')

        # ---------------------------------------
        # /cmd_vel publisher
        # ---------------------------------------
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # ---------------------------------------
        # Camera
        # /dev/video2 = camera index 2
        # ---------------------------------------
        self.cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error('video2 카메라를 열 수 없습니다.')
            raise RuntimeError('Camera open failed')

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ---------------------------------------
        # Yellow HSV range
        # ---------------------------------------
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])

        # 너무 작은 노란색은 무시
        self.MIN_AREA = 1500

        # 목표 count
        self.TARGET_COUNT = 3

        self.yellow_count = 0

        # 현재 노란색을 이미 센 상태인지
        self.yellow_active = False

        # 노란색이 사라졌다고 판단하기 위한 frame 수
        self.no_yellow_frames = 0
        self.RESET_FRAMES = 5

        # 종료 상태
        self.end_flag = False

        # 약 30Hz
        self.timer = self.create_timer(
            0.03,
            self.timer_callback
        )

        self.get_logger().info('Yellow detector START')
        self.get_logger().info('Camera = /dev/video2')
        self.get_logger().info('Target Count = 3')


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
                f'COUNT : {self.yellow_count}',
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            cv2.imshow('Yellow Detection', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.stop_robot()
                rclpy.shutdown()

            return

        # ---------------------------------------
        # BGR -> HSV
        # ---------------------------------------
        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        # ---------------------------------------
        # Yellow mask
        # ---------------------------------------
        mask = cv2.inRange(
            hsv,
            self.lower_yellow,
            self.upper_yellow
        )

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

        yellow_detected = False

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > self.MIN_AREA:

                yellow_detected = True

                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 255),
                    3
                )

                cv2.putText(
                    frame,
                    'YELLOW',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

        # ---------------------------------------
        # COUNT
        # ---------------------------------------
        if yellow_detected:

            self.no_yellow_frames = 0

            # 새 노란색이 처음 들어왔을 때만 count
            if not self.yellow_active:

                self.yellow_count += 1
                self.yellow_active = True

                self.get_logger().info(
                    f'YELLOW COUNT = {self.yellow_count}'
                )

        else:

            self.no_yellow_frames += 1

            # 노란색이 일정 시간 사라지면
            # 다음 물체를 count할 준비
            if self.no_yellow_frames >= self.RESET_FRAMES:

                self.yellow_active = False

        # ---------------------------------------
        # COUNT 3 -> END
        # ---------------------------------------
        if self.yellow_count >= self.TARGET_COUNT:

            self.end_flag = True

            self.stop_robot()

            self.get_logger().info('====================')
            self.get_logger().info('        END')
            self.get_logger().info('====================')

        # ---------------------------------------
        # 화면 표시
        # ---------------------------------------
        cv2.putText(
            frame,
            f'COUNT : {self.yellow_count}',
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
            'Yellow Detection',
            frame
        )

        cv2.imshow(
            'Yellow Mask',
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

    node = YellowCountNode()

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
