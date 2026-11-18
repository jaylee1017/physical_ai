"""카메라 열기 (두 노드 공용)"""
import cv2


def open_camera(index, width, height):
    """
    /dev/videoN 을 열고 (cap, 실제 포맷 문자열) 반환. 못 열면 (None, None)
    MJPG(압축)가 아니면 1280x720 에서 USB 대역폭을 넘어 화면이 가로줄로 깨짐.
    백엔드마다 적용 순서가 달라(V4L2: 포맷 먼저, Windows DSHOW: 해상도 먼저) 해상도 앞뒤로 두 번 지정.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, None
    mjpg = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('latin1')
    fmt = f'{fourcc} {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}'
    return cap, fmt
