"""
회전 원판 위 접시 사진 수집기 (YOLO 학습용)

라벨은 Roboflow 에서 'plate' 한 클래스로 박스만 그리면 됨 (빈 접시 / 음식 접시 모두 plate).
음식 유무는 YOLO 가 아니라 색 규칙(plate_state.py)이 판단하므로,
파일 이름 앞에 food_ / empty_ / background_ 를 붙여서 나중에 규칙 정확도 평가(eval_state.py)에 씀.

  [q]       촬영/저장
  [1][2][3] 모드 전환: 1=food(음식 있는 접시) 2=empty(빈 접시) 3=background(접시 없음)
  [ESC]     종료

사용 예 (C:\\Physical_AI 에서):
    python ros2_ws/src/plate_vision/tools/capture_dataset.py --cam 1
    → plate_data/images/food_20261031_143012_001.jpg ...
"""
import argparse
import os
import sys
from datetime import datetime

import cv2

MODES = {ord('1'): 'food', ord('2'): 'empty', ord('3'): 'background'}
COLORS = {'food': (0, 140, 255), 'empty': (0, 255, 0), 'background': (200, 200, 200)}


def open_camera(index, width, height):
    backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    # MJPG(압축)여야 1280x720 에서 USB 대역폭이 충분함. 백엔드마다 순서가 달라 앞뒤로 두 번 지정
    mjpg = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    for _ in range(10):  # 노출 조정 중인 초기 프레임 버림
        cap.read()
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=1)
    ap.add_argument('--out', default='plate_data/images')
    ap.add_argument('--mode', default='food', choices=list(COLORS))
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = open_camera(args.cam, args.width, args.height)
    if cap is None:
        print(f'[오류] 카메라 {args.cam}번을 열 수 없습니다.')
        return

    mode = args.mode
    files = os.listdir(args.out)
    counts = {m: sum(f.startswith(m + '_') for f in files) for m in COLORS}
    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    shot, flash = 0, 0
    print(f'저장 위치: {os.path.abspath(args.out)}')
    print('[q] 촬영  [1] food  [2] empty  [3] background  [ESC] 종료')

    while True:
        ok, frame = cap.read()
        if not ok:
            print('[오류] 프레임을 읽지 못했습니다.')
            break

        view = frame.copy()
        h, w = view.shape[:2]
        text = (f'MODE: {mode}   food {counts["food"]} | empty {counts["empty"]} | '
                f'bg {counts["background"]}   [q] shot [1/2/3] mode [ESC] quit')
        cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[mode], 2)
        if flash > 0:
            cv2.rectangle(view, (0, 0), (w - 1, h - 1), (255, 255, 255), 14)
            flash -= 1
        cv2.imshow('Capture', view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            shot += 1
            name = f'{mode}_{session}_{shot:03d}.jpg'
            cv2.imwrite(os.path.join(args.out, name), frame)  # 글자 없는 원본 저장
            counts[mode] += 1
            flash = 3
            print(f'[저장] {name}')
        elif key in MODES:
            mode = MODES[key]
            print(f'[모드] {mode}')
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'완료: food {counts["food"]} / empty {counts["empty"]} / background {counts["background"]}')


if __name__ == '__main__':
    main()
