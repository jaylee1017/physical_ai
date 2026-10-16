"""
접시 음식 유무 판정 (YOLO 박스 안쪽의 음식 픽셀 비율)

음식 픽셀 = 아래 둘 중 하나
  1) 색 있는 픽셀: 채도가 높음 (빨간 초밥 등). 검은/회색 접시와 배경은 채도가 낮음
  2) 하얀 덩어리: 아주 밝고 채도 낮은 픽셀이 '덩어리'로 모인 것
     (노란 계란초밥은 카메라 노출 때문에 거의 흰색으로 찍힘)
     접시 반사광도 밝지만 얇은 줄무늬라서, 접시 크기에 비례한 커널로 opening 하면 사라짐
"""
import cv2
import numpy as np


def food_ratio(frame, box, shrink, sat_min, val_min,
               white_min=240, white_sat_max=60, white_kernel=0.1):
    """박스 안쪽 타원(접시 면) 중 음식 픽셀 비율과 마스크. white_min > 255 이면 흰 덩어리 검사 끔"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, None

    ch, cw = crop.shape[:2]
    ellipse = np.zeros((ch, cw), np.uint8)
    # 테두리(색 띠 포함)와 박스 모서리 배경이 섞이지 않게 접시 안쪽만 사용
    cv2.ellipse(ellipse, (cw // 2, ch // 2),
                (int(cw / 2 * shrink), int(ch / 2 * shrink)), 0, 0, 360, 255, -1)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]

    # 1) 색 있는 음식
    colored = ((s >= sat_min) & (v >= val_min)).astype(np.uint8) * 255
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 2) 하얀 덩어리 음식 (반사광 줄무늬는 opening 으로 제거)
    if white_min <= 255:
        white = ((v >= white_min) & (s <= white_sat_max)).astype(np.uint8) * 255
        ks = max(3, int(min(cw, ch) * white_kernel)) | 1
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        colored = cv2.bitwise_or(colored, white)

    colored = cv2.bitwise_and(colored, ellipse)

    area = cv2.countNonZero(ellipse)
    return (cv2.countNonZero(colored) / area if area else 0.0), colored
