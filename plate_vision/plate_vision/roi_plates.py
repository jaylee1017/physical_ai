"""
ROI 안의 그릇(최대 2개)마다 빈 그릇/음식 그릇 + 띠 판정 (ROS 와 무관한 순수 로직)

  - 그릇 위치·빈 그릇/음식 그릇: YOLO 2클래스 (empty_plate, food_plate)
  - 띠: YOLO 가 아니라 색(HSV)으로 찾음
        그릇이 기울어져 있으면 YOLO 박스(축 정렬)가 서로 크게 겹쳐서, 박스로는 띠가 어느 그릇 것인지 구분이 안 됨
        → ROI 안의 파란/보라 덩어리를 찾아 '가장 가까운 그릇 중심'에 배정
        → 띠 실제 모양 그대로라 중심·각도도 정확함

  1) ROI 를 잘라 YOLO 실행 (박스 좌표는 전체 화면 기준으로 되돌림)
  2) ROI 안에 '완전히' 들어온 그릇 우선, 최대 2개 (가장자리에 걸친 그릇은 full=False → 판정 보류)
  3) 띠 덩어리마다 가장 가까운 그릇에 배정
"""
import cv2
import numpy as np

# 실제 띠 측정값(빈 그릇 157장): 파랑 H 103~113, 보라 H 120~127 → 117 로 구분
# 원판(짙은 남색)은 V ~32 라 V 하한으로 제외. 진짜 띠는 약 4700~5800px, 노이즈는 150~650px → 최소 1500px
BAND_LOWER = (100, 80, 60)
BAND_UPPER = (140, 255, 255)


def find_bands(frame, roi, lower=BAND_LOWER, upper=BAND_UPPER, min_area=1500):
    """ROI 안 파란/보라 덩어리 목록 [{center, angle, area, box, color}] (전체 화면 좌표)와 ROI 마스크"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (0, 0, w, h) if roi is None else [int(v) for v in roi]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    bands = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        hue = float(np.median(hsv[..., 0][cv2.drawContours(np.zeros_like(mask), [c], -1, 255, -1) > 0]))
        bx, by, bw, bh = cv2.boundingRect(c)
        bands.append({
            'center': (cx + x1, cy + y1),
            'angle': round(ang + 90 if rw < rh else ang, 1),          # 띠의 긴 방향
            'area': area,
            'box': (bx + x1, by + y1, bx + bw + x1, by + bh + y1),
            'color': 'blue' if hue < 117 else 'purple',
            'contour': c + np.array([[x1, y1]]),
        })
    return bands, mask


class RoiPlateJudge:

    def __init__(self, model, roi=None, conf=0.5, max_plates=2, edge_margin=5,
                 empty_cls='empty_plate', food_cls='food_plate', band_min_area=1500):
        self.model = model
        self.roi = roi                    # (x1, y1, x2, y2) 전체 화면 좌표, None 이면 화면 전체
        self.conf = conf
        self.max_plates = max_plates
        self.edge_margin = edge_margin    # ROI 경계에서 이 픽셀 안쪽이어야 '완전히 들어옴'
        self.cls = {empty_cls: 'empty', food_cls: 'food'}
        self.band_min_area = band_min_area

    def crop(self, frame):
        if self.roi is None:
            return frame, (0, 0)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in self.roi]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        return frame[y1:y2, x1:x2], (x1, y1)

    def judge(self, frame):
        roi_img, (ox, oy) = self.crop(frame)
        rh, rw = roi_img.shape[:2]
        r = self.model(roi_img, conf=self.conf, verbose=False)[0]

        plates = []
        m = self.edge_margin
        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            state = self.cls.get(r.names[int(k)])
            if state is None:
                continue
            full = b[0] >= m and b[1] >= m and b[2] <= rw - m and b[3] <= rh - m
            box = (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy)
            plates.append(dict(box=box, conf=c, state=state, full=full))

        # 같은 그릇을 empty/food 둘 다로 잡은 경우: 많이 겹치면 신뢰도 높은 쪽만 남김
        plates.sort(key=lambda p: -p['conf'])
        kept = []
        for p in plates:
            if all(_iou(p['box'], q['box']) < 0.6 for q in kept):
                kept.append(p)
        # 완전히 들어온 그릇 우선, 최대 2개, 왼쪽부터 번호
        kept.sort(key=lambda p: (not p['full'], -p['conf']))
        kept = sorted(kept[:self.max_plates], key=lambda p: p['box'][0])

        # 띠: 색으로 찾아서 가장 가까운 그릇 중심에 배정 (그릇마다 가장 큰 띠 하나)
        bands, band_mask = find_bands(frame, self.roi, min_area=self.band_min_area)
        for p in kept:
            p['cx'], p['cy'] = (p['box'][0] + p['box'][2]) / 2, (p['box'][1] + p['box'][3]) / 2
            p['band'] = None
        for b in bands:
            near = min(kept, key=lambda p: (p['cx'] - b['center'][0]) ** 2 + (p['cy'] - b['center'][1]) ** 2,
                       default=None)
            if near is not None and _inside(b['center'], near['box']) and \
                    (near['band'] is None or b['area'] > near['band']['area']):
                near['band'] = b

        result = []
        for i, p in enumerate(kept, 1):
            b = p['band']
            result.append({
                'id': i,
                'full': p['full'],                                  # False 면 ROI 경계에 걸침 → 판정 보류
                'state': p['state'],                                # 'empty' / 'food'
                'band': b is not None,
                'band_color': None if b is None else b['color'],    # 'blue' / 'purple'
                'band_center': None if b is None else [round(b['center'][0], 1), round(b['center'][1], 1)],
                'band_angle': None if b is None else b['angle'],
                'cx': round(p['cx'], 1),
                'cy': round(p['cy'], 1),
                'conf': round(p['conf'], 3),
                'box': [round(v, 1) for v in p['box']],
                '_band_contour': None if b is None else b['contour'],
            })
        return result, band_mask


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def draw(frame, plates, roi):
    view = frame.copy()
    if roi is not None:
        cv2.rectangle(view, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (0, 255, 255), 2)
        cv2.putText(view, 'ROI', (int(roi[0]) + 5, int(roi[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    for p in plates:
        x1, y1, x2, y2 = [int(v) for v in p['box']]
        if not p['full']:
            color = (160, 160, 160)                      # 경계에 걸친 그릇: 회색
        else:
            color = (0, 255, 0) if p['state'] == 'empty' else (0, 140, 255)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 3)
        if p['_band_contour'] is not None:
            cv2.drawContours(view, [p['_band_contour']], -1, (255, 255, 0), 2)   # 띠: 하늘색 윤곽
            bx, by = [int(v) for v in p['band_center']]
            cv2.line(view, (bx, by), (int(p['cx']), int(p['cy'])), (255, 255, 0), 1)
        label = f"#{p['id']} {p['state'].upper()}  band " + \
                (f"{p['band_color']} {p['band_angle']}deg" if p['band'] else 'X') + ('' if p['full'] else '  (edge)')
        cv2.putText(view, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(view, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return view
