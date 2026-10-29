"""
쌓인 접시의 파란 띠 개수 세기 (ROS 와 무관한 순수 알고리즘)

카메라 B 가 트레이 옆에서 쌓인 접시 옆면을 보고 있다고 가정.
한 프레임마다 '지금 화면에 보이는 띠 줄 수'를 새로 센다 (나타났다 사라지는 횟수를 세지 않음).

  1) ROI        : 트레이 부분만 잘라냄
  2) HSV 마스크  : 파란색 픽셀만 흰색
  3) 노이즈 정리 : 작은 점 제거 + '가로로만' 끊김 연결 (위아래 띠가 합쳐지지 않게)
  4) 행 투영     : 각 행(y)의 파란 픽셀 수를 셈 → 기준 이상이면 '띠 행'
                   기준 = max(가장 넓은 행 x rel_fill, ROI 폭 x min_width_ratio)
                   → ROI 를 딱 맞게 안 잡아도 되고, 작은 파란 물체는 최소 폭에서 걸러짐
  5) 구간 세기   : 연속된 '띠 행' 구간 하나 = 띠 하나 (너무 얇은 구간은 노이즈로 버림)
                   band_height 를 주면, 붙어서 두꺼워진 구간은 높이 / band_height 개로 계산
  6) 안정화      : 같은 개수가 stable_frames 동안 유지돼야 확정 (팔이 잠깐 가려도 안 흔들림)
"""
import cv2
import numpy as np


class BandCounter:

    def __init__(self,
                 lower_hsv=(100, 120, 60),     # 파란색 H 100~130, 진한 색(S>=120)만
                 upper_hsv=(130, 255, 255),
                 roi=None,                     # (x1, y1, x2, y2), None 이면 화면 전체
                 close_width=25,               # 가로 끊김 연결 커널 폭(px)
                 rel_fill=0.5,                 # 가장 넓은 행 대비 이 비율 이상이면 '띠 행'
                 min_width_ratio=0.15,         # 그래도 ROI 폭의 이 비율보다는 넓어야 함 (작은 물체 제외)
                 min_band_rows=3,              # 이보다 얇은 구간은 노이즈
                 band_height=0,                # 띠 1개 높이(px). 0 이면 끔, 주면 붙은 띠를 높이로 나눔
                 stable_frames=15):            # 약 0.5초 (30fps 기준)
        self.lower = np.array(lower_hsv)
        self.upper = np.array(upper_hsv)
        self.roi = roi
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 1))
        self.rel_fill = rel_fill
        self.min_width_ratio = min_width_ratio
        self.min_band_rows = min_band_rows
        self.band_height = band_height
        self.stable_frames = stable_frames
        self.reset()

    def reset(self):
        self.stable_count = 0      # 확정된 띠 개수
        self._last = 0             # 직전 프레임 개수
        self._same = 0             # 같은 개수가 이어진 프레임 수

    # ------------------------------------------------------------------
    def crop(self, frame):
        if self.roi is None:
            return frame, (0, 0)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in self.roi]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        return frame[y1:y2, x1:x2], (x1, y1)

    def make_mask(self, roi_img):
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # 점 노이즈 제거 (작은 정사각형)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # 띠 끊김 연결: 가로 N x 세로 1 커널 → 좌우로만 메움, 위아래 띠 사이 간격은 그대로
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        return mask

    def find_bands(self, mask):
        """행 투영으로 띠 구간 찾기. 반환: [(y_top, y_bottom, x_left, x_right), ...] (ROI 좌표)"""
        row_sum = (mask > 0).sum(axis=1)                    # 각 행의 파란 픽셀 수
        threshold = max(self.rel_fill * row_sum.max(initial=0),
                        self.min_width_ratio * mask.shape[1], 1)
        is_band = row_sum >= threshold                      # '띠 행' 여부

        bands = []
        y, n = 0, len(is_band)
        while y < n:
            if not is_band[y]:
                y += 1
                continue
            top = y
            while y < n and is_band[y]:
                y += 1
            bottom = y - 1
            if bottom - top + 1 >= self.min_band_rows:      # 너무 얇은 구간은 노이즈
                cols = np.flatnonzero(mask[top:bottom + 1].any(axis=0))
                bands.append((top, bottom, int(cols[0]), int(cols[-1])))
        return bands, row_sum, threshold

    def update(self, frame):
        """
        한 프레임 처리.
        반환: dict(raw=이번 프레임 개수, stable=확정 개수, bands=띠 위치(전체 화면 좌표), mask, row_sum)
        """
        roi_img, (ox, oy) = self.crop(frame)
        mask = self.make_mask(roi_img)
        bands, row_sum, threshold = self.find_bands(mask)
        if self.band_height > 0:
            # 붙어서 하나로 보이는 구간: 높이 24px, 띠 높이 12px → 2개
            raw = sum(max(1, round((b - t + 1) / self.band_height)) for t, b, _, _ in bands)
        else:
            raw = len(bands)

        # 같은 개수가 stable_frames 동안 이어져야 확정
        if raw == self._last:
            self._same += 1
        else:
            self._last, self._same = raw, 1
        if self._same >= self.stable_frames:
            self.stable_count = raw

        bands_full = [(x1 + ox, top + oy, x2 + ox, bottom + oy) for top, bottom, x1, x2 in bands]
        return dict(raw=raw, stable=self.stable_count, bands=bands_full,
                    mask=mask, row_sum=row_sum, threshold=threshold, offset=(ox, oy))


def draw_debug(frame, result, counter, target_count, extra_text=''):
    """띠 위치·개수·ROI 를 원본 위에 그리고, 마스크 옆에 행 투영 그래프를 붙인 디버그 화면을 반환"""
    view = frame.copy()
    if counter.roi is not None:
        x1, y1, x2, y2 = [int(v) for v in counter.roi]
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 1)
    for i, (x1, y1, x2, y2) in enumerate(result['bands'], 1):
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 165, 255), 2)   # 주황 (파란 띠 위에서 잘 보이게)
        cv2.putText(view, str(i), (x2 + 6, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    full = result['stable'] >= target_count
    color = (0, 0, 255) if full else (255, 255, 255)
    cv2.putText(view, f"BAND {result['stable']} / {target_count}  (now {result['raw']})",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if extra_text:
        cv2.putText(view, extra_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if full:
        cv2.putText(view, 'FULL', (view.shape[1] // 2 - 80, view.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)

    # 마스크 + 행 투영 그래프 (기준선 = row_fill)
    mask = cv2.cvtColor(result['mask'], cv2.COLOR_GRAY2BGR)
    h, w = mask.shape[:2]
    graph = np.zeros((h, 150, 3), np.uint8)
    if w > 0:
        lengths = (result['row_sum'] / w * 150).astype(int)
        for y, L in enumerate(lengths):
            cv2.line(graph, (0, y), (int(L), y), (200, 200, 200), 1)
    thr = int(result['threshold'] / max(w, 1) * 150)
    cv2.line(graph, (thr, 0), (thr, h - 1), (0, 0, 255), 1)   # 이 선을 넘는 행 = '띠 행'
    return view, np.hstack([mask, graph])
