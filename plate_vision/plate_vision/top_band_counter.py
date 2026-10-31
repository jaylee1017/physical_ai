"""
위에서 내려다보는 카메라로 트레이 위 파란 띠 세기 (ROS 와 무관한 순수 알고리즘)

트레이 영역(ROI) 안에서만 파란색을 찾는다 (회전 원판 위 접시의 띠는 세면 안 됨).

mode = 'blobs'  : 접시를 트레이에 '나란히' 놓을 때. 위에서 띠가 동시에 다 보임
                  → 파란 덩어리 개수 = 접시 수. 같은 개수가 stable_frames 유지되면 확정
mode = 'events' : 접시를 '겹쳐 쌓을' 때. 위에서는 맨 위 띠 하나만 보임
                  → 파란색이 reset_frames 이상 사라졌다가(로봇팔이 가림) 다시 보이면 +1
                    (팀원 원본 YellowCountNode 의 세는 방식)
"""
import cv2
import numpy as np


class TopBandCounter:

    def __init__(self,
                 roi=None,                     # 트레이 영역 (x1, y1, x2, y2). None 이면 화면 전체
                 lower_hsv=(100, 120, 60),     # 파란색
                 upper_hsv=(130, 255, 255),
                 mode='blobs',
                 min_area_ratio=0.003,         # ROI 넓이 대비 이보다 작은 파란 덩어리는 노이즈
                 stable_frames=15,             # blobs: 같은 개수가 이만큼 유지돼야 확정 (약 0.5초)
                 reset_frames=10):             # events: 이만큼 안 보여야 '사라짐' (약 0.3초)
        self.roi = roi
        self.lower = np.array(lower_hsv)
        self.upper = np.array(upper_hsv)
        self.mode = mode
        self.min_area_ratio = min_area_ratio
        self.stable_frames = stable_frames
        self.reset_frames = reset_frames
        self.reset()

    def reset(self):
        self.count = 0
        # blobs
        self._last, self._same = 0, 0
        # events
        self._active, self._absent = False, 0

    # ------------------------------------------------------------------
    def find_blobs(self, frame):
        """ROI 안 파란 덩어리 목록 [(x1, y1, x2, y2), ...] (전체 화면 좌표) 과 ROI 마스크"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (0, 0, w, h) if self.roi is None else [int(v) for v in self.roi]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return [], np.zeros((1, 1), np.uint8)

        mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # 점 노이즈 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # 띠 안의 작은 끊김 메움

        min_area = self.min_area_ratio * roi.shape[0] * roi.shape[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                blobs.append((bx + x1, by + y1, bx + bw + x1, by + bh + y1))
        return blobs, mask

    def update(self, frame):
        """한 프레임 처리. 반환: dict(count=확정 개수, now=이번 프레임 덩어리 수, blobs, mask, mode)"""
        blobs, mask = self.find_blobs(frame)
        now = len(blobs)

        if self.mode == 'blobs':
            if now == self._last:
                self._same += 1
            else:
                self._last, self._same = now, 1
            if self._same >= self.stable_frames:
                self.count = now
        else:  # events
            if now > 0:
                self._absent = 0
                if not self._active:          # 새로 나타남 → +1
                    self.count += 1
                    self._active = True
            else:
                self._absent += 1
                if self._absent >= self.reset_frames:
                    self._active = False      # 충분히 사라짐 → 다음 등장 때 셀 준비

        return dict(count=self.count, now=now, blobs=blobs, mask=mask, mode=self.mode)

    def toggle_mode(self):
        self.mode = 'events' if self.mode == 'blobs' else 'blobs'
        self.reset()
        return self.mode
