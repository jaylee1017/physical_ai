"""
학습한 모델 + 음식 유무 규칙을 사진 폴더로 평가

파일 이름이 food_ / empty_ 로 시작하면 정답으로 사용 (capture_dataset.py 로 찍은 사진).
  - 접시를 못 찾은 사진, empty/food 를 틀린 사진을 출력
  - 빈 접시 / 음식 접시의 food_ratio 분포를 보여 주고, 둘 사이의 기준값(food_thresh)을 추천
  - 결과 그림을 eval_out/ 에 저장 (박스 + 판정 + 음식 픽셀 보라색)

사용 예 (plate_vision 폴더에서):
    python tools/eval_state.py --weights models/best.pt --images C:/Physical_AI/plate_data/images
"""
import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plate_vision.plate_state import food_ratio  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--food-thresh', type=float, default=0.03)
    ap.add_argument('--sat-min', type=int, default=90)
    ap.add_argument('--white-min', type=int, default=256, help='흰 접시면 256(끔)')
    ap.add_argument('--out', default='eval_out')
    args = ap.parse_args()

    model = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(args.images, '*'))
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    ratios = {'food': [], 'empty': []}
    missed, wrong, extra = [], [], []
    for f in files:
        name = os.path.basename(f)
        truth = name.split('_')[0]
        img = cv2.imread(f)
        r = model(img, conf=args.conf, verbose=False)[0]
        boxes = r.boxes.xyxy.tolist()

        if truth == 'background':
            if boxes:
                extra.append(name)          # 접시 없는데 접시라고 함
            continue
        if not boxes:
            missed.append(name)
            continue

        view = img.copy()
        for box in boxes:
            ratio, mask = food_ratio(img, box, 0.85, args.sat_min, 60, white_min=args.white_min)
            state = 'food' if ratio >= args.food_thresh else 'empty'
            x1, y1, x2, y2 = [int(v) for v in box]
            if mask is not None:
                view[y1:y1 + mask.shape[0], x1:x1 + mask.shape[1]][mask > 0] = (255, 0, 255)
            color = (0, 140, 255) if state == 'food' else (0, 255, 0)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 3)
            cv2.putText(view, f'{state} {ratio * 100:.1f}%', (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if truth in ratios and len(boxes) == 1:   # 접시 1개짜리 사진만 정답 비교
                ratios[truth].append(ratio)
                if state != truth:
                    wrong.append(f'{name} → {state} ({ratio * 100:.1f}%)')
        cv2.imwrite(os.path.join(args.out, name), view)

    n = len([f for f in files if not os.path.basename(f).startswith('background')])
    print(f'\n사진 {len(files)}장 (접시 사진 {n}장)')
    print(f'접시 못 찾음: {len(missed)}장', missed[:10])
    print(f'배경인데 접시라고 함: {len(extra)}장', extra[:10])
    print(f'empty/food 틀림: {len(wrong)}장')
    for w in wrong[:20]:
        print('   ', w)
    for k in ('empty', 'food'):
        v = ratios[k]
        if v:
            print(f'{k:5s} food_ratio: 최소 {min(v) * 100:.1f}%  최대 {max(v) * 100:.1f}%  ({len(v)}장)')
    if ratios['empty'] and ratios['food']:
        lo, hi = max(ratios['empty']), min(ratios['food'])
        if lo < hi:
            print(f'→ 추천 food_thresh: {(lo + hi) / 2:.3f}  (빈 접시 최대 {lo * 100:.1f}% ~ 음식 접시 최소 {hi * 100:.1f}% 사이)')
        else:
            print('→ 빈 접시와 음식 접시 비율이 겹침: eval_out/ 그림에서 보라색(음식으로 본 픽셀)을 확인하세요')
    print(f'결과 그림: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
