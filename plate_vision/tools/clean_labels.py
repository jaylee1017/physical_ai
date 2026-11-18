"""
Roboflow 데이터셋의 중복 라벨 정리 (자동 라벨링을 두 번 돌려서 같은 그릇에 박스가 겹친 경우)

  - 같은 클래스 박스가 많이 겹침(IoU > 기준)  → 하나만 남김
  - 같은 자리에 empty_plate / food_plate 가 둘 다 있음(정답이 모순) → 그 사진은 제외
원본 폴더는 건드리지 않고 '<원본>_clean' 폴더에 정리본을 만든다. data.yaml 도 새 경로로 고쳐서 저장.

사용 예:
    python ros2_ws/src/plate_vision/tools/clean_labels.py --src plate_data/real_v2
    → plate_data/real_v2_clean/data.yaml 로 학습
"""
import argparse
import os
import shutil

import yaml

CONFLICT_GROUP = {'empty_plate', 'food_plate'}   # 서로 겹치면 모순인 클래스들


def iou(a, b):
    """YOLO 형식 (cx, cy, w, h) 두 박스의 IoU"""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def clean_file(rows, names, thr):
    """rows: [(cls, (cx, cy, w, h)), ...] → (정리된 rows, 지운 개수, 모순 여부)"""
    kept, removed, conflict = [], 0, False
    for cls, box in rows:
        dup = False
        for kcls, kbox in kept:
            if iou(box, kbox) > thr:
                if kcls == cls:
                    dup = True                                   # 같은 클래스 중복 → 버림
                    break
                if {names[cls], names[kcls]} <= CONFLICT_GROUP:
                    conflict = True                              # 빈 그릇이면서 음식 그릇 → 모순
        if dup:
            removed += 1
        else:
            kept.append((cls, box))
    return kept, removed, conflict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Roboflow 에서 받은 데이터셋 폴더 (data.yaml 이 있는 곳)')
    ap.add_argument('--dst', help='정리본 폴더 (기본: <src>_clean)')
    ap.add_argument('--iou', type=float, default=0.7, help='이보다 많이 겹치면 같은 박스로 봄')
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst or src.rstrip('\\/') + '_clean')
    with open(os.path.join(src, 'data.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    names = cfg['names'] if isinstance(cfg['names'], list) else list(cfg['names'].values())

    if os.path.exists(dst):
        shutil.rmtree(dst)
    total_removed, dropped = 0, []
    for split in ('train', 'valid', 'test'):
        img_dir, lbl_dir = os.path.join(src, split, 'images'), os.path.join(src, split, 'labels')
        if not os.path.isdir(img_dir):
            continue
        os.makedirs(os.path.join(dst, split, 'images'))
        os.makedirs(os.path.join(dst, split, 'labels'))
        before = after = 0
        for img in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(img)[0]
            lbl = os.path.join(lbl_dir, stem + '.txt')
            rows = []
            if os.path.exists(lbl):
                with open(lbl, encoding='utf-8') as f:
                    for line in f:
                        p = line.split()
                        if len(p) >= 5:
                            rows.append((int(p[0]), tuple(map(float, p[1:5]))))
            kept, removed, conflict = clean_file(rows, names, args.iou)
            if conflict:
                dropped.append(f'{split}/{img}')
                continue
            before += len(rows)
            after += len(kept)
            total_removed += removed
            shutil.copy2(os.path.join(img_dir, img), os.path.join(dst, split, 'images', img))
            with open(os.path.join(dst, split, 'labels', stem + '.txt'), 'w', encoding='utf-8') as f:
                f.writelines(f'{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n' for c, b in kept)

        counts = [0] * len(names)
        for fn in os.listdir(os.path.join(dst, split, 'labels')):
            with open(os.path.join(dst, split, 'labels', fn), encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        counts[int(line.split()[0])] += 1
        n_img = len(os.listdir(os.path.join(dst, split, 'images')))
        print(f'{split:5s}: 이미지 {n_img}장, 박스 {before} → {after}  |  ' +
              '  '.join(f'{n} {c}' for n, c in zip(names, counts)))

    cfg.update({'path': dst.replace('\\', '/'), 'train': 'train/images', 'val': 'valid/images', 'test': 'test/images'})
    with open(os.path.join(dst, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f'\n중복 박스 제거: {total_removed}개')
    print(f'모순 라벨(빈 그릇+음식 그릇)로 제외한 사진: {len(dropped)}장', dropped)
    print(f'정리본: {dst}\\data.yaml')


if __name__ == '__main__':
    main()
