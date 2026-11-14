"""
빈 그릇 / 음식 그릇 YOLO 학습 스크립트

  1) data.yaml 확인 (클래스, 이미지·라벨 개수, path 자동 수정)
  2) 학습 (GPU 있으면 자동 사용)
  3) test 세트로 클래스별 성능 출력
  4) best.pt 를 패키지 models/best.pt 로 복사 (기존 모델은 models/backup/ 에 보관)

사용 예 (C:\\Physical_AI 에서):
    python ros2_ws/src/plate_vision/tools/train.py --data plate_data/real_v1/data.yaml
"""
import argparse
import os
import shutil
import sys
from datetime import datetime

import yaml

PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
EXPECTED = ['empty_plate', 'food_plate']


def check_dataset(data_path):
    """data.yaml 을 읽어 점검하고, path 가 실제 폴더와 다르면 고쳐서 저장"""
    data_path = os.path.abspath(data_path)
    root = os.path.dirname(data_path)
    with open(data_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    names = cfg['names'] if isinstance(cfg['names'], list) else list(cfg['names'].values())
    print(f'클래스: {names}')
    if names != EXPECTED:
        print(f'[주의] 예상 클래스 {EXPECTED} 와 다름 → 노드의 클래스 이름 설정을 맞춰야 함')

    # Roboflow 기본값(../train/images)은 경로 오류가 잦아서 절대 경로로 고정
    fixed = {'path': root.replace('\\', '/'), 'train': 'train/images',
             'val': 'valid/images', 'test': 'test/images'}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        cfg.update(fixed)
        with open(data_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f'data.yaml 경로 수정 → path: {fixed["path"]}')

    for split in ('train', 'valid', 'test'):
        img_dir, lbl_dir = os.path.join(root, split, 'images'), os.path.join(root, split, 'labels')
        n_img = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
        counts = [0] * len(names)
        if os.path.isdir(lbl_dir):
            for fn in os.listdir(lbl_dir):
                with open(os.path.join(lbl_dir, fn), encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            counts[int(line.split()[0])] += 1
        print(f'  {split:5s}: 이미지 {n_img:4d}장  ' + '  '.join(f'{n} {c}개' for n, c in zip(names, counts)))
        if split in ('train', 'valid') and n_img == 0:
            sys.exit(f'[오류] {img_dir} 에 이미지가 없습니다.')
    return data_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Roboflow 에서 받은 data.yaml')
    ap.add_argument('--model', default='yolov8n.pt', help='시작 가중치 (yolov8n=빠름, yolov8s=더 정확)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8, help='GPU 메모리 부족(CUDA out of memory)이면 4')
    ap.add_argument('--name', default='plate_empty_food')
    ap.add_argument('--no-copy', action='store_true', help='학습 후 models/best.pt 로 복사하지 않음')
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'[device] {"GPU: " + torch.cuda.get_device_name(0) if device == 0 else "CPU (느림)"}')

    print('\n===== 1) 데이터셋 확인 =====')
    data = check_dataset(args.data)

    print('\n===== 2) 학습 =====')
    model = YOLO(args.model)
    model.train(
        data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        workers=0,              # Windows 데이터 로딩 오류 방지
        name=args.name,
        patience=30,            # 30 epoch 동안 나아지지 않으면 조기 종료
        hsv_h=0.0,              # 색상(Hue) 변형 끔: 파란 띠·음식 색이 바뀌지 않게
        degrees=180,            # 위에서 내려다보는 장면이라 아무 각도로 돌려도 자연스러움
        flipud=0.5,
    )
    best = os.path.join(str(model.trainer.save_dir), 'weights', 'best.pt')
    print(f'best.pt: {best}')

    print('\n===== 3) test 세트 평가 =====')
    metrics = YOLO(best).val(data=data, split='test', imgsz=args.imgsz, device=device, workers=0, verbose=False)
    names = metrics.names
    print(f'{"클래스":12s} {"Precision":>9s} {"Recall":>7s} {"mAP50":>7s} {"mAP50-95":>9s}')
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f'{names[int(c)]:12s} {p:9.3f} {r:7.3f} {ap50:7.3f} {ap:9.3f}')
    print(f'{"all":12s} {metrics.box.mp:9.3f} {metrics.box.mr:7.3f} {metrics.box.map50:7.3f} {metrics.box.map:9.3f}')

    if not args.no_copy:
        print('\n===== 4) 모델 교체 =====')
        dst = os.path.join(PKG_DIR, 'models', 'best.pt')
        if os.path.exists(dst):
            backup = os.path.join(PKG_DIR, 'models', 'backup')
            os.makedirs(backup, exist_ok=True)
            old = os.path.join(backup, f'best_{datetime.now():%Y%m%d_%H%M%S}.pt')
            shutil.copy2(dst, old)
            print(f'기존 모델 백업: {os.path.abspath(old)}')
        shutil.copy2(best, dst)
        print(f'새 모델 설치: {os.path.abspath(dst)}')


if __name__ == '__main__':
    main()
