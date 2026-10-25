#!/usr/bin/env python3
"""
Render video perbandingan dua model YOLO side-by-side.
Kiri = model A, kanan = model B. Parameter inferensi mengikuti dashcam/core/application.py.
"""

import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np
from ultralytics import YOLO

COLORS = {'crack': (0, 165, 255), 'pothole': (0, 0, 255)}  # BGR
HEADER_BG = {'KIRI': (90, 90, 90), 'KANAN': (140, 80, 20)}


def draw_side(frame, result, names, label, model_name, stats, scale):
    boxes = result.boxes
    n = len(boxes)
    stats['boxes'] += n
    stats['frames_detected'] += 1 if n else 0
    for (x1, y1, x2, y2), cls, conf in zip((boxes.xyxy.cpu().numpy() * scale).astype(int),
                                          boxes.cls.cpu().numpy().astype(int),
                                          boxes.conf.cpu().numpy()):
        name = names[cls]
        stats['per_class'][name] = stats['per_class'].get(name, 0) + 1
        color = COLORS.get(name, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(frame, text, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), HEADER_BG[label], -1)
    header = (f"{label}: {model_name}   kotak di frame ini: {n}   total kotak: {stats['boxes']}"
              f"   frame berdeteksi: {stats['frames_detected']}")
    cv2.putText(frame, header, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def has_nvenc():
    out = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    return 'h264_nvenc' in out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--model-left', required=True)
    p.add_argument('--model-right', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--side-width', type=int, default=960, help='Lebar tiap panel output')
    p.add_argument('--max-frames', type=int, default=0)
    args = p.parse_args()

    models = [YOLO(args.model_left), YOLO(args.model_right)]
    model_names = [os.path.splitext(os.path.basename(m))[0] for m in (args.model_left, args.model_right)]

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sw = args.side_width
    sh = int(round(h * sw / w / 2) * 2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    enc = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '23'] if has_nvenc() else \
          ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23']
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                           '-s', f'{sw * 2}x{sh}', '-r', f'{fps}', '-i', '-',
                           *enc, '-pix_fmt', 'yuv420p', args.out], stdin=subprocess.PIPE)

    stats = [{'boxes': 0, 'frames_detected': 0, 'per_class': {}} for _ in models]
    idx, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break
        panels = []
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        for i, (m, label) in enumerate(zip(models, ('KIRI', 'KANAN'))):
            res = m.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            device=0, quantize=16, verbose=False, max_det=100)[0]
            panels.append(draw_side(small.copy(), res, m.names, label, model_names[i], stats[i], sw / w))
        ff.stdin.write(np.hstack(panels).tobytes())
        idx += 1
        if idx % 500 == 0:
            el = time.time() - t0
            print(f"  {idx}/{total} frame  {idx / el:.1f} fps  ETA {(total - idx) / (idx / el) / 60:.1f} menit", flush=True)

    cap.release()
    ff.stdin.close()
    ff.wait()

    summary = {'video': args.video, 'frames': idx, 'conf': args.conf, 'imgsz': args.imgsz,
               'left': {'model': model_names[0], **stats[0]},
               'right': {'model': model_names[1], **stats[1]}}
    with open(os.path.splitext(args.out)[0] + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if ff.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
