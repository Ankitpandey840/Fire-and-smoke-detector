#!/usr/bin/env python3
"""
Mine hard negatives from footage you know contains no fire.

This is the highest-leverage step in the whole project and the one most
people skip. Point your trained model at an hour of ordinary footage from
the actual camera you plan to deploy on. Every detection it produces is,
by definition, a false positive. Saving those frames as background images
(an empty .txt label) and retraining teaches the model that this particular
sunset / heater / brake light / reflection is not fire.

One round of this typically cuts the false-alarm rate more than any
architecture change you could make.

Usage:
    # 1. Record or collect footage with NO fire in it
    python mine_hard_negatives.py --weights best.pt --source normal_day.mp4 \
        --out data/dfire --split train

    # 2. Retrain
    python train.py --data data/dfire/dfire.yaml

Tip: run with --conf below your deployment threshold. Near-misses are worth
catching too, since they're what a slightly different lighting condition
would turn into a real false alarm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Install dependencies first:  pip install ultralytics opencv-python")


def iter_sources(source: Path):
    """Yield (frame, tag) from a video file, a directory of videos, or images."""
    if source.is_dir():
        vids = sorted(
            p for p in source.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        imgs = sorted(
            p for p in source.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for p in imgs:
            img = cv2.imread(str(p))
            if img is not None:
                yield img, p.stem
        for v in vids:
            yield from _iter_video(v)
    elif source.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        img = cv2.imread(str(source))
        if img is not None:
            yield img, source.stem
    else:
        yield from _iter_video(source)


def _iter_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  could not open {path}")
        return
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, f"{path.stem}_{idx:06d}"
        idx += 1
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, type=Path,
                    help="Video, image, or directory of fire-FREE footage")
    ap.add_argument("--out", required=True, type=Path,
                    help="Dataset root created by prepare_dataset.py")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--conf", type=float, default=0.15,
                    help="Set below your deployment threshold to catch near-misses")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stride", type=int, default=5,
                    help="Process every Nth frame; consecutive frames are near-duplicates")
    ap.add_argument("--max-saves", type=int, default=500,
                    help="Cap so one bad video doesn't swamp the training set")
    ap.add_argument("--prefix", default="hardneg")
    ap.add_argument("--review-dir", type=Path, default=Path("hard_negatives_review"),
                    help="Annotated copies so you can eyeball what was caught")
    args = ap.parse_args()

    img_dir = args.out / "images" / args.split
    lbl_dir = args.out / "labels" / args.split
    if not img_dir.is_dir():
        raise SystemExit(f"Not a prepared dataset: {img_dir} missing. Run prepare_dataset.py first.")
    args.review_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    names = model.names

    seen = saved = 0
    print(f"Scanning {args.source} for false positives (conf >= {args.conf}) ...\n")

    for frame, tag in iter_sources(args.source):
        if seen % args.stride != 0:
            seen += 1
            continue
        seen += 1
        if saved >= args.max_saves:
            print(f"\nHit --max-saves limit ({args.max_saves}).")
            break

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)[0]
        if len(res.boxes) == 0:
            continue

        stem = f"{args.prefix}_{tag}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame)
        # Empty label file = background image. This is the whole point.
        (lbl_dir / f"{stem}.txt").write_text("")
        saved += 1

        annotated = frame.copy()
        detail = []
        for box in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            label = f"{names[int(box.cls)]} {float(box.conf):.2f}"
            detail.append(label)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 60, 245), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 245), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.review_dir / f"{stem}.jpg"), annotated)

        if saved % 25 == 0 or saved <= 5:
            print(f"  [{saved:>4}] {tag}  ->  {', '.join(detail[:3])}")

    processed = seen // max(args.stride, 1)
    rate = 100 * saved / processed if processed else 0
    print(f"\n  frames processed : {processed:,}")
    print(f"  false positives  : {saved:,}  ({rate:.1f}%)")
    print(f"\n  Added to: {img_dir}")
    print(f"  Review the annotated copies in {args.review_dir} before retraining.")
    print(f"  If any of them contain REAL fire, delete them - you'd be teaching")
    print(f"  the model to ignore actual fire.")
    print(f"\n  Then: python train.py --data {args.out / 'dfire.yaml'}")


if __name__ == "__main__":
    main()
