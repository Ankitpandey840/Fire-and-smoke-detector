#!/usr/bin/env python3
"""
Train a YOLO fire/smoke detector.

The defaults here are tuned for this specific task rather than copied from
the generic COCO recipe. The important deviations:

  hsv_h = 0.008   Hue is a primary cue for fire. The stock 0.015 rotates
                  orange flames toward green/blue and teaches the model to
                  ignore the single most discriminative feature it has.
  hsv_v = 0.5     Brightness jitter IS wanted - it simulates over/under
                  exposed cameras and night scenes.
  degrees = 5.0   Fire and smoke rise. Big rotations create physically
                  impossible scenes.
  flipud = 0.0    Same reason - never flip vertically.
  scale = 0.6     Aggressive scale jitter helps a lot, because distant
                  wildfire smoke and a nearby stovetop flame are the same
                  class at wildly different sizes.

Usage:
    python train.py --data data/dfire/dfire.yaml
    python train.py --data data/dfire/dfire.yaml --model yolo11s.pt --epochs 150
    python train.py --data data/dfire/dfire.yaml --resume runs/detect/firewatch/weights/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Install dependencies first:  pip install ultralytics")


HYPERPARAMS = dict(
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3.0,
    patience=30,          # early stop if val mAP stalls for 30 epochs
    close_mosaic=10,      # disable mosaic for the last 10 epochs
    # --- augmentation ---
    hsv_h=0.008,
    hsv_s=0.7,
    hsv_v=0.5,
    degrees=5.0,
    translate=0.1,
    scale=0.6,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    erasing=0.0,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="Path to dfire.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="Base weights")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", default=16, help="Int, or -1 for auto-batch on GPU")
    ap.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu, mps")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="firewatch")
    ap.add_argument("--resume", default=None, help="Path to last.pt to resume from")
    ap.add_argument("--export", action="store_true", help="Export ONNX after training")
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data config not found: {args.data}")

    batch = int(args.batch) if str(args.batch).lstrip("-").isdigit() else args.batch

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(args.data.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=args.device,
            workers=args.workers,
            name=args.name,
            project="runs/detect",
            exist_ok=False,
            val=True,
            plots=True,
            **HYPERPARAMS,
        )

    # Final evaluation on the held-out test split.
    print("\n" + "=" * 60)
    print("Evaluating on test split")
    print("=" * 60)
    metrics = model.val(split="test", imgsz=args.imgsz, device=args.device)

    print(f"\n  mAP@50      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   {metrics.box.map:.4f}")
    for i, name in enumerate(model.names.values()):
        try:
            print(
                f"  {name:<8} P={metrics.box.p[i]:.3f}  "
                f"R={metrics.box.r[i]:.3f}  mAP50={metrics.box.ap50[i]:.3f}"
            )
        except (IndexError, TypeError):
            pass

    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    print(f"\n  Best weights: {best}")

    if args.export and best.exists():
        print("\nExporting ONNX ...")
        YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)
        print(f"  -> {best.with_suffix('.onnx')}")

    print(f"\nNext: python detect_live.py --weights {best} --source 0")


if __name__ == "__main__":
    main()
