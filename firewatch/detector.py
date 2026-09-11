"""
Thin wrapper around an Ultralytics YOLO model.

Its job is to turn raw model output into a list of Detection objects using
Firewatch's own class names ("fire" / "smoke"), regardless of what the
downloaded model happens to call things, and to drop anything that falls
inside a configured ignore zone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("firewatch.detector")


@dataclass
class Detection:
    cls: str                      # normalised: "fire" or "smoke"
    conf: float
    xyxy: tuple[int, int, int, int]
    area_frac: float              # box area as a fraction of the frame


@dataclass
class IgnoreZone:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def contains_center(self, cx: float, cy: float) -> bool:
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


def _scalar(value) -> float:
    """
    Pull a Python number out of a box attribute.

    Ultralytics returns torch tensors, but ONNX/other backends can hand back
    numpy arrays, and numpy 2 refuses to coerce a 1-element array with int().
    This handles all of them.
    """
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except (ValueError, TypeError):
            pass
    try:
        return float(value)
    except TypeError:
        return float(value[0])


def _pick_device(requested: str) -> str:
    if str(requested).lower() != "auto":
        return str(requested)
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - torch missing or broken, fall back
        pass
    return "cpu"


class Detector:
    def __init__(self, cfg: dict, weights_path: Path):
        mcfg = cfg["model"]
        self.imgsz = int(mcfg["imgsz"])
        self.conf = float(mcfg["conf"])
        self.iou = float(mcfg["iou"])
        self.aliases: dict[str, str] = dict(mcfg["class_aliases"])
        self.device = _pick_device(mcfg["device"])

        self.zones: list[IgnoreZone] = []
        if cfg["zones"]["enabled"]:
            for z in cfg["zones"]["ignore"] or []:
                self.zones.append(
                    IgnoreZone(
                        name=z.get("name", "zone"),
                        x1=float(z["x1"]), y1=float(z["y1"]),
                        x2=float(z["x2"]), y2=float(z["y2"]),
                    )
                )

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}\n\n"
                f"Download a pretrained fire/smoke model with:\n"
                f"    python tools/fetch_model.py\n\n"
                f"or point model.weights in config.yaml at your own .pt file."
            )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise SystemExit(
                "ultralytics is not installed.\n"
                "    conda activate firewatch\n"
                "    pip install ultralytics"
            ) from e

        log.info("Loading model %s on device '%s' ...", weights_path.name, self.device)
        self.model = YOLO(str(weights_path))
        self.raw_names: dict[int, str] = dict(self.model.names)
        log.info("Model classes: %s", list(self.raw_names.values()))

        mapped = {
            raw: self.aliases[raw] for raw in self.raw_names.values()
            if raw in self.aliases
        }
        if not mapped:
            log.warning(
                "None of this model's classes %s match model.class_aliases in "
                "config.yaml. Every detection will be discarded. Add the "
                "correct names to class_aliases.",
                list(self.raw_names.values()),
            )
        else:
            log.info("Class mapping: %s", mapped)
        self.unmapped_warned: set[str] = set()

    def predict(self, frame) -> list[Detection]:
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        out: list[Detection] = []
        for box in result.boxes:
            raw_name = self.raw_names[int(_scalar(box.cls))]
            name = self.aliases.get(raw_name)
            if name is None:
                if raw_name not in self.unmapped_warned:
                    log.debug("Ignoring unmapped class %r", raw_name)
                    self.unmapped_warned.add(raw_name)
                continue

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h

            skip = next((z for z in self.zones if z.contains_center(cx, cy)), None)
            if skip is not None:
                log.debug("Detection suppressed by ignore zone %r", skip.name)
                continue

            out.append(
                Detection(
                    cls=name,
                    conf=_scalar(box.conf),
                    xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    area_frac=((x2 - x1) * (y2 - y1)) / frame_area,
                )
            )
        return out
