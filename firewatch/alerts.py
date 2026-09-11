"""Alert recording: CSV log, snapshots, and logger setup."""

from __future__ import annotations

import csv
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

import cv2

log = logging.getLogger("firewatch.alerts")

CSV_HEADER = [
    "timestamp", "class", "hits", "window", "peak_conf",
    "growth", "level", "snapshot", "annotated",
]


def setup_logging(cfg: dict, outdir: Path) -> None:
    level = getattr(logging, str(cfg["logging"]["level"]).upper(), logging.INFO)
    root = logging.getLogger("firewatch")
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if cfg["logging"]["to_file"]:
        path = Path(cfg["logging"]["file"])
        if not path.is_absolute():
            path = outdir.parent / path if path.parent != Path(".") else outdir / path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s %(name)s  %(message)s")
        )
        root.addHandler(handler)


class AlertSink:
    """Writes an alarm event to disk."""

    def __init__(self, cfg: dict, outdir: Path):
        ocfg = cfg["output"]
        self.dir = outdir
        self.snapshots = outdir / "snapshots"
        self.save_snapshot = bool(ocfg["save_snapshot"])
        self.save_annotated = bool(ocfg["save_annotated"])
        self.csv_enabled = bool(ocfg["csv_log"])
        self.csv_path = outdir / "alerts.csv"

        self.dir.mkdir(parents=True, exist_ok=True)
        if self.save_snapshot or self.save_annotated:
            self.snapshots.mkdir(parents=True, exist_ok=True)

        if self.csv_enabled and not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADER)

    def record(self, tracker, clean_frame, annotated_frame) -> tuple[str, str]:
        stamp = datetime.now()
        base = f"{stamp:%Y%m%d_%H%M%S}_{tracker.name}"

        snap_path = ""
        anno_path = ""
        if self.save_snapshot and clean_frame is not None:
            p = self.snapshots / f"{base}.jpg"
            cv2.imwrite(str(p), clean_frame)
            snap_path = str(p)
        if self.save_annotated and annotated_frame is not None:
            p = self.snapshots / f"{base}_boxed.jpg"
            cv2.imwrite(str(p), annotated_frame)
            anno_path = str(p)

        if self.csv_enabled:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    stamp.isoformat(timespec="seconds"),
                    tracker.name,
                    tracker.hit_count,
                    tracker.window,
                    f"{tracker.peak_conf:.3f}",
                    tracker.growth_label(),
                    tracker.level,
                    Path(snap_path).name if snap_path else "",
                    Path(anno_path).name if anno_path else "",
                ])

        growth = tracker.growth_label()
        log.warning(
            "ALARM  %s  %d/%d frames  peak %.2f%s",
            tracker.name.upper(),
            tracker.hit_count,
            tracker.window,
            tracker.peak_conf,
            f"  region {growth}" if growth else "",
        )
        if snap_path:
            log.info("       snapshot -> %s", snap_path)

        return snap_path, anno_path

    def manual_snapshot(self, frame) -> str:
        self.snapshots.mkdir(parents=True, exist_ok=True)
        p = self.snapshots / f"{datetime.now():%Y%m%d_%H%M%S}_manual.jpg"
        cv2.imwrite(str(p), frame)
        log.info("Snapshot saved -> %s", p)
        return str(p)
