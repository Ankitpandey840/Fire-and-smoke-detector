#!/usr/bin/env python3
"""
Firewatch - fire and smoke detection from a camera feed.

Everything is driven by config.yaml. Typical use:

    python main.py                                  # use config.yaml
    python main.py --config configs/plant_a.yaml    # a different config
    python main.py --set source.kind=webcam         # override one value
    python main.py --check                          # validate setup and exit

Keys in the preview window:
    Q / ESC   quit
    A         acknowledge (silence) a sounding alarm
    S         save a snapshot now
    P         pause / resume
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from firewatch import __version__, ui
from firewatch.alarm import AlarmPlayer, run_hook
from firewatch.alerts import AlertSink, setup_logging
from firewatch.config import ConfigError, load, resolve
from firewatch.detector import Detector
from firewatch.source import VideoSource
from firewatch.tracker import ALARM, build_trackers

log = logging.getLogger("firewatch.main")

# Stop after this many inference failures in a row - at that point something
# is genuinely broken, not just one corrupt frame.
MAX_CONSECUTIVE_ERRORS = 20

BANNER = r"""
  ___ _                        _      _
 | __(_)_ _ _____ __ ____ _ __| |_ __| |_
 | _|| | '_/ -_) V  V / _` / _| ' \\/ _| ' \
 |_| |_|_| \___|\_/\_/\__,_\__|_||_\__|_||_|  v{v}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config.yaml", help="Path to config file")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config value, e.g. --set source.kind=webcam",
    )
    p.add_argument(
        "--check", action="store_true",
        help="Validate config, model and camera, then exit without running",
    )
    p.add_argument("--version", action="version", version=f"Firewatch {__version__}")
    return p.parse_args()


def preflight(cfg: dict) -> bool:
    """Check the things that commonly go wrong, before the main loop."""
    ok = True

    weights = resolve(cfg, cfg["model"]["weights"])
    if weights.exists():
        size_mb = weights.stat().st_size / 1e6
        print(f"  [ok]   model weights      {weights.name} ({size_mb:.1f} MB)")
    else:
        print(f"  [FAIL] model weights      not found: {weights}")
        print("         run:  python tools/fetch_model.py")
        ok = False

    if cfg["alarm"]["sound_enabled"]:
        if cfg["alarm"]["sound_mode"] == "wav":
            wav = resolve(cfg, cfg["alarm"]["wav_file"])
            if wav.exists():
                print(f"  [ok]   alarm sound        {wav.name}")
            else:
                print(f"  [warn] alarm wav missing  {wav} (will beep instead)")
        else:
            print("  [ok]   alarm sound        beep mode")
    else:
        print("  [ok]   alarm sound        DISABLED (shadow mode)")

    src = cfg["source"]
    print(f"  [ok]   source             {src['kind']}")

    print("\n  Opening video source ...")
    vs = VideoSource(cfg)
    if vs.open():
        frame = None
        for _ in range(10):
            got, frame = vs.cap.read()
            if got:
                break
            time.sleep(0.2)
        if frame is not None:
            h, w = frame.shape[:2]
            print(f"  [ok]   video source       reading frames at {w}x{h}")
        else:
            print("  [FAIL] video source       opened but returned no frames")
            ok = False
        vs.close()
    else:
        print(f"  [FAIL] video source       could not open {vs.label}")
        if src["kind"] == "webcam":
            print("         try a different source.webcam_index (0, 1, 2 ...)")
            print("         and close any app already using the camera")
        ok = False

    return ok


def main() -> int:
    args = parse_args()
    print(BANNER.format(v=__version__))

    try:
        cfg = load(args.config, args.overrides)
    except ConfigError as e:
        print(f"Config error:\n  {e}", file=sys.stderr)
        return 2

    outdir = resolve(cfg, cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg, outdir)

    print(f"  config: {cfg['_config_path']}\n")
    print("  Preflight")
    print("  " + "-" * 52)
    healthy = preflight(cfg)
    print("  " + "-" * 52)

    if args.check:
        print("\n  " + ("All checks passed." if healthy else "Some checks FAILED."))
        return 0 if healthy else 1
    if not healthy:
        print("\n  Fix the failures above, then run again.")
        return 1

    weights = resolve(cfg, cfg["model"]["weights"])
    detector = Detector(cfg, weights)
    trackers = build_trackers(cfg)
    sink = AlertSink(cfg, outdir)

    wav = resolve(cfg, cfg["alarm"]["wav_file"]) if cfg["alarm"]["wav_file"] else None
    player = AlarmPlayer(cfg, wav)
    hook = str(cfg["alarm"]["hook_command"])

    pcfg = cfg["preview"]
    preview_on = bool(pcfg["enabled"])
    save_annotated = bool(cfg["output"]["save_annotated"])

    if not cfg["alarm"]["sound_enabled"]:
        log.info("SHADOW MODE - detections are logged, the siren stays silent.")
        log.info("Set alarm.sound_enabled: true in config.yaml to arm it.")

    rules = ", ".join(
        f"{t.name} {t.min_hits}/{t.window}@{t.conf:.2f}" for t in trackers.values()
    )
    log.info("Watching for: %s", rules)
    log.info("Press Ctrl-C to stop." if not preview_on else "Press Q in the window to stop.")

    fps = 0.0
    t_prev = time.time()
    paused = False
    processed = 0
    consecutive_errors = 0
    started = time.time()

    source = VideoSource(cfg)
    try:
        for frame in source.frames():
            if paused:
                if preview_on:
                    key = cv2.waitKey(50) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("p"):
                        paused = False
                        log.info("Resumed.")
                continue

            clean = frame.copy()

            # A single malformed frame must not take down a system that is
            # supposed to run for weeks. Log it, skip it, carry on - but give
            # up if it keeps happening, because that means something real is
            # broken rather than one bad frame.
            try:
                detections = detector.predict(frame)
                consecutive_errors = 0
            except Exception as e:  # noqa: BLE001
                consecutive_errors += 1
                log.error("Inference failed on frame %d: %s", processed + 1, e)
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log.critical(
                        "%d consecutive inference failures. Stopping.",
                        consecutive_errors,
                    )
                    break
                continue

            processed += 1

            if detections:
                log.debug(
                    "frame %d: %s", processed,
                    ", ".join(f"{d.cls}:{d.conf:.2f}" for d in detections),
                )

            # Boxes are needed for the preview AND for annotated snapshots, so
            # draw them whenever either is switched on - not just when the
            # window is visible.
            display = frame
            if (preview_on and pcfg["show_boxes"]) or save_annotated:
                display = ui.draw_boxes(display, detections, pcfg["show_labels"])
            if preview_on and detector.zones:
                display = ui.draw_zones(display, detector.zones)

            annotated_for_save = display.copy() if save_annotated else None

            for tracker in trackers.values():
                tracker.update(tracker.observe(detections))
                if tracker.should_alert(float(cfg["alarm"]["cooldown_sec"])):
                    snap, _ = sink.record(tracker, clean, annotated_for_save)
                    player.trigger()
                    if hook.strip():
                        run_hook(hook, tracker.name, snap, tracker.peak_conf)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            if preview_on:
                if pcfg["show_panel"]:
                    display = ui.draw_panel(
                        display, list(trackers.values()), fps,
                        player.is_sounding, paused,
                    )
                display = ui.draw_hint(display)
                cv2.imshow(pcfg["window_name"], ui.scaled(display, float(pcfg["scale"])))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("a"):
                    if not player.acknowledge():
                        log.info("Nothing to silence.")
                if key == ord("s"):
                    sink.manual_snapshot(clean)
                if key == ord("p"):
                    paused = True
                    log.info("Paused. Press P to resume.")

    except KeyboardInterrupt:
        print()
        log.info("Interrupted.")
    finally:
        player.shutdown()
        source.close()
        cv2.destroyAllWindows()

        mins = (time.time() - started) / 60
        print("\n  Session summary")
        print("  " + "-" * 52)
        print(f"   runtime          {mins:.1f} min")
        print(f"   frames processed {processed:,}")
        if source.reconnects:
            print(f"   reconnects       {source.reconnects}")
        for t in trackers.values():
            rate = t.total_alarms / (mins / 60) if mins > 0.5 else 0
            print(f"   {t.name:<6} alarms    {t.total_alarms}"
                  + (f"  ({rate:.1f}/hour)" if mins > 0.5 else ""))
        if cfg["output"]["csv_log"]:
            print(f"\n   alert log        {sink.csv_path}")
        print("  " + "-" * 52)

    return 0


if __name__ == "__main__":
    sys.exit(main())
