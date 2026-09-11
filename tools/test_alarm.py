#!/usr/bin/env python3
"""
Test the alarm sound without waiting for a real detection.

    python tools/test_alarm.py
    python tools/test_alarm.py --mode beep
    python tools/test_alarm.py --duration 5

Worth running before you deploy. Finding out the siren does not work is much
better now than during an actual fire.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firewatch.alarm import AlarmPlayer  # noqa: E402
from firewatch.config import load, resolve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["beep", "wav"], help="Override sound_mode")
    ap.add_argument("--duration", type=float, help="Override duration in seconds")
    args = ap.parse_args()

    cfg = load(args.config)

    # Force sound on for the test, whatever the config says.
    cfg["alarm"]["sound_enabled"] = True
    if args.mode:
        cfg["alarm"]["sound_mode"] = args.mode
    if args.duration:
        cfg["alarm"]["duration_sec"] = args.duration

    wav = resolve(cfg, cfg["alarm"]["wav_file"])
    mode = cfg["alarm"]["sound_mode"]
    dur = float(cfg["alarm"]["duration_sec"])

    print(f"\n  mode      {mode}")
    if mode == "wav":
        print(f"  file      {wav}  ({'found' if wav.exists() else 'MISSING'})")
    print(f"  duration  {dur:.0f}s")
    print("\n  Sounding alarm. Ctrl-C to stop early.\n")

    player = AlarmPlayer(cfg, wav)
    player.trigger()

    try:
        start = time.time()
        while player.is_sounding:
            left = dur - (time.time() - start)
            print(f"\r  sounding ... {max(left, 0):4.1f}s ", end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n  Stopped early.")
    finally:
        player.shutdown()

    print("\n\n  Done. If you heard nothing:")
    print("    - check your system volume is not muted")
    print("    - on Windows try:  python tools/test_alarm.py --mode beep")
    print("    - on Linux install alsa-utils (aplay) or use --mode beep\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
