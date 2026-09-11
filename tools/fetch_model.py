#!/usr/bin/env python3
"""
Download a pretrained fire/smoke model so you can start without training.

    python tools/fetch_model.py                 # list the options
    python tools/fetch_model.py --model general # download one
    python tools/fetch_model.py --repo someone/their-model --file best.pt

IMPORTANT - read this before you trust any of these.

None of these models were trained on YOUR camera, YOUR lighting, or YOUR
factory floor. A model's false-positive rate is not a property of the model,
it is a property of the model paired with a specific scene. Whichever one you
pick, run it in shadow mode (alarm.sound_enabled: false) on the real camera
for a couple of weeks and count the false alarms before you arm the siren.

After downloading, check the class names the model reports when Firewatch
starts. If they are not "fire" and "smoke", add them to model.class_aliases
in config.yaml or every detection will be silently discarded.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Community models on Hugging Face. Quality varies; treat the notes seriously.
MODELS = {
    "wildfire": {
        "repo": "pyronear/yolo11s_sensitive-detector",
        "file": "best.pt",
        "imgsz": 1024,
        "note": (
            "Best-engineered option here. PyroNear is a nonprofit running real "
            "wildfire detection towers, with versioned releases and ONNX/NCNN "
            "exports. Tuned for EARLY, LONG-RANGE OUTDOOR SMOKE at 1024px - "
            "the wrong prior for an indoor factory floor."
        ),
    },
    "general": {
        "repo": "TommyNgx/YOLOv10-Fire-and-Smoke-Detection",
        "file": "best.pt",
        "imgsz": 640,
        "note": (
            "General-purpose fire + smoke, YOLOv10. Reasonable starting point "
            "for indoor/mixed scenes. Community model, training data not fully "
            "documented."
        ),
    },
    "smoke": {
        "repo": "kittendev/YOLOv8m-smoke-detection",
        "file": "best.pt",
        "imgsz": 640,
        "note": (
            "Smoke only, no fire class. Medium-size model, slower. Useful if "
            "early smoke matters more to you than flame."
        ),
    },
}


def list_models() -> None:
    print("\nAvailable pretrained models\n" + "=" * 70)
    for key, m in MODELS.items():
        print(f"\n  {key}")
        print(f"    repo   {m['repo']}")
        print(f"    imgsz  {m['imgsz']}")
        print("    " + _wrap(m["note"], 66, "    "))
    print("\n" + "=" * 70)
    print("\nDownload with:  python tools/fetch_model.py --model general\n")
    print("Licence note: most YOLO weights are AGPL-3.0. Fine for a project or")
    print("internal use. If you ever want to sell this, read the licence first.\n")


def _wrap(text: str, width: int, indent: str) -> str:
    import textwrap

    return ("\n" + indent).join(textwrap.wrap(text, width))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), help="Preset to download")
    ap.add_argument("--repo", help="Any Hugging Face repo id (overrides --model)")
    ap.add_argument("--file", default="best.pt", help="Filename inside the repo")
    ap.add_argument("--out", default="weights/fire_smoke.pt",
                    help="Where to save (should match model.weights in config.yaml)")
    ap.add_argument("--list", action="store_true", help="Show the presets and exit")
    args = ap.parse_args()

    if args.list or (not args.model and not args.repo):
        list_models()
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed.\n    pip install huggingface_hub",
              file=sys.stderr)
        return 1

    if args.repo:
        repo, fname, imgsz, note = args.repo, args.file, None, ""
    else:
        m = MODELS[args.model]
        repo, fname, imgsz, note = m["repo"], m["file"], m["imgsz"], m["note"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {fname} from {repo} ...")
    try:
        cached = hf_hub_download(repo_id=repo, filename=fname)
    except Exception as e:  # noqa: BLE001
        print(f"\nDownload failed: {e}\n", file=sys.stderr)
        print("The repo or filename may have changed. Browse for alternatives at:")
        print("  https://huggingface.co/models?search=fire+smoke+yolo")
        print("then retry with:  --repo <repo_id> --file <filename>\n")
        return 1

    shutil.copy2(cached, out)
    size_mb = out.stat().st_size / 1e6
    print(f"Saved to {out}  ({size_mb:.1f} MB)")

    if note:
        print("\nAbout this model:")
        print("  " + _wrap(note, 70, "  "))
    if imgsz and imgsz != 640:
        print(f"\n  NOTE: this model expects imgsz {imgsz}.")
        print(f"  Set  model.imgsz: {imgsz}  in config.yaml or accuracy will suffer.")

    print("\nNext steps")
    print("  1. python main.py --check")
    print("  2. Check the class names it prints. If they are not fire/smoke,")
    print("     add them to model.class_aliases in config.yaml.")
    print("  3. python main.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
