# Firewatch

Fire and smoke detection from a webcam or RTSP camera, with a siren, a preview
window, and everything driven from a single `config.yaml`.

---

## ⚠️ Read this first

**This is not a fire alarm system.** It is a supplementary early-warning layer.

Smoke detectors sense particulates before anything is visible, work in total
darkness, and work around corners. A camera sees only what is in frame and lit.
In India your factory is already legally required to have proper detection under
Section 38 of the Factories Act 1948 and NBC 2016 Part 4, enforced through the
fire NOC. Firewatch does not satisfy any of that and must never be presented as
if it does.

If you install this somewhere real, get written permission first, and make sure
nobody comes to believe "we have AI fire detection now" — that belief is itself a
risk, no matter how well the code performs.

---

## Setup (Anaconda, Windows)

Open **Anaconda Prompt**, `cd` into this folder, then:

```bat
conda env create -f environment.yml
conda activate firewatch
```

That takes a few minutes. Then get a model:

```bat
python tools\fetch_model.py --list
python tools\fetch_model.py --model general
```

Check everything works:

```bat
python main.py --check
```

Run it:

```bat
python main.py
```

Or just double-click **`run.bat`**.

<details>
<summary>Not using conda?</summary>

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
</details>

---

## First run: test on your laptop camera

Out of the box `config.yaml` is set to `source.kind: webcam`, so `python main.py`
opens your built-in camera with a live preview. Hold up a lighter or a candle (or
just a photo of fire on your phone) and watch the `fire` meter fill.

If the wrong camera opens, change `source.webcam_index` to 1, 2, and so on.

**Keys in the preview window**

| Key | Action |
|---|---|
| `Q` / `Esc` | quit |
| `A` | silence a sounding alarm |
| `S` | save a snapshot now |
| `P` | pause / resume |

The status panel shows a hit meter per class. The white tick on each bar is the
alarm threshold — when the coloured fill passes the tick, the alarm fires.

---

## Switching to the factory camera

In `config.yaml`:

```yaml
source:
  kind: rtsp
  rtsp_url: "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"
```

Ask for the camera's **sub-stream**, not the main stream. You do not need 4K to
see a fire, and decoding 4K will eat your CPU for nothing. Channel `102` on
Hikvision and `subtype=1` on Dahua/CP Plus are the sub-streams.

You can test without editing the file:

```bat
python main.py --set source.kind=rtsp --set source.rtsp_url=rtsp://...
```

---

## How the alarm logic works

A single frame firing means nothing. A welding flash, a brake light, or sun on a
wall lights up a detector for a frame or two. Real fire persists.

So Firewatch keeps a sliding window over the last M processed frames and only
alarms when N of them contained the class:

```yaml
detection:
  fire:
    conf: 0.35      # a box must clear this to count as a hit
    window: 15      # M
    min_hits: 5     # N
```

At `target_fps: 4`, that means roughly 1.25 seconds of evidence before the siren.

It also reports whether the detected region is **growing**, by comparing box area
across the older and newer halves of the window. A spreading fire grows; a
reflection stays the same size. This is logged and displayed, not used as a gate —
a small steady fire is still a fire.

Smoke defaults are stricter (`9/20 @ 0.45`) because smoke is the noisier class:
steam, dust and exhaust all live there. If your plant has visible steam, set
`detection.smoke.enabled: false` at first and turn it on once fire is behaving.

---

## Shadow mode — do this before arming the siren

`alarm.sound_enabled` is **`false`** by default. That is deliberate.

Run it that way on the real camera for two to four weeks. It detects, logs to
`alerts/alerts.csv`, and saves snapshots, but stays silent. Then open the CSV and
count: **false alarms per camera per day**, broken down by hour.

That number is the only one that matters. A system with excellent accuracy that
cries wolf twice a shift gets muted by the night guard in week one.

The logged false positives are also exactly the training data you need — feed
them back with `tools/mine_hard_negatives.py` and retrain. You get your dataset
for free as a side effect of the deployment.

Only when the rate is acceptable, set `alarm.sound_enabled: true`.

---

## Masking out known nuisance sources

Got a welding bay or a window where the sun hits every afternoon? Mask it:

```yaml
zones:
  enabled: true
  ignore:
    - name: welding_bay
      x1: 0.60
      y1: 0.30
      x2: 0.95
      y2: 0.80
```

Coordinates are fractions of the frame, so they survive a resolution change.
Masked regions are outlined in grey in the preview.

---

## Alerting other systems

```yaml
alarm:
  hook_command: "notify.bat"
```

Called as `notify.bat <class> <snapshot_path> <confidence>`. Use it for Telegram,
ntfy, WhatsApp, SMS, or flipping a GPIO relay. It runs non-blocking, and a broken
hook can never take the detector down.

---

## Tuning

| Symptom | Change |
|---|---|
| Missing real fires | lower `detection.fire.min_hits` to 3, lower `conf` to 0.25 |
| Too many false alarms | raise `min_hits` to 7, or mine hard negatives |
| Alarm too slow | lower `window` to 10, or raise `source.target_fps` |
| Repeated alerts for one event | raise `alarm.cooldown_sec` |
| CPU pegged at 100% | lower `source.target_fps`, or `model.imgsz` to 416 |

Bias toward false positives. Missing a real fire costs far more than logging a
spurious alert, and the sliding window already absorbs most of the noise. Tune
`min_hits` before you touch confidence.

**Do not raise `model.conf` to reduce false alarms.** That filters boxes out
before the temporal logic ever sees them, so you lose the weak detections that
three consecutive frames would have confirmed. The config will refuse to start if
you set it above a class threshold.

---

## Known limitation: night

Most CCTV switches to IR mode after dark and the image goes monochrome. Colour is
the strongest cue a fire model has, and at night it is simply gone — so the
detector is weakest precisely during the unmanned night shift.

Test this early. It is the fastest way to find out whether a camera-based approach
is viable at your site at all. If night coverage matters, you need a thermal
camera, not a better model.

---

## Training your own model

Only worth doing once shadow mode has shown you a specific weakness. Then:

```bat
python tools\prepare_dataset.py --src path\to\D-Fire --dst data\dfire
python tools\train.py --data data\dfire\dfire.yaml
python tools\mine_hard_negatives.py --weights best.pt --source normal_day.mp4 --out data\dfire
python tools\train.py --data data\dfire\dfire.yaml
```

D-Fire (21,527 images, 26,557 fire/smoke boxes, includes negatives):
https://github.com/gaia-solutions-on-demand/DFireDataset

Expect mAP@50 around 0.60–0.68. That looks weak but is normal — the metric is
dragged down by tiny, hazy smoke boxes. Check the per-class numbers instead.

---

## Files

```
config.yaml              everything you configure
main.py                  entry point
run.bat                  Windows launcher
firewatch/
  config.py              load, validate, CLI overrides
  source.py              webcam / RTSP with auto-reconnect
  detector.py            model wrapper, class aliasing, ignore zones
  tracker.py             the N-of-M temporal alarm logic
  alarm.py               siren playback, cooldown, hooks
  alerts.py              CSV log and snapshots
  ui.py                  preview window drawing
tools/
  fetch_model.py         download a pretrained model
  test_alarm.py          check the siren works
  prepare_dataset.py     D-Fire -> YOLO splits
  train.py               training with fire-specific augmentation
  mine_hard_negatives.py harvest false positives for retraining
assets/alarm.wav         the siren
```

---

## Troubleshooting

**Camera will not open** — try a different `source.webcam_index`, and close
anything else using the camera (Teams, Zoom, the Camera app).

**"None of this model's classes match class_aliases"** — the model names its
classes something else. Firewatch prints the real names at startup; add them to
`model.class_aliases` in `config.yaml`.

**No sound** — `python tools\test_alarm.py`. If the WAV does not play, try
`--mode beep`, which uses `winsound` and needs nothing installed.

**RTSP is laggy or smeared** — keep `rtsp_use_tcp: true`, and make sure you are
on the sub-stream.

**Detections but no alarm** — that is the temporal logic doing its job. Watch the
hit meter; if it never reaches the white tick, lower `min_hits`.

---

## Licence note

Most YOLO weights are AGPL-3.0, inherited from Ultralytics. Fine for a college
project or internal use. If you ever want to sell this, read the licence
carefully — AGPL would require you to open-source your whole system. Apache-2.0
alternatives (RT-DETR, YOLOX) exist if that matters.
