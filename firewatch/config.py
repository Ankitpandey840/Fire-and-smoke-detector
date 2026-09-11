"""Load, validate and override the YAML configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when config.yaml is missing something or has a bad value."""


DEFAULTS: dict[str, Any] = {
    "source": {
        "kind": "webcam",
        "webcam_index": 0,
        "rtsp_url": "",
        "file_path": "",
        "rtsp_use_tcp": True,
        "target_fps": 4,
        "reconnect_delay_sec": 5,
        "max_reconnect_attempts": 0,
    },
    "model": {
        "weights": "weights/fire_smoke.pt",
        "imgsz": 640,
        "device": "auto",
        "conf": 0.20,
        "iou": 0.45,
        "class_aliases": {"fire": "fire", "smoke": "smoke"},
    },
    "detection": {
        "fire": {"enabled": True, "conf": 0.35, "window": 15, "min_hits": 5},
        "smoke": {"enabled": True, "conf": 0.45, "window": 20, "min_hits": 9},
        "growth_ratio_alert": 1.15,
    },
    "zones": {"enabled": False, "ignore": []},
    "alarm": {
        "sound_enabled": False,
        "sound_mode": "wav",
        "wav_file": "assets/alarm.wav",
        "beep_freq_hz": 1000,
        "beep_duration_ms": 400,
        "beep_repeat": 6,
        "duration_sec": 10,
        "cooldown_sec": 60,
        "allow_acknowledge": True,
        "hook_command": "",
    },
    "preview": {
        "enabled": True,
        "window_name": "Firewatch",
        "scale": 0.8,
        "show_boxes": True,
        "show_labels": True,
        "show_fps": True,
        "show_panel": True,
        "flash_on_alarm": True,
    },
    "output": {
        "dir": "alerts",
        "csv_log": True,
        "save_snapshot": True,
        "save_annotated": True,
        "clip_seconds_before": 0,
    },
    "logging": {"level": "INFO", "to_file": True, "file": "alerts/firewatch.log"},
}

CLASSES = ("fire", "smoke")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(text: str) -> Any:
    """Turn a CLI string into a bool / int / float / str."""
    low = text.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides of the form 'source.kind=webcam'."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Bad override {item!r}. Expected key.path=value")
        dotted, raw = item.split("=", 1)
        parts = dotted.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ConfigError(f"Unknown config section: {dotted}")
            node = node[part]
        if parts[-1] not in node:
            raise ConfigError(f"Unknown config key: {dotted}")
        node[parts[-1]] = _coerce(raw)
    return cfg


def validate(cfg: dict) -> None:
    """Catch the mistakes that would otherwise fail confusingly at runtime."""
    src = cfg["source"]
    if src["kind"] not in ("webcam", "rtsp", "file"):
        raise ConfigError(
            f"source.kind must be webcam, rtsp or file (got {src['kind']!r})"
        )
    if src["kind"] == "rtsp" and not str(src["rtsp_url"] or "").strip():
        raise ConfigError("source.kind is 'rtsp' but source.rtsp_url is empty")
    if src["kind"] == "file" and not str(src["file_path"] or "").strip():
        raise ConfigError("source.kind is 'file' but source.file_path is empty")

    det = cfg["detection"]
    enabled = []
    for name in CLASSES:
        rules = det.get(name)
        if not isinstance(rules, dict):
            raise ConfigError(f"Missing detection.{name} section")
        if rules["min_hits"] > rules["window"]:
            raise ConfigError(
                f"detection.{name}.min_hits ({rules['min_hits']}) cannot exceed "
                f"window ({rules['window']})"
            )
        if rules["min_hits"] < 1:
            raise ConfigError(f"detection.{name}.min_hits must be at least 1")
        if not 0.0 < rules["conf"] <= 1.0:
            raise ConfigError(f"detection.{name}.conf must be between 0 and 1")
        if rules["enabled"]:
            enabled.append(name)
    if not enabled:
        raise ConfigError("Both fire and smoke are disabled. Nothing to detect.")

    mdl = cfg["model"]
    if not 0.0 < mdl["conf"] <= 1.0:
        raise ConfigError("model.conf must be between 0 and 1")
    for name in CLASSES:
        if det[name]["enabled"] and det[name]["conf"] < mdl["conf"]:
            raise ConfigError(
                f"detection.{name}.conf ({det[name]['conf']}) is below "
                f"model.conf ({mdl['conf']}). The detector would filter those "
                f"boxes out before the rule ever sees them. Lower model.conf."
            )

    al = cfg["alarm"]
    if al["sound_mode"] not in ("beep", "wav"):
        raise ConfigError("alarm.sound_mode must be 'beep' or 'wav'")
    if al["cooldown_sec"] < al["duration_sec"]:
        raise ConfigError(
            "alarm.cooldown_sec should be >= alarm.duration_sec, otherwise the "
            "alarm can retrigger while it is still sounding"
        )

    if cfg["zones"]["enabled"]:
        for i, z in enumerate(cfg["zones"]["ignore"] or []):
            for k in ("x1", "y1", "x2", "y2"):
                if k not in z:
                    raise ConfigError(f"zones.ignore[{i}] is missing '{k}'")
                if not 0.0 <= float(z[k]) <= 1.0:
                    raise ConfigError(
                        f"zones.ignore[{i}].{k} must be a fraction between 0 and 1"
                    )


def load(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Read config.yaml, merge over defaults, apply overrides, validate."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}\n"
            f"Run from the project folder, or pass --config <path>."
        )

    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping")

    cfg = _deep_merge(DEFAULTS, user_cfg)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    validate(cfg)

    cfg["_config_path"] = str(path.resolve())
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def resolve(cfg: dict, relative: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return Path(cfg.get("_root", ".")) / p
