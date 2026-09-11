"""
Temporal alarm logic.

A single frame firing means nothing. A brake light, a welding flash, a
reflection off a window, or sun on a wall will each light up a detector for
a frame or two. Real fire persists.

So each class keeps a sliding window over the last M processed frames and
escalates:

    CLEAR   nothing seen recently
    WATCH   something is appearing, not yet enough to trust
    ALARM   at least N of the last M frames contained this class

It also tracks whether the detected region is growing, by comparing mean box
area across the older and newer halves of the window. A spreading fire grows;
a reflection holds steady. This is reported, not used as a gate, because a
small steady fire is still a fire.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

CLEAR = "CLEAR"
WATCH = "WATCH"
ALARM = "ALARM"


@dataclass
class Observation:
    """What one class looked like in one frame."""

    hit: bool = False
    conf: float = 0.0
    area: float = 0.0


@dataclass
class ClassTracker:
    name: str
    window: int
    min_hits: int
    conf: float
    growth_alert: float = 1.15

    history: deque = field(init=False, repr=False)
    level: str = field(default=CLEAR, init=False)
    previous_level: str = field(default=CLEAR, init=False)
    last_alarm_at: float = field(default=0.0, init=False)
    total_alarms: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.window)

    def observe(self, detections) -> Observation:
        """Collapse this frame's detections for this class into one observation."""
        obs = Observation()
        for d in detections:
            if d.cls != self.name:
                continue
            if d.conf >= self.conf:
                obs.hit = True
            obs.conf = max(obs.conf, d.conf)
            obs.area = max(obs.area, d.area_frac)
        return obs

    def update(self, obs: Observation) -> str:
        self.history.append(obs)
        self.previous_level = self.level

        hits = self.hit_count
        if hits >= self.min_hits:
            self.level = ALARM
        elif hits > 0:
            self.level = WATCH
        else:
            self.level = CLEAR
        return self.level

    @property
    def hit_count(self) -> int:
        return sum(1 for o in self.history if o.hit)

    @property
    def peak_conf(self) -> float:
        return max((o.conf for o in self.history), default=0.0)

    @property
    def just_escalated(self) -> bool:
        return self.level == ALARM and self.previous_level != ALARM

    def growth(self) -> float:
        """
        Ratio of mean box area in the newer half of the window to the older
        half. Returns 0.0 when there is not enough data to say anything.
        """
        areas = [o.area for o in self.history if o.hit]
        if len(areas) < 6:
            return 0.0
        mid = len(areas) // 2
        older = sum(areas[:mid]) / mid
        newer = sum(areas[mid:]) / (len(areas) - mid)
        if older <= 1e-9:
            return 0.0
        return newer / older

    def growth_label(self) -> str:
        g = self.growth()
        if not g:
            return ""
        if g >= self.growth_alert:
            return f"GROWING {g:.2f}x"
        if g <= 1.0 / self.growth_alert:
            return f"shrinking {g:.2f}x"
        return f"steady {g:.2f}x"

    def should_alert(self, cooldown: float, now: float | None = None) -> bool:
        """True once per alarm event, respecting the cooldown."""
        if self.level != ALARM:
            return False
        now = time.time() if now is None else now
        if now - self.last_alarm_at < cooldown:
            return False
        self.last_alarm_at = now
        self.total_alarms += 1
        return True

    def reset(self) -> None:
        self.history.clear()
        self.level = CLEAR
        self.previous_level = CLEAR


def build_trackers(cfg: dict) -> dict[str, ClassTracker]:
    """Create one tracker per enabled class from the config."""
    det = cfg["detection"]
    growth = float(det.get("growth_ratio_alert", 1.15))
    trackers: dict[str, ClassTracker] = {}
    for name in ("fire", "smoke"):
        rules = det[name]
        if not rules["enabled"]:
            continue
        trackers[name] = ClassTracker(
            name=name,
            window=int(rules["window"]),
            min_hits=int(rules["min_hits"]),
            conf=float(rules["conf"]),
            growth_alert=growth,
        )
    return trackers


def worst_level(trackers) -> str:
    levels = [t.level for t in trackers]
    if ALARM in levels:
        return ALARM
    if WATCH in levels:
        return WATCH
    return CLEAR
