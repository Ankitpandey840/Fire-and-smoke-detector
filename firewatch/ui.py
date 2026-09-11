"""Preview window drawing."""

from __future__ import annotations

import cv2

from .tracker import ALARM, CLEAR, WATCH, worst_level

# BGR
LEVEL_COLOR = {
    CLEAR: (120, 200, 120),
    WATCH: (60, 190, 245),
    ALARM: (60, 60, 245),
}
CLASS_COLOR = {
    "fire": (60, 90, 245),
    "smoke": (200, 200, 200),
}
PANEL_BG = (28, 28, 28)
TEXT = (225, 225, 225)
DIM = (140, 140, 140)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_boxes(frame, detections, show_labels: bool):
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        color = CLASS_COLOR.get(d.cls, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            label = f"{d.cls} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)),
                          (x1 + tw + 8, max(y1, th + 8)), color, -1)
            cv2.putText(frame, label, (x1 + 4, max(y1 - 5, th + 3)),
                        FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def draw_zones(frame, zones):
    h, w = frame.shape[:2]
    for z in zones:
        p1 = (int(z.x1 * w), int(z.y1 * h))
        p2 = (int(z.x2 * w), int(z.y2 * h))
        cv2.rectangle(frame, p1, p2, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ignore: {z.name}", (p1[0] + 5, p1[1] + 18),
                    FONT, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return frame


def draw_panel(frame, trackers, fps: float, sounding: bool, paused: bool):
    h, w = frame.shape[:2]
    rows = len(trackers)
    panel_w = 330
    panel_h = 46 + 30 * rows + (26 if sounding else 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), PANEL_BG, -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    level = worst_level(trackers)
    status = "PAUSED" if paused else level
    color = DIM if paused else LEVEL_COLOR[level]
    cv2.putText(frame, status, (14, 30), FONT, 0.8, color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:4.1f} fps", (panel_w - 78, 28),
                FONT, 0.45, DIM, 1, cv2.LINE_AA)

    y = 58
    for t in trackers:
        cv2.putText(frame, t.name, (14, y + 5), FONT, 0.5, TEXT, 1, cv2.LINE_AA)

        bar_x, bar_w = 82, 170
        cv2.rectangle(frame, (bar_x, y - 9), (bar_x + bar_w, y + 5), (62, 62, 62), -1)

        # Threshold marker: where min_hits sits on the bar.
        mark = bar_x + int(bar_w * t.min_hits / t.window)
        filled = int(bar_w * t.hit_count / t.window)
        if filled:
            cv2.rectangle(frame, (bar_x, y - 9), (bar_x + filled, y + 5),
                          LEVEL_COLOR[t.level], -1)
        cv2.line(frame, (mark, y - 11), (mark, y + 7), (255, 255, 255), 1)

        cv2.putText(frame, f"{t.hit_count}/{t.window}", (bar_x + bar_w + 8, y + 5),
                    FONT, 0.45, DIM, 1, cv2.LINE_AA)
        y += 30

    if sounding:
        cv2.putText(frame, "SIREN SOUNDING - press [A] to silence", (14, y + 4),
                    FONT, 0.45, LEVEL_COLOR[ALARM], 1, cv2.LINE_AA)

    if level == ALARM and not paused:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), LEVEL_COLOR[ALARM], 6)
    return frame


def draw_hint(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, "Q quit   A silence   S snapshot   P pause",
                (14, h - 14), FONT, 0.45, DIM, 1, cv2.LINE_AA)
    return frame


def scaled(frame, scale: float):
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)
