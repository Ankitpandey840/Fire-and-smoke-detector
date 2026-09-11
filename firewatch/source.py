"""
Video input with the awkward parts handled.

Three things this deals with that a bare cv2.VideoCapture does not:

1. RTSP streams drop. Network blips, camera reboots, NVR restarts. A system
   meant to run for weeks has to reconnect by itself.
2. RTSP over UDP loses packets and produces smeared, half-drawn frames that
   wreck detection. We force TCP.
3. On Windows, the default capture backend takes several seconds to open a
   webcam. DirectShow opens it immediately.
"""

from __future__ import annotations

import logging
import os
import platform
import time

import cv2

log = logging.getLogger("firewatch.source")

IS_WINDOWS = platform.system() == "Windows"


class VideoSource:
    """A reconnecting video source that yields frames at a capped rate."""

    def __init__(self, cfg: dict):
        src = cfg["source"]
        self.kind: str = src["kind"]
        self.rtsp_use_tcp: bool = bool(src["rtsp_use_tcp"])
        self.reconnect_delay: float = float(src["reconnect_delay_sec"])
        self.max_attempts: int = int(src["max_reconnect_attempts"])

        if self.kind == "webcam":
            self.target = int(src["webcam_index"])
            self.label = f"webcam {self.target}"
        elif self.kind == "rtsp":
            self.target = str(src["rtsp_url"])
            self.label = self._redact(self.target)
        else:
            self.target = str(src["file_path"])
            self.label = self.target

        fps = float(src["target_fps"])
        self.min_interval = 1.0 / fps if fps > 0 else 0.0

        self.cap: cv2.VideoCapture | None = None
        self.attempts = 0
        self._last_yield = 0.0
        self.frames_read = 0
        self.reconnects = 0

    @staticmethod
    def _redact(url: str) -> str:
        """Hide the password so it never reaches a log file."""
        if "@" not in url or "://" not in url:
            return url
        scheme, rest = url.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0] if ":" in creds else creds
        return f"{scheme}://{user}:****@{host}"

    def open(self) -> bool:
        self.close()

        if self.kind == "rtsp" and self.rtsp_use_tcp:
            # Must be set before the capture is constructed.
            existing = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
            if "rtsp_transport" not in existing:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        if self.kind == "webcam" and IS_WINDOWS:
            self.cap = cv2.VideoCapture(self.target, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.target)

        if not self.cap.isOpened():
            return False

        if self.kind == "rtsp":
            # Keep the decoder buffer tiny. A big buffer means you are looking
            # at footage from ten seconds ago, which is useless for an alarm.
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Opened %s  (%dx%d)", self.label, w, h)
        self.attempts = 0
        return True

    def connect(self) -> bool:
        """Open, retrying until the attempt limit is reached."""
        while True:
            if self.open():
                return True
            self.attempts += 1
            if self.max_attempts and self.attempts >= self.max_attempts:
                log.error("Giving up on %s after %d attempts", self.label, self.attempts)
                return False
            log.warning(
                "Cannot open %s (attempt %d). Retrying in %.0fs ...",
                self.label, self.attempts, self.reconnect_delay,
            )
            time.sleep(self.reconnect_delay)

    def frames(self):
        """Yield frames forever, reconnecting as needed. Stops at end of file."""
        if not self.connect():
            return

        while True:
            ok, frame = self.cap.read()

            if not ok:
                if self.kind == "file":
                    log.info("End of file reached.")
                    return
                log.warning("Stream dropped. Reconnecting in %.0fs ...",
                            self.reconnect_delay)
                time.sleep(self.reconnect_delay)
                self.reconnects += 1
                if not self.connect():
                    return
                continue

            self.frames_read += 1

            # Rate limit. For live sources we simply skip frames; for files we
            # sleep so playback stays watchable.
            if self.min_interval > 0:
                now = time.time()
                elapsed = now - self._last_yield
                if elapsed < self.min_interval:
                    if self.kind == "file":
                        time.sleep(self.min_interval - elapsed)
                    else:
                        continue
                self._last_yield = time.time()

            yield frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
