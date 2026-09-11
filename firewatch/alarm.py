"""
Making noise, and telling other systems about it.

Sound runs on a background thread so a sounding siren never stalls frame
processing - the detector must keep watching while the alarm is going off.

Windows uses winsound, which ships with Python and needs no extra packages.
Linux and macOS fall back to aplay / afplay, and then to the terminal bell,
so the project still runs if you develop on something other than Windows.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("firewatch.alarm")

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    try:
        import winsound
    except ImportError:  # pragma: no cover - Windows always has it
        winsound = None
else:
    winsound = None


class AlarmPlayer:
    """Plays the alarm sound, stops it on time, and can be acknowledged."""

    def __init__(self, cfg: dict, wav_path: Path | None):
        acfg = cfg["alarm"]
        self.enabled: bool = bool(acfg["sound_enabled"])
        self.mode: str = str(acfg["sound_mode"])
        self.wav_path = wav_path
        self.freq = int(acfg["beep_freq_hz"])
        self.beep_ms = int(acfg["beep_duration_ms"])
        self.beep_repeat = int(acfg["beep_repeat"])
        self.duration = float(acfg["duration_sec"])
        self.allow_ack = bool(acfg["allow_acknowledge"])

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        if self.enabled and self.mode == "wav":
            if self.wav_path is None or not self.wav_path.exists():
                log.warning(
                    "alarm.wav_file not found (%s). Falling back to beeps.",
                    self.wav_path,
                )
                self.mode = "beep"

        if self.enabled and not IS_WINDOWS and self.mode == "beep":
            log.warning(
                "alarm.sound_mode 'beep' uses winsound and only works on "
                "Windows. Using the terminal bell instead."
            )

    # ------------------------------------------------------------------ state
    @property
    def is_sounding(self) -> bool:
        # The worker thread can take a moment to notice the stop flag (it is
        # sleeping between beeps), so check the flag too. Otherwise the preview
        # keeps showing "SIREN SOUNDING" for a second after the operator
        # acknowledged it, which looks broken.
        if self._stop.is_set():
            return False
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # ---------------------------------------------------------------- control
    def trigger(self) -> None:
        """Start the alarm sound. No-op if disabled or already sounding."""
        if not self.enabled:
            return
        if self.is_sounding:
            return
        self._stop.clear()
        with self._lock:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def acknowledge(self) -> bool:
        """Silence a sounding alarm. Returns True if something was silenced."""
        if not self.allow_ack or not self.is_sounding:
            return False
        log.info("Alarm acknowledged by operator.")
        self.stop()
        return True

    def stop(self) -> None:
        self._stop.set()
        self._silence_platform()

    def shutdown(self) -> None:
        self.stop()
        with self._lock:
            t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)

    # --------------------------------------------------------------- internal
    def _run(self) -> None:
        deadline = time.time() + self.duration
        try:
            if self.mode == "wav" and IS_WINDOWS and winsound is not None:
                # SND_LOOP keeps playing until we purge it.
                winsound.PlaySound(
                    str(self.wav_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
                )
                while time.time() < deadline and not self._stop.is_set():
                    time.sleep(0.1)
                winsound.PlaySound(None, winsound.SND_PURGE)

            elif self.mode == "wav":
                self._loop_unix_wav(deadline)

            elif IS_WINDOWS and winsound is not None:
                while time.time() < deadline and not self._stop.is_set():
                    for _ in range(self.beep_repeat):
                        if self._stop.is_set() or time.time() >= deadline:
                            break
                        winsound.Beep(self.freq, self.beep_ms)
                        time.sleep(0.08)
                    time.sleep(0.3)

            else:
                while time.time() < deadline and not self._stop.is_set():
                    print("\a", end="", flush=True)
                    time.sleep(0.5)

        except Exception as e:  # noqa: BLE001 - never let audio kill detection
            log.error("Alarm sound failed: %s", e)

    def _loop_unix_wav(self, deadline: float) -> None:
        player = shutil.which("aplay") or shutil.which("afplay")
        if player is None:
            while time.time() < deadline and not self._stop.is_set():
                print("\a", end="", flush=True)
                time.sleep(0.5)
            return
        while time.time() < deadline and not self._stop.is_set():
            proc = subprocess.Popen(
                [player, str(self.wav_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while proc.poll() is None:
                if self._stop.is_set() or time.time() >= deadline:
                    proc.terminate()
                    break
                time.sleep(0.1)

    @staticmethod
    def _silence_platform() -> None:
        if IS_WINDOWS and winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:  # noqa: BLE001
                pass


def run_hook(command: str, cls: str, snapshot: str, conf: float) -> None:
    """
    Fire an external program. Called as:  <command> <class> <snapshot> <conf>

    Use it for Telegram, ntfy, WhatsApp, SMS, or flipping a GPIO relay.
    Non-blocking, and a broken hook must never take the detector down.
    """
    if not command.strip():
        return
    try:
        subprocess.Popen(
            [command, cls, snapshot, f"{conf:.3f}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("Hook dispatched: %s", command)
    except OSError as e:
        log.error("Hook failed (%s): %s", command, e)
