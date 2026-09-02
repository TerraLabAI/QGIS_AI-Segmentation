"""The plugin's own log lines for one Automatic run, shipped as one event.

A slow or broken run used to be readable only in the user's QGIS log panel,
so support meant asking them to find it, copy it and paste it. This module
listens to the "AI Segmentation" log from the Detect click to the review
ready line (or the failure, or the cancel), keeps the last 200 lines and
32 KB, redacts them, and sends them as ONE ``auto_run_log`` event through
``track()``, so the telemetry opt-out holds for it like for everything else.

Redaction, per line, before anything leaves: an absolute path keeps only its
last segment, a URL keeps its scheme and path but loses its host, an email
becomes ``[email]``. The lines themselves are the plugin's own log, which
never carries imagery, keys, or a zone geometry; that discipline is what
makes shipping them acceptable, and this module adds the belt on top.

Thread-safe: QGIS delivers a log line on the thread that wrote it.
"""
from __future__ import annotations

import re
import threading
from collections import deque
from datetime import datetime

from . import telemetry_events as ev
from .telemetry import track
from .telemetry_errors import keep_path_basenames

MAX_LINES = 200
MAX_BYTES = 32 * 1024
# How many error-time snapshots one run may send on top of its terminal one.
MAX_ERROR_SENDS_PER_RUN = 3
LOG_TAG = "AI Segmentation"

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Scheme and host, then the path if any. The path is kept inside angle
# brackets, which the path scrubber below never enters, so a route stays
# readable while the host is gone.
_URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s'\"<>]+(/[^\s'\"<>]*)?")
_LEVELS = {0: "I", 1: "W", 2: "C", 3: "S"}


def _url_without_host(match: re.Match) -> str:
    return f"<url:{match.group(1) or ''}>"


def redact_line(text: str) -> str:
    """One log line with no email, no URL host and no directory left in it."""
    if not text:
        return ""
    text = _EMAIL_RE.sub("[email]", text)
    text = _URL_RE.sub(_url_without_host, text)
    return keep_path_basenames(text)


class _RunLogCapture:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: deque[str] = deque()
        self._bytes = 0
        self._run_id = ""
        self._connected = False
        self._sent: set[str] = set()
        self.error_sends = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self, run_id: str) -> None:
        with self._lock:
            self._lines.clear()
            self._bytes = 0
            self._run_id = str(run_id or "")
            self._sent = set()
            self.error_sends = 0
        self._connect()

    def set_run_id(self, run_id: str) -> None:
        with self._lock:
            self._run_id = str(run_id or "")

    def active(self) -> bool:
        return self._connected

    def _connect(self) -> None:
        if self._connected:
            return
        try:
            from qgis.core import QgsApplication
            log = QgsApplication.messageLog()
            try:
                log.messageReceived.disconnect(self._on_message)
            except (TypeError, RuntimeError):
                pass
            log.messageReceived.connect(self._on_message)
            self._connected = True
        except Exception:  # noqa: BLE001 -- outside QGIS, nothing to listen to
            self._connected = False

    def stop(self) -> None:
        if not self._connected:
            return
        try:
            from qgis.core import QgsApplication
            QgsApplication.messageLog().messageReceived.disconnect(self._on_message)
        except (TypeError, RuntimeError, ImportError):
            pass
        self._connected = False

    # -- capture -----------------------------------------------------------

    def _on_message(self, message, tag, level) -> None:
        if tag != LOG_TAG:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            letter = _LEVELS.get(int(level), "?")
        except (TypeError, ValueError):
            letter = "?"
        with self._lock:
            for part in str(message).splitlines() or [""]:
                line = f"{stamp} {letter} {part}"
                self._lines.append(line)
                self._bytes += len(line.encode("utf-8", "ignore"))
                while self._lines and (len(self._lines) > MAX_LINES
                                       or self._bytes > MAX_BYTES):
                    gone = self._lines.popleft()
                    self._bytes -= len(gone.encode("utf-8", "ignore"))

    def snapshot(self) -> tuple[str, list[str]]:
        with self._lock:
            return self._run_id, [redact_line(line) for line in self._lines]

    # -- send --------------------------------------------------------------

    def send(self, reason: str, final: bool) -> bool:
        """Ship the captured lines once per (run, reason). A final send ends
        the capture. Returns whether an event went out."""
        run_id, lines = self.snapshot()
        key = f"{run_id}:{reason}"
        with self._lock:
            sent = bool(lines and run_id and key not in self._sent)
            if sent:
                self._sent.add(key)
        if sent:
            track(ev.AUTO_RUN_LOG, {
                "run_id": run_id,
                "reason": str(reason)[:64],
                "line_count": len(lines),
                "lines": lines,
            })
        if final:
            self.stop()
        return sent


_capture = _RunLogCapture()


def start_run_log(run_id: str) -> None:
    """Begin capturing at the Detect click. Idempotent per run."""
    try:
        _capture.start(run_id)
    except Exception:  # noqa: BLE001 -- a lost log is not a lost run  # nosec B110
        pass


def note_run_id(run_id: str) -> None:
    """The run id is minted after the click; attach it once known. A run
    that never went through the click (a headless one) starts its capture
    here instead, so its log still ships."""
    try:
        if _capture.active():
            _capture.set_run_id(run_id)
        else:
            _capture.start(run_id)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def send_run_log(reason: str) -> None:
    """The run's terminal: completed, failed or cancelled. Ends the capture."""
    try:
        _capture.send(reason, final=True)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def send_run_log_on_error(error_code: str) -> None:
    """A tracked plugin error mid-run ships the lines so far, capped per run.
    Called by track_plugin_error; a no-op outside a run."""
    try:
        if not _capture.active() or _capture.error_sends >= MAX_ERROR_SENDS_PER_RUN:
            return
        if _capture.send(f"error:{error_code}", final=False):
            _capture.error_sends += 1
    except Exception:  # noqa: BLE001  # nosec B110
        pass
