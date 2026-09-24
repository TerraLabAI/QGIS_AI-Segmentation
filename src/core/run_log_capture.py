
















from __future__ import annotations

import re
import threading
from collections import deque
from datetime import datetime

from . import telemetry_events as ev
from .log_scrub import bounded_log_tail, scrub_sensitive
from .telemetry import track
from .telemetry_errors import keep_path_basenames

MAX_LINES = 200
MAX_BYTES = 32 * 1024

MAX_ERROR_SENDS_PER_RUN = 3
LOG_TAG = "AI Segmentation"

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")



_URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s'\"<>]+(/[^\s'\"<>]*)?")
_LEVELS = {0: "I", 1: "W", 2: "C", 3: "S"}


def _url_without_host(match: re.Match) -> str:
    path = (match.group(1) or "").split("?", 1)[0].split("#", 1)[0]
    return f"<url:{path}>"


def redact_line(text: str) -> str:

    if not text:
        return ""
    text = _EMAIL_RE.sub("[email]", text)
    text = _URL_RE.sub(_url_without_host, text)
    return scrub_sensitive(keep_path_basenames(text))


class _RunLogCapture:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: deque[str] = deque()
        self._bytes = 0
        self._run_id = ""
        self._connected = False
        self._sent: set[str] = set()
        self.error_sends = 0



    def start(self, run_id: str) -> None:
        with self._lock:
            if run_id and self._connected and self._run_id == str(run_id):
                return
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
        except Exception:  # noqa: BLE001
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



    def _on_message(self, message, tag, level) -> None:
        try:
            self._capture_message(message, tag, level)
        except Exception:  # noqa: BLE001
            return

    def _capture_message(self, message, tag, level) -> None:
        if tag != LOG_TAG:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            letter = _LEVELS.get(int(level), "?")
        except (TypeError, ValueError):
            letter = "?"
        from .server_dials import dial_in_range
        max_lines = dial_in_range("tuning.notify.run_log_max_lines", MAX_LINES, 20, 2000)

        max_bytes = dial_in_range("tuning.notify.run_log_max_bytes", MAX_BYTES, 4096, 65536)
        bounded = bounded_log_tail(str(message), max_bytes)
        lines = [redact_line(part) for part in bounded.splitlines()]
        with self._lock:
            for part in lines or [""]:
                line = f"{stamp} {letter} {part}"
                self._lines.append(line)
                self._bytes += len(line.encode("utf-8", "ignore"))
                while self._lines and (len(self._lines) > max_lines
                                       or self._bytes > max_bytes):
                    gone = self._lines.popleft()
                    self._bytes -= len(gone.encode("utf-8", "ignore"))

    def snapshot(self) -> tuple[str, list[str]]:
        with self._lock:
            return self._run_id, list(self._lines)



    def send(self, reason: str, final: bool) -> bool:


        run_id, lines = self.snapshot()
        key = f"{run_id}:{reason}"
        with self._lock:
            sent = bool(lines and run_id and key not in self._sent)
            if sent:
                self._sent.add(key)
        try:
            if sent:
                track(ev.AUTO_RUN_LOG, {
                    "run_id": run_id,
                    "reason": str(reason)[:64],
                    "line_count": len(lines),
                    "lines": lines,
                })
        finally:
            if final:
                self.stop()
        return sent


_capture = _RunLogCapture()


def start_run_log(run_id: str) -> None:

    try:
        _capture.start(run_id)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def note_run_id(run_id: str) -> None:



    try:
        if _capture.active():
            _capture.set_run_id(run_id)
        else:
            _capture.start(run_id)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def send_run_log(reason: str) -> None:

    try:
        _capture.send(reason, final=True)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def send_run_log_on_error(error_code: str) -> None:


    try:
        from .server_dials import dial_in_range
        max_error_sends = dial_in_range(
            "tuning.notify.run_log_max_error_sends", MAX_ERROR_SENDS_PER_RUN, 0, 20)
        if not _capture.active() or _capture.error_sends >= max_error_sends:
            return
        if _capture.send(f"error:{error_code}", final=False):
            _capture.error_sends += 1
    except Exception:  # noqa: BLE001  # nosec B110
        pass
