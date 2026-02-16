










from __future__ import annotations

import os
import re
import sys
import threading
from collections import deque
from datetime import datetime



_LOG_BUFFER_LINES = 300
_COPIED_REPORT_LINES = 100
_log_buffer: deque[str] = deque(maxlen=_LOG_BUFFER_LINES)
_LOG_MESSAGE_MAX_CHARS = 4096
_log_buffer_lock = threading.Lock()
_log_collector_connected = False








_literal_home_memo: dict[str, list[re.Pattern[str]]] = {}


def _literal_home_patterns() -> list[re.Pattern[str]]:
    cached = _literal_home_memo.get("patterns")
    if cached is not None:
        return cached
    flags = re.IGNORECASE if sys.platform == "win32" else 0
    roots: list[str] = []
    try:
        roots.append(os.path.expanduser("~"))
    except Exception:
        pass  # nosec B110
    try:
        from .cache_paths import PLUGIN_CACHE_DIR
        roots.append(PLUGIN_CACHE_DIR)
    except Exception:
        pass  # nosec B110
    patterns: list[re.Pattern[str]] = []
    seen: set[str] = set()
    for root in roots:
        root = (root or "").rstrip("/\\")


        if len(root) < 4:
            continue


        for spelling in (root, root.replace("\\", "/"),
                         root.replace("\\", "\\\\")):
            key = spelling.lower() if flags else spelling
            if key in seen:
                continue
            seen.add(key)
            patterns.append(re.compile(re.escape(spelling), flags))
    _literal_home_memo["patterns"] = patterns
    return patterns








_POSIX_ACCOUNT = r"(?:[^/\s]+(?: [^/\s:]+){0,3}(?=/)|[^/\s]+(?=/|$|\s))"
_WINDOWS_ACCOUNT = (r"(?:[^/\\\s]+(?: [^/\\\s:]+){0,3}(?=[/\\])"
                    r"|[^/\\\s]+(?=[/\\]|$|\s))")


def anonymize_paths(text: str) -> str:










    if not text:
        return text



    for pattern in _literal_home_patterns():
        text = pattern.sub("<USER>", text)








    text = re.sub(r"/Users/" + _POSIX_ACCOUNT, "<USER>", text)


    text = re.sub(r"/home/" + _POSIX_ACCOUNT, "<USER>", text)












    if ":" in text:
        text = re.sub(r"[A-Za-z]:[/\\]+Users[/\\]+" + _WINDOWS_ACCOUNT, "<USER>", text,
                      flags=re.IGNORECASE)
    if "\\\\" not in text:
        return text






    return re.sub(r"(?<!\\)\\{2,}(?:[?.]\\+(?:UNC\\+)?)?[^\\]+\\+Users\\+" + _WINDOWS_ACCOUNT,
                  "<USER>", text, flags=re.IGNORECASE)







_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)










_HOST_RE = re.compile(
    r"(?<![\w.-])[\w.-]*[a-z0-9]\.(?:ai|io|cloud)\b",
    re.IGNORECASE,
)


_IPV4_RE = re.compile(r"\d(?<!\w\d)\d{0,2}(?:\.\d{1,3}){3}\b")



_AUTH_RE = re.compile(
    r"(?i)(?:\b(?:bearer|authorization|api[_-]?key)|(?<![\w-])x-[\w-]*auth[\w-]*"
    r"|\b(?:password|access[_-]?token|refresh[_-]?token|client[_-]?secret|cookie))\b"
    r"[\"']?\s*[:=]?\s*[\"']?[^\r\n]*",
)
_REPORT_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")



_KEY_RE = re.compile(r"(?i)\btl_[0-9a-f]{16,}\b")
_WEIGHTS_RE = re.compile(
    r"(?i)(?<![\w.-])[\w.\-]+\.(?:pth|pt|onnx|ckpt|safetensors)\b",
)




_AUTH_HINTS = ("bearer", "authorization", "api", "x-", "password", "token",
               "secret", "cookie")
_WEIGHTS_HINT_RE = re.compile(r"(?i)\.(?:pth|pt|onnx|ckpt|safetensors)\b")
_HOST_HINT_RE = re.compile(r"(?i)\.(?:ai|io|cloud)\b")


def scrub_sensitive(text: str) -> str:




    if not text:
        return text



    if "://" in text:
        text = _URL_RE.sub("<url>", text)
    lowered = text.lower()
    if any(word in lowered for word in _AUTH_HINTS):
        text = _AUTH_RE.sub("<auth>", text)
    if "tl_" in lowered:
        text = _KEY_RE.sub("<key>", text)
    if _WEIGHTS_HINT_RE.search(text):
        text = _WEIGHTS_RE.sub("<weights>", text)



    if "@" in text:
        text = _REPORT_EMAIL_RE.sub("<email>", text)
    if _HOST_HINT_RE.search(text):
        text = _HOST_RE.sub("<host>", text)
    return _IPV4_RE.sub("<ip>", text)


def bounded_log_tail(text: str, max_chars: int) -> str:

    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]

    _, separator, complete_lines = tail.partition("\n")
    return complete_lines if separator else "<truncated>"


def scrub_report(text: str) -> str:


    return scrub_sensitive(anonymize_paths(text))


def start_log_collector():


    global _log_collector_connected
    with _log_buffer_lock:
        if _log_collector_connected:
            return
        try:
            from qgis.core import QgsApplication
            log = QgsApplication.messageLog()





            try:
                log.messageReceived.disconnect(_on_log_message)
            except (TypeError, RuntimeError):
                pass
            log.messageReceived.connect(_on_log_message)
            _log_collector_connected = True
        except Exception:
            pass  # nosec B110


def stop_log_collector():

    global _log_collector_connected
    with _log_buffer_lock:
        if not _log_collector_connected:
            return
        try:
            from qgis.core import QgsApplication
            QgsApplication.messageLog().messageReceived.disconnect(_on_log_message)
        except (TypeError, RuntimeError, ImportError):
            pass
        _log_collector_connected = False


def _on_log_message(message, tag, level):

    if tag == "AI Segmentation":
        timestamp = datetime.now().strftime("%H:%M:%S")

        message = scrub_report(bounded_log_tail(str(message), _LOG_MESSAGE_MAX_CHARS))
        with _log_buffer_lock:
            _log_buffer.append(f"[{timestamp}] {message}")


def get_recent_logs() -> str:


    with _log_buffer_lock:
        if not _log_buffer:
            return "(No logs captured this session)"
        logs = "\n".join(list(_log_buffer)[-_COPIED_REPORT_LINES:])
    return scrub_report(logs)


def get_recent_log_lines(limit: int = _LOG_BUFFER_LINES) -> list[str]:


    with _log_buffer_lock:
        lines = list(_log_buffer)
    return lines[-max(0, int(limit)):] if limit else []
