"""Path anonymization + in-memory log capture, shared by telemetry and the UI.

This lives in core on purpose: core/telemetry.py scrubs every outgoing payload
with anonymize_paths, so the scrubber must never depend on a UI module (a UI
rename or refactor must not be able to silently disable PII scrubbing). The
error report dialog imports the same helpers from here.

Import-time dependencies are stdlib-only; the QGIS bindings are touched only
inside start/stop_log_collector, lazily.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from collections import deque
from datetime import datetime

# In-memory log buffer - captures AI Segmentation messages
_log_buffer: deque[str] = deque(maxlen=100)
_log_buffer_lock = threading.Lock()
_log_collector_connected = False


# Literal home patterns, built once. The /Users and /home shapes below only
# catch a home directory that sits where the OS puts it by default. A corporate
# profile redirected to another drive or another root carries the account name
# just as plainly and matches none of them, so the home directory is replaced
# by its own name first. The plugin cache root follows it, because an
# environment variable can move that one anywhere too.
_literal_home_res: list[re.Pattern[str]] | None = None


def _literal_home_patterns() -> list[re.Pattern[str]]:
    global _literal_home_res
    if _literal_home_res is not None:
        return _literal_home_res
    flags = re.IGNORECASE if sys.platform == "win32" else 0
    roots: list[str] = []
    try:
        roots.append(os.path.expanduser("~"))
    except Exception:
        pass  # nosec B110 - the shape patterns below still run
    try:
        from .cache_paths import PLUGIN_CACHE_DIR
        roots.append(PLUGIN_CACHE_DIR)
    except Exception:
        pass  # nosec B110 - optional, the home root carries the usual case
    patterns: list[re.Pattern[str]] = []
    seen: set[str] = set()
    for root in roots:
        root = (root or "").rstrip("/\\")
        # A one-character root would match half the text; "/" or "C:" is not a
        # home directory worth hiding.
        if len(root) < 4:
            continue
        for spelling in (root, root.replace("\\", "/")):
            key = spelling.lower() if flags else spelling
            if key in seen:
                continue
            seen.add(key)
            patterns.append(re.compile(re.escape(spelling), flags))
    _literal_home_res = patterns
    return patterns


def anonymize_paths(text: str) -> str:
    """
    Anonymize file paths to hide the username.

    Replaces patterns like:
    - /Users/<username>/... -> <USER>/...
    - /home/<username>/... -> <USER>/...
    - C:\\Users\\<username>\\... -> <USER>/...

    This protects user privacy when sending logs and reports.
    """
    if not text:
        return text

    # The literal home and cache roots go first, so a redirected profile is
    # covered before the shape patterns get their turn.
    for pattern in _literal_home_patterns():
        text = pattern.sub("<USER>", text)

    # The two POSIX patterns stay case-SENSITIVE. /Users and /home are fixed
    # case on disk and os.path.normcase is the identity there, so nothing
    # produces another spelling, while a case-insensitive match would eat
    # ordinary content: an API path like "/api/v1/users/me", a route, or a
    # /vsicurl/ raster source with "users" in it.

    # macOS: /Users/<username>/... (with or without trailing slash)
    text = re.sub(r"/Users/[^/\s]+(?=/|$|\s)", "<USER>", text)

    # Linux: /home/<username>/... (with or without trailing slash)
    text = re.sub(r"/home/[^/\s]+(?=/|$|\s)", "<USER>", text)

    # The two Windows patterns below are case-INSENSITIVE, because there the
    # text really does arrive in several spellings: a package manager prints
    # "c:\users\...", os.path.normcase lowercases a whole path, and the 8.3
    # short form comes back uppercase. A drive-letter path cannot be confused
    # with ordinary prose, so widening the match there costs nothing.

    # Windows: C:\Users\<username>\... (and other drive letters, forward/back slashes)
    text = re.sub(r"[A-Za-z]:[/\\]Users[/\\][^/\\\s]+(?=[/\\]|$|\s)", "<USER>", text,
                  flags=re.IGNORECASE)

    # Windows UNC paths: \\server\Users\<username>\...
    return re.sub(r"\\\\[^\\]+\\Users\\[^/\\\s]+(?=[/\\]|$|\s)", "<USER>", text,
                  flags=re.IGNORECASE)


# Generic redaction patterns. These match by SHAPE, never by literal name, so
# this file (which ships in the public repo) can strip backend URLs, hosts,
# auth headers, model-weight filenames and tokens out of a report WITHOUT
# spelling any of them out. A network error to the backend is the main way an
# endpoint URL would otherwise reach a user-visible report.
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
# Bare FQDNs on TLDs our infra can use, kept narrow so ordinary text survives:
# ".app"/".dev" are excluded on purpose (they would hit macOS bundles like
# QGIS.app in tracebacks); a scheme-bearing backend URL is already caught by
# _URL_RE above. Runs AFTER _URL_RE.
_HOST_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:ai|io|cloud)\b",
    re.IGNORECASE,
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Any X-*-Auth style header value, and bearer/token/api-key assignments. Runs
# to end of line: the bare wire form "Authorization: Bearer <key>" is two
# tokens, so consuming only one left the key sitting in the report.
_AUTH_RE = re.compile(
    r"(?i)\b(?:bearer|authorization|api[_-]?key|x-[\w-]*auth[\w-]*)\b"
    r"\s*[:=]?\s*[\"']?[^\r\n]*",
)
# The activation key by shape. Nothing writes one to a log today, and the
# checks that keep it out of URLs and redirects are elsewhere. This is the net
# under all of them, so a future caller cannot leak one into a support email.
_KEY_RE = re.compile(r"(?i)\btl_[0-9a-f]{16,}\b")
_WEIGHTS_RE = re.compile(
    r"(?i)\b[\w.\-]+\.(?:pth|pt|onnx|ckpt|safetensors)\b",
)


def scrub_sensitive(text: str) -> str:
    """Redact backend/infra and model identifiers by SHAPE (URLs, cloud hosts,
    IPs, auth headers, weight filenames). Pattern-based on purpose: no literal
    infra or model name lives in this public file. Leaves ordinary diagnostic
    text (versions, counts, CRS, run-id) intact."""
    if not text:
        return text
    text = _URL_RE.sub("<url>", text)
    text = _AUTH_RE.sub("<auth>", text)
    text = _KEY_RE.sub("<key>", text)
    text = _WEIGHTS_RE.sub("<weights>", text)
    text = _HOST_RE.sub("<host>", text)
    return _IPV4_RE.sub("<ip>", text)


def scrub_report(text: str) -> str:
    """Full report scrub: user paths first, then sensitive infra/model shapes.
    The single entry point the error report and any log export should use."""
    return scrub_sensitive(anonymize_paths(text))


def start_log_collector():
    """Connect to QgsMessageLog to capture AI Segmentation messages.
    Call once at plugin startup."""
    global _log_collector_connected
    with _log_buffer_lock:
        if _log_collector_connected:
            return
        try:
            from qgis.core import QgsApplication
            log = QgsApplication.messageLog()
            # The flag is a module global, so a reload reads False while the
            # old connection is still live on the same signal. Qt would then
            # hold two, and every line would be buffered twice for the rest of
            # the session. Dropping first makes this connect idempotent whatever
            # the flag says; a disconnect with nothing to drop raises TypeError.
            try:
                log.messageReceived.disconnect(_on_log_message)
            except (TypeError, RuntimeError):
                pass
            log.messageReceived.connect(_on_log_message)
            _log_collector_connected = True
        except Exception:
            pass  # nosec B110


def stop_log_collector():
    """Disconnect from QgsMessageLog. Call on plugin unload."""
    global _log_collector_connected
    with _log_buffer_lock:
        if not _log_collector_connected:
            return
        try:
            from qgis.core import QgsApplication
            QgsApplication.messageLog().messageReceived.disconnect(_on_log_message)
        except (TypeError, RuntimeError):
            pass
        _log_collector_connected = False


def _on_log_message(message, tag, level):
    """Callback for QgsMessageLog.messageReceived signal."""
    if tag == "AI Segmentation":
        timestamp = datetime.now().strftime("%H:%M:%S")
        with _log_buffer_lock:
            _log_buffer.append(f"[{timestamp}] {message}")


def get_recent_logs() -> str:
    """Return recent AI Segmentation log lines (paths anonymized AND infra/model
    shapes scrubbed)."""
    with _log_buffer_lock:
        if not _log_buffer:
            return "(No logs captured this session)"
        logs = "\n".join(_log_buffer)
    return scrub_report(logs)
