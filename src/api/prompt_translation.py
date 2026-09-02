"""Server-backed translation fallback for the Automatic prompt box.

The offline lexicon (server-delivered, read by ui/dock/prompt_guard.py)
resolves common words instantly; this module handles the long tail (any
language, rare words, or a missing lexicon) with ONE short blocking call at
commit time (Detect click), never per keystroke.

Caching keeps the cost at one round-trip per distinct word per machine:
successful translations persist in QSettings, and an answer of "nothing to
translate" is negative-cached for the session so the same word never costs a
second round trip. A call the server never answered is NOT cached: it says
nothing about the word, and caching it would run every later paid Detect on
the untranslated text until QGIS restarts. Never logs prompt text.
"""
from __future__ import annotations

import json
import re
import threading
import unicodedata

from qgis.PyQt.QtCore import QMetaObject, QObject, QSettings, Qt, QThread, pyqtSlot

from ..core import transport_dials as _td
from ..core.qt_compat import resolve_qt_enum

_CACHE_KEY = "TerraLab/AI_Segmentation/prompt_translations"
_CACHE_MAX = 300
# A prompt is a short object phrase ("building with a red roof"), not one word.
# Kept in step with the server (translate-prompt/route.ts MAX_CHARS/MAX_WORDS).
_MAX_PROMPT_CHARS = 60
_MAX_PROMPT_WORDS = 8

# norm -> token, or None for a word the server answered nothing for.
_session_cache: dict[str, str | None] = {}

# The persisted half, held in memory for the session. Every lookup runs on a
# background thread, and reading the whole blob back out of QSettings and
# re-serializing it there, once per word, is work the GUI thread's own settings
# writes have to interleave with. Loaded once, edited here, and written from
# the thread Qt wants it written on.
_disk_cache: dict | None = None
_disk_lock = threading.Lock()

# Error codes that mean the server never got to see the word: the link, not the
# answer, is what came back. Only these are kept out of the session cache; a
# code from a server that did answer (a rejection, a 5xx, a route that is not
# deployed) is a real verdict on the word and stays cached for the session.
_UNANSWERED_CODES = frozenset({
    "DNS_ERROR", "CONNECTION_REFUSED", "PROXY_ERROR", "NO_INTERNET",
    "TIMEOUT", "SSL_ERROR", "SERVICE_WARMING", "UNREADABLE_RESPONSE",
})


def _server_answered(resp) -> bool:
    """Whether ``resp`` is a verdict on the word rather than a dead link."""
    if not isinstance(resp, dict):
        return False
    code = str(resp.get("code") or "").strip().upper()
    return code not in _UNANSWERED_CODES


def _normalize_lookup_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower().strip("?.!,;:")


def _sanitize_token(token) -> str | None:
    """A usable English token: a short plain-ASCII noun phrase. Anything the
    server (or a tampered cache entry) returns beyond that is dropped."""
    if not isinstance(token, str):
        return None
    folded = (
        unicodedata.normalize("NFKD", _normalize_lookup_key(token))
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
    )
    if (not folded or len(folded) > _td.prompt_translate_max_chars(_MAX_PROMPT_CHARS)
            or len(folded.split(" ")) > _td.prompt_translate_max_words(_MAX_PROMPT_WORDS)):
        return None
    if not re.fullmatch(r"[a-z][a-z -]*", folded):
        return None
    return folded


def _load_disk_cache() -> dict:
    """The persisted translations, read once and then held for the session."""
    global _disk_cache
    with _disk_lock:
        if _disk_cache is not None:
            return _disk_cache
        try:
            raw = QSettings().value(_CACHE_KEY, "", type=str)
            data = json.loads(raw) if raw else {}
        except Exception:  # noqa: BLE001
            data = {}
        _disk_cache = data if isinstance(data, dict) else {}
        return _disk_cache


class _CacheWriter(QObject):
    """Writes the held cache on the thread Qt wants settings written on."""

    @pyqtSlot()
    def write(self) -> None:
        _write_disk_cache_now()


_writer: _CacheWriter | None = None


def _write_disk_cache_now() -> None:
    """Put the held cache into QSettings. Main thread only."""
    with _disk_lock:
        cache = dict(_disk_cache or {})
    try:
        QSettings().setValue(_CACHE_KEY, json.dumps(cache, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def _save_disk_cache(cache: dict) -> None:
    """Trim the held cache and get it written, from wherever we are.

    On the main thread it goes straight out. Off it the write is posted there
    and not waited on: nothing depends on it having landed, and a lookup must
    never block on a settings file.
    """
    with _disk_lock:
        cache_max = _td.prompt_translate_cache_max(_CACHE_MAX)
        while len(cache) > cache_max:
            cache.pop(next(iter(cache)))
    global _writer
    try:
        from qgis.core import QgsApplication

        app = QgsApplication.instance()
        if app is None or QThread.currentThread() == app.thread():
            _write_disk_cache_now()
            return
        if _writer is None:
            _writer = _CacheWriter()
            _writer.moveToThread(app.thread())
        # Only an explicit False is a refusal: this overload answers True on
        # PyQt5 and None on PyQt6.
        queued = resolve_qt_enum(Qt, "ConnectionType", "QueuedConnection")
        if QMetaObject.invokeMethod(_writer, "write", queued) is False:
            _write_disk_cache_now()
    except Exception:  # noqa: BLE001 -- a cache write must never break a lookup
        pass  # nosec B110


def resolve_english_prompt(text: str) -> str | None:
    """English token for ``text`` via the server, or None (already English,
    untranslatable, offline, or endpoint not deployed yet). Blocking: call it
    ONLY at commit time. The caller re-validates the token before use.

    A None from a dead link is not remembered, so the word is looked up again
    on the next Detect instead of running the rest of the session untranslated."""
    norm = _normalize_lookup_key(text)
    if not norm or len(norm) > _td.prompt_translate_max_chars(_MAX_PROMPT_CHARS):
        return None
    if norm in _session_cache:
        return _session_cache[norm]
    disk = _load_disk_cache()
    if norm in disk:
        token = _sanitize_token(disk[norm])
        _session_cache[norm] = token
        return token

    from .terralab_client import TerraLabClient

    # Send the activation Bearer when we have one so the call lands on the
    # per-key rate budget; unactivated users fall through to the anonymous
    # path. Best-effort: an auth-lookup failure must not block translation.
    auth = None
    try:
        from ..core.activation_manager import get_auth_header

        auth = get_auth_header() or None
    except Exception:  # noqa: BLE001
        auth = None  # nosec B110

    resp = TerraLabClient().translate_prompt(norm, auth=auth)
    token = None
    if isinstance(resp, dict) and not resp.get("error"):
        token = _sanitize_token(resp.get("token"))
    if token == norm:
        token = None  # already English: nothing to swap
    if _server_answered(resp):
        _session_cache[norm] = token
    if token:
        disk[norm] = token
        _save_disk_cache(disk)
    return token
