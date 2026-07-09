













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


_MAX_PROMPT_CHARS = 60
_MAX_PROMPT_WORDS = 8


_session_cache: dict[str, str | None] = {}






_disk_cache: dict | None = None
_disk_lock = threading.Lock()
_session_lock = threading.Lock()
_lookup_locks: dict = {}


def _server_answered(resp) -> bool:

    return isinstance(resp, dict) and "error" not in resp


def _normalize_lookup_key(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().lower().strip("?.!,;:")


def _sanitize_token(token) -> str | None:


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

    global _disk_cache
    with _disk_lock:
        if _disk_cache is not None:
            return _disk_cache
        try:
            raw = QSettings().value(_CACHE_KEY, "", type=str)
            data = json.loads(raw) if raw and len(raw) <= 1_048_576 else {}
        except Exception:  # noqa: BLE001
            data = {}
        cache_max = _td.prompt_translate_cache_max(_CACHE_MAX)
        items = data.items() if isinstance(data, dict) else ()
        _disk_cache = {}
        max_chars = _td.prompt_translate_max_chars(_MAX_PROMPT_CHARS)
        for key, value in items:
            if not isinstance(key, str) or len(key) > max_chars:
                continue
            token = _sanitize_token(value)
            if token is not None:
                _disk_cache[key] = token
        while len(_disk_cache) > cache_max:
            _disk_cache.pop(next(iter(_disk_cache)))
        return _disk_cache


class _CacheWriter(QObject):


    @pyqtSlot()
    def write(self) -> None:
        _write_disk_cache_now()


_writer: _CacheWriter | None = None


def _write_disk_cache_now() -> None:

    with _disk_lock:
        cache = dict(_disk_cache or {})
    try:
        QSettings().setValue(_CACHE_KEY, json.dumps(cache, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def _save_disk_cache(cache: dict) -> None:






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
        with _disk_lock:
            if _writer is None:
                _writer = _CacheWriter()
                _writer.moveToThread(app.thread())


        queued = resolve_qt_enum(Qt, "ConnectionType", "QueuedConnection")
        QMetaObject.invokeMethod(_writer, "write", queued)
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def resolve_english_prompt(text: str) -> str | None:

    norm = _normalize_lookup_key(text)
    if (not norm or len(norm) > _td.prompt_translate_max_chars(_MAX_PROMPT_CHARS)
            or len(norm.split()) > _td.prompt_translate_max_words(_MAX_PROMPT_WORDS)):
        return None
    with _session_lock:
        if norm in _session_cache:
            return _session_cache[norm]
        lock, users = _lookup_locks.get(norm, (threading.Lock(), 0))
        _lookup_locks[norm] = (lock, users + 1)
    try:
        with lock:
            return _resolve_english_prompt(norm)
    finally:
        with _session_lock:
            lock, users = _lookup_locks[norm]
            if users == 1:
                del _lookup_locks[norm]
            else:
                _lookup_locks[norm] = (lock, users - 1)


def _remember_session(norm: str, token: str | None) -> None:
    with _session_lock:
        _session_cache[norm] = token
        cache_max = _td.prompt_translate_cache_max(_CACHE_MAX)
        while len(_session_cache) > cache_max:
            _session_cache.pop(next(iter(_session_cache)))


def _resolve_english_prompt(text: str) -> str | None:






    norm = _normalize_lookup_key(text)
    if not norm or len(norm) > _td.prompt_translate_max_chars(_MAX_PROMPT_CHARS):
        return None
    with _session_lock:
        if norm in _session_cache:
            return _session_cache[norm]
    disk = _load_disk_cache()
    with _disk_lock:
        cached = disk.get(norm)
    if cached is not None:
        token = _sanitize_token(cached)
        if token is not None:
            _remember_session(norm, token)
            return token
        with _disk_lock:
            disk.pop(norm, None)

    from .terralab_client import TerraLabClient




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
        token = None
    if _server_answered(resp):
        _remember_session(norm, token)
    if token:
        with _disk_lock:
            disk[norm] = token
        _save_disk_cache(disk)
    return token
