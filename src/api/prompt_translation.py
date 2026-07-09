"""Server-backed translation fallback for the Automatic prompt box.

The offline lexicon (server-delivered, read by ui/dock/prompt_guard.py)
resolves common words instantly; this module handles the long tail (any
language, rare words, or a missing lexicon) with ONE short blocking call at
commit time (Detect click), never per keystroke.

Caching keeps the cost at one round-trip per distinct word per machine:
successful translations persist in QSettings, and failures (offline, server
down, untranslatable) are negative-cached for the session so a broken
connection never adds repeated latency to Detect. Never logs prompt text.
"""
from __future__ import annotations

import json
import re
import unicodedata

from qgis.PyQt.QtCore import QSettings

_CACHE_KEY = "TerraLab/AI_Segmentation/prompt_translations"
_CACHE_MAX = 300
_MAX_PROMPT_CHARS = 40

# norm -> token, or None for a lookup that already failed this session.
_session_cache: dict[str, str | None] = {}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower().strip("?.!,;:")


def _sanitize_token(token) -> str | None:
    """A usable English token: a short plain-ASCII noun phrase. Anything the
    server (or a tampered cache entry) returns beyond that is dropped."""
    if not isinstance(token, str):
        return None
    folded = (
        unicodedata.normalize("NFKD", _normalize(token))
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
    )
    if not folded or len(folded) > 30 or len(folded.split(" ")) > 3:
        return None
    if not re.fullmatch(r"[a-z][a-z -]*", folded):
        return None
    return folded


def _load_disk_cache() -> dict:
    try:
        raw = QSettings().value(_CACHE_KEY, "", type=str)
        data = json.loads(raw) if raw else {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _save_disk_cache(cache: dict) -> None:
    while len(cache) > _CACHE_MAX:
        cache.pop(next(iter(cache)))
    try:
        QSettings().setValue(_CACHE_KEY, json.dumps(cache, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def resolve_english_prompt(text: str) -> str | None:
    """English token for ``text`` via the server, or None (already English,
    untranslatable, offline, or endpoint not deployed yet). Blocking: call it
    ONLY at commit time. The caller re-validates the token before use."""
    norm = _normalize(text)
    if not norm or len(norm) > _MAX_PROMPT_CHARS:
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
    _session_cache[norm] = token
    if token:
        disk[norm] = token
        _save_disk_cache(disk)
    return token
