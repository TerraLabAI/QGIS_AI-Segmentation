"""Local 'recently detected' objects, persisted via QSettings.

Records the object (the English cloud-model token) of each committed Automatic
detection run so the Segment library can offer a Recent tab to re-run it. The
list is local-only - nothing leaves the machine.

One bucket per account, keyed by :func:`account_fingerprint`, plus a bucket for
the signed-out state. A signed-out user keeps seeing the objects they detected
signed out, and signing in to another account never surfaces the previous
account's list.

No favorites: segmentation recall is about quickly reusing the last objects you
actually detected, not curating a starred set. This is the trimmed sibling of AI Edit's ``prompt_history`` (Recent half
only, with segmentation-flavoured per-entry stats).
"""
from __future__ import annotations

import json
import time

from qgis.PyQt.QtCore import QSettings

from .run_history_cache import account_fingerprint

# Written by versions that kept one list for the whole machine, whichever
# account was signed in. Never read again, and dropped on every write and on
# every sign-in / sign-out.
_UNSCOPED_RECENT_KEY = "AISegmentation/recent_objects"
_RECENT_KEY = "AISegmentation/recent_objects_by_account"

# Bucket name for the signed-out state. Not a fingerprint, and no fingerprint
# can collide with it: fingerprints are lowercase hex.
_SIGNED_OUT_BUCKET = "signed_out"

# Cap on stored entries. The list is serialised as one JSON blob and rewritten
# on every committed run, so an uncapped list would bloat settings I/O. 200 is
# well past anyone's "recently used" memory and still loads instantly.
_RECENT_CAP = 200


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_dedupe_key(prompt: str) -> str:
    """Dedupe key. Whitespace-trim only; case is preserved."""
    return (prompt or "").strip()


def _recent_bucket_name() -> str:
    """Which stored list the current session reads and writes."""
    return account_fingerprint() or _SIGNED_OUT_BUCKET


def _read_buckets() -> dict:
    """All stored buckets, or {} when empty/corrupt."""
    raw = QSettings().value(_RECENT_KEY, "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, list)}


def get_recent() -> list[dict]:
    """Return Recent objects for the current account, newest first.

    Each entry: ``{prompt, ts, detections?, detail?}`` where ``prompt`` is the
    English token, ``detections`` the count exported by that run, ``detail`` the
    grid level used."""
    entries = _read_buckets().get(_recent_bucket_name(), [])
    return [e for e in entries if isinstance(e, dict)]


def add_recent(
    prompt: str,
    detections: int | None = None,
    detail: int | None = None,
) -> None:
    """Append a detected object to Recent, deduped + newest-first, capped.

    Re-detecting an object moves it back to the top and refreshes its stats."""
    text = _normalize_dedupe_key(prompt)
    if not text:
        return
    entries = [e for e in get_recent() if _normalize_dedupe_key(e.get("prompt", "")) != text]
    entry: dict = {"prompt": text, "ts": _now_iso()}
    if detections is not None:
        entry["detections"] = int(detections)
    if detail is not None:
        entry["detail"] = int(detail)
    entries.insert(0, entry)
    if len(entries) > _RECENT_CAP:
        entries = entries[:_RECENT_CAP]
    buckets = _read_buckets()
    buckets[_recent_bucket_name()] = entries
    settings = QSettings()
    settings.setValue(_RECENT_KEY, json.dumps(buckets, ensure_ascii=False))
    settings.remove(_UNSCOPED_RECENT_KEY)


def clear_unscoped_recent_objects() -> None:
    """Drop the machine-wide list older versions wrote.

    The per-account buckets are deliberately left alone: they are the user's
    own detections, unreachable from any other account, and deleting them on
    sign-out would lose history for no gain."""
    QSettings().remove(_UNSCOPED_RECENT_KEY)
