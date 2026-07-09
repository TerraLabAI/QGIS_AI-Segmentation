"""Local 'recently detected' objects, persisted via QSettings.

Records the object (the English cloud-model token) of each committed Automatic
detection run so the Segment library can offer a Recent tab to re-run it. The
list is local-only - nothing leaves the machine.

No favorites: segmentation recall is about quickly reusing the last objects you
actually detected, not curating a starred set. This is the trimmed sibling of AI Edit's ``prompt_history`` (Recent half
only, with segmentation-flavoured per-entry stats).
"""
from __future__ import annotations

import json
import time

from qgis.PyQt.QtCore import QSettings

_RECENT_KEY = "AISegmentation/recent_objects"

# Cap on stored entries. The list is serialised as one JSON blob and rewritten
# on every committed run, so an uncapped list would bloat settings I/O. 200 is
# well past anyone's "recently used" memory and still loads instantly.
_RECENT_CAP = 200


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize(prompt: str) -> str:
    """Dedupe key. Whitespace-trim only; case is preserved."""
    return (prompt or "").strip()


def get_recent() -> list[dict]:
    """Return Recent objects, newest first.

    Each entry: ``{prompt, ts, detections?, detail?}`` where ``prompt`` is the
    English token, ``detections`` the count exported by that run, ``detail`` the
    grid level used."""
    raw = QSettings().value(_RECENT_KEY, "")
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return data if isinstance(data, list) else []


def add_recent(
    prompt: str,
    detections: int | None = None,
    detail: int | None = None,
) -> None:
    """Append a detected object to Recent, deduped + newest-first, capped.

    Re-detecting an object moves it back to the top and refreshes its stats."""
    text = _normalize(prompt)
    if not text:
        return
    entries = [e for e in get_recent() if _normalize(e.get("prompt", "")) != text]
    entry: dict = {"prompt": text, "ts": _now_iso()}
    if detections is not None:
        entry["detections"] = int(detections)
    if detail is not None:
        entry["detail"] = int(detail)
    entries.insert(0, entry)
    if len(entries) > _RECENT_CAP:
        entries = entries[:_RECENT_CAP]
    QSettings().setValue(_RECENT_KEY, json.dumps(entries, ensure_ascii=False))
