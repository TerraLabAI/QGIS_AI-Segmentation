














from __future__ import annotations

import json
import time
from itertools import islice

from qgis.PyQt.QtCore import QSettings

from .run_history_cache import account_fingerprint




_UNSCOPED_RECENT_KEY = "AISegmentation/recent_objects"
_RECENT_KEY = "AISegmentation/recent_objects_by_account"



_SIGNED_OUT_BUCKET = "signed_out"




_RECENT_CAP = 200
_MAX_RECENT_BYTES = 256 * 1024


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_dedupe_key(prompt: str) -> str:

    return prompt.strip() if isinstance(prompt, str) else ""


def _recent_bucket_name() -> str:

    return account_fingerprint() or _SIGNED_OUT_BUCKET


def _read_buckets() -> dict:

    raw = QSettings().value(_RECENT_KEY, "")
    if not raw or not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_RECENT_BYTES:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, list)}


def _write_buckets(buckets: dict) -> bool:

    settings = QSettings()
    existing = settings.value(_RECENT_KEY, "")
    if isinstance(existing, str) and len(existing.encode("utf-8")) > _MAX_RECENT_BYTES:
        return False
    payload = json.dumps(buckets, ensure_ascii=False)
    if len(payload.encode("utf-8")) > _MAX_RECENT_BYTES:
        return False
    settings.setValue(_RECENT_KEY, payload)
    return True


def get_recent() -> list[dict]:





    from ..server_dials import dial_in_range
    recent_cap = dial_in_range("tuning.library.recent_objects_cap", _RECENT_CAP, 10, 1000)
    entries = _read_buckets().get(_recent_bucket_name(), [])
    return list(islice((e for e in entries if isinstance(e, dict)), recent_cap))


def add_recent(
    prompt: str,
    detections: int | None = None,
    detail: int | None = None,
) -> None:



    text = _normalize_dedupe_key(prompt) if isinstance(prompt, str) else ""
    if not text:
        return
    bucket = _recent_bucket_name()
    buckets = _read_buckets()
    entries = [e for e in buckets.get(bucket, []) if isinstance(e, dict)
               and _normalize_dedupe_key(e.get("prompt", "")) != text]
    entry: dict = {"prompt": text, "ts": _now_iso()}
    if detections is not None:
        entry["detections"] = int(detections)
    if detail is not None:
        entry["detail"] = int(detail)
    entries.insert(0, entry)
    from ..server_dials import dial_in_range
    recent_cap = dial_in_range("tuning.library.recent_objects_cap", _RECENT_CAP, 10, 1000)
    if len(entries) > recent_cap:
        entries = entries[:recent_cap]
    buckets[bucket] = entries
    if _write_buckets(buckets):
        QSettings().remove(_UNSCOPED_RECENT_KEY)


def clear_recent_objects_for_account() -> None:







    bucket = _recent_bucket_name()
    buckets = _read_buckets()
    if bucket not in buckets:
        return
    buckets.pop(bucket, None)
    if buckets:
        _write_buckets(buckets)
    else:
        existing = QSettings().value(_RECENT_KEY, "")
        if not (isinstance(existing, str) and len(existing.encode("utf-8")) > _MAX_RECENT_BYTES):
            QSettings().remove(_RECENT_KEY)


def clear_unscoped_recent_objects() -> None:





    QSettings().remove(_UNSCOPED_RECENT_KEY)
