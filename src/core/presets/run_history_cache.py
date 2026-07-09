



















from __future__ import annotations

import hashlib
import json
from itertools import islice

from qgis.PyQt.QtCore import QSettings




_UNSCOPED_RUN_CACHE_KEY = "AISegmentation/library_run_cache"
_RUN_CACHE_KEY = "AISegmentation/library_run_cache_by_account"



_RUNS_CAP = 50


_FINGERPRINT_LEN = 16
_MAX_RUN_CACHE_BYTES = 512 * 1024


_cached_fingerprint: str | None = None


def account_fingerprint() -> str:










    global _cached_fingerprint
    if _cached_fingerprint is not None:
        return _cached_fingerprint

    try:
        from ..activation_manager import get_auth_token

        key = (get_auth_token() or "").strip()
    except Exception:


        return ""
    if not key:
        return ""
    _cached_fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_FINGERPRINT_LEN]
    return _cached_fingerprint


def reset_account_fingerprint_cache() -> None:





    global _cached_fingerprint
    _cached_fingerprint = None


def get_runs() -> list[dict]:

    account = account_fingerprint()
    if not account:
        return []
    raw = QSettings().value(_RUN_CACHE_KEY, "")
    if not raw or not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_RUN_CACHE_BYTES:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict) or data.get("account") != account:
        return []
    runs = data.get("runs")
    if not isinstance(runs, list):
        return []
    from ..server_dials import dial_in_range
    runs_cap = dial_in_range("tuning.library.run_cache_cap", _RUNS_CAP, 5, 500)
    return list(islice((r for r in runs if isinstance(r, dict)), runs_cap))


def save_runs(runs: list[dict]) -> None:

    settings = QSettings()
    settings.remove(_UNSCOPED_RUN_CACHE_KEY)
    account = account_fingerprint()
    if not account:


        settings.remove(_RUN_CACHE_KEY)
        return
    from ..server_dials import dial_in_range
    runs_cap = dial_in_range("tuning.library.run_cache_cap", _RUNS_CAP, 5, 500)
    capped = list(islice((r for r in (runs or []) if isinstance(r, dict)), runs_cap))
    blob = {"account": account, "runs": capped}
    payload = json.dumps(blob, ensure_ascii=False)
    if len(payload.encode("utf-8")) <= _MAX_RUN_CACHE_BYTES:
        settings.setValue(_RUN_CACHE_KEY, payload)


def clear_run_history_cache() -> None:

    settings = QSettings()
    settings.remove(_RUN_CACHE_KEY)
    settings.remove(_UNSCOPED_RUN_CACHE_KEY)
