"""Persistent warm-start cache of the user's cloud run history (Library).

The library's History tab renders from this cache the instant it opens, then
refreshes from the server in the background. Without it the first open of a
session shows nothing until the network round-trip returns, which reads as a
slow pop-in. Server data stays the source of truth; this is a warm-start cache
only (same mechanism as the catalogue cache and AI Edit's history cache).

Stored as one JSON blob in QSettings under ``AISegmentation/library_run_cache``.
Only run metadata is cached (ids, counts, dates, prompt token) - never masks,
never coordinates beyond what the server list already returns.
"""
from __future__ import annotations

import json

from qgis.PyQt.QtCore import QSettings

_RUN_CACHE_KEY = "AISegmentation/library_run_cache"

# Matches the "keep the warm start cheap" cap: caching more than a few pages is
# wasted settings I/O (older pages re-fetch on demand via Load older runs).
_RUNS_CAP = 50


def get_runs() -> list[dict]:
    """Cached run list (newest first), or [] when empty/corrupt."""
    raw = QSettings().value(_RUN_CACHE_KEY, "")
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict)]


def save_runs(runs: list[dict]) -> None:
    """Persist the freshly synced run list (capped)."""
    capped = [r for r in (runs or []) if isinstance(r, dict)][:_RUNS_CAP]
    QSettings().setValue(_RUN_CACHE_KEY, json.dumps(capped, ensure_ascii=False))


def clear() -> None:
    QSettings().remove(_RUN_CACHE_KEY)
