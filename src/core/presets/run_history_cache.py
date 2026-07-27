"""Persistent warm-start cache of the user's cloud run history (Library).

The library's History tab renders from this cache the instant it opens, then
refreshes from the server in the background. Without it the first open of a
session shows nothing until the network round-trip returns, which reads as a
slow pop-in. Server data stays the source of truth; this is a warm-start cache
only (same mechanism as the catalogue cache and AI Edit's history cache).

Stored as one JSON blob in QSettings under
``AISegmentation/library_run_cache_by_account``. The blob carries the account
fingerprint it was written for, and the read path returns nothing when that
fingerprint does not match the account signed in right now: one account's runs
must never show up in another account's library. Only run metadata is cached
(ids, counts, dates, prompt token) - never masks, never coordinates beyond what
the server list already returns.

:func:`account_fingerprint` also scopes the two local stores
(``presets/segment_history`` and ``core/detection_history``), which is why it
lives here rather than being copied into each of them.
"""
from __future__ import annotations

import hashlib
import json

from qgis.PyQt.QtCore import QSettings

# Written by versions that had no account scoping. Never read again: an
# untagged blob cannot be attributed to an account, so it is dropped on sight
# rather than shown to whoever signs in next.
_UNSCOPED_RUN_CACHE_KEY = "AISegmentation/library_run_cache"
_RUN_CACHE_KEY = "AISegmentation/library_run_cache_by_account"

# Matches the "keep the warm start cheap" cap: caching more than a few pages is
# wasted settings I/O (older pages re-fetch on demand via Load older runs).
_RUNS_CAP = 50

# 16 hex chars (64 bits) of a SHA256 digest, same shape as the device hash.
_FINGERPRINT_LEN = 16

# "" is a real answer (signed out), so None means "not computed yet".
_cached_fingerprint: str | None = None


def account_fingerprint() -> str:
    """Short irreversible tag for the signed-in account, "" when signed out.

    Derived from the activation key with SHA256 and truncated, so nothing
    credential-like is written to disk and the raw key cannot be read back out
    of a cache blob or a directory name. Used only to keep one account's
    cached history out of the next account's view.

    Cached for the process lifetime: the library resolves it once per rendered
    card, and the key lookup can reach the auth database.
    """
    global _cached_fingerprint
    if _cached_fingerprint is not None:
        return _cached_fingerprint

    try:
        from ..activation_manager import get_auth_token

        key = (get_auth_token() or "").strip()
    except Exception:
        # Signed out is the fail-safe answer: it shows nothing and leaks
        # nothing. Not cached, so a transient lookup failure is retried.
        return ""
    _cached_fingerprint = (
        hashlib.sha256(key.encode("utf-8")).hexdigest()[:_FINGERPRINT_LEN]
        if key else ""
    )
    return _cached_fingerprint


def reset_account_fingerprint_cache() -> None:
    """Forget the cached fingerprint.

    Every write of the activation key MUST call this, or the stores keep
    answering for the account that just left.
    """
    global _cached_fingerprint
    _cached_fingerprint = None


def get_runs() -> list[dict]:
    """Cached run list (newest first), or [] when empty/corrupt/another account."""
    account = account_fingerprint()
    if not account:
        return []
    raw = QSettings().value(_RUN_CACHE_KEY, "")
    if not raw:
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
    return [r for r in runs if isinstance(r, dict)]


def save_runs(runs: list[dict]) -> None:
    """Persist the freshly synced run list (capped), tagged with the account."""
    settings = QSettings()
    settings.remove(_UNSCOPED_RUN_CACHE_KEY)
    account = account_fingerprint()
    if not account:
        # Signed out: there is no account to attribute these runs to, so there
        # is nothing safe to warm-start from next time.
        settings.remove(_RUN_CACHE_KEY)
        return
    capped = [r for r in (runs or []) if isinstance(r, dict)][:_RUNS_CAP]
    blob = {"account": account, "runs": capped}
    settings.setValue(_RUN_CACHE_KEY, json.dumps(blob, ensure_ascii=False))


def clear_run_history_cache() -> None:
    """Drop the warm start entirely. Called on sign-in and on every sign-out."""
    settings = QSettings()
    settings.remove(_RUN_CACHE_KEY)
    settings.remove(_UNSCOPED_RUN_CACHE_KEY)
