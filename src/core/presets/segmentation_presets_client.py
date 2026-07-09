"""Fetches the segment-library catalogue from the TerraLab backend and caches
it locally, mirroring AI Edit's ``prompt_presets_client``.

Endpoint: ``GET /api/ai-segmentation/presets`` (public, no auth) returns
``{"version", "categories": [...], "top_picks": [...]}`` where each preset
carries ``demo_url_before`` / ``demo_url_after`` (relative paths served by
``/api/ai-segmentation/template-demos/<id>/<which>``).

Resilience: a fresh result is cached 1h in QSettings; on failure we fall back
to a stale cache, then to the bundled offline catalogue. A short negative cache
avoids re-hitting the network every time the dialog opens before the endpoint
is deployed.
"""
from __future__ import annotations

import json
import time

from qgis.PyQt.QtCore import QSettings

from . import segmentation_presets as _fallback

_CACHE_KEY = "AISegmentation/server_catalog_v1"
_CACHE_TS_KEY = "AISegmentation/server_catalog_v1_ts"
_NEG_TS_KEY = "AISegmentation/server_catalog_v1_neg_ts"
_CACHE_TTL_S = 3600          # success: 1h
_NEG_TTL_S = 600             # failure: don't retry for 10 min


def base_url() -> str:
    try:
        from ...api.terralab_client import TerraLabClient
        return TerraLabClient().base_url
    except Exception:  # noqa: BLE001
        return "https://terra-lab.ai"


def absolute_demo_url(base: str, relative: str | None) -> str:
    """Resolve a (possibly relative) demo image path against the API base."""
    if not relative:
        return ""
    if relative.startswith("http://") or relative.startswith("https://"):
        return relative
    return f"{base.rstrip('/')}{relative}"


def _is_valid_catalog(data) -> bool:
    return isinstance(data, dict) and isinstance(data.get("categories"), list) and len(data["categories"]) > 0


def _read_cache(settings: QSettings, *, ignore_ttl: bool = False) -> dict | None:
    raw = settings.value(_CACHE_KEY)
    ts = settings.value(_CACHE_TS_KEY)
    if not raw:
        return None
    try:
        if not ignore_ttl:
            if ts is None or (time.time() - float(ts)) >= _CACHE_TTL_S:
                return None
        data = json.loads(raw)
        return data if _is_valid_catalog(data) else None
    except Exception:  # noqa: BLE001
        return None


def fetch_catalog(force: bool = False) -> dict | None:
    """Return the server catalogue (``{categories, top_picks}``) or None.

    Order: fresh cache -> network -> stale cache. Returns None only when there
    has never been a successful fetch (caller then uses the offline catalogue).
    """
    settings = QSettings()
    if not force:
        cached = _read_cache(settings)
        if cached is not None:
            return cached
        # Negative cache: skip the network if a recent fetch already failed.
        neg = settings.value(_NEG_TS_KEY)
        try:
            if neg is not None and (time.time() - float(neg)) < _NEG_TTL_S:
                return _read_cache(settings, ignore_ttl=True)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    resp = None
    try:
        from ...api.terralab_client import TerraLabClient
        resp = TerraLabClient()._request(
            "GET", "/api/ai-segmentation/presets", timeout_ms=10_000)
    except Exception:  # noqa: BLE001
        resp = None

    if _is_valid_catalog(resp):
        try:
            settings.setValue(_CACHE_KEY, json.dumps(resp))
            settings.setValue(_CACHE_TS_KEY, str(time.time()))
            settings.remove(_NEG_TS_KEY)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return resp

    # Network failed or returned an error payload: remember the failure and
    # fall back to whatever stale cache we have.
    try:
        settings.setValue(_NEG_TS_KEY, str(time.time()))
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return _read_cache(settings, ignore_ttl=True)


def _enrich_categories(cats: list[dict]) -> list[dict]:
    """Backfill a sidebar emoji for any category the source omitted (the server
    catalogue currently sends ``emoji: null``). Never overwrites a provided
    emoji, so a future server-supplied glyph still wins."""
    for cat in cats:
        if isinstance(cat, dict) and not cat.get("emoji"):
            cat["emoji"] = _fallback.category_emoji(cat.get("key", ""))
    return cats


def cached_or_offline_catalog() -> tuple[list[dict], list[str]]:
    """UI-thread-safe catalogue: the cached server catalogue (ignoring TTL) if
    one was ever fetched, else the bundled offline one. NEVER touches the
    network - keeping the library snappy is the background prefetch's job
    (``fetch_catalog(force=True)`` run from a QgsTask). Always non-empty, so the
    gallery opens instantly even on a cold cache."""
    data = _read_cache(QSettings(), ignore_ttl=True)
    if data:
        cats = data.get("categories") or []
        tops = data.get("top_picks") or []
        if cats:
            return _enrich_categories(cats), list(tops)
    return _fallback.fallback_categories(), list(_fallback.TOP_PICKS)
