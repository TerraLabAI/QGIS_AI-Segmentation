












from __future__ import annotations

import json
import math
import time
from urllib.parse import urlsplit

from qgis.PyQt.QtCore import QSettings

from .. import transport_dials as _td
from . import segmentation_presets as _fallback

_CACHE_KEY = "AISegmentation/server_catalog_v1"
_CACHE_TS_KEY = "AISegmentation/server_catalog_v1_ts"
_NEG_TS_KEY = "AISegmentation/server_catalog_v1_neg_ts"
_CACHE_TTL_S = 3600
_NEG_TTL_S = 600




_MAX_CATALOG_BYTES = 512 * 1024





_parsed_memo: tuple[float, str, dict] | None = None


def base_url() -> str:
    try:
        from ...api.terralab_client import TerraLabClient
        return TerraLabClient().base_url
    except Exception:  # noqa: BLE001
        return "https://terra-lab.ai"


def absolute_demo_url(base: str, relative: str | None) -> str:







    if not relative or not isinstance(relative, str):
        return ""
    if "://" in relative[:16]:
        from ..server_dials import safe_web_url

        return safe_web_url(relative, "")
    if any(ord(char) < 32 for char in relative) or "\\" in relative:
        return ""
    try:
        parts = urlsplit(relative)
        if parts.scheme or parts.netloc or any(part == ".." for part in parts.path.split("/")):
            return ""
        if relative.startswith("//") or not relative.startswith("/"):
            return ""
        from ..server_dials import safe_web_url

        return safe_web_url(f"{base.rstrip('/')}{relative}", "")
    except Exception:  # noqa: BLE001
        return ""


def _is_valid_catalog(data) -> bool:
    if not isinstance(data, dict):
        return False
    categories = data.get("categories")
    if not isinstance(categories, list) or not categories or len(categories) > 200:
        return False
    if not all(isinstance(category, dict) for category in categories):
        return False
    for category in categories:
        if not isinstance(category.get("key"), str) or not category["key"].strip():
            return False
        presets = category.get("presets", [])
        if not isinstance(presets, list) or len(presets) > 1000:
            return False
        for preset in presets:
            if (not isinstance(preset, dict)
                    or not isinstance(preset.get("id"), str) or not preset["id"].strip()
                    or not isinstance(preset.get("prompt"), str) or not preset["prompt"].strip()):
                return False
    top_picks = data.get("top_picks", [])
    return isinstance(top_picks, list) and len(top_picks) <= 200 and all(
        isinstance(item, str) and item.strip() for item in top_picks
    )


def _read_cache(settings: QSettings, *, ignore_ttl: bool = False) -> dict | None:
    raw = settings.value(_CACHE_KEY)
    ts = settings.value(_CACHE_TS_KEY)
    if not raw:
        return None
    global _parsed_memo
    try:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_CATALOG_BYTES:
            return None
        timestamp = float(ts)
        now = time.time()
        if not math.isfinite(timestamp) or timestamp > now + 300:
            return None
        if not ignore_ttl:
            if (now - timestamp) >= _td.catalog_ttl_s(_CACHE_TTL_S):
                return None
        if _parsed_memo is not None and _parsed_memo[:2] == (timestamp, raw):
            return _parsed_memo[2]
        data = json.loads(raw)
        if not _is_valid_catalog(data):
            return None
        _parsed_memo = (timestamp, raw, data)
        return data
    except Exception:  # noqa: BLE001
        return None


def fetch_catalog(force: bool = False) -> dict | None:





    settings = QSettings()
    cached = _read_cache(settings)
    if cached is not None and not force:
        return cached
    if not force:

        neg = settings.value(_NEG_TS_KEY)
        try:
            now = time.time()
            neg_stamp = float(neg)
            if (math.isfinite(neg_stamp) and neg_stamp <= now
                    and (now - neg_stamp) < _td.catalog_neg_ttl_s(_NEG_TTL_S)):
                return _read_cache(settings, ignore_ttl=True)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    resp = None
    try:
        from ...api.terralab_client import TerraLabClient
        from ..server_dials import dial_in_range
        timeout_ms = dial_in_range("tuning.library.catalog_fetch_timeout_ms", 10_000, 2000, 60_000)
        resp = TerraLabClient().get_segment_catalog(timeout_ms=timeout_ms)
    except Exception:  # noqa: BLE001
        resp = None

    if _is_valid_catalog(resp):
        try:
            payload = json.dumps(resp)
            if len(payload.encode("utf-8")) <= _MAX_CATALOG_BYTES:
                settings.setValue(_CACHE_KEY, payload)
                settings.setValue(_CACHE_TS_KEY, str(time.time()))
                settings.remove(_NEG_TS_KEY)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return resp



    try:
        settings.setValue(_NEG_TS_KEY, str(time.time()))
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return _read_cache(settings, ignore_ttl=True)


def _enrich_categories(cats: list[dict]) -> list[dict]:



    for cat in cats:
        if isinstance(cat, dict) and not cat.get("emoji"):
            cat["emoji"] = _fallback.category_emoji(cat.get("key", ""))
    return cats


def cached_or_offline_catalog() -> tuple[list[dict], list[str]]:





    data = _read_cache(QSettings(), ignore_ttl=True)
    if data:
        cats = data.get("categories") or []
        tops = data.get("top_picks") or []
        if cats:


            return _fallback.merged_categories(_enrich_categories(cats)), list(tops)
    return _fallback.fallback_categories(), list(_fallback.TOP_PICKS)
