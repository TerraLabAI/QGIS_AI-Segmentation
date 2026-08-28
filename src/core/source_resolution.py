


























from __future__ import annotations

import math

from .tile_manager import NATIVE_OVERSAMPLE_MAX












OVERSAMPLE_ALLOWANCE_MIN = 1.0
OVERSAMPLE_ALLOWANCE_MAX = 8.0



OVERSAMPLE_ALLOWANCE_DEFAULT = NATIVE_OVERSAMPLE_MAX


_SOURCE_TABLE_KEY = "source_native_mupp"


def oversample_allowance(policy: dict | None = None) -> float:







    from .detection_policy import native_oversample_max

    try:
        value = float(native_oversample_max(policy))
    except (TypeError, ValueError):
        return OVERSAMPLE_ALLOWANCE_DEFAULT
    if math.isnan(value) or value in (float("inf"), float("-inf")):
        return OVERSAMPLE_ALLOWANCE_DEFAULT
    if not OVERSAMPLE_ALLOWANCE_MIN <= value <= OVERSAMPLE_ALLOWANCE_MAX:
        return OVERSAMPLE_ALLOWANCE_DEFAULT
    return value


def source_floor_mupp_m(source_uri: str, policy: dict | None = None) -> float:











    entries = _source_table(policy)
    if not entries:
        return 0.0
    try:
        text = (source_uri or "").lower()
    except (AttributeError, TypeError):
        return 0.0
    if not text:
        return 0.0
    for entry in entries:
        match = entry.get("match")
        if not isinstance(match, str) or not match:
            continue
        if match.lower() not in text:
            continue
        mupp = _positive_float(entry.get("mupp"))
        if mupp > 0:
            return mupp
    return 0.0


def _source_table(policy: dict | None) -> list[dict]:





    from .detection_policy import seed_policy

    try:
        raw = seed_policy(policy).get(_SOURCE_TABLE_KEY)
    except (AttributeError, TypeError):
        return []
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _positive_float(value: object) -> float:





    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    if math.isnan(value) or value in (float("inf"), float("-inf")):
        return 0.0
    return value if value > 0 else 0.0
