










from __future__ import annotations

import math
from typing import NamedTuple

_OBJECT_FIELDS = ("objects_used_this_month", "objects_cap", "objects_remaining")
_KM2_FIELDS = ("counted_km2_this_month", "km2_cap", "km2_remaining")


class QuotaEnvelopes(NamedTuple):







    objects_used: int | None
    objects_cap: int | None
    objects_remaining: int | None
    km2_used: float | None
    km2_cap: float | None
    km2_remaining: float | None

    def has_objects_gauge(self) -> bool:

        return self.objects_used is not None and (self.objects_cap or 0) > 0

    def has_km2_gauge(self) -> bool:

        return self.km2_used is not None and (self.km2_cap or 0) > 0


def _as_count(value) -> int | None:

    if value is None or isinstance(value, bool):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _as_km2(value) -> float | None:

    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(out) or out in (float("inf"), float("-inf")):
        return None
    return max(0.0, out)


def quota_envelopes_from_account_row(row: dict) -> QuotaEnvelopes | None:






    if not isinstance(row, dict):
        return None
    if not any(key in row for key in _OBJECT_FIELDS + _KM2_FIELDS):
        return None
    return QuotaEnvelopes(
        objects_used=_as_count(row.get("objects_used_this_month")),
        objects_cap=_as_count(row.get("objects_cap")),
        objects_remaining=_as_count(row.get("objects_remaining")),
        km2_used=_as_km2(row.get("counted_km2_this_month")),
        km2_cap=_as_km2(row.get("km2_cap")),
        km2_remaining=_as_km2(row.get("km2_remaining")),
    )


def pick_segmentation_account_row(account: dict) -> dict | None:







    if not isinstance(account, dict):
        return None
    subs = account.get("subscriptions")
    if not isinstance(subs, list):
        return None
    rows = [row for row in subs
            if isinstance(row, dict)
            and str(row.get("product_id", "")).startswith("ai-segmentation")]
    for row in rows:
        if str(row.get("product_id", "")).endswith("-pro"):
            return row
    return rows[0] if rows else None


def format_km2_value(value: float) -> str:

    try:
        return f"{round(float(value), 2):g}"
    except (TypeError, ValueError):
        return "0"
