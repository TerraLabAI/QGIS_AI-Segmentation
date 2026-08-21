"""The two-envelope quota as the server tells it: cloud objects and km².

The account endpoint (/api/plugin/account) is the one surface that carries
the envelope fields; every other payload keeps the frozen wallet names. All
fields are additive and nullable, so this module parses defensively: a field
the server did not send is None, never 0. None means "unknown or exempt";
a display path shows nothing for it and a gate never fires on it.

Older servers send none of these fields, and then ``from_account_row``
answers None: every caller falls back to the wallet figures it shows today.
"""
from __future__ import annotations

from typing import NamedTuple

_OBJECT_FIELDS = ("objects_used_this_month", "objects_cap", "objects_remaining")
_KM2_FIELDS = ("counted_km2_this_month", "km2_cap", "km2_remaining")


class QuotaEnvelopes(NamedTuple):
    """What the account said about the two monthly envelopes.

    ``km2_cap`` None on a Pro row means the plan is never km²-gated (legacy
    subscription), not that the cap is zero. Any None used/remaining pair
    means the read failed server-side; show the wallet figure instead.
    """

    objects_used: int | None
    objects_cap: int | None
    objects_remaining: int | None
    km2_used: float | None
    km2_cap: float | None
    km2_remaining: float | None

    def has_objects_gauge(self) -> bool:
        """True when the objects envelope can be drawn as used-of-cap."""
        return self.objects_used is not None and (self.objects_cap or 0) > 0

    def has_km2_gauge(self) -> bool:
        """True when the km² envelope can be drawn as used-of-cap."""
        return self.km2_used is not None and (self.km2_cap or 0) > 0


def _as_count(value) -> int | None:
    """The value as a non-negative int, or None for null/absent/garbage."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _as_km2(value) -> float | None:
    """The value as a non-negative finite float, or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return max(0.0, out)


def quota_envelopes_from_account_row(row: dict) -> QuotaEnvelopes | None:
    """The envelope fields of one account subscription row, or None.

    None when the row carries none of the six envelope keys at all (an older
    server), so the caller keeps today's wallet display without a branch per
    field.
    """
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
    """The AI Segmentation subscription row of an /account payload, or None.

    Free and Pro rows both carry a product_id starting with "ai-segmentation",
    and an account can hold both. The paid row wins, so the envelope this
    module reads is the one the account dialog shows and the one the user pays
    for; taking whichever came first gave the two surfaces different quotas.
    """
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
    """A km² figure as the short string the UI prints (3, 0.04, 2.25)."""
    try:
        return f"{round(float(value), 2):g}"
    except (TypeError, ValueError):
        return "0"
