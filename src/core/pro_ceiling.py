"""The way out for a paid account that is near or at the end of its month.

A subscriber who spends the monthly km² or objects allowance used to meet a
wall with no exit: the free offer is hidden for them, and nothing said who to
write to. Every value here is a server dial under the ``pro_ceiling`` key
with one shipped fallback, so the address and the moment the nudge appears
can both move without a plugin release.

The address is shown in the card and copied on click, never opened as a
``mailto:`` link: a mail client that is not set up swallows the click and
the user learns nothing.

Pure Python: no Qt, no QGIS, no network. The dock reads it on the paint path,
so nothing here may raise.
"""
from __future__ import annotations

from .server_dials import dial_bool, dial_in_range, dial_str, feature_enabled

_DEFAULT_CONTACT_EMAIL = "yvann.barbot@terra-lab.ai"
_DEFAULT_LOW_FRACTION = 0.10
_MAX_CHARS = 120


def pro_ceiling_enabled() -> bool:
    """Both switches on: the feature kill switch (fail-open) and the config
    flag (shipped on). Either one can turn the whole path off fleet-wide."""
    return feature_enabled("pro_ceiling") and dial_bool("pro_ceiling.enabled", True)


def _looks_like_email(value: str) -> bool:
    """One "@" with something on both sides, no whitespace, no control
    characters, and short enough to sit on one line of a card."""
    if not value or len(value) > _MAX_CHARS or value.count("@") != 1:
        return False
    local, _, domain = value.partition("@")
    if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
        return False
    return all(ch.isprintable() and not ch.isspace() for ch in value)


def pro_ceiling_contact_email() -> str:
    """The address the card shows and the button copies. A served value that
    does not look like an address falls back to the shipped one."""
    served = dial_str("pro_ceiling.contact_email", _DEFAULT_CONTACT_EMAIL)
    return served if _looks_like_email(served) else _DEFAULT_CONTACT_EMAIL


def pro_ceiling_low_fraction() -> float:
    """Share of the monthly cap under which the "running low" nudge shows.
    Zero turns the nudge off; the cap of one half keeps it a nudge."""
    return float(dial_in_range(
        "pro_ceiling.low_fraction", _DEFAULT_LOW_FRACTION, 0.0, 0.5))
