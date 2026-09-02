"""Server dials for the dock's small surfaces: prompt help, links, the
Library, the install ETA.

Every getter here takes the shipped constant as its fallback and returns it
whenever the cached configuration has no usable value, so each call site keeps
working offline and on a cold cache. Every numeric dial is bounded, so a bad
served value cannot break the widget that reads it. Read at call time, never
at import time: the cache fills after import.

Pure Python with no Qt at import time, like ``server_dials``.
"""
from __future__ import annotations

import re

from .server_dials import dial_in_range, dial_url, feature_enabled, read_value

# -- prompt suggestions (the popup under the Automatic prompt box) -----------


def prompt_suggest_max_rows(fallback: int) -> int:
    """How many rows the suggestion list ranks."""
    return dial_in_range("ui.prompt_suggest.max_rows", fallback, 1, 50)


def prompt_suggest_visible_rows(fallback: int) -> int:
    """How many rows the suggestion list shows before it scrolls."""
    return dial_in_range("ui.prompt_suggest.visible_rows", fallback, 1, 30)


def prompt_suggest_recent_scan(fallback: int) -> int:
    """How many recent objects feed the top of the list. Zero means none."""
    return dial_in_range("ui.prompt_suggest.recent_scan", fallback, 0, 500)


def prompt_suggest_synonym_min_chars(fallback: int) -> int:
    """Shortest alphabet-script query that reaches the cross-language synonyms."""
    return dial_in_range("ui.prompt_suggest.synonym_min_chars", fallback, 1, 10)


# -- prompt guard: the "several objects at once" separators ------------------

# A separator is a short literal: a punctuation mark, or a connector word
# padded with spaces so it matches whole words only. The served list REPLACES
# the shipped one (a connector that reads wrong in one language has to be
# removable), and entries are taken verbatim: stripping them would turn
# " and " into "and" and match inside "sand".
_MAX_SEPARATORS = 32
_MAX_SEPARATOR_CHARS = 12

_SEPARATOR_MEMO: tuple[object, tuple[str, ...], re.Pattern] | None = None


def multi_object_separators(fallback: tuple[str, ...]) -> tuple[str, ...]:
    """The separators that mean "several objects at once", served or shipped."""
    try:
        value = read_value("prompt.multi_object_separators")
        if isinstance(value, (list, tuple)):
            kept = tuple(
                item for item in value[:_MAX_SEPARATORS]
                if isinstance(item, str) and item.strip()
                and len(item) <= _MAX_SEPARATOR_CHARS
                and item.isprintable()
            )
            if kept:
                return kept
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        pass
    return tuple(fallback)


def multi_object_pattern(fallback: tuple[str, ...]) -> re.Pattern:
    """Compiled alternation of ``multi_object_separators``, each entry escaped.

    Memoised on the served list object, so a keystroke never recompiles: the
    cache publishes a new list object only when the configuration changes.
    """
    global _SEPARATOR_MEMO
    try:
        token = read_value("prompt.multi_object_separators")
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        token = None
    memo = _SEPARATOR_MEMO
    if memo is not None and memo[0] is token and memo[1] == tuple(fallback):
        return memo[2]
    separators = multi_object_separators(fallback)
    pattern = re.compile("|".join(re.escape(sep) for sep in separators))
    _SEPARATOR_MEMO = (token, tuple(fallback), pattern)
    return pattern


# -- outbound links -----------------------------------------------------------


def guide_url_base(fallback: str) -> str:
    """Where the step-by-step guide lives, before the UTM stem is appended."""
    return dial_url("guidance.guide_url", fallback)


def cross_promo_url(fallback: str) -> str:
    """The sibling plugin's product page. Same key the sibling reads for ours."""
    return dial_url("cross_promo_url", fallback)


def cross_sell_ai_edit_enabled() -> bool:
    """Whether the sibling plugin is offered at all. Fail-open."""
    return feature_enabled("cross_sell_ai_edit")


# -- Library -------------------------------------------------------------------


def library_mask_budget_base_s(fallback: float) -> float:
    """Fixed part of the wall clock the Library's mask restore may spend."""
    return dial_in_range("library.mask_budget_base_s", fallback, 5, 600)


def library_mask_budget_per_tile_s(fallback: float) -> float:
    """Per-tile part of that wall clock. Zero means the base alone."""
    return dial_in_range("library.mask_budget_per_tile_s", fallback, 0, 60)


def library_mask_budget_max_s(fallback: float) -> float:
    """Hard ceiling on that wall clock, whatever the tile count."""
    return dial_in_range("library.mask_budget_max_s", fallback, 30, 3600)


def library_max_marks(fallback: int) -> int:
    """How many run marks the Library keeps, newest first."""
    return dial_in_range("library.max_marks", fallback, 10, 5000)


def library_demo_cache_ttl_s(fallback: int) -> int:
    """How long a cached demo image without a validator is trusted."""
    return dial_in_range("library.demo_cache_ttl_s", fallback, 3600, 31536000)


# -- install ETA ---------------------------------------------------------------


def install_eta_honest_s(fallback: int) -> int:
    """Past this many seconds the ETA says "more than N min" instead of a number."""
    return dial_in_range("install.eta_honest_s", fallback, 60, 86400)


def install_eta_ceiling_s(fallback: int) -> int:
    """The ETA is clamped here, so a stalled link never shows a day."""
    return dial_in_range("install.eta_ceiling_s", fallback, 600, 86400)


def removal_watchdog_ms(fallback: int) -> int:
    """How long the removal window waits for a worker that never reports back
    before it lets the user out."""
    return dial_in_range("install.removal_watchdog_ms", fallback, 10_000, 3_600_000)


# -- billing card ----------------------------------------------------------------


def pro_monthly_credits_fallback(fallback: int) -> int:
    """The Pro monthly allowance printed when the usage payload carries no
    limit of its own. Printed, never spent, so it only has to match the plan."""
    return dial_in_range("gate.pro_monthly_credits", fallback, 1, 1_000_000)


__all__ = [
    "cross_promo_url",
    "cross_sell_ai_edit_enabled",
    "guide_url_base",
    "install_eta_ceiling_s",
    "install_eta_honest_s",
    "library_demo_cache_ttl_s",
    "library_mask_budget_base_s",
    "library_mask_budget_max_s",
    "library_mask_budget_per_tile_s",
    "library_max_marks",
    "multi_object_pattern",
    "multi_object_separators",
    "prompt_suggest_max_rows",
    "prompt_suggest_recent_scan",
    "prompt_suggest_synonym_min_chars",
    "prompt_suggest_visible_rows",
    "pro_monthly_credits_fallback",
    "removal_watchdog_ms",
]
