










from __future__ import annotations

import re

from .server_dials import dial_in_range, dial_url, feature_enabled, read_value




def prompt_suggest_max_rows(fallback: int) -> int:

    return dial_in_range("ui.prompt_suggest.max_rows", fallback, 1, 50)


def prompt_suggest_visible_rows(fallback: int) -> int:

    return dial_in_range("ui.prompt_suggest.visible_rows", fallback, 1, 30)


def prompt_suggest_recent_scan(fallback: int) -> int:

    return dial_in_range("ui.prompt_suggest.recent_scan", fallback, 0, 500)


def prompt_suggest_synonym_min_chars(fallback: int) -> int:

    return dial_in_range("ui.prompt_suggest.synonym_min_chars", fallback, 1, 10)









_MAX_SEPARATORS = 32
_MAX_SEPARATOR_CHARS = 12

_separator_memo: dict[str, tuple[object, tuple[str, ...], re.Pattern]] = {}


def multi_object_separators(fallback: tuple[str, ...]) -> tuple[str, ...]:

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
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return tuple(fallback)


def multi_object_pattern(fallback: tuple[str, ...]) -> re.Pattern:





    try:
        token = read_value("prompt.multi_object_separators")
    except Exception:  # noqa: BLE001  # nosec B110
        token = None
    memo = _separator_memo.get("held")
    if memo is not None and memo[0] is token and memo[1] == tuple(fallback):
        return memo[2]
    separators = multi_object_separators(fallback)
    pattern = re.compile("|".join(re.escape(sep) for sep in separators))
    _separator_memo["held"] = (token, tuple(fallback), pattern)
    return pattern





def guide_url_base(fallback: str) -> str:

    return dial_url("guidance.guide_url", fallback)


def cross_promo_url(fallback: str) -> str:

    return dial_url("cross_promo_url", fallback)


def cross_sell_ai_edit_enabled() -> bool:

    return feature_enabled("cross_sell_ai_edit")





def library_mask_budget_base_s(fallback: float) -> float:

    return dial_in_range("library.mask_budget_base_s", fallback, 5, 600)


def library_mask_budget_per_tile_s(fallback: float) -> float:

    return dial_in_range("library.mask_budget_per_tile_s", fallback, 0, 60)


def library_mask_budget_max_s(fallback: float) -> float:

    return dial_in_range("library.mask_budget_max_s", fallback, 30, 3600)


def library_max_marks(fallback: int) -> int:

    return dial_in_range("library.max_marks", fallback, 10, 5000)


def library_demo_cache_ttl_s(fallback: int) -> int:

    return dial_in_range("library.demo_cache_ttl_s", fallback, 3600, 31536000)





def install_eta_honest_s(fallback: int) -> int:

    return dial_in_range("install.eta_honest_s", fallback, 60, 86400)


def install_eta_ceiling_s(fallback: int) -> int:

    return dial_in_range("install.eta_ceiling_s", fallback, 600, 86400)


def removal_watchdog_ms(fallback: int) -> int:


    return dial_in_range("install.removal_watchdog_ms", fallback, 10_000, 3_600_000)





def pro_monthly_credits_fallback(fallback: int) -> int:


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
