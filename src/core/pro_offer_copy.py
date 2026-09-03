


















from __future__ import annotations

from .i18n import tr



PRO_PRICING_KEY = "ai_segmentation_pro"




_PRICE_MIN_EUR = 1
_PRICE_MAX_EUR = 10_000

_CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£"}


def _pricing_blob() -> dict:
    try:
        from .config_cache import get_config
        pricing = get_config().get("pricing")
    except Exception:  # noqa: BLE001
        return {}
    return pricing if isinstance(pricing, dict) else {}


def _monthly_amount(product_key: str) -> int | None:

    entry = _pricing_blob().get(product_key)
    if not isinstance(entry, dict):
        return None
    raw = entry.get("month_eur")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    amount = int(round(raw))
    if not _PRICE_MIN_EUR <= amount <= _PRICE_MAX_EUR:
        return None
    return amount


def _trial_days() -> int | None:
    raw = _pricing_blob().get("trial_days")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    days = int(raw)
    return days if 1 <= days <= 90 else None


def format_price_amount(amount: int) -> str:

    currency = str(_pricing_blob().get("currency") or "EUR").strip().upper()
    symbol = _CURRENCY_SYMBOLS.get(currency)
    return f"{symbol}{amount}" if symbol else f"{amount} {currency}"


def pro_price_phrase(product_key: str = PRO_PRICING_KEY) -> str:







    amount = _monthly_amount(product_key)
    if amount is None:
        return ""
    price = format_price_amount(amount)
    days = _trial_days()
    if days is None:
        return tr("From {price}/month.").format(price=price)
    return tr("From {price}/month, {days}-day free trial.").format(
        price=price, days=days)


def join_offer_line(sentence: str | None, phrase: str) -> str:







    base = (sentence or "").strip()
    phrase = (phrase or "").strip()
    if not phrase:
        return base
    if not base:
        return phrase
    if not base.endswith((".", "!", "?", "\u3002")):
        base += "."
    return f"{base} {phrase}"
