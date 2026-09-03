"""The price line every Pro card shows, built from what the server serves.

A price belongs to the billing side, not to a plugin release. Users upgrade
through the marketplace over months, so a number compiled into a card is a
number we cannot change: the plugin would keep offering last season's price
long after the website moved. The server carries it in the plugin config
instead, under ``pricing``, and this module turns it into one short sentence.

    pricing = {
      "currency": ...,
      "trial_days": ...,
      "<product>": {"month_eur": ..., "year_eur": ...},
      ...
    }

Nothing here is required. An older server, a config that has not arrived yet
and a malformed number all return an empty string, and the caller shows its
line without a price rather than a placeholder.
"""
from __future__ import annotations

from .i18n import tr

# The product a plugin CTA sells. Named here so a future pack CTA reads its
# own key rather than a second copy of this function.
PRO_PRICING_KEY = "ai_segmentation_pro"

# A price outside this band is a unit mistake (cents served as euros, a test
# row, a decimal that lost its point), and a wrong price on a card is worse
# than no price at all.
_PRICE_MIN_EUR = 1
_PRICE_MAX_EUR = 10_000

_CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£"}


def _pricing_blob() -> dict:
    try:
        from .config_cache import get_config
        pricing = get_config().get("pricing")
    except Exception:  # noqa: BLE001 -- a card never fails on its price
        return {}
    return pricing if isinstance(pricing, dict) else {}


def _monthly_amount(product_key: str) -> int | None:
    """The whole euros a month, or None when the server serves no usable one."""
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
    """``39`` in the currency the server named, e.g. ``39 EUR`` or ``€39``."""
    currency = str(_pricing_blob().get("currency") or "EUR").strip().upper()
    symbol = _CURRENCY_SYMBOLS.get(currency)
    return f"{symbol}{amount}" if symbol else f"{amount} {currency}"


def pro_price_phrase(product_key: str = PRO_PRICING_KEY) -> str:
    """One short offer, or ``""`` when the server serves no price.

    "From EUR39/month, 7-day free trial." when both numbers are there, the
    price alone when the trial is not offered. "From" is not a flourish: the
    yearly plan costs less a month, and a plugin that quotes one number as
    the price contradicts the page the See all plans link opens.
    """
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
    """``sentence`` with ``phrase`` added to it, in the same line.

    The cards are already dense and a wall that grows a line to hold a price
    pushes its own button below the fold. So the price joins the sentence the
    card already had rather than taking one of its own, and a card with no
    such sentence gets the price by itself.
    """
    base = (sentence or "").strip()
    phrase = (phrase or "").strip()
    if not phrase:
        return base
    if not base:
        return phrase
    if not base.endswith((".", "!", "?", "\u3002")):
        base += "."
    return f"{base} {phrase}"
