"""The day a monthly quota comes back, formatted for display.

The server sends it as an ISO timestamp (``period_end``) and it is the least
guessable date there is: the quota renews on the sign-up anniversary, so a user
who signed up on the 7th has no way to work out that the counter refills on the
7th. Every surface that shows a balance shows this beside it, which is why the
formatting lives here once instead of at each call site.

Pure Python apart from one optional Qt call for the locale, so it imports and
tests in a bare interpreter.
"""
from __future__ import annotations

from datetime import datetime


def format_quota_reset_date(iso: str | None) -> str:
    """The reset day in the user's locale, or ``""`` when it is not known.

    Returns ``""`` for anything missing or unparseable rather than the raw
    input: an empty result hides the line, while an ISO timestamp on screen
    reads as a bug. An aware timestamp is converted to LOCAL time first, so a
    reset stamped just after midnight UTC does not display as the next day for
    a user west of it.
    """
    text = str(iso or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    try:
        from qgis.PyQt.QtCore import QDate, QLocale

        from .qt_compat import resolve_qt_enum

        long_format = resolve_qt_enum(QLocale, "FormatType", "LongFormat")
        formatted = QLocale().toString(
            QDate(parsed.year, parsed.month, parsed.day), long_format)
        if formatted:
            return str(formatted)
    except Exception:  # noqa: BLE001 -- outside QGIS, or a locale Qt cannot format  # nosec B110
        pass
    return parsed.strftime("%B %d, %Y")
