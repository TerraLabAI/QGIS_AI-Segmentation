










from __future__ import annotations

from datetime import datetime


def format_quota_reset_date(iso: str | None) -> str:








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
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return parsed.strftime("%B %d, %Y")
