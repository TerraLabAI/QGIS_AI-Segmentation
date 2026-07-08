







from __future__ import annotations




RECAP_LAYER_LINK = "recap-layer"


def format_ground_area(area_m2) -> str:







    try:
        m2 = float(area_m2 or 0.0)
    except (TypeError, ValueError):
        return ""
    if m2 <= 0.0:
        return ""

    from .ui_refresh_credits import grouped_locale
    loc = grouped_locale()
    if m2 < 10_000.0:

        return f"{m2:,.0f}".replace(",", " ") + " m²"
    if m2 < 1_000_000.0:
        return loc.toString(m2 / 10_000.0, "f", 2) + " ha"
    return loc.toString(m2 / 1_000_000.0, "f", 2) + " km²"
