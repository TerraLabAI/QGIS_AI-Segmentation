







from __future__ import annotations




RECAP_LAYER_LINK = "recap-layer"


def format_ground_area(area_m2) -> str:







    try:
        m2 = float(area_m2 or 0.0)
    except (TypeError, ValueError):
        return ""
    if m2 <= 0.0:
        return ""
    if m2 < 10_000.0:
        return f"{m2:,.0f}".replace(",", " ") + " m²"
    if m2 < 1_000_000.0:
        return f"{m2 / 10_000.0:.2f} ha"
    return f"{m2 / 1_000_000.0:.2f} km²"
