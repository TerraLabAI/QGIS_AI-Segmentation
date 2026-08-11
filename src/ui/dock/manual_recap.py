"""The two pieces the Automatic recap card is built from.

It was the Semi-Auto recap's own module until 2026-08-11, when that card was
removed: a line counting what the last session produced sat on the Start view
for the rest of the session, and the saved layer in the legend says the same
thing without taking a quarter of the panel. What is left here is what
``auto_recap.py`` still reads.
"""
from __future__ import annotations

# Href scheme for the layer link. The recap label carries the layer id, and the
# dock resolves it against the project when the link is clicked, so a layer the
# user has since removed simply does nothing.
RECAP_LAYER_LINK = "recap-layer"


def format_ground_area(area_m2) -> str:
    """Ground area in the unit that keeps the number readable.

    A handful of buildings covers a few thousand square metres, which prints as
    "0.00 km2" and carries no information. The unit follows the size instead:
    square metres under a hectare, hectares under a square kilometre, square
    kilometres above. Empty string when there is nothing to show.
    """
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
