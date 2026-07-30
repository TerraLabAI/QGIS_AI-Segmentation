"""What the Manual Start view says about the session that just ended.

One card, three facts: how many polygons were exported, how much ground they
cover, and the layer they landed in, as a link that frames it on the map. The
old line stopped at the count and a km2 figure that read "0.00" for anything
smaller than a city block, so it told a user nothing and pointed nowhere.
"""
from __future__ import annotations

from html import escape

from ...core.i18n import tr
from .styles import BRAND_BLUE

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


def manual_recap_html(count: int, area_m2, layer_name: str | None) -> str:
    """The recap card's rich text: the count, the area, and where it went.

    Rich text because the layer name is a link. Every value from the project
    is escaped: a layer called "Roofs & walls" must read as itself, not as an
    entity, and must not be able to inject markup.
    """
    head = tr("{count} polygon(s) exported").format(count=int(count))
    area = format_ground_area(area_m2)
    if area:
        head = f"{head} · {area}"
    line = f"<b>{escape(head)}</b>"
    name = (layer_name or "").strip()
    if not name:
        return line
    link = (f'<a href="{RECAP_LAYER_LINK}" style="color: {BRAND_BLUE};'
            f' text-decoration: none;">{escape(name)}</a>')
    return line + "<br>" + tr("Saved to {layer}").format(layer=link)
