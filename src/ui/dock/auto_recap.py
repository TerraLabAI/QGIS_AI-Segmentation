"""What the Automatic Start page says about the run that just finished.

Two lines about the same run: the success line right after Finish, and the quiet
session memory that takes over once it is dismissed. Both stop at the count and
the layer, and the layer name is a link that frames it on the map, because
reaching the result is what the user needs there. The zone area used to sit on
both lines and answered nothing (the footer credit ring already carries the
balance, and the zone is on screen while it is drawn).
"""
from __future__ import annotations

from html import escape

from qgis.PyQt.QtCore import QLocale

from ...core.i18n import tr
from .manual_recap import RECAP_LAYER_LINK
from .styles import BRAND_BLUE


def layer_link_html(layer_name: str, linked: bool) -> str:
    """The layer name for a recap line: a link when the dock can still resolve
    the layer, plain text when it cannot. Escaped either way, so a layer called
    "Roofs & walls" reads as itself and can never inject markup."""
    name = escape((layer_name or "").strip())
    if not name:
        return ""
    if not linked:
        return name
    return (f'<a href="{RECAP_LAYER_LINK}" style="color: {BRAND_BLUE};'
            f' text-decoration: none;">{name}</a>')


def auto_export_success_html(count: int, object_word: str, layer_name: str,
                             linked: bool = True) -> str:
    """The line shown right after Finish: how many objects were saved and where.
    Rich text, so it must be set on a RichText label."""
    count = int(count)
    word = (object_word or "").strip()
    if not word:
        # The generic fallback is the only word this function ever singles or
        # plurals itself: a user-typed class name is used exactly as given,
        # in both counts, because it cannot be conjugated by rule.
        word = tr("polygon") if count == 1 else tr("polygons")
    obj = escape(word)
    n = QLocale().toString(count)
    link = layer_link_html(layer_name, linked)
    if not link:
        return tr("{n} {object} saved").format(n=n, object=obj)
    return tr("{n} {object} saved to {layer}").format(
        n=n, object=obj, layer=link)


# auto_last_run_html was removed on 2026-08-11 with the card it filled. It read
# "Last run: 69 building in Building 4 (11 Aug) · 14 credits" and stayed on the
# Start page for the rest of the session, repeating the legend and the footer
# credit ring on the one screen that is about the NEXT run.
