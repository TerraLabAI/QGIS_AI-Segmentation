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
    obj = escape((object_word or "").strip() or tr("polygons"))
    link = layer_link_html(layer_name, linked)
    if not link:
        return tr("{n} {object} saved").format(n=int(count), object=obj)
    return tr("{n} {object} saved to {layer}").format(
        n=int(count), object=obj, layer=link)


def auto_last_run_html(count: int, object_word: str, layer_name: str,
                       credits_used, linked: bool = True) -> str:
    """The session memory the Start page keeps once the success line is gone:
    the same count and layer, plus what the run cost. Rich text."""
    obj = escape((object_word or "").strip() or tr("polygons"))
    link = layer_link_html(layer_name, linked)
    head = (tr("Last run: {n} {object} in {layer}").format(
                n=int(count), object=obj, layer=link)
            if link else
            tr("Last run: {n} {object}").format(n=int(count), object=obj))
    if credits_used is None:
        return head
    return head + " · " + escape(
        tr("{used} credits").format(used=credits_used))
