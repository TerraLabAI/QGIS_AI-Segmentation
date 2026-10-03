








from __future__ import annotations

from html import escape

from ...core.i18n import tr
from .manual_recap import RECAP_LAYER_LINK
from .styles import LINK_INK
from .ui_refresh_credits import grouped_locale


def layer_link_html(layer_name: str, linked: bool) -> str:



    name = escape((layer_name or "").strip())
    if not name:
        return ""
    if not linked:
        return name
    return (f'<a href="{RECAP_LAYER_LINK}" style="color: {LINK_INK};'
            f' text-decoration: none;">{name}</a>')


def auto_export_success_html(count: int, object_word: str, layer_name: str,
                             linked: bool = True) -> str:


    count = int(count)
    word = (object_word or "").strip()
    if not word:



        word = tr("polygon") if count == 1 else tr("polygons")
    obj = escape(word)
    n = grouped_locale().toString(count)
    link = layer_link_html(layer_name, linked)
    if not link:
        return tr("{n} {object} saved").format(n=n, object=obj)
    return tr("{n} {object} saved to {layer}").format(
        n=n, object=obj, layer=link)






