"""Starred catalogue templates, persisted via QSettings.

The Segment library's Favorites tab holds two halves. Detection runs are server
state, synced through the backend's per-run ``is_favorite``. Templates are
client-side catalogue entries, so starring one is a local preference: no field
for it exists on the server, the ids never leave the machine, and the list is
per profile like the Recent objects next door.

Stored as one JSON list of catalogue ids under
``AISegmentation/favorite_templates``, most recently starred first so the tab
has an order the user can predict.

Every read fails safe to an empty list. The library dialog builds its Favorites
tab from this on open, so an absent, truncated or hand-edited settings value
must cost the user their stars, never the dialog.
"""
from __future__ import annotations

import json

from qgis.PyQt.QtCore import QSettings

_FAVORITE_TEMPLATES_KEY = "AISegmentation/favorite_templates"

# Cap on stored ids. The catalogue holds a few dozen templates, so a list an
# order of magnitude longer can only come from a stale or corrupt value, and
# the whole list is rewritten as one JSON blob on every star.
_FAVORITE_TEMPLATES_CAP = 200


def favorite_template_ids() -> list[str]:
    """Starred catalogue ids, most recently starred first.

    Ids are returned as stored, never checked against the catalogue: the
    catalogue comes from the backend and can change, so a starred template that
    disappears simply stops being shown."""
    raw = QSettings().value(_FAVORITE_TEMPLATES_KEY, "")
    if not raw or not isinstance(raw, str):
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if not isinstance(item, str):
            continue
        template_id = item.strip()
        if template_id and template_id not in ids:
            ids.append(template_id)
    return ids[:_FAVORITE_TEMPLATES_CAP]


def is_favorite_template(template_id: str) -> bool:
    """True when this catalogue id is starred."""
    wanted = (template_id or "").strip()
    return bool(wanted) and wanted in favorite_template_ids()


def set_favorite_template(template_id: str, favorite: bool) -> None:
    """Star or unstar one template. Starring moves it back to the front."""
    wanted = (template_id or "").strip()
    if not wanted:
        return
    ids = [i for i in favorite_template_ids() if i != wanted]
    if favorite:
        ids.insert(0, wanted)
    _save_favorite_template_ids(ids)


def toggle_favorite_template(template_id: str) -> bool:
    """Flip one template's star and return the state it now has."""
    wanted = (template_id or "").strip()
    if not wanted:
        return False
    favorite = not is_favorite_template(wanted)
    set_favorite_template(wanted, favorite)
    return favorite


def clear_favorite_templates() -> None:
    """Drop every starred template."""
    QSettings().remove(_FAVORITE_TEMPLATES_KEY)


def _save_favorite_template_ids(ids: list[str]) -> None:
    QSettings().setValue(
        _FAVORITE_TEMPLATES_KEY, json.dumps(ids[:_FAVORITE_TEMPLATES_CAP], ensure_ascii=False))
