















from __future__ import annotations

import json

from qgis.PyQt.QtCore import QSettings

_FAVORITE_TEMPLATES_KEY = "AISegmentation/favorite_templates"




_FAVORITE_TEMPLATES_CAP = 200
_MAX_FAVORITE_BYTES = 128 * 1024


def favorite_template_ids() -> list[str]:





    raw = QSettings().value(_FAVORITE_TEMPLATES_KEY, "")
    if not raw or not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_FAVORITE_BYTES:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    from ..server_dials import dial_in_range
    cap = dial_in_range("tuning.library.favorite_templates_cap", _FAVORITE_TEMPLATES_CAP, 20, 2000)
    ids: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, str):
            continue
        template_id = item.strip()
        if template_id and template_id not in seen:
            seen.add(template_id)
            ids.append(template_id)
            if len(ids) >= cap:
                break
    return ids[:cap]


def is_favorite_template(template_id: str) -> bool:

    wanted = (template_id or "").strip()
    return bool(wanted) and wanted in favorite_template_ids()


def set_favorite_template(template_id: str, favorite: bool) -> None:

    wanted = (template_id or "").strip()
    if not wanted:
        return
    ids = [i for i in favorite_template_ids() if i != wanted]
    if favorite:
        ids.insert(0, wanted)
    _save_favorite_template_ids(ids)


def toggle_favorite_template(template_id: str) -> bool:

    wanted = (template_id or "").strip()
    if not wanted:
        return False
    favorite = not is_favorite_template(wanted)
    ids = [i for i in favorite_template_ids() if i != wanted]
    if favorite:
        ids.insert(0, wanted)
    if _save_favorite_template_ids(ids):
        return favorite
    return wanted in favorite_template_ids()


def clear_favorite_templates() -> None:

    QSettings().remove(_FAVORITE_TEMPLATES_KEY)


def _save_favorite_template_ids(ids: list[str]) -> bool:
    from ..server_dials import dial_in_range
    cap = dial_in_range("tuning.library.favorite_templates_cap", _FAVORITE_TEMPLATES_CAP, 20, 2000)
    settings = QSettings()
    existing = settings.value(_FAVORITE_TEMPLATES_KEY, "")
    if isinstance(existing, str) and len(existing.encode("utf-8")) > _MAX_FAVORITE_BYTES:
        return False
    payload = json.dumps(ids[:cap], ensure_ascii=False)
    if len(payload.encode("utf-8")) > _MAX_FAVORITE_BYTES:
        return False
    settings.setValue(_FAVORITE_TEMPLATES_KEY, payload)
    return True
