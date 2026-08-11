










from __future__ import annotations

import json

from qgis.PyQt.QtCore import QSettings


def _bounded_setting_oversized(value, max_bytes: int) -> bool:
    return isinstance(value, str) and len(value.encode("utf-8")) > max_bytes


def read_bounded_json_setting(key: str, max_bytes: int):

    raw = QSettings().value(key, "")
    if not raw or not isinstance(raw, str) or _bounded_setting_oversized(raw, max_bytes):
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def write_bounded_json_setting(key: str, value, max_bytes: int) -> bool:

    settings = QSettings()
    if _bounded_setting_oversized(settings.value(key, ""), max_bytes):
        return False
    payload = json.dumps(value, ensure_ascii=False)
    if _bounded_setting_oversized(payload, max_bytes):
        return False
    settings.setValue(key, payload)
    return True


def remove_bounded_json_setting(key: str, max_bytes: int) -> bool:

    if _bounded_setting_oversized(QSettings().value(key, ""), max_bytes):
        return False
    QSettings().remove(key)
    return True
