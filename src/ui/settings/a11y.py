












from __future__ import annotations

import re

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QAbstractButton

from ..dock.styles import (
    _BTN_SETTINGS_ACCENT,
    _BTN_SETTINGS_DANGER,
    _BTN_SETTINGS_GHOST,
    _BTN_SETTINGS_INK,
    _BTN_SETTINGS_RAIL_CTA,
    _BTN_SETTINGS_STEP,
    _BTN_SETTINGS_TEXT,
    BRAND_BLUE,
    DISABLED_FILL,
    DISABLED_INK,
    INK_3,
    LINK_INK,
    ORANGE_TEXT,
    RED_INK,
)






DANGER_INK = RED_INK

WARN_INK = ORANGE_TEXT

CONTROL_BORDER = INK_3

SWITCH_OFF_ALPHA = 0.55
FOCUS_RING = BRAND_BLUE

_PADDING = re.compile(r"padding:\s*(\d+)px\s+(\d+)px")


_GHOST_FOCUS = (f"QPushButton:focus {{ border: 2px solid {FOCUS_RING};"
                " padding: 5px 13px; }")


def _ring_without_border(qss: str, width: int) -> str:





    match = _PADDING.search(qss)
    extra = ""
    if match:
        vertical = max(0, int(match.group(1)) - width)
        horizontal = max(0, int(match.group(2)) - width)
        extra = f" padding: {vertical}px {horizontal}px;"
    ring = f"QPushButton {{ border: {width}px solid transparent;{extra} }}"
    return qss + ring + f"QPushButton:focus {{ border-color: {FOCUS_RING}; }}"


def accessible_qss(qss: str) -> str:





    qss = qss or ""
    if qss == _BTN_SETTINGS_ACCENT:



        waiting = (f"QPushButton:disabled {{ background: {DISABLED_FILL};"
                   f" color: {DISABLED_INK}; }}")
        return _ring_without_border(qss, 2) + waiting
    if qss in (_BTN_SETTINGS_INK, _BTN_SETTINGS_RAIL_CTA):
        return _ring_without_border(qss, 2)
    if qss == _BTN_SETTINGS_TEXT:

        return _ring_without_border(qss, 1)
    if qss == _BTN_SETTINGS_STEP:
        return qss + _GHOST_FOCUS
    if qss == _BTN_SETTINGS_DANGER:
        ink = f"QPushButton {{ border-color: {CONTROL_BORDER}; color: {DANGER_INK}; }}"
        return qss + ink + f"QPushButton:hover {{ border-color: {DANGER_INK}; }}" + _GHOST_FOCUS
    if qss == _BTN_SETTINGS_GHOST:
        return qss + f"QPushButton {{ border-color: {CONTROL_BORDER}; }}" + _GHOST_FOCUS
    return qss


def make_accessible(button: QAbstractButton, qss: str | None = None,
                    name: str = "") -> QAbstractButton:

    if qss is not None:
        button.setStyleSheet(accessible_qss(qss))
    button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    if name and not button.accessibleName():
        button.setAccessibleName(name)
    return button


__all__ = [
    "CONTROL_BORDER",
    "DANGER_INK",
    "FOCUS_RING",
    "LINK_INK",
    "SWITCH_OFF_ALPHA",
    "WARN_INK",
    "accessible_qss",
    "make_accessible",
]
