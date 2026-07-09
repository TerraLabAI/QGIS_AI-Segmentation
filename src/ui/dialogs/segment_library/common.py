





from __future__ import annotations

import calendar
import os
import time
from datetime import date

from qgis.PyQt.QtCore import QLocale
from qgis.PyQt.QtWidgets import QLabel, QWidget

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.presets.segmentation_presets_client import absolute_demo_url
from ...dock.font_scale import scale_qss_font_px as _scale_qss_font_px
from ...dock.styles import (
    _BTN_GHOST,
    _BTN_GREEN_STEP,
    _BTN_PRIMARY,
    _TOOLTIP_QSS,
    ACCENT_BORDER,
    ACCENT_BORDER_SOFT,
    BTN_SMALL_PX,
    FIELD,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    FONT_MICRO,
    HOVER,
    HOVER_ON,
    INK,
    INK_2,
    INK_3,
    INSET,
    LINE,
    LINE_INPUT,
    LINE_STRONG,
    PAGE,
    RADIUS_CARD,
    RADIUS_CHIP,
    RADIUS_CONTROL,
    SURFACE,
)



_RAIL_RECENT_TARGET = "__recent__"
_RAIL_FAVORITES_TARGET = "__favorites__"
_RAIL_POPULAR_TARGET = "__top__"
_RAIL_HISTORY_VIEWS = {
    _RAIL_RECENT_TARGET: "all",
    _RAIL_FAVORITES_TARGET: "favorites",
}









_DIALOG_QSS = (
    f"QDialog#segmentLibrary {{ background: {PAGE}; }}"
    + _TOOLTIP_QSS
)


def _apply_library_ground(dialog) -> None:

    dialog.setObjectName("segmentLibrary")
    dialog.setStyleSheet(_DIALOG_QSS)





_SEARCH_QSS = (
    f"QLineEdit {{ background: {FIELD}; border: 1px solid {LINE_INPUT};"
    f" border-radius: {RADIUS_CONTROL}px; padding: 7px 10px;"
    f" font-size: {FONT_BASE}px; color: {INK};"
    f" selection-background-color: {HOVER_ON}; selection-color: {INK}; }}"
    f"QLineEdit:hover {{ border-color: {ACCENT_BORDER_SOFT}; }}"
    f"QLineEdit:focus {{ border-color: {ACCENT_BORDER}; }}"
)


_CARD_NORMAL = (
    f"QFrame#card {{ border: 1px solid {LINE};"
    f" border-radius: {RADIUS_CARD}px; background: {SURFACE}; }}"
)
_CARD_HOVER = (
    f"QFrame#card {{ border: 1px solid {LINE_STRONG};"
    f" border-radius: {RADIUS_CARD}px; background: {HOVER}; }}"
)


_OVERLAY_BADGE_QSS = (
    "QLabel { background: rgba(0,0,0,0.62); color: #ffffff;"
    f" font-size: {FONT_MICRO}px; font-weight: 600; border: none;"
    f" border-radius: {RADIUS_CHIP}px; padding: 2px 7px; }}"
)
_META_QSS = (
    f"font-size: {FONT_HINT}px; color: {INK_2};"
    " background: transparent; border: none;"
)


_EMPTY_GLYPH = "background: transparent; border: none;"
_EMPTY_MSG = (
    f"color: {INK_2}; font-size: {FONT_BASE}px;"
    " background: transparent; border: none;"
)


_GHOST_BTN_QSS = _BTN_GHOST


_ICON_BTN_QSS = (
    "QToolButton { background: transparent; border: none; padding: 0;"
    f" border-radius: {RADIUS_CONTROL}px; }}"
    f"QToolButton:hover {{ background: {HOVER}; }}"
    f"QToolButton:pressed, QToolButton:checked:pressed {{ background: {HOVER_ON}; }}"
    "QToolButton:checked { background: transparent; }"
    "QToolButton:disabled { background: transparent; }"
    "QToolButton::menu-indicator { image: none; width: 0; }"
)
_STAR_BTN_QSS = _ICON_BTN_QSS




_TITLE_STYLE = (
    f"color: {INK}; font-size: {FONT_BASE + 5}px; font-weight: 600;"
    " background: transparent; border: none;"
)
_SECTION_STYLE = (
    f"color: {INK_2}; font-size: {FONT_HINT}px; font-weight: 600;"
    " background: transparent; border: none;"
)

_BADGE_STYLE = (
    f"QLabel {{ color: {INK_2}; background: {FIELD};"
    f" border: 1px solid {LINE}; border-radius: {RADIUS_CHIP}px;"
    f" font-size: {FONT_HINT}px; font-weight: 500; padding: 2px 8px; }}"
)
_SEPARATOR = f"background: {LINE}; border: none;"

_PROMPT_STYLE = (
    f"QLabel {{ color: {INK}; font-size: {FONT_BODY}px;"
    " font-family: Consolas, 'DejaVu Sans Mono', Menlo, monospace;"
    f" background: {FIELD}; border: 1px solid {LINE};"
    f" border-radius: {RADIUS_CONTROL}px; padding: 9px 12px; }}"
)


_COPY_BTN = (
    "QPushButton { background: transparent; border: none;"
    f" color: {INK_2}; font-size: {FONT_BODY}px; font-weight: 500;"
    f" padding: 4px 8px; border-radius: {RADIUS_CONTROL}px; }}"
    f"QPushButton:hover {{ background: {HOVER}; color: {INK}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; color: {INK}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; }}"
)
_CHIP_STYLE = (
    f"QFrame {{ background: {INSET};"
    f" border: 1px solid {LINE}; border-radius: {RADIUS_CONTROL}px; }}"
)
_CHIP_CAPTION = (
    f"color: {INK_2}; font-size: {FONT_MICRO}px; font-weight: 500;"
    " background: transparent; border: none;"
)
_CHIP_VALUE = (
    f"color: {INK}; font-size: {FONT_BODY}px; font-weight: 600;"
    " background: transparent; border: none;"
)
_ACTION_BTN = _BTN_GHOST
_PRIMARY_BTN = _BTN_PRIMARY


_PRIMARY_WIDE_BTN = _BTN_GREEN_STEP


_FS_BTN = (
    "QToolButton { background: rgba(0,0,0,0.55); border: none;"
    f" border-radius: {RADIUS_CONTROL}px; padding: 0; }}"
    "QToolButton:hover { background: rgba(0,0,0,0.8); }"
)
_DETAIL_STAR_BTN = (
    f"QToolButton {{ background: {SURFACE}; border: 1px solid {LINE_STRONG};"
    f" border-radius: {RADIUS_CONTROL}px; padding: 0; }}"
    f"QToolButton:hover {{ background: {HOVER}; }}"
    f"QToolButton:pressed {{ background: {HOVER_ON}; }}"
)








_RAIL_PANEL = (
    "QFrame#librail { border: none;"
    f" border-right: 1px solid {LINE}; background: transparent; }}"
)

_RAIL_GROUP = (
    f"QLabel {{ color: {INK_3}; font-size: {FONT_HINT}px; font-weight: 500;"
    " background: transparent; border: none; }"
)
_RAIL_ITEM_LABEL = (
    f"QLabel {{ color: {INK}; font-size: {FONT_BODY}px;"
    " background: transparent; border: none; }"
)

_RAIL_ITEM_COUNT = (
    f"QLabel {{ color: {INK_3}; font-size: {FONT_HINT}px;"
    " background: transparent; border: none; }"
)


def _rail_item_style(active: bool) -> str:



    base = (
        "QPushButton#railitem { text-align: left; border: none;"
        f" border-radius: {RADIUS_CONTROL}px; padding: 6px 10px;"
        f" font-size: {FONT_BODY}px; color: {INK};"
    )
    if active:
        qss = base + (f" border-left: 3px solid {ACCENT_BORDER};"
                      f" background: {HOVER_ON}; }}")
    else:
        qss = (base + " border-left: 3px solid transparent;"
               " background: transparent; }"
               f"QPushButton#railitem:hover {{ background: {HOVER}; }}"
               f"QPushButton#railitem:pressed {{ background: {HOVER_ON}; }}")
    return _scale_qss_font_px(qss)


def _rail_label_style(active: bool) -> str:


    if not active:
        return _RAIL_ITEM_LABEL
    return _scale_qss_font_px(
        f"QLabel {{ color: {INK}; font-size: {FONT_BODY}px;"
        " font-weight: 600; background: transparent; border: none; }"
    )







_EMPTY_GLYPH_PX = 28
_ICON_GLYPH_PX = 16


def _star_path(filled: bool):

    import math

    from qgis.PyQt.QtCore import QPointF
    from qgis.PyQt.QtGui import QPainterPath

    path = QPainterPath()
    cx, cy, outer, inner = 10.0, 10.6, 7.6, 3.4
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        angle = math.radians(-90 + i * 36)
        point = QPointF(cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        if i == 0:
            path.moveTo(point)
        else:
            path.lineTo(point)
    path.closeSubpath()
    return path


def _star_pixmap(widget, filled: bool, size: int = _ICON_GLYPH_PX):


    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtGui import QColor, QPainter, QPen, QPixmap

    from ...dock.font_scale import widget_pixel_ratio
    from ...icons import ink_of

    ratio = widget_pixel_ratio(widget)
    physical = max(1, int(round(size * ratio)))
    pixmap = QPixmap(physical, physical)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    ink = QColor(ink_of(widget))
    if not filled:
        ink.setAlphaF(0.72)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.scale(size / 20.0, size / 20.0)
        pen = QPen(ink, 1.5)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(ink if filled else Qt.BrushStyle.NoBrush)
        painter.drawPath(_star_path(filled))
    finally:
        painter.end()
    return pixmap


def _set_star_glyph(button, filled: bool, size: int = _ICON_GLYPH_PX) -> None:

    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QIcon

    button.setText("")
    button.setIcon(QIcon(_star_pixmap(button, filled, size)))
    button.setIconSize(QSize(size, size))


def _set_tool_glyph(button, name: str, size: int = _ICON_GLYPH_PX, color=None) -> None:

    from qgis.PyQt.QtCore import QSize

    from ...icons import icon_for

    button.setText("")
    button.setIcon(icon_for(button, name, size, color))
    button.setIconSize(QSize(size, size))


def _icon_button(parent, name: str, tooltip: str, accessible: str = ""):

    from qgis.PyQt.QtWidgets import QToolButton

    from ...dock.font_scale import scale_px_length

    btn = QToolButton(parent)
    btn.setStyleSheet(_ICON_BTN_QSS)
    btn.setCursor(QtC.PointingHandCursor)
    side = scale_px_length(BTN_SMALL_PX)
    btn.setFixedSize(side, side)
    _set_tool_glyph(btn, name)
    btn.setToolTip(tooltip)
    btn.setAccessibleName(accessible or tooltip)
    return btn



_EMPTY_GLYPH_DEFAULT = "layers"


def _empty_glyph_pixmap(widget, name: str):

    from qgis.PyQt.QtGui import QColor

    from ...icons import ink_of, pixmap_for

    size = _EMPTY_GLYPH_PX
    if name == "star":
        return _star_pixmap(widget, False, size)
    ink = QColor(ink_of(widget))
    ink.setAlphaF(0.55)
    return pixmap_for(widget, name or _EMPTY_GLYPH_DEFAULT, size, ink)


def _build_use_hint(parent) -> QLabel:

    hint = QLabel(parent)
    hint.setAttribute(QtC.WA_TransparentForMouseEvents)
    hint.setStyleSheet("background: transparent; border: none;")
    _set_use_hint(hint, False)
    return hint


def _set_use_hint(hint: QLabel, hovered: bool) -> None:
    from qgis.PyQt.QtGui import QColor

    from ...icons import ink_of, pixmap_for

    ink = QColor(ink_of(hint))
    if not hovered:
        ink.setAlphaF(0.45)
    hint.setPixmap(pixmap_for(hint, "chevron_right", _ICON_GLYPH_PX, ink))
    hint.setToolTip(tr("Use") if hovered else "")


class _AspectBox(QWidget):




    def __init__(self, child: QWidget, ratio: float, parent=None):
        super().__init__(parent)
        self._child = child
        child.setParent(self)
        self._ratio = ratio if ratio and ratio > 0 else 1.0
        self._overlay: QWidget | None = None
        self._overlay_margin = 10

    def set_ratio(self, ratio: float) -> None:
        self._ratio = ratio if ratio and ratio > 0 else 1.0
        self._relayout()

    def set_overlay(self, widget: QWidget) -> None:

        self._overlay = widget
        widget.setParent(self)
        widget.raise_()
        self._relayout()

    def resizeEvent(self, event):  # noqa: N802
        self._relayout()
        super().resizeEvent(event)

    def _relayout(self) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        if w / h > self._ratio:
            ch = h
            cw = int(round(h * self._ratio))
        else:
            cw = w
            ch = int(round(w / self._ratio))
        cx, cy = (w - cw) // 2, (h - ch) // 2
        self._child.setGeometry(cx, cy, cw, ch)
        if self._overlay is not None:
            ow = self._overlay.width()
            oh = self._overlay.height()
            m = self._overlay_margin
            self._overlay.move(cx + cw - ow - m, cy + ch - oh - m)
            self._overlay.raise_()







def _demo_url(base: str, preset: dict, which: str, preview: bool = False) -> str:

    rel = preset.get(f"demo_url_{which}") or (
        f"/api/ai-segmentation/template-demos/{preset.get('id', '')}/{which}")
    url = absolute_demo_url(base, rel)
    if preview and url:
        url += ("&" if "?" in url else "?") + "size=preview"
    return url


def _relative_when(ts: str) -> str:





    try:
        parsed = time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        stamp = calendar.timegm(parsed)
    except (ValueError, TypeError):
        return ""



    try:
        then = date(*time.localtime(stamp)[:3])
        days = (date.today() - then).days
    except (ValueError, OverflowError, OSError):
        days = int((time.time() - stamp) // 86400)
    if days <= 0:
        return tr("today")
    if days == 1:
        return tr("yesterday")
    if days < 7:
        return tr("{n} days ago").format(n=days)
    if days < 31:
        weeks = days // 7
        return tr("a week ago") if weeks == 1 else tr("{n} weeks ago").format(n=weeks)
    if days < 365:
        months = max(1, days // 30)
        return tr("a month ago") if months == 1 else tr("{n} months ago").format(n=months)
    years = days // 365
    return tr("a year ago") if years == 1 else tr("{n} years ago").format(n=years)


def _iso_norm(ts) -> str:






    ts = str(ts or "").strip()
    if len(ts) < 19:
        return ""
    body, tail = ts[:19], ts[19:]

    if tail.startswith("."):
        idx = 1
        while idx < len(tail) and tail[idx].isdigit():
            idx += 1
        tail = tail[idx:]
    if not tail or tail in ("Z", "z", "+00:00", "-00:00", "+0000", "-0000"):
        return body + "Z"
    sign = tail[0]
    if sign not in ("+", "-"):
        return body + "Z"
    digits = tail[1:].replace(":", "")
    if len(digits) < 4 or not digits[:4].isdigit():
        return body + "Z"
    try:
        parsed = time.strptime(body, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return body + "Z"
    shift = (int(digits[:2]) * 3600 + int(digits[2:4]) * 60) * (-1 if sign == "+" else 1)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                         time.gmtime(calendar.timegm(parsed) + shift))


def _fmt_count(value) -> str:

    try:
        return QLocale().toString(int(value or 0))
    except (TypeError, ValueError):
        return "0"


def _project_layer_reading(source: str):





    if not source:
        return None
    from qgis.core import QgsProject

    want = os.path.normcase(str(source))
    for layer in QgsProject.instance().mapLayers().values():
        try:
            if os.path.normcase(str(layer.source())) == want:
                return layer
        except (RuntimeError, AttributeError):
            continue
    return None


def _run_key(run: dict) -> str:

    return str(run.get("run_id") or run.get("group_key") or "")


def _history_error(resp) -> str | None:

    if not isinstance(resp, dict):
        return "parse_error"
    if resp.get("error"):
        return str(resp.get("code") or "SERVER_ERROR")
    return None







for _qss_name in (
    "_RAIL_GROUP",
    "_RAIL_ITEM_LABEL",
    "_RAIL_ITEM_COUNT",
    "_SEARCH_QSS",
    "_OVERLAY_BADGE_QSS",
    "_META_QSS",
    "_EMPTY_MSG",
    "_ICON_BTN_QSS",
    "_TITLE_STYLE",
    "_SECTION_STYLE",
    "_BADGE_STYLE",
    "_PROMPT_STYLE",
    "_COPY_BTN",
    "_CHIP_CAPTION",
    "_CHIP_VALUE",
    "_FS_BTN",
    "_DETAIL_STAR_BTN",
):
    globals()[_qss_name] = _scale_qss_font_px(globals()[_qss_name])
del _qss_name



__all__ = [
    "_ACTION_BTN",
    "_AspectBox",
    "_apply_library_ground",
    "_ICON_BTN_QSS",
    "_empty_glyph_pixmap",
    "_icon_button",
    "_set_star_glyph",
    "_set_tool_glyph",
    "_BADGE_STYLE",
    "_CARD_HOVER",
    "_CARD_NORMAL",
    "_CHIP_CAPTION",
    "_CHIP_STYLE",
    "_CHIP_VALUE",
    "_COPY_BTN",
    "_DETAIL_STAR_BTN",
    "_EMPTY_GLYPH",
    "_EMPTY_MSG",
    "_FS_BTN",
    "_GHOST_BTN_QSS",
    "_META_QSS",
    "_OVERLAY_BADGE_QSS",
    "_PRIMARY_BTN",
    "_PRIMARY_WIDE_BTN",
    "_PROMPT_STYLE",
    "_RAIL_FAVORITES_TARGET",
    "_RAIL_GROUP",
    "_RAIL_HISTORY_VIEWS",
    "_RAIL_ITEM_COUNT",
    "_RAIL_PANEL",
    "_RAIL_POPULAR_TARGET",
    "_RAIL_RECENT_TARGET",
    "_SEARCH_QSS",
    "_SECTION_STYLE",
    "_SEPARATOR",
    "_STAR_BTN_QSS",
    "_TITLE_STYLE",
    "_build_use_hint",
    "_demo_url",
    "_fmt_count",
    "_history_error",
    "_iso_norm",
    "_project_layer_reading",
    "_rail_item_style",
    "_rail_label_style",
    "_relative_when",
    "_run_key",
    "_set_use_hint",
]
