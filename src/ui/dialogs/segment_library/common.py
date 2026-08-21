"""Shared constants, styles, and small helpers for the Segment library.

The visual language mirrors AI Edit's prompt-templates dialog component for
component (navigation rail, cards, detail popup, buttons), with the accents
swapped to AI Segmentation's brand green.
"""
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
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    BRAND_GREEN,
    BTN_GREEN,
    BTN_GREEN_HOVER,
)

_BRAND_GREEN = BTN_GREEN       # AI Segmentation primary (Detect / Use)

# Favorite-star gold: a library-local accent for the checked/hover star. Kept
# here rather than in styles.py since nothing outside this dialog uses it.
_FAVORITE_STAR_GOLD = "#f6b100"

# Rail targets for the rows that are not a catalogue category. The two history
# rows map onto the server-side view names via _RAIL_HISTORY_VIEWS.
_RAIL_RECENT_TARGET = "__recent__"
_RAIL_FAVORITES_TARGET = "__favorites__"
_RAIL_POPULAR_TARGET = "__top__"
_RAIL_HISTORY_VIEWS = {
    _RAIL_RECENT_TARGET: "all",
    _RAIL_FAVORITES_TARGET: "favorites",
}

# ---------------------------------------------------------------------------
# QSS (mirrors AI Edit prompt_templates/common.py + generation_detail/styles.py)
# ---------------------------------------------------------------------------

_SEARCH_QSS = (
    "QLineEdit { border: 1px solid rgba(128,128,128,0.3);"
    " border-radius: 4px; padding: 6px 10px; font-size: 13px;"
    " color: palette(text); background: palette(base); }"
)
_CARD_NORMAL = (
    "QFrame#card { border: 1px solid rgba(128,128,128,0.30);"
    " border-radius: 6px; background: rgba(128,128,128,0.05); }"
)
# Hover is an active tint, so it belongs to the lime family. The CTA green is
# reserved for advance/commit and must not stand in for it.
_CARD_HOVER = (
    "QFrame#card { border: 1px solid rgba(139,172,39,0.75);"
    " border-radius: 6px; background: rgba(139,172,39,0.09); }"
)
# Count pill painted over the preview image, so the footer keeps the prompt and
# the date to itself. Dark scrim, because it sits on unpredictable imagery.
_OVERLAY_BADGE_QSS = (
    "QLabel { background: rgba(0,0,0,0.62); color: #ffffff; font-size: 10px;"
    " font-weight: 700; border: none; border-radius: 9px; padding: 2px 8px; }"
)
# Right-aligned click affordance on every card footer: a faint chevron at rest
# that becomes a green "Use ->" on hover. Swapped by each card's enter/leave.
_USE_HINT_REST = (
    "QLabel { color: rgba(128,128,128,0.60); font-size: 13px; font-weight: 700;"
    " background: transparent; border: none; }"
)
_USE_HINT_HOVER = (
    f"QLabel {{ color: {BTN_GREEN_HOVER}; font-size: 12px; font-weight: 700;"
    " background: transparent; border: none; }"
)
_META_QSS = (
    "font-size: 11px; color: rgba(128,128,128,0.85);"
    " background: transparent; border: none;"
)
# Empty states are hero-only: one glyph, one sentence, centered. The padding
# lives on the layout, not here, so the two parts stay a fixed distance apart.
_EMPTY_GLYPH = (
    "color: rgba(128,128,128,0.45); font-size: 34px;"
    " background: transparent; border: none;"
)
_EMPTY_MSG = (
    "color: rgba(128,128,128,0.95); font-size: 13px;"
    " background: transparent; border: none;"
)

_BLUE_BTN_QSS = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #ffffff; border: none;"
    " border-radius: 6px; padding: 9px 18px; font-weight: bold;"
    " font-size: 13px; }"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; }}"
    "QPushButton:disabled { background-color: rgba(128,128,128,0.25);"
    " color: rgba(128,128,128,0.8); }"
)
_GHOST_BTN_QSS = (
    "QPushButton { background: rgba(128,128,128,0.10); color: palette(text);"
    " border: none; border-radius: 6px; padding: 7px 14px; font-weight: 600;"
    " font-size: 12px; }"
    "QPushButton:hover { background: rgba(128,128,128,0.20); }"
)
_STAR_BTN_QSS = (
    "QToolButton { border: none; background: transparent; font-size: 16px;"
    " color: rgba(128,128,128,0.8); padding: 0 2px; }"
    f"QToolButton:checked {{ color: {_FAVORITE_STAR_GOLD}; }}"
    f"QToolButton:hover {{ color: {_FAVORITE_STAR_GOLD}; }}"
)

# ---- detail popup styles (AI Edit generation_detail/styles.py) ------------

_TITLE_STYLE = (
    "color: palette(text); font-size: 18px; font-weight: 800;"
    " letter-spacing: -0.2px; background: transparent; border: none;"
)
_SECTION_STYLE = (
    "color: rgba(128,128,128,0.95); font-size: 10px; font-weight: 700;"
    " letter-spacing: 1.2px; background: transparent; border: none;"
)
# Type/category tag above the title. Brand-green tint, hugs its content.
_BADGE_STYLE = (
    f"QLabel {{ color: {_BRAND_GREEN}; background: rgba(67,160,71,0.13);"
    " border: 1px solid rgba(67,160,71,0.40); border-radius: 9px;"
    " font-size: 10px; font-weight: 800; letter-spacing: 1.0px;"
    " padding: 2px 9px; }"
)
_SEPARATOR = "background: rgba(128,128,128,0.20); border: none;"
_PROMPT_STYLE = (
    "QLabel { color: palette(text); font-size: 12px;"
    " font-family: monospace;"
    " background: rgba(128,128,128,0.05); border: 1px solid rgba(128,128,128,0.15);"
    " border-radius: 4px; padding: 8px 10px; }"
)
_COPY_BTN = (
    "QPushButton { background: transparent; border: none;"
    " color: rgba(128,128,128,0.95); font-size: 11px; font-weight: 600;"
    " padding: 1px 6px; border-radius: 4px; }"
    "QPushButton:hover { background: rgba(128,128,128,0.14); color: palette(text); }"
)
_CHIP_STYLE = (
    "QFrame { background: rgba(128,128,128,0.06);"
    " border: 1px solid rgba(128,128,128,0.15); border-radius: 4px; }"
)
_CHIP_CAPTION = (
    "color: rgba(128,128,128,0.95); font-size: 9px; font-weight: 600;"
    " letter-spacing: 0.5px; background: transparent; border: none;"
)
_CHIP_VALUE = (
    "color: palette(text); font-size: 12px; font-weight: 600;"
    " background: transparent; border: none;"
)
_ACTION_BTN = (
    "QPushButton { background: transparent; border: 1px solid rgba(128,128,128,0.35);"
    " border-radius: 4px; padding: 7px 12px; font-size: 12px; color: palette(text); }"
    "QPushButton:hover { background: rgba(128,128,128,0.12);"
    " border-color: rgba(128,128,128,0.55); }"
    "QPushButton:disabled { color: rgba(128,128,128,0.5);"
    " border-color: rgba(128,128,128,0.15); }"
)
_PRIMARY_BTN = (
    f"QPushButton {{ background: {_BRAND_GREEN}; border: none; border-radius: 4px;"
    " padding: 8px 14px; font-size: 12px; font-weight: 600; color: #ffffff; }}"
    f"QPushButton:hover {{ background: {BTN_GREEN_HOVER}; }}"
    "QPushButton:disabled { background: rgba(128,128,128,0.25);"
    " color: rgba(128,128,128,0.6); }"
)
_FS_BTN = (
    "QToolButton { background: rgba(0,0,0,0.55); color: white; border: none;"
    " border-radius: 15px; font-size: 15px; }"
    "QToolButton:hover { background: rgba(0,0,0,0.8); }"
)
_DETAIL_STAR_BTN = (
    "QToolButton { background: transparent; border: 1px solid"
    " rgba(128,128,128,0.35); border-radius: 4px; font-size: 17px;"
    " color: rgba(128,128,128,0.8); }"
    "QToolButton:hover { background: rgba(128,128,128,0.15); }"
    f"QToolButton:checked {{ color: {_FAVORITE_STAR_GOLD}; }}"
)

# ---------------------------------------------------------------------------
# Navigation rail (left of the card grid). Mirrors AI Edit's library rail: the
# rows are deliberately plain (label + muted count, no glyph column and no
# per-entry hue) so the rail reads as a quiet index instead of a rainbow. The
# per-category tinted glyphs it replaces were the loudest thing in the dialog.
# ---------------------------------------------------------------------------

_RAIL_PANEL = (
    "QFrame#librail { border: none;"
    " border-right: 1px solid rgba(128,128,128,0.22); background: transparent; }"
)
# Group label above a cluster of rail rows. Sentence case, never uppercase:
# the design system bans uppercase across the plugin, so this is the one place
# the rail deviates from AI Edit's own sheet.
_RAIL_GROUP = (
    "QLabel { color: rgba(128,128,128,0.85); font-size: 10px; font-weight: 700;"
    " letter-spacing: 0.9px; background: transparent; border: none; }"
)
_RAIL_ITEM_LABEL = (
    "QLabel { color: palette(text); font-size: 13px;"
    " background: transparent; border: none; }"
)
# Muted count on the right of a rail row.
_RAIL_ITEM_COUNT = (
    "QLabel { color: rgba(128,128,128,0.85); font-size: 11px;"
    " background: transparent; border: none; }"
)


def _rail_item_style(active: bool) -> str:
    """QSS for one rail row. At rest: flat, transparent, hover tint. Active: a
    filled block, a bold label and a 3px left bar in the lime accent, so the
    selection reads as "you are here" rather than as a pressed button. The
    inactive row reserves the same 3px so the labels never shift sideways."""
    if active:
        qss = (
            "QPushButton#railitem { text-align: left; border: none;"
            f" border-left: 3px solid {BRAND_GREEN}; border-radius: 4px;"
            " padding: 8px 10px 8px 9px; font-size: 13px; font-weight: 700;"
            " color: palette(text); background: rgba(128,128,128,0.16); }"
        )
    else:
        qss = (
            "QPushButton#railitem { text-align: left; border: none;"
            " border-left: 3px solid transparent; border-radius: 4px;"
            " padding: 8px 10px 8px 9px; font-size: 13px;"
            " color: palette(text); background: transparent; }"
            "QPushButton#railitem:hover { background: rgba(128,128,128,0.10); }"
        )
    return _scale_qss_font_px(qss)


def _rail_label_style(active: bool) -> str:
    """QSS for a rail row's text. QSS does not cascade the button's color and
    weight into a child QLabel, so the active row restyles its label too."""
    if not active:
        return _RAIL_ITEM_LABEL
    return _scale_qss_font_px(
        "QLabel { color: palette(text); font-size: 13px; font-weight: 700;"
        " background: transparent; border: none; }"
    )


def _build_use_hint(parent) -> QLabel:
    hint = QLabel("›", parent)
    hint.setStyleSheet(_USE_HINT_REST)
    hint.setAttribute(QtC.WA_TransparentForMouseEvents)
    return hint


def _set_use_hint(hint: QLabel, hovered: bool) -> None:
    if hovered:
        hint.setText(f"{tr('Use')} →")
        hint.setStyleSheet(_USE_HINT_HOVER)
    else:
        hint.setText("›")
        hint.setStyleSheet(_USE_HINT_REST)


class _AspectBox(QWidget):
    """Keeps its single child at a fixed width:height ratio, centered. The
    before/after slider draws cover-fit, so matching the box ratio to the
    image ratio shows the whole image with no crop."""

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
        """Float ``widget`` over the bottom-right corner of the image rect."""
        self._overlay = widget
        widget.setParent(self)
        widget.raise_()
        self._relayout()

    def resizeEvent(self, event):  # noqa: N802 - Qt signature
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


# ---------------------------------------------------------------------------
# Small data helpers
# ---------------------------------------------------------------------------


def _demo_url(base: str, preset: dict, which: str, preview: bool = False) -> str:
    """Resolve a preset's demo image URL (server path or synthesized from id)."""
    rel = preset.get(f"demo_url_{which}") or (
        f"/api/ai-segmentation/template-demos/{preset.get('id', '')}/{which}")
    url = absolute_demo_url(base, rel)
    if preview and url:
        url += ("&" if "?" in url else "?") + "size=preview"
    return url


def _relative_when(ts: str) -> str:
    """Relative age of a UTC ISO timestamp, coarsening as it gets older.

    Past a week, counting days stops helping ("412 days ago" tells nobody
    anything), so the unit grows with the distance.
    """
    try:
        parsed = time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        stamp = calendar.timegm(parsed)
    except (ValueError, TypeError):
        return ""
    # Whole calendar days apart, on the user's own clock, not blocks of 24
    # hours: a run saved late last night reads "yesterday" this morning, where
    # counting elapsed hours still called it today.
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
    """Normalize a server ISO timestamp to the '%Y-%m-%dT%H:%M:%SZ' shape
    _relative_when parses.

    The offset is read, not discarded: chopping at 19 characters would turn a
    '+02:00' stamp into a UTC one and shift the age by the offset.
    """
    ts = str(ts or "").strip()
    if len(ts) < 19:
        return ""
    body, tail = ts[:19], ts[19:]
    # Drop fractional seconds, then look at what is left for an offset.
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
    """Group a count's thousands so 125549 reads as a magnitude, not a serial."""
    try:
        return QLocale().toString(int(value or 0))
    except (TypeError, ValueError):
        return "0"


def _project_layer_reading(source: str):
    """A layer already in the project reading this exact source, or None.

    Compared with normcase, because Windows hands the same file back in more
    than one casing and two spellings of one path would read as two files.
    """
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
    """Stable identity of a history run (real run_id or legacy group key)."""
    return str(run.get("run_id") or run.get("group_key") or "")


def _history_error(resp) -> str | None:
    """Error string of a history response, or None when it is usable."""
    if not isinstance(resp, dict):
        return "parse_error"
    if resp.get("error"):
        return str(resp.get("code") or "SERVER_ERROR")
    return None


# The panel follows the text size the user set in QGIS (see font_scale). A
# constant applied at build time is caught by the pass over the finished panel,
# but the same constant re-applied on a state change is not, and the widget
# would snap back to the base size mid-session. Growing them here, once, covers
# both. A no-op on the default text size, and outside QGIS.
for _qss_name in (
    "_RAIL_GROUP",
    "_RAIL_ITEM_LABEL",
    "_RAIL_ITEM_COUNT",
    "_SEARCH_QSS",
    "_OVERLAY_BADGE_QSS",
    "_USE_HINT_REST",
    "_USE_HINT_HOVER",
    "_META_QSS",
    "_EMPTY_GLYPH",
    "_EMPTY_MSG",
    "_BLUE_BTN_QSS",
    "_GHOST_BTN_QSS",
    "_STAR_BTN_QSS",
    "_TITLE_STYLE",
    "_SECTION_STYLE",
    "_BADGE_STYLE",
    "_PROMPT_STYLE",
    "_COPY_BTN",
    "_CHIP_CAPTION",
    "_CHIP_VALUE",
    "_ACTION_BTN",
    "_PRIMARY_BTN",
    "_FS_BTN",
    "_DETAIL_STAR_BTN",
):
    globals()[_qss_name] = _scale_qss_font_px(globals()[_qss_name])
del _qss_name
