"""Shared constants, styles, and small helpers for the Segment library.

The visual language mirrors AI Edit's prompt-templates dialog component for
component (sidebar buttons, cards, detail popup, buttons), with the accents
swapped to AI Segmentation's brand green.
"""
from __future__ import annotations

import calendar
import html
import time

from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.presets.segmentation_presets_client import absolute_demo_url

_BRAND_GREEN = "#43a047"       # AI Segmentation primary (Detect / Use)

# ---------------------------------------------------------------------------
# QSS (mirrors AI Edit prompt_templates/common.py + generation_detail/styles.py)
# ---------------------------------------------------------------------------

_SIDEBAR_ITEM = (
    "QPushButton { text-align: left; border: none; border-radius: 4px;"
    " padding: 10px 10px; font-size: 13px; color: palette(text);"
    " background: transparent; }"
    "QPushButton:hover { background: rgba(128,128,128,0.12); }"
)
_SIDEBAR_ITEM_ACTIVE = (
    "QPushButton { text-align: left; border: none; border-radius: 4px;"
    " padding: 10px 10px; font-size: 13px; font-weight: bold;"
    " color: palette(text); background: rgba(128,128,128,0.18); }"
)
_SECTION_HEADER = (
    "color: rgba(128,128,128,0.95); font-size: 10px; font-weight: 700;"
    " letter-spacing: 0.8px; background: transparent; border: none;"
    " padding: 6px 12px 2px 12px;"
)
_SEARCH_QSS = (
    "QLineEdit { border: 1px solid rgba(128,128,128,0.3);"
    " border-radius: 4px; padding: 6px 10px; font-size: 13px;"
    " color: palette(text); background: palette(base); }"
)
_CARD_NORMAL = (
    "QFrame#card { border: 1px solid rgba(128,128,128,0.30);"
    " border-radius: 6px; background: rgba(128,128,128,0.05); }"
)
_CARD_HOVER = (
    "QFrame#card { border: 1px solid rgba(67,160,71,0.75);"
    " border-radius: 6px; background: rgba(67,160,71,0.09); }"
)
# Right-aligned click affordance on every card footer: a faint chevron at rest
# that becomes a green "Use ->" on hover. Swapped by each card's enter/leave.
_USE_HINT_REST = (
    "QLabel { color: rgba(128,128,128,0.60); font-size: 13px; font-weight: 700;"
    " background: transparent; border: none; }"
)
_USE_HINT_HOVER = (
    "QLabel { color: #2e7d32; font-size: 12px; font-weight: 700;"
    " background: transparent; border: none; }"
)
_META_QSS = (
    "font-size: 11px; color: rgba(128,128,128,0.85);"
    " background: transparent; border: none;"
)
_EMPTY_MSG = "color: palette(text); padding: 24px;"

_BLUE_BTN_QSS = (
    "QPushButton { background-color: #1e88e5; color: #ffffff; border: none;"
    " border-radius: 6px; padding: 9px 18px; font-weight: bold;"
    " font-size: 13px; }"
    "QPushButton:hover { background-color: #1565c0; }"
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
    "QToolButton:checked { color: #f6b100; }"
    "QToolButton:hover { color: #f6b100; }"
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
    "QLabel { color: #43a047; background: rgba(67,160,71,0.13);"
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
    "QPushButton:hover { background: #2e7d32; }"
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
    "QToolButton:checked { color: #f6b100; }"
)

# ---------------------------------------------------------------------------
# Sidebar glyphs: tinted TEXT-presentation Unicode (never color emoji, which
# render inconsistently in Qt buttons - AI Edit's proven approach). Keys cover
# both the offline taxonomy and the server taxonomy.
# ---------------------------------------------------------------------------

_SIDEBAR_GLYPHS: dict[str, tuple[str, str]] = {
    "__recent__": ("◷", "#5ca0c0"),
    "__favorites__": ("☆", "#e57373"),
    "__top__": ("★", "#d4a548"),
    # offline taxonomy
    "buildings": ("⌂", "#b08858"),
    "vehicles_transport": ("➤", "#b07878"),
    "aircraft_vessels": ("✈", "#7aa0c4"),
    "energy_industrial": ("☀", "#d4a548"),
    "sport_recreation": ("⚑", "#c08fa0"),
    "land_water": ("◉", "#68a868"),
    # server taxonomy
    "transport": ("❖", "#9880b0"),
    "vehicles": ("➤", "#b07878"),
    "aircraft_maritime": ("✈", "#7aa0c4"),
    "energy": ("☀", "#d4a548"),
    "water": ("≈", "#3b8fb0"),
    "vegetation": ("✺", "#4d8c3f"),
    "agriculture": ("✿", "#c4a548"),
    "sports": ("⚑", "#c08fa0"),
    "land": ("◉", "#68a868"),
}
_GLYPH_DEFAULT = ("◈", "#8c6a4b")


def _sidebar_icon_html(key: str) -> str:
    glyph, color = _SIDEBAR_GLYPHS.get(key, _GLYPH_DEFAULT)
    return f'<span style="color:{color}; font-size:15px;">{glyph}</span>'


def _tab_label_html(label: str, count: int | None = None) -> str:
    """Sidebar label HTML - name with an optional muted count badge. The
    label is HTML-escaped ("Buildings & structures" needs &amp; in rich
    text, the sip mnemonic problem's rich-text cousin)."""
    label = html.escape(str(label or ""))
    count_html = ""
    if count is not None and count > 0:
        count_html = (
            f' <span style="color:rgba(128,128,128,0.8); font-size:11px;">'
            f'({count})</span>'
        )
    return (
        f'<span style="font-size:13px; color:palette(text);">{label}</span>'
        f'{count_html}'
    )


class _SidebarButton(QPushButton):
    """Sidebar tab entry: colored HTML glyph + label (+ optional count)."""

    def __init__(self, icon_html: str, label_html: str, parent=None):
        super().__init__(parent)
        self.setText("")
        self._label = QLabel(f"{icon_html}&nbsp;&nbsp;{label_html}")
        self._label.setTextFormat(QtC.RichText)
        self._label.setAttribute(QtC.WA_TransparentForMouseEvents)
        self._label.setStyleSheet(
            "background: transparent; border: none; padding: 0px;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.addWidget(self._label)

    def set_label_html(self, icon_html: str, label_html: str) -> None:
        self._label.setText(f"{icon_html}&nbsp;&nbsp;{label_html}")


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
    """Coarse 'today / yesterday / N days ago' from a UTC ISO timestamp."""
    try:
        parsed = time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        secs = time.time() - calendar.timegm(parsed)
    except (ValueError, TypeError):
        return ""
    days = int(secs // 86400)
    if days <= 0:
        return tr("today")
    if days == 1:
        return tr("yesterday")
    return tr("{n} days ago").format(n=days)


def _iso_norm(ts) -> str:
    """Normalize a server ISO timestamp (offset / fractional seconds tolerated)
    to the plain '%Y-%m-%dT%H:%M:%SZ' shape _relative_when parses."""
    ts = str(ts or "")
    return ts[:19] + "Z" if len(ts) >= 19 else ""


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
