"""Gallery card widgets: template card, local-recent card, history run card.

Card anatomy mirrors AI Edit's library cards: a 175px before/after preview,
a compact single-line footer (title + chevron that becomes "Use ->" on
hover), and the leaf-green lift on hover.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QPoint, Qt, pyqtSignal
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.presets.segmentation_presets import pick_label
from ...before_after_slider import BeforeAfterSlider
from ...template_demo_loader import TemplateDemoLoader
from .common import (
    _CARD_HOVER,
    _CARD_NORMAL,
    _META_QSS,
    _STAR_BTN_QSS,
    _build_use_hint,
    _demo_url,
    _iso_norm,
    _relative_when,
    _set_use_hint,
)

# Small ghost action buttons on a Recent card (lighter than the shared ghost
# QSS: these sit inside a card, so they must not read as heavy furniture).
_RECENT_ACTION_QSS = (
    "QPushButton { background: rgba(128,128,128,0.10); color: palette(text);"
    " border: none; border-radius: 5px; padding: 5px 10px; font-weight: 600;"
    " font-size: 11px; }"
    "QPushButton:hover { background: rgba(67,160,71,0.18); color: #2e7d32; }"
)

_TITLE_QSS = (
    "font-size: 13px; font-weight: 600; color: palette(text);"
    " background: transparent; border: none;"
)
# Shared preview height: every card in the grid aligns at the image edge.
_PREVIEW_H = 175


class _PresetCard(QFrame):
    """One template card: before/after slider + localized label + use hint.

    Demo images are requested lazily (only when the card scrolls into view)
    via ``request_demos``; the call is idempotent.
    """

    activated = pyqtSignal(dict)

    def __init__(self, preset: dict, parent=None):
        super().__init__(parent)
        self._preset = preset
        self._missing: set[str] = set()
        self._requested = False
        self.setObjectName("card")
        self.setStyleSheet(_CARD_NORMAL)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.slider = BeforeAfterSlider(self, auto_loop=False, show_badges=False)
        self.slider.setFixedHeight(_PREVIEW_H)
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.set_placeholder_text(tr("Loading..."))
        # The image emits its own click; the card handles footer/margin clicks.
        # Both route through the single `_fire` slot, and the dialog guards
        # against re-entry, so one user click can only open one detail popup.
        self.slider.clicked.connect(self._fire)
        lay.addWidget(self.slider)

        footer_wrap = QWidget(self)
        footer = QHBoxLayout(footer_wrap)
        footer.setContentsMargins(10, 8, 10, 10)
        footer.setSpacing(6)
        title = QLabel(pick_label(preset.get("label"), preset.get("prompt", "")))
        title.setStyleSheet(_TITLE_QSS)
        footer.addWidget(title)
        footer.addStretch()
        self._hint = _build_use_hint(self)
        footer.addWidget(self._hint)
        lay.addWidget(footer_wrap)

    def request_demos(self, loader: TemplateDemoLoader, base: str) -> None:
        if self._requested:
            return
        self._requested = True
        loader.request(self._preset["id"], "before", _demo_url(base, self._preset, "before"))
        loader.request(self._preset["id"], "after", _demo_url(base, self._preset, "after"))

    def set_image(self, which: str, pixmap) -> None:
        if which == "before":
            self.slider.set_before(pixmap)
        elif which == "after":
            self.slider.set_after(pixmap)

    def mark_missing(self, which: str) -> None:
        self._missing.add(which)
        if {"before", "after"} <= self._missing and not self.slider.has_images():
            self.slider.set_placeholder_text(tr("No preview"))

    def _fire(self) -> None:
        self.activated.emit(self._preset)

    def enterEvent(self, ev):  # noqa: N802 - Qt signature
        self.setStyleSheet(_CARD_HOVER)
        _set_use_hint(self._hint, True)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802 - Qt signature
        self.setStyleSheet(_CARD_NORMAL)
        _set_use_hint(self._hint, False)
        super().leaveEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802 - Qt signature
        # Root-cause guard for the historical double-popup: a click on the
        # slider is delivered there first, but the default QWidget handler lets
        # the release propagate up to this card too. The slider already emits
        # its own click for that case, so a release whose position lands inside
        # the slider's rect must NOT fire again here.
        pt = QtC.event_pos(ev)
        if not self.slider.geometry().contains(QPoint(int(pt.x()), int(pt.y()))):
            self._fire()
        super().mouseReleaseEvent(ev)


class _RecentCard(QFrame):
    """One local-recent card: zone thumbnail + object name + run stats, laid
    out like the template cards (same preview band + footer) so the Recent
    grid reads as one gallery. Shown as the signed-out / no-server-history
    face of the Recent tab.

    A single click restores the run, not just the prompt: the dialog reuses
    the object, zooms the canvas back to the detected zone and re-activates
    the exported layer (see SegmentLibraryDialog._on_recent_activated).
    Entries recorded before thumbnails existed (or whose PNG was cleaned up)
    show the shared no-preview placeholder.

    Two ghost buttons turn the card into a re-run loop: "Run again here"
    rebuilds the exact zone + prompt (only shown when the entry stored an
    extent), and "Same object, new zone" prefills the prompt for a fresh
    zone. Both are emitted upward; the dialog relays them to the dock."""

    activated = pyqtSignal(dict)
    rerun_requested = pyqtSignal(dict)       # "Run again here": same zone + prompt
    reuse_prompt_requested = pyqtSignal(dict)  # "Same object, new zone": prompt only

    def __init__(self, entry: dict, parent=None, *, view_only: bool = False):
        super().__init__(parent)
        self._entry = entry
        self._view_only = bool(view_only)
        self.setObjectName("card")
        self.setStyleSheet(_CARD_NORMAL)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Preview band: same height + rounded top as _RunCard / _PresetCard so
        # every card in the grid aligns at the image edge (no floating).
        self._thumb = QLabel()
        self._thumb.setFixedHeight(_PREVIEW_H)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet(
            "QLabel { background: rgba(128,128,128,0.12);"
            " border-top-left-radius: 6px; border-top-right-radius: 6px;"
            " color: rgba(128,128,128,0.7); font-size: 11px; }")
        self._thumb.setText(tr("No preview"))
        thumb_path = entry.get("_thumb") or ""
        if thumb_path:
            pixmap = QPixmap(thumb_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    max(self.width(), 200), _PREVIEW_H,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation)
                self._thumb.setText("")
                self._thumb.setPixmap(scaled)
        lay.addWidget(self._thumb)

        footer_wrap = QWidget(self)
        footer = QVBoxLayout(footer_wrap)
        footer.setContentsMargins(10, 8, 10, 10)
        footer.setSpacing(3)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel(entry.get("label") or entry.get("prompt", ""))
        title.setStyleSheet(_TITLE_QSS)
        title_row.addWidget(title, 1)
        self._hint = _build_use_hint(self)
        title_row.addWidget(self._hint)
        footer.addLayout(title_row)

        meta = entry.get("_meta") or ""
        if meta:
            meta_lbl = QLabel(meta)
            meta_lbl.setStyleSheet(_META_QSS)
            footer.addWidget(meta_lbl)

        # Re-run loop: two ghost actions. "Run again here" needs a stored zone
        # extent (old entries without one show only the prompt-reuse action).
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 4, 0, 0)
        actions.setSpacing(6)
        # While the library is view-only (a run is in flight) the re-run actions
        # are disabled + greyed: browsing is fine, starting another run is not.
        busy_tip = tr("Available when detection finishes")
        has_zone = bool(entry.get("extent") and entry.get("crs"))
        if has_zone:
            again_btn = QPushButton(tr("Run again here"))
            again_btn.setStyleSheet(_RECENT_ACTION_QSS)
            again_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            again_btn.setToolTip(
                busy_tip if self._view_only
                else tr("Reload this zone and object, ready to detect."))
            again_btn.setEnabled(not self._view_only)
            again_btn.clicked.connect(
                lambda: self.rerun_requested.emit(self._entry))
            actions.addWidget(again_btn)
        new_zone_btn = QPushButton(tr("Same object, new zone"))
        new_zone_btn.setStyleSheet(_RECENT_ACTION_QSS)
        new_zone_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_zone_btn.setToolTip(
            busy_tip if self._view_only
            else tr("Keep this object and draw a new zone on the map."))
        new_zone_btn.setEnabled(not self._view_only)
        new_zone_btn.clicked.connect(
            lambda: self.reuse_prompt_requested.emit(self._entry))
        actions.addWidget(new_zone_btn)
        actions.addStretch()
        footer.addLayout(actions)
        lay.addWidget(footer_wrap)

    def enterEvent(self, ev):  # noqa: N802 - Qt signature
        self.setStyleSheet(_CARD_HOVER)
        _set_use_hint(self._hint, True)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802 - Qt signature
        self.setStyleSheet(_CARD_NORMAL)
        _set_use_hint(self._hint, False)
        super().leaveEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802 - Qt signature
        self.activated.emit(self._entry)
        super().mouseReleaseEvent(ev)


class _RunCard(QFrame):
    """One history run card: overlay thumbnail + prompt + stats + star."""

    opened = pyqtSignal(dict)
    star_toggled = pyqtSignal(dict, bool)

    def __init__(self, run: dict, view: str, parent=None):
        super().__init__(parent)
        self._run = run
        self._view = view
        self.setObjectName("card")
        self.setStyleSheet(_CARD_NORMAL)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._thumb = QLabel()
        self._thumb.setFixedHeight(_PREVIEW_H)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet(
            "QLabel { background: rgba(128,128,128,0.12);"
            " border-top-left-radius: 6px; border-top-right-radius: 6px;"
            " color: rgba(128,128,128,0.7); font-size: 11px; }")
        self._thumb.setText(tr("Loading..."))
        lay.addWidget(self._thumb)

        body_wrap = QWidget(self)
        body = QVBoxLayout(body_wrap)
        body.setContentsMargins(10, 8, 10, 10)
        body.setSpacing(3)

        prompt = (run.get("prompt") or "").strip() or tr("Older detection")
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel(prompt)
        title.setStyleSheet(_TITLE_QSS)
        title_row.addWidget(title, 1)
        self.star_btn = QToolButton()
        self.star_btn.setCheckable(True)
        self.star_btn.setStyleSheet(_STAR_BTN_QSS)
        self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_favorite(bool(run.get("is_favorite")))
        # Legacy day-bucket pseudo-runs have no server row to star.
        if not run.get("run_id"):
            self.star_btn.setVisible(False)
        self.star_btn.clicked.connect(self._on_star_clicked)
        title_row.addWidget(self.star_btn)
        body.addLayout(title_row)

        meta_bits = [tr("{tiles} tiles · {objects} objects · {credits} credits").format(
            tiles=run.get("tiles") or 0,
            objects=run.get("objects") or 0,
            credits=run.get("credits") or 0,
        )]
        when = _relative_when(_iso_norm(
            run.get("started_at") or run.get("created_at")))
        if when:
            meta_bits.append(when)
        meta = QLabel("  ·  ".join(meta_bits))
        meta.setStyleSheet(_META_QSS)
        body.addWidget(meta)

        lay.addWidget(body_wrap)

    def _on_star_clicked(self, checked: bool) -> None:
        self.set_favorite(checked)  # glyph follows the optimistic flip at once
        self.star_toggled.emit(self._run, checked)

    def set_favorite(self, fav: bool) -> None:
        self.star_btn.blockSignals(True)
        self.star_btn.setChecked(fav)
        self.star_btn.setText("★" if fav else "☆")
        self.star_btn.blockSignals(False)

    def set_thumb(self, pixmap: QPixmap) -> None:
        if pixmap is None or pixmap.isNull():
            return
        w = max(self.width(), 200)
        scaled = pixmap.scaled(
            w, _PREVIEW_H,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation)
        self._thumb.setText("")
        self._thumb.setPixmap(scaled)

    def mark_thumb_missing(self) -> None:
        if self._thumb.pixmap() is None or self._thumb.pixmap().isNull():
            self._thumb.setText(tr("No preview"))

    def enterEvent(self, ev):  # noqa: N802 - Qt signature
        self.setStyleSheet(_CARD_HOVER)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802 - Qt signature
        self.setStyleSheet(_CARD_NORMAL)
        super().leaveEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802 - Qt signature
        self.opened.emit(self._run)
        super().mouseReleaseEvent(ev)
