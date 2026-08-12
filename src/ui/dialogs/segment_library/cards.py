"""Gallery card widgets: template card, local-recent card, history run card.

Card anatomy mirrors AI Edit's library cards: a 175px before/after preview,
a compact single-line footer (title + chevron that becomes "Use ->" on
hover), and the leaf-green lift on hover.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QPoint, Qt, QTimer, pyqtSignal
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
    _OVERLAY_BADGE_QSS,
    _STAR_BTN_QSS,
    _build_use_hint,
    _demo_url,
    _fmt_count,
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
    star_toggled = pyqtSignal(dict, bool)

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

        self.slider = BeforeAfterSlider(
            self, auto_loop=False, show_badges=False, handle_grab_only=True)
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
        self.star_btn = QToolButton()
        self.star_btn.setCheckable(True)
        self.star_btn.setStyleSheet(_STAR_BTN_QSS)
        self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.star_btn.setToolTip(tr("Keep this object in Favorites"))
        self.set_favorite(False)
        self.star_btn.clicked.connect(self._on_star_clicked)
        footer.addWidget(self.star_btn)
        self._hint = _build_use_hint(self)
        footer.addWidget(self._hint)
        lay.addWidget(footer_wrap)

    def _on_star_clicked(self, checked: bool) -> None:
        self.set_favorite(checked)  # glyph follows the optimistic flip at once
        self.star_toggled.emit(self._preset, checked)

    def set_favorite(self, fav: bool) -> None:
        self.star_btn.blockSignals(True)
        self.star_btn.setChecked(fav)
        self.star_btn.setText("★" if fav else "☆")
        self.star_btn.blockSignals(False)

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
        if which not in ("before", "after"):
            return
        self._missing.add(which)
        # Settle the failed half now. Waiting for both to fail leaves a card
        # whose other half arrived stuck on "Loading..." for the life of the
        # dialog, with the slider holding an empty side.
        self.slider.mark_unavailable(which)
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

        # Preview band: same widget as the other two cards, so every card in the
        # grid aligns at the image edge and clips to the same rounded corners.
        # A QLabel cannot do that: QSS radius does not clip a pixmap, so a
        # QLabel thumbnail paints square corners inside a rounded card.
        # A local entry has one image and nothing to compare it to, so the
        # "before" half is declared absent and the thumbnail paints full bleed.
        self._thumb = BeforeAfterSlider(
            self, auto_loop=False, show_badges=False, handle_grab_only=True)
        self._thumb.setFixedHeight(_PREVIEW_H)
        self._thumb.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._thumb.set_placeholder_text(tr("No preview"))
        self._thumb.mark_unavailable("before")
        self._thumb.clicked.connect(self._fire)
        thumb_path = entry.get("_thumb") or ""
        if thumb_path:
            pixmap = QPixmap(thumb_path)
            if not pixmap.isNull():
                self._thumb.set_after(pixmap)
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

    def _fire(self) -> None:
        self.activated.emit(self._entry)

    def mouseReleaseEvent(self, ev):  # noqa: N802 - Qt signature
        # The preview emits its own click; without this guard a release over it
        # would fire twice (see _PresetCard.mouseReleaseEvent).
        pt = QtC.event_pos(ev)
        if not self._thumb.geometry().contains(QPoint(int(pt.x()), int(pt.y()))):
            self._fire()
        super().mouseReleaseEvent(ev)


# The two halves of a run card's comparison, in loader terms. "input" is the
# imagery exactly as it was sent; "preview" is the same tile with the detected
# masks painted on. The names are the artifact types the image route serves.
_RUN_BEFORE = "input"
_RUN_AFTER = "preview"


class _RunCard(QFrame):
    """One history run card: input/result comparison + prompt + stats + star.

    The preview band shows what the run was run ON as much as what it produced:
    a card that only shows masks is unreadable, since every run of the same
    object looks alike once the imagery underneath is hidden.
    """

    opened = pyqtSignal(dict)
    star_toggled = pyqtSignal(dict, bool)

    def __init__(self, run: dict, view: str, parent=None):
        super().__init__(parent)
        self._run = run
        self._view = view
        self._requested = False
        self._missing: set[str] = set()
        self.setObjectName("card")
        self.setStyleSheet(_CARD_NORMAL)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Same shape as AI Edit's gallery cards: no idle animation, no badges,
        # and the divider only grabs on its handle so the card stays clickable.
        # The labelled before/after lives in the detail popup the card opens.
        self.slider = BeforeAfterSlider(
            self, auto_loop=False, show_badges=False, handle_grab_only=True)
        self.slider.setFixedHeight(_PREVIEW_H)
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.set_placeholder_text(tr("Loading..."))
        self.slider.clicked.connect(self._fire)
        lay.addWidget(self.slider)

        # Object count rides on the image, which frees the footer for the prompt
        # and the date and puts the number where the eye already is.
        self._count_badge = QLabel(self.slider)
        self._count_badge.setStyleSheet(_OVERLAY_BADGE_QSS)
        self._count_badge.setAttribute(QtC.WA_TransparentForMouseEvents, True)
        objects = int(run.get("objects") or 0)
        self._count_badge.setText(
            tr("1 object") if objects == 1
            else tr("{n} objects").format(n=_fmt_count(objects)))
        self._count_badge.adjustSize()
        self._count_badge.move(8, _PREVIEW_H - self._count_badge.height() - 8)

        body_wrap = QWidget(self)
        body = QVBoxLayout(body_wrap)
        body.setContentsMargins(10, 8, 10, 10)
        body.setSpacing(3)

        # A run with no prompt was driven by boxes drawn on the map, so name it
        # for what it was rather than for its age.
        prompt = (run.get("prompt") or "").strip()
        if not prompt:
            prompt = (tr("Drawn examples") if run.get("has_exemplars")
                      else tr("Older detection"))
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel(prompt)
        title.setStyleSheet(_TITLE_QSS)
        title_row.addWidget(title, 1)
        self.star_btn = QToolButton()
        self.star_btn.setCheckable(True)
        self.star_btn.setStyleSheet(_STAR_BTN_QSS)
        self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.star_btn.setToolTip(tr("Keep this detection in Favorites"))
        self.set_favorite(bool(run.get("is_favorite")))
        # Legacy day-bucket pseudo-runs have no server row to star.
        if not run.get("run_id"):
            self.star_btn.setVisible(False)
        self.star_btn.clicked.connect(self._on_star_clicked)
        title_row.addWidget(self.star_btn)
        body.addLayout(title_row)

        # One credit per tile, so printing both numbers says the same thing
        # twice. The run detail carries the full billing breakdown.
        meta_bits = [tr("{tiles} cloud detections").format(
            tiles=_fmt_count(run.get("tiles")))]
        tiles = int(run.get("tiles") or 0)
        credits = int(run.get("credits") or 0)
        if credits != tiles:
            meta_bits.append(
                tr("{credits} charged").format(credits=_fmt_count(credits)))
        when = _relative_when(_iso_norm(
            run.get("started_at") or run.get("created_at")))
        if when:
            meta_bits.append(when)
        meta = QLabel("  ·  ".join(meta_bits))
        meta.setStyleSheet(_META_QSS)
        body.addWidget(meta)

        lay.addWidget(body_wrap)

    # ---- artifacts -------------------------------------------------------

    def request_artifacts(self, loader: TemplateDemoLoader, urls: dict,
                          variant: str | None = None) -> None:
        """Fetch the comparison halves. Idempotent, so a re-scroll is free.

        ``urls`` maps "input"/"preview" to (url, headers); a half with no URL is
        marked missing at once rather than left spinning. ``variant`` names the
        size being asked for, so the card copy and the full one cached by the
        detail popup do not overwrite each other.

        A finished run's archived tile is written once and never rewritten, so
        these are cached as immutable: no expiry, and no revalidation traffic.
        """
        if self._requested:
            return
        self._requested = True
        key = self.artifact_key()
        if not key:
            self.mark_missing(_RUN_BEFORE)
            self.mark_missing(_RUN_AFTER)
            return
        for which in (_RUN_BEFORE, _RUN_AFTER):
            entry = urls.get(which)
            if not entry or not entry[0]:
                self.mark_missing(which)
                continue
            loader.request(key, which, entry[0], headers=entry[1],
                           variant=variant, immutable=True)

    def artifact_key(self) -> str:
        """Cache/routing key for this run's images (the archived tile's id)."""
        return str(self._run.get("preview_request_id") or "")

    def adopt_run(self, run: dict) -> None:
        """Take the freshly synced payload for the run this card already shows.

        Called when a sync returns a page the grid is already painting: the
        cards are kept, so they must stop carrying the copy the disk cache
        handed them at open time.
        """
        self._run = run

    def set_image(self, which: str, pixmap: QPixmap) -> None:
        if which == _RUN_BEFORE:
            self.slider.set_before(pixmap)
        elif which == _RUN_AFTER:
            self.slider.set_after(pixmap)

    def mark_missing(self, which: str) -> None:
        if which not in (_RUN_BEFORE, _RUN_AFTER):
            return
        self._missing.add(which)
        self.slider.mark_unavailable(
            "before" if which == _RUN_BEFORE else "after")
        if {_RUN_BEFORE, _RUN_AFTER} <= self._missing:
            self.slider.set_placeholder_text(tr("No preview"))

    # ---- interaction -----------------------------------------------------

    def _on_star_clicked(self, checked: bool) -> None:
        self.set_favorite(checked)  # glyph follows the optimistic flip at once
        self.star_toggled.emit(self._run, checked)

    def set_favorite(self, fav: bool) -> None:
        self.star_btn.blockSignals(True)
        self.star_btn.setChecked(fav)
        self.star_btn.setText("★" if fav else "☆")
        self.star_btn.blockSignals(False)

    def _fire(self) -> None:
        # Deferred: opening the detail rebuilds the grid, and destroying the card
        # from inside its own signal handler aborts QGIS on Qt6.
        QTimer.singleShot(0, self._do_fire)

    def _do_fire(self) -> None:
        from qgis.PyQt import sip

        if sip.isdeleted(self) is True:
            return
        self.opened.emit(self._run)

    def enterEvent(self, ev):  # noqa: N802 - Qt signature
        self.setStyleSheet(_CARD_HOVER)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802 - Qt signature
        self.setStyleSheet(_CARD_NORMAL)
        super().leaveEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802 - Qt signature
        # The slider emits its own click; without this guard a release over it
        # would open the detail twice (see _PresetCard.mouseReleaseEvent).
        pt = QtC.event_pos(ev)
        if not self.slider.geometry().contains(QPoint(int(pt.x()), int(pt.y()))):
            self._fire()
        super().mouseReleaseEvent(ev)
