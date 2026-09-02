"""The run history's own widgets: one card per past run, one header per day.

A run card is not a template card. A template is a word the user might pick; a
run is work they already paid for, so the card has to answer "which run was
this" (the imagery, the object, the day, the ground it covered) and then let
them act on it without opening anything.

Anatomy, top to bottom: the archived tile with its result painted over it, the
object count riding on the image, the prompt with its star, two muted lines of
facts, and the actions. The actions are HIDDEN, never greyed, when the window
cannot run them (no account, no plugin, a run already in flight): a dead
control on every card reads as a broken list.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QPoint, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ...before_after_slider import BeforeAfterSlider
from ...template_demo_loader import TemplateDemoLoader
from .cards import _RECENT_ACTION_QSS, _TITLE_QSS, _preview_height
from .common import (
    _CARD_HOVER,
    _CARD_NORMAL,
    _META_QSS,
    _OVERLAY_BADGE_QSS,
    _STAR_BTN_QSS,
)
from .run_summary import (
    run_detections_text,
    run_objects_text,
    run_status_text,
    run_time_text,
    run_title_text,
    run_zone_area_text,
)

# The two halves of a run card's comparison, in loader terms. "input" is the
# imagery exactly as it was sent; "preview" is the same tile with the detected
# masks painted on. The names are the artifact types the image route serves.
_RUN_BEFORE = "input"
_RUN_AFTER = "preview"

_ELIDE_RIGHT = QtC.resolve_qt_enum(Qt, "TextElideMode", "ElideRight")

# Header over one day's runs. A section label, not a card: it carries the date
# and nothing else, so it stays typographic (the design system's rule for
# headers) instead of growing a frame of its own.
_DAY_HEADER_QSS = (
    "QLabel { color: palette(text); font-size: 12px; font-weight: 700;"
    " background: transparent; border: none; }"
)


class _DayHeader(QWidget):
    """A full-width row naming one group of runs, with an optional action.

    The action is what the signed-out and offline faces of the list hang off:
    the header says which state the user is looking at, and carries the one
    thing they can do about it, so no card below has to.
    """

    def __init__(self, text: str, parent=None, *, note: str = "",
                 action: tuple | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 10, 2, 2)
        row.setSpacing(8)
        label = QLabel(text, self)
        label.setTextFormat(QtC.PlainText)
        label.setStyleSheet(_DAY_HEADER_QSS)
        row.addWidget(label)
        if note:
            note_label = QLabel(note, self)
            note_label.setTextFormat(QtC.PlainText)
            note_label.setStyleSheet(_META_QSS)
            row.addWidget(note_label)
        row.addStretch(1)
        if action is not None:
            caption, callback = action
            btn = QPushButton(caption, self)
            btn.setStyleSheet(_RECENT_ACTION_QSS)
            btn.setAutoDefault(False)
            btn.setCursor(QtC.PointingHandCursor)
            btn.clicked.connect(callback)
            row.addWidget(btn)


class _RunCard(QFrame):
    """One past run, with everything needed to recognise it and act on it."""

    opened = pyqtSignal(dict)
    star_toggled = pyqtSignal(dict, bool)
    restore_requested = pyqtSignal(dict)
    rerun_requested = pyqtSignal(dict)
    export_requested = pyqtSignal(dict)
    delete_requested = pyqtSignal(dict)

    def __init__(self, run: dict, view: str, parent=None, *,
                 can_star: bool = True, can_act: bool = False):
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

        self.slider = BeforeAfterSlider(
            self, auto_loop=False, show_badges=False, handle_grab_only=True)
        self.slider.setFixedHeight(_preview_height())
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.set_placeholder_text(tr("Loading..."))
        self.slider.clicked.connect(self._fire)
        lay.addWidget(self.slider)

        # Object count rides on the image, which frees the footer for the words
        # and puts the number where the eye already is.
        self._count_badge = QLabel(self.slider)
        self._count_badge.setStyleSheet(_OVERLAY_BADGE_QSS)
        self._count_badge.setAttribute(QtC.WA_TransparentForMouseEvents, True)
        self._count_badge.setText(run_objects_text(run))

        body_wrap = QWidget(self)
        body = QVBoxLayout(body_wrap)
        body.setContentsMargins(10, 8, 10, 10)
        body.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(4)
        self._title_text = run_title_text(run)
        self._title = QLabel(self._title_text, body_wrap)
        self._title.setTextFormat(QtC.PlainText)
        self._title.setStyleSheet(_TITLE_QSS)
        self._title.setToolTip(self._title_text)
        self._title.setMinimumWidth(24)
        self._title.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        title_row.addWidget(self._title, 1)
        self.star_btn = QToolButton(body_wrap)
        self.star_btn.setCheckable(True)
        self.star_btn.setStyleSheet(_STAR_BTN_QSS)
        self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_favorite(bool(run.get("is_favorite")))
        # No server row to flip: a legacy day-bucket pseudo-run, or a library
        # opened with no account behind it. The star would light up and be gone
        # on the next open, so the control is not offered at all.
        if not run.get("run_id") or not can_star:
            self.star_btn.setVisible(False)
        self.star_btn.clicked.connect(self._on_star_clicked)
        title_row.addWidget(self.star_btn)
        body.addLayout(title_row)

        self._facts = QLabel(self._facts_text(), body_wrap)
        self._facts.setTextFormat(QtC.PlainText)
        self._facts.setStyleSheet(_META_QSS)
        body.addWidget(self._facts)

        self._when = QLabel(self._when_text(), body_wrap)
        self._when.setTextFormat(QtC.PlainText)
        self._when.setStyleSheet(_META_QSS)
        body.addWidget(self._when)

        self._actions = self._build_actions(body_wrap)
        self._actions.setVisible(bool(can_act) and bool(run.get("run_id")))
        body.addWidget(self._actions)

        lay.addWidget(body_wrap)
        self._place_count_badge()

    # ---- text ------------------------------------------------------------

    def _facts_text(self) -> str:
        """Line one: how much ground, and what the run cost."""
        bits = [t for t in (run_zone_area_text(self._run),
                            run_detections_text(self._run)) if t]
        return "  ·  ".join(bits)

    def _when_text(self) -> str:
        """Line two: the time of day (the header carries the date), plus what
        this computer has already done with the run."""
        bits = [t for t in (run_time_text(self._run),
                            run_status_text(self._run)) if t]
        return "  ·  ".join(bits)

    def refresh_texts(self) -> None:
        """Re-read the run after something changed it (a status mark)."""
        try:
            self._facts.setText(self._facts_text())
            self._when.setText(self._when_text())
            self._count_badge.setText(run_objects_text(self._run))
            self._place_count_badge()
        except RuntimeError:
            pass  # nosec B110 - the card went while the fetch was in flight

    # ---- actions ---------------------------------------------------------

    def _build_actions(self, parent: QWidget) -> QWidget:
        host = QWidget(parent)
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(6)
        self.restore_btn = QPushButton(tr("Restore to map"), host)
        self.restore_btn.setStyleSheet(_RECENT_ACTION_QSS)
        self.restore_btn.setAutoDefault(False)
        self.restore_btn.setCursor(QtC.PointingHandCursor)
        self.restore_btn.setToolTip(
            tr("Reopens this run's review at the same place, with its "
               "imagery. Free, and it costs no cloud detections."))
        self.restore_btn.clicked.connect(
            lambda: self.restore_requested.emit(self._run))
        row.addWidget(self.restore_btn)
        self.rerun_btn = QPushButton(tr("Run again"), host)
        self.rerun_btn.setStyleSheet(_RECENT_ACTION_QSS)
        self.rerun_btn.setAutoDefault(False)
        self.rerun_btn.setCursor(QtC.PointingHandCursor)
        self.rerun_btn.setToolTip(
            tr("Points the map back at this run, ready to detect the same "
               "object again. Nothing is spent until you do."))
        self.rerun_btn.clicked.connect(
            lambda: self.rerun_requested.emit(self._run))
        row.addWidget(self.rerun_btn)
        row.addStretch(1)
        # Export and Delete are rarer than the two above and one of them cannot
        # be undone, so they sit one step in rather than beside them.
        self.more_btn = QPushButton(tr("More"), host)
        self.more_btn.setStyleSheet(_RECENT_ACTION_QSS)
        self.more_btn.setAutoDefault(False)
        self.more_btn.setCursor(QtC.PointingHandCursor)
        self.more_btn.clicked.connect(self._open_more_menu)
        row.addWidget(self.more_btn)
        return host

    def set_actions_enabled(self, usable: bool) -> None:
        """Hide the action row while the window is busy with another run.

        Hidden, not greyed: the design system's rule for a control that does
        not apply to the state the window is in.
        """
        try:
            self._actions.setVisible(bool(usable))
        except RuntimeError:
            pass  # nosec B110

    def _open_more_menu(self) -> None:
        menu = QMenu(self)
        export = menu.addAction(tr("Export..."))
        delete = menu.addAction(tr("Delete run"))
        chosen = menu.exec(self.more_btn.mapToGlobal(
            QPoint(0, self.more_btn.height())))
        if chosen is export:
            self.export_requested.emit(self._run)
        elif chosen is delete:
            self.delete_requested.emit(self._run)

    # ---- layout ----------------------------------------------------------

    def _place_count_badge(self) -> None:
        """Re-measure the badge and pin it to the image's bottom-left corner."""
        try:
            self._count_badge.adjustSize()
            self._count_badge.move(
                8, _preview_height() - self._count_badge.height() - 8)
        except (RuntimeError, AttributeError):
            pass  # nosec B110 - the card went before the layout settled

    def _apply_title_elide(self) -> None:
        """Cut a long prompt with an ellipsis instead of letting Qt clip it.

        The label is Ignored horizontally so it never widens the card, which
        means Qt hands it whatever is left and truncates mid-letter.
        """
        try:
            width = max(24, self._title.width())
            self._title.setText(
                self._title.fontMetrics().elidedText(
                    self._title_text, _ELIDE_RIGHT, width))
        except (RuntimeError, AttributeError, TypeError):
            pass  # nosec B110

    def on_font_scale_applied(self) -> None:
        """The card was measured, then the font pass grew its text.

        The badge floats over the image rather than sitting in a layout, so
        nothing moves it back: at any scale above 100% it was left overflowing
        its own corner.
        """
        self._place_count_badge()
        self._apply_title_elide()

    def resizeEvent(self, ev):  # noqa: N802 - Qt signature
        super().resizeEvent(ev)
        self._apply_title_elide()

    # ---- artifacts -------------------------------------------------------

    def request_artifacts(self, loader: TemplateDemoLoader, urls: dict,
                          variant: str | None = None) -> None:
        """Fetch the comparison halves. Idempotent, so a re-scroll is free.

        ``urls`` maps "input"/"preview" to (url, headers) or, when the same
        bytes have a second address to try, to (url, headers, fallback_url,
        fallback_headers). A half with no URL is marked missing at once rather
        than left spinning. ``variant`` names the size being asked for, so the
        card copy and the full one cached by the detail popup do not overwrite
        each other.

        The two halves are asked for separately and land separately, so the
        imagery paints as soon as it arrives instead of waiting on the overlay.

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
            entry = tuple(urls.get(which) or ())
            if not entry or not entry[0]:
                self.mark_missing(which)
                continue
            url, headers = entry[0], entry[1] if len(entry) > 1 else None
            fallback_url = entry[2] if len(entry) > 2 else None
            fallback_headers = entry[3] if len(entry) > 3 else None
            loader.request(key, which, url, headers=headers,
                           variant=variant, immutable=True,
                           fallback_url=fallback_url,
                           fallback_headers=fallback_headers)

    def artifact_key(self) -> str:
        """Cache/routing key for this run's images (the archived tile's id).

        The preview tile's id when there is one, else the tile whose imagery
        the server picked. A run that detected nothing has imagery and no
        overlay, and used to answer no key at all, which left its card blank.
        """
        return str(self._run.get("preview_request_id")
                   or self._run.get("input_request_id") or "")

    def adopt_run(self, run: dict) -> None:
        """Take the freshly synced payload for the run this card already shows.

        Called when a sync returns a page the grid is already painting: the
        cards are kept, so they must stop carrying the copy the disk cache
        handed them at open time.
        """
        self._run = run
        self.refresh_texts()

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
        self.star_btn.setToolTip(
            tr("Remove from favorites") if fav else tr("Add to favorites"))
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
        # The slider and the footer buttons emit their own click; without this
        # guard a release over one of them would also open the detail.
        pt = QtC.event_pos(ev)
        point = QPoint(int(pt.x()), int(pt.y()))
        if self.slider.geometry().contains(point):
            super().mouseReleaseEvent(ev)
            return
        child = self.childAt(point)
        while child is not None and child is not self:
            if isinstance(child, (QPushButton, QToolButton)):
                super().mouseReleaseEvent(ev)
                return
            child = child.parentWidget()
        self._fire()
        super().mouseReleaseEvent(ev)
