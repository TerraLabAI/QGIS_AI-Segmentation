












from __future__ import annotations

from qgis.PyQt.QtCore import QPoint, Qt, pyqtSignal
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
from ...dock.font_scale import scale_px_length, scale_qss_font_px
from ...dock.styles import _MENU_QSS, BTN_SMALL_PX, FONT_BODY, INK, RADIUS_CARD
from ...template_demo_loader import TemplateDemoLoader
from .cards import (
    _RECENT_ACTION_QSS,
    _TITLE_QSS,
    _card_min_width,
    _preview_height,
    _released_on_card,
)
from .common import (
    _CARD_HOVER,
    _CARD_NORMAL,
    _META_QSS,
    _OVERLAY_BADGE_QSS,
    _STAR_BTN_QSS,
    _icon_button,
    _set_star_glyph,
)
from .run_summary import (
    run_detections_text,
    run_objects_text,
    run_status_text,
    run_time_text,
    run_title_text,
    run_zone_area_text,
)




_RUN_BEFORE = "input"
_RUN_AFTER = "preview"

_ELIDE_RIGHT = QtC.resolve_qt_enum(Qt, "TextElideMode", "ElideRight")




_DAY_HEADER_QSS = scale_qss_font_px(
    f"QLabel {{ color: {INK}; font-size: {FONT_BODY}px; font-weight: 600;"
    " background: transparent; border: none; }"
)


def _set_icon_text(button: QPushButton, glyph: str) -> None:

    from qgis.PyQt.QtCore import QSize

    from ...icons import icon_for

    button.setIcon(icon_for(button, glyph, 14))
    button.setIconSize(QSize(14, 14))


class _DayHeader(QWidget):







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
        self.setMinimumWidth(_card_min_width())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.slider = BeforeAfterSlider(
            self, auto_loop=False, show_badges=False, handle_grab_only=True)
        self.slider.setFixedHeight(_preview_height())


        self.slider.set_card_corners(RADIUS_CARD, top_only=True)
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.set_placeholder_text(tr("Loading..."))
        self.slider.clicked.connect(self._fire)
        lay.addWidget(self.slider)



        self._count_badge = QLabel(self.slider)
        self._count_badge.setStyleSheet(_OVERLAY_BADGE_QSS)
        self._count_badge.setAttribute(QtC.WA_TransparentForMouseEvents, True)
        self._count_badge.setText(run_objects_text(run))

        body_wrap = QWidget(self)
        body = QVBoxLayout(body_wrap)
        body.setContentsMargins(12, 8, 8, 8)
        body.setSpacing(2)

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
        self.star_btn.setFixedSize(scale_px_length(BTN_SMALL_PX),
                                   scale_px_length(BTN_SMALL_PX))
        self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_favorite(bool(run.get("is_favorite")))



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



    def _facts_text(self) -> str:

        bits = [t for t in (run_zone_area_text(self._run),
                            run_detections_text(self._run)) if t]
        return "  ·  ".join(bits)

    def _when_text(self) -> str:


        bits = [t for t in (run_time_text(self._run),
                            run_status_text(self._run)) if t]
        return "  ·  ".join(bits)

    def refresh_texts(self) -> None:

        try:
            self._facts.setText(self._facts_text())
            self._when.setText(self._when_text())
            self._count_badge.setText(run_objects_text(self._run))
            self._place_count_badge()
        except RuntimeError:
            pass  # nosec B110



    def _build_actions(self, parent: QWidget) -> QWidget:
        host = QWidget(parent)
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(2)
        self.restore_btn = QPushButton(tr("Restore to map"), host)
        self.restore_btn.setStyleSheet(_RECENT_ACTION_QSS)
        self.restore_btn.setAutoDefault(False)
        self.restore_btn.setCursor(QtC.PointingHandCursor)
        self.restore_btn.setToolTip(
            tr("Reopens this run's review at the same place, with its "
               "imagery. Free, and it costs no cloud detections."))
        self.restore_btn.clicked.connect(
            lambda: self.restore_requested.emit(self._run))
        _set_icon_text(self.restore_btn, "layers")
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
        _set_icon_text(self.rerun_btn, "play")
        row.addWidget(self.rerun_btn)
        row.addStretch(1)




        self.more_btn = _icon_button(host, "kebab", tr("More"), tr("More"))
        self.more_btn.clicked.connect(self._open_more_menu)
        row.addWidget(self.more_btn)
        return host

    def set_actions_enabled(self, usable: bool) -> None:





        try:
            self._actions.setVisible(bool(usable))
        except RuntimeError:
            pass  # nosec B110

    def _open_more_menu(self) -> None:
        menu = QMenu(self)


        menu.setStyleSheet(_MENU_QSS)
        export = menu.addAction(tr("Export..."))
        delete = menu.addAction(tr("Delete run"))
        chosen = menu.exec(self.more_btn.mapToGlobal(
            QPoint(0, self.more_btn.height())))
        if chosen is export:
            self.export_requested.emit(self._run)
        elif chosen is delete:
            self.delete_requested.emit(self._run)



    def _place_count_badge(self) -> None:

        try:
            self._count_badge.adjustSize()
            self._count_badge.move(
                8, _preview_height() - self._count_badge.height() - 8)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _apply_title_elide(self) -> None:





        try:
            width = max(24, self._title.width())
            self._title.setText(
                self._title.fontMetrics().elidedText(
                    self._title_text, _ELIDE_RIGHT, width))
        except (RuntimeError, AttributeError, TypeError):
            pass  # nosec B110

    def on_font_scale_applied(self) -> None:






        self._place_count_badge()
        self._apply_title_elide()

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        self._apply_title_elide()



    def request_artifacts(self, loader: TemplateDemoLoader, urls: dict,
                          variant: str | None = None) -> None:















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






        return str(self._run.get("preview_request_id")
                   or self._run.get("input_request_id") or "")

    def adopt_run(self, run: dict) -> None:






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



    def _on_star_clicked(self, checked: bool) -> None:
        self.set_favorite(checked)
        self.star_toggled.emit(self._run, checked)

    def set_favorite(self, fav: bool) -> None:
        self.star_btn.blockSignals(True)
        self.star_btn.setChecked(fav)
        _set_star_glyph(self.star_btn, fav)
        tip = tr("Remove from favorites") if fav else tr("Add to favorites")
        self.star_btn.setToolTip(tip)
        self.star_btn.setAccessibleName(tip)
        self.star_btn.blockSignals(False)

    def _fire(self) -> None:



        QtC.safe_single_shot(0, self, self._do_fire)

    def _do_fire(self) -> None:
        try:
            self.opened.emit(self._run)
        except RuntimeError:
            pass  # nosec B110

    def enterEvent(self, ev):  # noqa: N802
        self.setStyleSheet(_CARD_HOVER)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802
        self.setStyleSheet(_CARD_NORMAL)
        super().leaveEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802


        pt = QtC.event_pos(ev)
        point = QPoint(int(pt.x()), int(pt.y()))
        if (not _released_on_card(self, ev)
                or self.slider.geometry().contains(point)):
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
