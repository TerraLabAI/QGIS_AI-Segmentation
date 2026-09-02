










from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from ...core.i18n import tr
from ...core.interaction_dials import confirm_reset_ms
from ...core.qt_compat import safe_single_shot
from ...core.review_corrections import RetryLinkState
from .font_scale import scale_qss_font_px
from .styles import _BTN_LINK_MUTED, _BTN_LINK_QUIET, FONT_HINT, INK_2, INK_3

_SUMMARY_TEXT_QSS = scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2};"
    " background: transparent; border: none;")
_SUMMARY_DOT_QSS = scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_3};"
    " background: transparent; border: none;")




CLEAR_CONFIRM_RESET_MS = 6000


def clear_all_rest_label(count: int) -> str:


    if int(count) > 1:
        return tr("Clear all {n}").format(n=int(count))
    return tr("Clear all")


class ClearAllConfirm:






    def __init__(self, button) -> None:
        self._button = button
        self._state = RetryLinkState()
        self._count = 0

    @property
    def armed(self) -> bool:

        return self._state.armed

    def clicked(self) -> bool:


        if self._state.activate():
            self._apply_label()
            return True
        self._apply_label()
        safe_single_shot(confirm_reset_ms(CLEAR_CONFIRM_RESET_MS), self._button, self.reset)
        return False

    def reset(self) -> None:


        if not self._state.armed:
            return
        self._state.reset()
        self._apply_label()

    def set_count(self, count: int) -> None:


        self._count = max(0, int(count))
        self._state.reset()
        self._apply_label()

    def _apply_label(self) -> None:
        button = self._button
        if button is None:
            return
        try:
            if self._state.armed:



                from .auto_review_build import _BTN_LINK_CONFIRM
                button.setText(tr("Undo every correction? Confirm"))
                button.setStyleSheet(_BTN_LINK_CONFIRM)
            else:
                button.setText(clear_all_rest_label(self._count))
                button.setStyleSheet(_BTN_LINK_MUTED)
        except (RuntimeError, AttributeError, ImportError):

            pass


def build_correction_summary_row(dock, lay) -> None:






    dock.auto_correct_summary_row = QWidget()
    row = QHBoxLayout(dock.auto_correct_summary_row)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    dock.auto_correct_summary_label = QLabel("")
    dock.auto_correct_summary_label.setStyleSheet(_SUMMARY_TEXT_QSS)
    dock.auto_correct_undo_btn = QPushButton(tr("Undo last"))
    dock.auto_correct_undo_btn.setStyleSheet(_BTN_LINK_QUIET)
    dock.auto_correct_undo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    dock.auto_correct_undo_btn.clicked.connect(
        dock.auto_correction_undo_requested.emit)
    dock.auto_correct_clear_btn = QPushButton(clear_all_rest_label(0))
    dock.auto_correct_clear_btn.setStyleSheet(_BTN_LINK_MUTED)
    dock.auto_correct_clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    dock.auto_correct_clear_btn.setToolTip(tr(
        "Undo every correction of this round at once. The count is in the "
        "label, so you can see what goes. It asks once first."))


    dock._correct_clear_confirm = ClearAllConfirm(dock.auto_correct_clear_btn)
    dock.auto_correct_clear_btn.clicked.connect(dock._on_correct_clear_clicked)
    row.addWidget(dock.auto_correct_summary_label)
    for widget in (dock.auto_correct_undo_btn, dock.auto_correct_clear_btn):
        dot = QLabel("\u00b7")
        dot.setStyleSheet(_SUMMARY_DOT_QSS)



        if widget is dock.auto_correct_undo_btn:
            dot.setContentsMargins(6, 0, 0, 0)
        row.addWidget(dot)
        row.addWidget(widget)
    row.addStretch(1)
    dock.auto_correct_summary_row.setVisible(False)
    lay.addWidget(dock.auto_correct_summary_row)
