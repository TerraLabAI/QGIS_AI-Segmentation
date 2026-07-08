






















from __future__ import annotations

from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    _BTN_GHOST,
    _BTN_GREEN,
    _BTN_RED_OUTLINE,
    BTN_PRIMARY_WIDE_PX,
    BTN_PX,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    HUE_LOCAL,
    INK,
    INK_2,
    _btn_start_qss,
    _msg_label_qss,
    category_progress_qss,
    category_tile,
)



_DIALOG_WIDTH = 400




_INSTALL_MINUTES = 10





_STALL_SECONDS = 180






_HOLD_SECONDS = 4 * 60




_CHOSE_CLOUD_CODE = 2




_TITLE_QSS = scale_qss_font_px(
    f"font-size: {FONT_BASE + 2}px; font-weight: 600; color: {INK};")
_BODY_QSS = scale_qss_font_px(f"font-size: {FONT_BODY}px; color: {INK};")
_PROGRESS_NOTE_QSS = scale_qss_font_px(
    f"font-size: {FONT_BODY}px; color: {INK}; padding-bottom: 4px;")
_WAIT_NOTE_QSS = scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2};"
    " padding-top: 6px; padding-bottom: 8px;")

_DIALOG_MARGIN = 16




def local_install_minutes() -> int:








    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range(
            "install.local_minutes", _INSTALL_MINUTES, 1, 120))
    except Exception:  # noqa: BLE001
        return _INSTALL_MINUTES


def local_install_disk_figures() -> tuple[float, float | None]:








    try:
        from ...core.venv_manager import resolved_min_free_gb_full

        need = float(resolved_min_free_gb_full())
    except Exception:  # noqa: BLE001
        need = 0.0
    free: float | None = None
    try:
        import os
        import shutil

        from ...core.cache_paths import PLUGIN_CACHE_DIR





        os.makedirs(PLUGIN_CACHE_DIR, exist_ok=True)
        measured = shutil.disk_usage(PLUGIN_CACHE_DIR).free / (1024 ** 3)
        free = measured if measured >= 0.001 else None
    except Exception:  # noqa: BLE001
        free = None
    return need, free


class ManualLocalInstallDialog(QDialog):




    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("The offline AI"))
        self.setMinimumWidth(_DIALOG_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*(_DIALOG_MARGIN,) * 4)
        layout.setSpacing(8)

        self._title = QLabel("")
        self._title.setWordWrap(True)
        self._title.setStyleSheet(_TITLE_QSS)


        _head = QHBoxLayout()
        _head.setContentsMargins(0, 0, 0, 0)
        _head.setSpacing(10)
        _head.addWidget(category_tile("package", HUE_LOCAL), 0,
                        Qt.AlignmentFlag.AlignVCenter)
        _head.addWidget(self._title, 1, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(_head)
        layout.addSpacing(4)


        self._offer_lines: list[QWidget] = []
        need, free = local_install_disk_figures()
        for line in self._offer_copy(need):
            body = QLabel(line)
            body.setWordWrap(True)
            body.setTextFormat(Qt.TextFormat.RichText)
            body.setStyleSheet(_BODY_QSS)
            layout.addWidget(body)
            self._offer_lines.append(body)



        self._short_disk = QLabel("")
        self._short_disk.setWordWrap(True)
        self._short_disk.setStyleSheet(_msg_label_qss("warning"))
        self._short_disk.setVisible(False)
        if need > 0 and free is not None and free < need:
            self._short_disk.setText(tr(
                "This drive has {free} GB free, under the {need} GB the "
                "install needs. Free some space, or use Cloud AI.").format(
                    free=f"{free:.1f}", need=f"{need:g}"))
            self._short_disk.setVisible(True)
        layout.addWidget(self._short_disk)
        self._offer_lines.append(self._short_disk)

        layout.addSpacing(6)






        self._install_btn = QPushButton(tr("Install it now"))



        self._install_btn.setFixedHeight(scale_px_length(BTN_PRIMARY_WIDE_PX))
        self._install_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._install_btn.setStyleSheet(_btn_start_qss(_BTN_GREEN))
        self._install_btn.setDefault(True)
        self._install_btn.clicked.connect(self.accept)
        layout.addWidget(self._install_btn)
        self._offer_lines.append(self._install_btn)




        self._cloud_btn = QPushButton(tr("Use Cloud AI instead"))
        self._cloud_btn.setFixedHeight(scale_px_length(BTN_PX))


        self._cloud_btn.setAutoDefault(False)
        self._cloud_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cloud_btn.setStyleSheet(_BTN_GHOST)



        self._cloud_btn.clicked.connect(lambda: self.done(_CHOSE_CLOUD_CODE))
        layout.addWidget(self._cloud_btn)
        self._offer_lines.append(self._cloud_btn)


        self._progress_lines: list[QWidget] = []
        self._progress_note = QLabel("")
        self._progress_note.setWordWrap(True)
        self._progress_note.setStyleSheet(_PROGRESS_NOTE_QSS)
        layout.addWidget(self._progress_note)
        self._progress_lines.append(self._progress_note)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(category_progress_qss(HUE_LOCAL))
        layout.addWidget(self._progress_bar)
        self._progress_lines.append(self._progress_bar)



        self._stall_note = QLabel(tr(
            "The install has not reported anything for a while. QGIS is yours "
            "again: leave this running, or stop it and use Cloud AI."))
        self._stall_note.setWordWrap(True)
        self._stall_note.setStyleSheet(_msg_label_qss("warning"))
        self._stall_note.setVisible(False)
        layout.addWidget(self._stall_note)
        self._progress_lines.append(self._stall_note)


        self._wait_note = QLabel(tr(
            "QGIS waits while this installs. To segment right away, stop "
            "the install and use Cloud AI."))
        self._wait_note.setWordWrap(True)
        self._wait_note.setStyleSheet(_WAIT_NOTE_QSS)
        layout.addWidget(self._wait_note)
        self._progress_lines.append(self._wait_note)

        self._cancel_btn = QPushButton(tr("Stop the install"))
        self._cancel_btn.setFixedHeight(scale_px_length(BTN_PX))


        self._cancel_btn.setAutoDefault(False)
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.setStyleSheet(_BTN_RED_OUTLINE)
        self._cancel_btn.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self._cancel_btn)
        self._progress_lines.append(self._cancel_btn)

        self._installing = False


        from ...core.server_dials import dial_in_range

        self._stall_timer = QTimer(self)
        self._stall_timer.setSingleShot(True)
        self._stall_timer.setInterval(dial_in_range(
            "tuning.install.stall_seconds", _STALL_SECONDS, 30, 900) * 1000)
        self._stall_timer.timeout.connect(self._on_install_stalled)


        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(dial_in_range(
            "tuning.install.hold_seconds", _HOLD_SECONDS, 60, 1800) * 1000)
        self._hold_timer.timeout.connect(self._release_application)
        self._show_offer()



    def _offer_copy(self, need_gb: float) -> list[str]:







        lines = [
            tr("The offline AI answers your clicks on this computer. Your "
               "imagery stays here, and every click is free."),
        ]
        if need_gb > 0:
            lines.append(tr("It needs <b>{gb} GB</b> of free disk space.")
                         .format(gb=f"{need_gb:g}"))
        lines.append(
            tr("Downloading and setting it up takes <b>about {n} minutes</b>, "
               "once.").format(n=local_install_minutes()))
        return lines



    def _show_offer(self) -> None:
        self._installing = False
        self._title.setText(tr("Install the offline AI"))
        for widget in self._offer_lines:
            widget.setVisible(bool(widget.text()) if widget is self._short_disk
                              else True)
        for widget in self._progress_lines:
            widget.setVisible(False)

    def _show_progress(self) -> None:
        self._installing = True
        self._title.setText(tr("Installing the offline AI"))
        for widget in self._offer_lines:
            widget.setVisible(False)
        for widget in self._progress_lines:
            widget.setVisible(True)



        self._stall_note.setVisible(False)
        self.adjustSize()



    def ask(self) -> str:








        self._show_offer()
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        result = self.exec()
        if result == QDialog.DialogCode.Accepted:
            return "install"
        return "cloud" if result == _CHOSE_CLOUD_CODE else "closed"

    def begin_progress(self) -> None:






        if not self.isVisible():
            self.setModal(True)
            self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._show_progress()
        self.set_progress(0, tr("Preparing the install..."))
        self.show()
        self.raise_()
        self._stall_timer.start()
        self._hold_timer.start()

    def is_installing(self) -> bool:

        return self._installing

    def mark_install_ended(self) -> None:






        self._installing = False
        self._stall_timer.stop()
        self._hold_timer.stop()



    def _confirm_stop(self) -> bool:



        from ..dialogs.confirm_dialog import (
            DANGER,
            SECONDARY,
            ChoiceButton,
            ask_choice,
        )

        choice = ask_choice(
            self, tr("Stop the install?"),
            tr("The offline AI is not installed yet. Stop the install?"),
            [ChoiceButton("keep", tr("Keep installing"), SECONDARY),
             ChoiceButton("stop", tr("Stop the install"), DANGER)],
            default=None, escape="keep", tone="warning")
        return choice == "stop"

    def _on_stop_clicked(self) -> None:

        if self._confirm_stop():
            self.cancel_requested.emit()

    def reject(self) -> None:

        if self._installing:
            if self._confirm_stop():
                self.cancel_requested.emit()
            return
        super().reject()

    def closeEvent(self, event) -> None:







        if self._installing:
            event.ignore()
            if self._confirm_stop():
                self.cancel_requested.emit()
            return
        super().closeEvent(event)

    def set_progress(self, percent: int, message: str) -> None:

        try:
            self._progress_bar.setValue(max(0, min(100, int(percent))))
            if message:
                self._progress_note.setText(message)




            self._stall_note.setVisible(False)
            self._stall_timer.start()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass  # nosec B110

    def _on_install_stalled(self) -> None:








        if not self._installing:
            return
        self._stall_note.setVisible(True)
        self._release_application()

    def _release_application(self) -> None:





        if not self._installing:
            return
        try:
            if self.windowModality() != Qt.WindowModality.NonModal:






                self.hide()
                self.setModal(False)
                self.setWindowModality(Qt.WindowModality.NonModal)
                self.show()
                self.raise_()
            self.adjustSize()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
