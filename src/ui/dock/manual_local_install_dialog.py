"""The window that asks before the offline AI lands on the machine.

Semi-Auto is the only mode with two engines, so it is the only place this
opens. Automatic answers every click off the machine and never downloads a
model.

It replaced an install card wedged into the panel. That card was on screen from
the first open, above the layer picker and above the Start button, on a page
where nobody had asked for a download: the user read a setup they did not want
before they read the mode they came for. It also sold the wait without the
price, because the disk figure it was meant to carry never rendered.

One window, two states. The offer states the two things a download costs, disk
and time, and the way out of it is the engine that costs neither. Once the
install runs the same window carries the bar, so the panel stays the page the
user came for from the first open to the last.

Modal through both states, on purpose. The install holds QGIS until it is
done, so the wait is felt where it is chosen, and the engine that costs no
wait stays the visible way out. Closing the window mid-install is a request to
stop it, asked back as a confirmation, never a way to park the download out of
sight: nothing about this install runs in the background.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from .styles import (
    _BTN_GHOST,
    _BTN_GREEN,
    _BTN_RED_OUTLINE,
    _PROGRESS_THIN_QSS,
    _btn_start_qss,
    _msg_label_qss,
)

# Same width the cloud notice uses. The two windows say the two halves of one
# decision, so they wrap their lines alike rather than each picking a size.
_DIALOG_WIDTH = 400

# What the download plus the build take on a normal connection, end to end.
# A figure, not a range: "5 to 20 minutes" reads as nobody having measured it.
# The shipped figure, and the fallback the reader below falls to.
_INSTALL_MINUTES = 10

# What `ask()` returns when the user picked the other engine. Qt keeps 0 and 1
# for Rejected and Accepted, so a window shut with Escape or the title bar X
# stays 0 and can be told apart from a button somebody actually pressed.
_CHOSE_CLOUD_CODE = 2


# The public name for the wait. Every screen that quotes it reads it here, so
# the window and the panel can never print two numbers at the same user.
def local_install_minutes() -> int:
    """How long the offer says the install takes, end to end, in minutes.

    Server-tunable, because this is a claim the user checks against a clock and
    a wrong one has to be correctable without a plugin release. The other half
    of the same claim, the disk figure, is already read that way. An absent
    key, an empty cache or a figure outside 1 to 120 leaves the shipped one in
    place, so an offline machine offers exactly what it shipped with.
    """
    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range(
            "install.local_minutes", _INSTALL_MINUTES, 1, 120))
    except Exception:  # noqa: BLE001 -- a bad config must never break the offer
        return _INSTALL_MINUTES


def local_install_disk_figures() -> tuple[float, float | None]:
    """What the install needs and what the volume has, both in GB.

    Free space comes back None when it cannot be read, and a reading under a
    gigabyte is treated the same way: on Windows that is a quota'd, redirected
    or virtualised path misreporting itself, not a full disk, and the install
    preflight already refuses to act on one. A figure this window cannot trust
    is a figure it does not show.
    """
    try:
        from ...core.venv_manager import resolved_min_free_gb_full

        need = float(resolved_min_free_gb_full())
    except Exception:  # noqa: BLE001 -- an unknown floor drops the figure
        need = 0.0
    free: float | None = None
    try:
        import os
        import shutil

        from ...core.cache_paths import PLUGIN_CACHE_DIR

        # The directory first, like every other reader of this volume. On a
        # machine that has never installed anything the path does not exist
        # yet, and the reading raised: the short-disk warning went missing on
        # exactly the machines it is written for.
        os.makedirs(PLUGIN_CACHE_DIR, exist_ok=True)
        measured = shutil.disk_usage(PLUGIN_CACHE_DIR).free / (1024 ** 3)
        free = measured if measured >= 0.001 else None
    except Exception:  # noqa: BLE001 -- an unreadable volume says nothing
        free = None
    return need, free


class ManualLocalInstallDialog(QDialog):
    """Ask for the offline install, then carry it. One window, two states."""

    # The install is the plugin's to stop, not this window's: the worker, the
    # lock and the half-written venv all live out there.
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("The offline AI"))
        self.setMinimumWidth(_DIALOG_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        self._title = QLabel("")
        self._title.setWordWrap(True)
        self._title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: palette(text);")
        layout.addWidget(self._title)
        layout.addSpacing(4)

        # -- the offer ------------------------------------------------------
        self._offer_lines: list[QWidget] = []
        need, free = local_install_disk_figures()
        for line in self._offer_copy(need):
            body = QLabel(line)
            body.setWordWrap(True)
            body.setTextFormat(Qt.TextFormat.RichText)
            body.setStyleSheet("font-size: 12px; color: palette(text);")
            layout.addWidget(body)
            self._offer_lines.append(body)

        # Only on a volume that cannot take it. A line reporting free space on
        # a machine with room to spare is a worry the user did not have.
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

        # Enabled even when the drive is short. The preflight is what refuses,
        # and it reclaims old environments before it does, so it clears
        # machines this reading would have turned away. Greying the one button
        # the window exists for is the wall of dead controls the design system
        # bans.
        self._install_btn = QPushButton(tr("Install it now"))
        self._install_btn.setMinimumHeight(38)
        self._install_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._install_btn.setStyleSheet(_btn_start_qss(_BTN_GREEN))
        self._install_btn.setDefault(True)
        self._install_btn.clicked.connect(self.accept)
        layout.addWidget(self._install_btn)
        self._offer_lines.append(self._install_btn)

        # A real button, not a Cancel. The user came here to segment, and the
        # other engine does that with nothing to download, so the way out has
        # to name it rather than dead-end them on the page they started on.
        self._cloud_btn = QPushButton(tr("Use Cloud AI instead"))
        self._cloud_btn.setMinimumHeight(34)
        self._cloud_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cloud_btn.setStyleSheet(_BTN_GHOST)
        # Its own result code, not reject(). Escape and the title bar X also
        # reject, and moving the engine on those would answer a question
        # nobody read.
        self._cloud_btn.clicked.connect(lambda: self.done(_CHOSE_CLOUD_CODE))
        layout.addWidget(self._cloud_btn)
        self._offer_lines.append(self._cloud_btn)

        # -- the install ----------------------------------------------------
        self._progress_lines: list[QWidget] = []
        self._progress_note = QLabel("")
        self._progress_note.setWordWrap(True)
        self._progress_note.setStyleSheet(
            "font-size: 12px; color: palette(text); padding-bottom: 4px;")
        layout.addWidget(self._progress_note)
        self._progress_lines.append(self._progress_note)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(_PROGRESS_THIN_QSS)
        layout.addWidget(self._progress_bar)
        self._progress_lines.append(self._progress_bar)

        # Says the wait out loud, and keeps the free way out of it in view.
        self._wait_note = QLabel(tr(
            "QGIS waits while this installs. To segment right away, stop "
            "the install and use Cloud AI."))
        self._wait_note.setWordWrap(True)
        self._wait_note.setStyleSheet(
            "font-size: 11px; color: rgba(128, 128, 128, 0.95);"
            " padding-top: 6px; padding-bottom: 8px;")
        layout.addWidget(self._wait_note)
        self._progress_lines.append(self._wait_note)

        self._cancel_btn = QPushButton(tr("Stop the install"))
        self._cancel_btn.setMinimumHeight(34)
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.setStyleSheet(_BTN_RED_OUTLINE)
        self._cancel_btn.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self._cancel_btn)
        self._progress_lines.append(self._cancel_btn)

        self._installing = False
        self._show_offer()

    # -- copy ---------------------------------------------------------------

    def _offer_copy(self, need_gb: float) -> list[str]:
        """The three facts, in the order a user weighing a download asks them.

        What it buys first, then the two prices. Bold on the figures alone, so
        the disk and the wait can be read without reading the sentences.
        """
        # Where the imagery is, never what does not happen to it. Denying a
        # risk is how you plant it, and on this half there is none to plant.
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

    # -- states -------------------------------------------------------------

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
        self.adjustSize()

    # -- what the dock drives -----------------------------------------------

    def ask(self) -> str:
        """Put the offer on screen, and say what came back.

        Three answers, not two: ``"install"`` they asked for the download,
        ``"cloud"`` they pressed the button for the other engine, ``"closed"``
        they shut the window without pressing either. The last one is not a
        refusal, and treating it as one moved the engine under a user who had
        answered nothing.
        """
        self._show_offer()
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        result = self.exec()
        if result == QDialog.DialogCode.Accepted:
            return "install"
        return "cloud" if result == _CHOSE_CLOUD_CODE else "closed"

    def begin_progress(self) -> None:
        """Reopen on the bar, and keep QGIS held until the install ends.

        Modality is set while the window is hidden, which is the only moment
        Qt honours a change to it: either exec() has just returned, or this is
        a fresh window put up over an install already running.
        """
        if not self.isVisible():
            self.setModal(True)
            self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._show_progress()
        self.set_progress(0, tr("Preparing the install..."))
        self.show()
        self.raise_()

    def is_installing(self) -> bool:
        """Whether this window is carrying an install right now."""
        return self._installing

    def mark_install_ended(self) -> None:
        """Let the window close without asking: the install is over.

        The dock calls this right before it closes the window, whichever way
        the install ended. Without it the close guards below would put a
        confirmation over a window the plugin itself is taking down.
        """
        self._installing = False

    # -- the ways out, all of them asked back ---------------------------------

    def _confirm_stop(self) -> bool:
        """Ask whether the install should really stop. True on a yes."""
        box = QMessageBox(self)
        box.setWindowTitle(tr("Stop the install?"))
        box.setText(tr(
            "The offline AI is not installed yet. Stop the install?"))
        stop_btn = box.addButton(
            tr("Stop the install"), QMessageBox.ButtonRole.DestructiveRole)
        keep_btn = box.addButton(
            tr("Keep installing"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep_btn)
        box.exec()
        return box.clickedButton() is stop_btn

    def _on_stop_clicked(self) -> None:
        """The Stop button: confirmed first, like every other way out."""
        if self._confirm_stop():
            self.cancel_requested.emit()

    def reject(self) -> None:
        """Escape during an install asks, it never slips the window away."""
        if self._installing:
            if self._confirm_stop():
                self.cancel_requested.emit()
            return
        super().reject()

    def closeEvent(self, event) -> None:
        """The title bar X during an install: same question, same two answers.

        The event is refused either way. On a yes the plugin stops the
        install and takes the window down itself; on a no the bar stays. A
        close that went through here silently would leave the download
        running with nothing on screen, which this window exists to prevent.
        """
        if self._installing:
            event.ignore()
            if self._confirm_stop():
                self.cancel_requested.emit()
            return
        super().closeEvent(event)

    def set_progress(self, percent: int, message: str) -> None:
        """Mirror one progress tick. Never raises: it is on a worker's path."""
        try:
            self._progress_bar.setValue(max(0, min(100, int(percent))))
            if message:
                self._progress_note.setText(message)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass  # nosec B110 -- a closed window must not break an install
