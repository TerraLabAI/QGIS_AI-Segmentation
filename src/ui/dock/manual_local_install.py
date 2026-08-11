"""The door to the offline install, and the one window that carries it.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py); methods
here are plain mixin members and the window lives on the dock instance.

Every way into an offline install from Semi-Auto comes through here: picking
"My computer" on the engine cards, and pressing Start on a machine where a
previous attempt failed. One door, so the download can never start from a
control the user read as something else.

The panel used to be the door. It opened on an install card, above the layer
picker, before the user had chosen anything, and it stayed there until the
download was done. Nothing on that page had asked for it, and a mode whose
first screen is a setup reads as a mode you cannot use yet. The card is still
built, and `_update_full_ui_interactive` still shows it for the two states this
window cannot own: an install with no cloud engine to fall back on, and one
started somewhere else that would otherwise run with nothing on screen.

Refusing is a real answer, not an error. It puts the engine back on the cloud,
which needs no download, rather than leaving the user on a half of the switch
that cannot answer a click.
"""
from __future__ import annotations

from .manual_local_install_dialog import ManualLocalInstallDialog

# How long the window waits for the first progress tick before it hands the
# page back to the panel. It has to outlast the install's own retry ladder,
# which defers the start while the on-device pipe is still busy: a watchdog
# that fires first closes the window on an install that was about to begin.
# The shipped wait, and the fallback the reader below falls to.
_INSTALL_START_WATCHDOG_MS = 20000


# The wait in force. Read per call, never at import, so a configuration that
# lands later still reaches a window built before it.
def _install_start_watchdog_ms() -> int:
    """How long to wait for the first progress tick, in milliseconds.

    Server-tunable because the retry ladder this has to outlast is itself
    server-tunable, so the two must be able to move together. An absent key, an
    empty cache or a figure outside 5 to 120 seconds leaves the shipped wait in
    place.
    """
    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range(
            "install.start_watchdog_ms", _INSTALL_START_WATCHDOG_MS,
            5_000, 120_000))
    except Exception:  # noqa: BLE001 -- a bad config must never close a window early
        return _INSTALL_START_WATCHDOG_MS


class DockManualLocalInstallMixin:
    """Open, hold and close the offline-install window."""

    # -- questions the rest of the dock asks --------------------------------

    def _manual_install_window_owns_it(self) -> bool:
        """Whether a live window is already showing the install.

        What the panel reads before it fills itself with a second copy of the
        same progress bar.
        """
        dialog = getattr(self, "_manual_install_dialog", None)
        if dialog is None:
            return False
        try:
            return bool(dialog.is_installing())
        except RuntimeError:
            return False  # C++ side gone: nothing is showing anything

    # -- the door -----------------------------------------------------------

    def _open_manual_local_install(self) -> bool:
        """Ask for the offline install, and start it on a yes.

        True when a download is now running or already was. False when the
        user chose the cloud instead, and the caller must not treat that as a
        failure: it is the second engine doing its job.
        """
        # Already downloading. Put the window back up rather than asking for
        # an install that is minutes into itself.
        if self._manual_install_running():
            dialog = self._ensure_manual_install_dialog()
            if dialog is not None:
                try:
                    dialog.begin_progress()
                except RuntimeError:
                    self._manual_install_dialog = None
            # The route moves with the card. Without this the cards read My
            # computer while the stored answer was still the cloud, so Start
            # opened a billed session under a card that said otherwise. After
            # the window is up, so the redraw it triggers finds it there.
            self._set_manual_engine_cloud(False)
            return True

        dialog = self._ensure_manual_install_dialog()
        if dialog is None:
            # No window to ask with. Fall back to the panel card, which is the
            # behaviour every version before this one had.
            self._setup_section_wanted = True
            self._manual_install_wants_model = True
            self._update_full_ui()
            self.install_requested.emit()
            return True

        try:
            answer = dialog.ask()
        except RuntimeError:
            self._manual_install_dialog = None
            return False
        if answer == "closed":
            # Shut with Escape or the title bar X. That answers neither half
            # of the question, so the engine stays where it was and Start asks
            # again. Moving it here changed the mode under a user who had read
            # nothing and pressed nothing.
            self._close_manual_install_window()
            self._update_full_ui()
            return False
        if answer == "cloud":
            # `_set_manual_engine_cloud` stores the route, counts the choice
            # under the install gate and redraws the page, which is the whole
            # of what a refusal has to do.
            self._close_manual_install_window()
            self._set_manual_engine_cloud(True)
            return False

        # Stored BEFORE the request goes out: the plugin reads the dock to
        # decide whether this install carries the model, and Automatic's
        # lighter install is what it would pick from the wrong answer.
        self._set_manual_engine_cloud(False)
        try:
            dialog.begin_progress()
        except RuntimeError:
            self._manual_install_dialog = None
        self._setup_section_wanted = True
        # This one carries the on-device AI, which is what tells it apart from
        # the light install Automatic asks for.
        self._manual_install_wants_model = True
        self.install_requested.emit()
        self._update_full_ui()
        self._watch_manual_install_started()
        return True

    def _watch_manual_install_started(self) -> None:
        """Take the window down if the install never got off the ground.

        `_on_install_requested` has early returns that report their own error
        and start no worker: a Flatpak or Snap QGIS, an Intel Mac with no
        installable runtime, another install already in flight. None of them
        ever sends a progress tick, so the window would sit on an empty bar
        forever. The panel has the real message in every one of those cases,
        so hand the page back to it.

        The backstop, not the first answer: the two paths that report a
        refused install through `set_dependency_status` take the window down
        the moment they land. This catches the ones that report nothing.

        A timer rather than a check on the spot: the worker start is not
        guaranteed to have reported anything by the time the signal returns.
        """
        from qgis.PyQt.QtCore import QTimer

        def _check() -> None:
            try:
                if not self._manual_install_window_owns_it():
                    return
                if self._manual_install_running():
                    return
                self._finish_manual_install_window(False)
            except (RuntimeError, AttributeError):
                pass  # nosec B110 -- the dock can be gone by now

        QTimer.singleShot(_install_start_watchdog_ms(), _check)

    def _ensure_manual_install_dialog(self) -> ManualLocalInstallDialog | None:
        """The window, built once per dock and reused. None if Qt refused it."""
        dialog = getattr(self, "_manual_install_dialog", None)
        if dialog is not None:
            try:
                dialog.isVisible()  # cheap probe for a dead C++ side
                return dialog
            except RuntimeError:
                dialog = None
        try:
            dialog = ManualLocalInstallDialog(self)
            dialog.cancel_requested.connect(self._on_manual_install_stop)
            dialog.finished.connect(self._on_manual_install_window_closed)
        except (RuntimeError, TypeError):
            return None
        self._manual_install_dialog = dialog
        return dialog

    # -- what the install reports back --------------------------------------

    def _mirror_manual_install_progress(self, percent: int, message: str) -> None:
        """Show one progress tick in the window, when one is up.

        Called from `set_install_progress`, so it runs on every install the
        plugin starts, including the ones this window did not ask for. Those
        have no window and this is a no-op for them.
        """
        if not self._manual_install_window_owns_it():
            return
        try:
            self._manual_install_dialog.set_progress(percent, message)
        except (RuntimeError, AttributeError):
            self._manual_install_dialog = None

    def _finish_manual_install_window(self, ok: bool) -> None:
        """Close the window: the install ended, whichever way it ended.

        A failure leaves the panel to report it. The card carries the real
        message and the Retry button, and duplicating either into a window
        that is about to close would say the same thing twice.
        """
        if not self._manual_install_window_owns_it():
            return
        self._close_manual_install_window()
        if ok:
            return
        # Nothing to segment with on this half of the switch. The panel takes
        # the page back, with the error and the way to try again. Set here
        # rather than only where the failure is worded, because an install that
        # never started reports nothing at all and needs the same handover.
        self._manual_install_failed = True
        self._setup_section_wanted = True
        self._update_full_ui()

    def _close_manual_install_window(self) -> None:
        """Take the window down without treating that as an answer."""
        dialog = getattr(self, "_manual_install_dialog", None)
        self._manual_install_dialog = None
        if dialog is None:
            return
        try:
            dialog.blockSignals(True)
            # Ends the install as far as the window knows, so its close guard
            # does not put a stop confirmation over a close the plugin chose.
            dialog.mark_install_ended()
            dialog.close()
            dialog.deleteLater()
        except RuntimeError:
            pass  # nosec B110 -- already gone

    # -- the clicks ---------------------------------------------------------

    def _end_manual_install_as_stopped(self) -> None:
        """The one ending a stopped install gets, wherever Stop was pressed.

        The window's Stop and the panel's Cancel are the same act, and they
        used to part ways here: one moved the engine back to Cloud AI, the
        other left the switch on a half with nothing behind it, so the next
        Start ran into a download that had just been refused.

        The cards move back to Cloud AI, the Start button goes green again,
        and what downloaded stays on disk, so a second attempt resumes rather
        than starting over. With no second engine on offer there is nowhere to
        move to, and the panel keeps the install card and its Retry.
        """
        self._close_manual_install_window()
        self._manual_install_wants_model = False
        try:
            self.cancel_install_requested.emit()
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- teardown
        if self._manual_engine_offered():
            self._set_manual_engine_cloud(True)

    def _on_manual_install_stop(self) -> None:
        """The window's Stop button: cancel the download, keep the page usable."""
        self._end_manual_install_as_stopped()

    def _on_manual_install_window_closed(self, _result: int = 0) -> None:
        """A window gone mid-install stops the install. The backstop.

        The window's own guards ask before they let any close through, and
        every close the plugin chooses arrives here with signals blocked. So
        a finish that lands while the install still runs is a close that
        slipped past both, and the one thing it must not leave behind is a
        download running with nothing on screen: this install never runs in
        the background.
        """
        dialog = getattr(self, "_manual_install_dialog", None)
        if dialog is None:
            return
        try:
            if not dialog.is_installing():
                return
        except RuntimeError:
            self._manual_install_dialog = None
            return
        try:
            self._end_manual_install_as_stopped()
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- teardown
        try:
            self._update_full_ui()
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- teardown
