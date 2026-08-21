"""Keeps the Automatic balance in step with the account while QGIS stays open.

The balance used to be read at five moments and never again: the first dock
open, sign-in, the switch into Automatic, the end of a run, and a re-run from
the library. Anything that changed the account from the outside was invisible
until one of those came round. The case that matters is the one the plugin
itself sets up: the panel says "subscribe", the user pays in a browser tab,
comes back, and finds the same figure and the same refused Detect.

Three hooks close it, all of them here:

- the QGIS window coming back to the front, which IS the return from the
  payment page;
- a slow repeat while Automatic is on screen and QGIS is the active window,
  for an account that changes from the other side (credits granted to a user);
- one re-read before Detect is refused for want of credits, so a stale figure
  can never be the reason a funded run is blocked.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out so
agents and humans work on one concern per file. Methods are plain mixin
members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import time

from qgis.PyQt.QtCore import QObject, Qt, pyqtSignal

from ...core.i18n import tr

# The timer only wakes the decision below; how often it actually reads the
# balance is the gap, not this. Kept short so an activation that arrives while
# the window was already active is still picked up quickly.
_WATCH_TICK_MS = 15_000
# Alt-tabbing between QGIS and a browser must not become one request per
# switch.
_ACTIVATE_GAP_S = 10.0
# A refusal is worth one read, not one per slider move.
_UNDERFUNDED_GAP_S = 30.0
# The generic client cadence when the server names none.
_IDLE_GAP_S = 180


class MainWindowActivationRelay(QObject):
    """Emits ``activated`` when QGIS comes back to the front.

    It listens to the application's own state signal. The first shape of this
    filtered every event the main window received, which is a Python call per
    mouse move to catch a moment that happens a few times an hour. A QObject
    of its own because a signal needs one and the plugin controller is not one.
    """

    activated = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._listening = False
        try:
            from qgis.PyQt.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.applicationStateChanged.connect(self._on_application_state)
                self._listening = True
        except (RuntimeError, AttributeError, TypeError):
            self._listening = False

    def _on_application_state(self, state) -> None:
        # Never raises: this runs inside Qt's own emission.
        try:
            if state == Qt.ApplicationState.ApplicationActive:
                self.activated.emit()
        except (RuntimeError, AttributeError, TypeError):
            pass

    def detach(self) -> None:
        """Stop listening. Safe to call more than once."""
        if not self._listening:
            return
        self._listening = False
        try:
            from qgis.PyQt.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.applicationStateChanged.disconnect(self._on_application_state)
        except (RuntimeError, AttributeError, TypeError):
            pass


class AutoCreditsWatchMixin:
    """Reads the Automatic balance again when it may have changed elsewhere."""

    def _arm_credits_watch(self) -> None:
        """Start watching. Idempotent, so it is safe on every dock start.

        Nothing here talks to the network: it creates a timer and an event
        filter, and every read is decided later, on its own conditions.
        """
        if self.dock_widget is None:
            return
        if self._credits_activation_relay is None:
            try:
                relay = MainWindowActivationRelay(self.dock_widget)
                relay.activated.connect(self._on_qgis_window_activated)
                self._credits_activation_relay = relay
            except (RuntimeError, AttributeError):
                self._credits_activation_relay = None
        if self._credits_watch_timer is None:
            from qgis.PyQt.QtCore import QTimer

            # Parented to the dock, never to the controller: the controller is
            # not a QObject, and an unparented timer outlives the plugin.
            timer = QTimer(self.dock_widget)
            timer.setInterval(_WATCH_TICK_MS)
            timer.timeout.connect(self._on_credits_watch_tick)
            timer.start()
            self._credits_watch_timer = timer

    def _disarm_credits_watch(self) -> None:
        """Stop the watch. Called from unload, before the tasks are cancelled:
        a tick landing during teardown would queue a fetch on a controller
        that is already coming apart."""
        timer = self._credits_watch_timer
        self._credits_watch_timer = None
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                pass  # the dock took its children with it
        relay = self._credits_activation_relay
        self._credits_activation_relay = None
        if relay is not None:
            try:
                relay.detach()
            except (RuntimeError, AttributeError):
                pass

    def _on_credits_watch_tick(self) -> None:
        """One tick: read the balance only if a change would show on screen
        and the gap since the last read has passed."""
        if not self._credits_watch_wanted():
            return
        if not self._qgis_window_is_active():
            # Nothing to poll for while the user is elsewhere: the window
            # coming back to the front reads the balance on its own, and that
            # is the very moment a payment made in a browser tab lands.
            return
        if (time.time() - self._credits_last_read_unix) < self._credits_refresh_gap_s():
            return
        self._read_credits_now()

    def _on_qgis_window_activated(self) -> None:
        """QGIS is back in front. The user may have just paid, or just been
        given credits, in the tab they came from."""
        if not self._credits_watch_wanted():
            return
        if (time.time() - self._credits_last_read_unix) < _ACTIVATE_GAP_S:
            return
        self._read_credits_now()

    def _on_account_usage_loaded(self, usage: dict) -> None:
        """The account window read the balance for its own card: take it.

        One fetch, both surfaces. Opening that window used to show a figure
        the footer ring never picked up, so the panel and the ring sat there
        contradicting each other.
        """
        self._credits_last_read_unix = time.time()
        try:
            self._apply_usage_payload(usage)
        except (RuntimeError, AttributeError):
            pass

    def _announce_plan_upgrade(self) -> None:
        """The account became Pro while this session was open. Say so, once.

        The user paid in a browser and came back to a panel that had been
        asking them to subscribe. Without a word here the only proof the
        payment landed is a number they were not watching.
        """
        if self._plan_upgrade_announced:
            return
        self._plan_upgrade_announced = True
        try:
            from qgis.core import Qgis

            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                # No figure here. This fires for a subscriber, and quota_limit
                # is a purchase-time snapshot, so a legacy plan can be 10,000.
                # The footer ring already shows the real balance.
                tr("Pro is active on this account. Your cloud detections are "
                   "ready."),
                level=Qgis.MessageLevel.Success,
            )
        except (RuntimeError, AttributeError):
            pass

    def _credits_watch_wanted(self) -> bool:
        """True while a fresher balance would change what is on screen.

        Never during a run: the balance is moving, the run reads its own, and
        the terminal funnel refreshes as soon as it stops.

        Semi-Auto counts too, but only when its clicks are answered on
        TerraLab's servers. That is the one shape of the mode that spends
        anything, and its Save is refused against this figure, so a stale one
        would turn a funded account away.
        """
        from ..ai_segmentation_dockwidget import Mode

        dock = self.dock_widget
        if dock is None or self._auto_worker is not None:
            return False
        try:
            if not dock.isVisible():
                return False
            if dock._mode == Mode.AUTOMATIC:
                return True
            return bool(getattr(self, "_manual_credit_ledger", None) is not None)
        except (RuntimeError, AttributeError):
            return False

    def _qgis_window_is_active(self) -> bool:
        """True while QGIS is the window the user is actually looking at."""
        try:
            return bool(self.iface.mainWindow().isActiveWindow())
        except (RuntimeError, AttributeError):
            return False

    @staticmethod
    def _credits_refresh_gap_s() -> float:
        """Seconds between two background reads of the balance.

        A dial on itself, so the cadence can be tightened the day a top-up has
        to show up faster, or loosened if it ever costs more than it is worth.
        Bounded so no served value can turn this into a polling loop.
        """
        try:
            from ...core.server_dials import dial_in_range

            return float(dial_in_range("credits_refresh_seconds", _IDLE_GAP_S, 30, 3600))
        except Exception:  # noqa: BLE001 -- a cadence must never break on config
            return float(_IDLE_GAP_S)

    def _read_credits_now(self) -> None:
        """One balance read, stamped so every throttle above holds.

        Stamped even when the fetch declines to start (not activated, one
        already in flight): the next attempt is one gap away either way.
        """
        self._credits_last_read_unix = time.time()
        try:
            self._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass
