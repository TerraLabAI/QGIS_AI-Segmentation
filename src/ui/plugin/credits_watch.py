

























from __future__ import annotations

import time

from qgis.PyQt.QtCore import QObject, Qt, pyqtSignal

from ...core.i18n import tr




_WATCH_TICK_MS = 15_000


_ACTIVATE_GAP_S = 10.0

_IDLE_GAP_S = 180




_USER_IDLE_AFTER_S = 600
_USER_IDLE_GAP_FACTOR = 4
_MAX_GAP_S = 3600


_HARD_MAX_GAP_S = 7200.0





_STABLE_READS_BEFORE_BACKOFF = 2
_STABLE_MAX_GAP_S = 900


class MainWindowActivationRelay(QObject):








    activated = pyqtSignal()

    def __init__(self, parent=None, canvas=None) -> None:
        super().__init__(parent)
        self._listening = False
        self._canvas = None



        self.last_input_unix = time.time()
        try:
            from qgis.PyQt.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.applicationStateChanged.connect(self._on_application_state)
                app.focusChanged.connect(self._on_user_input)
                self._listening = True
        except (RuntimeError, AttributeError, TypeError):
            self._listening = False
        if canvas is not None:
            try:
                canvas.extentsChanged.connect(self._on_user_input)
                self._canvas = canvas
            except (RuntimeError, AttributeError, TypeError):
                self._canvas = None

    def _on_user_input(self, *_args) -> None:
        self.last_input_unix = time.time()

    def seconds_since_user_input(self) -> float:
        return max(0.0, time.time() - self.last_input_unix)

    def _on_application_state(self, state) -> None:

        try:
            if state == Qt.ApplicationState.ApplicationActive:
                self.activated.emit()
        except (RuntimeError, AttributeError, TypeError):
            pass

    def detach(self) -> None:

        if not self._listening:
            return
        self._listening = False
        try:
            from qgis.PyQt.QtWidgets import QApplication

            app = QApplication.instance()
        except (RuntimeError, AttributeError, TypeError):
            app = None
        if app is not None:
            try:
                app.applicationStateChanged.disconnect(self._on_application_state)
            except (RuntimeError, AttributeError, TypeError):
                pass
            try:
                app.focusChanged.disconnect(self._on_user_input)
            except (RuntimeError, AttributeError, TypeError):
                pass
        canvas, self._canvas = self._canvas, None
        if canvas is not None:
            try:
                canvas.extentsChanged.disconnect(self._on_user_input)
            except (RuntimeError, AttributeError, TypeError):
                pass


class AutoCreditsWatchMixin:


    def _arm_credits_watch(self) -> None:





        if self.dock_widget is None:
            return
        if self._credits_activation_relay is None:
            try:
                canvas = None
                try:
                    canvas = self.iface.mapCanvas()
                except (RuntimeError, AttributeError):
                    canvas = None
                relay = MainWindowActivationRelay(self.dock_widget, canvas=canvas)
                relay.activated.connect(self._on_qgis_window_activated)
                self._credits_activation_relay = relay
            except (RuntimeError, AttributeError):
                self._credits_activation_relay = None
        if self._credits_watch_timer is None:
            from qgis.PyQt.QtCore import QTimer



            timer = QTimer(self.dock_widget)
            from ...core.server_dials import dial_in_range

            timer.setInterval(int(dial_in_range(
                "tuning.credits.watch_tick_ms", _WATCH_TICK_MS, 5_000, 60_000)))
            timer.timeout.connect(self._on_credits_watch_tick)
            timer.start()
            self._credits_watch_timer = timer

    def _disarm_credits_watch(self) -> None:



        timer = self._credits_watch_timer
        self._credits_watch_timer = None
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                pass
        relay = self._credits_activation_relay
        self._credits_activation_relay = None
        if relay is not None:
            try:
                relay.detach()
            except (RuntimeError, AttributeError):
                pass

    def _set_credits_watch_paused(self, paused: bool) -> None:







        timer = self._credits_watch_timer
        if timer is None:
            return
        try:
            if paused:
                timer.stop()
            elif not timer.isActive():
                timer.start()
        except RuntimeError:
            self._credits_watch_timer = None

    def _on_credits_watch_tick(self) -> None:


        if not self._credits_watch_wanted():
            return
        if not self._qgis_window_is_active():



            return
        if (time.time() - self._credits_last_read_unix) < self._credits_effective_gap_s():
            return
        self._read_credits_now()

    def _credits_effective_gap_s(self) -> float:






        gap = self._credits_stable_gap_s(self._credits_refresh_gap_s())
        relay = self._credits_activation_relay
        if relay is None:
            return gap
        try:
            idle_s = relay.seconds_since_user_input()
        except (RuntimeError, AttributeError):
            return gap
        try:
            from ...core.server_dials import dial_in_range

            idle_after = dial_in_range(
                "tuning.credits.user_idle_after_s", _USER_IDLE_AFTER_S, 60, 3600)
        except Exception:  # noqa: BLE001
            idle_after = _USER_IDLE_AFTER_S
        if idle_s < idle_after:
            return gap
        try:
            from ...core.server_dials import dial_in_range

            idle_factor = dial_in_range(
                "tuning.credits.user_idle_gap_factor", _USER_IDLE_GAP_FACTOR, 1, 10)
        except Exception:  # noqa: BLE001
            idle_factor = _USER_IDLE_GAP_FACTOR
        return float(min(gap * idle_factor, self._credits_max_gap_s()))

    @staticmethod
    def _credits_max_gap_s() -> float:




        try:
            from ...core.server_dials import dial_in_range

            return float(dial_in_range(
                "tuning.credits.max_gap_s", _MAX_GAP_S, 600.0, _HARD_MAX_GAP_S))
        except Exception:  # noqa: BLE001
            return float(_MAX_GAP_S)

    def _credits_stable_gap_s(self, gap: float) -> float:








        stable = int(getattr(self, "_credits_stable_reads", 0) or 0)
        try:
            from ...core.server_dials import dial_in_range




            threshold = int(dial_in_range(
                "tuning.credits.stable_reads_before_backoff",
                _STABLE_READS_BEFORE_BACKOFF, 1, 10))
        except Exception:  # noqa: BLE001
            threshold = int(_STABLE_READS_BEFORE_BACKOFF)
        if stable < threshold:
            return float(gap)
        try:
            from ...core.server_dials import dial_in_range

            stable_max = dial_in_range(
                "tuning.credits.stable_max_gap_s", _STABLE_MAX_GAP_S, 60, 3600)
        except Exception:  # noqa: BLE001
            stable_max = _STABLE_MAX_GAP_S
        stretched = float(gap) * (2 ** (stable - threshold + 1))
        return float(min(max(stretched, gap), stable_max))

    def _reset_credits_backoff(self) -> None:



        self._credits_stable_reads = 0

    def _note_credits_reading(self, fingerprint) -> None:

        if fingerprint is None:
            return
        if fingerprint == getattr(self, "_credits_last_fingerprint", None):
            self._credits_stable_reads = int(getattr(self, "_credits_stable_reads", 0) or 0) + 1
            return
        self._credits_last_fingerprint = fingerprint
        self._credits_stable_reads = 0

    @staticmethod
    def _credits_reading_fingerprint(usage, account=None):






        if not isinstance(usage, dict):
            return None
        parts = [
            str(usage.get(key)) for key in (
                "is_free_tier", "free_detections_remaining", "free_detections_total",
                "images_used", "images_limit", "reset_date", "period_end",
            )
        ]
        envelopes = None
        try:
            from ...core.quota_envelopes import (
                pick_segmentation_account_row,
                quota_envelopes_from_account_row,
            )
            row = pick_segmentation_account_row(account) if account else None
            envelopes = quota_envelopes_from_account_row(row) if row else None
        except Exception:  # noqa: BLE001
            envelopes = None
        if envelopes is not None:
            parts.extend(str(value) for value in envelopes)
        return "|".join(parts)

    def _on_qgis_window_activated(self) -> None:


        self._reset_credits_backoff()
        if not self._credits_watch_wanted():
            return
        try:
            from ...core.server_dials import dial_in_range

            gap = dial_in_range("tuning.credits.activate_gap_s", _ACTIVATE_GAP_S, 1.0, 60.0)
        except Exception:  # noqa: BLE001
            gap = _ACTIVATE_GAP_S
        if (time.time() - self._credits_last_read_unix) < gap:
            return
        self._read_credits_now()

    def _recheck_plan_after_free_wall(self) -> None:






        self._reset_credits_backoff()
        try:
            from ...core.server_dials import dial_in_range

            gap = dial_in_range("tuning.credits.activate_gap_s", _ACTIVATE_GAP_S, 1.0, 60.0)
        except Exception:  # noqa: BLE001
            gap = _ACTIVATE_GAP_S
        if (time.time() - self._credits_last_read_unix) < gap:
            return
        self._read_credits_now()

    def _on_account_usage_loaded(self, usage: dict) -> None:






        self._credits_last_read_unix = time.time()


        self._reset_credits_backoff()
        try:
            self._apply_usage_payload(usage)
        except (RuntimeError, AttributeError):
            pass

    def _announce_plan_upgrade(self) -> None:






        if self._plan_upgrade_announced:
            return
        self._plan_upgrade_announced = True
        try:
            from qgis.core import Qgis

            self.iface.messageBar().pushMessage(
                "AI Segmentation",



                tr("Pro is active on this account. Your cloud detections are "
                   "ready."),
                level=Qgis.MessageLevel.Success,
            )
        except (RuntimeError, AttributeError):
            pass

    def _credits_watch_wanted(self) -> bool:










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

        try:
            return bool(self.iface.mainWindow().isActiveWindow())
        except (RuntimeError, AttributeError):
            return False

    def _credits_refresh_gap_s(self) -> float:










        try:
            from ...core.server_dials import dial_in_range

            return float(dial_in_range(
                "credits_refresh_seconds", _IDLE_GAP_S, 30, self._credits_max_gap_s()))
        except Exception:  # noqa: BLE001
            return float(_IDLE_GAP_S)

    def _read_credits_now(self) -> None:





        self._credits_last_read_unix = time.time()
        try:
            self._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass
