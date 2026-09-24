








from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

from ...core.i18n import tr
from ...core.qt_compat import safe_disconnect
from ...core.window_focus import bring_qgis_window_to_front
from .env_setup_account import _drop_untagged_account_history


_BILLING_NOTICE_S = 15


class EnvSetupActivationMixin:


    def _refresh_activation_async(self, force: bool = False):








        import time

        from ...core.activation_manager import is_plugin_activated
        if not self.dock_widget:
            return



        self._start_sibling_sign_in()
        if not is_plugin_activated():
            if self._report_locked_activation_key():
                return


            if not self.dock_widget.is_activated() and self.dock_widget.dock_content_built:
                self.dock_widget._update_full_ui()
            return


        from ...core.server_dials import dial_in_range
        throttle_s = dial_in_range(
            "tuning.install.key_revalidate_throttle_s", 900, 60, 3600)
        if force or (time.time() - self._last_key_validation_unix) >= throttle_s:





            self._key_revalidate_pending = True




        self._reset_credits_backoff()
        self._refresh_auto_credits()

    def _report_locked_activation_key(self) -> bool:









        from ...core.auth_helper import activation_key_is_locked
        if not activation_key_is_locked():
            return False
        self.dock_widget.set_activation_message(
            tr("You are signed in on this computer, but QGIS cannot read your "
               "sign-in until you enter its master password."),
            is_error=False, kind="warning",
        )
        if not getattr(self, "_locked_key_recheck_armed", False):
            self._locked_key_recheck_armed = True
            from qgis.PyQt.QtCore import QTimer

            from ...core.server_dials import dial_in_range
            recheck_ms = dial_in_range(
                "tuning.install.locked_key_recheck_ms", 60000, 10000, 300000)
            QTimer.singleShot(recheck_ms, self._recheck_locked_activation_key)
        return True

    def _recheck_locked_activation_key(self) -> None:





        self._locked_key_recheck_armed = False
        if getattr(self, "dock_widget", None) is None:
            return
        self._refresh_activation_async()

    def _on_key_revalidate_ok(self, _usage: object) -> None:
        import time
        self._key_revalidate_pending = False
        self._last_key_validation_unix = time.time()

    def _on_key_revalidate_failed(self, message: str, code: str) -> None:
        self._key_revalidate_pending = False
        normalized = (code or "").strip().upper()






        if normalized == "INVALID_KEY":
            from ...core.activation_manager import clear_auth
            clear_auth()
            _drop_untagged_account_history()
            self._last_key_validation_unix = 0.0

            self._end_cloud_click_session()
            self._refresh_config_for_account_change()
            if self.dock_widget:
                self.dock_widget.set_activated_state(False)


            try:
                self.iface.messageBar().pushWarning(
                    "AI Segmentation",
                    tr("You have been signed out. Sign in again to keep "
                       "using the cloud features."),
                )
            except (RuntimeError, AttributeError):
                pass


            try:
                from ...core import telemetry_errors
                telemetry_errors.track_plugin_error(
                    stage="activate",
                    error_code=(code or "key_rejected").lower(),
                    message="stored key rejected on revalidation",
                )
            except Exception:
                pass  # nosec B110
            return
        if normalized == "SUBSCRIPTION_INACTIVE":
            self._notify_billing_problem()
            return
        if normalized in ("DEVICE_LIMIT_EXCEEDED", "DEVICE_LIMIT"):
            self._notify_device_limit()
            return
        self._notify_revalidate_failure(normalized, message)

    def _notify_device_limit(self) -> None:







        try:
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("Your plan is already running on its maximum number of "
                   "computers. Close AI Segmentation on one of them, then "
                   "run Detect again."),
            )
        except (RuntimeError, AttributeError):
            pass

    def _notify_billing_problem(self) -> None:





        if getattr(self, "_billing_warning_shown", False):
            return
        self._billing_warning_shown = True
        try:
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("There's a problem with your subscription. Open Settings "
                   "to update your payment method or review your plan."),
                level=Qgis.MessageLevel.Warning,


                duration=_BILLING_NOTICE_S,
            )
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def _notify_revalidate_failure(self, code: str, message: str) -> None:







        if (code or "").strip().upper() in self._CONNECTIVITY_CODES:
            self._notify_connection_issue(code, message)
            return
        import time

        from ...core.server_dials import dial_in_range
        now = time.monotonic()
        last = getattr(self, "_last_auth_notice_monotonic", 0.0)
        min_gap_s = dial_in_range(
            "tuning.network.conn_notice_min_gap_s", self._CONN_NOTICE_MIN_GAP_S, 10, 600)
        if now - last < min_gap_s:
            return
        self._last_auth_notice_monotonic = now
        QgsMessageLog.logMessage(
            f"Key revalidation failed ({code or 'unknown'})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        try:
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Could not check your AI Segmentation account. If this "
                   "lasts, sign out and sign in again."),
                level=Qgis.MessageLevel.Warning,
                duration=dial_in_range("tuning.network.connection_issue_notice_s", 8, 4, 10),
            )
        except (RuntimeError, AttributeError):  # nosec B110
            pass



    def _on_pairing_requested(self, code: str):









        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices

        from ...api.terralab_client import TerraLabClient
        from ...workers.pairing_poll_task import PairingPollTask

        client = TerraLabClient()
        self._start_pairing_poll(client, code)


        from ...core.device_id import get_device_hash




        url = (
            f"{client.base_url}/connect?code={code}&product=ai-segmentation"
            f"&device_id={get_device_hash()}"
            f"&ttl={int(PairingPollTask.CODE_TTL_S)}"
            "&utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
            "&utm_content=connect"
        )



        self._pairing_url = url
        if not QDesktopServices.openUrl(QUrl(url)):
            self._show_pairing_address(url)

    def _start_pairing_poll(self, client, code: str) -> None:







        worker = self._pairing_worker
        if worker is not None and worker.is_active():
            if worker.pairing_code == code:
                return


            for signal_name in ("pairing_succeeded", "pairing_failed",
                                "pairing_timeout", "pairing_stalled"):
                safe_disconnect(worker, signal_name)
            try:
                worker.cancel()
            except RuntimeError:
                pass  # nosec B110
        from qgis.core import QgsApplication

        from ...workers.pairing_poll_task import PairingPollTask
        self._pairing_worker = PairingPollTask(client, code)
        self._pairing_worker.pairing_succeeded.connect(self._on_pairing_succeeded)
        self._pairing_worker.pairing_failed.connect(self._on_pairing_failed)
        self._pairing_worker.pairing_timeout.connect(self._on_pairing_timeout)



        self._pairing_worker.pairing_stalled.connect(self._on_pairing_stalled)
        QgsApplication.taskManager().addTask(self._pairing_worker)
        import time as _time
        self._pairing_t0 = _time.monotonic()
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pairing_started()
        except Exception:
            pass  # nosec B110
        QgsMessageLog.logMessage(
            "Pairing started", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _show_pairing_address(self, url: str) -> None:













        if not self.dock_widget:
            return
        copied = False
        try:
            from qgis.PyQt.QtWidgets import QApplication

            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(url)
                copied = True
        except (RuntimeError, AttributeError, ImportError):
            copied = False
        try:


            from qgis.PyQt.QtCore import Qt

            label = self.dock_widget.activation_message_label
            label.setWordWrap(True)
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
        except (RuntimeError, AttributeError, ImportError):
            pass
        if copied:
            message = tr(
                "QGIS could not open a browser. The sign-in address is "
                "copied to your clipboard: paste it into a browser to "
                "finish, then come back here. It works once.")
        else:
            message = tr(
                "QGIS could not open a browser. Open this address to finish "
                "signing in, then come back here. It works once:\n{}").format(url)
        self.dock_widget.set_activation_message(message, is_error=True)

    def _clear_pairing_address(self) -> None:






        self._pairing_url = ""
        if not self.dock_widget:
            return
        try:
            label = self.dock_widget.activation_message_label
            label.clear()
            label.setVisible(False)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _pairing_elapsed_ms(self) -> int | None:

        try:
            import time as _time
            t0 = getattr(self, "_pairing_t0", None)
            return int((_time.monotonic() - t0) * 1000) if t0 else None
        except Exception:
            return None

    def _adopt_signed_in_key(self, key: str) -> None:


        from ...core.activation_manager import save_auth_token
        save_auth_token(key)

        _drop_untagged_account_history()

        self._refresh_config_for_account_change()


        self._last_key_validation_unix = 0.0
        if self.dock_widget:
            self.dock_widget.set_activated_state(True)



            self._reset_credits_backoff()
            self._refresh_auto_credits()

    def _start_sibling_sign_in(self) -> None:
        from ...core.server_dials import feature_enabled
        if not feature_enabled("sibling_sign_in"):
            return
        from ...api.terralab_client import TerraLabClient
        from ...core import sibling_sign_in
        try:
            from ...core.device_id import get_device_hash
            device = get_device_hash()
        except Exception:  # noqa: BLE001
            device = ""
        sibling_sign_in.start("ai-segmentation", TerraLabClient().base_url, device,
                              self._on_sibling_sign_in)

    def _on_sibling_sign_in(self, result: dict) -> None:






        from ...core.activation_manager import ACTIVATION_KEY_RE, is_plugin_activated
        if not result.get("ok") or not self.dock_widget or is_plugin_activated():
            return
        worker = self._pairing_worker
        if worker is not None and worker.is_active():
            return
        key = str(result.get("key") or "")
        if not ACTIVATION_KEY_RE.match(key):
            return
        self._adopt_signed_in_key(key)
        email, label = str(result.get("email") or ""), str(result.get("label") or "")
        message = (tr("Signed in as {} (from {}).").format(email, label) if email
                   else tr("Signed in (from {}).").format(label))
        self.dock_widget.set_activation_message(message, is_error=False, kind="success")
        try:
            self.iface.messageBar().pushMessage(
                "AI Segmentation", message, level=Qgis.MessageLevel.Success, duration=10)
        except (RuntimeError, AttributeError):
            pass
        try:
            from ...core.telemetry_session_events import track_plugin_activated
            track_plugin_activated(duration_ms=None)
        except Exception:  # nosec B110
            pass
        QgsMessageLog.logMessage(
            f"Signed in with the account of {label}", "AI Segmentation",
            level=Qgis.MessageLevel.Info)

    def _on_pairing_succeeded(self, key: str):
        self._adopt_signed_in_key(key)

        self._clear_pairing_address()


        try:
            bring_qgis_window_to_front(self.iface.mainWindow(), self.dock_widget)
        except Exception:  # nosec B110
            pass
        try:
            from ...core.telemetry_session_events import track_plugin_activated
            track_plugin_activated(duration_ms=self._pairing_elapsed_ms())
        except Exception:  # nosec B110
            pass
        QgsMessageLog.logMessage(
            "Pairing successful", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _on_pairing_failed(self, message: str, code: str):
        if self.dock_widget:
            self.dock_widget.show_pairing_idle()
            self.dock_widget.set_activation_message(message, is_error=True)
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pairing_failed(
                error_code=code or "unknown",
                duration_ms=self._pairing_elapsed_ms(),
            )
        except Exception:
            pass  # nosec B110
        QgsMessageLog.logMessage(
            f"Pairing failed ({code})", "AI Segmentation",
            level=Qgis.MessageLevel.Warning)

    def _on_pairing_stalled(self, reason: str = ""):










        if not self.dock_widget:
            return
        from ...workers.pairing_poll_task import PairingPollTask

        if reason == PairingPollTask.STALL_CODE_EXPIRED:


            self.dock_widget.set_activation_message(
                tr("This sign-in code has expired. Click Cancel, then Sign in "
                   "to get a new one."),
                is_error=True,
            )
            return
        message = tr("Still waiting for the sign-in page. If no browser "
                     "opened, or the page shows an error, click Cancel and "
                     "try again.")



        url = getattr(self, "_pairing_url", "")
        if url:
            message = "{}\n\n{}".format(
                message,
                tr("You can also open this address by hand:\n{}").format(url))
        self.dock_widget.set_activation_message(message, is_error=False, kind="info")

    def _on_pairing_timeout(self):
        if self.dock_widget:
            self.dock_widget.show_pairing_idle()
            self.dock_widget.set_activation_message(
                tr("Sign-in timed out. Click Sign in to try again."),
                is_error=True,
            )
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pairing_failed(
                error_code="timeout",
                duration_ms=self._pairing_elapsed_ms(),
            )
        except Exception:
            pass  # nosec B110
        QgsMessageLog.logMessage(
            "Pairing timed out", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _cancel_pairing_worker(self):

        if self._pairing_worker is not None and self._pairing_worker.is_active():
            try:
                self._pairing_worker.cancel()
            except RuntimeError:
                pass

    def _cancel_task(self, attr: str):







        task = getattr(self, attr, None)
        if task is None:
            return



        safe_disconnect(task, "succeeded")
        safe_disconnect(task, "failed")
        try:
            if task.is_active():
                task.cancel()
        except Exception:  # nosec B110
            pass
        setattr(self, attr, None)



    _CONNECTIVITY_CODES = frozenset({
        "DNS_ERROR", "CONNECTION_REFUSED", "TIMEOUT",
        "SSL_ERROR", "PROXY_ERROR", "NO_INTERNET",
    })

    _CONN_NOTICE_MIN_GAP_S = 60.0

    def _notify_connection_issue(self, code: str, message: str):




        import time

        from ...core.server_dials import dial_in_range

        if (code or "").strip().upper() not in self._CONNECTIVITY_CODES:
            return
        now = time.monotonic()
        min_gap_s = dial_in_range(
            "tuning.network.conn_notice_min_gap_s", self._CONN_NOTICE_MIN_GAP_S, 10, 600)
        if now - self._last_conn_notice_monotonic < min_gap_s:
            return
        self._last_conn_notice_monotonic = now
        try:
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                message or tr("Network error. Check your internet connection."),
                level=Qgis.MessageLevel.Warning,
                duration=dial_in_range("tuning.network.connection_issue_notice_s", 8, 4, 10),
            )
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def _on_cancel_pairing(self, code: str = ""):
        self._cancel_pairing_worker()

        self._clear_pairing_address()
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pairing_cancelled(duration_ms=self._pairing_elapsed_ms())
        except Exception:
            pass  # nosec B110
        if code:


            from qgis.core import QgsApplication, QgsTask

            from ...api.terralab_client import TerraLabClient
            client = TerraLabClient()
            self._pairing_cancel_task = QgsTask.fromFunction(
                tr("Cancelling sign-in"),
                lambda task, c=code: client.cancel_pairing(c),
            )
            QgsApplication.taskManager().addTask(self._pairing_cancel_task)
        QgsMessageLog.logMessage(
            "Pairing cancelled", "AI Segmentation", level=Qgis.MessageLevel.Info)
