





from __future__ import annotations

from ..core.i18n import tr
from ..core.qt_compat import safe_disconnect
from ..core.server_dials import dial_in_range
from ..workers.generic_request_task import GenericRequestTask
from .account_settings_plan import (
    _as_int,
)
from .account_settings_session import (
    _ACCOUNT_OFFLINE_CODES,
)


def _format_purge_date(value) -> str:








    from ..core.quota_reset_date import format_quota_reset_date

    return format_quota_reset_date(value)






_DELETE_ACCOUNT_WATCHDOG_MS = 60_000


class AccountDeletionMixin:





    def _on_delete_account_clicked(self):






        from qgis.PyQt.QtWidgets import QDialog as _QDialog

        from .dialogs.confirm_dialog import warning_box

        if self._delete_running or self._removal_running:
            return

        email = (self._account_email or "").strip()
        if not email:
            warning_box(self, tr(
                "Your account address has not loaded yet. Close this window, "
                "open it again, then try."))
            return

        from .dialogs.delete_account_dialog import DeleteAccountDialog

        confirm_dialog = DeleteAccountDialog(email, self)
        try:
            answer = confirm_dialog.exec()
            typed = confirm_dialog.typed_email()
        finally:
            confirm_dialog.deleteLater()
        if answer != _QDialog.DialogCode.Accepted or not typed:
            return

        self._delete_running = True
        self._delete_generation += 1
        generation = self._delete_generation


        self._last_delete_error = None
        if self._delete_btn is not None:
            self._delete_btn.setEnabled(False)
            self._delete_btn.setText(tr("Deleting..."))
        self._set_delete_status(tr("Scheduling the deletion..."))

        from qgis.core import QgsApplication

        client, auth = self._client, self._auth

        def call_delete():



            result = client.delete_account(auth, typed)
            if isinstance(result, dict) and "error" in result:
                self._last_delete_error = dict(result)
            return result

        task = GenericRequestTask(
            tr("Deleting account..."), call_delete, hidden=True)
        task.succeeded.connect(self._on_delete_account_done)
        task.failed.connect(self._on_delete_account_failed)
        self._delete_task = task
        QgsApplication.taskManager().addTask(task)




        try:
            from ..core.qt_compat import safe_single_shot
            watchdog_ms = dial_in_range(
                "tuning.account.delete_watchdog_ms", _DELETE_ACCOUNT_WATCHDOG_MS,
                10_000, 600_000)
            safe_single_shot(
                watchdog_ms, self,
                lambda g=generation: self._on_delete_account_watchdog(g))
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _on_delete_account_watchdog(self, generation: int) -> None:



        if generation != getattr(self, "_delete_generation", 0):
            return
        if not self._delete_running:
            return
        try:



            self._cancel_delete_task()
            self._on_delete_account_failed(
                tr("The deletion did not get an answer. Check your connection, "
                   "then try again."), "")
        except RuntimeError:

            self._delete_running = False

    def _on_delete_account_done(self, data: dict):


        from .dialogs.confirm_dialog import success_box

        if not self._delete_running:
            return
        self._delete_running = False
        self._delete_task = None
        self._set_delete_status("")
        self._reset_delete_button()

        purge_date = _format_purge_date(
            (data or {}).get("purge_after") if isinstance(data, dict) else None)
        headline = tr(
            "Your account is scheduled for deletion. Every TerraLab plugin is "
            "signed out on this computer now.")
        if purge_date:
            message = tr(
                "Your data is erased for good on {}. Until then, sign in on "
                "terra-lab.ai to cancel it.").format(purge_date)
        else:
            message = tr(
                "After the grace period your data is erased for good. Until "
                "then, sign in on terra-lab.ai to cancel it.")



        self.account_deleted.emit()
        success_box(self, headline, message)
        self.accept()

    def _on_delete_account_failed(self, message: str, code: str = ""):

        from .dialogs.confirm_dialog import warning_box

        if not self._delete_running:
            return
        self._delete_running = False
        self._delete_task = None
        self._set_delete_status("")
        self._reset_delete_button()
        warning_box(self, self._delete_failure_text(message, code))

    def _delete_failure_text(self, message: str, code: str) -> str:






        payload = self._last_delete_error or {}
        self._last_delete_error = None
        key = (code or "").strip().upper()

        if key == "CONFIRM_MISMATCH":
            return tr(
                "That address does not match the one on your account. Check "
                "it and try again.")
        if key == "ALREADY_SCHEDULED":
            purge_date = _format_purge_date(payload.get("purge_after"))
            if purge_date:
                return tr(
                    "This account is already scheduled for deletion. Its data "
                    "is erased on {}. To cancel, sign in on terra-lab.ai."
                ).format(purge_date)
            return tr(
                "This account is already scheduled for deletion. To cancel, "
                "sign in on terra-lab.ai.")
        if key == "ACCOUNT_DELETION_SCHEDULED":
            return tr(
                "This account is already scheduled for deletion, so it can no "
                "longer be used from QGIS. To cancel, sign in on "
                "terra-lab.ai.")
        if key == "RATE_LIMITED":
            seconds = _as_int(payload.get("retry_after"))
            if seconds and seconds > 0:
                return tr(
                    "Too many attempts. Wait {} seconds, then try again."
                ).format(seconds)
            return tr("Too many attempts. Wait a moment, then try again.")
        if key == "NO_ACCOUNT":
            return tr(
                "This computer is not linked to a TerraLab account, so there "
                "is nothing to delete here.")
        if key in ("NO_AUTH", "INVALID_KEY"):
            return tr(
                "This computer is no longer signed in. Sign in again, then "
                "try.")
        if key == "SUBSCRIPTION_INACTIVE":
            return tr(
                "Your subscription is not active, so the service refused the "
                "request. Open your account on terra-lab.ai, then try again.")
        if key in _ACCOUNT_OFFLINE_CODES:
            return tr(
                "The request did not reach the service. Check your connection, "
                "then try again.")
        return message or tr(
            "The deletion could not be started. Try again in a few minutes.")

    def _set_delete_status(self, text: str):

        label = getattr(self, "_delete_status", None)
        if label is None:
            return
        try:
            label.setText(text)
            label.setVisible(bool(text))
        except RuntimeError:
            self._delete_status = None

    def _reset_delete_button(self):

        btn = getattr(self, "_delete_btn", None)
        if btn is None:
            return
        try:
            btn.setEnabled(bool(self._account_email))
            btn.setText(tr("Delete"))
        except RuntimeError:
            self._delete_btn = None

    def _cancel_delete_task(self):






        task = self._delete_task
        self._delete_task = None
        if task is None:
            return
        safe_disconnect(task, "succeeded")
        safe_disconnect(task, "failed")
        try:
            task.cancel()
        except Exception:  # nosec B110
            pass
