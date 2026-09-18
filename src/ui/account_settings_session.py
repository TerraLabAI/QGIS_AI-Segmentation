







from __future__ import annotations

from ..core.activation_manager import PRODUCT_ID
from ..core.i18n import tr
from ..core.qt_compat import safe_disconnect
from ..workers.generic_request_task import GenericRequestTask


def _load_account_and_usage(client, auth) -> dict:










    try:
        try:
            account, usage = client.get_account_and_usage(auth=auth)
        except Exception:  # nosec B110
            account = client.get_account(auth=auth)
            usage = {} if "error" in account else client.get_usage(auth=auth)
    finally:




        try:
            client.release_thread_nam()
        except Exception:  # noqa: BLE001
            pass  # nosec B110
    if not isinstance(account, dict):
        return {"error": "Invalid server response", "code": "SERVER_ERROR"}
    if "error" in account:
        return account
    if not isinstance(usage, dict) or "error" in usage:
        usage = {}
    return {"account": account, "usage": usage}





_ACCOUNT_OFFLINE_CODES = frozenset({
    "NO_INTERNET", "DNS_ERROR", "CONNECTION_REFUSED", "PROXY_ERROR",
    "TIMEOUT", "SSL_ERROR",
})


class AccountSessionMixin:





    def _fetch_account(self):
        self._paint_account_state({"kind": "loading"})

        from qgis.core import QgsApplication


        self._cancel_worker()
        client, auth = self._client, self._auth

        def load_account():


            result = _load_account_and_usage(client, auth)
            if isinstance(result, dict) and "error" in result:
                self._last_error_payload = dict(result)
            return result

        self._worker = GenericRequestTask(
            tr("Loading account info..."), load_account, hidden=True,
        )
        self._worker.succeeded.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        QgsApplication.taskManager().addTask(self._worker)

    def _cancel_worker(self):




        if self._worker is None:
            return


        safe_disconnect(self._worker, "succeeded")
        safe_disconnect(self._worker, "failed")
        try:
            self._worker.cancel()
        except Exception:  # nosec B110
            pass
        self._worker = None

    def _on_loaded(self, data: dict):


        account_data = data.get("account", data)
        usage_data = data.get("usage", {})



        if isinstance(usage_data, dict) and usage_data:
            self.usage_loaded.emit(usage_data)




        email = account_data.get("email")
        self._account_email = str(email).strip() if isinstance(email, str) else ""
        self._paint_account_state({
            "kind": "loaded",
            "account": account_data,
            "usage": usage_data if isinstance(usage_data, dict) else {},
        })

    def _on_failed(self, message: str, code: str = ""):






        from qgis.core import Qgis

        from ..core.logging_utils import log

        log(f"Account load failed ({code or 'unknown'}): {str(message)[:200]}",
            Qgis.MessageLevel.Warning)


        self._paint_account_state({
            "kind": "error",
            "code": str(code or ""),
            "payload": dict(getattr(self, "_last_error_payload", None) or {}),
        })

    @staticmethod
    def _find_subscription(data: dict) -> dict | None:


        subs = data.get("subscriptions") or []
        for pid in (f"{PRODUCT_ID}-pro", PRODUCT_ID):
            for s in subs:
                if isinstance(s, dict) and s.get("product_id") == pid:
                    return s
        return None

    def _on_sign_out(self, source: str = "account_card"):
        from .dialogs.confirm_dialog import PRIMARY, SECONDARY, ChoiceButton, ask_choice



        choice = ask_choice(
            self, tr("Sign out of AI Segmentation?"),
            tr("You can sign back in anytime from QGIS."),
            [ChoiceButton("cancel", tr("Cancel"), SECONDARY),
             ChoiceButton("sign_out", tr("Sign out"), PRIMARY)],
            default=None, escape="cancel")
        if choice != "sign_out":
            return



        try:
            from ..core import telemetry_session_events
            telemetry_session_events.track_account_signed_out(source)
        except Exception:
            pass  # nosec B110
        self.sign_out_requested.emit()
        self.accept()
