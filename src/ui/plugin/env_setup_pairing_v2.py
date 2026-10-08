













from __future__ import annotations

from qgis.core import Qgis, QgsApplication, QgsMessageLog

from ...core.i18n import tr
from ...core.qt_compat import safe_disconnect



_FALLBACK_AFTER_S = 15.0
_CLEARED = ""


class _PairingV2SignIn:


    def __init__(self, attempt_id: str) -> None:


        self.attempt_id = attempt_id
        self.secret = _CLEARED
        self.code = ""
        self.connect_url = ""
        self.legacy = False
        self.ended = False
        self.listener = None
        self.start_task = None
        self.fallback_timer = None
        self.grant_claim = None
        self.typed_claim = None

        self.typed_held = False


class EnvSetupPairingV2Mixin:





    def _pairing_v2_begin(self, attempt_id: str) -> None:

        from ...api.terralab_client import TerraLabClient
        from ...workers.pairing_listener import PairingListener
        from ...workers.pairing_poll_task import PairingPollTask
        from ...workers.pairing_v2 import PairingStartTask, make_secret, secret_hash

        self._pairing_v2_end()
        self._announce_pairing_started()
        state = _PairingV2SignIn(attempt_id)
        self._pairing_v2 = state
        state.secret = make_secret()
        listener = PairingListener(
            lambda grant, reply, s=state: self._pairing_v2_on_grant(s, grant, reply))
        port = listener.start()
        state.listener = listener if port else None
        task = PairingStartTask(TerraLabClient(), secret_hash(state.secret), port or None,
                                int(PairingPollTask.CODE_TTL_S))
        task.started.connect(
            lambda code, url, expires_in, s=state: self._pairing_v2_on_started(
                s, code, url, expires_in))
        task.use_legacy.connect(lambda reason, s=state: self._pairing_v2_on_legacy(s, reason))
        state.start_task = task
        QgsApplication.taskManager().addTask(task)

    def _pairing_v2_current(self, state: _PairingV2SignIn) -> bool:
        return state is self._pairing_v2 and not state.ended

    def _pairing_v2_reopen(self, attempt_id: str) -> bool:


        state = self._pairing_v2
        if state is None or state.ended or state.attempt_id != attempt_id:
            return False
        if state.legacy:
            return False
        if state.connect_url:
            self._pairing_v2_open_browser(state.connect_url)
        return True

    def _pairing_v2_open_browser(self, connect_url: str) -> None:
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices

        from ...core.device_id import get_device_hash


        joiner = "&" if "?" in connect_url else "?"
        url = f"{connect_url}{joiner}device_id={get_device_hash()}"
        self._pairing_url = url
        if not QDesktopServices.openUrl(QUrl(url)):
            self._show_pairing_address(url)

    def _pairing_v2_on_started(self, state: _PairingV2SignIn, code: str,
                               connect_url: str, expires_in: int) -> None:
        state.start_task = None
        if not self._pairing_v2_current(state):
            return
        from ...api.terralab_client import TerraLabClient

        state.code = code
        state.connect_url = connect_url
        if state.listener is not None:
            state.listener.set_code(code)

        self._start_pairing_poll(TerraLabClient(), code, status_only=True,
                                 total_timeout_s=float(expires_in))
        worker = self._pairing_worker
        if worker is not None:
            worker.pairing_confirmed.connect(
                lambda s=state: self._pairing_v2_on_confirmed(s))
        self._pairing_v2_open_browser(connect_url)
        if state.listener is None:


            self._pairing_v2_show_code_entry(state)

    def _pairing_v2_on_legacy(self, state: _PairingV2SignIn, reason: str) -> None:
        state.start_task = None
        if not self._pairing_v2_current(state):
            return
        QgsMessageLog.logMessage(
            f"Pairing: using the previous sign-in for this attempt ({reason})",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        state.legacy = True
        state.secret = _CLEARED
        if state.listener is not None:
            state.listener.close()
            state.listener = None
        self._start_legacy_pairing(state.attempt_id)

    def _pairing_v2_on_confirmed(self, state: _PairingV2SignIn) -> None:
        if not self._pairing_v2_current(state) or state.fallback_timer is not None:
            return
        if self.dock_widget is None:
            return
        from qgis.PyQt.QtCore import QTimer

        timer = QTimer(self.dock_widget)
        timer.setSingleShot(True)
        timer.setInterval(int(_FALLBACK_AFTER_S * 1000))
        timer.timeout.connect(lambda s=state: self._pairing_v2_show_code_entry(s))
        state.fallback_timer = timer
        timer.start()

    def _pairing_v2_show_code_entry(self, state: _PairingV2SignIn) -> None:

        if not self._pairing_v2_current(state) or not self.dock_widget:
            return
        try:
            self.dock_widget.show_pairing_code_entry()
        except RuntimeError:  # nosec B110
            return
        QgsMessageLog.logMessage(
            "Pairing: asking for the code shown in the browser",
            "AI Segmentation", level=Qgis.MessageLevel.Info)



    def _pairing_v2_on_grant(self, state: _PairingV2SignIn, grant: str, reply) -> None:

        if (not self._pairing_v2_current(state) or not state.code
                or state.grant_claim is not None):
            reply.go_back()
            return
        from ...api.terralab_client import TerraLabClient
        from ...workers.pairing_v2 import PairingClaimTask

        task = PairingClaimTask(TerraLabClient(), state.code, state.secret, grant=grant)
        task.answered.connect(
            lambda result, s=state, r=reply: self._pairing_v2_on_grant_answer(s, result, r))
        state.grant_claim = task
        QgsApplication.taskManager().addTask(task)

    def _on_pairing_code_entered(self, typed: str) -> None:

        state = self._pairing_v2
        if (state is None or state.ended or state.legacy or not state.code
                or state.typed_claim is not None or not self.dock_widget):
            return
        from ...workers.pairing_v2 import normalize_user_code

        user_code = normalize_user_code(typed)
        if not user_code:
            self._pairing_v2_wrong_code()
            return
        from ...api.terralab_client import TerraLabClient
        from ...workers.pairing_v2 import PairingClaimTask

        self.dock_widget.set_pairing_code_busy(True)
        task = PairingClaimTask(TerraLabClient(), state.code, state.secret,
                                user_code=user_code)
        task.answered.connect(
            lambda result, s=state: self._pairing_v2_on_typed_answer(s, result))
        state.typed_claim = task
        QgsApplication.taskManager().addTask(task)

    def _pairing_v2_on_grant_answer(self, state: _PairingV2SignIn, result: dict,
                                    reply) -> None:
        state.grant_claim = None
        status = result.get("status")
        if not self._pairing_v2_current(state):
            reply.go_back()
            return
        if status == "ready":
            reply.redirect(f"{self._pairing_v2_site()}/connect/done?product=ai-segmentation")
            self._on_pairing_succeeded(result.get("key", ""))
            return


        reply.go_back()
        QgsMessageLog.logMessage(
            f"Pairing: the browser came back, claim answered {status}",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        self._pairing_v2_claim_ended(state, status, typed=False)

    def _pairing_v2_on_typed_answer(self, state: _PairingV2SignIn, result: dict) -> None:
        state.typed_claim = None
        if not self._pairing_v2_current(state):
            return
        status = result.get("status")
        if status == "ready":
            if self.dock_widget:
                self.dock_widget.set_pairing_code_busy(False)
            self._on_pairing_succeeded(result.get("key", ""))
            return
        QgsMessageLog.logMessage(
            f"Pairing: the typed code was answered {status}",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        self._pairing_v2_claim_ended(state, status, typed=True)

    def _pairing_v2_claim_ended(self, state: _PairingV2SignIn, status, typed: bool) -> None:








        others_pending = (state.typed_claim if not typed else state.grant_claim) is not None
        if status == "not_found" and others_pending:
            if typed:

                state.typed_held = True
            QgsMessageLog.logMessage(
                "Pairing: holding not_found while another claim of the code is out",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            return
        typed_held = state.typed_held and not others_pending
        if typed_held:
            state.typed_held = False
        if typed or typed_held:
            if self.dock_widget:
                self.dock_widget.set_pairing_code_busy(False)
        if status == "locked":
            self._on_pairing_failed(
                tr("Too many wrong codes. Click Sign in to start again."), "LOCKED")
            return
        if status == "not_found":
            self._on_pairing_failed(
                tr("This sign-in has expired. Click Sign in to start again."), "NOT_FOUND")
            return
        if self._pairing_v2_end_on_terminal(status):
            return
        if typed and status in ("invalid", "invalid_request"):
            self._pairing_v2_wrong_code()
            return
        if typed or typed_held:
            if self.dock_widget:
                self.dock_widget.set_activation_message(
                    tr("Could not check the code. Try again."), is_error=True)


    def _pairing_v2_end_on_terminal(self, status) -> bool:


        from ...workers.pairing_v2 import pairing_cancelled_message, pairing_no_plan_message

        if status == "no_plan":
            self._on_pairing_failed(pairing_no_plan_message(), "NO_PLAN")
            return True
        if status == "cancelled":
            self._on_pairing_failed(pairing_cancelled_message(), "CANCELLED")
            return True
        return False

    def _pairing_v2_wrong_code(self) -> None:
        if self.dock_widget:
            self.dock_widget.set_activation_message(
                tr("Wrong code. Try again."), is_error=True)

    def _pairing_v2_site(self) -> str:
        from ...api.terralab_client import TerraLabClient

        return TerraLabClient().base_url



    def _pairing_v2_server_code(self) -> str:

        state = self._pairing_v2
        if state is None or state.legacy:
            return ""
        return state.code

    def _pairing_v2_active(self) -> bool:
        state = self._pairing_v2
        return state is not None and not state.ended

    def _pairing_v2_end(self) -> None:


        state, self._pairing_v2 = self._pairing_v2, None
        if state is None:
            return
        state.ended = True
        state.secret = _CLEARED
        timer, state.fallback_timer = state.fallback_timer, None
        if timer is not None:
            try:
                timer.stop()
                timer.deleteLater()
            except RuntimeError:  # nosec B110
                pass
        listener, state.listener = state.listener, None
        if listener is not None:
            listener.close()
        task, state.start_task = state.start_task, None
        if task is not None:
            safe_disconnect(task, "started")
            safe_disconnect(task, "use_legacy")
            try:
                if task.is_active():
                    task.cancel()
            except RuntimeError:  # nosec B110
                pass
        if state.legacy or not state.code:
            return
        worker = self._pairing_worker
        if worker is not None and worker.pairing_code == state.code:
            for signal_name in ("pairing_succeeded", "pairing_failed", "pairing_timeout",
                                "pairing_stalled", "pairing_confirmed"):
                safe_disconnect(worker, signal_name)
            try:
                if worker.is_active():
                    worker.cancel()
            except RuntimeError:  # nosec B110
                pass
