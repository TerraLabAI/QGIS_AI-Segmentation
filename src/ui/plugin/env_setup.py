"""First-time setup, install/download workers, activation, pairing, settings.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import sys

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

from ...core.i18n import tr
from ..background_workers import DepsInstallWorker, DownloadWorker, VerifyWorker
from ..error_report_dialog import show_error_report
from .shared import (
    _get_change_path_instructions,
)


class EnvSetupMixin:
    """First-time setup, install/download workers, activation, pairing, settings."""

    def toggle_dock_widget(self):
        just_created = not self._dock_created
        self._ensure_dock_widget()
        if self.dock_widget:
            if just_created or not self.dock_widget.isVisible():
                self.dock_widget.show()
                self.dock_widget.raise_()
                QgsMessageLog.logMessage(
                    f"Dock shown (first_create={just_created})",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            else:
                self.dock_widget.hide()
                QgsMessageLog.logMessage(
                    "Dock hidden", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _do_first_time_setup(self):
        QgsMessageLog.logMessage(
            "Panel opened - checking dependencies...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        # All environment checks (legacy cleanup, venv status, checkpoint
        # presence) run off the UI thread: get_venv_status walks the venv and
        # hashes dependency specs, which froze the dock for 1-2 s on open.
        if self._startup_check_worker is not None and self._startup_check_worker.isRunning():
            return
        from ..background_workers import StartupCheckWorker
        self._startup_check_worker = StartupCheckWorker()
        self._startup_check_worker.done.connect(self._on_startup_check_finished)
        self._startup_check_worker.start()

        # Check for plugin updates with retries - QGIS repo metadata may not
        # be ready after just a few seconds, especially on slower connections.
        from qgis.PyQt.QtCore import QTimer
        self._update_check_delays = [5000, 30000, 60000, 120000]
        self._update_check_index = 0
        QTimer.singleShot(
            self._update_check_delays[0], self._check_for_plugin_update)

    def _on_startup_check_finished(self, venv_ready: bool, message: str, checkpoint_ok: bool):
        # Cache the venv-installed state for the Refine-in-Manual env gate.
        self._env_ready = bool(venv_ready)
        if not self.dock_widget:
            return

        if message.startswith("startup_error:"):
            detail = message[len("startup_error:"):].strip()
            QgsMessageLog.logMessage(
                f"Dependency check error: {detail}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_dependency_status(False, f"Error: {detail[:50]}")
            # Do NOT leave the env-check latched done on an error: reset the guard
            # so a later Manual/Refine re-entry re-attempts the check instead of
            # stranding the error status until a manual Install click.
            self._interactive_setup_done = False
            return

        if venv_ready:
            self.dock_widget.set_dependency_status(True, "✓ " + tr("Dependencies ready"))
            QgsMessageLog.logMessage(
                "✓ Virtual environment verified successfully",
                "AI Segmentation",
                level=Qgis.MessageLevel.Success
            )
            self._start_device_info_worker()
            if checkpoint_ok:
                self.dock_widget.set_checkpoint_status(True, "SAM model ready")
                self._load_predictor()
                self._refresh_activation_async()
            else:
                # Model missing but deps ok.
                self.dock_widget.set_dependency_status(
                    True, tr("Dependencies ready, model not downloaded"))
                # A pending Refine handoff / background install is waiting on the
                # model: download it now so the deferred import can complete,
                # instead of stranding the user behind a manual Download button.
                if (self._pending_refine_import
                        or getattr(self, "_refine_install_pending", False)):  # noqa: W503
                    self._auto_download_checkpoint()
                    return
                # Show the Download-Model button ONLY in Manual mode: without the
                # mode guard this install UI leaked onto the Automatic page when a
                # StartupCheckWorker finished after a switch back to Automatic.
                from ..ai_segmentation_dockwidget import Mode
                if self.dock_widget._mode == Mode.INTERACTIVE:
                    self.dock_widget.install_button.setVisible(True)
                    self.dock_widget.install_button.setEnabled(True)
                    self.dock_widget.install_button.setText(tr("Download Model"))
                    self.dock_widget.setup_group.setVisible(True)
                    # Model download is pending: keep the setup section across a
                    # mode round trip (see dock _setup_section_wanted).
                    self.dock_widget._setup_section_wanted = True
        else:
            self.dock_widget.set_dependency_status(False, message)
            QgsMessageLog.logMessage(
                f"Virtual environment status: {message}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
            # Auto-trigger install for upgrades
            if "need updating" in message:
                self._on_install_requested()

    def _check_for_plugin_update(self):
        """Trigger the update check on the dock widget, retrying if needed."""
        if not self.dock_widget:
            return
        self.dock_widget.check_for_updates()

        # If notification is still hidden and we have retries left, schedule next
        if (not self.dock_widget.update_notification_widget.isVisible()
                and hasattr(self, "_update_check_delays")):  # noqa: W503
            self._update_check_index += 1
            if self._update_check_index < len(self._update_check_delays):
                from qgis.PyQt.QtCore import QTimer
                delay = self._update_check_delays[self._update_check_index]
                QTimer.singleShot(delay, self._check_for_plugin_update)

    def _load_predictor(self):
        """Kick off predictor initialization in the background (#34)."""
        from ..background_workers import PredictorLoadWorker
        if self.dock_widget:
            self.dock_widget.set_checkpoint_status(True, tr("Loading AI model..."))
        self._predictor_worker = PredictorLoadWorker()
        self._predictor_worker.done.connect(self._on_predictor_loaded)
        self._predictor_worker.start()

    def _on_predictor_loaded(self, predictor, err_msg: str):
        if predictor is None:
            QgsMessageLog.logMessage(
                f"Failed to initialize predictor: {err_msg}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            if self.dock_widget:
                self.dock_widget.set_checkpoint_status(
                    False, tr("Model load failed"))
            # A background install started from the Automatic review can no longer
            # complete: drop the pending refine and re-enable the Refine button.
            self._abort_refine_install()
            return
        self.predictor = predictor
        QgsMessageLog.logMessage(
            "SAM predictor initialized (subprocess mode)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        if self.dock_widget:
            self.dock_widget.set_checkpoint_status(True, tr("SAM model ready"))
            self.dock_widget.set_install_progress(100, tr("Ready"))
        # Warm the model NOW when Manual use is predictable, so the first
        # session's first click never waits out the model load: right after a
        # fresh install/model download (the user is onboarding, about to try),
        # or when this machine ran a Manual session recently (regular Manual
        # users reach their resident-model steady state at startup instead of
        # at their first click; the model stays loaded until QGIS closes
        # anyway once a session runs). Automatic-only users trip neither
        # signal, so they never pay the model's RAM. warm_up() is idempotent
        # and cheap on this thread (Popen + one init line; the load happens
        # inside the subprocess). Best-effort: the Start-time warm_up retries.
        try:
            if not self._headless and (
                getattr(self, "_warm_predictor_on_ready", False)
                or self._manual_used_recently()  # noqa: W503
            ):
                self._warm_predictor_on_ready = False
                predictor.warm_up()
                QgsMessageLog.logMessage(
                    "Pre-warming the SAM worker (Manual use predicted)",
                    "AI Segmentation", level=Qgis.MessageLevel.Info
                )
        except Exception:  # noqa: BLE001 - prediction of intent must never break load
            pass  # nosec B110
        # D1: a background install kicked off from the Automatic review (the user
        # clicked Refine with no local AI) has finished. Clear the inline banner
        # and, since the user already asked to refine, open the handoff now on the
        # still-intact review. If they left the review meanwhile, just release the
        # flag (the AI is now ready for next time).
        if getattr(self, "_refine_install_pending", False):
            self._refine_install_pending = False
            if self.dock_widget:
                try:
                    self.dock_widget.set_auto_review_installing(False)
                except (RuntimeError, AttributeError):
                    pass
            if self._auto_review:
                self._on_refine_in_manual_clicked()
            return
        # Complete a Refine-in-Manual handoff that arrived before the model was
        # loaded: now that the predictor is up, start the Manual session and
        # import the held detections as editable polygons.
        if self._pending_refine_import and self._refine_handoff_active:
            self._pending_refine_import = False
            layer = getattr(self, "_handoff_source_layer", None)
            review = self._auto_review
            if layer is not None and review:
                if self.dock_widget:
                    try:
                        self.dock_widget.set_refine_handoff_preparing(False)
                    except (RuntimeError, AttributeError):
                        pass
                    try:
                        combo = self.dock_widget.layer_combo
                        combo.blockSignals(True)
                        combo.setLayer(layer)
                        combo.blockSignals(False)
                    except (RuntimeError, AttributeError):
                        pass
                self._on_start_segmentation(layer)
                self._import_review_geoms_as_saved(review)

    def _manual_used_recently(self, days: int = 14) -> bool:
        """True when this machine ran a Manual session within ``days``.

        The timestamp is written by _on_start_segmentation. Drives the
        predictive model warm-up in _on_predictor_loaded: a machine that
        segmented manually this fortnight will almost surely do it again,
        while an Automatic-only machine never gets the key at all."""
        try:
            import time
            from qgis.PyQt.QtCore import QSettings
            ts = QSettings().value(
                "AISegmentation/last_manual_session_ts", 0, type=int)
            return 0 < ts and (time.time() - ts) < days * 86400
        except Exception:  # noqa: BLE001 - heuristic only
            return False

    def _start_device_info_worker(self):
        """Detect and log the compute device off the UI thread (diagnostics)."""
        if self._device_info_worker is not None and self._device_info_worker.isRunning():
            return
        from ..background_workers import DeviceInfoWorker
        self._device_info_worker = DeviceInfoWorker()
        self._device_info_worker.done.connect(self._on_device_info)
        self._device_info_worker.start()

    def _on_device_info(self, ok: bool, info: str):
        if ok:
            QgsMessageLog.logMessage(
                f"Device info: {info}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
            return

        # Check if this is a PyTorch DLL error (Windows)
        if "DLL" in info or "shm.dll" in info:
            QgsMessageLog.logMessage(
                f"PyTorch DLL error: {info}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            if self._headless:
                return
            # Show user-friendly error dialog
            from qgis.PyQt.QtWidgets import QMessageBox
            msg_box = QMessageBox(self.iface.mainWindow())
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle(tr("PyTorch Error"))
            msg_box.setText(tr("PyTorch cannot load on Windows"))
            msg_box.setInformativeText(
                tr("The plugin requires Visual C++ Redistributables to run PyTorch.\n\n"
                   "Please download and install:\n"
                   "https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
                   "After installation, restart QGIS and try again."))
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
        else:
            QgsMessageLog.logMessage(
                f"Could not determine device info: {info}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    def _on_install_requested(self):
        # Manual (local) install needs to download/execute a standalone
        # Python and build a multi-GB venv, both unreliable or blocked under
        # Flatpak/Snap confinement. Fail fast with a clear message instead of
        # letting the install grind through a cryptic low-level failure;
        # Automatic mode needs no local install and works in the sandbox.
        from ...core.python_manager import is_sandboxed_linux
        if is_sandboxed_linux():
            QgsMessageLog.logMessage(
                "Manual install blocked: running inside a Flatpak/Snap sandbox",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_dependency_status(
                False, tr("Manual mode is not supported in this QGIS installation"))
            show_error_report(
                self.iface.mainWindow(),
                tr("Manual Mode Not Supported"),
                tr(
                    "Manual mode needs to install local dependencies, which is "
                    "not supported inside this sandboxed QGIS installation "
                    "(Flatpak or Snap). Please use Automatic mode instead, "
                    "which runs fully in the cloud and needs no local install."
                ),
                error_code="sandboxed_linux_manual_blocked",
            )
            return

        # Intel Macs on QGIS 4 (Python 3.13+) have no installable local runtime
        # (the Intel-Mac fallback has no build for this Python). Fail fast with a
        # clear message instead of a cryptic dependency-resolver error; Automatic
        # mode runs in the cloud and needs no local install.
        from ...core.model_config import MACOS_X86_NO_LOCAL_INFERENCE
        if MACOS_X86_NO_LOCAL_INFERENCE:
            QgsMessageLog.logMessage(
                "Manual install blocked: no local inference build for this Mac + Python",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_dependency_status(
                False, tr("Manual mode is not supported in this QGIS installation"))
            show_error_report(
                self.iface.mainWindow(),
                tr("Manual Mode Not Supported"),
                tr(
                    "Manual mode installs local components that are not available "
                    "for this Mac with this version of QGIS. Please use Automatic "
                    "mode instead, which runs fully in the cloud and needs no "
                    "local install."
                ),
                error_code="macos_x86_no_local_inference",
            )
            return

        # Guard: prevent concurrent installs
        if self.deps_install_worker is not None and self.deps_install_worker.isRunning():
            QgsMessageLog.logMessage(
                "Install already in progress, ignoring duplicate request",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            return

        from ...core.venv_manager import get_venv_status

        is_ready, message = get_venv_status()
        if is_ready:
            # Deps already installed, just need model download
            self.dock_widget.set_dependency_status(True, "✓ " + tr("Dependencies ready"))
            self._auto_download_checkpoint()
            return

        QgsMessageLog.logMessage(
            "Starting virtual environment creation and dependency installation...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        QgsMessageLog.logMessage(
            f"Platform: {sys.platform}, Python: {sys.version}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        self.dock_widget.set_install_progress(0, "Preparing installation...")

        import time as _time
        self._install_t0 = _time.monotonic()
        try:
            from ...core import telemetry
            telemetry.track_install_started()
        except Exception:
            pass  # nosec B110

        self.deps_install_worker = DepsInstallWorker()
        self.deps_install_worker.progress.connect(self._on_deps_install_progress)
        self.deps_install_worker.done.connect(self._on_deps_install_finished)
        self.deps_install_worker.start()

        # Surface the sign-in section 2 seconds after install starts
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(2000, self._refresh_activation_async)

    def _refresh_activation_async(self, force: bool = False):
        """Refresh the signed-in state without ever blocking the UI thread.

        Re-validates the stored key in a hidden QgsTask; the UI keeps its
        current state meanwhile (benefit of the doubt). Two safeguards for flaky
        networks: a connectivity failure never signs the user out (see
        _on_key_revalidate_failed), and the check is throttled to once per 15 min
        so a restored-open dock does not re-poll on every visibility event.
        """
        import time

        from ...core.activation_manager import is_plugin_activated
        if not self.dock_widget:
            return
        if not is_plugin_activated():
            if not self.dock_widget.is_activated():
                self.dock_widget._update_full_ui()
            return
        # Activated on startup (stored key): the dock restores straight to its
        # last tab, so if that tab is Automatic, _on_mode_changed never fires and
        # the upsell would stay visible for a Pro account until the user toggles
        # modes. Fetch usage now so the Pro/Free state is correct immediately.
        self._refresh_auto_credits()
        # Throttle: one revalidation per 15 min. Sign-in / sign-out reset the
        # stamp (to 0.0) so a fresh state revalidates immediately.
        if not force and (time.time() - self._last_key_validation_unix) < 900:
            return
        if self._key_revalidate_task is not None and self._key_revalidate_task.is_active():
            return
        from qgis.core import QgsApplication

        # Auth header is read here (main thread); the task only talks HTTP.
        from ...api.terralab_client import TerraLabClient
        from ...core.activation_manager import get_auth_header
        from ...workers.generic_request_task import GenericRequestTask
        auth = get_auth_header()
        client = TerraLabClient()
        self._key_revalidate_task = GenericRequestTask(
            tr("Checking your AI Segmentation subscription"),
            lambda: client.get_usage(auth=auth),
            hidden=True,
        )
        self._key_revalidate_task.succeeded.connect(self._on_key_revalidate_ok)
        self._key_revalidate_task.failed.connect(self._on_key_revalidate_failed)
        QgsApplication.taskManager().addTask(self._key_revalidate_task)

    def _on_key_revalidate_ok(self, _usage: object) -> None:
        import time
        self._key_revalidate_task = None
        self._last_key_validation_unix = time.time()

    def _on_key_revalidate_failed(self, message: str, code: str) -> None:
        self._key_revalidate_task = None
        from ...core.activation_manager import is_rejection_code
        # Only a genuine auth rejection signs the user out. A network blip must
        # NOT clear the stored key (that was the old bug that logged users out
        # offline); we keep the session and surface a non-blocking notice.
        if is_rejection_code(code):
            from ...core.activation_manager import clear_auth
            clear_auth()
            self._last_key_validation_unix = 0.0
            if self.dock_widget:
                self.dock_widget.set_activated_state(False)
            # A forced sign-out is a churn-grade moment (device limit hit,
            # subscription lapsed, key revoked): make it visible post-hoc.
            try:
                from ...core import telemetry
                telemetry.track_plugin_error(
                    stage="activate",
                    error_code=(code or "key_rejected").lower(),
                    message="stored key rejected on revalidation",
                )
            except Exception:
                pass  # nosec B110
            return
        self._notify_connection_issue(code, message)

    # --- One-click connect (browser pairing handoff) ------------------------

    def _on_pairing_requested(self, code: str):
        """Open the browser to /connect and start polling for the key.

        Re-entrant: if a poll is already running (the user clicked "open the
        page again"), we only re-open the browser instead of starting a second
        worker.
        """
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices

        from ...api.terralab_client import TerraLabClient

        client = TerraLabClient()
        # Build the connect URL from the client base so .env.local
        # TERRALAB_BASE_URL is honored in dev. Never log the code or full URL.
        url = (
            f"{client.base_url}/connect?code={code}&product=ai-segmentation"
            "&utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
            "&utm_content=connect"
        )
        opened = QDesktopServices.openUrl(QUrl(url))
        if not opened:
            self.dock_widget.show_pairing_idle()
            self.dock_widget.set_activation_message(
                tr("Couldn't open your browser. Check your connection and click "
                   "Sign in / Sign up to start again."),
                is_error=True,
            )
            return

        if self._pairing_worker is not None and self._pairing_worker.is_active():
            # Already polling for this code; the browser was just re-opened.
            return

        from qgis.core import QgsApplication

        from ...workers.pairing_poll_task import PairingPollTask
        self._pairing_worker = PairingPollTask(client, code)
        self._pairing_worker.pairing_succeeded.connect(self._on_pairing_succeeded)
        self._pairing_worker.pairing_failed.connect(self._on_pairing_failed)
        self._pairing_worker.pairing_timeout.connect(self._on_pairing_timeout)
        QgsApplication.taskManager().addTask(self._pairing_worker)
        import time as _time
        self._pairing_t0 = _time.monotonic()
        try:
            from ...core import telemetry
            telemetry.track_pairing_started()
        except Exception:
            pass  # nosec B110
        QgsMessageLog.logMessage(
            "Pairing started", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _pairing_elapsed_ms(self) -> int | None:
        """Elapsed ms since the current browser sign-in poll started."""
        try:
            import time as _time
            t0 = getattr(self, "_pairing_t0", None)
            return int((_time.monotonic() - t0) * 1000) if t0 else None
        except Exception:
            return None

    def _on_pairing_succeeded(self, key: str):
        from ...core.activation_manager import save_auth_token
        save_auth_token(key)
        # Fresh sign-in: clear the revalidation throttle so the next dock event
        # re-checks the new key immediately instead of waiting out the window.
        self._last_key_validation_unix = 0.0
        if self.dock_widget:
            self.dock_widget.set_activated_state(True)
            # Reflect the real plan (Pro vs free) in the Automatic panel right
            # away, so the upsell card is hidden immediately for a Pro account
            # instead of waiting for the next mode toggle.
            self._refresh_auto_credits()
        # Bring QGIS back to front so the user sees the activated dock.
        try:
            mw = self.iface.mainWindow()
            mw.activateWindow()
            mw.raise_()
            if self.dock_widget:
                self.dock_widget.raise_()
        except Exception:  # nosec B110
            pass
        try:
            from ...core.telemetry import track_plugin_activated
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
            from ...core import telemetry
            telemetry.track_pairing_failed(
                error_code=code or "unknown",
                duration_ms=self._pairing_elapsed_ms(),
            )
        except Exception:
            pass  # nosec B110
        QgsMessageLog.logMessage(
            f"Pairing failed ({code})", "AI Segmentation",
            level=Qgis.MessageLevel.Warning)

    def _on_pairing_timeout(self):
        if self.dock_widget:
            self.dock_widget.show_pairing_idle()
            self.dock_widget.set_activation_message(
                tr("Sign-in timed out. Click Connect to try again."),
                is_error=True,
            )
        try:
            from ...core import telemetry
            telemetry.track_pairing_failed(
                error_code="timeout",
                duration_ms=self._pairing_elapsed_ms(),
            )
        except Exception:
            pass  # nosec B110
        QgsMessageLog.logMessage(
            "Pairing timed out", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _cancel_pairing_worker(self):
        """Cancel any in-flight pairing poll (never terminate())."""
        if self._pairing_worker is not None and self._pairing_worker.is_active():
            try:
                self._pairing_worker.cancel()
            except RuntimeError:
                pass

    def _cancel_task(self, attr: str):
        """Cooperatively cancel a GenericRequestTask held on ``self.<attr>``.

        Disconnect its signals so a late result can't fire into torn-down UI,
        then cancel it. We NEVER terminate a network-bound thread: the task
        manager drains run() on its own, and terminate() on a wedged socket
        crashes QGIS.
        """
        task = getattr(self, attr, None)
        if task is None:
            return
        try:
            task.succeeded.disconnect()
            task.failed.disconnect()
        except (RuntimeError, TypeError):  # nosec B110
            pass
        try:
            if task.is_active():
                task.cancel()
        except Exception:  # nosec B110
            pass
        setattr(self, attr, None)

    # Network error codes (from terralab_client._classify_network_error) that
    # mean "no/unstable connection" rather than an auth or app error.
    _CONNECTIVITY_CODES = frozenset({
        "DNS_ERROR", "CONNECTION_REFUSED", "TIMEOUT",
        "SSL_ERROR", "PROXY_ERROR", "NO_INTERNET",
    })
    _CONN_NOTICE_MIN_GAP_S = 60.0

    def _notify_connection_issue(self, code: str, message: str):
        """Show ONE transient, dismissible 'no connection' notice in the message
        bar. Non-blocking (never freezes), throttled so a flaky network does not
        spam the user, and silent for non-connectivity codes (auth/app errors
        surface through their own paths)."""
        import time

        if (code or "").strip().upper() not in self._CONNECTIVITY_CODES:
            return
        now = time.monotonic()
        if now - self._last_conn_notice_monotonic < self._CONN_NOTICE_MIN_GAP_S:
            return
        self._last_conn_notice_monotonic = now
        try:
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                message or tr("Network error. Check your internet connection."),
                level=Qgis.MessageLevel.Warning,
                duration=8,
            )
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def _on_cancel_pairing(self, code: str = ""):
        self._cancel_pairing_worker()
        try:
            from ...core import telemetry
            telemetry.track_pairing_cancelled(duration_ms=self._pairing_elapsed_ms())
        except Exception:
            pass  # nosec B110
        if code:
            # Retire the code server-side so a later Confirm in the browser
            # shows "expired" instead of binding a key nobody is polling for.
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

    # --- Settings / sign out -------------------------------------------------

    def _on_settings_clicked(self):
        from ...core.activation_manager import get_auth_header, get_auth_token, is_plugin_activated
        if not is_plugin_activated():
            return
        from ...api.terralab_client import TerraLabClient
        from ..account_settings_dialog import AccountSettingsDialog
        client = TerraLabClient()
        dlg = AccountSettingsDialog(
            client=client,
            auth=get_auth_header(),
            activation_key=get_auth_token(),
            parent=self.iface.mainWindow(),
            on_remove_ai_data=self.remove_local_ai_data,
            is_busy_check=self.is_local_ai_busy,
        )
        dlg.sign_out_requested.connect(self._on_sign_out_requested)
        dlg.exec()

    def is_local_ai_busy(self) -> bool:
        """True while any local install/download/verify/model-load worker or a
        detection run is in flight, so removing the downloaded AI data would
        corrupt an operation. A loaded-but-idle predictor is NOT busy: it is
        shut down cleanly by remove_local_ai_data before any deletion. Fail
        safe: if the state cannot be read, report busy so nothing is deleted
        out from under a live operation."""
        try:
            for attr in (
                "deps_install_worker", "download_worker", "_verify_worker",
                "_predictor_worker", "_startup_check_worker", "_auto_worker",
                # The device probe imports torch from the venv; deleting the
                # venv under it locks files on Windows and faults the probe.
                "_device_info_worker",
                # The async crop encode talks to the SAM subprocess; deleting
                # the venv under it kills the process mid-round-trip.
                "_manual_encode_worker",
            ):
                worker = getattr(self, attr, None)
                if worker is None:
                    continue
                try:
                    if worker.isRunning():
                        return True
                except (RuntimeError, AttributeError):
                    continue
            return False
        except Exception:  # nosec B110
            return True

    def remove_local_ai_data(self) -> tuple[bool, str]:
        """Delete every trace of the local AI install this plugin created: the
        isolated venv + downloaded model weights (all under CACHE_DIR), the
        stored activation key (QgsAuthManager + the QSettings fallback) and this
        plugin's own QSettings groups. Signs the user out as a side effect.

        Returns (ok, message). Refuses (returns False) while an install or a
        detection run is in flight. Never raises: any partial-failure detail is
        folded into the returned message so the dialog can show it."""
        import os
        import shutil

        from ...core.venv_manager import CACHE_DIR

        if self.is_local_ai_busy():
            return False, tr(
                "An install or detection is still running. Wait for it to "
                "finish, then try again.")

        # 1. Shut the loaded predictor subprocess down first so its handles on
        #    the venv are released before we delete it (a live process locks
        #    those files on Windows). Bounded, mirrors unload.
        if getattr(self, "predictor", None) is not None:
            import threading
            pred = self.predictor
            self.predictor = None
            try:
                t = threading.Thread(target=lambda: pred.cleanup(), daemon=True)
                t.start()
                t.join(timeout=8)
            except Exception:  # nosec B110
                pass
        self._env_ready = False
        self._first_time_setup_done = False

        # 2. Clear the activation key (removes the QgsAuthManager config via the
        #    authcfg_id, which still lives in QSettings at this point) BEFORE
        #    wiping the QSettings groups, else the stored config would leak.
        errors: list[str] = []
        try:
            from ...core.activation_manager import clear_auth
            clear_auth()
        except Exception as err:  # nosec B110
            errors.append(str(err)[:80])
        self._last_key_validation_unix = 0.0

        # 3. Wipe this plugin's own QSettings groups. Leave the shared TerraLab/
        #    keys (telemetry opt-out, device seed) untouched: they are shared
        #    with the sibling plugin and are not "downloaded AI data".
        try:
            from qgis.PyQt.QtCore import QSettings
            settings = QSettings()
            for group in ("AISegmentation", "AI_Segmentation",
                          "TerraLab/AI_Segmentation"):
                settings.remove(group)
            settings.sync()
        except Exception as err:  # nosec B110
            errors.append(str(err)[:80])

        # 4. Delete the venv + weights + feature cache (all under CACHE_DIR).
        freed = ""
        try:
            if os.path.isdir(CACHE_DIR):
                freed = self._dir_size_label(CACHE_DIR)
                shutil.rmtree(CACHE_DIR, ignore_errors=True)
            if os.path.isdir(CACHE_DIR):
                # Some files survived (locked / permissions): report it rather
                # than claim a clean removal.
                errors.append(tr("some files could not be deleted"))
        except Exception as err:  # nosec B110
            errors.append(str(err)[:80])

        # 5. Reflect the sign-out in the dock so the UI is consistent.
        if self.dock_widget:
            try:
                self.dock_widget.set_activated_state(False)
            except (RuntimeError, AttributeError):
                pass

        QgsMessageLog.logMessage(
            "Local AI data removed ({}), errors={}".format(
                freed or "0 MB", len(errors)),
            "AI Segmentation", level=Qgis.MessageLevel.Info)

        if errors:
            return True, tr(
                "AI data removed, but some items could not be fully cleared. "
                "You can delete the folder manually.")
        return True, tr("Downloaded AI data removed. You have been signed out.")

    @staticmethod
    def _dir_size_label(path: str) -> str:
        """Human-readable total size of a directory tree (for the removal log
        line). Best-effort, never raises."""
        import os
        total = 0
        try:
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:  # nosec B112
                        continue
        except OSError:
            return "-"
        mb = total / (1024 * 1024)
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"

    def _on_sign_out_requested(self):
        from ...core.activation_manager import clear_auth
        clear_auth()
        self._last_key_validation_unix = 0.0
        if self.dock_widget:
            self.dock_widget.set_activated_state(False)

    def _on_deps_install_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return
        # Scale deps progress to 0-80% (model download gets 80-100%)
        scaled = int(percent * 0.8)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_deps_install_finished(self, success: bool, message: str):
        if not self.dock_widget:
            return

        if success:
            # install_completed fires in _on_verify_finished: a venv that fails
            # verification (broken torch DLL) is a FAILED install to the user,
            # so counting it completed here skewed the install funnel.
            # Run verification + device detection off main thread
            self.dock_widget.set_install_progress(80, tr("Verifying installation..."))
            self._verify_worker = VerifyWorker()
            self._verify_worker.progress.connect(self._on_verify_progress)
            self._verify_worker.done.connect(self._on_verify_finished)
            self._verify_worker.start()
        else:
            self.dock_widget.set_install_progress(100, "Failed")
            error_msg = message[:300] if message else tr("Unknown error")
            self.dock_widget.set_dependency_status(False, tr("Installation failed"))
            # A background install from the Automatic review can no longer finish.
            self._abort_refine_install()

            error_title = tr("Installation Failed")
            error_code = "installation_failed"
            msg_lower = message.lower() if message else ""
            from ...core.pip_diagnostics import is_disk_full
            if "not enough free disk space" in msg_lower or is_disk_full(msg_lower):
                # Preflight or mid-install disk exhaustion: the message already
                # carries the free-up / AI_SEGMENTATION_CACHE_DIR guidance.
                error_title = tr("Not Enough Disk Space")
                error_code = "disk_space"
            elif any(p in msg_lower for p in [
                "ssl", "certificate verify", "sslerror",
                "unable to get local issuer",
                # uv (rustls) wording, no "ssl" substring anywhere
                "invalid peer certificate", "unknownissuer",
            ]):
                error_title = tr("SSL Certificate Error")
                error_code = "ssl_certificate_error"
            elif "file in use by qgis" in msg_lower:
                # Native module (.pyd/.dll) locked by the running QGIS process
                # during a torch upgrade - no dialog-splash of AV advice here,
                # the actionable fix is just "restart QGIS".
                from ...core.pip_diagnostics import get_file_locked_help
                error_title = tr("Restart QGIS Required")
                error_msg = get_file_locked_help()
                error_code = "restart_qgis_required"
            elif any(p in msg_lower for p in [
                "access is denied", "winerror 5", "winerror 225",
                "permission denied", "blocked",
                "cannot write to install",
                "cannot open the device or file",
            ]):
                error_title = tr("Installation Blocked")
                error_msg = f"{error_msg}\n\n{_get_change_path_instructions()}"
                error_code = "installation_blocked"
            elif any(p in msg_lower for p in [
                "network error", "connection aborted", "connection reset",
                "timed out", "timeout", "network connection failed",
                "connection broken", "could not resolve",
            ]):
                error_title = tr("Network Connection Problem")
                error_msg = "{}\n\n{}".format(
                    error_msg,
                    tr(
                        "Your connection appears unstable or blocked. "
                        "Check: (1) your internet is working, "
                        "(2) QGIS > Settings > Options > Network has a proxy "
                        "configured if you are on a corporate network, "
                        "(3) your firewall allows connections to pypi.org "
                        "and files.pythonhosted.org."
                    ),
                )
                error_code = "network_connection_problem"

            show_error_report(
                self.iface.mainWindow(),
                error_title,
                error_msg,
                error_code=error_code,
            )
            try:
                import time as _time
                from ...core import telemetry
                t0 = getattr(self, "_install_t0", 0.0)
                telemetry.track_install_failed(
                    error_class=error_code,
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None,
                    python_minor=sys.version_info.minor,
                )
            except Exception:
                pass  # nosec B110

    def _on_verify_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return
        # Scale verify progress (0-100%) into the 80-95% range
        scaled = 80 + int(percent * 0.15)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_verify_finished(self, is_valid: bool, message: str):
        if not self.dock_widget:
            return
        if is_valid:
            # The install only counts as completed once the venv VERIFIES:
            # firing at deps-install success counted broken-torch installs
            # (DLL blocked) as completed and understated install_failed.
            try:
                import time as _time
                from ...core import telemetry
                t0 = getattr(self, "_install_t0", 0.0)
                if t0:
                    telemetry.track_install_completed(
                        duration_ms=int((_time.monotonic() - t0) * 1000),
                        python_minor=sys.version_info.minor,
                    )
                    self._install_t0 = 0.0
            except Exception:
                pass  # nosec B110
            self.dock_widget.set_dependency_status(True, "✓ " + tr("Dependencies ready"))
            if message and not message.startswith("device_error"):
                QgsMessageLog.logMessage(
                    f"Device info: {message}",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            elif message.startswith("device_error"):
                QgsMessageLog.logMessage(
                    "Could not determine device info: {}".format(
                        message.replace("device_error: ", "")),
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            try:
                self._auto_download_checkpoint()
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Auto-download checkpoint failed: {e}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                self.dock_widget.set_install_progress(100, "Failed")
                self.dock_widget.set_dependency_status(
                    True, tr("Dependencies ready, model download failed"))
                self._abort_refine_install()
        else:
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                False, "{} {}".format(tr("Verification failed:"), message))
            self._abort_refine_install()
            # A broken native module (torch DLL blocked by a missing VC++
            # runtime) is the dominant verify failure: route it to the
            # actionable fix-it steps instead of a dead-end generic dialog.
            from ...core.pip_diagnostics import get_vcpp_help, is_dll_init_error
            error_title = tr("Verification Failed")
            error_code = "verification_failed"
            body = "{}\n{}".format(
                tr("Virtual environment was created but verification failed:"),
                message)
            low = (message or "").lower()
            if is_dll_init_error(low) or "required dll failed to initialize" in low:
                error_title = tr("A Component Failed to Load")
                error_code = "dll_init_failed"
                body = "{}\n\n{}".format(message, get_vcpp_help())
            show_error_report(
                self.iface.mainWindow(),
                error_title,
                body,
                error_code=error_code)
            try:
                import time as _time
                from ...core import telemetry
                t0 = getattr(self, "_install_t0", 0.0)
                telemetry.track_install_failed(
                    error_class=error_code,
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None,
                    python_minor=sys.version_info.minor,
                )
            except Exception:
                pass  # nosec B110

    def _on_cancel_install(self):
        if self.deps_install_worker and self.deps_install_worker.isRunning():
            self.deps_install_worker.cancel()
            try:
                import time as _time

                from ...core import telemetry
                t0 = getattr(self, "_install_t0", 0.0)
                telemetry.track_install_cancelled(
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None)
            except Exception:
                pass  # nosec B110
            QgsMessageLog.logMessage(
                "Installation cancelled by user",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    def _auto_download_checkpoint(self):
        """Auto-download model after deps install if not already present."""
        # This path only runs at the tail of a fresh install/repair: the user
        # is onboarding and will click Start within moments, so ask
        # _on_predictor_loaded to warm the model as soon as it is ready.
        self._warm_predictor_on_ready = True
        from ...core.checkpoint_manager import checkpoint_exists
        try:
            if checkpoint_exists():
                self.dock_widget.set_install_progress(95, tr("Loading AI model..."))
                self._load_predictor()
                self._refresh_activation_async()
                return
        except Exception:
            pass  # nosec B110

        self.dock_widget.set_install_progress(80, tr("Downloading AI model..."))
        try:
            import time as _time
            self._model_download_t0 = _time.monotonic()
            self.download_worker = DownloadWorker()
            self.download_worker.progress.connect(self._on_download_progress)
            self.download_worker.done.connect(self._on_download_finished)
            self.download_worker.start()
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to start model download: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                True, tr("Dependencies ready, model download failed"))
            self._abort_refine_install()

    def _on_download_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return
        # Scale model download to 80-100% of the unified progress
        scaled = 80 + int(percent * 0.2)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_download_finished(self, success: bool, message: str):
        if not self.dock_widget:
            return
        if success:
            try:
                import time as _time

                from ...core import telemetry
                from ...core.model_config import USE_SAM2
                t0 = getattr(self, "_model_download_t0", 0.0)
                telemetry.track_model_download_completed(
                    model="sam2" if USE_SAM2 else "sam1",
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None)
            except Exception:
                pass  # nosec B110
            self.dock_widget.set_install_progress(95, tr("Loading AI model..."))
            self._load_predictor()
            self._refresh_activation_async()
        else:
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                True, tr("Dependencies ready, model download failed"))
            self._abort_refine_install()

            show_error_report(
                self.iface.mainWindow(),
                tr("Download Failed"),
                "{}\n{}".format(
                    tr("Failed to download model:"),
                    message),
                error_code="download_failed",
            )

    def _recover_corrupt_checkpoint(self, deleted: bool) -> bool:
        """Recover from a corrupt model checkpoint detected during encoding.

        ``deleted`` reports whether the bad file was removed. When it was, we
        kick off a fresh download (which re-verifies the hash and reloads the
        predictor on success). When it could not be removed, we ask the user
        to clear the cache folder manually. Always returns False so the caller
        aborts the current encode. (#65)
        """
        from ...core.checkpoint_manager import get_checkpoints_dir

        if not deleted:
            msg = tr(
                "The AI model file is corrupted but could not be removed "
                "automatically. Please delete this folder and restart QGIS:"
            ) + "\n" + get_checkpoints_dir()
            if self._headless:
                self._headless_error = msg
                return False
            show_error_report(
                self.iface.mainWindow(),
                tr("Model File Corrupted"),
                msg,
                error_code="checkpoint_corrupt",
            )
            return False

        msg = tr(
            "The AI model file was corrupted and is being re-downloaded. "
            "Please try your selection again once it finishes."
        )
        if self._headless:
            self._headless_error = msg
            return False

        from qgis.PyQt.QtWidgets import QMessageBox
        QMessageBox.information(
            self.iface.mainWindow(), tr("Re-downloading Model"), msg)
        try:
            self._auto_download_checkpoint()
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to re-download checkpoint after corruption: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False

    def _recover_broken_venv(self, error: str) -> bool:
        """Recover from a venv that can no longer run.

        Happens when the base Python the venv points at was deleted by a
        cleanup tool or its recorded path got corrupted: site-packages look
        complete, the panel says ready, but every worker spawn fails.
        Reinstalling recreates the runtime in place; the package cache makes
        it a short repair, not a full re-download. Always returns False so
        the caller aborts the current encode. (#64)
        """
        QgsMessageLog.logMessage(
            f"Broken Python runtime detected, starting repair: {error[:200]}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)

        msg = tr(
            "The Python runtime used by the AI engine is damaged "
            "(this can be caused by a disk cleanup tool or antivirus). "
            "It will now be repaired automatically. "
            "Please try your selection again once the repair finishes."
        )
        if self._headless:
            self._headless_error = msg
            return False

        from qgis.PyQt.QtWidgets import QMessageBox
        QMessageBox.information(
            self.iface.mainWindow(), tr("Repairing Installation"), msg)
        self.dock_widget.set_dependency_status(
            False, tr("Repairing installation..."))
        try:
            self._on_install_requested()
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to start repair install: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False
