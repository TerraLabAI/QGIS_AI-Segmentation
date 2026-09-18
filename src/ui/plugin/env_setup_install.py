








from __future__ import annotations

import sys

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

from ...core.i18n import tr
from ...core.interaction_dials import install_pipe_wait_turns
from ..background_workers import (
    DepsInstallWorker,
    DownloadWorker,
    VerifyWorker,
)
from ..error_report_dialog import show_error_report
from .shared import _get_change_path_instructions

_INSTALL_ATTEMPT_KEY = "TerraLab/ai_seg_install_attempts"




_INSTALL_PIPE_WAIT_MAX = 8


def _bump_install_attempt() -> int:








    try:
        from qgis.PyQt.QtCore import QSettings
        s = QSettings()
        n = int(s.value(_INSTALL_ATTEMPT_KEY, 0, type=int)) + 1
        s.setValue(_INSTALL_ATTEMPT_KEY, n)
        return n
    except Exception:  # nosec B110
        return 0


def _clear_install_attempts() -> None:

    try:
        from qgis.PyQt.QtCore import QSettings
        QSettings().setValue(_INSTALL_ATTEMPT_KEY, 0)
    except Exception:  # nosec B110
        pass


class EnvSetupInstallMixin:


    def _on_install_requested(self, include_local_model: bool | None = None):


        if self.dock_widget is None:
            return



        if include_local_model is None:
            include_local_model = self._install_wants_local_model()
        self._install_includes_local_model = bool(include_local_model)







        if self._sam_pipe_busy():
            waited = getattr(self, "_install_pipe_waits", 0)
            if waited < install_pipe_wait_turns(_INSTALL_PIPE_WAIT_MAX):
                from functools import partial

                from qgis.PyQt.QtCore import QTimer as _QTimer

                from ...core.server_dials import dial_in_range
                self._install_pipe_waits = waited + 1


                retry_ms = dial_in_range(
                    "tuning.install.pipe_wait_retry_ms", 1500, 500, 5000)
                _QTimer.singleShot(retry_ms, partial(
                    self._on_install_requested, include_local_model))


                self.dock_widget.show_install_waiting_notice(
                    tr("Finishing the current AI task, then the install "
                       "starts."))
                return
            QgsMessageLog.logMessage(
                "Starting the install with the local AI pipe still busy: "
                "waited out the retry budget",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self._install_pipe_waits = 0






        from ...core.python_manager import is_sandboxed_linux
        if is_sandboxed_linux():
            QgsMessageLog.logMessage(
                "Manual install blocked: running inside a Flatpak/Snap sandbox",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_dependency_status(
                False,
                tr("Semi-Auto mode is not supported in this QGIS installation"))
            show_error_report(
                self.iface.mainWindow(),
                tr("Semi-Auto mode is not supported"),
                tr(
                    "Semi-Auto mode needs to install local dependencies, "
                    "which is not supported inside this sandboxed QGIS installation "
                    "(Flatpak or Snap). Please use Automatic mode instead, "
                    "which runs fully in the cloud and needs no local install."
                ),
                error_code="sandboxed_linux_manual_blocked",
            )
            return








        from ...core.model_config import MACOS_X86_NO_LOCAL_INFERENCE
        if MACOS_X86_NO_LOCAL_INFERENCE and include_local_model:
            QgsMessageLog.logMessage(
                "Manual install blocked: no local inference build for this Mac + Python",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_dependency_status(
                False,
                tr("Semi-Auto mode is not supported in this QGIS installation"))
            show_error_report(
                self.iface.mainWindow(),
                tr("Semi-Auto mode is not supported"),
                tr(
                    "Semi-Auto mode installs local components that are not "
                    "available for this Mac with this version of QGIS. Please use Automatic "
                    "mode instead, which runs fully in the cloud and needs no "
                    "local install."
                ),
                error_code="macos_x86_no_local_inference",
            )
            return


        if self.deps_install_worker is not None and self.deps_install_worker.isRunning():
            QgsMessageLog.logMessage(
                "Install already in progress, ignoring duplicate request",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            return





        remover = getattr(self, "_remove_data_worker", None)
        if remover is not None:
            try:
                running = remover.isRunning()
            except RuntimeError:
                running = False
            if running:
                QgsMessageLog.logMessage(
                    "Install refused: the downloaded AI data is being removed",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                self.dock_widget.set_dependency_status(
                    False, tr("Removing the downloaded AI data..."))
                return

        from ...core.venv_manager import get_venv_status, local_model_ready






        is_ready, message = get_venv_status(allow_subprocess_probe=False)







        model_ready, _model_msg = local_model_ready()
        if is_ready and not include_local_model:



            self.dock_widget.set_dependency_status(
                True, tr("Ready for Automatic mode"))
            self._refresh_activation_async()
            return
        if is_ready and model_ready:

            self.dock_widget.set_dependency_status(True, tr("AI ready"))
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

        self.dock_widget.set_install_progress(0, tr("Preparing installation..."))

        import time as _time
        self._install_t0 = _time.monotonic()
        self._install_attempt = _bump_install_attempt()
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_install_started(
                entry=self._install_entry_kind())
        except Exception:
            pass  # nosec B110







        predictor = getattr(self, "predictor", None)
        self.predictor = None
        self.deps_install_worker = DepsInstallWorker(
            predictor, include_local_model=include_local_model)
        self.deps_install_worker.progress.connect(self._on_deps_install_progress)
        self.deps_install_worker.done.connect(self._on_deps_install_finished)
        self.deps_install_worker.start()
        if not (self.deps_install_worker.isRunning() or self.deps_install_worker.isFinished()):





            self.predictor = predictor
            self._on_deps_install_finished(
                False, "the installation thread did not start")
            return


        from qgis.PyQt.QtCore import QTimer

        from ...core.server_dials import dial_in_range
        reveal_ms = dial_in_range(
            "tuning.install.signin_reveal_delay_ms", 2000, 500, 10000)
        QTimer.singleShot(reveal_ms, self._refresh_activation_async)

    def _on_deps_install_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return



        scaled = int(percent * 0.7)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_deps_install_finished(self, success: bool, message: str):
        if not self.dock_widget:
            return

        if success:




            self.dock_widget.set_install_progress(70, tr("Verifying installation..."))



            if self._verify_worker is not None and self._verify_worker.isRunning():
                return
            self._verify_worker = VerifyWorker(
                include_local_model=getattr(
                    self, "_install_includes_local_model", True))
            self._verify_worker.progress.connect(self._on_verify_progress)
            self._verify_worker.done.connect(self._on_verify_finished)
            self._verify_worker.start()
        else:




















            _cancelled = bool(
                getattr(self.deps_install_worker, "_cancelled", False))
            _msg_lower_early = (message or "").lower()
            if (_cancelled or "installation cancelled" in _msg_lower_early or "download cancelled" in _msg_lower_early):
                self.dock_widget.set_install_progress(
                    100, tr("Cancelled"), state="cancelled")
                self.dock_widget.set_dependency_status(
                    False, tr("Installation cancelled"))
                self._release_local_ai_install()





                self._load_predictor()
                return

            self.dock_widget.set_install_progress(
                100, tr("Failed"), state="failed")
            error_msg = message[:300] if message else tr(
                "Unknown error. Try again, or use Cloud AI instead.")
            self.dock_widget.set_dependency_status(False, tr("Installation failed"))

            self._release_local_ai_install()

            error_title = tr("Installation Failed")
            error_code = "installation_failed"
            msg_lower = message.lower() if message else ""








            from ...core.cache_paths import PLUGIN_CACHE_DIR
            from ...core.pip_diagnostics import (
                get_app_control_help,
                get_broken_python_runtime_help,
                get_corrupt_venv_help,
                get_crash_help,
                get_dependency_conflict_help,
                get_file_locked_help,
                get_glibc_too_old_help,
                get_invalid_path_help,
                get_macos_intel_help,
                get_pip_antivirus_help,
                get_ssl_error_help,
                get_vcpp_help,
                is_antivirus_error,
                is_app_control_error,
                is_broken_python_runtime,
                is_corrupt_venv,
                is_dependency_conflict,
                is_disk_full,
                is_dll_init_error,
                is_file_locked_error,
                is_glibc_too_old,
                is_index_forbidden_error,
                is_invalid_path_error,
                is_macos_intel_no_wheel,
                is_proxy_auth_error,
                is_rename_or_record_error,
                is_unable_to_create_process,
            )
            from ...core.venv_manager import mark_venv_for_rebuild
            if "another qgis window is installing" in msg_lower:





                self.dock_widget.set_dependency_status(
                    False, tr("Installation running in another window"))
                from ..dialogs.confirm_dialog import info_box
                info_box(
                    self.iface.mainWindow(),
                    tr("Installation Already Running"),
                    tr(
                        "Another QGIS window is installing the AI components. "
                        "Wait for it to finish, then try again."
                    ),
                )
                return
            if "not enough free disk space" in msg_lower or is_disk_full(msg_lower):


                error_title = tr("Not Enough Disk Space")
                error_code = "disk_space"
            elif is_broken_python_runtime(msg_lower):




                error_title = tr("AI Environment Damaged")
                error_msg = get_broken_python_runtime_help(PLUGIN_CACHE_DIR)
                error_code = "broken_python_runtime"
                mark_venv_for_rebuild()
            elif is_corrupt_venv(msg_lower):



                error_title = tr("AI Environment Damaged")
                error_msg = get_corrupt_venv_help()
                error_code = "corrupt_venv"
                mark_venv_for_rebuild()
            elif is_invalid_path_error(msg_lower):


                error_title = tr("Installation Path Problem")
                error_msg = get_invalid_path_help(PLUGIN_CACHE_DIR)
                error_code = "invalid_install_path"
            elif is_dependency_conflict(msg_lower):
                error_title = tr("Package Versions Conflict")
                error_msg = get_dependency_conflict_help()
                error_code = "dependency_conflict"
            elif any(p in msg_lower for p in [
                "ssl", "certificate verify", "sslerror",
                "unable to get local issuer",

                "invalid peer certificate", "unknownissuer",
            ]):


                error_title = tr("SSL Certificate Error")
                error_msg = get_ssl_error_help(msg_lower, PLUGIN_CACHE_DIR)
                error_code = "ssl_certificate_error"
            elif "file in use by qgis" in msg_lower or is_file_locked_error(msg_lower):



                error_title = tr("Restart QGIS Required")
                error_msg = get_file_locked_help()
                error_code = "restart_qgis_required"
            elif is_index_forbidden_error(msg_lower):



                error_title = tr("Downloads Blocked by Your Network")
                error_msg = tr(
                    "The package index refused the download (error 403).\n\n"
                    "This is usually a company or campus network filtering "
                    "downloads. Ask your IT administrator to allow "
                    "pypi.org and files.pythonhosted.org, or run the install "
                    "from another network.\n\n"
                    "Automatic (cloud) mode does not need this download."
                )
                error_code = "index_forbidden"
            elif is_app_control_error(msg_lower):



                error_title = tr("Blocked by IT Security Policy")
                error_msg = get_app_control_help(PLUGIN_CACHE_DIR)
                error_code = "app_control_policy"
            elif is_dll_init_error(msg_lower):



                error_title = tr("Missing System Component")
                error_msg = get_vcpp_help()
                error_code = "dll_init_error"
            elif is_antivirus_error(msg_lower):





                error_title = tr("Blocked by Antivirus or Security Software")
                error_msg = get_pip_antivirus_help(PLUGIN_CACHE_DIR)
                error_code = "antivirus_blocked"
            elif "cannot write to install" in msg_lower:



                error_title = tr("Installation Blocked")
                error_msg = f"{error_msg}\n\n{_get_change_path_instructions()}"
                error_code = "installation_blocked"
            elif is_proxy_auth_error(msg_lower):


                error_title = tr("Proxy Authentication Required")
                error_msg = "{}\n\n{}".format(
                    error_msg,
                    tr(
                        "Your network proxy requires a username and password. "
                        "Enter them in QGIS > Settings > Options > Network, "
                        "then restart QGIS and try again."
                    ),
                )
                error_code = "proxy_auth_required"
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
            elif is_unable_to_create_process(msg_lower):




                error_title = tr("Installation Failed")
                error_msg = "{}\n\n{}".format(
                    error_msg,
                    tr(
                        "The installer could not start a helper process (a "
                        "damaged Python launcher). Click Reinstall Dependencies "
                        "to rebuild the environment from scratch."
                    ),
                )
                error_code = "broken_pip_shim"
                mark_venv_for_rebuild()
            elif is_rename_or_record_error(msg_lower):


                error_title = tr("Restart QGIS Required")
                error_msg = get_crash_help(PLUGIN_CACHE_DIR)
                error_code = "dist_info_rename"
            elif is_glibc_too_old(msg_lower):

                error_title = tr("Linux System Too Old")
                error_msg = get_glibc_too_old_help()
                error_code = "glibc_too_old"
            elif is_macos_intel_no_wheel(msg_lower):

                error_title = tr("Unsupported Mac and Python Combination")
                error_msg = get_macos_intel_help()
                error_code = "macos_intel_no_wheel"





            elif "process crashed" in msg_lower:


                error_msg = get_crash_help(PLUGIN_CACHE_DIR)
                error_code = "process_crash"
            elif any(marker in msg_lower for marker in (
                "failed to create venv",
                "failed to bootstrap pip",
                "virtual environment does not exist",
                "virtual environment not found",
            )):


                error_title = tr("AI Environment Damaged")
                error_msg = get_corrupt_venv_help()
                error_code = "venv_create_failed"
                mark_venv_for_rebuild()
            elif "is broken" in msg_lower:


                error_title = tr("AI Environment Damaged")
                error_msg = get_corrupt_venv_help()
                error_code = "package_broken"
                mark_venv_for_rebuild()

            show_error_report(
                self.iface.mainWindow(),
                error_title,
                error_msg,
                error_code=error_code,
            )
            try:
                import time as _time

                from ...core import telemetry_session_events
                t0 = getattr(self, "_install_t0", 0.0)
                telemetry_session_events.track_install_failed(
                    error_class=error_code,
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None,
                    python_minor=sys.version_info.minor,
                    retry_count=getattr(self, "_install_attempt", 0),
                    detail=message,
                    entry=self._install_entry_kind(),
                )
            except Exception:
                pass  # nosec B110

    def _on_verify_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return

        scaled = 70 + int(percent * 0.1)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_verify_finished(self, is_valid: bool, message: str):
        if not self.dock_widget:
            return
        if is_valid:





            model_ok = True
            try:
                from ...core.venv_manager import local_model_ready
                model_ok, _why = local_model_ready()
            except Exception:  # noqa: BLE001
                pass  # nosec B110



            try:
                import time as _time

                from ...core import telemetry_session_events
                t0 = getattr(self, "_install_t0", 0.0)
                if t0:
                    telemetry_session_events.track_install_completed(
                        duration_ms=int((_time.monotonic() - t0) * 1000),
                        python_minor=sys.version_info.minor,
                        retry_count=getattr(self, "_install_attempt", 0),
                        entry=self._install_entry_kind(),
                        local_model_ready=model_ok,
                    )
                    self._install_t0 = 0.0
                _clear_install_attempts()
            except Exception:
                pass  # nosec B110





            if model_ok:
                self.dock_widget.set_dependency_status(
                    True, tr("AI ready"))
            else:
                self.dock_widget.set_dependency_status(
                    True, tr("Ready for Automatic mode"))




                if getattr(self, "_install_includes_local_model", True):
                    try:
                        self.iface.messageBar().pushWarning(
                            "AI Segmentation",
                            tr("Automatic mode is ready. The on-device AI could "
                               "not be installed, so Semi-Auto mode and the "
                               "AI fix are off until it is. Everything else "
                               "works."),
                        )
                    except (RuntimeError, AttributeError):
                        pass
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
                self.dock_widget.set_install_progress(
                    100, tr("Failed"), state="failed")


                self.dock_widget.set_dependency_status(
                    False, tr("Almost ready: the AI file did not download."))
                self._release_local_ai_install()
        else:



            cancelled = bool(getattr(self._verify_worker, "_cancelled", False))
            if cancelled or "installation cancelled" in (message or "").lower():
                self.dock_widget.set_install_progress(
                    100, tr("Cancelled"), state="cancelled")
                self.dock_widget.set_dependency_status(
                    False, tr("Installation cancelled"))
                self._release_local_ai_install()
                return
            self.dock_widget.set_install_progress(
                100, tr("Failed"), state="failed")
            self.dock_widget.set_dependency_status(
                False, "{} {}".format(tr("Verification failed:"), message))
            self._release_local_ai_install()



            from ...core.pip_diagnostics import (
                get_app_control_help,
                get_vcpp_help,
                is_app_control_error,
                is_dll_init_error,
            )
            error_title = tr("Verification Failed")
            error_code = "verification_failed"
            body = "{}\n{}".format(
                tr("The AI was set up but could not start."),
                message)
            low = (message or "").lower()
            if is_app_control_error(low):



                from ...core.cache_paths import PLUGIN_CACHE_DIR
                error_title = tr("Blocked by IT Security Policy")
                error_code = "app_control_policy"
                body = f"{message}\n\n{get_app_control_help(PLUGIN_CACHE_DIR)}"
            elif is_dll_init_error(low) or "required dll failed to initialize" in low:
                error_title = tr("A Component Failed to Load")
                error_code = "dll_init_failed"
                body = f"{message}\n\n{get_vcpp_help()}"
            show_error_report(
                self.iface.mainWindow(),
                error_title,
                body,
                error_code=error_code)
            try:
                import time as _time

                from ...core import telemetry_session_events
                t0 = getattr(self, "_install_t0", 0.0)
                telemetry_session_events.track_install_failed(
                    error_class=error_code,
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None,
                    python_minor=sys.version_info.minor,
                    retry_count=getattr(self, "_install_attempt", 0),
                    detail=message,
                    entry=self._install_entry_kind(),
                )
            except Exception:
                pass  # nosec B110

    def _on_cancel_install(self):



        if self.download_worker is not None and self.download_worker.isRunning():
            self._download_cancelled = True
            try:
                from ...core.checkpoint_manager import request_download_cancel
                request_download_cancel()
            except Exception:
                pass  # nosec B110
            QgsMessageLog.logMessage(
                "Model download cancelled by user",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )



        verify_worker = getattr(self, "_verify_worker", None)
        if verify_worker is not None:
            try:
                if verify_worker.isRunning():
                    verify_worker.cancel()
                    QgsMessageLog.logMessage(
                        "Verification cancelled by user",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning)
            except (RuntimeError, AttributeError):
                pass  # nosec B110
        if self.deps_install_worker and self.deps_install_worker.isRunning():
            self.deps_install_worker.cancel()
            try:
                import time as _time

                from ...core import telemetry_session_events
                t0 = getattr(self, "_install_t0", 0.0)
                telemetry_session_events.track_install_cancelled(
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None,
                    entry=self._install_entry_kind())
            except Exception:
                pass  # nosec B110
            QgsMessageLog.logMessage(
                "Installation cancelled by user",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    def _auto_download_checkpoint(self):




        if self.download_worker is not None and self.download_worker.isRunning():
            return







        from ...core.venv_manager import local_model_ready
        model_ready, _model_msg = local_model_ready()
        if not model_ready:
            QgsMessageLog.logMessage(
                "Skipping the AI model download: the packages that load it are "
                "not installed. Automatic mode is unaffected.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_dependency_status(
                True, tr("Automatic mode ready"))



            self._release_local_ai_install()
            self._refresh_activation_async()
            return




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
        self._download_cancelled = False
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
            self.dock_widget.set_install_progress(
                100, tr("Failed"), state="failed")


            self.dock_widget.set_dependency_status(
                False, tr("Almost ready: the AI file did not download."))
            self._release_local_ai_install()

    def _on_download_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return

        scaled = 80 + int(percent * 0.2)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_download_finished(self, success: bool, message: str):
        if not self.dock_widget:
            return
        if success:
            try:
                import time as _time

                from ...core import telemetry_session_events
                from ...core.model_config import USE_SAM2
                t0 = getattr(self, "_model_download_t0", 0.0)
                telemetry_session_events.track_model_download_completed(
                    model="sam2" if USE_SAM2 else "sam1",
                    duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None)
            except Exception:
                pass  # nosec B110
            self.dock_widget.set_install_progress(95, tr("Loading AI model..."))
            self._load_predictor()
            self._refresh_activation_async()
        else:



            if getattr(self, "_download_cancelled", False):
                self._download_cancelled = False
                self.dock_widget.set_install_progress(
                    100, tr("Cancelled"), state="cancelled")


                self.dock_widget.set_dependency_status(
                    False, tr("Almost ready: the AI file is still missing."))
                self._release_local_ai_install()
                return

            self.dock_widget.set_install_progress(
                100, tr("Failed"), state="failed")


            self.dock_widget.set_dependency_status(
                False, tr("Almost ready: the AI file did not download."))
            self._release_local_ai_install()

            show_error_report(
                self.iface.mainWindow(),
                tr("Download Failed"),
                "{}\n{}".format(
                    tr("Failed to download model:"),
                    message),
                error_code="download_failed",
            )

    def _recover_corrupt_checkpoint(self, deleted: bool) -> bool:








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

        from ..dialogs.confirm_dialog import info_box
        info_box(
            self.iface.mainWindow(), tr("Re-downloading Model"), msg)
        try:
            self._auto_download_checkpoint()
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to re-download checkpoint after corruption: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False

    def _recover_broken_venv(self, error: str) -> bool:









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

        from ..dialogs.confirm_dialog import info_box
        info_box(
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
