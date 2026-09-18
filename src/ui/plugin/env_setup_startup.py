








from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

from ...core.i18n import tr
from ...core.interaction_dials import (
    vcredist_url,
    warm_recent_manual_days,
)
from ..error_report_dialog import show_error_report
from .shared import SETTINGS_KEY_LAST_MANUAL_SESSION_TS




_VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


class EnvSetupStartupMixin:


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



                self.dock_widget.close()
                QgsMessageLog.logMessage(
                    "Dock hidden", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _do_first_time_setup(self):
        QgsMessageLog.logMessage(
            "Panel opened - checking dependencies...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )





        self._warm_local_ai_for_manual()




        if self._startup_check_worker is not None and self._startup_check_worker.isRunning():
            return
        from ..background_workers import StartupCheckWorker
        self._startup_check_worker = StartupCheckWorker()
        self._startup_check_worker.done.connect(self._on_startup_check_finished)
        self._startup_check_worker.start()



        from qgis.PyQt.QtCore import QTimer

        from ...core.server_dials import dial_in_range
        self._update_check_delays = [
            dial_in_range("tuning.install.update_check_delay1_ms", 5000, 2000, 30000),
            dial_in_range("tuning.install.update_check_delay2_ms", 30000, 10000, 120000),
            dial_in_range("tuning.install.update_check_delay3_ms", 60000, 20000, 300000),
            dial_in_range("tuning.install.update_check_delay4_ms", 120000, 30000, 600000),
        ]
        self._update_check_index = 0
        QTimer.singleShot(
            self._update_check_delays[0], self._check_for_plugin_update)

    def _on_startup_check_finished(self, venv_ready: bool, message: str, checkpoint_ok: bool):
        try:
            self._apply_startup_check(venv_ready, message, checkpoint_ok)
        finally:



            self._refresh_activation_async()

    def _apply_startup_check(self, venv_ready: bool, message: str, checkpoint_ok: bool):

        self._env_ready = bool(venv_ready)
        if not self.dock_widget:
            return

        if message.startswith("startup_error:"):
            detail = message[len("startup_error:"):].strip()
            QgsMessageLog.logMessage(
                f"Dependency check error: {detail}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)



            self.dock_widget.set_dependency_status(
                False, tr("Could not check the AI components. See the log for details."))



            self._interactive_setup_done = False
            return

        if venv_ready:
            self.dock_widget.set_dependency_status(True, tr("AI ready"))
            QgsMessageLog.logMessage(
                "✓ Virtual environment verified successfully",
                "AI Segmentation",
                level=Qgis.MessageLevel.Success
            )
            self._start_device_info_worker()
            if checkpoint_ok:
                self.dock_widget.set_checkpoint_status(True, tr("AI ready"))
                self._load_predictor()
            else:







                model_ok = True
                try:
                    from ...core.venv_manager import local_model_ready
                    model_ok, _why = local_model_ready()
                except Exception:  # noqa: BLE001
                    pass  # nosec B110
                if not model_ok:
                    self.dock_widget.set_dependency_status(
                        False, "Local model packages are not installed")
                    return









                from ..ai_segmentation_dockwidget import Mode
                interactive = self.dock_widget._mode == Mode.INTERACTIVE
                if interactive:


                    self.dock_widget._setup_section_wanted = True
                self.dock_widget.set_dependency_status(
                    True, tr("Almost ready: the AI file is still missing."),
                    mark=False)



                if self._pending_refine_import or getattr(self, "_refine_install_pending", False):
                    self._auto_download_checkpoint()
                    return
                if interactive:
                    self.dock_widget.install_button.setVisible(True)
                    self.dock_widget.install_button.setEnabled(True)
                    self.dock_widget.install_button.setText(tr("Download AI model"))
        else:
            self.dock_widget.set_dependency_status(False, message)
            QgsMessageLog.logMessage(
                f"Virtual environment status: {message}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )


            from ..dock.setup_status_text import STATUS_NEEDS_UPDATE, setup_status_code
            if setup_status_code(message) == STATUS_NEEDS_UPDATE:
                self._on_install_requested()

    def _check_for_plugin_update(self):

        if not self.dock_widget:
            return
        self.dock_widget.check_for_updates()


        if not self.dock_widget.is_update_offer_shown() and hasattr(self, "_update_check_delays"):
            self._update_check_index += 1
            if self._update_check_index < len(self._update_check_delays):
                from qgis.PyQt.QtCore import QTimer
                delay = self._update_check_delays[self._update_check_index]
                QTimer.singleShot(delay, self._check_for_plugin_update)

    def _load_predictor(self):





        if self._cloud_correct_predictor_active():
            return


        worker = getattr(self, "_predictor_worker", None)
        if worker is not None and worker.isRunning():
            return















        try:
            from ...core.venv_manager import local_model_ready
        except ImportError:
            local_model_ready = None
        if local_model_ready is not None:
            try:
                model_ok, why = local_model_ready()
            except Exception as exc:  # noqa: BLE001
                model_ok, why = False, f"cannot read the environment: {exc}"
        else:
            model_ok, why = True, ""
        if not model_ok:









            self._on_predictor_loaded(None, why)
            return
        from ..background_workers import PredictorLoadWorker
        if self.dock_widget:
            self.dock_widget.set_checkpoint_status(True, tr("Loading AI model..."))
        self._predictor_worker = PredictorLoadWorker()
        self._predictor_worker.done.connect(self._on_predictor_loaded)
        self._predictor_worker.start()

    def _abandon_local_ai_session(self, err_msg: str, from_install: bool = False) -> None:














        in_session = bool(getattr(self, "_refine_handoff_active", False)
                          or getattr(self, "_pending_refine_import", False))
        if not (in_session or from_install):
            return
        if in_session:




            if getattr(self, "_refine_add_mode_active", False):
                try:
                    self._exit_ai_add_mode()
                except (RuntimeError, AttributeError):
                    pass



            for step in (self._collect_manual_refine_into_review,
                         self._restore_auto_review_after_handoff):
                try:
                    step()
                except (RuntimeError, AttributeError):
                    pass
            try:
                self.dock_widget.leave_ai_reshape_state()
            except (RuntimeError, AttributeError):
                pass
            try:
                self._arm_correct_select()
            except (RuntimeError, AttributeError):
                pass
        try:
            from ...core.server_dials import dial_copy

            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                dial_copy(
                    "install.local_ai_unavailable_notice",
                    tr("The on-device AI could not start, so the AI fix is off. "
                       "Your detections are safe: switch the fix method to Manual "
                       "to keep correcting, or save them as they are."),
                ),
            )
        except (RuntimeError, AttributeError):
            pass
        QgsMessageLog.logMessage(
            f"Local AI unavailable for this session: {err_msg}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        try:
            from ...core.telemetry_errors import report_exception
            report_exception(
                RuntimeError(err_msg or "predictor load failed"),
                stage="local_ai_load", module="env_setup")
        except Exception:  # noqa: BLE001
            pass  # nosec B110

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


            self._local_ai_load_failed = True





            was_install = self._local_ai_install_pending()


            self._release_local_ai_install()
            self._abandon_local_ai_session(err_msg, from_install=was_install)
            return
        self._local_ai_load_failed = False
        if self._cloud_correct_predictor_active():




            self._local_predictor_held = predictor
        else:
            self.predictor = predictor
        QgsMessageLog.logMessage(
            "SAM predictor initialized (subprocess mode)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        if self.dock_widget:
            self.dock_widget.set_checkpoint_status(True, tr("AI ready"))
            self.dock_widget.set_install_progress(100, tr("Ready"))












        try:
            if not self._headless:
                should_warm = getattr(self, "_warm_predictor_on_ready", False)
                should_warm = should_warm or self._manual_used_recently()
                if should_warm:
                    self._warm_predictor_on_ready = False
                    predictor.warm_up()
                    QgsMessageLog.logMessage(
                        "Pre-warming the SAM worker (Manual use predicted)",
                        "AI Segmentation", level=Qgis.MessageLevel.Info
                    )
        except Exception:  # noqa: BLE001
            pass  # nosec B110






        if self._local_ai_install_pending():
            resume_add = bool(getattr(self, "_ai_add_install_pending", False))
            self._release_local_ai_install()
            if self._auto_review:
                if resume_add:
                    self._on_ai_add_requested()
                else:
                    self._on_reshape_ai_requested()
            return



        if self._pending_refine_import and self._refine_handoff_active:
            self._pending_refine_import = False
            layer = getattr(self, "_handoff_source_layer", None)
            review = self._auto_review
            if layer is not None and review:
                if self.dock_widget:
                    try:
                        combo = self.dock_widget.layer_combo
                        combo.blockSignals(True)
                        combo.setLayer(layer)
                        combo.blockSignals(False)
                    except (RuntimeError, AttributeError):
                        pass
                self._on_start_segmentation(layer)




                self._seed_refine_from_review()
                self._import_review_geoms_as_saved(review)


                self._open_reshape_target()
            return



        self._replay_hover_warm_when_ready()

    def _manual_used_recently(self, days: int = 14) -> bool:






        try:
            import time

            from qgis.PyQt.QtCore import QSettings
            ts = QSettings().value(
                SETTINGS_KEY_LAST_MANUAL_SESSION_TS, 0, type=int)
            window_days = warm_recent_manual_days(days)
            return ts > 0 and (time.time() - ts) < window_days * 86400
        except Exception:  # noqa: BLE001
            return False

    def _start_device_info_worker(self):

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


        if "DLL" in info or "shm.dll" in info:
            QgsMessageLog.logMessage(
                f"PyTorch DLL error: {info}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            if self._headless:
                return


            from ...core.server_dials import dial_copy

            show_error_report(
                self.iface.mainWindow(),
                dial_copy(
                    "install.dll_error_title",
                    tr("The AI engine cannot load on Windows"),
                ),
                dial_copy(
                    "install.dll_error_body",
                    tr("The plugin requires Visual C++ Redistributables to run the "
                       "local AI engine.\n\n"
                       "Please download and install:\n"
                       "https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
                       "After installation, restart QGIS and try again."),
                ).replace(_VCREDIST_URL, vcredist_url(_VCREDIST_URL)),
                error_code="pytorch_dll_error",
            )
        else:
            QgsMessageLog.logMessage(
                f"Could not determine device info: {info}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    def _sam_pipe_busy(self) -> bool:







        for attr in ("_manual_encode_worker", "_predictor_worker"):
            worker = getattr(self, attr, None)
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    return True
            except (RuntimeError, AttributeError):
                continue
        return False

    def _install_wants_local_model(self) -> bool:














        for flag in ("_refine_install_pending", "_ai_add_install_pending",
                     "_refine_handoff_active", "_pending_refine_import"):
            if getattr(self, flag, False):
                return True
        try:
            from ..ai_segmentation_dockwidget import Mode
            return self.dock_widget._mode != Mode.AUTOMATIC
        except (RuntimeError, AttributeError):
            return True

    def _install_entry_kind(self) -> str:






        return ("semi_auto_modal"
                if getattr(self, "_install_includes_local_model", False)
                else "background")
