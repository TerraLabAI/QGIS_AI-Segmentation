








from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

from ...core.i18n import tr


def _notify_ui(callback, *args) -> None:





    if callback is None:
        return
    try:
        callback(*args)
    except RuntimeError:
        pass  # nosec B110


def _drop_untagged_account_history() -> None:








    try:
        from ...core.presets.run_history_cache import (
            clear_run_history_cache,
            reset_account_fingerprint_cache,
        )
        from ...core.presets.segment_history import clear_unscoped_recent_objects



        reset_account_fingerprint_cache()
        clear_run_history_cache()
        clear_unscoped_recent_objects()
    except Exception:
        pass  # nosec B110


class EnvSetupAccountMixin:




    def _auto_run_in_flight(self) -> bool:






        if getattr(self, "_auto_worker", None) is not None:
            return True
        try:
            return bool(getattr(self.dock_widget, "_auto_run_active", False))
        except (RuntimeError, AttributeError):
            return False

    def _show_sign_in_page(self) -> None:






        dock = self.dock_widget
        if dock is None:
            return
        try:
            dock.setVisible(True)
            dock.raise_()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        from ...core.server_dials import dial_copy

        try:
            dock.set_activated_state(False)
            dock.set_activation_message(
                dial_copy(
                    "install.session_expired_notice",
                    tr("Session expired. Sign in again to continue."),
                ),
                True)
        except (RuntimeError, AttributeError):
            try:
                self.iface.messageBar().pushWarning(
                    "AI Segmentation",
                    tr("Session expired. Open the AI Segmentation panel and "
                       "sign in again."))
            except (RuntimeError, AttributeError):
                pass  # nosec B110

    def _on_settings_clicked(self):
        self.open_settings_page("")

    def open_settings_page(self, page: str = "") -> bool:







        from ...core.activation_manager import get_auth_header, get_auth_token, is_plugin_activated
        if not is_plugin_activated():
            if not page:
                self._show_sign_in_page()
            return False
        from ...api.terralab_client import TerraLabClient
        from ..account_settings_dialog import AccountSettingsDialog
        existing = getattr(self, "_account_dialog", None)
        if existing is not None:
            try:
                if page:
                    existing.show_settings_page(page)
                existing.raise_()
                existing.activateWindow()
                return True
            except (RuntimeError, AttributeError):
                self._account_dialog = None
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
        dlg.account_deleted.connect(self._on_account_deleted)


        dlg.usage_loaded.connect(self._on_account_usage_loaded)
        if page:
            dlg.show_settings_page(page)
        if self._auto_run_in_flight():



            self._account_dialog = dlg
            dlg.setModal(False)
            dlg.finished.connect(lambda _r: setattr(self, "_account_dialog", None))
            dlg.finished.connect(dlg.deleteLater)
            dlg.show()
            dlg.raise_()
            return True
        dlg.exec()






        dlg.deleteLater()
        return True

    def is_local_ai_busy(self) -> bool:






        try:
            for attr in (
                "deps_install_worker", "download_worker", "_verify_worker",
                "_predictor_worker", "_startup_check_worker", "_auto_worker",


                "_device_info_worker",


                "_manual_encode_worker",


                "_remove_data_worker",
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

    def remove_local_ai_data(self, on_progress=None, on_finished=None) -> tuple[bool, str]:


















        worker = getattr(self, "_remove_data_worker", None)
        if worker is not None:
            try:
                if worker.isRunning():
                    return False, tr("The removal is already running.")
            except RuntimeError:
                pass
        if self.is_local_ai_busy():
            return False, tr(
                "An install or detection is still running. Wait for it to "
                "finish, then try again.")





        from ...core.install_lock import InstallLock
        from ...core.venv_manager import INSTALL_LOCK_FILE

        lock = InstallLock(INSTALL_LOCK_FILE)
        if not lock.acquire():
            return False, tr(
                "Another QGIS window is installing the AI engine. Wait for it "
                "to finish, then try again.")

        errors = self._clear_local_ai_account_state()

        predictor = getattr(self, "predictor", None)
        self.predictor = None
        try:
            from ..background_workers import RemoveAiDataWorker
            from .shared import park_orphaned_worker

            worker = RemoveAiDataWorker(predictor)
            self._remove_data_worker = worker
            if on_progress is not None:
                worker.progress.connect(
                    lambda text, cb=on_progress: _notify_ui(cb, text))
            worker.done.connect(
                lambda nothing_left, freed, error, lk=lock, errs=errors, cb=on_finished:
                self._on_remove_ai_data_done(nothing_left, freed, error, lk, errs, cb))



            park_orphaned_worker(worker)
            worker.start()



            if not (worker.isRunning() or worker.isFinished()):
                raise RuntimeError("the removal thread did not start")
        except Exception as err:  # noqa: BLE001




            QgsMessageLog.logMessage(
                f"Could not start the AI data removal: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical)
            self._remove_data_worker = None
            self.predictor = predictor
            try:
                lock.release()
            except Exception:  # nosec B110
                pass
            from ...core.server_dials import dial_copy

            return False, dial_copy(
                "install.removal_start_failed_notice",
                tr("The removal could not start. You are signed out, but the "
                   "downloaded AI data is still on this computer. Try again."))
        return True, tr("Removing the downloaded AI data...")

    def _clear_local_ai_account_state(self) -> list:





        self._env_ready = False
        self._first_time_setup_done = False




        errors: list[str] = []
        try:
            from ...core.activation_manager import clear_auth
            clear_auth()
        except Exception as err:  # nosec B110
            errors.append(str(err)[:80])
        _drop_untagged_account_history()
        self._last_key_validation_unix = 0.0

        self._end_cloud_click_session()




        try:
            from qgis.PyQt.QtCore import QSettings
            settings = QSettings()
            for group in ("AISegmentation", "AI_Segmentation",
                          "TerraLab/AI_Segmentation"):
                settings.remove(group)
            settings.sync()
        except Exception as err:  # nosec B110
            errors.append(str(err)[:80])

        if self.dock_widget:
            try:
                self.dock_widget.set_activated_state(False)
            except (RuntimeError, AttributeError):
                pass
        return errors

    def _on_remove_ai_data_done(self, nothing_left, freed, error,
                                lock, errors, on_finished) -> None:


        self._remove_data_worker = None
        errors = list(errors or [])
        if error:
            errors.append(str(error)[:80])
        elif not nothing_left:


            errors.append("leftover files")

        lock.release()



        import os

        from ...core.cache_paths import PLUGIN_CACHE_DIR
        try:
            os.rmdir(PLUGIN_CACHE_DIR)
        except OSError:
            pass  # nosec B110

        QgsMessageLog.logMessage(
            "Local AI data removed ({}), errors={}".format(
                freed or "0 MB", len(errors)),
            "AI Segmentation", level=Qgis.MessageLevel.Info)

        from ...core.server_dials import dial_copy

        if errors:
            message = dial_copy(
                "install.removal_partial_notice",
                tr("AI data removed, but some items could not be fully cleared. "
                   "You can delete the folder manually."))
        else:
            message = dial_copy(
                "install.removal_done_notice",
                tr("Downloaded AI data removed. You have been signed out."))
        _notify_ui(on_finished, True, message)

    def _on_sign_out_requested(self):
        from ...core.activation_manager import clear_auth
        clear_auth()
        _drop_untagged_account_history()
        self._last_key_validation_unix = 0.0



        self._cancel_task("_usage_fetch_task")
        self._key_revalidate_pending = False

        self._end_cloud_click_session()

        self._end_manual_credit_session()

        self._refresh_config_for_account_change()




        self._last_usage = {}
        self._usage_applied_at = None
        self._billing_warning_shown = False
        self._plan_upgrade_announced = False
        if self.dock_widget:
            try:
                self.dock_widget._low_credit_note_seen = False
            except (RuntimeError, AttributeError):
                pass
            self.dock_widget.set_activated_state(False)

    def _on_account_deleted(self):











        try:
            from ...core.detection_history import clear_detection_history
            clear_detection_history()
        except Exception:
            pass  # nosec B110
        try:
            from ...core.presets.segment_history import (
                clear_recent_objects_for_account,
            )
            clear_recent_objects_for_account()
        except Exception:
            pass  # nosec B110


        self._on_sign_out_requested()
