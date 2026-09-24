






from __future__ import annotations

import time

from qgis.PyQt.QtCore import QThread, pyqtSignal

from ..core.i18n import tr
from ..core.server_dials import dial_in_range





_PREDICTOR_SHUTDOWN_TIMEOUT_S = 8


def _predictor_shutdown_timeout_s() -> float:
    return dial_in_range(
        "tuning.install.predictor_shutdown_timeout_s", _PREDICTOR_SHUTDOWN_TIMEOUT_S, 5, 60)




def _log_worker_warning(message: str) -> None:
    from qgis.core import Qgis, QgsMessageLog
    QgsMessageLog.logMessage(message, "AI Segmentation",
                             level=Qgis.MessageLevel.Warning)


class DepsInstallWorker(QThread):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(bool, str)

    def __init__(self, predictor=None, parent=None, include_local_model: bool = True):
        super().__init__(parent)
        self._cancelled = False
        self._predictor = predictor


        self.include_local_model = bool(include_local_model)

    def cancel(self):
        self._cancelled = True

    def _shutdown_predictor(self) -> bool:











        predictor = self._predictor
        self._predictor = None
        if predictor is None:
            return True
        self.progress.emit(0, tr("Stopping the local AI..."))
        try:
            import threading
            thread = threading.Thread(target=predictor.cleanup, daemon=True)
            thread.start()
            thread.join(timeout=_predictor_shutdown_timeout_s())
            if thread.is_alive():



                _log_worker_warning(
                    "Model shutdown did not finish in time; the install was "
                    "stopped rather than delete files still held open")
                return False
        except Exception as err:  # noqa: BLE001



            _log_worker_warning(f"Model shutdown failed: {type(err).__name__}")
            return False
        return True

    def run(self):
        from ..core.power_inhibit import begin_keep_awake, end_keep_awake



        activity = begin_keep_awake("AI Segmentation dependency install")
        try:
            if not self._shutdown_predictor():
                self.done.emit(False, tr(
                    "The local AI did not stop in time, so the install was "
                    "not started. Close and reopen QGIS, then try again."))
                return
            from ..core.venv_manager import create_venv_and_install
            success, message = create_venv_and_install(
                progress_callback=lambda percent, msg: self.progress.emit(percent, msg),
                cancel_check=lambda: self._cancelled,
                include_local_model=self.include_local_model,
            )
            self.done.emit(success, message)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.done.emit(False, error_msg)
        finally:
            end_keep_awake(activity)


class DownloadWorker(QThread):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        from ..core.power_inhibit import begin_keep_awake, end_keep_awake



        activity = begin_keep_awake("AI Segmentation model download")
        try:
            from ..core.checkpoint_manager import download_checkpoint
            success, message = download_checkpoint(
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )
            self.done.emit(success, message)
        except Exception as e:





            self.done.emit(False, str(e) or type(e).__name__)
        finally:
            end_keep_awake(activity)


class SetImageWorker(QThread):














    done = pyqtSignal(int, bool, str)

    def __init__(self, predictor, image_np, generation: int, parent=None):
        super().__init__(parent)
        self._predictor = predictor
        self._image_np = image_np
        self._generation = generation


        self.encode_s = None

    def run(self):
        try:


            started = time.perf_counter()
            self._predictor.set_image(self._image_np)
            self.encode_s = time.perf_counter() - started
            self.done.emit(self._generation, True, "")
        except Exception as e:






            self.done.emit(self._generation, False, str(e) or type(e).__name__)


class PredictorLoadWorker(QThread):

    done = pyqtSignal(object, str)

    def run(self):
        try:
            from ..core.checkpoint_manager import get_checkpoint_path
            from ..core.sam_predictor import SamPredictor, build_sam_predictor_config
            sam_config = build_sam_predictor_config(checkpoint=get_checkpoint_path())
            predictor = SamPredictor(sam_config)
            self.done.emit(predictor, "")
        except Exception as e:
            import traceback
            self.done.emit(None, f"{e}\n{traceback.format_exc()}")


class StartupCheckWorker(QThread):






    done = pyqtSignal(bool, str, bool)

    def run(self):
        try:
            from ..core.checkpoint_manager import cleanup_legacy_sam1_data
            cleanup_legacy_sam1_data()
        except Exception:

            pass  # nosec B110

        try:


            from ..core.cache_paths import PLUGIN_CACHE_DIR
            from ..core.install_temp_sweep import sweep_stale_install_temp_files
            sweep_stale_install_temp_files(PLUGIN_CACHE_DIR)
        except Exception:
            pass  # nosec B110

        try:
            from ..core.venv_manager import cleanup_old_libs, get_venv_status
            cleanup_old_libs()
            venv_ready, message = get_venv_status()
        except Exception as e:
            import traceback



            try:
                from ..core.telemetry_errors import report_exception
                report_exception(e, stage="install", module="background_workers")
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            self.done.emit(
                False, f"startup_error: {e}\n{traceback.format_exc()}", False)
            return

        checkpoint_ok = False
        if venv_ready:
            try:
                from ..core.checkpoint_manager import checkpoint_exists
                checkpoint_ok = checkpoint_exists()
            except Exception:
                checkpoint_ok = False
        self.done.emit(venv_ready, message, checkpoint_ok)


class DeviceInfoWorker(QThread):





    done = pyqtSignal(bool, str)

    def run(self):
        try:
            from ..core.venv_manager import ensure_venv_packages_available
            ensure_venv_packages_available()
            from ..core.device_manager import get_device_info
            info = get_device_info()






            try:
                import rasterio  # noqa: F401
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            self.done.emit(True, str(info or ""))
        except RuntimeError as e:
            self.done.emit(False, str(e))
        except Exception as e:
            self.done.emit(True, f"device_info_unavailable: {e}")


class RemoveAiDataWorker(QThread):











    progress = pyqtSignal(str)

    done = pyqtSignal(bool, str, str)

    def __init__(self, predictor=None, parent=None):
        super().__init__(parent)
        self._cancelled = False
        self._predictor = predictor

    def cancel(self):
        self._cancelled = True

    def run(self):
        import os

        predictor = self._predictor
        self._predictor = None
        if predictor is not None:
            self.progress.emit(tr("Stopping the local AI..."))
            try:
                import threading
                thread = threading.Thread(target=predictor.cleanup, daemon=True)
                thread.start()
                thread.join(timeout=_predictor_shutdown_timeout_s())
                if thread.is_alive():





                    _log_worker_warning(
                        "Model shutdown did not finish in time; the removal was "
                        "stopped rather than delete files still held open")


                    self.done.emit(
                        False, "", "the local AI did not stop; nothing deleted")
                    return
            except Exception:  # noqa: BLE001
                pass  # nosec B110

        freed = ""
        try:
            from ..core.cache_paths import PLUGIN_CACHE_DIR
            from ..core.venv_manager import purge_cache_dir
            from .plugin.shared import dir_size_label
            if not os.path.isdir(PLUGIN_CACHE_DIR):
                self.done.emit(True, "", "")
                return
            self.progress.emit(tr("Measuring the downloaded data..."))
            freed = dir_size_label(PLUGIN_CACHE_DIR)
            self.progress.emit(tr("Deleting the downloaded data..."))
            nothing_left = purge_cache_dir(cancel_check=lambda: self._cancelled)
        except Exception as e:  # noqa: BLE001
            self.done.emit(False, freed, str(e)[:80])
            return
        if self._cancelled:




            self.done.emit(
                False, "", "the removal was stopped; the data is partly deleted")
            return
        self.done.emit(bool(nothing_left), freed, "")


class VerifyWorker(QThread):







    done = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)

    def __init__(self, include_local_model: bool = True, parent=None):
        super().__init__(parent)
        self.include_local_model = bool(include_local_model)
        self._cancelled = False

    def cancel(self):

        self._cancelled = True

    def run(self):
        try:
            from ..core.venv_manager import verify_venv
            is_valid, msg = verify_venv(
                progress_callback=lambda pct, m: self.progress.emit(pct, m),
                include_local_model=self.include_local_model,
                cancel_check=lambda: self._cancelled)
            if not is_valid:
                self.done.emit(False, msg)
                return
            if self._cancelled:
                self.done.emit(False, "Installation cancelled")
                return
            if not self.include_local_model:
                self.done.emit(True, "")
                return
            self.progress.emit(100, tr("Detecting device..."))
            try:
                from ..core.venv_manager import ensure_venv_packages_available
                ensure_venv_packages_available()
                from ..core.device_manager import get_device_info
                info = get_device_info()
                self.done.emit(True, info or "")
            except Exception as e:
                self.done.emit(True, f"device_error: {str(e)}")
        except Exception as e:
            self.done.emit(False, str(e))
