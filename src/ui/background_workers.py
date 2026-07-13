"""Background QThread workers for dependency install, model download, and verification.

Completion is reported via a `done` signal, deliberately NOT named `finished`:
QThread already owns a built-in `finished` signal that fires when the OS thread
actually returns. Shadowing it would hide that lifecycle signal, which unload()
relies on to safely park a still-running worker (see park_orphaned_worker).
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QThread, pyqtSignal

from ..core.i18n import tr


class DepsInstallWorker(QThread):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from ..core.power_inhibit import begin_activity, end_activity
        # A multi-GB torch install over a slow link can be interrupted by
        # system sleep if the user walks away; hold a keep-awake activity for
        # the whole install. Best-effort, always released.
        activity = begin_activity("AI Segmentation dependency install")
        try:
            from ..core.venv_manager import create_venv_and_install
            success, message = create_venv_and_install(
                progress_callback=lambda percent, msg: self.progress.emit(percent, msg),
                cancel_check=lambda: self._cancelled,
            )
            self.done.emit(success, message)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.done.emit(False, error_msg)
        finally:
            end_activity(activity)


class DownloadWorker(QThread):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            from ..core.checkpoint_manager import download_checkpoint
            success, message = download_checkpoint(
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )
            self.done.emit(success, message)
        except Exception as e:
            self.done.emit(False, str(e))


class SetImageWorker(QThread):
    """Runs SamPredictor.set_image() off the GUI thread (PERF-01).

    Only the encode round-trip moves here. Crop extraction and everything that
    touches QGIS/canvas objects stays on the main thread; this worker just
    carries the ~3-8s stdin/stdout round-trip to the SAM subprocess off the UI
    thread so QGIS never freezes on a new crop.

    Transport safety: the predictor is a subprocess-over-pipes object whose
    stdin is written from exactly one place at a time. While this worker runs,
    the main thread holds the `_encoding_in_progress` lock and never touches the
    predictor pipe, so the JSON-RPC stream is never interleaved. The `done`
    signal carries the generation the encode started with so a stale completion
    (after a teardown that bumped the counter) is dropped by the main thread.
    """
    done = pyqtSignal(int, bool, str)  # (generation, ok, error_message)

    def __init__(self, predictor, image_np, generation: int, parent=None):
        super().__init__(parent)
        self._predictor = predictor
        self._image_np = image_np
        self._generation = generation

    def run(self):
        try:
            # set_image classifies its own transport errors and raises; the
            # main thread reproduces the sync error handling from the message.
            self._predictor.set_image(self._image_np)
            self.done.emit(self._generation, True, "")
        except Exception as e:
            # Pass the message only (matching the sync path's str(e), which the
            # corrupt-checkpoint / broken-venv classifiers read on the main
            # thread). A traceback would pollute the classifier match.
            self.done.emit(self._generation, False, str(e))


class PredictorLoadWorker(QThread):
    """Initializes the SAM predictor off the UI thread (#34)."""
    done = pyqtSignal(object, str)  # (predictor_or_None, err_msg)

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
    """First-open environment checks, off the UI thread.

    The dock must paint instantly when the toolbar icon is clicked. Venv
    status (filesystem walks, hashing, sometimes a subprocess probe on
    Windows) and checkpoint presence run here and report back when done.
    """
    done = pyqtSignal(bool, str, bool)  # (venv_ready, message, checkpoint_ok)

    def run(self):
        try:
            from ..core.checkpoint_manager import cleanup_legacy_sam1_data
            cleanup_legacy_sam1_data()
        except Exception:
            # Logged internally; never block startup on legacy cleanup.
            pass  # nosec B110

        try:
            from ..core.venv_manager import cleanup_old_libs, get_venv_status
            cleanup_old_libs()
            venv_ready, message = get_venv_status()
        except Exception as e:
            import traceback
            # This crash otherwise only reaches a status label + a log line
            # (no telemetry), so capture it: a startup-check failure blocks the
            # whole Manual path. Off the GUI thread; the next flush ships it.
            try:
                from ..core.telemetry import report_exception
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
    """Logs compute-device info off the UI thread (diagnostics only).

    Emits ok=False with the error text when PyTorch cannot load (Windows
    DLL error) so the plugin can show the fix-it dialog on the main thread.
    """
    done = pyqtSignal(bool, str)  # (ok, info_or_error)

    def run(self):
        try:
            from ..core.venv_manager import ensure_venv_packages_available
            ensure_venv_packages_available()
            from ..core.device_manager import get_device_info
            info = get_device_info()
            # Pre-import the crop stack while we are already off-thread: the
            # first Manual crop extraction otherwise pays the cold rasterio
            # import (0.5-2s of native-lib loading, worse behind antivirus) ON
            # the GUI thread at session start. Importing here makes that first
            # extraction a plain windowed read. Best-effort: GDAL-only setups
            # extract through GDAL instead, so a missing rasterio is fine.
            try:
                import rasterio  # noqa: F401
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            self.done.emit(True, str(info or ""))
        except RuntimeError as e:
            self.done.emit(False, str(e))
        except Exception as e:
            self.done.emit(True, f"device_info_unavailable: {e}")


class VerifyWorker(QThread):
    """Runs venv verification + device detection off the main thread."""
    done = pyqtSignal(bool, str)  # (is_valid, message)
    progress = pyqtSignal(int, str)   # (percent, message)

    def run(self):
        try:
            from ..core.venv_manager import verify_venv
            is_valid, msg = verify_venv(
                progress_callback=lambda pct, m: self.progress.emit(pct, m))
            if not is_valid:
                self.done.emit(False, msg)
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
