"""Background QThread workers for dependency install, model download, and verification."""

from qgis.PyQt.QtCore import QThread, pyqtSignal

from ..core.i18n import tr


class DepsInstallWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from ..core.venv_manager import create_venv_and_install
            success, message = create_venv_and_install(
                progress_callback=lambda percent, msg: self.progress.emit(percent, msg),
                cancel_check=lambda: self._cancelled,
            )
            self.finished.emit(success, message)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


class DownloadWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            from ..core.checkpoint_manager import download_checkpoint
            success, message = download_checkpoint(
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, str(e))


class PredictorLoadWorker(QThread):
    """Initializes the SAM predictor off the UI thread (#34)."""
    finished = pyqtSignal(object, str)  # (predictor_or_None, err_msg)

    def run(self):
        try:
            from ..core.checkpoint_manager import get_checkpoint_path
            from ..core.sam_predictor import SamPredictor, build_sam_predictor_config
            sam_config = build_sam_predictor_config(checkpoint=get_checkpoint_path())
            predictor = SamPredictor(sam_config)
            self.finished.emit(predictor, "")
        except Exception as e:
            import traceback
            self.finished.emit(None, f"{e}\n{traceback.format_exc()}")


class StartupCheckWorker(QThread):
    """First-open environment checks, off the UI thread.

    The dock must paint instantly when the toolbar icon is clicked. Venv
    status (filesystem walks, hashing, sometimes a subprocess probe on
    Windows) and checkpoint presence run here and report back when done.
    """
    finished = pyqtSignal(bool, str, bool)  # (venv_ready, message, checkpoint_ok)

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
            self.finished.emit(
                False, f"startup_error: {e}\n{traceback.format_exc()}", False)
            return

        checkpoint_ok = False
        if venv_ready:
            try:
                from ..core.checkpoint_manager import checkpoint_exists
                checkpoint_ok = checkpoint_exists()
            except Exception:
                checkpoint_ok = False
        self.finished.emit(venv_ready, message, checkpoint_ok)


class DeviceInfoWorker(QThread):
    """Logs compute-device info off the UI thread (diagnostics only).

    Emits ok=False with the error text when PyTorch cannot load (Windows
    DLL error) so the plugin can show the fix-it dialog on the main thread.
    """
    finished = pyqtSignal(bool, str)  # (ok, info_or_error)

    def run(self):
        try:
            from ..core.venv_manager import ensure_venv_packages_available
            ensure_venv_packages_available()
            from ..core.device_manager import get_device_info
            info = get_device_info()
            self.finished.emit(True, str(info or ""))
        except RuntimeError as e:
            self.finished.emit(False, str(e))
        except Exception as e:
            self.finished.emit(True, f"device_info_unavailable: {e}")


class KeyRevalidateWorker(QThread):
    """Re-checks the stored activation key against the server off-thread.

    The old synchronous call froze the dock for seconds on slow networks.
    The auth header is read on the main thread and passed in, and any
    clearing of a rejected key happens back on the main thread: settings and
    QgsAuthManager are never touched from this worker.
    """
    finished = pyqtSignal(bool)  # (key_still_valid)

    def __init__(self, auth: dict, parent=None):
        super().__init__(parent)
        self._auth = auth

    def run(self):
        try:
            from ..api.terralab_client import TerraLabClient
            from ..core.activation_manager import is_rejection_code
            result = TerraLabClient().get_usage(auth=self._auth)
            if "error" in result and is_rejection_code(result.get("code", "")):
                self.finished.emit(False)
                return
            self.finished.emit(True)
        except Exception:
            # Benefit of the doubt: never sign the user out on a local error.
            self.finished.emit(True)


class KeyValidateWorker(QThread):
    """Validates a pasted activation key without blocking the UI thread.

    Network-only (check_key_with_server never persists); the caller saves
    the key on the main thread when success comes back.
    """
    finished = pyqtSignal(bool, str, str)  # (success, message, key)

    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self._key = key

    def run(self):
        try:
            from ..core.activation_manager import check_key_with_server
            success, message = check_key_with_server(self._key)
            self.finished.emit(success, message, self._key)
        except Exception as e:
            self.finished.emit(False, str(e), self._key)


class VerifyWorker(QThread):
    """Runs venv verification + device detection off the main thread."""
    finished = pyqtSignal(bool, str)  # (is_valid, message)
    progress = pyqtSignal(int, str)   # (percent, message)

    def run(self):
        try:
            from ..core.venv_manager import verify_venv
            is_valid, msg = verify_venv(
                progress_callback=lambda pct, m: self.progress.emit(pct, m))
            if not is_valid:
                self.finished.emit(False, msg)
                return
            self.progress.emit(100, tr("Detecting device..."))
            try:
                from ..core.venv_manager import ensure_venv_packages_available
                ensure_venv_packages_available()
                from ..core.device_manager import get_device_info
                info = get_device_info()
                self.finished.emit(True, info or "")
            except Exception as e:
                self.finished.emit(True, f"device_error: {str(e)}")
        except Exception as e:
            self.finished.emit(False, str(e))
