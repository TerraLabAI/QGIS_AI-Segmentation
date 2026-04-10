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
