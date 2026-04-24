"""
Error report dialog for the AI Segmentation plugin.
Minimal dialog: error message + copy logs + email contact + TerraLab link.
Also provides a bug report dialog for user-initiated reports.
"""

from __future__ import annotations

import os
import platform
import re
import sys
import threading
from collections import deque
from datetime import datetime

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..core.i18n import tr

SUPPORT_EMAIL = "yvann.barbot@terra-lab.ai"

# In-memory log buffer - captures AI Segmentation messages
_log_buffer = deque(maxlen=100)
_log_buffer_lock = threading.Lock()
_log_collector_connected = False


def _anonymize_paths(text: str) -> str:
    """
    Anonymize file paths to hide the username.

    Replaces patterns like:
    - /Users/<username>/... -> <USER>/...
    - /home/<username>/... -> <USER>/...
    - C:\\Users\\<username>\\... -> <USER>/...

    This protects user privacy when sending logs and reports.
    """
    if not text:
        return text

    # macOS: /Users/<username>/... (with or without trailing slash)
    text = re.sub(r"/Users/[^/\s]+(?=/|$|\s)", "<USER>", text)

    # Linux: /home/<username>/... (with or without trailing slash)
    text = re.sub(r"/home/[^/\s]+(?=/|$|\s)", "<USER>", text)

    # Windows: C:\Users\<username>\... (and other drive letters, forward/back slashes)
    text = re.sub(r"[A-Za-z]:[/\\]Users[/\\][^/\\\s]+(?=[/\\]|$|\s)", "<USER>", text)

    # Windows UNC paths: \\server\Users\<username>\...
    return re.sub(r"\\\\[^\\]+\\Users\\[^/\\\s]+(?=[/\\]|$|\s)", "<USER>", text)


def start_log_collector():
    """Connect to QgsMessageLog to capture AI Segmentation messages.
    Call once at plugin startup."""
    global _log_collector_connected
    with _log_buffer_lock:
        if _log_collector_connected:
            return
        try:
            from qgis.core import QgsApplication
            QgsApplication.messageLog().messageReceived.connect(_on_log_message)
            _log_collector_connected = True
        except Exception:
            pass  # nosec B110


def stop_log_collector():
    """Disconnect from QgsMessageLog. Call on plugin unload."""
    global _log_collector_connected
    with _log_buffer_lock:
        if not _log_collector_connected:
            return
        try:
            from qgis.core import QgsApplication
            QgsApplication.messageLog().messageReceived.disconnect(_on_log_message)
        except (TypeError, RuntimeError):
            pass
        _log_collector_connected = False


def _on_log_message(message, tag, level):
    """Callback for QgsMessageLog.messageReceived signal."""
    if tag == "AI Segmentation":
        timestamp = datetime.now().strftime("%H:%M:%S")
        with _log_buffer_lock:
            _log_buffer.append(f"[{timestamp}] {message}")


def _get_recent_logs() -> str:
    """Return recent AI Segmentation log lines (with paths anonymized)."""
    with _log_buffer_lock:
        if not _log_buffer:
            return "(No logs captured this session)"
        logs = "\n".join(_log_buffer)
    return _anonymize_paths(logs)


def _collect_diagnostic_info(error_message: str) -> str:
    """Collect system diagnostic info for error reporting."""
    lines = []
    lines.append("=== AI Segmentation - Error Report ===")
    lines.append("")

    # Error
    if error_message:
        lines.append("--- Error ---")
        lines.append(error_message)
        lines.append("")

    # Plugin version
    lines.append("--- Plugin ---")
    try:
        plugin_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        metadata_path = os.path.join(plugin_dir, "metadata.txt")
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("version="):
                        lines.append("Version: {}".format(line.strip().split("=", 1)[1]))
                        break
    except Exception:
        lines.append("Version: unknown")
    lines.append("")

    # System info
    lines.append("--- System ---")
    lines.append(f"OS: {sys.platform} ({platform.system()} {platform.release()})")
    from ..core.model_config import IS_ROSETTA
    arch_str = platform.machine()
    if IS_ROSETTA:
        arch_str += " (Rosetta on Apple Silicon)"
    lines.append(f"Architecture: {arch_str}")
    lines.append(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    try:
        from qgis.core import Qgis
        lines.append(f"QGIS: {Qgis.QGIS_VERSION}")
    except Exception:
        lines.append("QGIS: unknown")
    lines.append("")

    # Device info
    lines.append("--- Device ---")
    try:
        from ..core.venv_manager import ensure_venv_packages_available
        ensure_venv_packages_available()
        import torch
        if sys.platform == "darwin" and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                lines.append("Device: Apple Silicon (MPS)")
            else:
                lines.append(f"Device: CPU ({os.cpu_count()} cores)")
        else:
            lines.append(f"Device: CPU ({os.cpu_count()} cores)")
        lines.append(f"PyTorch: {torch.__version__}")
    except Exception:
        lines.append("Device: could not detect (dependencies not installed)")
    lines.append("")

    # Installed packages
    lines.append("--- Packages ---")
    try:
        from ..core.subprocess_utils import get_clean_env_for_venv, get_subprocess_kwargs
        from ..core.venv_manager import get_venv_python_path, venv_exists
        if venv_exists():
            import subprocess  # nosec B404
            python_path = get_venv_python_path()
            env = get_clean_env_for_venv()
            kwargs = get_subprocess_kwargs()
            result = subprocess.run(  # nosec B603
                [python_path, "-m", "pip", "list", "--format=columns"],
                capture_output=True, text=True, timeout=5,
                stdin=subprocess.DEVNULL, env=env, **kwargs
            )
            if result.returncode == 0:
                for pkg_line in result.stdout.strip().split("\n"):
                    lines.append(pkg_line)
            else:
                lines.append("pip list failed")
        else:
            lines.append("Virtual environment not found")
    except Exception as e:
        lines.append(f"Could not list packages: {str(e)[:100]}")
    lines.append("")

    # Last encoded image info
    lines.append("--- Last Encoded Image ---")
    try:
        from ..core.checkpoint_manager import FEATURES_DIR
        if os.path.isdir(FEATURES_DIR):
            subdirs = [
                os.path.join(FEATURES_DIR, d)
                for d in os.listdir(FEATURES_DIR)
                if os.path.isdir(os.path.join(FEATURES_DIR, d))
            ]
            if subdirs:
                latest = max(subdirs, key=os.path.getmtime)
                folder_name = os.path.basename(latest)
                # Anonymize raster name for privacy - only show extension
                lines.append("Raster: xxx.tif")

                csv_path = os.path.join(latest, folder_name + ".csv")
                tif_count = len([
                    f for f in os.listdir(latest) if f.endswith(".tif")
                ])
                lines.append(f"Tiles: {tif_count}")

                if os.path.exists(csv_path):
                    import csv as csv_mod
                    with open(csv_path, encoding="utf-8") as cf:
                        reader = csv_mod.DictReader(cf)
                        rows = list(reader)
                    if rows:
                        first = rows[0]
                        crs_val = first.get("crs", "unknown")
                        res_val = first.get("res", "unknown")
                        lines.append(f"CRS: {crs_val}")
                        lines.append(f"Resolution: {res_val}")

                        all_minx = [float(r["minx"]) for r in rows]
                        all_maxx = [float(r["maxx"]) for r in rows]
                        all_miny = [float(r["miny"]) for r in rows]
                        all_maxy = [float(r["maxy"]) for r in rows]
                        lines.append(
                            f"Bounds: [{min(all_minx):.2f}, {max(all_maxx):.2f}, "
                            f"{min(all_miny):.2f}, {max(all_maxy):.2f}]"
                        )
                else:
                    lines.append("(CSV index not found)")
            else:
                lines.append("(No encoded images found)")
        else:
            lines.append("(No features directory)")
    except Exception as e:
        lines.append(f"Could not read: {str(e)[:100]}")
    lines.append("")

    # Recent logs from the in-memory buffer
    lines.append("--- Recent Logs ---")
    lines.append(_get_recent_logs())

    lines.append("")
    lines.append("=== End of Report ===")
    # Anonymize all paths in the report to protect user privacy
    report = "\n".join(lines)
    return _anonymize_paths(report)


class ErrorReportDialog(QDialog):
    """
    Minimal error report dialog.
    Shows error message, copy-logs button, email button, and TerraLab link.
    """

    def __init__(self, error_title: str, error_message: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Segmentation")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMaximumWidth(500)

        self._error_title = error_title
        self._error_message = error_message
        self._diagnostic_info = _collect_diagnostic_info(error_message)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        # Error message
        error_label = QLabel(self._error_message[:500])
        error_label.setWordWrap(True)
        error_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(error_label)

        # Help text
        help_label = QLabel(
            "{}\n\n{}".format(
                tr("Copy your logs with the button below and send them to our email."),
                tr("We'll fix your issue :)"))
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        # Step 1: Copy logs button (full width)
        self._copy_btn = QPushButton(tr("1. Click to copy logs"))
        self._copy_btn.clicked.connect(self._on_copy)
        layout.addWidget(self._copy_btn)

        # Arrow pointing down, centered
        arrow_label = QLabel("\u25BC")
        arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(arrow_label)

        # Step 2: Email button (full width) - opens mailto link
        self._email_btn = QPushButton(tr("2. Click to send to {}").format(SUPPORT_EMAIL))
        self._email_btn.setToolTip(tr("Open email client"))
        self._email_btn.clicked.connect(self._on_open_email)
        layout.addWidget(self._email_btn)

    def _on_copy(self):
        """Copy diagnostic info to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self._diagnostic_info)
        self._copy_btn.setText(tr("Copied!"))
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self._copy_btn.setText(tr("1. Click to copy logs")))

    def _on_open_email(self):
        """Open email client with support address."""
        from urllib.parse import quote

        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        subject = quote("AI Segmentation - Bug Report")
        QDesktopServices.openUrl(QUrl(f"mailto:{SUPPORT_EMAIL}?subject={subject}"))


class BugReportDialog(QDialog):
    """
    User-initiated bug report dialog.
    Friendly tone, no error context - user reports something on their own.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Report a Bug"))
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMaximumWidth(500)

        self._diagnostic_info = _collect_diagnostic_info("")

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        # Friendly message
        msg_label = QLabel(
            "{}\n\n{}".format(
                tr("Something not working?"),
                tr("Copy your logs and send them to us, we'll look into it :)"))
        )
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)

        # Step 1: Copy logs button (full width)
        self._copy_btn = QPushButton(tr("1. Click to copy logs"))
        self._copy_btn.clicked.connect(self._on_copy)
        layout.addWidget(self._copy_btn)

        # Arrow pointing down, centered
        arrow_label = QLabel("\u25BC")
        arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(arrow_label)

        # Step 2: Email button (full width) - opens mailto link
        self._email_btn = QPushButton(tr("2. Click to send to {}").format(SUPPORT_EMAIL))
        self._email_btn.setToolTip(tr("Open email client"))
        self._email_btn.clicked.connect(self._on_open_email)
        layout.addWidget(self._email_btn)

    def _on_copy(self):
        """Copy diagnostic info to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self._diagnostic_info)
        self._copy_btn.setText(tr("Copied!"))
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self._copy_btn.setText(tr("1. Click to copy logs")))

    def _on_open_email(self):
        """Open email client with support address."""
        from urllib.parse import quote

        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        subject = quote("AI Segmentation - Bug Report")
        QDesktopServices.openUrl(QUrl(f"mailto:{SUPPORT_EMAIL}?subject={subject}"))


def show_error_report(parent, error_title: str, error_message: str):
    """Convenience function to show the error report dialog."""
    try:
        from ..core.telemetry import track_plugin_error
        stage = _infer_stage(error_title, error_message)
        first_line = (error_message or "").splitlines()[0] if error_message else ""
        track_plugin_error(
            stage=stage,
            error_code=_short_code(error_title),
            message=first_line,
        )
    except Exception:
        pass  # nosec B110

    dialog = ErrorReportDialog(error_title, error_message, parent)
    dialog.exec()


def _infer_stage(title: str, message: str) -> str:
    """Heuristic mapping from error dialog title/message to a telemetry stage.

    Stages: install | download | activate | segment | export | other.
    """
    haystack = f"{title} {message}".lower()
    if any(k in haystack for k in ("venv", "pip", "install", "dependency", "dependencies")):
        return "install"
    if any(k in haystack for k in ("download", "checkpoint", "model")):
        return "download"
    if "activation" in haystack or "activate" in haystack:
        return "activate"
    if "export" in haystack:
        return "export"
    if any(k in haystack for k in ("segmentation", "segment", "mask", "predict")):
        return "segment"
    return "other"


def _short_code(title: str) -> str:
    """Compact, lowercase error_code from the dialog title."""
    import re as _re
    if not title:
        return "unknown"
    code = _re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return code[:40] or "unknown"


def show_bug_report(parent):
    """Convenience function to show the bug report dialog."""
    dialog = BugReportDialog(parent)
    dialog.exec()
