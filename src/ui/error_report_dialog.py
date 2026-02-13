"""
Error report dialog for the AI Segmentation plugin.
Minimal dialog: error message + copy logs + email contact + TerraLab link.
Also provides a bug report dialog for user-initiated reports.
"""

import sys
import os
import platform
from collections import deque
from datetime import datetime

from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QApplication,
)
from qgis.PyQt.QtCore import Qt
import re

from ..core.i18n import tr

SUPPORT_EMAIL = "yvann.barbot@terra-lab.ai"
TERRALAB_URL = "https://terra-lab.ai/ai-segmentation"

# In-memory log buffer - captures AI Segmentation messages
_log_buffer = deque(maxlen=100)
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

    # macOS: /Users/<username>/...
    text = re.sub(r'/Users/[^/\s]+/', '<USER>/', text)

    # Linux: /home/<username>/...
    text = re.sub(r'/home/[^/\s]+/', '<USER>/', text)

    # Windows: C:\Users\<username>\... (and other drive letters)
    # Handle both forward and backslashes
    text = re.sub(r'[A-Za-z]:\\Users\\[^\\]+\\', '<USER>/', text)
    text = re.sub(r'[A-Za-z]:/Users/[^/]+/', '<USER>/', text)

    return text


def start_log_collector():
    """Connect to QgsMessageLog to capture AI Segmentation messages.
    Call once at plugin startup."""
    global _log_collector_connected
    if _log_collector_connected:
        return
    try:
        from qgis.core import QgsApplication
        QgsApplication.messageLog().messageReceived.connect(_on_log_message)
        _log_collector_connected = True
    except Exception:
        pass


def _on_log_message(message, tag, level):
    """Callback for QgsMessageLog.messageReceived signal."""
    if tag == "AI Segmentation":
        timestamp = datetime.now().strftime("%H:%M:%S")
        _log_buffer.append("[{}] {}".format(timestamp, message))


def _get_recent_logs() -> str:
    """Return recent AI Segmentation log lines (with paths anonymized)."""
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
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("version="):
                        lines.append("Version: {}".format(line.strip().split("=", 1)[1]))
                        break
    except Exception:
        lines.append("Version: unknown")
    lines.append("")

    # System info
    lines.append("--- System ---")
    lines.append("OS: {} ({} {})".format(sys.platform, platform.system(), platform.release()))
    lines.append("Architecture: {}".format(platform.machine()))
    lines.append("Python: {}.{}.{}".format(
        sys.version_info.major, sys.version_info.minor, sys.version_info.micro))

    try:
        from qgis.core import Qgis
        lines.append("QGIS: {}".format(Qgis.QGIS_VERSION))
    except Exception:
        lines.append("QGIS: unknown")
    lines.append("")

    # GPU info
    lines.append("--- GPU ---")
    try:
        from ..core.venv_manager import ensure_venv_packages_available
        ensure_venv_packages_available()
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                lines.append("CUDA {}: {} ({:.1f}GB)".format(i, gpu_name, gpu_mem))
        elif sys.platform == "darwin" and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                lines.append("GPU: Apple Silicon (MPS)")
            else:
                lines.append("GPU: MPS not available")
        else:
            lines.append("GPU: CPU only ({} cores)".format(os.cpu_count()))
        lines.append("PyTorch: {}".format(torch.__version__))
    except Exception:
        lines.append("GPU: could not detect (dependencies not installed)")
    lines.append("")

    # Installed packages
    lines.append("--- Packages ---")
    try:
        from ..core.venv_manager import (
            get_venv_python_path, venv_exists,
            _get_clean_env_for_venv, _get_subprocess_kwargs
        )
        if venv_exists():
            import subprocess
            python_path = get_venv_python_path()
            env = _get_clean_env_for_venv()
            kwargs = _get_subprocess_kwargs()
            result = subprocess.run(
                [python_path, "-m", "pip", "list", "--format=columns"],
                capture_output=True, text=True, timeout=15,
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
        lines.append("Could not list packages: {}".format(str(e)[:100]))
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
                    f for f in os.listdir(latest) if f.endswith('.tif')
                ])
                lines.append("Tiles: {}".format(tif_count))

                if os.path.exists(csv_path):
                    import csv as csv_mod
                    with open(csv_path, "r", encoding="utf-8") as cf:
                        reader = csv_mod.DictReader(cf)
                        rows = list(reader)
                    if rows:
                        first = rows[0]
                        crs_val = first.get("crs", "unknown")
                        res_val = first.get("res", "unknown")
                        lines.append("CRS: {}".format(crs_val))
                        lines.append("Resolution: {}".format(res_val))

                        all_minx = [float(r["minx"]) for r in rows]
                        all_maxx = [float(r["maxx"]) for r in rows]
                        all_miny = [float(r["miny"]) for r in rows]
                        all_maxy = [float(r["maxy"]) for r in rows]
                        lines.append("Bounds: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
                            min(all_minx), max(all_maxx),
                            min(all_miny), max(all_maxy)))
                else:
                    lines.append("(CSV index not found)")
            else:
                lines.append("(No encoded images found)")
        else:
            lines.append("(No features directory)")
    except Exception as e:
        lines.append("Could not read: {}".format(str(e)[:100]))
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
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Error message
        error_label = QLabel(self._error_message[:500])
        error_label.setWordWrap(True)
        error_label.setTextFormat(Qt.PlainText)
        layout.addWidget(error_label)

        # Help text
        help_label = QLabel(
            "{}\n\n{}".format(
                tr("Copy your logs with the button below and send them to our email."),
                tr("We'll fix your issue :)"))
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: palette(text); margin-top: 6px;")
        layout.addWidget(help_label)

        # Action buttons row
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        # Copy logs button
        self._copy_btn = QPushButton(tr("Copy log to clipboard"))
        self._copy_btn.clicked.connect(self._on_copy)
        action_layout.addWidget(self._copy_btn)

        # Email button - copies the email address to clipboard
        self._email_btn = QPushButton(SUPPORT_EMAIL)
        self._email_btn.setToolTip(tr("Copy email address"))
        self._email_btn.clicked.connect(self._on_copy_email)
        action_layout.addWidget(self._email_btn)

        layout.addLayout(action_layout)

        # TerraLab link (discreet, at the bottom)
        link_label = QLabel(
            '<a href="{}" style="color: palette(link);">terra-lab.ai</a>'.format(TERRALAB_URL)
        )
        link_label.setOpenExternalLinks(True)
        link_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(link_label)

    def _on_copy(self):
        """Copy diagnostic info to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self._diagnostic_info)
        self._copy_btn.setText(tr("Copied!"))
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self._copy_btn.setText(tr("Copy log to clipboard")))

    def _on_copy_email(self):
        """Copy support email address to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(SUPPORT_EMAIL)
        self._email_btn.setText(tr("Email copied!"))
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self._email_btn.setText(SUPPORT_EMAIL))


class BugReportDialog(QDialog):
    """
    User-initiated bug report dialog.
    Friendly tone, no error context - user reports something on their own.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Segmentation")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMaximumWidth(500)

        self._diagnostic_info = _collect_diagnostic_info("")

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Friendly message
        msg_label = QLabel(
            "{}\n\n{}".format(
                tr("Something not working?"),
                tr("Copy your logs and send them to us, we'll look into it :)"))
        )
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)

        # Action buttons row
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        # Copy logs button
        self._copy_btn = QPushButton(tr("Copy log to clipboard"))
        self._copy_btn.clicked.connect(self._on_copy)
        action_layout.addWidget(self._copy_btn)

        # Email button - copies the email address to clipboard
        self._email_btn = QPushButton(SUPPORT_EMAIL)
        self._email_btn.setToolTip(tr("Copy email address"))
        self._email_btn.clicked.connect(self._on_copy_email)
        action_layout.addWidget(self._email_btn)

        layout.addLayout(action_layout)

        # TerraLab link (discreet, at the bottom)
        link_label = QLabel(
            '<a href="{}" style="color: palette(link);">terra-lab.ai</a>'.format(TERRALAB_URL)
        )
        link_label.setOpenExternalLinks(True)
        link_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(link_label)

    def _on_copy(self):
        """Copy diagnostic info to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self._diagnostic_info)
        self._copy_btn.setText(tr("Copied!"))
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self._copy_btn.setText(tr("Copy log to clipboard")))

    def _on_copy_email(self):
        """Copy support email address to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(SUPPORT_EMAIL)
        self._email_btn.setText(tr("Email copied!"))
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self._email_btn.setText(SUPPORT_EMAIL))


def show_error_report(parent, error_title: str, error_message: str):
    """Convenience function to show the error report dialog."""
    dialog = ErrorReportDialog(error_title, error_message, parent)
    dialog.exec_()


def show_bug_report(parent):
    """Convenience function to show the bug report dialog."""
    dialog = BugReportDialog(parent)
    dialog.exec_()


CALENDLY_URL = "https://calendly.com/barbot-yvann/30min"


class SuggestFeatureDialog(QDialog):
    """Dialog for users to suggest features or share feedback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Segmentation")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMaximumWidth(500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        msg_label = QLabel(
            "{}\n\n{}".format(
                tr("We're building AI Segmentation for real-world geospatial workflows."),
                tr("We'd love to understand your use case and make the plugin more useful for you :)"))
        )
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        # Email button (blue)
        self._email_btn = QPushButton(SUPPORT_EMAIL)
        self._email_btn.setToolTip(tr("Copy email address"))
        self._email_btn.setStyleSheet(
            "background-color: #1976d2; color: white; padding: 6px 12px;"
        )
        self._email_btn.clicked.connect(self._on_copy_email)
        action_layout.addWidget(self._email_btn)

        # Take a call button (green)
        call_btn = QPushButton(tr("Take a call"))
        call_btn.setStyleSheet(
            "background-color: #2e7d32; color: white; padding: 6px 12px;"
        )
        call_btn.clicked.connect(self._on_take_call)
        action_layout.addWidget(call_btn)

        layout.addLayout(action_layout)

        # TerraLab link
        link_label = QLabel(
            '<a href="{}" style="color: palette(link);">terra-lab.ai</a>'.format(
                TERRALAB_URL)
        )
        link_label.setOpenExternalLinks(True)
        link_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(link_label)

    def _on_copy_email(self):
        """Copy support email address to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(SUPPORT_EMAIL)
        self._email_btn.setText(tr("Email copied!"))
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(
            2000, lambda: self._email_btn.setText(SUPPORT_EMAIL))

    def _on_take_call(self):
        """Open Calendly link in browser."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(CALENDLY_URL))


def show_suggest_feature(parent):
    """Convenience function to show the suggest feature dialog."""
    dialog = SuggestFeatureDialog(parent)
    dialog.exec_()
