"""
Error report dialog for the AI Segmentation plugin.
Minimal dialog: error message + copy logs + email contact + TerraLab link.
"""

from __future__ import annotations

import os
import platform
import sys

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..core.i18n import tr

# Path anonymization + log capture live in core/log_scrub.py (telemetry
# scrubs every payload with them, so they must not depend on this UI module).
# Re-exported under the historical names for existing callers.
from ..core.log_scrub import (  # noqa: F401
    anonymize_paths as _anonymize_paths,
)
from ..core.log_scrub import (
    get_recent_logs as _get_recent_logs,
)
from ..core.log_scrub import (
    scrub_report as _scrub_report,
)
from .dock.styles import _BTN_BLUE, _BTN_GREEN

SUPPORT_EMAIL = "yvann.barbot@terra-lab.ai"

# Subtle down-arrow between step 1 and step 2: a muted, small glyph so it
# reads as flow guidance, not content.
_ARROW_STYLE = "color: rgba(128,128,128,0.65); font-size: 10px;"


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

    # Last Automatic run correlation id: lets support match this report to the
    # server-archived run. Omitted entirely when no run happened this session.
    try:
        from ..core.telemetry import get_last_run_id
        run_id = get_last_run_id()
    except Exception:
        run_id = None
    if run_id:
        lines.append("--- Run ---")
        lines.append(f"Run ID: {run_id}")
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

    # Device info. Never IMPORT torch here: this dialog opens on the GUI
    # thread right after something already failed, and a first-time torch
    # import (native extensions, antivirus rescans) can freeze QGIS for
    # seconds at the worst possible moment. Report from the already-loaded
    # module when the session has one; otherwise say so and move on.
    lines.append("--- Device ---")
    try:
        torch = sys.modules.get("torch")
        if torch is not None:
            if sys.platform == "darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                lines.append("Device: Apple Silicon (MPS)")
            else:
                lines.append(f"Device: CPU ({os.cpu_count()} cores)")
            lines.append(f"PyTorch: {torch.__version__}")
        else:
            lines.append(f"Device: CPU ({os.cpu_count()} cores), AI engine not loaded this session")
    except Exception:
        lines.append("Device: could not detect (dependencies not installed)")
    lines.append("")

    # Environment: the isolated venv path (anonymized on the way out) so a
    # report says WHICH environment failed, plus whether it exists at all.
    lines.append("--- Environment ---")
    venv_python = None
    try:
        from ..core.venv_manager import get_venv_python_path, venv_exists
        if venv_exists():
            venv_python = get_venv_python_path()
            lines.append(f"Venv Python: {venv_python}")
        else:
            lines.append("Venv: not found (dependencies not installed)")
    except Exception as e:
        lines.append(f"Venv: could not resolve ({str(e)[:80]})")
    lines.append("")

    # Import status: whether the heavy native packages actually IMPORT in the
    # venv. This is the single most useful install signal: pip can list a
    # package that then fails to import (broken native extension, antivirus
    # quarantine, arch mismatch), which is exactly the case pip list hides.
    lines.append("--- Import Status ---")
    if venv_python:
        try:
            import subprocess  # nosec B404

            from ..core.subprocess_utils import get_clean_env_for_venv, get_subprocess_kwargs
            probe = (
                "import importlib\n"
                "for m in ['numpy','torch','torchvision','rasterio','scipy','cv2','PIL']:\n"
                "    try:\n"
                "        mod=importlib.import_module(m)\n"
                "        print(m+': OK '+str(getattr(mod,'__version__','')))\n"
                "    except Exception as e:\n"
                "        print(m+': FAIL '+type(e).__name__+': '+str(e)[:120])\n"
            )
            env = get_clean_env_for_venv()
            kwargs = get_subprocess_kwargs()
            result = subprocess.run(  # nosec B603
                [venv_python, "-c", probe],
                capture_output=True, text=True,
                # The child is forced to write UTF-8 (PYTHONIOENCODING), so it
                # must be decoded as UTF-8: the Windows ANSI code page turns an
                # accented path into mojibake, or raises and loses the whole
                # section.
                encoding="utf-8", errors="replace", timeout=20,
                stdin=subprocess.DEVNULL, env=env, **kwargs
            )
            out = (result.stdout or "").strip()
            lines.append(out or "(no import output)")
            err = (result.stderr or "").strip()
            if err:
                lines.append("stderr: " + err.splitlines()[-1][:120])
        except Exception as e:
            lines.append(f"Could not check imports: {str(e)[:100]}")
    else:
        lines.append("(skipped: no venv)")
    lines.append("")

    # Installed packages
    lines.append("--- Packages ---")
    try:
        from ..core.subprocess_utils import get_clean_env_for_venv, get_subprocess_kwargs
        if venv_python:
            import subprocess  # nosec B404
            env = get_clean_env_for_venv()
            kwargs = get_subprocess_kwargs()
            result = subprocess.run(  # nosec B603
                [venv_python, "-m", "pip", "list", "--format=columns"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=5,
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
                        # SIZE, not position. The raster name is withheld just
                        # above for privacy, and printing the corners to two
                        # decimals in a projected CRS would put the site back
                        # in the report to the centimetre. What diagnoses a
                        # bug is how big the area is, with the CRS and the
                        # resolution already above.
                        lines.append(
                            f"Extent size: {max(all_maxx) - min(all_minx):.0f} x "
                            f"{max(all_maxy) - min(all_miny):.0f} (CRS units)"
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
    # Scrub the WHOLE report: user paths (privacy) AND any backend/infra/model
    # shape (a network error can carry an endpoint URL into the error text).
    report = "\n".join(lines)
    return _scrub_report(report)


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
        # Diagnostics run two subprocesses in the isolated environment (up to
        # 25 s together). Collected on the copy click, not here: gathering them
        # up front froze QGIS before the dialog even painted, so the user saw
        # nothing at all after the failure.
        self._diagnostic_info: str | None = None

        self._setup_ui()

    def _get_diagnostic_info(self) -> str:
        """Collect the diagnostics once, on first use."""
        if self._diagnostic_info is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._diagnostic_info = _collect_diagnostic_info(self._error_message)
            finally:
                QApplication.restoreOverrideCursor()
        return self._diagnostic_info

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Error message
        error_label = QLabel(self._error_message[:500])
        error_label.setWordWrap(True)
        error_label.setTextFormat(Qt.TextFormat.PlainText)
        error_label.setStyleSheet("font-size: 12px; color: palette(text);")
        layout.addWidget(error_label)

        # Help text
        help_label = QLabel(
            "{}\n\n{}".format(
                tr("Copy your logs with the button below and send them to our support email."),
                tr("We'll get this fixed for you :)"))
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("font-size: 11px; color: palette(text);")
        layout.addWidget(help_label)

        # Step 1: Copy logs button (full width) - green primary
        self._copy_btn = QPushButton(tr("1. Click to copy logs"))
        self._copy_btn.setStyleSheet(_BTN_GREEN)
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.clicked.connect(self._on_copy)
        layout.addWidget(self._copy_btn)

        # Arrow pointing down, centered (muted flow hint)
        arrow_label = QLabel("\u25BC")
        arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow_label.setStyleSheet(_ARROW_STYLE)
        layout.addWidget(arrow_label)

        # Step 2: Email button (full width) - blue secondary, opens mailto link
        self._email_btn = QPushButton(tr("2. Click to send to {}").format(SUPPORT_EMAIL))
        self._email_btn.setToolTip(tr("Open email client"))
        self._email_btn.setStyleSheet(_BTN_BLUE)
        self._email_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._email_btn.clicked.connect(self._on_open_email)
        layout.addWidget(self._email_btn)

    def _on_copy(self):
        """Copy diagnostic info to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self._get_diagnostic_info())
        self._copy_btn.setText(tr("Copied!"))
        # Bound to the dialog: a bare singleShot keeps the lambda and the
        # button alive in the global event loop, so closing the dialog inside
        # the two seconds landed the call on a freed widget.
        from ..core.qt_compat import safe_single_shot
        safe_single_shot(
            2000, self, lambda: self._copy_btn.setText(tr("1. Click to copy logs")))

    def _on_open_email(self):
        """Open email client with support address."""
        from urllib.parse import quote

        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        subject = quote("AI Segmentation - Bug Report")
        QDesktopServices.openUrl(QUrl(f"mailto:{SUPPORT_EMAIL}?subject={subject}"))


def show_error_report(
    parent, error_title: str, error_message: str, error_code: str = "",
    track: bool = True,
):
    """Convenience function to show the error report dialog.

    `error_code` should be a canonical English snake_case identifier
    (e.g. "crop_error"). If omitted, falls back to the language-independent
    `_short_code()` constant - it never derives a code from the translated
    `error_title`, so dashboards aggregate cleanly across UI languages.
    Callers should still pass an explicit code so the breakdown stays useful.

    `track=False` skips the internal plugin_error event: callers that already
    tracked the same failure (or that re-open this dialog from a persistent
    "Report this problem" link) pass False so one failure counts exactly once.
    """
    if track:
        try:
            from ..core.telemetry import track_plugin_error
            stage = _infer_stage(error_title, error_message)
            first_line = (error_message or "").splitlines()[0] if error_message else ""
            code = error_code or _short_code(error_title)
            # Install/download logs are pip and download output (no user
            # geodata, paths and coords scrubbed anyway); the tail is what
            # turns a useless "network error" event into a diagnosable one.
            track_plugin_error(
                stage=stage,
                error_code=code,
                message=first_line,
                include_log_tail=stage in ("install", "download"),
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


def _short_code(_title: str = "") -> str:
    """Last-resort error_code when a caller omits an explicit `error_code=`.

    A dialog title is a user-facing string passed through `tr()`, so it is
    translated per QGIS locale. Slugifying it (the old behavior) produced a
    DIFFERENT code per language for the SAME error - e.g. the Spanish "Crop
    Error" title "Error de recorte" became `error_de_recorte` while English
    stayed `crop_error`, fragmenting the analytics breakdown by UI language.
    Note this happens even for accent-free titles, so an ASCII check is not
    enough: a localized title must NEVER become an error_code.

    We therefore return a single language-independent constant and ignore the
    title entirely. Every real error path passes a canonical English
    snake_case `error_code=`; reaching this constant means a call site forgot
    to, which is a code smell to fix at the call site, not here.
    """
    return "unspecified_error"
