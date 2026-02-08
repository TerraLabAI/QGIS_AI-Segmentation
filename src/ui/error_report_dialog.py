




from __future__ import annotations

import os
import platform
import sys

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ..core.activation_manager import get_support_email
from ..core.i18n import tr




from ..core.log_scrub import (
    get_recent_logs as _get_recent_logs,
)
from ..core.log_scrub import (
    scrub_report as _scrub_report,
)
from ..core.server_dials import dial_in_range
from .dock.styles import (
    _BTN_SETTINGS_ACCENT,
    _BTN_SETTINGS_GHOST,
    _SCROLL_AREA_QSS,
    FIELD,
    FONT_BODY,
    FONT_HINT,
    INK,
    INK_2,
    LINE,
    RADIUS_CONTROL,
)
from .settings.a11y import make_accessible
from .settings.settings_widgets import TITLE_PX


SUPPORT_EMAIL = "yvann.barbot@terra-lab.ai"

_DIALOG_QSS = (
    "QDialog#errorReportDialog { background: palette(window); }"
    f"QLabel#reportTitle {{ font-size: {TITLE_PX}px; font-weight: 600; color: {INK}; }}"
    f"QLabel#reportHelp {{ font-size: {FONT_BODY}px; color: {INK_2}; }}"
    f"QLabel#reportAddress {{ font-size: {FONT_HINT}px; color: {INK_2}; }}"
)



_MESSAGE_BOX_STYLE = (
    f"QPlainTextEdit {{ background-color: {FIELD}; color: {INK};"
    f" border: 1px solid {LINE}; border-radius: {RADIUS_CONTROL}px;"
    f" font-size: {FONT_BODY}px; padding: 8px 10px; }}"
    + _SCROLL_AREA_QSS
)
_MESSAGE_BOX_MAX_H = 160
_DIALOG_MAX_W = 500


def _collect_diagnostic_info(error_message: str) -> str:

    lines = []
    lines.append("=== AI Segmentation - Error Report ===")
    lines.append("")


    if error_message:
        lines.append("--- Error ---")
        lines.append(error_message)
        lines.append("")


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



    try:
        from ..core.telemetry import get_last_run_id
        run_id = get_last_run_id()
    except Exception:
        run_id = None
    if run_id:
        lines.append("--- Run ---")
        lines.append(f"Run ID: {run_id}")
        lines.append("")


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


    lines.append("--- Recent Logs ---")
    lines.append(_get_recent_logs())




    try:
        from ..core.venv_manager import read_install_log_tail
        install_tail = read_install_log_tail(60)
    except Exception:  # noqa: BLE001
        install_tail = ""
    if install_tail:
        lines.append("")
        lines.append("--- Install Log (last lines) ---")
        lines.append(install_tail)

    lines.append("")
    lines.append("=== End of Report ===")


    report = "\n".join(lines)
    return _scrub_report(report)


class _DiagnosticsCollector(QThread):







    collected = pyqtSignal(str)

    def __init__(self, error_message: str):
        super().__init__()
        self._error_message = error_message

    def run(self):
        try:
            report = _collect_diagnostic_info(self._error_message)
        except Exception as err:  # noqa: BLE001
            report = _scrub_report(
                "=== AI Segmentation - Error Report ===\n\n"
                f"{self._error_message}\n\n"
                f"(The rest could not be collected: {str(err)[:120]})"
            )
        self.collected.emit(report)


class ErrorReportDialog(QDialog):





    def __init__(self, error_title: str, error_message: str, parent=None):
        super().__init__(parent)

        self.setWindowTitle(error_title or "AI Segmentation")
        self.setModal(True)
        from .dock.font_scale import scale_px_length

        self.setMinimumWidth(scale_px_length(400))


        self.setMaximumWidth(scale_px_length(_DIALOG_MAX_W))

        self._error_title = error_title
        self._error_message = error_message





        self._diagnostic_info: str | None = None

        self._setup_ui()
        self._start_collecting()

    def _start_collecting(self) -> None:






        try:
            collector = _DiagnosticsCollector(self._error_message)
            collector.collected.connect(self._on_diagnostics_collected)
            from .plugin.shared import park_orphaned_worker

            park_orphaned_worker(collector)
            collector.start()
        except Exception:  # noqa: BLE001
            self._set_copy_ready(True)

    def _on_diagnostics_collected(self, report: str) -> None:
        self._diagnostic_info = report
        self._set_copy_ready(True)

    def _set_copy_ready(self, ready: bool) -> None:






        try:
            self._copy_btn.setEnabled(ready)
            self._copy_btn.setText(
                tr("1. Click to copy logs") if ready
                else tr("Reading your logs..."))
        except RuntimeError:
            pass  # nosec B110
        try:
            self._email_btn.setEnabled(ready)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _get_diagnostic_info(self) -> str:

        if self._diagnostic_info is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._diagnostic_info = _collect_diagnostic_info(self._error_message)
            finally:
                QApplication.restoreOverrideCursor()
        return self._diagnostic_info

    def _setup_ui(self):
        from .dock.font_scale import scale_px_length, scale_qss_font_px

        self.setObjectName("errorReportDialog")
        self.setStyleSheet(scale_qss_font_px(_DIALOG_QSS))
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(22, 20, 22, 18)



        title_label = QLabel(self._error_title or tr("Report a problem"))
        title_label.setObjectName("reportTitle")
        title_label.setWordWrap(True)
        title_label.setTextFormat(Qt.TextFormat.PlainText)
        from .settings.category_tile import category_icon_tile, tile_beside


        self._title_tile = category_icon_tile("warning", "coral", self)
        layout.addLayout(tile_beside(self._title_tile, title_label))



        if self._error_message:
            error_box = QPlainTextEdit(self._error_message)
            error_box.setReadOnly(True)
            error_box.setStyleSheet(scale_qss_font_px(_MESSAGE_BOX_STYLE))
            error_box.setMaximumHeight(scale_px_length(_MESSAGE_BOX_MAX_H))
            layout.addWidget(error_box)

        help_label = QLabel(
            tr("Copy your logs, then send them to us.")
        )
        help_label.setObjectName("reportHelp")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        layout.addSpacing(4)



        self._copy_btn = QPushButton(tr("Reading your logs..."))
        self._copy_btn.setEnabled(False)
        make_accessible(self._copy_btn, _BTN_SETTINGS_ACCENT)
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.setDefault(True)
        self._copy_btn.clicked.connect(self._on_copy)
        layout.addWidget(self._copy_btn)



        self._email_btn = QPushButton(tr("2. Send to support"))
        self._email_btn.setToolTip(tr("Open email client"))
        make_accessible(self._email_btn, _BTN_SETTINGS_GHOST)
        self._email_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._email_btn.setEnabled(False)
        self._email_btn.setAutoDefault(False)
        self._email_btn.clicked.connect(self._on_open_email)
        layout.addWidget(self._email_btn)

        address_label = QLabel(get_support_email(SUPPORT_EMAIL))
        address_label.setObjectName("reportAddress")
        address_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        address_label.setTextFormat(Qt.TextFormat.PlainText)
        address_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(address_label)



        close_row = QHBoxLayout()
        close_row.addStretch(1)
        self._close_btn = QPushButton(tr("Close"))
        make_accessible(self._close_btn, _BTN_SETTINGS_GHOST)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setAutoDefault(False)
        self._close_btn.clicked.connect(self.reject)
        close_row.addWidget(self._close_btn)
        layout.addLayout(close_row)

        from .dock.font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(self)



        inner = self.minimumWidth() - 44
        help_label.ensurePolished()
        help_label.setMinimumHeight(help_label.heightForWidth(inner))
        inner -= self._title_tile.width() + 12
        for label in (title_label,):
            label.ensurePolished()
            label.setMinimumHeight(label.heightForWidth(inner))

    def _on_copy(self):

        clipboard = QApplication.clipboard()
        clipboard.setText(self._get_diagnostic_info())
        self._flash_copy_text(tr("Copied!"))

    def _flash_copy_text(self, text: str) -> None:

        try:
            self._copy_btn.setText(text)
        except (RuntimeError, AttributeError):
            return



        from ..core.qt_compat import safe_single_shot
        reset_ms = dial_in_range("tuning.ui.copy_feedback_ms", 2000, 500, 6000)
        safe_single_shot(reset_ms, self, self._restore_copy_label)

    def _restore_copy_label(self) -> None:
        try:
            self._copy_btn.setText(tr("1. Click to copy logs"))
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _put_report_on_clipboard(self, report: str) -> bool:






        if not report:
            return False
        try:
            clipboard = QApplication.clipboard()
            if clipboard is None:
                return False
            clipboard.setText(report)
        except (RuntimeError, AttributeError):
            return False
        self._flash_copy_text(tr("Report copied: paste it into your email"))
        return True

    def _on_open_email(self):











        from urllib.parse import quote

        from .external_links import open_email
        report = self._get_diagnostic_info()
        self._put_report_on_clipboard(report)
        subject = quote("AI Segmentation - Bug Report")
        support_email = get_support_email(SUPPORT_EMAIL)
        opened = open_email(
            f"mailto:{support_email}?subject={subject}", support_email, parent=self)
        if not opened:
            self._put_report_on_clipboard(report)


def _telemetry_summary(error_message: str) -> str:






    lines = [line.strip() for line in (error_message or "").splitlines() if line.strip()]
    if not lines:
        return ""
    head = lines[0]
    if head.endswith((":", "\uff1a")) and len(lines) > 1:
        return f"{head} {lines[1]}"
    return head


def show_error_report(
    parent, error_title: str, error_message: str, error_code: str = "",
    track: bool = True,
):












    if track:
        try:
            from ..core.telemetry_errors import track_plugin_error
            stage = _stage_for_code(error_code) or _infer_stage(
                error_title, error_message)
            first_line = _telemetry_summary(error_message)
            code = error_code or _short_code(error_title)



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





_STAGE_CODE_MARKERS = (
    ("install", ("install", "venv", "pip", "dependen", "uv_", "python_")),
    ("download", ("download", "checkpoint", "weight")),
    ("activate", ("activat", "sign_in", "signin", "license", "licence", "key_")),
    ("export", ("export", "gpkg", "save_")),
    ("segment", ("segment", "mask", "predict", "detect", "crop", "refine")),
)


def _stage_for_code(error_code: str) -> str | None:






    code = (error_code or "").lower()
    if not code or code == _short_code():
        return None
    for stage, markers in _STAGE_CODE_MARKERS:
        if any(marker in code for marker in markers):
            return stage
    return "other"


def _infer_stage(title: str, message: str) -> str:




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















    return "unspecified_error"
