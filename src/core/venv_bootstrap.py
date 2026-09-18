








from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess  # nosec B404
import sys
from typing import Callable

from qgis.core import Qgis

from . import install_config
from .cache_paths import PLUGIN_CACHE_DIR
from .logging_utils import log as _log
from .model_config import IS_ROSETTA
from .pip_diagnostics import get_ssl_error_help
from .pip_diagnostics import is_antivirus_error as _is_antivirus_error
from .pip_diagnostics import is_dll_init_error as _is_dll_init_error
from .pip_diagnostics import is_network_error as _is_network_error
from .pip_diagnostics import is_unable_to_create_process as _is_unable_to_create_process
from .pip_diagnostics import is_windows_process_crash as _is_windows_process_crash
from .subprocess_utils import run_unthrottled  # nosec B404


def _uv_would_choke_on(python_path: str) -> bool:








    if sys.platform != "win32" or " " not in python_path:
        return False
    return " " in _win_short_path(python_path)


def _get_qgis_python() -> str | None:








    if sys.platform != "win32":
        return None


    python_path = os.path.join(sys.prefix, "python.exe")
    if not os.path.exists(python_path):

        python_path = os.path.join(sys.prefix, "python3.exe")

    if not os.path.exists(python_path):
        _log("QGIS bundled Python not found at sys.prefix", Qgis.MessageLevel.Warning)
        return None


    try:






        env = _get_clean_env_for_venv()
        env["PYTHONIOENCODING"] = "utf-8"

        result = run_unthrottled(
            [python_path, "-c", "import sys; print(sys.version)"],
            text=True, encoding="utf-8", errors="replace", timeout=15,
            env=env, **_get_subprocess_kwargs(),
        )
        if result.returncode == 0:
            _log(f"QGIS Python verified: {result.stdout.strip()}", Qgis.MessageLevel.Info)
            return python_path
        _log(f"QGIS Python failed verification: {result.stderr}", Qgis.MessageLevel.Warning)
        return None
    except Exception as e:
        _log(f"QGIS Python verification error: {e}", Qgis.MessageLevel.Warning)
        return None


def _system_python_matches_target(python3_path: str) -> bool:







    from .python_manager import get_qgis_python_version
    try:
        env = _get_clean_env_for_venv()
        result = run_unthrottled(
            [python3_path, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            text=True, encoding="utf-8", errors="replace", timeout=15,
            env=env, **_get_subprocess_kwargs(),
        )
        if result.returncode != 0:
            return False
        parts = result.stdout.strip().split()
        if len(parts) != 2:
            return False
        installed = (int(parts[0]), int(parts[1]))
        return installed == get_qgis_python_version()
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        _log(f"System Python version check failed: {e}", Qgis.MessageLevel.Warning)
        return False


def _fallback_python_for_platform() -> str | None:








    if sys.platform == "win32":
        return _get_qgis_python()

    candidates: list[str] = []
    if sys.platform == "darwin":
        for prefix in (sys.prefix, sys.base_prefix):
            candidates.append(os.path.join(prefix, "bin", "python3"))
    else:




        versioned = f"python{sys.version_info.major}.{sys.version_info.minor}"
        for name in (versioned, "python3"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
            candidates.append(os.path.join("/usr/bin", name))

    for candidate in candidates:
        if os.path.exists(candidate) and _system_python_matches_target(candidate):
            return candidate
    return None


def _get_system_python() -> str:












    from .python_manager import (
        get_standalone_python_path,
        is_unsupported_windows,
        remove_standalone_python,
        standalone_python_exists,
        verify_standalone_python,
    )

    unsupported, why = is_unsupported_windows()
    if unsupported:
        raise RuntimeError(why)

    if standalone_python_exists():
        ok, msg = verify_standalone_python()
        if ok:
            python_path = get_standalone_python_path()
            _log(f"Using standalone Python: {python_path}", Qgis.MessageLevel.Info)
            return python_path
        _log(
            f"Standalone Python broken ({msg}), removing...",
            Qgis.MessageLevel.Warning
        )
        remove_standalone_python()









    if sys.platform in ("linux", "darwin"):
        fallback = _fallback_python_for_platform()
        if fallback:
            _log(f"Falling back to system Python: {fallback}", Qgis.MessageLevel.Info)
            return fallback


    if sys.platform == "win32":
        qgis_python = _get_qgis_python()
        if qgis_python:
            _log(
                "Standalone Python unavailable, using QGIS Python as fallback",
                Qgis.MessageLevel.Warning
            )
            return qgis_python


    raise RuntimeError(
        "Python standalone not installed. "
        "Please click 'Install Dependencies' to download Python automatically."
    )


def _ensurepip_missing_help(error_text: str) -> str:






    opening = tr("Failed to create venv: this Python is missing its venv support.\n\n")



    if not shutil.which("apt-get"):
        return (
            opening
            + tr(
                "Install the venv module for your Python with your system's "
                "package manager, then retry."
            )
        )
    match = re.search(r"(python3(?:\.\d+)?-venv)", error_text)
    package = match.group(1) if match else "python3-venv"
    return (
        opening
        + tr("Install it, then retry:\n")
        + f"    sudo apt install {package}"
    )




_PIP_BOOTSTRAP_MARKERS = ("ensurepip", "--default-pip", "_setup_pip", "install_pip")


def _venv_failed_on_pip_bootstrap(error_text: str) -> bool:





    lowered = (error_text or "").lower()
    return any(marker in lowered for marker in _PIP_BOOTSTRAP_MARKERS)


def _venv_failure_is_blocked(error_text: str) -> bool:





    text = error_text or ""
    if "errno 13" in text.lower():
        return True
    return _is_antivirus_error(text)


def _uv_binary_is_unusable(error_text: str, returncode: int | None = None) -> bool:









    if returncode is not None and _is_windows_process_crash(returncode):
        return True
    text = error_text or ""
    lowered = text.lower()
    return (
        _is_dll_init_error(text)
        or _is_unable_to_create_process(text)
        or "exec format error" in lowered
        or "is not recognized as an internal or external command" in lowered
    )


def _retry_venv_without_pip(
    system_python: str,
    venv_dir: str,
    env: dict,
    subprocess_kwargs: dict,
    cancel_check: Callable[[], bool] | None,
) -> str:

    _cleanup_partial_venv(venv_dir)
    _log("Retrying venv creation with --without-pip...", Qgis.MessageLevel.Warning)
    nopip_cmd = [system_python, "-m", "venv", "--without-pip", venv_dir]
    try:
        result = _run_with_cancel(nopip_cmd, 300, env, subprocess_kwargs, cancel_check)
        if cancel_check and cancel_check():
            return "cancelled"
        if result.returncode != 0:
            err = result.stderr or result.stdout or ""
            _log(f"Retry --without-pip failed: {err[:200]}", Qgis.MessageLevel.Warning)
            return "failed"
    except Exception as e:
        _log(f"Retry --without-pip exception: {e}", Qgis.MessageLevel.Warning)
        return "failed"

    installed, cancelled = _bootstrap_pip_in_venv(
        venv_dir, env, subprocess_kwargs, cancel_check)
    if cancelled:
        return "cancelled"
    return "ok" if installed else "failed"


def _win_long_paths_state() -> str:






    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            value, _kind = winreg.QueryValueEx(key, "LongPathsEnabled")
        return "on" if int(value) == 1 else "off"
    except (OSError, ValueError, ImportError):
        return "unknown"


def _log_system_info():

    try:
        qgis_version = Qgis.QGIS_VERSION
    except Exception:
        qgis_version = "Unknown"

    custom_cache = os.environ.get("AI_SEGMENTATION_CACHE_DIR")
    info_lines = [
        "=" * 50,
        "Installation Environment:",
        f"  OS: {sys.platform} ({platform.system()} {platform.release()})",
        "  Architecture: {}{}".format(
            platform.machine(),
            " (Rosetta on Apple Silicon)" if IS_ROSETTA else ""
        ),
        f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"  QGIS: {qgis_version}",
        f"  Install dir: {PLUGIN_CACHE_DIR}",
    ]
    if custom_cache:
        info_lines.append("  (via AI_SEGMENTATION_CACHE_DIR)")
    if sys.platform == "win32":
        info_lines.append(f"  Long paths: {_win_long_paths_state()}, "
                          f"venv path length {len(VENV_DIR)}")
    info_lines.append("=" * 50)
    for line in info_lines:
        _log(line, Qgis.MessageLevel.Info)


def _check_rosetta_warning() -> str | None:




    if not IS_ROSETTA:
        return None

    return (
        "Rosetta detected: QGIS is running as x86_64 on Apple Silicon. "
        "Installing native ARM64 Python 3.10+ for the AI engine."
    )




_QT_NETWORK_ERROR_PATTERNS = (
    "connection refused",
    "connection closed",
    "remote host closed",
    "unreachable",
    "ssl handshake failed",
    "temporary network failure",
    "firewall or proxy",
)


_QT_HOST_NOT_FOUND_RE = re.compile(r"\bhost\b.{0,120}?\bnot found\b", re.IGNORECASE | re.DOTALL)


def _is_download_network_error(text: str) -> bool:








    if _is_network_error(text):
        return True
    if _QT_HOST_NOT_FOUND_RE.search(text):
        return True
    lower = text.lower()
    return any(
        p in lower
        for p in install_config.classifier_markers(
            "qt_network", _QT_NETWORK_ERROR_PATTERNS))


def _check_gdal_available() -> tuple[bool, str]:












    if sys.platform != "darwin":
        return True, ""

    try:
        result = subprocess.run(  # nosec B603
            ["gdal-config", "--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
        )
        if result.returncode == 0:
            return True, f"GDAL {result.stdout.strip()} found"
        return False, ""
    except FileNotFoundError:




        return False, ""
    except Exception:
        return True, ""


def _get_ssl_error_help(error_text: str = "") -> str:

    return get_ssl_error_help(error_text, cache_dir=PLUGIN_CACHE_DIR)








from .venv_deps import (  # noqa: E402
    tr,
)
from .venv_install import (  # noqa: E402
    _bootstrap_pip_in_venv,
)
from .venv_paths import (  # noqa: E402
    VENV_DIR,
    _cleanup_partial_venv,
    _win_short_path,
)
from .venv_subprocess import (  # noqa: E402
    _get_clean_env_for_venv,
    _get_subprocess_kwargs,
    _run_with_cancel,
)
