







from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys

from qgis.core import Qgis

from . import install_config
from .cache_paths import PLUGIN_CACHE_DIR
from .logging_utils import log as _log
from .model_config import SAM_PACKAGE, TORCH_MIN, TORCHVISION_MIN
from .pip_diagnostics import is_antivirus_error as _is_antivirus_error
from .pip_diagnostics import is_app_control_error as _is_app_control_error
from .pip_diagnostics import is_disk_full as _is_disk_full
from .pip_diagnostics import is_dll_init_error as _is_dll_init_error
from .pip_diagnostics import is_file_locked_error as _is_file_locked_error
from .pip_diagnostics import is_rename_or_record_error as _is_rename_or_record_error
from .pip_diagnostics import is_windows_process_crash as _is_windows_process_crash
from .venv_paths import (
    DEPS_HASH_FILE,
    INSTALL_MARKER_FILE,
    VENV_DIR,
    get_venv_site_packages,
)


def _numpy_version_spec() -> str:





    if sys.version_info >= (3, 13):
        return ">=2.0.0,<3.0.0"
    return ">=1.26.0,<2.0.0"


def _with_upper_bound(spec: str, cap: str) -> str:









    if "<" in spec:
        return spec
    return f"{spec},{cap}"


REQUIRED_PACKAGES = [
    ("setuptools", ">=70.0,<100.0"),
    ("numpy", _numpy_version_spec()),



    ("rasterio", ">=1.3.0,<2.0.0"),
    ("torch", _with_upper_bound(TORCH_MIN, "<3.0.0")),
    ("torchvision", _with_upper_bound(TORCHVISION_MIN, "<1.0.0")),
    (SAM_PACKAGE[0], _with_upper_bound(SAM_PACKAGE[1], "<2.0.0")),
]








MANUAL_ONLY_PACKAGES = {"torch", "torchvision", SAM_PACKAGE[0]}





NON_ESSENTIAL_PACKAGES = {"pandas"}



TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"


PIP_RETRIES = 10

PIP_TIMEOUT_S = 30




UV_HTTP_TIMEOUT_S = 300

UV_HTTP_RETRIES = 5




VERIFY_TIMEOUT_TORCH_S = 120

VERIFY_TIMEOUT_HEAVY_S = 180

VERIFY_TIMEOUT_DEFAULT_S = 30



PACKAGE_TIMEOUTS_S = {"torch": 5400, "torchvision": 1200}

PACKAGE_TIMEOUT_DEFAULT_S = 600



NETWORK_RETRY_ATTEMPTS = 4

NETWORK_RETRY_BACKOFF_S = 5



































_INSTALL_LOGIC_VERSION = "7"


MIN_FREE_GB_FULL = 5.0




MIN_FREE_GB_AUTOMATIC = 1.5


def tr(text: str) -> str:






    try:
        from .i18n import tr as translate

        return translate(text)
    except Exception:  # noqa: BLE001
        return text


def resolved_min_free_gb_full() -> float:

    try:
        return install_config.min_free_gb_full(MIN_FREE_GB_FULL)
    except Exception:  # noqa: BLE001
        return MIN_FREE_GB_FULL


def resolved_min_free_gb_automatic() -> float:

    try:
        return install_config.min_free_gb_automatic(MIN_FREE_GB_AUTOMATIC)
    except Exception:  # noqa: BLE001
        return MIN_FREE_GB_AUTOMATIC


def packages_skipped_for_disk_space() -> set[str]:





    try:
        os.makedirs(PLUGIN_CACHE_DIR, exist_ok=True)
        free_gb = shutil.disk_usage(PLUGIN_CACHE_DIR).free / (1024 ** 3)
    except OSError:
        return set()


    if free_gb < 0.001 or free_gb >= resolved_min_free_gb_full():
        return set()
    return set(MANUAL_ONLY_PACKAGES) - _manual_packages_already_installed()


def _manual_packages_already_installed() -> set[str]:







    try:
        site_packages = get_venv_site_packages()
        if not os.path.isdir(site_packages):
            return set()
        installed = _installed_dist_names(site_packages)
    except OSError:
        return set()
    return {
        name for name in MANUAL_ONLY_PACKAGES
        if _normalize_dist_name(name) in installed
    }


def resolved_packages() -> list[tuple[str, str]]:







    try:
        return install_config.package_specs(REQUIRED_PACKAGES)
    except Exception:  # noqa: BLE001
        return list(REQUIRED_PACKAGES)


def _resolved_install_logic_version() -> str:

    try:
        return install_config.install_logic_version(_INSTALL_LOGIC_VERSION)
    except Exception:  # noqa: BLE001
        return _INSTALL_LOGIC_VERSION


def _compute_deps_hash() -> str:








    data = repr(resolved_packages()).encode("utf-8")
    data += _resolved_install_logic_version().encode("utf-8")
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _read_deps_hash() -> str | None:

    try:
        with open(DEPS_HASH_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _write_deps_hash():

    try:
        hash_dir = os.path.dirname(DEPS_HASH_FILE)
        os.makedirs(hash_dir, exist_ok=True)
        tmp_path = DEPS_HASH_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(_compute_deps_hash())
        os.replace(tmp_path, DEPS_HASH_FILE)
    except OSError as e:
        _log(f"Failed to write deps hash: {e}", Qgis.MessageLevel.Warning)


def _write_install_marker():

    try:
        with open(INSTALL_MARKER_FILE, "w", encoding="utf-8") as f:
            f.write(os.path.basename(VENV_DIR))
    except OSError as e:
        _log(f"Could not write install marker: {e}", Qgis.MessageLevel.Warning)


def _clear_install_marker():
    try:
        os.unlink(INSTALL_MARKER_FILE)
    except OSError:
        pass  # nosec B110


def mark_venv_for_rebuild():







    _write_install_marker()


def _install_marker_present() -> bool:

    try:
        with open(INSTALL_MARKER_FILE, encoding="utf-8") as f:
            return f.read().strip() == os.path.basename(VENV_DIR)
    except OSError:
        return False


def _failure_is_machine_level(error_text: str, returncode: int | None) -> bool:









    if returncode is not None and _is_windows_process_crash(returncode):
        return True
    checks = [_is_file_locked_error, _is_disk_full,
              _is_app_control_error, _is_antivirus_error]
    if sys.platform == "win32":
        checks += [_is_dll_init_error, _is_rename_or_record_error]
    return any(check(error_text) for check in checks)


def _normalize_dist_name(name: str) -> str:

    return re.sub(r"[-_.]+", "_", name).lower()


def _installed_dist_names(site_packages: str) -> set:









    return {
        _normalize_dist_name(entry.split("-", 1)[0])
        for entry in os.listdir(site_packages)
        if entry.endswith(".dist-info")
    }


def _get_verification_timeout(package_name: str) -> int:










    if package_name == "torch":
        shipped = VERIFY_TIMEOUT_TORCH_S
    elif package_name in ("torchvision", "pandas"):


        shipped = VERIFY_TIMEOUT_HEAVY_S
    else:
        shipped = VERIFY_TIMEOUT_DEFAULT_S
    return install_config.verify_timeout_s(package_name, shipped)
