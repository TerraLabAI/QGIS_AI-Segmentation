







from __future__ import annotations

import os
import shutil
from typing import Callable

from qgis.core import Qgis

from .cache_paths import PLUGIN_CACHE_DIR
from .install_lock import InstallLock, lock_age_seconds
from .logging_utils import log as _log
from .venv_deps import (
    _clear_install_marker,
    _install_marker_present,
    _write_install_marker,
    resolved_min_free_gb_automatic,
    resolved_min_free_gb_full,
    tr,
)
from .venv_install import (
    _create_venv_and_install,
)
from .venv_paths import (
    INSTALL_LOCK_FILE,
    VENV_DIR,
    _cleanup_partial_venv,
    _clear_installer_caches,
    cleanup_old_libs,
    cleanup_old_venv_directories,
)


def create_venv_and_install(
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    include_local_model: bool = True,
) -> tuple[bool, str]:






















    lock = InstallLock(INSTALL_LOCK_FILE)
    if not lock.acquire():



        age = lock_age_seconds(INSTALL_LOCK_FILE)
        busy = tr("Another QGIS window is installing the AI components.")
        if age is not None and age >= 60:
            busy += " " + tr("It started {minutes} minutes ago.").format(
                minutes=int(age // 60))
        busy += " " + tr("Wait for it to finish, then try again.")
        _log(f"Install lock held elsewhere (age {int(age or 0)}s)",
             Qgis.MessageLevel.Warning)
        return False, busy

    try:














        min_free_gb = resolved_min_free_gb_full()
        min_free_gb_auto = resolved_min_free_gb_automatic()
        if not include_local_model:



            min_free_gb = min_free_gb_auto
        try:
            os.makedirs(PLUGIN_CACHE_DIR, exist_ok=True)
            free_gb = shutil.disk_usage(PLUGIN_CACHE_DIR).free / (1024 ** 3)
        except OSError:

            free_gb = None







        if free_gb is not None and free_gb < 0.001:
            _log(
                f"Disk-space probe read ~0 GB free at {PLUGIN_CACHE_DIR}; treating it as "
                "unmeasurable (likely a quota'd or redirected path) and "
                "continuing.",
                Qgis.MessageLevel.Warning,
            )
            free_gb = None
        if free_gb is not None and free_gb < min_free_gb:





            removed = cleanup_old_venv_directories()
            cleanup_old_libs()




            _clear_installer_caches(include_tmp=True)
            _log(
                f"Preflight short on space, removed {len(removed)} old "
                "venv(s), cleared the installer caches and re-measured.",
                Qgis.MessageLevel.Info)
            try:
                free_gb = shutil.disk_usage(PLUGIN_CACHE_DIR).free / (1024 ** 3)
            except OSError:
                free_gb = None
        if free_gb is not None and free_gb < min_free_gb_auto:
            hint = tr(
                "Not enough free disk space to install dependencies: "
                "{free_gb:.1f} GB available at {cache_dir}, "
                "at least {min_free_gb:.1f} GB is required.\n\n"
                "Free up disk space, or set the AI_SEGMENTATION_CACHE_DIR "
                "environment variable to a directory on a larger drive, "
                "then restart QGIS."
            ).format(free_gb=free_gb, cache_dir=PLUGIN_CACHE_DIR, min_free_gb=min_free_gb_auto)
            _log(hint, Qgis.MessageLevel.Critical)
            return False, hint
        if free_gb is not None and free_gb < min_free_gb:
            _log(
                tr(
                    "{free_gb:.1f} GB free at {cache_dir}, under the "
                    "{min_free_gb:.0f} GB the local model needs. Installing the "
                    "Automatic packages only. Free up space and install again to "
                    "turn Semi-Auto mode on."
                ).format(free_gb=free_gb, cache_dir=PLUGIN_CACHE_DIR, min_free_gb=min_free_gb),
                Qgis.MessageLevel.Warning,
            )





        if _install_marker_present():
            _log(
                "Previous installation was interrupted, "
                "recreating the virtual environment...",
                Qgis.MessageLevel.Warning)
            if not _cleanup_partial_venv(VENV_DIR):
                from .pip_diagnostics import get_file_locked_help
                return False, get_file_locked_help()

        _write_install_marker()






        ok, message = _create_venv_and_install(
            progress_callback, cancel_check, include_local_model)
        if ok:
            _clear_install_marker()
        return ok, message
    finally:
        lock.release()
