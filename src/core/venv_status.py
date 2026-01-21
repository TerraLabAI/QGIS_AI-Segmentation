








from __future__ import annotations

import os
import re
import subprocess  # nosec B404
import sys

from qgis.core import Qgis

from .install_degraded import degraded_packages as recorded_degraded_packages
from .logging_utils import log as _log
from .subprocess_utils import run_unthrottled  # nosec B404


def _quick_check_packages(venv_dir: str | None = None) -> tuple[bool, str]:






    if venv_dir is None:
        venv_dir = VENV_DIR

    site_packages = get_venv_site_packages(venv_dir)
    if not os.path.exists(site_packages):
        _log(f"Quick check: site-packages not found: {site_packages}",
             Qgis.MessageLevel.Warning)
        return False, "site-packages directory not found"











    package_markers = {
        name: name.replace("-", "_")
        for name, _spec in resolved_packages()
        if name != "setuptools" and name not in MANUAL_ONLY_PACKAGES
    }



    try:
        dist_names = _installed_dist_names(site_packages)
    except OSError as e:
        _log(f"Quick check: cannot list site-packages: {e}",
             Qgis.MessageLevel.Warning)
        return False, "site-packages directory not readable"

    for package_name, dir_name in package_markers.items():
        pkg_dir = os.path.join(site_packages, dir_name)
        if not os.path.exists(pkg_dir):
            _log(f"Quick check: {package_name} not found at {pkg_dir}", Qgis.MessageLevel.Warning)
            return False, f"Package {package_name} not found"
        if _normalize_dist_name(package_name) not in dist_names:
            _log(
                f"Quick check: {package_name} has no dist-info in "
                f"{site_packages} (broken install, e.g. antivirus quarantine)",
                Qgis.MessageLevel.Warning,
            )
            return False, f"Package {package_name} is damaged"

    _log(f"Quick check: all packages found in {site_packages}",
         Qgis.MessageLevel.Info)
    return True, "All packages found"


def _packages_missing_from_venv(venv_dir: str, package_names: list[str]) -> list[str]:










    try:
        site_packages = get_venv_site_packages(venv_dir)
        dist_names = _installed_dist_names(site_packages)
    except OSError as e:
        _log(f"Cannot read the environment site-packages: {e}", Qgis.MessageLevel.Warning)
        return list(package_names)
    except Exception as e:
        _log(f"Package placement check skipped: {e}", Qgis.MessageLevel.Warning)
        return []
    return [name for name in package_names
            if _normalize_dist_name(name) not in dist_names]


def local_model_ready(venv_dir: str | None = None) -> tuple[bool, str]:






    from .local_model_cache import cached_answer, remember

    site_packages = get_venv_site_packages(venv_dir or VENV_DIR)
    held = cached_answer(site_packages)
    if held is not None:
        return held
    answer = _probe_local_model(venv_dir)
    remember(site_packages, answer)
    return answer


def _probe_local_model(venv_dir: str | None = None) -> tuple[bool, str]:












    if venv_dir is None:
        venv_dir = VENV_DIR
    site_packages = get_venv_site_packages(venv_dir)
    if not os.path.exists(site_packages):
        return False, "site-packages directory not found"







    try:
        dist_names = _installed_dist_names(site_packages)
    except OSError as e:
        _log(f"Local model check: cannot list site-packages: {e}",
             Qgis.MessageLevel.Warning)
        return False, "site-packages directory not readable"
    for name, _spec in resolved_packages():
        if name not in MANUAL_ONLY_PACKAGES:
            continue
        if not os.path.exists(os.path.join(site_packages, name.replace("-", "_"))):
            return False, f"Package {name} not found"
        if _normalize_dist_name(name) not in dist_names:
            _log(
                f"Local model check: {name} has no dist-info in "
                f"{site_packages} (broken install, e.g. antivirus quarantine)",
                Qgis.MessageLevel.Warning,
            )
            return False, f"Package {name} is damaged"
    return True, "Local model packages found"


_MISSING_MODULE_RE = re.compile(r"No module named '([A-Za-z_][\w.]*)'")


def package_missing_behind_error(error_text: str, venv_dir: str = None) -> str | None:










    modules = {name.split(".", 1)[0] for name in _MISSING_MODULE_RE.findall(error_text or "")}
    if not modules:
        return None
    if venv_dir is None:
        venv_dir = VENV_DIR
    site_packages = get_venv_site_packages(venv_dir)
    try:
        dist_names = _installed_dist_names(site_packages)
    except OSError:
        return None
    for name, _spec in resolved_packages():
        dir_name = name.replace("-", "_")
        if dir_name not in modules:
            continue
        package_dir = os.path.join(site_packages, dir_name)
        if (not os.path.isfile(os.path.join(package_dir, "__init__.py"))
                or _normalize_dist_name(name) not in dist_names):
            return name
    return None


def _fallback_python_answers(allow_subprocess_probe: bool) -> bool:







    if not allow_subprocess_probe and sys.platform == "win32":
        return any(os.path.isfile(os.path.join(sys.prefix, name))
                   for name in ("python.exe", "python3.exe"))
    return bool(_fallback_python_for_platform())


def get_venv_status(allow_subprocess_probe: bool = True) -> tuple[bool, str]:









    from .python_manager import get_python_full_version, standalone_python_exists


    if os.path.exists(LIBS_DIR):
        _log(f"get_venv_status: old libs/ detected at {LIBS_DIR}",
             Qgis.MessageLevel.Warning)
        return False, "Old installation detected. Migration required."



    if _install_marker_present():
        _log("get_venv_status: previous installation was interrupted",
             Qgis.MessageLevel.Warning)
        return False, "Previous installation was interrupted"




    if not standalone_python_exists():




        if not (venv_exists() and _fallback_python_answers(allow_subprocess_probe)):
            _log("get_venv_status: standalone Python not found", Qgis.MessageLevel.Info)
            return False, "Dependencies not installed"


    if not venv_exists():
        _log(f"get_venv_status: venv not found at {VENV_DIR}",
             Qgis.MessageLevel.Info)
        return False, "Virtual environment not configured"




    base_ok, base_msg = _venv_base_python_ok()
    if not base_ok:
        _log(f"get_venv_status: {base_msg}", Qgis.MessageLevel.Warning)
        return False, "Python runtime is damaged. Reinstall required."


    is_present, msg = _quick_check_packages()
    if is_present:

        stored_hash = _read_deps_hash()
        current_hash = _compute_deps_hash()
        if stored_hash is not None and stored_hash != current_hash:
            _log(
                "get_venv_status: deps hash mismatch "
                f"(stored={stored_hash}, current={current_hash})",
                Qgis.MessageLevel.Warning
            )
            return False, "Dependencies need updating"
        if stored_hash is None:







            python_path = get_venv_python_path()
            if python_path:
                if not allow_subprocess_probe:







                    _log(
                        "get_venv_status: skipping the import probe "
                        "(caller cannot block), reporting unverified ready",
                        Qgis.MessageLevel.Info
                    )
                    unverified_version = get_python_full_version()
                    return True, f"Ready (Python {unverified_version}, not verified)"
                try:
                    env = _get_clean_env_for_venv()
                    kwargs = _get_subprocess_kwargs()






                    torch_timeout = _get_verification_timeout("torch")
                    try:
                        probe = run_unthrottled(
                            [python_path, "-c", "import torch"],
                            text=True,
                            encoding="utf-8", errors="replace",
                            timeout=torch_timeout, env=env, **kwargs,
                        )
                    except subprocess.TimeoutExpired:



                        _log(
                            "get_venv_status: torch probe timed out, retrying once",
                            Qgis.MessageLevel.Warning
                        )
                        probe = run_unthrottled(
                            [python_path, "-c", "import torch"],
                            text=True,
                            encoding="utf-8", errors="replace",
                            timeout=torch_timeout, env=env, **kwargs,
                        )
                    if probe.returncode != 0:
                        _log(
                            "get_venv_status: torch import failed "
                            "(possible DLL error), marking as incomplete",
                            Qgis.MessageLevel.Warning
                        )
                        return False, "Package verification failed (torch import error)"
                except Exception as exc:
                    _log(
                        f"get_venv_status: torch probe failed: {exc}",
                        Qgis.MessageLevel.Warning
                    )
                    return False, "Package verification failed"
            _log(
                "get_venv_status: no deps hash file, "
                "writing current hash (packages already present)",
                Qgis.MessageLevel.Info
            )
            _write_deps_hash()
        python_version = get_python_full_version()



        short = recorded_degraded_packages(VENV_DIR)
        if short:
            _log("get_venv_status: ready, short of " + ", ".join(short),
                 Qgis.MessageLevel.Warning)
            return True, f"Ready (Python {python_version}, without {', '.join(short)})"
        _log("get_venv_status: ready (quick check passed)", Qgis.MessageLevel.Success)
        return True, f"Ready (Python {python_version})"
    _log(f"get_venv_status: quick check failed: {msg}",
         Qgis.MessageLevel.Warning)
    return False, f"Virtual environment incomplete: {msg}"








from .venv_bootstrap import (  # noqa: E402
    _fallback_python_for_platform,
)
from .venv_deps import (  # noqa: E402
    MANUAL_ONLY_PACKAGES,
    _compute_deps_hash,
    _get_verification_timeout,
    _install_marker_present,
    _installed_dist_names,
    _normalize_dist_name,
    _read_deps_hash,
    _write_deps_hash,
    resolved_packages,
)
from .venv_paths import (  # noqa: E402
    LIBS_DIR,
    VENV_DIR,
    get_venv_python_path,
    get_venv_site_packages,
    venv_exists,
)
from .venv_repair import (  # noqa: E402
    _venv_base_python_ok,
)
from .venv_subprocess import (  # noqa: E402
    _get_clean_env_for_venv,
    _get_subprocess_kwargs,
)
