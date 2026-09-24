










from __future__ import annotations

import os
import subprocess  # nosec B404
import sys
import tempfile
import time
from typing import Callable

from qgis.core import Qgis

from . import install_config
from .cache_paths import PLUGIN_CACHE_DIR
from .install_degraded import record_degraded
from .install_progress_text import install_display_name
from .logging_utils import log as _log
from .model_config import SAM_PACKAGE
from .pip_diagnostics import first_run_hosts_sentence as _first_run_hosts_sentence
from .pip_diagnostics import get_app_control_help as _get_app_control_help
from .pip_diagnostics import get_corrupt_venv_help as _get_corrupt_venv_help
from .pip_diagnostics import get_crash_help as _get_crash_help
from .pip_diagnostics import get_disk_full_help as _get_disk_full_help
from .pip_diagnostics import get_file_locked_help as _get_file_locked_help
from .pip_diagnostics import get_glibc_too_old_help as _get_glibc_too_old_help
from .pip_diagnostics import get_macos_intel_help as _get_macos_intel_help
from .pip_diagnostics import get_pip_antivirus_help as _get_pip_antivirus_help
from .pip_diagnostics import get_pip_ssl_bypass_flags as _get_pip_ssl_bypass_flags
from .pip_diagnostics import get_vcpp_help as _get_vcpp_help
from .pip_diagnostics import is_antivirus_error as _is_antivirus_error
from .pip_diagnostics import is_app_control_error as _is_app_control_error
from .pip_diagnostics import is_disk_full as _is_disk_full
from .pip_diagnostics import is_dll_init_error as _is_dll_init_error
from .pip_diagnostics import is_file_locked_error as _is_file_locked_error
from .pip_diagnostics import is_glibc_too_old as _is_glibc_too_old
from .pip_diagnostics import is_hash_mismatch as _is_hash_mismatch
from .pip_diagnostics import is_index_forbidden_error as _is_index_forbidden_error
from .pip_diagnostics import is_macos_intel_no_wheel as _is_macos_intel_no_wheel
from .pip_diagnostics import is_network_error as _is_network_error
from .pip_diagnostics import is_proxy_auth_error as _is_proxy_auth_error
from .pip_diagnostics import is_rename_or_record_error as _is_rename_or_record_error
from .pip_diagnostics import is_ssl_error as _is_ssl_error
from .pip_diagnostics import is_unable_to_create_process as _is_unable_to_create_process
from .pip_diagnostics import is_windows_process_crash as _is_windows_process_crash
from .uv_manager import download_uv, get_uv_path, remove_uv, uv_exists, verify_uv
from .venv_network import (
    _get_pip_proxy_args,
    _insecure_install_opt_in,
    env_without_ca_bundle_overrides,
    windows_trust_store_bundle,
)


_uv_available = False

_uv_path: str | None = None

_uv_probe = {"done": False}


def _ensure_uv_state() -> None:








    global _uv_available, _uv_path
    if _uv_available or _uv_probe["done"]:
        return
    _uv_probe["done"] = True
    try:
        if uv_exists() and verify_uv():
            _uv_available = True
            _uv_path = get_uv_path()
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def _drop_uv(reason: str) -> None:

    global _uv_available, _uv_path
    _log(f"Removing the uv binary ({reason})", Qgis.MessageLevel.Warning)
    remove_uv()
    _uv_available = False
    _uv_path = None


def _build_install_cmd(python_path: str, pip_args: list) -> list:







    _ensure_uv_state()
    if _uv_available and _uv_path and not _uv_would_choke_on(python_path):
        cmd = [_uv_path, "pip"]
        skip_next = False
        for i, arg in enumerate(pip_args):
            if skip_next:
                skip_next = False
                continue
            if arg == "--disable-pip-version-check" or arg == "--no-warn-script-location" or arg == "--prefer-binary":
                continue
            if arg in ("--retries", "--timeout"):
                skip_next = True
                continue
            if arg == "--no-cache-dir":
                cmd.append("--no-cache")
                continue
            if arg == "--force-reinstall":
                cmd.append("--reinstall")
                continue
            if arg == "--trusted-host":
                cmd.append("--allow-insecure-host")
                continue
            if arg == "--proxy":

                skip_next = True
                continue
            if arg == "--constraint" and i + 1 < len(pip_args):







                cmd.append("numpy" + _numpy_version_spec())
                skip_next = True
                continue
            cmd.append(arg)
        cmd.extend(["--python", _win_short_path(python_path)])
        return cmd






    return [python_path, "-m", "pip"] + pip_args


def create_venv(
    venv_dir: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    _log(f"Creating virtual environment at: {venv_dir}", Qgis.MessageLevel.Info)

    if progress_callback:
        progress_callback(10, tr("Creating virtual environment..."))

    system_python = _get_system_python()
    _log(f"Using Python: {system_python}", Qgis.MessageLevel.Info)



    env = _get_clean_env_for_venv()


    if _uv_available and _uv_path:
        _log("Creating venv with uv...", Qgis.MessageLevel.Info)



        uv_python = _win_short_path(_win_long_path(system_python))
        uv_cmd = [_uv_path, "venv", "--python", uv_python, venv_dir]
        try:
            subprocess_kwargs = _get_subprocess_kwargs()

            result = _run_with_cancel(
                uv_cmd, 120, env, subprocess_kwargs, cancel_check)
            if cancel_check and cancel_check():
                _cleanup_partial_venv(venv_dir)
                return False, "Installation cancelled"
            if result.returncode == 0:
                _log("Virtual environment created with uv", Qgis.MessageLevel.Success)
                if progress_callback:
                    progress_callback(20, tr("Virtual environment created (uv)"))
                return True, "Virtual environment created"
            error_msg = result.stderr or result.stdout or ""
            _log(f"uv venv creation failed: {error_msg[:200]}", Qgis.MessageLevel.Warning)
            _cleanup_partial_venv(venv_dir)



            if _uv_binary_is_unusable(error_msg, result.returncode):
                _drop_uv("it could not run")
            _log("Falling back to python -m venv", Qgis.MessageLevel.Warning)
        except Exception as e:
            _log(f"uv venv exception: {e}, falling back to python -m venv", Qgis.MessageLevel.Warning)
            _cleanup_partial_venv(venv_dir)


            if isinstance(e, OSError) or _uv_binary_is_unusable(str(e)):
                _drop_uv("it could not be started")


    cmd = [system_python, "-m", "venv", venv_dir]
    try:
        subprocess_kwargs = _get_subprocess_kwargs()

        result = _run_with_cancel(cmd, 300, env, subprocess_kwargs, cancel_check)
        if cancel_check and cancel_check():
            _cleanup_partial_venv(venv_dir)
            return False, "Installation cancelled"

        if result.returncode == 0:
            _log("Virtual environment created successfully", Qgis.MessageLevel.Success)


            pip_path = get_venv_pip_path(venv_dir)
            if not os.path.exists(pip_path):
                _log("pip not found in venv, bootstrapping with ensurepip...", Qgis.MessageLevel.Info)
                python_in_venv = get_venv_python_path(venv_dir)
                ensurepip_cmd = [python_in_venv, "-m", "ensurepip", "--upgrade"]
                ensurepip_ok = False
                try:
                    ensurepip_result = _run_with_cancel(
                        ensurepip_cmd, 120, env, subprocess_kwargs, cancel_check)
                    if cancel_check and cancel_check():
                        _cleanup_partial_venv(venv_dir)
                        return False, "Installation cancelled"
                    if ensurepip_result.returncode == 0:
                        _log("pip bootstrapped via ensurepip", Qgis.MessageLevel.Success)
                        ensurepip_ok = True
                    else:
                        err = ensurepip_result.stderr or ensurepip_result.stdout or ""
                        _log(f"ensurepip failed: {err[:200]}",
                             Qgis.MessageLevel.Warning)
                except Exception as e:
                    _log(f"ensurepip exception: {e}",
                         Qgis.MessageLevel.Warning)

                if not ensurepip_ok:


                    if _uv_available and _uv_path:
                        _log(
                            "ensurepip unavailable but uv is present, "
                            "continuing without pip",
                            Qgis.MessageLevel.Warning)
                    else:
                        _cleanup_partial_venv(venv_dir)
                        return False, (
                            "Failed to bootstrap pip (ensurepip unavailable). "
                            "This often happens with Anaconda Python."
                        )

            if progress_callback:
                progress_callback(20, tr("Virtual environment created"))
            return True, "Virtual environment created"
        error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
        _log(f"Failed to create venv: {error_msg}", Qgis.MessageLevel.Critical)
        _cleanup_partial_venv(venv_dir)



        if "ensurepip is not available" in error_msg.lower():
            failure_message = _ensurepip_missing_help(error_msg)
        else:
            failure_message = f"Failed to create venv: {error_msg[:600]}"




        if (_venv_failed_on_pip_bootstrap(error_msg)
                and not _venv_failure_is_blocked(error_msg)):
            outcome = _retry_venv_without_pip(
                system_python, venv_dir, env, subprocess_kwargs, cancel_check)
            if outcome == "cancelled":
                _cleanup_partial_venv(venv_dir)
                return False, "Installation cancelled"
            if outcome == "ok":
                _log("Virtual environment created without the built-in pip step",
                     Qgis.MessageLevel.Success)
                if progress_callback:
                    progress_callback(20, tr("Virtual environment created"))
                return True, "Virtual environment created"
            _cleanup_partial_venv(venv_dir)
        return False, failure_message

    except subprocess.TimeoutExpired:
        _log("Venv creation timed out, retrying with --without-pip...", Qgis.MessageLevel.Warning)
        _cleanup_partial_venv(venv_dir)

        try:
            nopip_cmd = [system_python, "-m", "venv", "--without-pip", venv_dir]
            result2 = _run_with_cancel(
                nopip_cmd, 300, env, subprocess_kwargs, cancel_check)
            if cancel_check and cancel_check():
                _cleanup_partial_venv(venv_dir)
                return False, "Installation cancelled"
            if result2.returncode == 0:
                _log("Venv created (--without-pip), bootstrapping pip...", Qgis.MessageLevel.Info)
                python_in_venv = get_venv_python_path(venv_dir)
                ensurepip_cmd = [python_in_venv, "-m", "ensurepip", "--upgrade"]
                ep_result = _run_with_cancel(
                    ensurepip_cmd, 120, env, subprocess_kwargs, cancel_check)
                if cancel_check and cancel_check():
                    _cleanup_partial_venv(venv_dir)
                    return False, "Installation cancelled"
                if ep_result.returncode == 0:
                    _log("pip bootstrapped via ensurepip", Qgis.MessageLevel.Success)
                    if progress_callback:
                        progress_callback(20, tr("Virtual environment created"))
                    return True, "Virtual environment created"
                err = ep_result.stderr or ep_result.stdout or ""
                _log(f"ensurepip failed: {err[:200]}", Qgis.MessageLevel.Warning)
                if _uv_available and _uv_path:
                    _log(
                        "ensurepip unavailable but uv present, "
                        "continuing without pip",
                        Qgis.MessageLevel.Warning)
                    if progress_callback:
                        progress_callback(20, tr("Virtual environment created"))
                    return True, "Virtual environment created"
                _cleanup_partial_venv(venv_dir)
                return False, f"Failed to bootstrap pip: {err[:200]}"
            err = result2.stderr or result2.stdout or ""
            _log(f"Retry --without-pip failed: {err[:200]}", Qgis.MessageLevel.Critical)
            _cleanup_partial_venv(venv_dir)
            return False, "Virtual environment creation timed out"
        except Exception as e2:
            _log(f"Retry --without-pip exception: {e2}", Qgis.MessageLevel.Critical)
            _cleanup_partial_venv(venv_dir)
            return False, "Virtual environment creation timed out"
    except FileNotFoundError:
        _log(f"Python executable not found: {system_python}", Qgis.MessageLevel.Critical)
        return False, f"Python not found: {system_python}"
    except Exception as e:
        _log(f"Exception during venv creation: {str(e)}", Qgis.MessageLevel.Critical)
        _cleanup_partial_venv(venv_dir)
        return False, f"Error: {str(e)[:200]}"


def _bootstrap_pip_in_venv(
    venv_dir: str,
    env: dict,
    subprocess_kwargs: dict,
    cancel_check: Callable[[], bool] | None,
) -> tuple[bool, bool]:






    python_in_venv = get_venv_python_path(venv_dir)

    if _uv_available and _uv_path and not _uv_would_choke_on(python_in_venv):
        uv_cmd = [_uv_path, "pip", "install", "pip",
                  "--python", _win_short_path(python_in_venv)]
        try:
            uv_result = _run_with_cancel(uv_cmd, 180, env, subprocess_kwargs, cancel_check)
            if cancel_check and cancel_check():
                return False, True
            if uv_result.returncode == 0:
                _log("pip bootstrapped with uv", Qgis.MessageLevel.Success)
                return True, False
            err = uv_result.stderr or uv_result.stdout or ""
            _log(f"uv could not bootstrap pip: {err[:200]}", Qgis.MessageLevel.Warning)
        except Exception as e:
            _log(f"uv pip bootstrap exception: {e}", Qgis.MessageLevel.Warning)

    ensurepip_cmd = [python_in_venv, "-m", "ensurepip", "--upgrade"]
    try:
        ep_result = _run_with_cancel(ensurepip_cmd, 120, env, subprocess_kwargs, cancel_check)
        if cancel_check and cancel_check():
            return False, True
        if ep_result.returncode == 0:
            _log("pip bootstrapped via ensurepip", Qgis.MessageLevel.Success)
            return True, False
        err = ep_result.stderr or ep_result.stdout or ""
        _log(f"ensurepip failed: {err[:200]}", Qgis.MessageLevel.Warning)
    except Exception as e:
        _log(f"ensurepip exception: {e}", Qgis.MessageLevel.Warning)
    return False, False


def install_dependencies(
    venv_dir: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    include_local_model: bool = True,
) -> tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment does not exist"

    pip_path = get_venv_pip_path(venv_dir)
    _log(f"Installing dependencies using: {pip_path}", Qgis.MessageLevel.Info)




    python_path_pre = get_venv_python_path(venv_dir)
    if _uv_available:
        _log("Using uv for installation, skipping pip upgrade", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(20, tr("Using uv package installer..."))
    else:
        if progress_callback:
            progress_callback(20, tr("Upgrading pip..."))
        try:
            _log("Upgrading pip to latest version...", Qgis.MessageLevel.Info)
            upgrade_cmd = [
                python_path_pre, "-m", "pip", "install",
                "--upgrade", "pip",
                "--disable-pip-version-check",
                "--no-warn-script-location",
            ]
            upgrade_result = _run_with_cancel(
                upgrade_cmd,
                timeout=120,
                env=_get_clean_env_for_venv(),
                subprocess_kwargs=_get_subprocess_kwargs(),
                cancel_check=cancel_check,
            )
            if upgrade_result.returncode == -1 and upgrade_result.stderr == "cancelled":
                return False, "Installation cancelled"
            if upgrade_result.returncode == 0:
                _log("pip upgraded successfully", Qgis.MessageLevel.Success)
            else:
                _log("pip upgrade failed (non-critical): {}".format(
                    (upgrade_result.stderr or upgrade_result.stdout or "")[:200]),
                    Qgis.MessageLevel.Warning)
        except Exception as e:
            _log(f"pip upgrade failed (non-critical): {str(e)[:200]}",
                 Qgis.MessageLevel.Warning)

    packages = resolved_packages()




    skipped_for_space = packages_skipped_for_disk_space()
    left_out = set(skipped_for_space)
    if not include_local_model:




        left_out |= set(MANUAL_ONLY_PACKAGES)
    if left_out:
        packages = [(n, s) for n, s in packages if n not in left_out]
    total_packages = len(packages)
    base_progress = 20
    progress_range = 80



    _wmap = {
        "numpy": 5, "torch": 30, "torchvision": 15,
        "rasterio": 10,
    }
    _weights = [_wmap.get(name, 10) for name, _ in packages]
    weight_total = sum(_weights)

    _cumulative = [0]
    for w in _weights:
        _cumulative.append(_cumulative[-1] + w)

    def _pkg_progress_start(idx):
        return base_progress + int(progress_range * _cumulative[idx] / weight_total)

    def _pkg_progress_end(idx):
        return base_progress + int(progress_range * _cumulative[idx + 1] / weight_total)

    python_path = get_venv_python_path(venv_dir)





    os.makedirs(PLUGIN_CACHE_DIR, exist_ok=True)
    constraints_fd, constraints_path_str = tempfile.mkstemp(
        suffix=".txt", prefix="pip_constraints_", dir=PLUGIN_CACHE_DIR
    )
    constraints_path: str | None = constraints_path_str
    try:
        with os.fdopen(constraints_fd, "w", encoding="utf-8") as f:
            if sys.version_info >= (3, 13):
                f.write("numpy<3.0.0\n")
            else:
                f.write("numpy<2.0.0\n")
        _log(f"Created pip constraints file: {constraints_path}", Qgis.MessageLevel.Info)
    except Exception as e:
        _log(f"Failed to write constraints file: {e}", Qgis.MessageLevel.Warning)

        try:
            os.close(constraints_fd)
        except Exception:
            pass  # nosec B110
        try:
            os.unlink(constraints_path_str)
        except Exception:
            pass  # nosec B110
        constraints_path = None

    try:




        learned_ssl_env: dict | None = None





        degraded_packages: list[str] = list(skipped_for_space)

        for i, (package_name, version_spec) in enumerate(packages):
            if cancel_check and cancel_check():
                _log("Installation cancelled by user", Qgis.MessageLevel.Warning)
                return False, "Installation cancelled"

            package_spec = f"{package_name}{version_spec}"
            pkg_start = _pkg_progress_start(i)
            pkg_end = _pkg_progress_end(i)

            if progress_callback:
                progress_callback(
                    pkg_start,
                    tr("Installing {package}... ({done}/{total})").format(
                        package=install_display_name(package_name),
                        done=i + 1, total=total_packages))

            _log(f"[{i + 1}/{total_packages}] Installing {package_spec}...", Qgis.MessageLevel.Info)

            pip_args = [
                "install",
                "--upgrade",
                "--no-warn-script-location",
                "--disable-pip-version-check",
                "--prefer-binary",





                "--retries", str(install_config.pip_retries(PIP_RETRIES)),
                "--timeout", str(install_config.pip_timeout_s(PIP_TIMEOUT_S)),
            ]




            if _builds_from_source(package_name):
                pip_args.append("--no-build-isolation")


            if package_name == "rasterio":
                pip_args.extend(["--only-binary", "rasterio"])



            if constraints_path:
                pip_args.extend(["--constraint", constraints_path])
            pip_args.extend(_get_pip_proxy_args())
            pip_args.append(package_spec)


            env = _get_clean_env_for_venv()
            if learned_ssl_env is not None:
                env = dict(learned_ssl_env)

            subprocess_kwargs = _get_subprocess_kwargs()


            pkg_timeout = install_config.package_timeout_s(
                package_name,
                PACKAGE_TIMEOUTS_S.get(package_name, PACKAGE_TIMEOUT_DEFAULT_S))

            install_failed = False
            install_error_msg = ""
            last_returncode = None
            base_cmd = None

            try:

                base_cmd = _build_install_cmd(python_path, pip_args)






                use_cpu_index = (
                    package_name in ("torch", "torchvision") and sys.platform.startswith("linux")
                )
                if use_cpu_index:
                    cpu_args = pip_args[:-1] + [
                        "--index-url",
                        install_config.torch_index_url(TORCH_CPU_INDEX_URL),
                        pip_args[-1]]
                    base_cmd = _build_install_cmd(python_path, cpu_args)


                result = _run_pip_install(
                    cmd=base_cmd,
                    timeout=pkg_timeout,
                    env=env,
                    subprocess_kwargs=subprocess_kwargs,
                    package_name=package_name,
                    package_index=i,
                    total_packages=total_packages,
                    progress_start=pkg_start,
                    progress_end=pkg_end,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )


                if result.returncode == -1 and "cancelled" in (result.stderr or "").lower():
                    _log("Installation cancelled by user", Qgis.MessageLevel.Warning)
                    return False, "Installation cancelled"




                if use_cpu_index and result.returncode != 0:
                    _log(
                        "CPU wheel index failed for "
                        f"{package_name} (code {result.returncode}), "
                        "retrying from the default index...",
                        Qgis.MessageLevel.Warning
                    )
                    base_cmd = _build_install_cmd(python_path, pip_args)
                    result = _run_pip_install(
                        cmd=base_cmd,
                        timeout=pkg_timeout,
                        env=env,
                        subprocess_kwargs=subprocess_kwargs,
                        package_name=package_name,
                        package_index=i,
                        total_packages=total_packages,
                        progress_start=pkg_start,
                        progress_end=pkg_end,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                    )
                    if result.returncode == -1 and "cancelled" in (result.stderr or "").lower():
                        _log("Installation cancelled by user", Qgis.MessageLevel.Warning)
                        return False, "Installation cancelled"



                if not _uv_available and _is_windows_process_crash(result.returncode):
                    _log(
                        f"Process crash detected (code {result.returncode}), "
                        "retrying with pip.exe...",
                        Qgis.MessageLevel.Warning
                    )
                    if progress_callback:
                        progress_callback(
                            pkg_start,
                            tr("Retrying {package}... ({done}/{total})").format(
                                package=package_name, done=i + 1, total=total_packages)
                        )

                    fallback_cmd = [pip_path] + pip_args
                    result = _run_pip_install(
                        cmd=fallback_cmd,
                        timeout=pkg_timeout,
                        env=env,
                        subprocess_kwargs=subprocess_kwargs,
                        package_name=package_name,
                        package_index=i,
                        total_packages=total_packages,
                        progress_start=pkg_start,
                        progress_end=pkg_end,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                    )



                if not _uv_available and result.returncode != 0:
                    error_output = result.stderr or result.stdout or ""
                    if _is_unable_to_create_process(error_output):
                        _log(
                            "Unable to create process detected, "
                            "retrying with pip.exe...",
                            Qgis.MessageLevel.Warning
                        )
                        if progress_callback:
                            progress_callback(
                                pkg_start,
                                tr("Retrying {package}... ({done}/{total})").format(
                                    package=package_name, done=i + 1, total=total_packages)
                            )
                        fallback_cmd = [pip_path] + pip_args
                        result = _run_pip_install(
                            cmd=fallback_cmd,
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            package_name=package_name,
                            package_index=i,
                            total_packages=total_packages,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )








                if result.returncode != 0 and not _is_windows_process_crash(result.returncode):
                    error_output = result.stderr or result.stdout or ""

                    if _is_ssl_error(error_output):
                        if _uv_available:

                            ssl_env, _dropped = (
                                env_without_ca_bundle_overrides(env))
                            os_bundle, n_certs = windows_trust_store_bundle(
                                PLUGIN_CACHE_DIR)
                            if os_bundle:
                                ssl_env["SSL_CERT_FILE"] = os_bundle
                            _log(
                                "SSL error detected, retrying with the "
                                "machine's own certificates ({})...".format(
                                    f"{n_certs} read from the Windows store"
                                    if os_bundle else "system store"),
                                Qgis.MessageLevel.Warning
                            )
                            if progress_callback:
                                progress_callback(
                                    pkg_start,
                                    tr(
                                        "SSL error, retrying {package} (system certs)... "
                                        "({done}/{total})"
                                    ).format(
                                        package=package_name, done=i + 1, total=total_packages)
                                )









                            ssl_cmd_safe = base_cmd + ["--native-tls"]
                            result = _run_pip_install(
                                cmd=ssl_cmd_safe,
                                timeout=pkg_timeout,
                                env=ssl_env,
                                subprocess_kwargs=subprocess_kwargs,
                                package_name=package_name,
                                package_index=i,
                                total_packages=total_packages,
                                progress_start=pkg_start,
                                progress_end=pkg_end,
                                progress_callback=progress_callback,
                                cancel_check=cancel_check,
                            )
                            if result.returncode == 0:
                                env = ssl_env
                                learned_ssl_env = ssl_env
                        else:








                            os_bundle, n_certs = windows_trust_store_bundle(
                                PLUGIN_CACHE_DIR)
                            if os_bundle:
                                ssl_env, _dropped = (
                                    env_without_ca_bundle_overrides(env))
                                ssl_env["SSL_CERT_FILE"] = os_bundle
                                _log(
                                    "SSL error detected, retrying with the "
                                    "machine's own certificates "
                                    f"({n_certs} read from the Windows store)...",
                                    Qgis.MessageLevel.Warning
                                )
                                if progress_callback:
                                    progress_callback(
                                        pkg_start,
                                        tr(
                                            "SSL error, retrying {package} (system certs)... "
                                            "({done}/{total})"
                                        ).format(
                                            package=package_name, done=i + 1,
                                            total=total_packages)
                                    )
                                result = _run_pip_install(
                                    cmd=base_cmd + ["--cert", os_bundle],
                                    timeout=pkg_timeout,
                                    env=ssl_env,
                                    subprocess_kwargs=subprocess_kwargs,
                                    package_name=package_name,
                                    package_index=i,
                                    total_packages=total_packages,
                                    progress_start=pkg_start,
                                    progress_end=pkg_end,
                                    progress_callback=progress_callback,
                                    cancel_check=cancel_check,
                                )
                                if result.returncode == 0:
                                    env = ssl_env
                                    learned_ssl_env = ssl_env





                        if result.returncode != 0 and not _insecure_install_opt_in():
                            _log(
                                "SSL error persists after the retry on this machine's own "
                                "certificates, so the certificate signing these downloads "
                                "is not installed here. Not disabling TLS verification "
                                "(secure default): ask IT to install the network's root "
                                "certificate.",
                                Qgis.MessageLevel.Warning
                            )
                        elif result.returncode != 0:
                            _log(
                                "SSL error persists, retrying with TLS verification bypass "
                                "(user opted in via allow_insecure_install)...",
                                Qgis.MessageLevel.Warning
                            )
                            if progress_callback:
                                progress_callback(
                                    pkg_start,
                                    tr("SSL bypass retry for {package}... ({done}/{total})").format(
                                        package=package_name, done=i + 1, total=total_packages)
                                )



                            ssl_bypass_pip_args = list(pip_args) + _get_pip_ssl_bypass_flags()
                            ssl_cmd_bypass = _build_install_cmd(python_path, ssl_bypass_pip_args)
                            result = _run_pip_install(
                                cmd=ssl_cmd_bypass,
                                timeout=pkg_timeout,
                                env=env,
                                subprocess_kwargs=subprocess_kwargs,
                                package_name=package_name,
                                package_index=i,
                                total_packages=total_packages,
                                progress_start=pkg_start,
                                progress_end=pkg_end,
                                progress_callback=progress_callback,
                                cancel_check=cancel_check,
                            )


                if result.returncode != 0 and not _is_windows_process_crash(result.returncode):
                    error_output = result.stderr or result.stdout or ""

                    if _is_hash_mismatch(error_output):
                        _log(
                            "Hash mismatch detected (corrupted cache), "
                            "retrying with --no-cache-dir...",
                            Qgis.MessageLevel.Warning
                        )






                        _clear_installer_caches()
                        if progress_callback:
                            progress_callback(
                                pkg_start,
                                tr("Cache error, retrying {package}... ({done}/{total})").format(
                                    package=package_name, done=i + 1, total=total_packages)
                            )

                        nocache_flag = "--no-cache" if _uv_available else "--no-cache-dir"
                        nocache_cmd = base_cmd + [nocache_flag]
                        result = _run_pip_install(
                            cmd=nocache_cmd,
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            package_name=package_name,
                            package_index=i,
                            total_packages=total_packages,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )







                if result.returncode != 0 and not _is_windows_process_crash(result.returncode):
                    error_output = result.stderr or result.stdout or ""







                    if (_is_network_error(error_output)
                            and not _is_antivirus_error(error_output)
                            and not _is_index_forbidden_error(error_output)):
                        cancelled, retried = _retry_install_with_backoff(
                            reason="Network error detected",
                            cmd=base_cmd,
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            package_name=package_name,
                            package_index=i,
                            total_packages=total_packages,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )
                        if cancelled:
                            return False, "Installation cancelled"
                        if retried is not None:
                            result = retried


                if result.returncode != 0 and package_name in ("torch", "torchvision"):
                    error_output = result.stderr or result.stdout or ""
                    err_lower = error_output.lower()
                    no_dist = "no matching distribution" in err_lower
                    if no_dist:
                        _log(
                            f"No matching distribution for {package_name}, "
                            "retrying with --no-cache-dir...",
                            Qgis.MessageLevel.Warning
                        )
                        nocache2 = "--no-cache" if _uv_available else "--no-cache-dir"
                        nocache_cmd = base_cmd + [nocache2]
                        result = _run_pip_install(
                            cmd=nocache_cmd,
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            package_name=package_name,
                            package_index=i,
                            total_packages=total_packages,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )


                if result.returncode != 0:
                    error_output = result.stderr or result.stdout or ""
                    if _is_rename_or_record_error(error_output):
                        _log(
                            f"Stale dist-info detected for {package_name}, cleaning and "
                            "retrying with --force-reinstall...",
                            Qgis.MessageLevel.Warning
                        )
                        try:
                            site_pkgs = get_venv_site_packages()
                            if os.path.isdir(site_pkgs):
                                import glob as _glob
                                import shutil as _shutil





                                pattern = os.path.join(
                                    site_pkgs,
                                    f"{package_name.replace('-', '_')}-*.dist-info")
                                for dist_dir in _glob.glob(pattern):
                                    _shutil.rmtree(dist_dir, ignore_errors=True)
                                    _log(f"Removed stale {dist_dir}",
                                         Qgis.MessageLevel.Warning)
                        except Exception as exc:
                            _log(f"Failed to clean dist-info: {exc}",
                                 Qgis.MessageLevel.Warning)

                        if progress_callback:
                            progress_callback(
                                pkg_start,
                                tr("Retrying {package}... ({done}/{total})").format(
                                    package=package_name, done=i + 1, total=total_packages))
                        reinstall_cmd = base_cmd + ["--force-reinstall"]
                        result = _run_pip_install(
                            cmd=reinstall_cmd,
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            package_name=package_name,
                            package_index=i,
                            total_packages=total_packages,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )

                if result.returncode == 0:
                    _log(f"Successfully installed {package_spec}", Qgis.MessageLevel.Success)
                    if progress_callback:
                        progress_callback(
                            pkg_end, tr("{package} installed").format(
                                package=install_display_name(package_name)))
                else:
                    error_msg = _scrub_credentials(
                        result.stderr or result.stdout or f"Return code {result.returncode}")
                    _log(f"✗ Failed to install {package_spec}: {error_msg[:500]}", Qgis.MessageLevel.Critical)
                    install_failed = True
                    install_error_msg = error_msg
                    last_returncode = result.returncode

            except subprocess.TimeoutExpired:
                _log(f"Installation of {package_spec} timed out", Qgis.MessageLevel.Critical)
                install_failed = True
                install_error_msg = f"Installation of {package_name} timed out"





                if base_cmd is not None:
                    cancelled, retried = _retry_install_with_backoff(
                        reason="Download stalled",
                        cmd=base_cmd,
                        timeout=pkg_timeout,
                        env=env,
                        subprocess_kwargs=subprocess_kwargs,
                        package_name=package_name,
                        package_index=i,
                        total_packages=total_packages,
                        progress_start=pkg_start,
                        progress_end=pkg_end,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                    )
                    if cancelled:
                        return False, "Installation cancelled"
                    if retried is not None and retried.returncode == 0:
                        _log(f"Successfully installed {package_spec}", Qgis.MessageLevel.Success)
                        if progress_callback:
                            progress_callback(
                                pkg_end, tr("{package} installed").format(
                                    package=install_display_name(package_name)))
                        install_failed = False
                        install_error_msg = ""
                    elif retried is not None:
                        install_error_msg = _scrub_credentials(
                            retried.stderr or retried.stdout or install_error_msg)
                        last_returncode = retried.returncode
            except Exception as e:
                _log(f"Exception during installation of {package_spec}: {str(e)}", Qgis.MessageLevel.Critical)
                install_failed = True
                install_error_msg = f"Error installing {package_name}: {str(e)[:200]}"

            if (install_failed and package_name in MANUAL_ONLY_PACKAGES
                    and not _failure_is_machine_level(
                        install_error_msg, last_returncode)):




                install_error_msg = _scrub_credentials(install_error_msg)
                _log(
                    f"{package_name} failed to install, so Semi-Auto mode "
                    f"stays unavailable. Automatic mode does not need it and is "
                    f"unaffected. Reason: {install_error_msg[-300:]}",
                    Qgis.MessageLevel.Warning,
                )
                if _is_index_forbidden_error(install_error_msg):
                    _log(
                        "The package index refused that download (HTTP 403), "
                        "which is usually a company or campus network "
                        "filtering it. Ask your IT administrator to allow "
                        f"{_first_run_hosts_sentence()}, then install "
                        "again to turn Semi-Auto mode on.",
                        Qgis.MessageLevel.Warning,
                    )
                degraded_packages.append(package_name)
                if package_name == "torch" and SAM_PACKAGE[0] not in degraded_packages:




                    degraded_packages.append(SAM_PACKAGE[0])
                    _log(
                        f"{SAM_PACKAGE[0]} is skipped as well: it is built "
                        "against torch, which did not install.",
                        Qgis.MessageLevel.Warning,
                    )
                if progress_callback:
                    progress_callback(
                        pkg_end, tr("{package} unavailable").format(package=package_name))
                continue

            if install_failed:




                install_error_msg = _scrub_credentials(install_error_msg)
                _log(f"pip error output (tail): {install_error_msg[-1000:]}", Qgis.MessageLevel.Critical)


                if last_returncode is not None and _is_windows_process_crash(last_returncode):
                    _log(_get_crash_help(venv_dir), Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: process crashed (code {last_returncode})"


                is_dll_err = sys.platform == "win32" and _is_dll_init_error(install_error_msg)
                if is_dll_err and package_name in ("torch", "torchvision"):
                    _log(_get_vcpp_help(), Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: {_get_vcpp_help()}"





                if _is_file_locked_error(install_error_msg):
                    _log(_get_file_locked_help(), Qgis.MessageLevel.Warning)
                    return False, (
                        f"Failed to install {package_name}: file in use by QGIS. "
                        "Please close and reopen QGIS, then retry."
                    )







                if _is_disk_full(install_error_msg):
                    _log(_get_disk_full_help(PLUGIN_CACHE_DIR), Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: no space left on device"





                if _is_app_control_error(install_error_msg):
                    _log(_get_app_control_help(PLUGIN_CACHE_DIR), Qgis.MessageLevel.Warning)
                    return False, (
                        f"Failed to install {package_name}: blocked by an "
                        "application control policy"
                    )








                if _is_antivirus_error(install_error_msg):
                    _log(_get_pip_antivirus_help(PLUGIN_CACHE_DIR), Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: blocked by antivirus or security policy"



                if sys.platform == "win32" and _is_rename_or_record_error(install_error_msg):
                    help_msg = (
                        f"Failed to install {package_name}: file rename blocked.\n\n"
                        "This is typically caused by antivirus or security software "
                        "scanning files during installation.\n\n"
                        "Please try:\n"
                        "  1. Temporarily disable real-time antivirus scanning\n"
                        f"  2. Add an exclusion for: {PLUGIN_CACHE_DIR}\n"
                        "  3. Restart QGIS and reinstall dependencies"
                    )
                    _log(help_msg, Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: blocked by antivirus (rename failed)"





                if _is_ssl_error(install_error_msg):
                    _log(_get_ssl_error_help(install_error_msg), Qgis.MessageLevel.Warning)
                    return False, (
                        f"Failed to install {package_name}: SSL error: "
                        f"{install_error_msg[-400:]}"
                    )


                if _is_proxy_auth_error(install_error_msg):
                    _log(
                        "Proxy authentication failed (HTTP 407). "
                        "Configure proxy credentials in: "
                        "QGIS > Settings > Options > Network > Proxy "
                        "(User and Password fields).",
                        Qgis.MessageLevel.Warning
                    )
                    return False, f"Failed to install {package_name}: proxy authentication required (407)"




                if _is_index_forbidden_error(install_error_msg):
                    _log(
                        "The package index refused the download (HTTP 403). "
                        "This is usually a company or campus network filtering "
                        "downloads. Ask your IT administrator to allow "
                        f"{_first_run_hosts_sentence()}, or run the "
                        "install from another network.",
                        Qgis.MessageLevel.Warning,
                    )
                    return False, (
                        f"Failed to install {package_name}: the package index "
                        "refused the download (403 Forbidden). Your network is "
                        "blocking it."
                    )


                if _is_network_error(install_error_msg):
                    _log(
                        "Network connection failed after multiple retries. "
                        "Check internet connection, VPN/proxy settings, "
                        "and firewall rules for "
                        f"{_first_run_hosts_sentence()}.",
                        Qgis.MessageLevel.Warning
                    )
                    return False, f"Failed to install {package_name}: network error"




                if sys.platform.startswith("linux") and _is_glibc_too_old(install_error_msg):
                    _log(_get_glibc_too_old_help(), Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: glibc too old for this AI engine"





                if _is_macos_intel_no_wheel(install_error_msg):
                    _log(_get_macos_intel_help(), Qgis.MessageLevel.Warning)
                    return False, (
                        f"Failed to install {package_name}: no AI engine build for "
                        "this Intel Mac + Python version"
                    )


                if "no matching platform tag" in install_error_msg.lower():
                    return False, (
                        f"Failed to install {package_name}: no wheels with a matching "
                        "platform tag for this OS/architecture."
                    )


                if _is_unable_to_create_process(install_error_msg):
                    return False, (
                        f"Failed to install {package_name}: unable to create process.\n\n"
                        "Please try:\n"
                        f"  1. Delete the folder: {PLUGIN_CACHE_DIR}\n"
                        "  2. Restart QGIS and reinstall dependencies"
                    )




                if package_name == "rasterio" and "gdal" in install_error_msg.lower():
                    gdal_ok, gdal_help = _check_gdal_available()
                    if not gdal_ok and gdal_help:
                        _log(gdal_help, Qgis.MessageLevel.Warning)
                        return False, f"Failed to install {package_name}: GDAL library not found"




                return False, f"Failed to install {package_name}: {install_error_msg[-400:]}"







            _clear_installer_caches()



        _repin_numpy(venv_dir, cancel_check)





        landed_expected = [
            name for name, _spec in packages
            if name != "setuptools" and name not in degraded_packages
        ]
        misplaced = _packages_missing_from_venv(venv_dir, landed_expected)
        if misplaced:
            missing = ", ".join(misplaced)
            _log(
                f"Installed but absent from the environment: {missing}",
                Qgis.MessageLevel.Critical)



            return False, _get_corrupt_venv_help()

        record_degraded(venv_dir, degraded_packages)
        if degraded_packages:
            short = ", ".join(degraded_packages)
            if progress_callback:
                progress_callback(100, tr("Automatic mode ready"))
            _log("=" * 50, Qgis.MessageLevel.Warning)
            _log(
                f"Installed everything except {short}. Automatic (cloud) mode "
                "is ready. Semi-Auto mode and the AI correction tool stay "
                "off "
                "until that package installs.",
                Qgis.MessageLevel.Warning,
            )
            _log(f"Virtual environment: {venv_dir}", Qgis.MessageLevel.Info)
            _log("=" * 50, Qgis.MessageLevel.Warning)
            return True, (
                f"Automatic mode is ready. {short} could not be installed, so "
                "Semi-Auto mode and the AI correction tool are unavailable. "
                "Everything else works."
            )

        if progress_callback:
            progress_callback(100, tr("All dependencies installed"))

        _log("=" * 50, Qgis.MessageLevel.Success)
        _log("All dependencies installed successfully!", Qgis.MessageLevel.Success)
        _log(f"Virtual environment: {venv_dir}", Qgis.MessageLevel.Success)
        _log("=" * 50, Qgis.MessageLevel.Success)

        return True, "All dependencies installed successfully"

    finally:

        if constraints_path:
            for _attempt in range(3):
                try:
                    os.unlink(constraints_path)
                    break
                except PermissionError:
                    time.sleep(0.5)
                except Exception:
                    break


def _create_venv_and_install(
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    include_local_model: bool = True,
) -> tuple[bool, str]:

    from .python_manager import (
        download_python_standalone,
        get_python_full_version,
        remove_standalone_python,
        standalone_python_exists,
        standalone_python_is_current,
        verify_standalone_python,
    )




    _log_system_info()


    try:
        os.makedirs(PLUGIN_CACHE_DIR, exist_ok=True)






        with tempfile.TemporaryFile(dir=PLUGIN_CACHE_DIR, suffix=".tmp") as probe:
            probe.write(b"ok")
    except OSError as e:
        hint = (
            f"Cannot write to install directory: {PLUGIN_CACHE_DIR}\n"
            f"Error: {e}\n\n"
            "Set the AI_SEGMENTATION_CACHE_DIR environment variable "
            "to a writable directory, then restart QGIS."
        )
        _log(hint, Qgis.MessageLevel.Critical)
        return False, hint


    rosetta_warning = _check_rosetta_warning()
    if rosetta_warning:
        _log(rosetta_warning, Qgis.MessageLevel.Warning)


    removed_venvs = cleanup_old_venv_directories()
    if removed_venvs:
        _log(f"Removed {len(removed_venvs)} old venv directories", Qgis.MessageLevel.Info)

    cleanup_old_libs()



    _sweep_pending_delete()


    if standalone_python_exists() and not standalone_python_is_current():
        _log(
            "Standalone Python version mismatch, re-downloading...",
            Qgis.MessageLevel.Warning)
        remove_standalone_python()

        if venv_exists():
            try:



                _cleanup_partial_venv(VENV_DIR)
                _log("Removed stale venv after Python version mismatch", Qgis.MessageLevel.Info)
            except Exception as e:
                _log(f"Failed to remove stale venv: {e}", Qgis.MessageLevel.Warning)


    from .python_manager import is_nixos
    need_python = not standalone_python_exists()
    if not need_python and not verify_standalone_python()[0]:




        _log(
            "The downloaded Python is present but does not run; "
            "replacing it.",
            Qgis.MessageLevel.Warning)
        remove_standalone_python()
        need_python = True
    if need_python:
        if is_nixos():
            _log("NixOS detected, using system Python", Qgis.MessageLevel.Info)
            if progress_callback:
                progress_callback(10, tr("Using system Python (NixOS)..."))
        else:
            python_version = get_python_full_version()
            _log(f"Downloading Python {python_version} standalone...", Qgis.MessageLevel.Info)

            def python_progress(percent, msg):

                if progress_callback:
                    progress_callback(int(percent * 0.10), msg)

            success, msg = download_python_standalone(
                progress_callback=python_progress,
                cancel_check=cancel_check
            )

            if not success:




                from .python_manager import is_unsupported_python_version
                py_unsupported, py_why = is_unsupported_python_version()
                if py_unsupported:
                    _log(py_why, Qgis.MessageLevel.Critical)
                    return False, py_why




                fallback_python = _fallback_python_for_platform()
                if fallback_python:
                    _log(
                        "Standalone Python download failed, falling back to "
                        f"{fallback_python}: {msg}",
                        Qgis.MessageLevel.Warning
                    )
                    if progress_callback:
                        progress_callback(10, tr("Using system Python (fallback)..."))
                elif _is_download_network_error(msg):


                    return False, (
                        f"Failed to download Python: network error - {msg}")
                else:
                    return False, f"Failed to download Python: {msg}"

        if cancel_check and cancel_check():
            return False, "Installation cancelled"
    else:
        _log("Python standalone already installed", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(10, tr("Python standalone ready"))


    global _uv_available, _uv_path
    if uv_exists() and verify_uv():
        _uv_available = True
        _uv_path = get_uv_path()
        _log("uv already installed, using for package management", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(13, tr("uv package installer ready"))
    else:
        if progress_callback:
            progress_callback(10, tr("Downloading uv package installer..."))
        try:
            def uv_progress(percent, uv_msg):
                if progress_callback:
                    progress_callback(10 + int(percent * 0.03), uv_msg)

            uv_ok, uv_msg = download_uv(
                progress_callback=uv_progress,
                cancel_check=cancel_check,
            )
            if uv_ok:
                _uv_available = True
                _uv_path = get_uv_path()
                _log("uv downloaded, using for package management", Qgis.MessageLevel.Info)
            else:
                _uv_available = False
                _uv_path = None
                _log(f"uv download failed (non-fatal), using pip: {uv_msg}", Qgis.MessageLevel.Warning)
        except Exception as e:
            _uv_available = False
            _uv_path = None
            _log(f"uv download failed (non-fatal): {e}", Qgis.MessageLevel.Warning)
        if progress_callback:
            progress_callback(
                13, tr("uv: ready") if _uv_available else tr("uv: unavailable, using pip"))

    if cancel_check and cancel_check():
        return False, "Installation cancelled"





    def venv_progress(percent, msg):

        if progress_callback:
            progress_callback(13 + int(percent * 0.05), msg)

    if venv_exists() and _venv_is_functional():
        _log("Virtual environment already exists", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(18, tr("Virtual environment ready"))
    else:
        if venv_exists():
            _log(
                "Existing virtual environment is broken or incomplete, "
                "recreating it...", Qgis.MessageLevel.Warning)
            if not _cleanup_partial_venv(VENV_DIR):
                return False, _get_file_locked_help()

        success, msg = create_venv(
            progress_callback=venv_progress, cancel_check=cancel_check)
        if not success:
            return False, msg





        if not _venv_is_functional():



            _log("The new environment does not report itself as one; rebuilding "
                 "on the next attempt.", Qgis.MessageLevel.Critical)
            _cleanup_partial_venv(VENV_DIR)
            return False, _get_corrupt_venv_help()

        if cancel_check and cancel_check():
            return False, "Installation cancelled"


    def deps_progress(percent, msg):

        if progress_callback:
            mapped = 18 + int((percent - 20) * (95 - 18) / 80)
            progress_callback(min(mapped, 95), msg)

    success, msg = install_dependencies(
        progress_callback=deps_progress,
        cancel_check=cancel_check,
        include_local_model=include_local_model,
    )

    if not success:
        return False, msg


    def verify_progress(percent: int, msg: str):

        if progress_callback:

            mapped = 95 + int(percent * 0.04)
            progress_callback(min(mapped, 99), msg)

    is_valid, verify_msg = verify_venv(
        progress_callback=verify_progress,
        include_local_model=include_local_model,
        cancel_check=cancel_check)
    if cancel_check and cancel_check():
        return False, "Installation cancelled"

    if not is_valid:
        return False, f"Verification failed: {verify_msg}"


    _write_deps_hash()

    from .local_model_cache import invalidate as _forget_local_model
    _forget_local_model()


    _clear_installer_caches()

    if progress_callback:
        progress_callback(100, tr("All dependencies installed"))

    return True, "Virtual environment ready"








from .venv_bootstrap import (  # noqa: E402
    _check_gdal_available,
    _check_rosetta_warning,
    _ensurepip_missing_help,
    _fallback_python_for_platform,
    _get_ssl_error_help,
    _get_system_python,
    _is_download_network_error,
    _log_system_info,
    _retry_venv_without_pip,
    _uv_binary_is_unusable,
    _uv_would_choke_on,
    _venv_failed_on_pip_bootstrap,
    _venv_failure_is_blocked,
)
from .venv_deps import (  # noqa: E402
    MANUAL_ONLY_PACKAGES,
    PACKAGE_TIMEOUT_DEFAULT_S,
    PACKAGE_TIMEOUTS_S,
    PIP_RETRIES,
    PIP_TIMEOUT_S,
    TORCH_CPU_INDEX_URL,
    _failure_is_machine_level,
    _numpy_version_spec,
    _write_deps_hash,
    packages_skipped_for_disk_space,
    resolved_packages,
    tr,
)
from .venv_paths import (  # noqa: E402
    VENV_DIR,
    _cleanup_partial_venv,
    _clear_installer_caches,
    _sweep_pending_delete,
    _win_long_path,
    _win_short_path,
    cleanup_old_libs,
    cleanup_old_venv_directories,
    get_venv_pip_path,
    get_venv_python_path,
    get_venv_site_packages,
    venv_exists,
)
from .venv_pip_run import (  # noqa: E402
    _retry_install_with_backoff,
    _run_pip_install,
    _scrub_credentials,
)
from .venv_repair import (  # noqa: E402
    _builds_from_source,
    _repin_numpy,
    _venv_is_functional,
    verify_venv,
)
from .venv_status import (  # noqa: E402
    _packages_missing_from_venv,
)
from .venv_subprocess import (  # noqa: E402
    _get_clean_env_for_venv,
    _get_subprocess_kwargs,
    _run_with_cancel,
)
