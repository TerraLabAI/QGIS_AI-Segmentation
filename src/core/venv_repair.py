









from __future__ import annotations

import glob
import os
import subprocess  # nosec B404
import sys
from typing import Callable

from qgis.core import Qgis

from . import install_config
from .cache_paths import PLUGIN_CACHE_DIR, remove_tree_quietly
from .logging_utils import log as _log
from .model_config import TORCH_MIN, TORCHVISION_MIN
from .pip_diagnostics import get_app_control_help as _get_app_control_help
from .pip_diagnostics import get_file_locked_help as _get_file_locked_help
from .pip_diagnostics import get_pip_antivirus_help as _get_pip_antivirus_help
from .pip_diagnostics import get_vcpp_help as _get_vcpp_help
from .pip_diagnostics import install_again_step as _install_again_step
from .pip_diagnostics import is_antivirus_error as _is_antivirus_error
from .pip_diagnostics import is_app_control_error as _is_app_control_error
from .pip_diagnostics import is_dll_init_error as _is_dll_init_error
from .subprocess_utils import run_unthrottled  # nosec B404


def _venv_is_functional(venv_dir: str | None = None) -> bool:















    if venv_dir is None:
        venv_dir = VENV_DIR

    python_path = get_venv_python_path(venv_dir)
    if not os.path.exists(python_path):
        return False
    if not os.path.exists(os.path.join(venv_dir, "pyvenv.cfg")):
        _log("Existing venv has no pyvenv.cfg (incomplete), will recreate",
             Qgis.MessageLevel.Warning)
        return False
    probe_cmd = [python_path, "-c", "import sys; sys.stdout.write(sys.prefix)"]
    from .server_dials import dial_in_range
    probe_timeout_s = dial_in_range(
        "tuning.install.repair_probe_timeout_s", 30, 10, 120)
    try:
        try:
            result = run_unthrottled(
                probe_cmd, text=True, encoding="utf-8",
                errors="replace", timeout=probe_timeout_s, env=_get_clean_env_for_venv(),
                **_get_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired:



            if sys.platform != "win32":
                raise
            _log("Venv interpreter was slow to start, asking once more",
                 Qgis.MessageLevel.Info)
            result = run_unthrottled(
                probe_cmd, text=True, encoding="utf-8",
                errors="replace", timeout=probe_timeout_s, env=_get_clean_env_for_venv(),
                **_get_subprocess_kwargs(),
            )
        if result.returncode != 0:
            return False
        reported_prefix = (result.stdout or "").strip()
        if not reported_prefix:
            _log("Venv interpreter reported no prefix, will recreate",
                 Qgis.MessageLevel.Warning)
            return False


        if (os.path.normcase(os.path.realpath(reported_prefix))
                != os.path.normcase(os.path.realpath(venv_dir))):
            _log(
                f"Venv interpreter answers for another prefix ({reported_prefix}), "
                "will recreate", Qgis.MessageLevel.Warning)
            return False
        return True
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        _log(f"Existing venv interpreter is not runnable, will recreate: {e}",
             Qgis.MessageLevel.Warning)
        return False


def _venv_base_python_ok(venv_dir: str | None = None) -> tuple[bool, str]:










    if venv_dir is None:
        venv_dir = VENV_DIR

    cfg_path = os.path.join(venv_dir, "pyvenv.cfg")
    home = None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            for line in f:
                key, sep, value = line.partition("=")
                if sep and key.strip().lower() == "home":
                    home = value.strip()
                    break
    except OSError as e:
        return False, f"pyvenv.cfg unreadable: {e}"
    if not home:
        return False, "pyvenv.cfg has no home entry"
    if '"' in home or "'" in home:
        return False, f"pyvenv.cfg home path is corrupt: {home}"
    if not os.path.isdir(home):
        return False, f"venv base Python missing: {home}"
    try:
        entries = os.listdir(home)
    except OSError:
        entries = []
    if not any(name.startswith("python") for name in entries):
        return False, f"no python executable in venv base: {home}"
    return True, "venv base Python present"


def venv_needs_repair(allow_subprocess_probe: bool = True) -> bool:












    if not venv_exists():
        return False
    base_ok, base_msg = _venv_base_python_ok()
    if not base_ok:
        _log(f"venv_needs_repair: {base_msg}", Qgis.MessageLevel.Warning)
        return True
    if not allow_subprocess_probe:
        return False
    return not _venv_is_functional()


def _package_loaded_in_process(dir_name: str) -> bool:








    if sys.platform != "win32":
        return False
    return dir_name.split(".", 1)[0] in sys.modules




_STAGED_REMOVAL_PREFIX = "_removing_"


def _remove_package_dir(target: str) -> bool:







    if not os.path.isdir(target):
        return True
    parent = os.path.dirname(target)
    staged = os.path.join(
        parent, f"{_STAGED_REMOVAL_PREFIX}{os.path.basename(target)}.{os.getpid()}")
    try:
        os.replace(_win_extended_path(target), _win_extended_path(staged))
    except PermissionError:
        return False
    except OSError as e:
        _log(f"Could not move {target} aside: {e}", Qgis.MessageLevel.Warning)
        return False
    remove_tree_quietly(_win_extended_path(staged))
    return True


def _sweep_staged_removals(site_packages: str) -> None:

    pattern = os.path.join(_win_extended_path(site_packages), f"{_STAGED_REMOVAL_PREFIX}*")
    for leftover in glob.glob(pattern):
        remove_tree_quietly(leftover)


def purge_package_from_venv(package_name: str, venv_dir: str | None = None) -> bool:












    if venv_dir is None:
        venv_dir = VENV_DIR
    site_packages = get_venv_site_packages(venv_dir)
    if not site_packages or not os.path.isdir(site_packages):
        return True
    dir_name = package_name.replace("-", "_")
    if _package_loaded_in_process(dir_name):
        _log(
            f"Not purging {package_name}: it is loaded in this QGIS process "
            "and a partial delete would break the reinstall. Restart QGIS.",
            Qgis.MessageLevel.Warning
        )
        return False
    _sweep_staged_removals(site_packages)
    targets = [
        os.path.join(site_packages, dir_name),


        os.path.join(site_packages, dir_name + ".libs"),
    ]
    targets.extend(glob.glob(os.path.join(site_packages, f"{dir_name}-*.dist-info")))
    purged = True
    for target in targets:
        if not os.path.exists(target):
            continue
        try:
            if os.path.isdir(target):
                if not _remove_package_dir(target):
                    _log(
                        f"Could not purge {target}: it is in use. Restart QGIS.",
                        Qgis.MessageLevel.Warning
                    )
                    purged = False
                    continue
            else:
                os.remove(target)
            _log(f"Purged broken package artifact: {target}", Qgis.MessageLevel.Info)
        except PermissionError as e:
            _log(f"Could not purge {target}: {e}", Qgis.MessageLevel.Warning)
            purged = False
        except OSError as e:
            _log(f"Could not purge {target}: {e}", Qgis.MessageLevel.Warning)
    return purged


def _remove_torch_dirs(site_pkgs: str) -> bool:








    if _package_loaded_in_process("torch") or _package_loaded_in_process("torchvision"):
        _log(
            "torch is loaded in this QGIS process, so its files cannot be "
            "replaced. Restart QGIS to finish the repair.",
            Qgis.MessageLevel.Warning
        )
        return False
    _sweep_staged_removals(site_pkgs)
    site_pkgs = _win_extended_path(site_pkgs)
    removed_all = True
    for pattern in ("torch*", "torchvision*"):
        for target in glob.glob(os.path.join(site_pkgs, pattern)):
            if os.path.isdir(target) and not _remove_package_dir(target):
                removed_all = False
    return removed_all








_SOURCE_BUILT_PACKAGES = ("sam2", "segment-anything")


def _builds_from_source(package_name: str) -> bool:

    return package_name in _SOURCE_BUILT_PACKAGES


def _repair_install_args(package_name: str, pkg_spec: str) -> list:







    args = ["install", "--force-reinstall", "--disable-pip-version-check"]
    if _builds_from_source(package_name):
        args.append("--no-build-isolation")
    else:
        args.append("--no-deps")
    args += ["--prefer-binary", pkg_spec]
    return args


def _repin_numpy(venv_dir: str, cancel_check: Callable[[], bool] | None = None):








    if sys.version_info >= (3, 13):
        _log("Python >= 3.13: numpy 2.x is expected, skipping repin",
             Qgis.MessageLevel.Info)
        return
    if _package_loaded_in_process("numpy"):




        _log("numpy is already loaded in QGIS; its version is left alone until "
             "the next start.", Qgis.MessageLevel.Info)
        return

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    try:
        result = _run_with_cancel(
            [python_path, "-c", "import numpy; print(numpy.__version__)"],
            30, env, subprocess_kwargs, cancel_check)
        if result.returncode != 0:
            return



        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return
        version_str = lines[-1]
        major = int(version_str.split(".")[0])
        if major >= 2:
            _log(
                f"numpy {version_str} detected (>=2.0), forcing downgrade to <2.0.0...",
                Qgis.MessageLevel.Warning
            )
            downgrade_args = [
                "install", "--force-reinstall", "--no-deps",
                "--disable-pip-version-check",
            ] + [
                "numpy>=1.26.0,<2.0.0",
            ]
            downgrade_cmd = _build_install_cmd(python_path, downgrade_args)
            downgrade_result = _run_with_cancel(
                downgrade_cmd, 120, env, subprocess_kwargs, cancel_check)
            if downgrade_result.returncode == -1 and cancel_check and cancel_check():
                return
            if downgrade_result.returncode == 0:
                _log("numpy downgraded successfully to <2.0.0", Qgis.MessageLevel.Success)
            else:
                err = downgrade_result.stderr or downgrade_result.stdout or ""
                _log(f"numpy downgrade failed: {err[:200]}", Qgis.MessageLevel.Warning)
    except Exception as e:
        _log(f"numpy version check failed: {e}", Qgis.MessageLevel.Warning)


def _get_verification_code(package_name: str) -> str:





    if package_name == "pandas":

        return "import pandas as pd; df = pd.DataFrame({'a': [1, 2, 3]}); print(df.sum())"
    if package_name == "numpy":

        return "import numpy as np; a = np.array([1, 2, 3]); print(np.sum(a))"
    if package_name == "torch":

        return "import torch; t = torch.tensor([1, 2, 3]); print(t.sum())"
    if package_name == "rasterio":

        return "import rasterio; print(rasterio.__version__)"
    if package_name == "sam2":
        return "from sam2.build_sam import build_sam2; print('ok')"
    if package_name == "segment-anything":
        return "from segment_anything import sam_model_registry; print('ok')"
    if package_name == "torchvision":
        return "import torchvision; print(torchvision.__version__)"
    import_name = package_name.replace("-", "_")
    return f"import {import_name}"


def _repair_timeout_s(package_name: str) -> int:






    return install_config.package_timeout_s(
        package_name, PACKAGE_TIMEOUTS_S.get(package_name, PACKAGE_TIMEOUT_DEFAULT_S))


def _run_repair_step(cmd, timeout, env, subprocess_kwargs, cancel_check) -> _PipResult | None:





    try:
        return _run_with_cancel(cmd, timeout, env, subprocess_kwargs, cancel_check)
    except subprocess.TimeoutExpired:
        return None


def _repair_cancelled(result: _PipResult | None, cancel_check) -> bool:

    return (result is not None and result.returncode == -1
            and bool(cancel_check) and bool(cancel_check()))


def verify_venv(
    venv_dir: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    include_local_model: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[bool, str]:






    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment not found"

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    packages = resolved_packages()
    if not include_local_model:



        packages = [(n, s) for n, s in packages
                    if n not in MANUAL_ONLY_PACKAGES]



    unavailable_manual: list[str] = []
    total_packages = len(packages)
    for i, (package_name, _) in enumerate(packages):
        if cancel_check and cancel_check():
            _log("Verification cancelled by user", Qgis.MessageLevel.Warning)
            return False, "Installation cancelled"
        if progress_callback:

            percent = int((i / total_packages) * 100)
            progress_callback(
                percent,
                tr("Verifying {package}... ({done}/{total})").format(
                    package=package_name, done=i + 1, total=total_packages))


        verify_code = _get_verification_code(package_name)
        cmd = [python_path, "-c", verify_code]
        pkg_timeout = _get_verification_timeout(package_name)

        try:
            result = _run_with_cancel(
                cmd, pkg_timeout, env, subprocess_kwargs, cancel_check)
            if result.returncode == -1 and cancel_check and cancel_check():
                _log("Verification cancelled by user", Qgis.MessageLevel.Warning)
                return False, "Installation cancelled"

            if result.returncode != 0:






                full_error = result.stderr or result.stdout or ""
                error_detail = full_error[-400:] if full_error else ""
                _log(
                    f"Package {package_name} verification failed: {error_detail}",
                    Qgis.MessageLevel.Warning
                )





                if package_name in NON_ESSENTIAL_PACKAGES:
                    _log(
                        f"Package {package_name} is not required by the plugin; "
                        "skipping it and continuing. If you need it, install the "
                        "Microsoft Visual C++ Redistributable (x64) and restart "
                        "QGIS.",
                        Qgis.MessageLevel.Warning
                    )
                    continue












                if (package_name in MANUAL_ONLY_PACKAGES
                        and "no module named" in full_error.lower()
                        and not _failure_is_machine_level(full_error, None)):
                    _log(
                        f"Package {package_name} did not verify, so Semi-Auto "
                        "mode and the AI correction tool stay unavailable. Automatic "
                        "(cloud) mode does not use it and is unaffected.",
                        Qgis.MessageLevel.Warning
                    )
                    unavailable_manual.append(package_name)
                    continue







                if _is_app_control_error(full_error):



                    _log(_get_app_control_help(PLUGIN_CACHE_DIR), Qgis.MessageLevel.Warning)
                    return False, (
                        f"Package {package_name} is blocked by an application "
                        "control policy"
                    )
                if _is_antivirus_error(full_error):
                    _log(_get_pip_antivirus_help(PLUGIN_CACHE_DIR), Qgis.MessageLevel.Warning)
                    return False, (
                        f"Package {package_name} is blocked by a security policy "
                        "(antivirus, AppLocker or application control).\n\n"
                        f"{_get_pip_antivirus_help(PLUGIN_CACHE_DIR)}"
                    )



                if _is_dll_init_error(full_error):
                    _log(
                        f"DLL init error for {package_name}, attempting "
                        "force-reinstall...",
                        Qgis.MessageLevel.Warning
                    )
                    pkg_spec = package_name
                    for name, spec in packages:
                        if name == package_name:
                            pkg_spec = f"{name}{spec}"
                            break
                    reinstall_cmd = _build_install_cmd(
                        python_path,
                        _repair_install_args(package_name, pkg_spec))
                    try:
                        step = _run_repair_step(
                            reinstall_cmd, _repair_timeout_s(package_name),
                            env, subprocess_kwargs, cancel_check)
                        if _repair_cancelled(step, cancel_check):
                            return False, "Installation cancelled"
                        result2 = _run_repair_step(
                            cmd, pkg_timeout, env, subprocess_kwargs, cancel_check)
                        if _repair_cancelled(result2, cancel_check):
                            return False, "Installation cancelled"
                        if result2 is not None and result2.returncode == 0:
                            _log(
                                f"Package {package_name} fixed after "
                                "force-reinstall",
                                Qgis.MessageLevel.Success)
                            continue
                    except Exception:
                        pass  # nosec B110





                    if package_name not in ("torch", "torchvision"):
                        _log(_get_vcpp_help(), Qgis.MessageLevel.Warning)
                        return False, (
                            f"Package {package_name} failed: {_get_vcpp_help()}"
                        )


                    _log(
                        "Force-reinstall did not fix DLL error for "
                        f"{package_name}. Nuking and reinstalling...",
                        Qgis.MessageLevel.Warning
                    )
                    if not _remove_torch_dirs(get_venv_site_packages(venv_dir)):
                        _log(_get_file_locked_help(), Qgis.MessageLevel.Warning)
                        return False, (
                            f"Package {package_name} failed: file in use by QGIS. "
                            "Please close and reopen QGIS, then retry."
                        )
                    try:

                        _specs = dict(packages)
                        torch_spec = f"torch{_specs.get('torch', TORCH_MIN)}"
                        tv_spec = (
                            f"torchvision{_specs.get('torchvision', TORCHVISION_MIN)}")
                        nuke_cmd = _build_install_cmd(
                            python_path,
                            ["install", "--prefer-binary",
                             torch_spec, tv_spec])
                        step = _run_repair_step(
                            nuke_cmd, _repair_timeout_s("torch"),
                            env, subprocess_kwargs, cancel_check)
                        if _repair_cancelled(step, cancel_check):
                            return False, "Installation cancelled"
                        result3 = _run_repair_step(
                            cmd, pkg_timeout, env, subprocess_kwargs, cancel_check)
                        if _repair_cancelled(result3, cancel_check):
                            return False, "Installation cancelled"
                        if result3 is not None and result3.returncode == 0:
                            _log(
                                f"Package {package_name} fixed after nuke "
                                "reinstall",
                                Qgis.MessageLevel.Success)
                            continue
                    except Exception:
                        pass  # nosec B110



                    from .model_config import (
                        TORCH_WINDOWS_FALLBACK,
                        TORCHVISION_WINDOWS_FALLBACK,
                    )


                    _torch_pin = install_config.version_pin(
                        "install.windows_fallback.torch", TORCH_WINDOWS_FALLBACK)
                    _tv_pin = install_config.version_pin(
                        "install.windows_fallback.torchvision",
                        TORCHVISION_WINDOWS_FALLBACK)
                    if _torch_pin:
                        _log(
                            "Nuke reinstall did not fix DLL error. "
                            f"Trying pinned torch{_torch_pin} fallback...",
                            Qgis.MessageLevel.Warning
                        )
                        if not _remove_torch_dirs(
                                get_venv_site_packages(venv_dir)):
                            _log(_get_file_locked_help(), Qgis.MessageLevel.Warning)
                            return False, (
                                f"Package {package_name} failed: file in use by "
                                "QGIS. Please close and reopen QGIS, then retry."
                            )
                        try:
                            fallback_cmd = _build_install_cmd(
                                python_path,
                                ["install", "--prefer-binary",
                                 f"torch{_torch_pin}",
                                 f"torchvision{_tv_pin}"])
                            step = _run_repair_step(
                                fallback_cmd, _repair_timeout_s("torch"),
                                env, subprocess_kwargs, cancel_check)
                            if _repair_cancelled(step, cancel_check):
                                return False, "Installation cancelled"
                            result4 = _run_repair_step(
                                cmd, pkg_timeout, env, subprocess_kwargs, cancel_check)
                            if _repair_cancelled(result4, cancel_check):
                                return False, "Installation cancelled"
                            if result4 is not None and result4.returncode == 0:
                                _log(
                                    f"Package {package_name} fixed with pinned "
                                    f"torch{_torch_pin} fallback",
                                    Qgis.MessageLevel.Success)
                                continue
                        except Exception:
                            pass  # nosec B110

                    _log(_get_vcpp_help(), Qgis.MessageLevel.Warning)
                    return False, (
                        f"Package {package_name} failed: {_get_vcpp_help()}"
                    )


                error_lower = full_error.lower()
                broken_markers = install_config.classifier_markers(
                    "broken_extension",
                    [
                        "no module named", "_libs",
                        "dll load failed", "importerror",
                        "applocker", "application control",
                        "blocked by your organization",
                    ],
                )
                is_broken = any(m in error_lower for m in broken_markers)

                if is_broken:
                    _log(
                        f"Package {package_name} has broken C extensions, "
                        "attempting force-reinstall...",
                        Qgis.MessageLevel.Warning
                    )

                    pkg_spec = package_name
                    for name, spec in packages:
                        if name == package_name:
                            pkg_spec = f"{name}{spec}"
                            break
                    reinstall_cmd = _build_install_cmd(
                        python_path,
                        _repair_install_args(package_name, pkg_spec))
                    try:
                        step = _run_repair_step(
                            reinstall_cmd, _repair_timeout_s(package_name),
                            env, subprocess_kwargs, cancel_check)
                        if _repair_cancelled(step, cancel_check):
                            return False, "Installation cancelled"
                    except Exception:
                        pass  # nosec B110

                    try:
                        result2 = _run_repair_step(
                            cmd, pkg_timeout, env, subprocess_kwargs, cancel_check)
                        if _repair_cancelled(result2, cancel_check):
                            return False, "Installation cancelled"
                        if result2 is not None and result2.returncode == 0:
                            _log(
                                f"Package {package_name} fixed after force-reinstall",
                                Qgis.MessageLevel.Success
                            )
                            continue
                    except Exception:
                        pass  # nosec B110

                    detail_lower = full_error.lower()
                    applocker_markers = [
                        "applocker", "application control",
                        "blocked by your organization",
                    ]
                    if any(m in detail_lower for m in applocker_markers):
                        return False, (
                            f"Package {package_name} is blocked by AppLocker or "
                            "application control policy.\n\n"
                            "Ask your IT administrator to whitelist "
                            f"this folder:\n  {venv_dir}\n\n"
                            "Then restart QGIS and reinstall "
                            "dependencies."
                        )
                    return False, (
                        f"Package {package_name} is broken (antivirus may be "
                        f"interfering): {error_detail[:200]}"
                    )

                if sys.platform == "win32":
                    from .interaction_dials import vcredist_url
                    vcpp_url = vcredist_url(
                        "https://aka.ms/vs/17/release/vc_redist.x64.exe"
                    )
                    if package_name == "torch":
                        hint = (
                            "\n\nPlease try:\n"
                            "  1. Install Visual C++ "
                            "Redistributable:\n"
                            f"     {vcpp_url}\n"
                            "  2. Add an antivirus exclusion for "
                            "the plugin folder\n"
                            "  3. Restart QGIS"
                        )
                    else:
                        hint = (
                            "\n\nPlease try:\n"
                            "  1. Install Visual C++ "
                            "Redistributable:\n"
                            f"     {vcpp_url}\n"
                            f"  2. {_install_again_step()}\n"
                            "  3. Restart QGIS"
                        )
                else:
                    hint = (
                        "\n\nPlease try:\n"
                        f"  1. {_install_again_step()}\n"
                        "  2. Restart QGIS"
                    )
                return False, f"Package {package_name} is broken: {error_detail[:200]}{hint}"

        except subprocess.TimeoutExpired:


            _log(
                f"Verification of {package_name} timed out ({pkg_timeout}s), retrying...",
                Qgis.MessageLevel.Info
            )
            try:
                result = _run_with_cancel(
                    cmd, pkg_timeout, env, subprocess_kwargs, cancel_check)
                if result.returncode == -1 and cancel_check and cancel_check():
                    _log("Verification cancelled by user", Qgis.MessageLevel.Warning)
                    return False, "Installation cancelled"
                if result.returncode != 0:



                    retry_error = result.stderr or result.stdout or ""
                    error_detail = retry_error[-400:]
                    _log(
                        f"Package {package_name} verification failed on retry: {error_detail}",
                        Qgis.MessageLevel.Warning
                    )

                    retry_lower = retry_error.lower()
                    applocker_kw = [
                        "applocker", "application control",
                        "blocked by your organization",
                    ]
                    if (any(m in retry_lower for m in applocker_kw)
                            or _is_app_control_error(retry_error)):
                        return False, (
                            f"Package {package_name} is blocked by AppLocker "
                            "or application control policy.\n\n"
                            "Ask your IT administrator to "
                            "whitelist this folder:\n"
                            f"  {venv_dir}\n\nThen restart QGIS and "
                            "reinstall dependencies."
                        )
                    return False, f"Package {package_name} is broken: {error_detail[:200]}"
            except subprocess.TimeoutExpired:
                _log(
                    f"Verification of {package_name} timed out twice",
                    Qgis.MessageLevel.Warning
                )
                return False, (
                    f"Verification error: {package_name} "
                    "(timed out - antivirus may be blocking)"
                )
            except Exception as e:
                _log(
                    f"Failed to verify {package_name} on retry: {str(e)}",
                    Qgis.MessageLevel.Warning
                )
                return False, f"Verification error: {package_name}"

        except Exception as e:
            _log(
                f"Failed to verify {package_name}: {str(e)}",
                Qgis.MessageLevel.Warning
            )
            return False, f"Verification error: {package_name}"

    if progress_callback:
        progress_callback(100, tr("Verification complete"))

    if unavailable_manual:
        short = ", ".join(unavailable_manual)
        _log(
            f"Virtual environment verified for Automatic mode. {short} did not "
            "verify, so Semi-Auto mode and the AI correction tool stay off.",
            Qgis.MessageLevel.Warning,
        )
        return True, (
            f"Ready for Automatic mode. {short} is unavailable, so "
            "Semi-Auto mode and the AI correction tool are off."
        )

    _log("Virtual environment verified successfully", Qgis.MessageLevel.Success)
    return True, "Virtual environment ready"








from .venv_deps import (  # noqa: E402
    MANUAL_ONLY_PACKAGES,
    NON_ESSENTIAL_PACKAGES,
    PACKAGE_TIMEOUT_DEFAULT_S,
    PACKAGE_TIMEOUTS_S,
    _failure_is_machine_level,
    _get_verification_timeout,
    resolved_packages,
    tr,
)
from .venv_install import (  # noqa: E402
    _build_install_cmd,
)
from .venv_paths import (  # noqa: E402
    VENV_DIR,
    _win_extended_path,
    get_venv_python_path,
    get_venv_site_packages,
    venv_exists,
)
from .venv_subprocess import (  # noqa: E402
    _get_clean_env_for_venv,
    _get_subprocess_kwargs,
    _PipResult,
    _run_with_cancel,
)
