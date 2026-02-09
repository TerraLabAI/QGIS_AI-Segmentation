from __future__ import annotations

import functools
import os
import subprocess  # nosec B404
import sys


def _sits_inside(path: str, directory: str) -> bool:






    if not path or not directory:
        return False
    try:
        parent = os.path.normcase(os.path.realpath(directory))
        child = os.path.normcase(os.path.realpath(path))
    except (OSError, ValueError):
        return False
    return child == parent or child.startswith(parent + os.sep)


def _qgis_install_roots() -> list[str]:










    roots = []
    for name in ("OSGEO4W_ROOT", "QGIS_PREFIX_PATH"):
        value = os.environ.get(name, "")
        if value:
            roots.append(value)
    if os.name == "nt":
        prefix = os.environ.get("QGIS_PREFIX_PATH", "")
        if prefix:
            grandparent = os.path.dirname(os.path.dirname(prefix))

            if grandparent and len(grandparent.rstrip("\\/")) > 2:
                roots.append(grandparent)
    return roots





_SHARED_PREFIXES = frozenset({
    "/", "/usr", "/usr/local", "/opt", "/opt/local", "/opt/homebrew",
})


def _qgis_only_roots(roots: list[str]) -> list[str]:

    own = []
    for root in roots:
        try:
            resolved = os.path.realpath(root)
        except (OSError, ValueError):
            continue
        trimmed = resolved.rstrip("\\/")
        if trimmed in _SHARED_PREFIXES:
            continue

        if len(trimmed) <= 2:
            continue
        own.append(resolved)
    return own


def _strip_qgis_from_path(env: dict) -> None:













    path = env.get("PATH", "")
    if not path:
        return
    own = _qgis_only_roots(_qgis_install_roots())
    if not own:
        return
    kept = _path_without_roots(path, tuple(own))
    if kept:
        env["PATH"] = kept


@functools.lru_cache(maxsize=8)
def _path_without_roots(path: str, own: tuple) -> str:






    kept = [entry for entry in path.split(os.pathsep)
            if entry and not any(_sits_inside(entry, root) for root in own)]
    return os.pathsep.join(kept)


def get_clean_env_for_venv() -> dict:

    env = os.environ.copy()
    qgis_roots = _qgis_install_roots()
    vars_to_remove = [
        "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",











        "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__",




        "PIP_USER", "PIP_TARGET", "PIP_PREFIX", "PYTHONUSERBASE",
        "QGIS_PREFIX_PATH", "QGIS_PLUGINPATH",
        "PROJ_DATA", "PROJ_LIB",
        "GDAL_DATA", "GDAL_DRIVER_PATH",







        "LD_LIBRARY_PATH", "LD_PRELOAD",
        "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
    ]
    if sys.platform == "win32":






        vars_to_remove += ["QT_PLUGIN_PATH", "PDAL_DRIVER_PATH", "GISBASE"]
        vars_to_remove += [name for name in env if name.upper().startswith("GRASS_")]
    for var in vars_to_remove:
        env.pop(var, None)
    _strip_qgis_from_path(env)









    for store_var, exists in (("SSL_CERT_DIR", os.path.isdir),
                              ("SSL_CERT_FILE", os.path.isfile),
                              ("REQUESTS_CA_BUNDLE", os.path.isfile),
                              ("CURL_CA_BUNDLE", os.path.isfile)):
        store = env.get(store_var, "")
        if not store:
            continue
        if not exists(store) or any(_sits_inside(store, r) for r in qgis_roots):
            env.pop(store_var, None)
    env["PYTHONIOENCODING"] = "utf-8"



    env["PYTHONNOUSERSITE"] = "1"





    env["PYTHONSAFEPATH"] = "1"
    return env


def get_subprocess_kwargs() -> dict:

    kwargs = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def _taskkill_tree(pid: int) -> bool:

    system_root = os.environ.get("SYSTEMROOT") or "C:" + os.sep + "Windows"
    taskkill = os.path.join(system_root, "System32", "taskkill.exe")
    if not os.path.isfile(taskkill):
        return False
    try:
        result = subprocess.run(  # nosec B603
            [taskkill, "/T", "/F", "/PID", str(int(pid))],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=15, check=False,
            **get_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return result.returncode == 0


def stop_process_tree(process, grace: float = 10.0, hard: float = 5.0) -> bool:









    if process is None or process.poll() is not None:
        return True
    if sys.platform == "win32" and _taskkill_tree(process.pid):
        try:
            process.wait(timeout=hard)
            return True
        except subprocess.TimeoutExpired:
            pass
    try:
        process.terminate()
        process.wait(timeout=grace)
        return True
    except subprocess.TimeoutExpired:
        pass
    except OSError:
        return process.poll() is not None
    try:
        process.kill()
        process.wait(timeout=hard)
        return True
    except (subprocess.TimeoutExpired, OSError):
        return process.poll() is not None


def keep_child_off_power_throttling(proc) -> bool:










    if sys.platform != "win32":
        return False
    handle = int(getattr(proc, "_handle", 0) or 0)
    return bool(handle) and _unthrottle_handle(handle)


def _unthrottle_handle(handle: int) -> bool:

    try:
        import ctypes
        from ctypes import wintypes

        class _ThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG),
                        ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]



        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        setter = kernel32.SetProcessInformation
        setter.restype = wintypes.BOOL
        setter.argtypes = [wintypes.HANDLE, ctypes.c_int,
                           ctypes.c_void_p, wintypes.DWORD]

        state = _ThrottlingState(1, 0x1, 0x0)
        return bool(setter(handle, 4, ctypes.byref(state),
                           ctypes.sizeof(state)))
    except Exception:  # noqa: BLE001
        return False


def keep_descendants_off_power_throttling(proc) -> int:







    if sys.platform != "win32":
        return 0
    try:
        from .windows_process_tree import descendant_pids, open_for_set_info
    except ImportError:
        return 0
    count = 0
    for pid in descendant_pids(int(proc.pid)):
        handle = open_for_set_info(pid)
        if handle is None:
            continue
        try:
            count += _unthrottle_handle(handle)
        finally:
            _close_handle(handle)
    return count


def _close_handle(handle: int) -> None:
    try:
        import ctypes
        ctypes.WinDLL("kernel32").CloseHandle(ctypes.c_void_p(handle))
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def run_unthrottled(cmd: list, timeout: float | None = None, **kwargs) -> subprocess.CompletedProcess:







    if sys.platform != "win32":
        return subprocess.run(  # nosec B603
            cmd, capture_output=True, timeout=timeout, check=False, **kwargs)
    kwargs.setdefault("stdout", subprocess.PIPE)
    kwargs.setdefault("stderr", subprocess.PIPE)
    with subprocess.Popen(cmd, **kwargs) as process:  # nosec B603
        keep_child_off_power_throttling(process)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process_tree(process, grace=2.0, hard=5.0)
            process.communicate()
            raise
        except BaseException:
            stop_process_tree(process, grace=2.0, hard=5.0)
            raise
        return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
