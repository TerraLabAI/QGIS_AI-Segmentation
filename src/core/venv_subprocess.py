







from __future__ import annotations

import os
import subprocess  # nosec B404
import tempfile
import time
from typing import Callable

from qgis.core import Qgis

from . import install_config
from .cache_paths import PLUGIN_CACHE_DIR
from .logging_utils import log as _log
from .subprocess_utils import get_clean_env_for_venv as _get_base_clean_env  # nosec B404
from .subprocess_utils import get_subprocess_kwargs as _get_base_subprocess_kwargs  # nosec B404
from .subprocess_utils import (  # nosec B404
    keep_child_off_power_throttling,
    keep_descendants_off_power_throttling,
    stop_process_tree,
)
from .venv_deps import (
    UV_HTTP_RETRIES,
    UV_HTTP_TIMEOUT_S,
)
from .venv_network import _get_effective_proxy_url, _get_qgis_no_proxy_hosts


def _read_text_file(path: str, max_bytes: int = 2 * 1024 * 1024) -> str:

    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read(max_bytes).decode("utf-8", "replace")
    except OSError:
        return ""


def _stop_process(process) -> bool:





    return stop_process_tree(process)


class _PipResult:


    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run_with_cancel(
    cmd: list[str],
    timeout: int,
    env: dict,
    subprocess_kwargs: dict,
    cancel_check: Callable[[], bool] | None = None,
    poll_interval: float = 1.0,
) -> _PipResult:







    if cancel_check and cancel_check():
        return _PipResult(-1, "", "cancelled")
    poll_interval = max(0.01, min(float(poll_interval), 1.0))

    try:
        os.makedirs(PLUGIN_CACHE_DIR, exist_ok=True)
        tmp_dir: str | None = PLUGIN_CACHE_DIR
    except OSError:
        tmp_dir = None
    out_file = err_file = None
    out_path = err_path = None
    process = None
    try:
        out_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix="_out.txt", prefix="run_",
            delete=False, dir=tmp_dir)
        out_path = out_file.name
        err_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix="_err.txt", prefix="run_",
            delete=False, dir=tmp_dir)
        err_path = err_file.name
        process = subprocess.Popen(  # nosec B603
            cmd, stdout=out_file, stderr=err_file, text=True,
            encoding="utf-8", errors="replace", env=env, **subprocess_kwargs)
        keep_child_off_power_throttling(process)
        start = time.monotonic()
        while True:
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                _stop_process(process)
                raise subprocess.TimeoutExpired(cmd, timeout)
            try:
                process.wait(timeout=min(poll_interval, remaining))
                break
            except subprocess.TimeoutExpired:
                pass
            keep_descendants_off_power_throttling(process)
            if cancel_check and cancel_check():
                _stop_process(process)
                return _PipResult(-1, "", "cancelled")
            if time.monotonic() - start >= timeout:
                _stop_process(process)
                raise subprocess.TimeoutExpired(cmd, timeout)

        out_file.close()
        err_file.close()
        out_file = None
        err_file = None
        return _PipResult(
            process.returncode, _read_text_file(out_path), _read_text_file(err_path))
    except BaseException:
        if process and process.poll() is None:
            _stop_process(process)
        raise
    finally:
        for handle in (out_file, err_file):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass  # nosec B110
        for leftover in (out_path, err_path):
            if leftover is None:
                continue
            try:
                os.unlink(leftover)
            except OSError:
                pass  # nosec B110


_dirs_made: set[str] = set()


def _ensure_dir_once(path: str) -> None:






    if path in _dirs_made and os.path.isdir(path):
        return
    os.makedirs(path, exist_ok=True)
    _dirs_made.add(path)


def _apply_cache_containment(env: dict) -> None:















    uv_cache = os.path.join(PLUGIN_CACHE_DIR, "uv_cache")
    pip_cache = os.path.join(PLUGIN_CACHE_DIR, "pip_cache")
    tmp_dir = os.path.join(PLUGIN_CACHE_DIR, "tmp")
    try:
        for path in (uv_cache, pip_cache, tmp_dir):
            _ensure_dir_once(path)
    except OSError as e:
        _log(
            f"Cache containment skipped, using default locations: {e}",
            Qgis.MessageLevel.Info)
        return
    env["UV_CACHE_DIR"] = uv_cache
    env["PIP_CACHE_DIR"] = pip_cache
    env["TMPDIR"] = tmp_dir
    env["TEMP"] = tmp_dir
    env["TMP"] = tmp_dir


def _get_clean_env_for_venv() -> dict:


    env = _get_base_clean_env()



    _apply_cache_containment(env)


    env["SAM2_BUILD_CUDA"] = "0"


    env["CUDA_VISIBLE_DEVICES"] = ""


    env["UV_HTTP_TIMEOUT"] = str(install_config.uv_http_timeout_s(UV_HTTP_TIMEOUT_S))





    env.setdefault("UV_NATIVE_TLS", "1")



    env.setdefault(
        "UV_HTTP_RETRIES", str(install_config.uv_http_retries(UV_HTTP_RETRIES)))





    proxy_url = _get_effective_proxy_url()
    if proxy_url:



        env["HTTP_PROXY"] = proxy_url
        env["HTTPS_PROXY"] = proxy_url
        env["http_proxy"] = proxy_url
        env["https_proxy"] = proxy_url




        no_proxy = _get_qgis_no_proxy_hosts()
        if no_proxy:
            env.setdefault("NO_PROXY", no_proxy)
            env.setdefault("no_proxy", no_proxy)

    return env


def _get_subprocess_kwargs() -> dict:









    _ensure_dir_once(PLUGIN_CACHE_DIR)
    kwargs = dict(_get_base_subprocess_kwargs())



    kwargs["cwd"] = PLUGIN_CACHE_DIR
    kwargs["stdin"] = subprocess.DEVNULL
    return kwargs
