







from __future__ import annotations

import os
import re
import subprocess  # nosec B404
import tempfile
import time
from typing import IO, Callable

from qgis.core import Qgis

from . import install_config
from .cache_paths import PLUGIN_CACHE_DIR
from .install_progress_text import download_size_of, install_display_name
from .logging_utils import log as _log
from .streamed_download import sleep_unless_cancelled as _sleep_unless_cancelled
from .subprocess_utils import (  # nosec B404
    keep_child_off_power_throttling,
    keep_descendants_off_power_throttling,
)
from .venv_deps import (
    NETWORK_RETRY_ATTEMPTS,
    NETWORK_RETRY_BACKOFF_S,
    tr,
)
from .venv_paths import (
    INSTALL_LOG_FILE,
    INSTALL_LOG_MAX_BYTES,
    INSTALL_LOG_STREAM_TAIL,
)
from .venv_subprocess import (
    _PipResult,
    _read_text_file,
    _stop_process,
)


def _parse_pip_download_line(line: str) -> str | None:







    m = re.search(r"Downloading\s+(\S+)\s+\(([^)]+)\)", line)
    if not m:
        return None

    raw_name = m.group(1)
    size = m.group(2)


    if "/" in raw_name:
        raw_name = raw_name.rsplit("/", 1)[-1]



    name_match = re.match(r"([A-Za-z][A-Za-z0-9_]*)", raw_name)
    pkg_name = name_match.group(1) if name_match else raw_name


    size_match = re.match(r"([\d.]+)\s*(kB|MB|GB)", size)
    if size_match:
        try:
            num = float(size_match.group(1))
        except ValueError:
            return None
        unit = size_match.group(2)
        if unit == "MB" and num >= 1000:
            size = f"{num / 1000:.1f} GB"

    return f"Downloading {pkg_name} ({size})"


def _tail_text(path: str, max_bytes: int = 4096) -> str:






    try:
        if max_bytes <= 0:
            return ""
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read(max_bytes).decode("utf-8", "replace")
    except OSError:
        return ""


def _stream_bytes(path: str) -> int:






    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def read_install_log_tail(max_lines: int = 60) -> str:






    if max_lines <= 0:
        return ""
    tail = _tail_text(INSTALL_LOG_FILE, max_bytes=64 * 1024)
    if not tail:
        return ""
    lines = tail.splitlines()

    if len(lines) > 1 and _stream_bytes(INSTALL_LOG_FILE) > 64 * 1024:
        lines = lines[1:]
    return "\n".join(lines[-max_lines:])




_DOWNLOAD_RATIO_RE = re.compile(
    r"([\d.]+)\s*/\s*([\d.]+)\s*(kB|KiB|MB|MiB|GB|GiB)", re.IGNORECASE)


def _parse_download_ratio(line: str) -> float | None:






    m = _DOWNLOAD_RATIO_RE.search(line)
    if not m:
        return None
    try:
        done = float(m.group(1))
        total = float(m.group(2))
    except ValueError:
        return None
    if total <= 0:
        return None
    return max(0.0, min(done / total, 1.0))



_URL_CREDENTIALS_RE = re.compile(r"://[^\s/@]+@")


def _scrub_credentials(text: str) -> str:






    if not text:
        return text
    from .log_scrub import scrub_sensitive

    return scrub_sensitive(_URL_CREDENTIALS_RE.sub("://", text))


def _rotate_install_log() -> None:

    from .server_dials import dial_in_range
    max_bytes = dial_in_range(
        "tuning.install.install_log_max_bytes", INSTALL_LOG_MAX_BYTES,
        256 * 1024, 32 * 1024 * 1024)
    try:
        if os.path.getsize(INSTALL_LOG_FILE) < max_bytes:
            return
    except OSError:
        return
    try:
        os.replace(INSTALL_LOG_FILE, INSTALL_LOG_FILE + ".1")
    except OSError as e:
        _log(f"Could not rotate the install log: {e}", Qgis.MessageLevel.Info)


def _append_install_log(
    package_name: str,
    cmd: list[str],
    returncode: int | None,
    stdout_path: str,
    stderr_path: str,
) -> None:






    try:
        _rotate_install_log()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        stdout_text = _scrub_credentials(_tail_text(stdout_path, INSTALL_LOG_STREAM_TAIL))
        stderr_text = _scrub_credentials(_tail_text(stderr_path, INSTALL_LOG_STREAM_TAIL))
        header = (
            f"\n===== {stamp} | {package_name} | exit={returncode} =====\n"
            f"$ {_scrub_credentials(' '.join(cmd))}\n"
        )
        with open(INSTALL_LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(header)
            for label, text in (("stdout", stdout_text), ("stderr", stderr_text)):
                if not text.strip():
                    continue
                if len(text) > INSTALL_LOG_STREAM_TAIL:
                    f.write(f"--- {label} (earlier output dropped) ---\n")
                    text = text[-INSTALL_LOG_STREAM_TAIL:]
                else:
                    f.write(f"--- {label} ---\n")
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
    except Exception as e:  # noqa: BLE001
        _log(f"Could not write the install log: {e}", Qgis.MessageLevel.Info)


def _run_pip_install(
    cmd: list[str],
    timeout: int,
    env: dict,
    subprocess_kwargs: dict,
    package_name: str,
    package_index: int,
    total_packages: int,
    progress_start: int,
    progress_end: int,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> _PipResult:







    poll_interval = 2





    try:
        os.makedirs(PLUGIN_CACHE_DIR, exist_ok=True)
        _tmp_dir: str | None = PLUGIN_CACHE_DIR
    except OSError:
        _tmp_dir = None
    stdout_fd, stdout_path = tempfile.mkstemp(
        suffix="_stdout.txt", prefix="pip_", dir=_tmp_dir
    )
    stderr_fd, stderr_path = tempfile.mkstemp(
        suffix="_stderr.txt", prefix="pip_", dir=_tmp_dir
    )

    stdout_file: IO[str] | None
    stderr_file: IO[str] | None
    try:
        stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
        stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
    except Exception:

        try:
            os.close(stdout_fd)
        except Exception:
            pass  # nosec B110
        try:
            os.close(stderr_fd)
        except Exception:
            pass  # nosec B110
        for leaked_path in (stdout_path, stderr_path):
            try:
                os.unlink(leaked_path)
            except OSError:
                pass  # nosec B110
        raise

    process = None
    try:
        process = subprocess.Popen(  # nosec B603
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            encoding="utf-8", errors="replace",
            env=env,
            **subprocess_kwargs,
        )


        keep_child_off_power_throttling(process)

        start_time = time.monotonic()
        last_download_status = ""


        last_logged_status = ""
        download_ratio: float | None = None




        last_progress_at = start_time
        stream_bytes = {stdout_path: _stream_bytes(stdout_path),
                        stderr_path: _stream_bytes(stderr_path)}

        while True:
            try:
                process.wait(timeout=poll_interval)

                break
            except subprocess.TimeoutExpired:
                pass


            keep_descendants_off_power_throttling(process)

            elapsed = int(time.monotonic() - start_time)


            if cancel_check and cancel_check():
                if not _stop_process(process):



                    _log("Cancelled install: the process did not exit "
                         "after being killed", Qgis.MessageLevel.Warning)
                return _PipResult(-1, "", "Installation cancelled")


            if time.monotonic() - last_progress_at >= timeout:
                _stop_process(process)
                raise subprocess.TimeoutExpired(cmd, timeout)




            for stream_path in (stdout_path, stderr_path):
                grown = _stream_bytes(stream_path)
                if grown > stream_bytes.get(stream_path, 0):
                    stream_bytes[stream_path] = grown
                    last_progress_at = time.monotonic()





            for stream_path in (stdout_path, stderr_path):
                tail = _tail_text(stream_path)
                if not tail:
                    continue
                lines = tail.strip().split("\n")
                found_status = None
                found_ratio = None
                for line in reversed(lines):
                    if found_ratio is None:
                        found_ratio = _parse_download_ratio(line)
                    if found_status is None:
                        found_status = _parse_pip_download_line(line)
                    if found_status is not None and found_ratio is not None:
                        break
                if found_status and found_status != last_download_status:
                    last_download_status = found_status
                    download_ratio = None
                    last_progress_at = time.monotonic()
                if found_ratio is not None and found_ratio != download_ratio:
                    download_ratio = found_ratio
                    last_progress_at = time.monotonic()
                if found_status is not None or found_ratio is not None:
                    break


            if elapsed >= 60:
                elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"
            else:
                elapsed_str = f"{elapsed}s"





            shown_name = install_display_name(package_name)
            if last_download_status:
                if last_download_status != last_logged_status:
                    last_logged_status = last_download_status
                    _log(last_download_status, Qgis.MessageLevel.Info)
                size = download_size_of(last_download_status)
                if size:
                    msg = tr("Downloading {package} ({size})... {elapsed}").format(
                        package=shown_name, size=size, elapsed=elapsed_str)
                else:
                    msg = tr("Downloading {package}... {elapsed}").format(
                        package=shown_name, elapsed=elapsed_str)
            else:
                msg = tr("Installing {package}... {elapsed}").format(
                    package=shown_name, elapsed=elapsed_str)




            progress_range = progress_end - progress_start
            if download_ratio is not None:

                fraction = min(download_ratio, 0.95)
            elif timeout > 0:
                fraction = min((time.monotonic() - start_time) / timeout, 0.9)
            else:
                fraction = 0
            interpolated = progress_start + int(progress_range * fraction)
            interpolated = min(interpolated, progress_end - 1)

            if progress_callback:
                progress_callback(interpolated, msg)


        if cancel_check and cancel_check():
            return _PipResult(-1, "", "Installation cancelled")


        stdout_file.close()
        stderr_file.close()
        stdout_file = None
        stderr_file = None

        full_stdout = _read_text_file(stdout_path)
        full_stderr = _read_text_file(stderr_path)

        return _PipResult(process.returncode, full_stdout, full_stderr)

    except subprocess.TimeoutExpired:
        raise
    except Exception:
        if process and process.poll() is None:
            _stop_process(process)
        raise
    finally:

        if stdout_file is not None:
            try:
                stdout_file.close()
            except Exception:
                pass  # nosec B110
        if stderr_file is not None:
            try:
                stderr_file.close()
            except Exception:
                pass  # nosec B110


        _append_install_log(
            package_name, cmd,
            process.returncode if process is not None else None,
            stdout_path, stderr_path,
        )

        try:
            os.unlink(stdout_path)
        except Exception:
            pass  # nosec B110
        try:
            os.unlink(stderr_path)
        except Exception:
            pass  # nosec B110


def _retry_install_with_backoff(
    reason: str,
    cmd: list[str],
    timeout: int,
    env: dict,
    subprocess_kwargs: dict,
    package_name: str,
    package_index: int,
    total_packages: int,
    progress_start: int,
    progress_end: int,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[bool, _PipResult | None]:










    attempts = install_config.network_retry_attempts(NETWORK_RETRY_ATTEMPTS)
    result: _PipResult | None = None
    for attempt in range(1, attempts + 1):
        wait = install_config.network_retry_backoff_s(
            attempt, NETWORK_RETRY_BACKOFF_S)
        _log(
            f"{reason}, retrying in {wait}s (attempt {attempt}/{attempts})...",
            Qgis.MessageLevel.Warning
        )
        if progress_callback:
            progress_callback(
                progress_start,
                tr("Network error, retry {attempt}/{total} in {wait}s...").format(
                    attempt=attempt, total=attempts, wait=wait)
            )
        if _sleep_unless_cancelled(wait, cancel_check):
            return True, result
        try:
            result = _run_pip_install(
                cmd=cmd,
                timeout=timeout,
                env=env,
                subprocess_kwargs=subprocess_kwargs,
                package_name=package_name,
                package_index=package_index,
                total_packages=total_packages,
                progress_start=progress_start,
                progress_end=progress_end,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
        except subprocess.TimeoutExpired:
            _log(
                f"{package_name} stalled again on attempt {attempt}/{attempts}",
                Qgis.MessageLevel.Warning
            )
            continue
        if result.returncode == -1 and "cancelled" in (result.stderr or "").lower():
            return True, result
        if result.returncode == 0:
            break
    return False, result
