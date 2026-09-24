















from __future__ import annotations

import logging
import os
import subprocess  # nosec B404
import sys

logger = logging.getLogger(__name__)

_IS_MACOS = sys.platform == "darwin"
_IS_WINDOWS = sys.platform == "win32"
_IS_LINUX = sys.platform == "linux"





_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001




_ES_AWAYMODE_REQUIRED = 0x00000040


def begin_keep_awake(reason: str = "AI Segmentation task"):





    if _IS_MACOS:
        from .macos_activity import begin_app_nap_activity as _mac_begin
        return ("macos", _mac_begin(reason))
    if _IS_WINDOWS:
        try:
            import ctypes



            set_state = ctypes.windll.kernel32.SetThreadExecutionState  # type: ignore[attr-defined]
            previous = set_state(
                _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_AWAYMODE_REQUIRED)
            if not previous:
                previous = set_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
            return ("windows", int(previous)) if previous else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("power_inhibit: SetThreadExecutionState failed: %s", exc)
            return None
    if _IS_LINUX:
        try:





            read_fd, write_fd = os.pipe()
            try:
                proc = subprocess.Popen(  # nosec B603 B607
                    [
                        "systemd-inhibit", "--what=sleep:idle", f"--why={reason}",
                        "--mode=block", "sh", "-c", "read _",
                    ],
                    stdin=read_fd,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            finally:
                os.close(read_fd)
            return ("linux", (proc, write_fd))
        except (OSError, subprocess.SubprocessError) as exc:


            logger.debug("power_inhibit: systemd-inhibit unavailable: %s", exc)
            return None
    return None


def end_keep_awake(token) -> None:

    if token is None:
        return
    kind, payload = token
    if kind == "macos":
        from .macos_activity import end_app_nap_activity as _mac_end
        _mac_end(payload)
        return
    if kind == "windows":
        try:
            import ctypes

            restore = payload if isinstance(payload, int) and payload else _ES_CONTINUOUS
            ctypes.windll.kernel32.SetThreadExecutionState(restore)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.debug("power_inhibit: reset SetThreadExecutionState failed: %s", exc)
        return
    if kind == "linux":
        proc, write_fd = payload



        try:
            os.close(write_fd)
        except OSError as exc:
            logger.debug("power_inhibit: closing the inhibit pipe failed: %s", exc)
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:


            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception as exc:  # noqa: BLE001
                logger.debug("power_inhibit: systemd-inhibit kill failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("power_inhibit: systemd-inhibit teardown failed: %s", exc)
