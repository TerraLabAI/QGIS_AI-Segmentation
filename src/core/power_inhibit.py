"""Cross-platform keep-awake facade for long installs and Auto runs.

A multi-GB dependency install or a large Automatic run over a slow link can
be interrupted by system sleep if the user walks away; an interrupted Auto
run can waste already-billed detections. This wraps platform-specific sleep/
idle inhibition behind one begin_keep_awake/end_keep_awake pair so callers don't
need per-OS branches:

- macOS: NSProcessInfo App Nap suppression (macos_activity.py, unchanged).
- Windows: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
- Linux: best-effort ``systemd-inhibit`` subprocess if present, else no-op.

Everything here is best-effort and fail-open: any failure just means the
system might sleep, never a hard error surfaced to the user. Dependency-free
(ctypes/subprocess only, both stdlib).
"""
from __future__ import annotations

import logging
import os
import subprocess  # nosec B404
import sys

logger = logging.getLogger(__name__)

_IS_MACOS = sys.platform == "darwin"
_IS_WINDOWS = sys.platform == "win32"
_IS_LINUX = sys.platform == "linux"

# Windows SetThreadExecutionState flags (winbase.h). ES_CONTINUOUS informs the
# system the state should remain in effect until reset; combined with
# ES_SYSTEM_REQUIRED it resets the system idle timer without also forcing the
# display to stay on.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def begin_keep_awake(reason: str = "AI Segmentation task"):
    """Start a keep-awake hold for the duration of a task.

    Returns an opaque token to pass to end_keep_awake(), or None on any
    platform/failure. Callers must treat None as a no-op and never raise.
    """
    if _IS_MACOS:
        from .macos_activity import begin_app_nap_activity as _mac_begin
        return ("macos", _mac_begin(reason))
    if _IS_WINDOWS:
        try:
            import ctypes
            ok = ctypes.windll.kernel32.SetThreadExecutionState(
                _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
            return ("windows", bool(ok)) if ok else None
        except Exception as exc:  # noqa: BLE001 - power management must never break a run
            logger.debug("power_inhibit: SetThreadExecutionState failed: %s", exc)
            return None
    if _IS_LINUX:
        try:
            # The child waits on a pipe we hold rather than on `sleep infinity`,
            # so it releases the inhibitor on its own the moment this process
            # goes away. With a plain sleep, a QGIS that crashes or is killed
            # leaves the child reparented to init, holding the machine awake
            # until the user logs out, with nothing on screen to explain it.
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
            # Most commonly FileNotFoundError: systemd-inhibit is absent
            # (non-systemd distros); no-op, never fatal.
            logger.debug("power_inhibit: systemd-inhibit unavailable: %s", exc)
            return None
    return None


def end_keep_awake(token) -> None:
    """End a keep-awake hold started by begin_keep_awake(). Safe to call with None."""
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
            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("power_inhibit: reset SetThreadExecutionState failed: %s", exc)
        return
    if kind == "linux":
        proc, write_fd = payload
        # Closing our end of the pipe is what actually ends the hold: the
        # child's `read` returns and systemd-inhibit exits. terminate() stays
        # as the belt to that braces.
        try:
            os.close(write_fd)
        except OSError as exc:
            logger.debug("power_inhibit: closing the inhibit pipe failed: %s", exc)
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception as exc:  # noqa: BLE001
            logger.debug("power_inhibit: systemd-inhibit teardown failed: %s", exc)
