"""Cross-platform keep-awake facade for long installs and Auto runs.

A multi-GB dependency install or a large Automatic run over a slow link can
be interrupted by system sleep if the user walks away; an interrupted Auto
run can waste already-billed detections. This wraps platform-specific sleep/
idle inhibition behind one begin_activity/end_activity pair so callers don't
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


def begin_activity(reason: str = "AI Segmentation task"):
    """Start a keep-awake hold for the duration of a task.

    Returns an opaque token to pass to end_activity(), or None on any
    platform/failure. Callers must treat None as a no-op and never raise.
    """
    if _IS_MACOS:
        from .macos_activity import begin_activity as _mac_begin
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
            proc = subprocess.Popen(  # nosec B603 B607
                [
                    "systemd-inhibit", "--what=sleep:idle", f"--why={reason}",
                    "--mode=block", "sleep", "infinity",
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return ("linux", proc)
        except (OSError, subprocess.SubprocessError) as exc:
            # Most commonly FileNotFoundError: systemd-inhibit is absent
            # (non-systemd distros); no-op, never fatal.
            logger.debug("power_inhibit: systemd-inhibit unavailable: %s", exc)
            return None
    return None


def end_activity(token) -> None:
    """End a keep-awake hold started by begin_activity(). Safe to call with None."""
    if token is None:
        return
    kind, payload = token
    if kind == "macos":
        from .macos_activity import end_activity as _mac_end
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
        try:
            payload.terminate()
            payload.wait(timeout=2)
        except Exception as exc:  # noqa: BLE001
            logger.debug("power_inhibit: systemd-inhibit teardown failed: %s", exc)
