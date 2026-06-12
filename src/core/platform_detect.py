"""Low-level platform detection shared by the Python and uv managers.

Stdlib-only (no QGIS, no plugin imports) so it can be imported from any
manager without circular-import risk. Detects the two environments whose
default platform triple would otherwise be wrong:

- musl Linux (Alpine and derivatives): the glibc build boots with a cryptic
  dynamic-loader error, so we must pick the -musl python-build-standalone /
  uv triples instead.
- Windows on ARM (Snapdragon laptops): QGIS itself usually runs as emulated
  x86_64, so the real machine arch is hidden behind WOW64; detect it for
  logging and future native support without changing the safe x86_64 default.
"""
from __future__ import annotations

import glob
import os
import platform
import subprocess  # nosec B404
import sys


def is_musl_linux() -> bool:
    """True when running on a musl-libc Linux (Alpine and friends).

    platform.libc_ver() is unreliable here (it reads the running Python's
    own libc tag and returns ("", "") on several musl builds), so the
    authoritative signal is the presence of the musl dynamic loader, with
    `ldd --version` as a secondary check.
    """
    if sys.platform != "linux":
        return False
    try:
        libc, _ver = platform.libc_ver()
        if libc and "glibc" in libc.lower():
            return False
    except Exception:
        pass  # nosec B110
    if glob.glob("/lib/ld-musl-*") or glob.glob("/usr/lib/ld-musl-*"):
        return True
    try:
        result = subprocess.run(  # nosec B603 B607
            ["ldd", "--version"],
            capture_output=True, text=True, errors="replace", timeout=5,
        )
        return "musl" in (result.stdout + result.stderr).lower()
    except Exception:
        return False


def is_windows_arm64() -> bool:
    """True when the real machine is ARM64 Windows, even under x86 emulation.

    A 32-bit-on-64-bit or emulated process sees PROCESSOR_ARCHITEW6432 set to
    the native arch; on ARM64 Windows it reads "ARM64". platform.machine() is
    checked too for native-arch processes.
    """
    if sys.platform != "win32":
        return False
    arch_envs = (
        os.environ.get("PROCESSOR_ARCHITEW6432", ""),
        os.environ.get("PROCESSOR_ARCHITECTURE", ""),
    )
    if any("arm64" in a.lower() for a in arch_envs):
        return True
    return platform.machine().lower() in ("arm64", "aarch64")
