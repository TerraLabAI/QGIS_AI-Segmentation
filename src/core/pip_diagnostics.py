"""Pip/uv install error detection and user-facing help messages.

Pure functions that classify pip/uv stderr/stdout output into actionable
error categories. Reusable across any QGIS plugin that installs Python
packages via pip or uv.
"""
from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# SSL / certificate errors
# ---------------------------------------------------------------------------

_SSL_ERROR_PATTERNS = [
    "ssl error",
    "ssl:",
    "sslerror",
    "sslcertverificationerror",
    "certificate verify failed",
    "CERTIFICATE_VERIFY_FAILED",
    "tlsv1 alert",
    "unable to get local issuer certificate",
    "self signed certificate in certificate chain",
]


def is_ssl_error(stderr: str) -> bool:
    """Detect SSL/certificate errors in pip output."""
    stderr_lower = stderr.lower()
    return any(pattern.lower() in stderr_lower for pattern in _SSL_ERROR_PATTERNS)


def is_hash_mismatch(output: str) -> bool:
    """Detect pip hash mismatch errors (corrupted cache from interrupted download)."""
    output_lower = output.lower()
    return "do not match the hashes" in output_lower or "hash mismatch" in output_lower


def get_pip_ssl_flags() -> list[str]:
    """Get pip flags to bypass SSL verification for corporate proxies.

    Note: --trusted-host may be deprecated in future pip versions (>= 21.0),
    but is still needed for older pip on QGIS 3.22-3.28 bundled Python.
    """
    return [
        "--trusted-host", "pypi.org",
        "--trusted-host", "pypi.python.org",
        "--trusted-host", "files.pythonhosted.org",
    ]


def is_ssl_module_missing(error_text: str) -> bool:
    """Check if the error is about a missing SSL module (not a certificate issue)."""
    lower = error_text.lower()
    patterns = ["ssl module is not available", "no module named '_ssl'",
                "ssl module", "importerror: _ssl"]
    return any(p in lower for p in patterns)


def get_ssl_error_help(error_text: str = "", cache_dir: str = "") -> str:
    """Get actionable help message for SSL errors.

    Differentiates between SSL module missing (broken Python) and
    SSL certificate errors (network/proxy).
    """
    if is_ssl_module_missing(error_text):
        return (
            "Installation failed: Python's SSL module is not available.\n\n"
            "This usually means the Python installation is incomplete or corrupted.\n"
            "Please try:\n"
            f"  1. Delete the folder: {cache_dir}\n"
            "  2. Restart QGIS and try again\n"
            "  3. If the issue persists, reinstall QGIS"
        )
    return (
        "Installation failed due to network restrictions.\n\n"
        "Please contact your IT department to allow access to:\n"
        "  - pypi.org\n"
        "  - files.pythonhosted.org\n"
        "  - download.pytorch.org\n\n"
        "You can also try checking your proxy settings in QGIS "
        "(Settings > Options > Network)."
    )


# ---------------------------------------------------------------------------
# Network / connectivity errors
# ---------------------------------------------------------------------------

_NETWORK_ERROR_PATTERNS = [
    "connectionreseterror",
    "connection aborted",
    "connection was forcibly closed",
    "remotedisconnected",
    "connectionerror",
    "newconnectionerror",
    "maxretryerror",
    "protocolerror",
    "readtimeouterror",
    "connecttimeouterror",
    "urlib3.exceptions",
    "requests.exceptions.connectionerror",
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
    "network timeout",
    "failed to download",
]


def is_network_error(output: str) -> bool:
    """Detect transient network/connection errors in pip output."""
    output_lower = output.lower()
    # Exclude SSL errors — they have their own retry path
    if is_ssl_error(output):
        return False
    return any(p in output_lower for p in _NETWORK_ERROR_PATTERNS)


def is_proxy_auth_error(output: str) -> bool:
    """Detect proxy authentication errors (HTTP 407)."""
    output_lower = output.lower()
    patterns = [
        "407 proxy authentication",
        "proxy authentication required",
        "proxyerror",
    ]
    return any(p in output_lower for p in patterns)


# ---------------------------------------------------------------------------
# Windows process / DLL errors
# ---------------------------------------------------------------------------

def is_unable_to_create_process(output: str) -> bool:
    """Detect 'unable to create process' errors on Windows (broken pip.exe shim)."""
    return "unable to create process" in output.lower()


def is_dll_init_error(output: str) -> bool:
    """Detect DLL initialization failures (missing VC++ Redistributables)."""
    lower = output.lower()
    patterns = [
        "winerror 1114",
        "dll initialization routine failed",
        "dll load failed",
        "_load_dll_libraries",
    ]
    return any(p in lower for p in patterns)


def get_vcpp_help() -> str:
    """Get actionable help for DLL init errors (missing VC++ Redistributables)."""
    return (
        "A required DLL failed to initialize.\n\n"
        "Try these steps in order:\n"
        "  1. Install the latest VC++ Redistributable (x64):\n"
        "     https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
        "  2. Restart your computer after installing\n"
        "  3. If the error persists after reboot, click 'Reinstall Dependencies'\n"
        "     to force a clean reinstall of PyTorch\n"
        "  4. Check that no other Python (Anaconda, Miniconda, standalone Python)\n"
        "     puts conflicting torch DLLs on your system PATH.\n"
        "     Open a terminal and run: where python\n"
        "     If you see multiple results, remove the extra ones from PATH"
    )


# ---------------------------------------------------------------------------
# Antivirus / permission errors
# ---------------------------------------------------------------------------

def is_antivirus_error(stderr: str) -> bool:
    """Detect antivirus/permission blocking in pip output."""
    stderr_lower = stderr.lower()
    patterns = [
        "access is denied",
        "winerror 5",
        "winerror 225",
        "permission denied",
        "operation did not complete successfully because the file contains a virus",
        "blocked by your administrator",
        "blocked by group policy",
        "application control policy",
        "control de aplicaciones",
        "applocker",
        "blocked by your organization",
    ]
    return any(p in stderr_lower for p in patterns)


def get_pip_antivirus_help(venv_dir: str) -> str:
    """Get actionable help message for antivirus blocking pip."""
    steps = (
        "Installation was blocked, likely by antivirus software "
        "or security policy.\n\n"
        "Please try:\n"
        "  1. Temporarily disable real-time antivirus scanning\n"
        "  2. Add an exclusion for the plugin folder:\n"
        f"     {venv_dir}\n"
    )
    if sys.platform == "win32":
        steps += (
            "  3. Run QGIS as administrator "
            "(right-click > Run as administrator)\n"
            "  4. Try the installation again"
        )
    else:
        steps += (
            "  3. Check folder permissions: "
            f'chmod -R u+rwX "{venv_dir}"\n'
            "  4. Try the installation again"
        )
    return steps


# ---------------------------------------------------------------------------
# Windows crash / dist-info errors
# ---------------------------------------------------------------------------

# Windows NTSTATUS crash codes (both signed and unsigned representations)
_WINDOWS_CRASH_CODES = {
    3221225477,   # 0xC0000005 unsigned - ACCESS_VIOLATION
    -1073741819,  # 0xC0000005 signed   - ACCESS_VIOLATION
    3221225725,   # 0xC00000FD unsigned - STACK_OVERFLOW
    -1073741571,  # 0xC00000FD signed   - STACK_OVERFLOW
    3221225781,   # 0xC0000135 unsigned - DLL_NOT_FOUND
    -1073741515,  # 0xC0000135 signed   - DLL_NOT_FOUND
}


def is_windows_process_crash(returncode: int) -> bool:
    """Detect Windows process crashes (ACCESS_VIOLATION, STACK_OVERFLOW, etc.)."""
    if sys.platform != "win32":
        return False
    return returncode in _WINDOWS_CRASH_CODES


def is_rename_or_record_error(output: str) -> bool:
    """Detect dist-info rename/RECORD errors during torch upgrade on Windows."""
    lower = output.lower()
    if "rename" in lower and "dist-info" in lower:
        return True
    if "record" in lower and "dist-info" in lower:
        return True
    # uv may truncate "dist-info" from output
    return bool("failed to install" in lower and "failed to rename" in lower)


def get_crash_help(venv_dir: str) -> str:
    """Get actionable help for Windows process crash during pip install."""
    return (
        "The installer process crashed unexpectedly (access violation).\n\n"
        "This is usually caused by:\n"
        "  - Antivirus software (Windows Defender, etc.) blocking pip\n"
        "  - Corrupted virtual environment\n\n"
        "Please try:\n"
        "  1. Temporarily disable real-time antivirus scanning\n"
        "  2. Add an exclusion for the plugin folder:\n"
        f"     {venv_dir}\n"
        "  3. Click 'Reinstall Dependencies' to recreate the environment\n"
        "  4. If the issue persists, run QGIS as administrator"
    )
