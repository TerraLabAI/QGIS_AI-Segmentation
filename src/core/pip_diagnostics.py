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


def get_pip_ssl_bypass_flags() -> list[str]:
    """Get pip flags to bypass SSL/TLS certificate verification.

    WARNING: These flags disable TLS verification for the listed hosts,
    exposing downloads to potential MITM attacks. Only use as a fallback
    after a verified SSL error — never as default install flags.

    Intended for users behind corporate proxies with custom CA certificates
    that pip/uv cannot verify against their bundled certificate store.
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

# Localized variants of "access is denied" from Windows error messages.
# uv surfaces the raw OS message (in the user's system language), so our
# English-only classifier was silently missing these cases (#bug-lukas).
_ACCESS_DENIED_LOCALIZED = [
    "access is denied",             # en
    "zugriff verweigert",           # de
    "acces refuse",                 # fr (no accents, defensive)
    "accès refusé",                 # fr
    "l'accès est refusé",           # fr (alt form)
    "acceso denegado",              # es
    "acesso negado",                # pt-BR / pt
    "accesso negato",               # it
    "toegang geweigerd",            # nl
    "отказано в доступе",           # ru
    "アクセスが拒否されました",       # ja
    "拒绝访问",                       # zh-CN
    "存取被拒",                       # zh-TW
]


def is_antivirus_error(stderr: str) -> bool:
    """Detect antivirus/permission blocking in pip/uv output.

    Also catches corporate application-control policies (AppLocker, etc.)
    and localized Windows "access denied" messages.
    """
    stderr_lower = stderr.lower()
    patterns = [
        *_ACCESS_DENIED_LOCALIZED,
        "winerror 5",
        "winerror 225",
        "winerror 110",        # ERROR_OPEN_FAILED — AV scan race on new wheel
        "os error 5",          # uv format on Windows (distinct from winerror 5)
        "os error 13",         # uv format on POSIX (EACCES)
        # Windows text for ERROR_OPEN_FAILED; seen during install of fresh
        # wheels (e.g. tqdm) when Defender holds the file during real-time
        # scan. HRESULT 0x8007006E shows up as signed "os error -2147024786".
        "cannot open the device or file",
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


# Binary extensions whose presence in a "failed to remove" message points
# to file-locking by the current process, not to antivirus interference.
_BINARY_MODULE_EXTENSIONS = (".pyd", ".dll", ".so", ".dylib")


def is_file_locked_error(output: str) -> bool:
    """Detect native-module file-lock errors during package upgrade.

    These occur when a .pyd/.dll/.so is already imported into the current
    process — Windows and some filesystems refuse to delete an in-use
    binary. The fix is to close the app holding the module (restart QGIS),
    NOT to exclude the folder from antivirus.

    Example (uv on Windows, German locale):
        error: failed to remove file
        `...\\site-packages\\torch/_C.cp312-win_amd64.pyd`:
        Zugriff verweigert (os error 5)
    """
    lower = output.lower()
    # "failed to remove" + a binary extension + any access-denied variant
    has_remove_verb = (
        "failed to remove" in lower
        or "could not remove" in lower
        or "unable to remove" in lower
    )
    if not has_remove_verb:
        return False
    has_binary_ext = any(ext in lower for ext in _BINARY_MODULE_EXTENSIONS)
    if not has_binary_ext:
        return False
    if any(p in lower for p in _ACCESS_DENIED_LOCALIZED):
        return True
    return (
        "os error 5" in lower
        or "os error 13" in lower
        or "winerror 5" in lower
        or "permission denied" in lower
    )


def get_file_locked_help() -> str:
    """Action-only instructions for file-lock errors (no diagnostic text)."""
    return (
        "How to fix this:\n\n"
        "  1. Close all QGIS windows (File > Exit)\n"
        "  2. Reopen QGIS\n"
        "  3. Open the AI Segmentation panel — installation will resume\n\n"
        "If it still fails after restarting QGIS:\n"
        "  4. Uninstall the plugin "
        "(Plugins > Manage and Install Plugins > Installed > AI Segmentation)\n"
        "  5. Restart QGIS\n"
        "  6. Reinstall the plugin"
    )


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
