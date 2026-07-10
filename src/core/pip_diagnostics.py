"""Pip/uv install error detection and user-facing help messages.

Pure functions that classify pip/uv stderr/stdout output into actionable
error categories. Reusable across any QGIS plugin that installs Python
packages via pip or uv.
"""
from __future__ import annotations

import re
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
    # uv is written in Rust: its TLS errors come from rustls, not OpenSSL,
    # so none of the patterns above match them. Seen behind corporate
    # MITM proxies: "invalid peer certificate: UnknownIssuer".
    "invalid peer certificate",
    "unknownissuer",
    "self-signed certificate",
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
    after a verified SSL error - never as default install flags.

    Intended for users behind corporate proxies with custom CA certificates
    that pip/uv cannot verify against their bundled certificate store.
    """
    return [
        "--trusted-host", "pypi.org",
        "--trusted-host", "pypi.python.org",
        "--trusted-host", "files.pythonhosted.org",
    ]


def is_ssl_module_missing(error_text: str) -> bool:
    """Check if the error is about a missing SSL module (not a certificate issue).

    Seen on Windows 7 + Python 3.9 QGIS shipments where the embedded Python
    lacks `_ssl`, breaking all pip calls. The classifier-driven help text
    nudges users toward reinstalling QGIS on a supported OS.
    """
    lower = error_text.lower()
    patterns = [
        "ssl module is not available",
        "no module named '_ssl'",
        "ssl module",
        "importerror: _ssl",
        "can't connect to https url because the ssl module is not available",
        "the ssl module in python is not available",
    ]
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
    # Localized DNS variants. Scoped to the plugin's officially supported
    # locales (fr, pt-BR, es) - the OS-language error bubbles up via getaddrinfo
    # even when the plugin UI is in English, so missing these silently
    # mis-classifies real DNS failures as generic install errors.
    "hôte inconnu",                # fr
    "hote inconnu",                # fr (no accent, defensive)
    "host non trouvé",             # fr (alt)
    "host desconhecido",           # pt-BR
    "host desconocido",            # es
]


def is_network_error(output: str) -> bool:
    """Detect transient network/connection errors in pip output."""
    output_lower = output.lower()
    # Exclude SSL errors - they have their own retry path
    if is_ssl_error(output):
        return False
    return any(p in output_lower for p in _NETWORK_ERROR_PATTERNS)


def is_proxy_auth_error(output: str) -> bool:
    """Detect proxy authentication errors (HTTP 407).

    pip/OpenSSL and uv/rustls word this differently. pip says "407 Proxy
    Authentication Required" / "ProxyError"; uv (Rust) surfaces the CONNECT
    tunnel rejection as "tunnel error: proxy authorization required" (note
    "authoriZation", a different spelling). Both mean the same thing: the
    corporate proxy needs credentials the installer was not given.
    """
    output_lower = output.lower()
    patterns = [
        "407 proxy authentication",
        "proxy authentication required",
        "proxyerror",
        # uv / rustls vocabulary (seen behind authenticated corporate proxies)
        "proxy authorization required",
        "tunnel error",
    ]
    return any(p in output_lower for p in patterns)


# ---------------------------------------------------------------------------
# Disk-full errors (can happen mid-install, after the 4 GB preflight passed)
# ---------------------------------------------------------------------------

_DISK_FULL_PATTERNS = [
    "no space left on device",
    "errno 28",
    "enospc",
    "os error 28",                       # uv / Rust on POSIX
    "not enough space on the disk",      # en (Windows)
    "there is not enough space on the disk",
    "espace disque insuffisant",         # fr
    "espacio en disco insuficiente",     # es
    "espaco em disco insuficiente",      # pt-BR (no cedilla, defensive)
    "espaço em disco insuficiente",      # pt-BR
]


def is_disk_full(output: str) -> bool:
    """Detect out-of-disk errors during install.

    A 4 GB preflight runs before install, but torch + CUDA wheels can still
    exhaust the disk mid-extract. Must be checked BEFORE is_antivirus_error:
    a failed write from a full disk also surfaces as a permission/access
    error on Windows, which the antivirus classifier would misattribute.
    """
    lower = output.lower()
    return any(p in lower for p in _DISK_FULL_PATTERNS)


def get_disk_full_help(cache_dir: str = "") -> str:
    """Actionable help for a disk-full install failure."""
    location = cache_dir or "~/.qgis_ai_segmentation"
    return (
        "Installation failed: your disk ran out of space.\n\n"
        "The AI engine needs roughly 4 GB free during installation.\n"
        "Please try:\n"
        "  1. Free up disk space (empty the trash, remove large unused files)\n"
        f"  2. The environment is installed under: {location}\n"
        "  3. To install on another drive, set the AI_SEGMENTATION_CACHE_DIR\n"
        "     environment variable to a folder on a disk with more space,\n"
        "     then restart QGIS and try again"
    )


# ---------------------------------------------------------------------------
# Linux glibc-too-old errors (Ubuntu 18.04, CentOS 7, ...)
# ---------------------------------------------------------------------------

def is_glibc_too_old(output: str) -> bool:
    """Detect a glibc older than what modern PyTorch wheels require.

    Current torch wheels are manylinux_2_28: on older distros pip reports
    "no matching distribution" and the loader complains about a missing
    GLIBC_2.xx symbol. uv mentions the manylinux/glibc tag instead.
    """
    lower = output.lower()
    if re.search(r"glibc_2\.\d+'? not found", lower):
        return True
    has_manylinux = "manylinux" in lower
    has_no_match = any(
        phrase in lower
        for phrase in (
            # pip vocabulary
            "no matching distribution",
            "could not find a version",
            "is not a supported wheel",
            # uv (rustls) vocabulary: it names the platform tag instead
            "no wheels",
            "none of the wheels",
            "matching platform tag",
            "compatible with your platform",
            "compatible with the current platform",
        )
    )
    if has_manylinux and has_no_match:
        return True
    return bool("requires a newer" in lower and "glibc" in lower)


def get_glibc_too_old_help() -> str:
    """Actionable help for a glibc-too-old Linux install failure."""
    return (
        "Installation failed: your Linux distribution is too old for the\n"
        "current AI engine. PyTorch wheels now require a recent system\n"
        "library (glibc 2.28+, i.e. Ubuntu 20.04 / Debian 10 / CentOS 8 or\n"
        "newer).\n\n"
        "Please try:\n"
        "  1. Upgrade your distribution to a version released after 2019\n"
        "  2. If you cannot upgrade, this plugin's AI engine is unfortunately\n"
        "     not supported on this machine"
    )


# ---------------------------------------------------------------------------
# macOS Intel: no torch wheel for newer Python
# ---------------------------------------------------------------------------

def is_macos_intel_no_wheel(output: str) -> bool:
    """Detect 'no torch wheel' failures specific to Intel (x86_64) macOS.

    PyTorch's last Intel-mac release is 2.2.2 (cp38-cp312). With a newer
    Python (3.13+), no wheel exists and pip/uv report no matching
    distribution while naming the macosx_x86_64 platform tag.
    """
    if sys.platform != "darwin":
        return False
    lower = output.lower()
    mentions_torch = "torch" in lower
    no_match = any(
        phrase in lower
        for phrase in ("no matching distribution", "could not find a version")
    )
    mentions_x86 = "macosx" in lower and ("x86_64" in lower or "x86-64" in lower)
    return mentions_torch and no_match and mentions_x86


def get_macos_intel_help() -> str:
    """Actionable help for the Intel-mac unsupported-Python combination."""
    return (
        "Installation failed: no compatible AI engine build exists for this\n"
        "combination of Intel Mac and Python version.\n\n"
        "Intel (x86_64) Macs are supported only up to PyTorch 2.2.2, which\n"
        "ships for Python 3.8 to 3.12. Your Python is newer than that.\n\n"
        "Please try:\n"
        "  1. Use a QGIS build bundling Python 3.12 or older, or\n"
        "  2. On Apple Silicon, run the native (arm64) QGIS rather than the\n"
        "     Intel build under Rosetta"
    )


# ---------------------------------------------------------------------------
# Windows process / DLL errors
# ---------------------------------------------------------------------------

def is_unable_to_create_process(output: str) -> bool:
    """Detect 'unable to create process' errors on Windows (broken pip.exe shim)."""
    return "unable to create process" in output.lower()


def is_dll_init_error(output: str) -> bool:
    """Detect DLL initialization failures (missing VC++ Redistributables).

    A DLL that is *blocked by a security policy* (AppLocker, WDAC, corporate
    application-control) or quarantined by antivirus also prints "DLL load
    failed", but the fix there is to whitelist the plugin folder, not to install
    the VC++ runtime. Defer those to is_antivirus_error so the user gets the
    right guidance instead of a dead-end "install Visual C++" message
    (#bug-kees).
    """
    lower = output.lower()
    patterns = [
        "winerror 1114",
        "dll initialization routine failed",
        "dll load failed",
        "_load_dll_libraries",
    ]
    if not any(p in lower for p in patterns):
        return False
    # A policy/antivirus block is not a missing-runtime error; let the
    # antivirus classifier own it (it carries the whitelist guidance).
    if is_antivirus_error(output):
        return False
    return True


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
        "winerror 110",        # ERROR_OPEN_FAILED - AV scan race on new wheel
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
        # Windows "an application control policy has blocked this file",
        # locale-independent numeric forms (Python "[WinError 4551]" and
        # uv "(os error 4551)").
        "winerror 4551",
        "os error 4551",
        # Localized Windows wording for an application-control policy block;
        # the OS message ships in the system language, so the English phrase
        # above never matches on these machines.
        "strategie de controle d'application",   # fr (accents stripped in logs)
        "stratégie de contrôle d'application",   # fr
        "beleid voor toepassingsbeheer",         # nl
    ]
    return any(p in stderr_lower for p in patterns)


# Binary extensions whose presence in a "failed to remove" message points
# to file-locking by the current process, not to antivirus interference.
_BINARY_MODULE_EXTENSIONS = (".pyd", ".dll", ".so", ".dylib")


def is_file_locked_error(output: str) -> bool:
    """Detect native-module file-lock errors during package upgrade.

    These occur when a .pyd/.dll/.so is already imported into the current
    process - Windows and some filesystems refuse to delete an in-use
    binary. The fix is to close the app holding the module (restart QGIS),
    NOT to exclude the folder from antivirus.

    Example (uv on Windows, German locale):
        error: failed to remove file
        `...\\site-packages\\torch/_C.cp312-win_amd64.pyd`:
        Zugriff verweigert (os error 5)
    """
    lower = output.lower()
    # "failed to remove" + a binary extension + any access-denied variant
    has_remove_verb = any(
        phrase in lower
        for phrase in ("failed to remove", "could not remove", "unable to remove")
    )
    if not has_remove_verb:
        return False
    has_binary_ext = any(ext in lower for ext in _BINARY_MODULE_EXTENSIONS)
    if not has_binary_ext:
        return False
    if any(p in lower for p in _ACCESS_DENIED_LOCALIZED):
        return True
    return any(
        phrase in lower
        for phrase in ("os error 5", "os error 13", "winerror 5", "permission denied")
    )


def get_file_locked_help() -> str:
    """Action-only instructions for file-lock errors (no diagnostic text)."""
    return (
        "How to fix this:\n\n"
        "  1. Close all QGIS windows (File > Exit)\n"
        "  2. Reopen QGIS\n"
        "  3. Open the AI Segmentation panel - installation will resume\n\n"
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
