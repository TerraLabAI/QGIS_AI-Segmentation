"""Pip/uv install error detection and user-facing help messages.

Pure functions that classify pip/uv stderr/stdout output into actionable
error categories. Reusable across any QGIS plugin that installs Python
packages via pip or uv.

Each keyword table below is the SHIPPED base of a server-extensible vocabulary
(``install_config.classifier_markers``, one ``install.classifier.*`` key per
classifier). The union is additive, so a deploy can teach the fleet a phrase a
newer installer or a system language started printing, and can never drop a
shipped phrase. Reading the configuration is cache-only and never raises, so an
offline machine classifies exactly on what shipped.

Two things stay out of that: the numeric OS-code tables, whose regexes carry
word-boundary anchoring a served string list could not express, and the order
the classifiers run in, which is written into the call sites and carries
meaning of its own.
"""
from __future__ import annotations

import re
import sys

from .install_config import classifier_markers

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
    return any(p in stderr_lower for p in classifier_markers("ssl", _SSL_ERROR_PATTERNS))


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


def is_untrusted_certificate_error(error_text: str) -> bool:
    """True when the download reached the server and refused its certificate.

    That is a different problem from a blocked address, and it needs different
    advice: the connection works, but something re-signed it with a certificate
    this machine does not vouch for.
    """
    lower = error_text.lower()
    patterns = [
        "unknownissuer",
        "unable to get local issuer certificate",
        "self signed certificate",
        "self-signed certificate",
        "certificate verify failed",
        "invalid peer certificate",
        "certificate is not trusted",
    ]
    return any(p in lower for p in patterns)


def get_ssl_error_help(error_text: str = "", cache_dir: str = "") -> str:
    """Get actionable help message for SSL errors.

    Differentiates between SSL module missing (broken Python), a certificate
    this machine does not trust, and a blocked address.
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
    if is_untrusted_certificate_error(error_text):
        return (
            "Installation failed: the download server presented a certificate "
            "this computer does not trust.\n\n"
            "Your network inspects secure connections and re-signs them with "
            "its own certificate, and that certificate is not in the "
            "computer's certificate store.\n\n"
            "Ask your IT department to either:\n"
            "  - install the network's root certificate on this machine, or\n"
            "  - exclude pypi.org, files.pythonhosted.org and "
            "download.pytorch.org from inspection."
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
    # uv (Rust/reqwest) vocabulary: none of the Python spellings above appear
    # in its output, so its network failures fell into the generic bucket.
    "request failed after",        # "error: Request failed after 5 retries"
    "error sending request",       # reqwest transport error
    "client error (connect)",      # reqwest connect phase
    "dns error",
    "operation timed out",
]


def is_network_error(output: str) -> bool:
    """Detect transient network/connection errors in pip output."""
    output_lower = output.lower()
    # Exclude SSL errors - they have their own retry path
    if is_ssl_error(output):
        return False
    return any(
        p in output_lower for p in classifier_markers("network", _NETWORK_ERROR_PATTERNS))


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
    return any(p in output_lower for p in classifier_markers("proxy_auth", patterns))


# ---------------------------------------------------------------------------
# Package index refusing the download outright (HTTP 403)
# ---------------------------------------------------------------------------

_INDEX_FORBIDDEN_PATTERNS = [
    "403 forbidden",
    "client error (403",
    "403 client error",
    "status client error (403",
]


def is_index_forbidden_error(output: str) -> bool:
    """Detect a package index or mirror answering the download with HTTP 403.

    Different from 407, where the proxy asks for credentials, and from a
    transient network fault: the request reaches a host that is entitled to
    answer and refuses. A filtering gateway or a locked-down internal mirror
    does that. Retrying never clears it, so the advice has to be "get the
    package index allowed", not "check your connection".
    """
    return any(
        p in output.lower()
        for p in classifier_markers("index_forbidden", _INDEX_FORBIDDEN_PATTERNS)
    )


# ---------------------------------------------------------------------------
# Disk-full errors (can happen mid-install, after the 4 GB preflight passed)
# ---------------------------------------------------------------------------

_DISK_FULL_PATTERNS = [
    "no space left on device",
    "enospc",
    "not enough space on the disk",      # en (Windows)
    "there is not enough space on the disk",
    "espace disque insuffisant",         # fr
    "espacio en disco insuficiente",     # es
    "espaco em disco insuficiente",      # pt-BR (no cedilla, defensive)
    "espaço em disco insuficiente",      # pt-BR
]

# The numeric forms: 28 is POSIX ENOSPC, 112 is Windows ERROR_DISK_FULL. Both
# are needed, because the prose above only covers four of the languages Windows
# ships in and everything else falls through to the wrong classifier. Anchored
# with a word boundary so a code is never read as the prefix of a longer one.
_DISK_FULL_CODE_RE = re.compile(r"(?:errno|os error|winerror)\s+(?:28|112)\b")


def is_disk_full(output: str) -> bool:
    """Detect out-of-disk errors during install.

    A 4 GB preflight runs before install, but torch + CUDA wheels can still
    exhaust the disk mid-extract. Must be checked BEFORE is_antivirus_error:
    a failed write from a full disk also surfaces as a permission/access
    error on Windows, which the antivirus classifier would misattribute.
    """
    lower = output.lower()
    if _DISK_FULL_CODE_RE.search(lower):
        return True
    return any(p in lower for p in classifier_markers("disk_full", _DISK_FULL_PATTERNS))


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
    no_match_phrases = (
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
    has_no_match = any(
        phrase in lower for phrase in classifier_markers("glibc", no_match_phrases))
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
    return not is_antivirus_error(output)


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
# Application-control policy blocks (AppLocker / WDAC style)
# ---------------------------------------------------------------------------

# Corporate application-control policies block unsigned executables and DLLs
# (the standalone Python, uv, and package DLLs the plugin installs). Unlike an
# antivirus quarantine, neither disabling real-time scanning nor running QGIS
# as administrator helps: only an IT-managed allow rule does. (#82)
_APP_CONTROL_PATTERNS = [
    # "an application control policy has blocked this file"
    "application control",
    "applocker",
    "blocked by group policy",
    "blocked by your organization",
    # Locale-independent numeric forms: Python "[WinError 4551]" and
    # uv "(os error 4551)".
    "winerror 4551",
    "os error 4551",
    # Localized Windows wording for an application-control policy block;
    # the OS message ships in the system language, so the English phrase
    # above never matches on these machines.
    "control de aplicaciones",                # es
    "strategie de controle d'application",    # fr (accents stripped in logs)
    "stratégie de contrôle d'application",    # fr
    "beleid voor toepassingsbeheer",          # nl
    # "This program is blocked by group policy" also arrives as WinError 1260
    # (ERROR_ACCESS_DISABLED_BY_POLICY). The numeric form is locale-independent;
    # the localized phrases cover logs where only the OS sentence survives.
    "winerror 1260",
    "os error 1260",
    "directiva de grupo",                     # es
    "diretiva de grupo",                      # pt-BR
    "gruppenrichtlinie",                      # de
    "groepsbeleid",                           # nl
    "strategie de groupe",                    # fr (accents stripped in logs)
    "stratégie de groupe",                    # fr
    "criteri di gruppo",                      # it
]


def is_app_control_error(output: str) -> bool:
    """Detect an application-control policy block (AppLocker/WDAC style).

    A strict subset of is_antivirus_error: policy blocks need different
    guidance (a path-based allow rule set by IT) because the antivirus
    advice (disable scanning, run as admin) is a dead end against a
    centrally managed policy.
    """
    lower = output.lower()
    return any(p in lower for p in classifier_markers("app_control", _APP_CONTROL_PATTERNS))


def get_app_control_help(install_dir: str = "") -> str:
    """Actionable help when an application-control policy blocks the plugin.

    The binaries the plugin installs are downloaded from their official
    open-source sources and are not code-signed, so strict policies block
    them. A path-based allow rule on the install folder survives plugin
    updates (no per-update re-signing), which is why it is the recommended
    fix for IT departments.
    """
    location = install_dir or "~/.qgis_ai_segmentation"
    return (
        "Your organization's security policy (application control, "
        "e.g. AppLocker or WDAC)\n"
        "is blocking the plugin's local AI environment.\n\n"
        "Disabling antivirus or running QGIS as administrator will not help.\n\n"
        "Ask your IT department to add a path-based allow rule "
        "for this folder:\n"
        f"  {location}\n\n"
        "The plugin always uses this folder, so one rule keeps working "
        "across updates.\n"
        "It contains a standalone Python runtime, the uv installer and "
        "Python packages,\n"
        "all downloaded from their official open-source sources.\n\n"
        "Once the rule is in place, restart QGIS and try again."
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


# Numeric codes that mean a scanner or a policy held the file:
# 5 ERROR_ACCESS_DENIED, 110 ERROR_OPEN_FAILED (scan race on a new wheel),
# 225 ERROR_VIRUS_INFECTED, and os error 13 for uv's POSIX EACCES.
# Anchored with a word boundary: matched as a bare substring, "winerror 5" also
# covers the whole 50-to-59 network and device block, so "[winerror 53] the
# network path was not found" told a user on a mapped drive to disable their
# antivirus and run QGIS as administrator.
_BLOCKED_ERROR_CODE_RE = re.compile(
    r"winerror\s+(?:5|110|225)\b|os error\s+(?:5|13)\b")


def is_antivirus_error(stderr: str) -> bool:
    """Detect antivirus/permission blocking in pip/uv output.

    Also catches corporate application-control policies (AppLocker, etc.)
    and localized Windows "access denied" messages.
    """
    stderr_lower = stderr.lower()
    if _BLOCKED_ERROR_CODE_RE.search(stderr_lower):
        return True
    patterns = [
        *classifier_markers("access_denied", _ACCESS_DENIED_LOCALIZED),
        # Application-control policy blocks are a subset: is_app_control_error
        # narrows them down for the targeted IT allow-rule guidance.
        *classifier_markers("app_control", _APP_CONTROL_PATTERNS),
        # Windows text for ERROR_OPEN_FAILED; seen during install of fresh
        # wheels (e.g. tqdm) when Defender holds the file during real-time
        # scan. HRESULT 0x8007006E shows up as signed "os error -2147024786".
        "cannot open the device or file",
        "permission denied",
        "operation did not complete successfully because the file contains a virus",
        "blocked by your administrator",
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
    if any(p in lower for p in classifier_markers("access_denied", _ACCESS_DENIED_LOCALIZED)):
        return True
    # Anchored, for the same reason as _BLOCKED_ERROR_CODE_RE above: the bare
    # substring "os error 5" is a prefix of the whole 50-to-59 network block,
    # so a cache on a mapped drive ("the network path was not found (os error
    # 53)") read as a locked binary and sent the user to restart QGIS.
    return bool(_BLOCKED_ERROR_CODE_RE.search(lower)) or "permission denied" in lower


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


def get_pip_antivirus_help(exclude_dir: str) -> str:
    """Get actionable help message for antivirus blocking pip.

    `exclude_dir` must be the whole plugin cache folder, never the venv alone.
    Callers used to pass the venv, and a real 2026-07-27 install failure was
    antivirus locking a file under `uv_cache`, a SIBLING of the venv: the user
    was told to exclude a folder that could not stop the failure they had.
    """
    steps = (
        "Installation was blocked, likely by antivirus software "
        "or security policy.\n\n"
        "Please try:\n"
        "  1. Temporarily disable real-time antivirus scanning\n"
        "  2. Add an exclusion for the plugin folder:\n"
        f"     {exclude_dir}\n"
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
            f'chmod -R u+rwX "{exclude_dir}"\n'
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


# ---------------------------------------------------------------------------
# Invalid install path (cloud-synced, special characters, over-long)
# ---------------------------------------------------------------------------

_INVALID_PATH_PATTERNS = [
    # pip: "OSError: [Errno 22] Invalid argument: 'C:\\Users\\...'"
    "errno 22",
    # Windows "The filename, directory name, or volume label syntax is
    # incorrect", numeric forms first (locale-independent).
    "winerror 123",
    "os error 123",
    "volume label syntax is incorrect",
    # Windows path-length limit (260 chars) hit inside a deep venv tree.
    "winerror 206",
    "os error 206",
]


def is_invalid_path_error(output: str) -> bool:
    """Detect a file path the OS refuses to write during install.

    Seen in production as pip's "Could not install packages due to an
    OSError: [Errno 22] Invalid argument: '<path>'" on Windows homes that
    live under a cloud-synced or redirected folder (OneDrive placeholders
    reject normal writes), contain unusual characters, or exceed the legacy
    260-character path limit.
    """
    lower = output.lower()
    return any(p in lower for p in _INVALID_PATH_PATTERNS)


def get_invalid_path_help(cache_dir: str = "") -> str:
    """Actionable help when the install path itself is the problem."""
    location = cache_dir or "~/.qgis_ai_segmentation"
    return (
        "Windows refused a file path during installation.\n\n"
        "This usually means the install folder is cloud-synced "
        "(OneDrive/Dropbox), contains unusual characters, or the path grew "
        "past the Windows length limit.\n\n"
        f"The environment installs under: {location}\n\n"
        "Please try:\n"
        "  1. If that folder is inside OneDrive or another sync tool, pause\n"
        "     syncing (or mark the folder 'Always keep on this device')\n"
        "  2. Or set the AI_SEGMENTATION_CACHE_DIR environment variable to a\n"
        "     short local folder outside any synced area (e.g. C:\\qgis_ai),\n"
        "     then restart QGIS\n"
        "  3. Click 'Install Dependencies' again"
    )


# ---------------------------------------------------------------------------
# Broken local Python runtime (gutted stdlib)
# ---------------------------------------------------------------------------

_BROKEN_RUNTIME_PATTERNS = [
    # The interpreter cannot even start: stdlib missing or unreachable.
    "no module named 'encodings'",
    "no module named encodings",
    "init_fs_encoding",
    "failed to get the python codec of the filesystem encoding",
]


def is_broken_python_runtime(output: str) -> bool:
    """Detect a damaged local Python runtime (stdlib gutted or unreachable).

    Antivirus quarantine or an interrupted first install can leave the
    standalone Python (or the venv that points at it) unable to boot at all:
    "Fatal Python error: init_fs_encoding ... No module named 'encodings'".
    Every later package build then fails with the same wall of text.
    """
    lower = output.lower()
    return any(
        p in lower for p in classifier_markers("broken_runtime", _BROKEN_RUNTIME_PATTERNS))


def get_broken_python_runtime_help(cache_dir: str = "") -> str:
    """Actionable help for a damaged local Python runtime."""
    location = cache_dir or "~/.qgis_ai_segmentation"
    return (
        "The plugin's local Python runtime is damaged and cannot start.\n"
        "This is usually caused by antivirus quarantine or an interrupted\n"
        "first installation.\n\n"
        "The next installation will rebuild it from scratch automatically.\n\n"
        "Please try:\n"
        "  1. Add an antivirus exclusion for the folder:\n"
        f"     {location}\n"
        "  2. Click 'Install Dependencies' again to rebuild everything"
    )


# ---------------------------------------------------------------------------
# Corrupt virtual environment (survives superficial checks)
# ---------------------------------------------------------------------------

_CORRUPT_VENV_PATTERNS = [
    # venv python without its bundled pip: "python.exe: No module named pip"
    "no module named pip",
    "no module named 'pip'",
    # uv cannot read the venv's interpreter metadata
    "failed to inspect python interpreter",
]


def is_corrupt_venv(output: str) -> bool:
    """Detect a virtual environment too damaged to install into.

    The venv directory exists (so quick checks pass) but its interpreter or
    bundled pip is gone: cleanup tools, quarantine, or a kill mid-build.
    """
    lower = output.lower()
    return any(p in lower for p in classifier_markers("corrupt_venv", _CORRUPT_VENV_PATTERNS))


def get_corrupt_venv_help() -> str:
    """Actionable help for a corrupt virtual environment."""
    return (
        "The plugin's Python environment is damaged (files are missing "
        "inside it).\n\n"
        "Click 'Install Dependencies' again: the environment will be "
        "rebuilt from scratch automatically."
    )


# ---------------------------------------------------------------------------
# Dependency resolution conflicts
# ---------------------------------------------------------------------------

_DEPENDENCY_CONFLICT_PATTERNS = [
    "conflicting dependencies",       # pip
    "resolutionimpossible",           # pip resolver class name
    "no solution found when resolving",  # uv
]


def is_dependency_conflict(output: str) -> bool:
    """Detect a package-resolver conflict (no installable version set)."""
    lower = output.lower()
    return any(
        p in lower
        for p in classifier_markers("dependency_conflict", _DEPENDENCY_CONFLICT_PATTERNS))


def get_dependency_conflict_help() -> str:
    """Actionable help for a dependency-resolution conflict."""
    return (
        "The package resolver could not find a compatible set of versions.\n"
        "This usually comes from stale cached package data or a Python\n"
        "version the AI packages no longer support.\n\n"
        "Please try:\n"
        "  1. Click 'Reinstall Dependencies' to rebuild with fresh data\n"
        "  2. If it persists, update QGIS to the latest LTR release\n"
        "     (newer QGIS ships a newer Python) and try again"
    )
