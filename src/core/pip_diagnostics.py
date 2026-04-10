

















from __future__ import annotations

import re
import sys

from .install_config import classifier_markers


def tr(text: str) -> str:






    try:
        from .i18n import tr as translate

        return translate(text)
    except Exception:  # noqa: BLE001
        return text





def install_again_step() -> str:

    return tr("Open the AI Segmentation panel and click Install")






def _weights_hosts() -> tuple[str, ...]:







    try:
        from urllib.parse import urlparse

        from .install_config import checkpoint_source
        from .model_config import (
            CHECKPOINT_FILENAME,
            CHECKPOINT_SHA256,
            CHECKPOINT_URL,
        )

        url, _digest, mirrors = checkpoint_source(
            CHECKPOINT_FILENAME, CHECKPOINT_URL, CHECKPOINT_SHA256)
        hosts: list[str] = []
        for address in (url,) + tuple(mirrors or ()):
            host = urlparse(address or "").netloc or ""
            if host and host not in hosts:
                hosts.append(host)
        return tuple(hosts)
    except Exception:  # noqa: BLE001
        return ()


def first_run_hosts() -> tuple[str, ...]:









    hosts = [
        "pypi.org",
        "files.pythonhosted.org",
        "download.pytorch.org",
        "github.com",
        "objects.githubusercontent.com",
    ]
    for weights in _weights_hosts():
        if weights not in hosts:
            hosts.append(weights)
    return tuple(hosts)


def first_run_hosts_sentence() -> str:

    hosts = first_run_hosts()
    return ", ".join(hosts[:-1]) + " and " + hosts[-1]


def first_run_hosts_bullets(indent: str = "  - ") -> str:

    return "\n".join(indent + host for host in first_run_hosts())






_SSL_ERROR_PATTERNS = [
    "ssl error",
    "ssl:",
    "sslerror",
    "sslcertverificationerror",
    "certificate verify failed",
    "certificate_verify_failed",
    "tlsv1 alert",
    "unable to get local issuer certificate",
    "self signed certificate in certificate chain",



    "invalid peer certificate",
    "unknownissuer",
    "self-signed certificate",
]


def is_ssl_error(stderr: str) -> bool:

    stderr_lower = stderr.lower()
    return any(p in stderr_lower for p in classifier_markers("ssl", _SSL_ERROR_PATTERNS))


def is_hash_mismatch(output: str) -> bool:

    output_lower = output.lower()
    return "do not match the hashes" in output_lower or "hash mismatch" in output_lower


def get_pip_ssl_bypass_flags() -> list[str]:









    return [
        "--trusted-host", "pypi.org",
        "--trusted-host", "pypi.python.org",
        "--trusted-host", "files.pythonhosted.org",
    ]


def is_ssl_module_missing(error_text: str) -> bool:






    lower = error_text.lower()
    patterns = [
        "ssl module is not available",
        "no module named '_ssl'",
        "ssl module",
        "importerror: _ssl",
        "can't connect to https url because the ssl module is not available",
        "the ssl module in python is not available",
    ]
    return any(p in lower for p in classifier_markers("ssl_module", patterns))


def is_untrusted_certificate_error(error_text: str) -> bool:






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
    return any(
        p in lower for p in classifier_markers("untrusted_certificate", patterns))


def get_ssl_error_help(error_text: str = "", cache_dir: str = "") -> str:





    if is_ssl_module_missing(error_text):
        return (
            tr("Installation failed: Python's SSL module is not available.") + "\n\n"
            + tr("This usually means the Python installation is incomplete or corrupted.")
            + "\n"
            + tr("Please try:") + "\n"
            + tr("  1. Delete the folder: {folder}").format(folder=cache_dir) + "\n"
            + tr("  2. Restart QGIS and try again") + "\n"
            + tr("  3. If the issue persists, reinstall QGIS")
        )
    if is_untrusted_certificate_error(error_text):
        return (
            tr(
                "Installation failed: the download server presented a certificate "
                "this computer does not trust."
            ) + "\n\n"
            + tr(
                "Your network inspects secure connections and re-signs them with "
                "its own certificate, and that certificate is not in the "
                "computer's certificate store."
            ) + "\n\n"
            + tr("Ask your IT department to either:") + "\n"
            + tr("  - install the network's root certificate on this machine, or") + "\n"
            + tr("  - exclude these hosts from inspection:") + "\n"
            + f"{first_run_hosts_bullets('      - ')}"
        )
    return (
        tr("Installation failed due to network restrictions.") + "\n\n"
        + tr("Please contact your IT department to allow access to:") + "\n"
        + f"{first_run_hosts_bullets()}\n\n"
        + tr(
            "You can also try checking your proxy settings in QGIS "
            "(Settings > Options > Network)."
        )
    )






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
    "urllib3.exceptions",
    "requests.exceptions.connectionerror",
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
    "network timeout",
    "failed to download",




    "hôte inconnu",
    "hote inconnu",
    "host non trouvé",
    "host desconhecido",
    "host desconocido",


    "request failed after",
    "error sending request",
    "client error (connect)",
    "dns error",
    "operation timed out",
]


def is_network_error(output: str) -> bool:

    output_lower = output.lower()

    if is_ssl_error(output):
        return False
    return any(
        p in output_lower for p in classifier_markers("network", _NETWORK_ERROR_PATTERNS))


def is_proxy_auth_error(output: str) -> bool:








    output_lower = output.lower()
    patterns = [
        "407 proxy authentication",
        "proxy authentication required",
        "proxyerror",

        "proxy authorization required",
        "tunnel error",
    ]
    return any(p in output_lower for p in classifier_markers("proxy_auth", patterns))






_INDEX_FORBIDDEN_PATTERNS = [
    "403 forbidden",
    "client error (403",
    "403 client error",
    "status client error (403",
]


def is_index_forbidden_error(output: str) -> bool:








    return any(
        p in output.lower()
        for p in classifier_markers("index_forbidden", _INDEX_FORBIDDEN_PATTERNS)
    )






_DISK_FULL_PATTERNS = [
    "no space left on device",
    "enospc",
    "not enough space on the disk",
    "there is not enough space on the disk",
    "espace disque insuffisant",
    "espacio en disco insuficiente",
    "espaco em disco insuficiente",
    "espaço em disco insuficiente",
]





_DISK_FULL_CODE_RE = re.compile(r"(?:errno|os error|winerror)\s+(?:28|112)\b")


def is_disk_full(output: str) -> bool:







    lower = output.lower()
    if _DISK_FULL_CODE_RE.search(lower):
        return True
    return any(p in lower for p in classifier_markers("disk_full", _DISK_FULL_PATTERNS))


def _min_free_gb_full_display() -> str:








    try:
        from .venv_manager import resolved_min_free_gb_full

        return f"{resolved_min_free_gb_full():.0f}"
    except Exception:  # noqa: BLE001
        return "4"


def get_disk_full_help(cache_dir: str = "") -> str:

    location = cache_dir or "~/.qgis_ai_segmentation"
    return (
        tr("Installation failed: your disk ran out of space.") + "\n\n"
        + tr("The AI engine needs roughly {gb} GB free during installation.").format(
            gb=_min_free_gb_full_display()) + "\n"
        + tr("Please try:") + "\n"
        + tr("  1. Free up disk space (empty the trash, remove large unused files)") + "\n"
        + tr("  2. The environment is installed under: {location}").format(
            location=location) + "\n"
        + tr(
            "  3. To install on another drive, set the AI_SEGMENTATION_CACHE_DIR\n"
            "     environment variable to a folder on a disk with more space,\n"
            "     then restart QGIS and try again"
        )
    )










_NO_WHEEL_PHRASES = (

    "no matching distribution",
    "could not find a version",
    "is not a supported wheel",

    "no wheels",
    "none of the wheels",
    "matching platform tag",
    "compatible with your platform",
    "compatible with the current platform",
)


def is_glibc_too_old(output: str) -> bool:






    lower = output.lower()
    if re.search(r"glibc_2\.\d+'? not found", lower):
        return True
    has_manylinux = "manylinux" in lower
    has_no_match = any(
        phrase in lower for phrase in classifier_markers("glibc", _NO_WHEEL_PHRASES))
    if has_manylinux and has_no_match:
        return True
    return bool("requires a newer" in lower and "glibc" in lower)


def get_glibc_too_old_help() -> str:

    return (
        tr(
            "Installation failed: your Linux distribution is too old for the\n"
            "current AI engine. PyTorch wheels now require a recent system\n"
            "library (glibc 2.28+, i.e. Ubuntu 20.04 / Debian 10 / CentOS 8 or\n"
            "newer)."
        ) + "\n\n"
        + tr("Please try:") + "\n"
        + tr("  1. Upgrade your distribution to a version released after 2019") + "\n"
        + tr(
            "  2. If you cannot upgrade, this plugin's AI engine is unfortunately\n"
            "     not supported on this machine"
        )
    )






def is_macos_intel_no_wheel(output: str) -> bool:








    if sys.platform != "darwin":
        return False
    lower = output.lower()
    mentions_torch = "torch" in lower
    no_match = any(
        phrase in lower
        for phrase in classifier_markers("macos_intel", _NO_WHEEL_PHRASES)
    )
    mentions_x86 = "macosx" in lower and ("x86_64" in lower or "x86-64" in lower)
    return mentions_torch and no_match and mentions_x86


def get_macos_intel_help() -> str:

    return (
        tr(
            "Installation failed: no compatible AI engine build exists for this\n"
            "combination of Intel Mac and Python version."
        ) + "\n\n"
        + tr(
            "Intel (x86_64) Macs are supported only up to PyTorch 2.2.2, which\n"
            "ships for Python 3.8 to 3.12. Your Python is newer than that."
        ) + "\n\n"
        + tr("Please try:") + "\n"
        + tr("  1. Use a QGIS build bundling Python 3.12 or older, or") + "\n"
        + tr(
            "  2. On Apple Silicon, run the native (arm64) QGIS rather than the\n"
            "     Intel build under Rosetta"
        )
    )






def is_unable_to_create_process(output: str) -> bool:

    return "unable to create process" in output.lower()


def is_dll_init_error(output: str) -> bool:









    lower = output.lower()
    patterns = [
        "winerror 1114",
        "dll initialization routine failed",
        "dll load failed",
        "_load_dll_libraries",
    ]
    if not any(p in lower for p in classifier_markers("dll_init", patterns)):
        return False


    return not is_antivirus_error(output)


def get_vcpp_help() -> str:

    from .interaction_dials import vcredist_url
    vc_redist_url = vcredist_url("https://aka.ms/vs/17/release/vc_redist.x64.exe")
    return (
        tr("A required DLL failed to initialize.") + "\n\n"
        + tr("Try these steps in order:") + "\n"
        + tr("  1. Install the latest VC++ Redistributable (x64):\n     {url}").format(
            url=vc_redist_url) + "\n"
        + tr("  2. Restart your computer after installing") + "\n"
        + tr("  3. If the error is still there after the reboot:") + "\n"
        + tr("     {step} to build the AI engine again").format(
            step=install_again_step()) + "\n"
        + tr(
            "  4. Check that no other Python (Anaconda, Miniconda, standalone Python)\n"
            "     puts conflicting torch DLLs on your system PATH.\n"
            "     Open a terminal and run: where python\n"
            "     If you see multiple results, remove the extra ones from PATH"
        )
    )










_APP_CONTROL_PATTERNS = [

    "application control",
    "applocker",
    "blocked by group policy",
    "blocked by your organization",


    "winerror 4551",
    "os error 4551",



    "control de aplicaciones",
    "strategie de controle d'application",
    "stratégie de contrôle d'application",
    "beleid voor toepassingsbeheer",



    "winerror 1260",
    "os error 1260",
    "directiva de grupo",
    "diretiva de grupo",
    "gruppenrichtlinie",
    "groepsbeleid",
    "strategie de groupe",
    "stratégie de groupe",
    "criteri di gruppo",
]


def is_app_control_error(output: str) -> bool:







    lower = output.lower()
    return any(p in lower for p in classifier_markers("app_control", _APP_CONTROL_PATTERNS))


def get_app_control_help(install_dir: str = "") -> str:








    location = install_dir or "~/.qgis_ai_segmentation"
    return (
        tr(
            "Your organization's security policy (application control, "
            "e.g. AppLocker or WDAC)\n"
            "is blocking the plugin's local AI environment."
        ) + "\n\n"
        + tr("Disabling antivirus or running QGIS as administrator will not help.") + "\n\n"
        + tr(
            "Ask your IT department to add a path-based allow rule "
            "for this folder:"
        ) + "\n"
        + f"  {location}\n\n"
        + tr(
            "The plugin always uses this folder, so one rule keeps working "
            "across updates.\n"
            "It contains a standalone Python runtime, the uv installer and "
            "Python packages,\n"
            "all downloaded from their official open-source sources."
        ) + "\n\n"
        + tr("Once the rule is in place, restart QGIS and try again.")
    )










_ACCESS_DENIED_LOCALIZED = [
    "access is denied",
    "zugriff verweigert",
    "acces refuse",
    "accès refusé",
    "l'accès est refusé",
    "acceso denegado",
    "acesso negado",
    "accesso negato",
    "toegang geweigerd",
    "отказано в доступе",
    "アクセスが拒否されました",
    "拒绝访问",
    "存取被拒",
]









_BLOCKED_ERROR_CODE_RE = re.compile(
    r"winerror\s+(?:5|110|225)\b|os error\s+(?:5|13)\b")


def is_antivirus_error(stderr: str) -> bool:





    stderr_lower = stderr.lower()
    if _BLOCKED_ERROR_CODE_RE.search(stderr_lower):
        return True
    patterns = [
        *classifier_markers("access_denied", _ACCESS_DENIED_LOCALIZED),


        *classifier_markers("app_control", _APP_CONTROL_PATTERNS),



        "cannot open the device or file",



        "blocked by antivirus",
        "permission denied",
        "operation did not complete successfully because the file contains a virus",
        "blocked by your administrator",
    ]
    return any(p in stderr_lower for p in patterns)




_BINARY_MODULE_EXTENSIONS = (".pyd", ".dll", ".so", ".dylib")






_SHARING_VIOLATION_RE = re.compile(r"winerror\s+32\b|os error\s+32\b")


def is_file_locked_error(output: str) -> bool:












    lower = output.lower()

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




    return (bool(_BLOCKED_ERROR_CODE_RE.search(lower))
            or bool(_SHARING_VIOLATION_RE.search(lower))
            or "permission denied" in lower)


def get_file_locked_help() -> str:

    return (
        tr("How to fix this:") + "\n\n"
        + tr("  1. Close all QGIS windows (File > Exit)") + "\n"
        + tr("  2. Reopen QGIS") + "\n"
        + tr("  3. Open the AI Segmentation panel - installation will resume") + "\n\n"
        + tr("If it still fails after restarting QGIS:") + "\n"
        + tr(
            "  4. Uninstall the plugin "
            "(Plugins > Manage and Install Plugins > Installed > AI Segmentation)"
        ) + "\n"
        + tr("  5. Restart QGIS") + "\n"
        + tr("  6. Reinstall the plugin")
    )


def get_pip_antivirus_help(exclude_dir: str) -> str:







    steps = (
        tr(
            "Installation was blocked, likely by antivirus software "
            "or security policy."
        ) + "\n\n"
        + tr("Please try:") + "\n"
        + tr("  1. Temporarily disable real-time antivirus scanning") + "\n"
        + tr("  2. Add an exclusion for the plugin folder:") + "\n"
        + f"     {exclude_dir}\n"
    )
    if sys.platform == "win32":
        steps += (
            tr(
                "  3. Run QGIS as administrator "
                "(right-click > Run as administrator)"
            ) + "\n"
            + tr("  4. Try the installation again")
        )
    else:
        steps += (
            tr("  3. Check folder permissions: {command}").format(
                command=f'chmod -R u+rwX "{exclude_dir}"'
            ) + "\n"
            + tr("  4. Try the installation again")
        )
    return steps







_WINDOWS_CRASH_CODES = {
    3221225477,
    -1073741819,
    3221225725,
    -1073741571,
    3221225781,
    -1073741515,
}


def is_windows_process_crash(returncode: int) -> bool:

    if sys.platform != "win32":
        return False
    return returncode in _WINDOWS_CRASH_CODES


def is_rename_or_record_error(output: str) -> bool:

    lower = output.lower()
    if "rename" in lower and "dist-info" in lower:
        return True
    if "record" in lower and "dist-info" in lower:
        return True

    return bool("failed to install" in lower and "failed to rename" in lower)


def get_crash_help(venv_dir: str) -> str:

    return (
        tr("The installer process crashed unexpectedly (access violation).") + "\n\n"
        + tr("This is usually caused by:") + "\n"
        + tr("  - Antivirus software (Windows Defender, etc.) blocking pip") + "\n"
        + tr("  - Corrupted virtual environment") + "\n\n"
        + tr("Please try:") + "\n"
        + tr("  1. Temporarily disable real-time antivirus scanning") + "\n"
        + tr("  2. Add an exclusion for the plugin folder:") + "\n"
        + f"     {venv_dir}\n"
        + tr("  3. {step} to build it again").format(step=install_again_step()) + "\n"
        + tr("  4. If the issue persists, run QGIS as administrator")
    )






_INVALID_PATH_PATTERNS = [


    "volume label syntax is incorrect",
]





_INVALID_PATH_CODE_RE = re.compile(
    r"(?:errno|os error|winerror)\s+(?:22|123|206)\b")


def is_invalid_path_error(output: str) -> bool:








    lower = output.lower()
    if _INVALID_PATH_CODE_RE.search(lower):
        return True
    return any(
        p in lower for p in classifier_markers("invalid_path", _INVALID_PATH_PATTERNS))


def get_invalid_path_help(cache_dir: str = "") -> str:

    location = cache_dir or "~/.qgis_ai_segmentation"
    return (
        tr("Windows refused a file path during installation.") + "\n\n"
        + tr(
            "This usually means the install folder is cloud-synced "
            "(OneDrive/Dropbox), contains unusual characters, or the path grew "
            "past the Windows length limit."
        ) + "\n\n"
        + tr("The environment installs under: {location}").format(location=location) + "\n\n"
        + tr("Please try:") + "\n"
        + tr(
            "  1. If that folder is inside OneDrive or another sync tool, pause\n"
            "     syncing (or mark the folder 'Always keep on this device')"
        ) + "\n"
        + tr(
            "  2. Or set the AI_SEGMENTATION_CACHE_DIR environment variable to a\n"
            "     short local folder outside any synced area (e.g. C:\\qgis_ai),\n"
            "     then restart QGIS"
        ) + "\n"
        + tr("  3. {step} again").format(step=install_again_step())
    )






_BROKEN_RUNTIME_PATTERNS = [

    "no module named 'encodings'",
    "no module named encodings",
    "init_fs_encoding",
    "failed to get the python codec of the filesystem encoding",
]


def is_broken_python_runtime(output: str) -> bool:







    lower = output.lower()
    return any(
        p in lower for p in classifier_markers("broken_runtime", _BROKEN_RUNTIME_PATTERNS))


def get_broken_python_runtime_help(cache_dir: str = "") -> str:

    location = cache_dir or "~/.qgis_ai_segmentation"
    return (
        tr(
            "The plugin's local Python runtime is damaged and cannot start.\n"
            "This is usually caused by antivirus quarantine or an interrupted\n"
            "first installation."
        ) + "\n\n"
        + tr("The next installation will rebuild it from scratch automatically.") + "\n\n"
        + tr("Please try:") + "\n"
        + tr("  1. Add an antivirus exclusion for the folder:") + "\n"
        + f"     {location}\n"
        + tr("  2. {step} to build everything again").format(step=install_again_step())
    )






_CORRUPT_VENV_PATTERNS = [

    "no module named pip",
    "no module named 'pip'",

    "failed to inspect python interpreter",
]


def is_corrupt_venv(output: str) -> bool:





    lower = output.lower()
    return any(p in lower for p in classifier_markers("corrupt_venv", _CORRUPT_VENV_PATTERNS))


def get_corrupt_venv_help() -> str:

    return (
        tr(
            "The plugin's Python environment is damaged (files are missing "
            "inside it)."
        ) + "\n\n"
        + tr("{step}. The plugin builds it again from scratch.").format(
            step=install_again_step())
    )






_DEPENDENCY_CONFLICT_PATTERNS = [
    "conflicting dependencies",
    "resolutionimpossible",
    "no solution found when resolving",
]


def is_dependency_conflict(output: str) -> bool:

    lower = output.lower()
    return any(
        p in lower
        for p in classifier_markers("dependency_conflict", _DEPENDENCY_CONFLICT_PATTERNS))


def get_dependency_conflict_help() -> str:

    return (
        tr(
            "The package resolver could not find a compatible set of versions.\n"
            "This usually comes from stale cached package data or a Python\n"
            "version the AI packages no longer support."
        ) + "\n\n"
        + tr("Please try:") + "\n"
        + tr("  1. {step} to build again with fresh data").format(
            step=install_again_step()) + "\n"
        + tr(
            "  2. If it persists, update QGIS to the latest LTR release\n"
            "     (newer QGIS ships a newer Python) and try again"
        )
    )
