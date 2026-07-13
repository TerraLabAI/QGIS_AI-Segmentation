"""
Python Standalone Manager for QGIS AI-Segmentation Plugin.

Downloads and manages a standalone Python interpreter that matches
the QGIS Python version, ensuring 100% compatibility.

Source: https://github.com/astral-sh/python-build-standalone
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess  # nosec B404
import sys
import tarfile
import tempfile
import time
import zipfile
from typing import Callable

from qgis.core import Qgis, QgsBlockingNetworkRequest
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from .archive_utils import safe_extract_tar as _safe_extract_tar
from .archive_utils import safe_extract_zip as _safe_extract_zip
from .logging_utils import log as _log
from .model_config import IS_ROSETTA
from .subprocess_utils import get_subprocess_kwargs  # nosec B404 - our helper, name merely starts with "subprocess"
from .uv_manager import DOWNLOAD_TIMEOUT_MS

PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("AI_SEGMENTATION_CACHE_DIR") or os.path.expanduser("~/.qgis_ai_segmentation")
STANDALONE_DIR = os.path.join(CACHE_DIR, "python_standalone")


def _download_tmp_dir() -> str | None:
    """Containment temp dir on the CACHE_DIR volume, mirroring venv_manager's
    _apply_cache_containment. The download temp file must land on the same
    volume the rest of the install uses, or a full SYSTEM drive still ENOSPCs
    the multi-hundred-MB download even when AI_SEGMENTATION_CACHE_DIR points at a
    roomy secondary drive. Returns None on any failure so the caller falls back
    to the system default temp (``dir=None`` == the old behaviour)."""
    tmp_dir = os.path.join(CACHE_DIR, "tmp")
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir
    except OSError:
        return None


# Release tag from python-build-standalone
# Update this periodically to get newer Python builds
RELEASE_TAG = "20251014"

# Mapping of Python minor versions to their latest patch versions in the release
PYTHON_VERSIONS = {
    (3, 9): "3.9.24",
    (3, 10): "3.10.19",
    (3, 11): "3.11.14",
    (3, 12): "3.12.12",
    (3, 13): "3.13.9",
    (3, 14): "3.14.0",
}

# SHA256 of every standalone-Python archive get_download_urls() can request,
# copied from the release's official SHA256SUMS. Public integrity checks, not
# secrets. Covers all PYTHON_VERSIONS x every platform string _get_platform_info()
# emits x both variants (install_only_stripped preferred, install_only fallback).
# MUST be regenerated in the same commit as any RELEASE_TAG or PYTHON_VERSIONS
# bump: the download fails closed on a mismatch or a missing entry, so a stale
# or partial table would brick installs on the affected platform.
PYTHON_STANDALONE_SHA256 = {
    "cpython-3.9.24+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "6292c6484ab2c96c80116f4bdb3da638d816206fe11a102e83787a2f75591b94",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-aarch64-apple-darwin-install_only.tar.gz": "6b65213e639e91eb8072db80ed9c140d769af1d5e0386efd8f153449c3694714",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "eafceb263d9507ff0052ae9d6f1c415bb99299dcb202a931865b8ca044a5e40e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-apple-darwin-install_only.tar.gz": "14beda9465feb6991f73d6f6cb9e69afc576c5cac8c185bd729f491aa4305bfb",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "fc9b3af198bdc85ff532eade79825d18b4a4d4036caf8f895922e97e3378c642",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "a2fdaf290361386396bbfaa08e13fc2b88e1149f870adf18836e262c609406db",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "4a5faa90b3f76b235f2706be501605fc8f57e4f1e2c6c596e6fb328639e0d65a",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "d840efd9d81ad557019ebd0d435828fc32101cd01be82046087b4aee463dca0c",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "0339804b69bc00d5dde58c6694174c8e97e6f16c8ace90fe9a1b1a15456ac510",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "866745efbee219a3f9b9d54ee1477ebf92542bb9ff9f6591a7e5a3643a0d4214",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "37931bdcd24496bf57415e34f93dcf360f80b6a2b5bf91d32ceecde14fe9f29f",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-aarch64-apple-darwin-install_only.tar.gz": "06cfdfa8966dfd86204d45c6a241dd37cb0b3ede90986591fc0b0dbe576848de",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "82703c1d9de3b6b686269361dd61c29aa65f52d04dbf0c4f53fc6fd8faf38dfd",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-apple-darwin-install_only.tar.gz": "b4e0c82f350f18a8fb1b1982f03c1c90aaba5d9ab74fe6ede9896306f64a287c",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "47ffefa9240d7354c086e9eb84e917d2460c6ae2a719281337218a2a3c83e4cf",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "e2d9193b2d2fd99fac3fb90eda216100b64cd7cf14f291d9425436ea9b1eaa04",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "6ea9ff46ae3e0eb551558754c78a41cb90968b1950e1a2c716e339e6264bcc96",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "c4c760f49dbba10a0f91b2fd52c847dd50cbe7cb8cb19bb7598c4dc38a358e9c",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "c0b09b744293f2aad85b1ef84544f8a7ba383675b29d1f7efd1e96bb9984399f",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "85c96114de83d783db18137f3858bcd3b5a9c4cbe9053f0072d7b5f52154a8c9",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "cd54cc868a9b1056fbd4654509431f402f0365329618e2583b60c82f73da4e56",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-aarch64-apple-darwin-install_only.tar.gz": "99d98bf73d9906d18a9184054a328288ede2cb4a2d245a05411a28e8d023aab6",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "a1eb602d2bbbfdbba005c54334b33779de8e0f2225f1d5e03c7a1e3e95cb822e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-apple-darwin-install_only.tar.gz": "d234fa6518634daf3aa812895ec757d0e0b1fea3335fd0c5038d4e2bcc5d7ee5",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "c2a7b0bb86eb9f1cc094a01bdabef7ddc77f89f8e45161fa7819f2b4a7ba7bc1",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "80022423ca581c88d5bb7beb889f10c12d3d8d2e5cc6422fd2b060b52e45aa05",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "b59afc432a64df8fdcfeab5bca98e66c7272cd3e6bc3611b9b48996f714ae15a",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "8b033614f3a6969d86c20f9b823277ee8e1f72788307c082a44d2ad4cc856e2b",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "91ab434738647ab45d630aaa02e3808bb516239beeb52f7799c12a12d1557a38",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "d0623c777fb89b904b56cd5aba51af29cbb34b1f9d45f0672f90f6dce30fa93e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "84cb7acbf75264982c8bdd818bfa1ff0f1eb76007b48a5f3e01d28633b46afdf",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-aarch64-apple-darwin-install_only.tar.gz": "6ceba34fe78802853a30bde6f303a0a54f71f6ab07a673da34e90c0aa06c786e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "f76a921e71e9c8954cccd00f176b7083041527b3b4223670d05bbb2f51209d3f",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-apple-darwin-install_only.tar.gz": "9b8589eefb153cbe7cb652993d0ecc94aeb2fa13c1a2e8bc240f5f74f23bb21b",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "3c8b9b10a933909c98b9916297e2093b24a9c2abaa23df1c2622c2bfe052cb94",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "2d670beb3b930d30e3a13cc909923a001dbdfcb5537692d5da40b6b41643ce1c",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "d2a6c0d4ceea088f635b309a59d5d700a256656423225f96ddfb71d532adb1aa",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "d32487b853d6f5709019a471770be5e5d3e6bd2ac507e5629e2d6825565d3e71",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "c74addcd1b033a6e4d60ead3ab47fcc995569027e01d3061c4a934f363c4a0cf",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "1ab2b6594d1c3d76cbebea09d6bc3e6ba68d8eb3b6322080375c4cc3dd188f34",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "52721745b0fa3196e4d0381fa5c06dda1d54343b90d49d90c3bba52d1171bd98",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-aarch64-apple-darwin-install_only.tar.gz": "931db8f735e18700d4eab9ee39dbbd0b4c114d7d039dd2707b2d932ded039698",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "7c33b153a69c6255e6f2659cf39738f316b03969d6230d7bc47c73b7fde9a0d4",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-apple-darwin-install_only.tar.gz": "9f6bc3c15e2f9e2c9c90db2c8b3ee94598e777789f8aea6e36b69ae55d007d01",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "76e0a9749c4deeb975a4b6b36d54be4e43f0c2a4c654bedab5d2e4d62dbc3006",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "8b0efc2674bb293ce2d423d59765b1ca3a2d80dc0ca6168f6279cb569e72b55e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "baa3d107d17e4328448e30c3c9c83cca0eb41ca7a37c10982e14d46a5c3db07d",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "c86606a45fb6540b1b66d9c52c6f5466fba8affb29acb9ab6a0b7f5ad54e588a",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "c0ccda275948c79996e46993c2c5476ff5cca606dee530f1dea712179131b348",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "b4b0204658930337c85c321b49ed2585fe544097a72bc76dcf0b77e49fff8473",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "057476264b07222a2baeff68a733647f91a9d61c94f79beba46a44eb42101749",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-aarch64-apple-darwin-install_only.tar.gz": "1333ce2807fbea673eb242edbf4997ea1e2f6cbc01cd80dec1f9d19de2cd63ed",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "56dcb0cdafabac9d6d976690fb05d9ee92d20ce798c3aabe9049259ebe7d3e0d",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-apple-darwin-install_only.tar.gz": "0a4cc33ca56830b92545950aacdde8925c9d4259e4f00ceda04fedf853f70679",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "b064fca740da03dbae1bad7f73fcaabbc76681ad635b9897ed3808c3eecff122",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "d90e97fe69b819f0a776cd665d06fef6526a4259211d11f00e501688659f1c0e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "7dbb43b742c040835a277318355fb359b41e509dbf4fbb614da38005a9290e16",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "e613f44e60227b3423a994698426698569e055c24447c10dd9c1c022cf511f05",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "493c477b4a88bb1ea2f6c6f57fa0e88ffbe55d9e7b1405c4699f2d41c04eb154",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "74d4516a64abc63ae4bcbffb35482879a85b7faa187fcfa47c1ca8f00faebf5f",  # noqa: E501  # pragma: allowlist secret
}


def is_nixos() -> bool:
    """Detect NixOS where standalone Python binaries cannot run."""
    if sys.platform != "linux":
        return False
    nix_env = os.environ.get("NIX_PROFILES")
    return os.path.exists("/etc/NIXOS") or bool(nix_env)


def is_flatpak() -> bool:
    """Detect a Flatpak-sandboxed QGIS (e.g. org.qgis.qgis on Flathub)."""
    if sys.platform != "linux":
        return False
    return os.path.exists("/.flatpak-info") or bool(os.environ.get("FLATPAK_ID"))


def is_snap() -> bool:
    """Detect a Snap-confined QGIS (the Ubuntu Snap Store package)."""
    if sys.platform != "linux":
        return False
    return bool(os.environ.get("SNAP")) and bool(os.environ.get("SNAP_NAME"))


def is_sandboxed_linux() -> bool:
    """True when running inside a Flatpak or Snap sandbox on Linux.

    Manual (local) mode needs to download/execute a standalone Python and
    build a multi-GB venv; both are unreliable or blocked under strict
    Flatpak/Snap confinement. Automatic mode needs no local install and is
    unaffected.
    """
    return is_flatpak() or is_snap()


def is_unsupported_windows() -> tuple[bool, str]:
    """Detect Windows versions below the standalone Python's official support.

    astral-sh/python-build-standalone targets Windows 8+. Windows 7 binaries
    boot but commonly miss runtime APIs (e.g. ssl module loading fails because
    schannel-related symbols are absent in older kernel32), producing the
    "Can't connect to HTTPS URL because the SSL module is not available"
    failure observed in user reports. Detect early so we surface a clear
    "OS not supported" message instead of letting the install loop on a
    download that will never produce a working interpreter.
    """
    if sys.platform != "win32":
        return False, ""
    release = platform.release() or ""
    if release in ("7", "Vista", "XP", "2003Server", "post2003"):
        return True, (
            f"Windows {release} is not supported by AI Segmentation. "
            "The bundled Python interpreter requires Windows 8 or later. "
            "Please upgrade to Windows 10 or 11."
        )
    return False, ""


def is_unsupported_python_version() -> tuple[bool, str]:
    """Detect a QGIS Python major.minor with no matching standalone build.

    We only ship interpreters for the versions in PYTHON_VERSIONS. Installing a
    mismatched standalone would pull ABI-incompatible wheels that are then
    imported in-process (numpy/rasterio/scipy on the polygon path), which can
    crash all of QGIS. Surface a clean "not supported" message here instead,
    mirroring is_unsupported_windows so the install path fails fast.
    """
    major, minor = get_qgis_python_version()
    if (major, minor) in PYTHON_VERSIONS:
        return False, ""
    return True, (
        f"Python {major}.{minor} is not supported by AI Segmentation. "
        "Please use a QGIS build with a supported Python version."
    )


def _get_windows_antivirus_help(plugin_path: str) -> str:
    """
    Return help message for Windows antivirus issues.
    """
    return (
        "Installation failed - this may be caused by antivirus software blocking the extraction.\n"
        "Please try:\n"
        "  1. Temporarily disable your antivirus (Windows Defender, etc.)\n"
        "  2. Add an exclusion for the QGIS plugins folder\n"
        "  3. Try the installation again\n"
        f"Folder to exclude: {plugin_path}"
    )


def get_qgis_python_version() -> tuple[int, int]:
    """Get the target Python version for the standalone interpreter.

    Under Rosetta, returns (3, 10) so we download ARM64 Python 3.10+
    for SAM2 support instead of matching QGIS's x86_64 Python 3.9.
    """
    if IS_ROSETTA:
        return (3, 10)
    return (sys.version_info.major, sys.version_info.minor)


def get_python_full_version() -> str:
    """Get the full Python version string for download (e.g., '3.12.12')."""
    version_tuple = get_qgis_python_version()
    if version_tuple in PYTHON_VERSIONS:
        return PYTHON_VERSIONS[version_tuple]
    # Fallback: use 3.13 (newest well-tested version) instead of X.Y.0
    # which likely doesn't exist in the release assets
    _log(
        f"Python {version_tuple[0]}.{version_tuple[1]} not in PYTHON_VERSIONS, falling back to 3.13",
        Qgis.MessageLevel.Warning)
    return PYTHON_VERSIONS[(3, 13)]


def _create_python_symlinks(python_dir: str) -> None:
    """Create python3 symlink if only versioned binary exists (e.g. python3.12)."""
    bin_dir = os.path.join(python_dir, "bin")
    python3_path = os.path.join(bin_dir, "python3")
    if os.path.exists(python3_path):
        return
    # Find versioned binary like python3.12
    major, minor = get_qgis_python_version()
    versioned = os.path.join(bin_dir, f"python{major}.{minor}")
    if os.path.exists(versioned):
        os.symlink(f"python{major}.{minor}", python3_path)
        _log(f"Created python3 symlink -> python{major}.{minor}")


def get_standalone_python_path() -> str:
    """Get the path to the standalone Python executable."""
    python_dir = os.path.join(STANDALONE_DIR, "python")

    if sys.platform == "win32":
        return os.path.join(python_dir, "python.exe")
    return os.path.join(python_dir, "bin", "python3")


def standalone_python_exists() -> bool:
    """Check if standalone Python is already installed."""
    python_path = get_standalone_python_path()
    return os.path.exists(python_path)


def standalone_python_is_current() -> bool:
    """Check if installed standalone Python matches QGIS Python major.minor.

    Returns False if standalone doesn't exist or version doesn't match.
    """
    python_path = get_standalone_python_path()
    if not os.path.exists(python_path):
        return False

    try:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(  # nosec B603
            [python_path, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=15, env=env, **get_subprocess_kwargs(),
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) == 2:
                installed = (int(parts[0]), int(parts[1]))
                expected = get_qgis_python_version()
                if installed != expected:
                    _log(
                        f"Standalone Python {installed[0]}.{installed[1]} "
                        f"doesn't match QGIS {expected[0]}.{expected[1]}",
                        Qgis.MessageLevel.Warning)
                    return False
                return True
    except Exception as e:
        _log(f"Failed to check standalone Python version: {e}", Qgis.MessageLevel.Warning)

    return False


def _get_platform_info() -> tuple[str, str]:
    """Get platform and architecture info for download URL."""
    system = sys.platform
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64") or IS_ROSETTA:
            return ("aarch64-apple-darwin", ".tar.gz")
        return ("x86_64-apple-darwin", ".tar.gz")
    if system == "win32":
        return ("x86_64-pc-windows-msvc", ".tar.gz")
    # Linux
    if machine in ("arm64", "aarch64"):
        return ("aarch64-unknown-linux-gnu", ".tar.gz")
    return ("x86_64-unknown-linux-gnu", ".tar.gz")


def get_download_urls() -> list[str]:
    """Candidate download URLs for the standalone Python, preferred first.

    install_only_stripped is the same build minus native debug symbols
    (about half the download on Windows, a third of it on Linux); it is
    what uv itself ships. Plain install_only stays as a fallback in case
    a release or platform lacks the stripped variant.
    """
    python_version = get_python_full_version()
    platform_str, ext = _get_platform_info()
    base = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{RELEASE_TAG}"
    )
    prefix = f"cpython-{python_version}+{RELEASE_TAG}-{platform_str}"
    return [
        f"{base}/{prefix}-install_only_stripped{ext}",
        f"{base}/{prefix}-install_only{ext}",
    ]


def _sha256_file(filepath: str) -> str:
    """Stream a file through SHA256 in 4096-byte blocks.

    Mirrors checkpoint_manager.verify_checkpoint_hash so the archive (tens of
    MB) is not held in memory a second time.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _verify_python_payload(filepath: str, asset_name: str) -> tuple[bool, str]:
    """Fail-closed SHA256 integrity check of a downloaded standalone Python archive.

    Returns (ok, message). Both a missing pin and a digest mismatch fail, so a
    tampered CDN response or an unpinned variant is never extracted or executed.
    """
    expected = PYTHON_STANDALONE_SHA256.get(asset_name, "")
    if not expected:
        return False, f"No pinned digest for {asset_name}; refusing to install"
    if _sha256_file(filepath) != expected:
        return False, "Python download failed integrity verification"
    return True, ""


def download_python_standalone(
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None
) -> tuple[bool, str]:
    """
    Download and install Python standalone using QGIS network manager.

    Uses QgsBlockingNetworkRequest to respect QGIS proxy settings.

    Args:
        progress_callback: Function called with (percent, message) for progress updates
        cancel_check: Function that returns True if operation should be cancelled

    Returns:
        Tuple of (success: bool, message: str)
    """
    unsupported, why = is_unsupported_windows()
    if unsupported:
        _log(why, Qgis.MessageLevel.Critical)
        return False, why

    unsupported, why = is_unsupported_python_version()
    if unsupported:
        _log(why, Qgis.MessageLevel.Critical)
        return False, why

    if standalone_python_exists():
        _log("Python standalone already exists", Qgis.MessageLevel.Info)
        return True, "Python standalone already installed"

    urls = get_download_urls()
    python_version = get_python_full_version()

    _log(f"Downloading Python {python_version} from: {urls[0]}", Qgis.MessageLevel.Info)

    if progress_callback:
        progress_callback(0, f"Downloading Python {python_version}...")

    # Create temp file for download, contained on the CACHE_DIR volume so a
    # full system drive does not ENOSPC the download (see _download_tmp_dir).
    fd, temp_path = tempfile.mkstemp(suffix=".tar.gz", dir=_download_tmp_dir())
    os.close(fd)

    try:
        if cancel_check and cancel_check():
            return False, "Download cancelled"

        if progress_callback:
            progress_callback(5, "Connecting to download server...")

        # Try each URL variant (stripped first, plain as fallback), each
        # with up to 3 attempts and exponential backoff. QGIS network
        # manager is used so QGIS proxy settings are respected. A variant is
        # abandoned for the next one BOTH when it 404s and when it downloads
        # but its interpreter fails the post-extract self-check: a stripped
        # archive can be published yet unable to create a venv, and only the
        # plain build recovers that machine.
        max_retries = 3
        last_error = ""
        for url_idx, url in enumerate(urls):
            qurl = QUrl(url)
            err = None
            error_msg = ""
            request = None
            for attempt in range(max_retries):
                if cancel_check and cancel_check():
                    return False, "Download cancelled"

                request = QgsBlockingNetworkRequest()
                net_req = QNetworkRequest(qurl)
                if hasattr(net_req, "setTransferTimeout"):
                    net_req.setTransferTimeout(DOWNLOAD_TIMEOUT_MS)
                err = request.get(net_req)

                if err == QgsBlockingNetworkRequest.ErrorCode.NoError:
                    break

                error_msg = request.errorMessage()
                if "404" in error_msg or "Not Found" in error_msg:
                    # No point retrying a 404; move on to the next variant
                    break

                if attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)  # 5, 10s
                    _log(
                        f"Download failed (attempt {attempt + 1}/{max_retries}): {error_msg}. "
                        f"Retrying in {wait}s...",
                        Qgis.MessageLevel.Warning
                    )
                    if progress_callback:
                        progress_callback(
                            5, f"Network error, retrying in {wait}s...")
                    time.sleep(wait)

            if err != QgsBlockingNetworkRequest.ErrorCode.NoError:
                is_404 = "404" in error_msg or "Not Found" in error_msg
                if is_404:
                    if url_idx + 1 < len(urls):
                        _log(
                            f"Archive variant not published ({url}), "
                            "trying the fallback variant...",
                            Qgis.MessageLevel.Warning)
                        continue
                    error_msg = (
                        f"Python {python_version} not available for this platform. "
                        f"URL: {url}"
                    )
                    _log(error_msg, Qgis.MessageLevel.Critical)
                    return False, error_msg
                error_msg = f"Download failed: {error_msg}"
                _log(error_msg, Qgis.MessageLevel.Critical)
                return False, error_msg

            if cancel_check and cancel_check():
                return False, "Download cancelled"

            reply = request.reply()
            content = reply.content()

            content_size = len(content)
            if content_size == 0:
                # A bad payload from one variant should not abort the run: record
                # it and try the next variant, returning False only after the last.
                last_error = "Download failed: received empty file (0 bytes)"
                _log(last_error, Qgis.MessageLevel.Warning)
                continue
            min_expected = 10 * 1024 * 1024  # 10 MB
            if content_size < min_expected:
                _log(
                    f"Download suspiciously small: {content_size} bytes (expected >10 MB)", Qgis.MessageLevel.Warning)
                last_error = (
                    f"Download failed: file too small ({content_size / (1024 * 1024):.1f} MB). "
                    "A firewall or proxy may be blocking the download."
                )
                continue

            if progress_callback:
                total_mb = content_size / (1024 * 1024)
                progress_callback(50, f"Downloaded {total_mb:.1f} MB, saving...")

            # Write content to temp file
            with open(temp_path, "wb") as f:
                f.write(content.data())

            # Validate archive magic bytes (catch proxy/firewall HTML pages)
            with open(temp_path, "rb") as f:
                magic = f.read(4)
            is_gzip = magic[:2] == b"\x1f\x8b"
            is_zip = magic[:2] == b"PK"
            if not is_gzip and not is_zip:
                try:
                    preview_text = bytes(content.data()[:200]).decode(
                        "utf-8", errors="replace")[:150]
                except Exception:
                    preview_text = "(binary data)"
                last_error = (
                    "Download failed: file is not a valid archive. "
                    "A firewall or proxy may have returned an error page. "
                    f"Preview: {preview_text}"
                )
                _log(last_error, Qgis.MessageLevel.Warning)
                continue

            # Cryptographically verify the payload BEFORE extracting/executing it.
            asset_name = url.rsplit("/", 1)[-1]
            ok, verify_msg = _verify_python_payload(temp_path, asset_name)
            if not ok:
                _log(verify_msg, Qgis.MessageLevel.Warning)
                last_error = verify_msg
                continue

            _log(f"Download complete ({content_size} bytes), extracting...", Qgis.MessageLevel.Info)

            if progress_callback:
                progress_callback(55, "Extracting Python...")

            # Remove existing standalone dir if it exists
            if os.path.exists(STANDALONE_DIR):
                shutil.rmtree(STANDALONE_DIR)

            os.makedirs(STANDALONE_DIR, exist_ok=True)

            # Extract archive with path traversal protection
            if temp_path.endswith(".tar.gz") or temp_path.endswith(".tgz"):
                with tarfile.open(temp_path, "r:gz") as tar:
                    _safe_extract_tar(tar, STANDALONE_DIR)
            else:
                with zipfile.ZipFile(temp_path, "r") as z:
                    _safe_extract_zip(z, STANDALONE_DIR)

            # Create python3 symlink if missing (archive symlinks skipped for safety)
            if sys.platform != "win32":
                _create_python_symlinks(os.path.join(STANDALONE_DIR, "python"))

            if progress_callback:
                progress_callback(80, "Verifying Python installation...")

            # Verify installation
            success, verify_msg = verify_standalone_python()

            if success:
                if progress_callback:
                    progress_callback(100, f"✓ Python {python_version} installed")
                _log("Python standalone installed successfully", Qgis.MessageLevel.Success)
                return True, f"Python {python_version} installed successfully"
            # Clean up broken installation so _get_system_python() won't find it
            remove_standalone_python()
            last_error = f"Verification failed: {verify_msg}"
            if url_idx + 1 < len(urls):
                _log(
                    f"Extracted interpreter failed its self-check ({verify_msg}); "
                    "trying the fallback archive variant...",
                    Qgis.MessageLevel.Warning)
                continue
            return False, last_error

        return False, last_error or "Download failed"

    except InterruptedError:
        return False, "Download cancelled"
    except Exception as e:
        error_msg = f"Installation failed: {str(e)}"
        _log(error_msg, Qgis.MessageLevel.Critical)

        # On Windows, check for antivirus blocking (permission/access errors)
        if sys.platform == "win32":
            error_lower = str(e).lower()
            if "denied" in error_lower or "access" in error_lower or "permission" in error_lower:
                antivirus_help = _get_windows_antivirus_help(STANDALONE_DIR)
                _log(antivirus_help, Qgis.MessageLevel.Warning)
                error_msg = f"{error_msg}\n\n{antivirus_help}"

        return False, error_msg
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def verify_standalone_python() -> tuple[bool, str]:
    """Verify that the standalone Python installation works."""
    python_path = get_standalone_python_path()

    if not os.path.exists(python_path):
        return False, f"Python executable not found at {python_path}"

    # On Unix, make sure it's executable
    if sys.platform != "win32":
        try:
            import stat
            # Set executable permission (owner rwx, group rx, others rx)
            os.chmod(python_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        except OSError:
            pass

    try:
        # Test basic execution
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONIOENCODING"] = "utf-8"

        # Probe BOTH the version AND that `import subprocess` works: a broken
        # macOS standalone can print its version yet fail to import subprocess
        # (which pulls the _posixsubprocess C extension), a failure that used to
        # stay hidden until it resurfaced as a cryptic "No module named
        # '_posixsubprocess'" crash at venv-creation time. Importing it here
        # catches the broken build now, so it is removed and re-downloaded
        # instead of being trusted (#bug-anehm).
        result = subprocess.run(  # nosec B603
            [python_path, "-c", "import subprocess, sys; print(sys.version)"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
            **get_subprocess_kwargs(),
        )

        if result.returncode == 0:
            version_output = result.stdout.strip().split()[0]
            expected_version = get_python_full_version()

            # Require the FULL version to match what we downloaded. A major.minor
            # check let a wrong interpreter (e.g. the host's 3.9.5 instead of the
            # bundled 3.9.24) pass verification, masking a broken extraction that
            # only failed later at venv creation (#bug-anehm).
            if version_output != expected_version:
                msg = f"Python version mismatch: got {version_output}, expected {expected_version}"
                _log(msg, Qgis.MessageLevel.Warning)
                return False, f"Version mismatch: downloaded {version_output}, expected {expected_version}"

            _log(f"Verified Python standalone: {version_output}", Qgis.MessageLevel.Success)
            return True, f"Python {version_output} verified"
        error = result.stderr or "Unknown error"
        _log(f"Python verification failed: {error}", Qgis.MessageLevel.Warning)
        return False, f"Verification failed: {error[:100]}"

    except subprocess.TimeoutExpired:
        return False, "Python verification timed out"
    except Exception as e:
        return False, f"Verification error: {str(e)[:100]}"


def remove_standalone_python() -> tuple[bool, str]:
    """Remove the standalone Python installation."""
    if not os.path.exists(STANDALONE_DIR):
        return True, "Standalone Python not installed"

    try:
        shutil.rmtree(STANDALONE_DIR)
        _log("Removed standalone Python installation", Qgis.MessageLevel.Success)
        return True, "Standalone Python removed"
    except Exception as e:
        error_msg = f"Failed to remove: {str(e)}"
        _log(error_msg, Qgis.MessageLevel.Warning)
        return False, error_msg
