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
import stat
import subprocess  # nosec B404
import sys
import tarfile
import tempfile
import time
import zipfile
from typing import Callable

from qgis.core import Qgis

from .archive_utils import safe_extract_tar as _safe_extract_tar
from .archive_utils import safe_extract_zip as _safe_extract_zip
from .cache_paths import PLUGIN_CACHE_DIR, plugin_cache_tmp_dir
from .logging_utils import log as _log
from .model_config import IS_ROSETTA
from .python_release_pins import (
    PYTHON_STANDALONE_SHA256,
    PYTHON_VERSIONS,
    RELEASE_TAG,
)
from .streamed_download import discard_part_file, stream_url_to_file
from .subprocess_utils import (  # nosec B404 - our helper, name merely starts with "subprocess"
    get_clean_env_for_venv,
    get_subprocess_kwargs,
)
from .uv_manager import (
    DOWNLOAD_TIMEOUT_MS,
    unsupported_download_platform_reason,
)

# RELEASE_TAG, PYTHON_VERSIONS and PYTHON_STANDALONE_SHA256 above are the
# pinned release, its per-minor patch versions and the digest of every archive
# a download can ask for. They live in one small file because a release bump
# moves all three at once, and the download fails closed on a missing digest.

# Ceiling on one attempt end to end. DOWNLOAD_TIMEOUT_MS is the silence a
# transfer may go through; this is the total, so a slow but live link is left
# alone and a transfer that will never end still stops.
_DOWNLOAD_HARD_TIMEOUT_MS = 60 * 60 * 1000


def tr(text: str) -> str:
    """Translate one user-facing line, falling back to English.

    The translator is imported inside the call, so this module keeps importing
    with no translator loaded and a lookup can never break an install.
    """
    try:
        from .i18n import tr as translate

        return translate(text)
    except Exception:  # noqa: BLE001 -- English is a fine answer here
        return text


STANDALONE_DIR = os.path.join(PLUGIN_CACHE_DIR, "python_standalone")

#: The two 4xx answers that mean "ask again later" rather than "not here":
#: a request timeout and a rate limit. Every other 4xx is about the address.
_RETRIABLE_STATUSES = (408, 429)


def _asset_unavailable(status, error_msg: str) -> bool:
    """Whether this address will never serve the archive.

    403, 404 and 410 all say the asset is not published for this build, and so
    does any other 4xx that is not a timeout or a rate limit. They read the
    same to a caller: retrying the same address cannot change the answer, and
    the fallback archive is the only thing that recovers the machine. Treating
    403 and 410 as plain failures ended the install with a variant left
    untried.
    """
    try:
        code = int(status)
    except (TypeError, ValueError):
        code = 0
    if 400 <= code < 500 and code not in _RETRIABLE_STATUSES:
        return True
    return "404" in error_msg or "Not Found" in error_msg


def resolved_release_tag() -> str:
    """RELEASE_TAG, or the interpreter release the server asked for.

    The release and its digest table move together: see
    ``resolved_python_digests``.
    """
    try:
        from .install_config import python_release_tag

        return python_release_tag(RELEASE_TAG)
    except Exception:  # noqa: BLE001 -- a bad config must never block an install
        return RELEASE_TAG


def resolved_python_versions() -> dict:
    """PYTHON_VERSIONS, with any server-corrected patch version.

    Only a patch version for a minor version this build already fetches is
    accepted: a new key would name an interpreter the rest of the plugin has no
    wheels for.

    And only one the digest table can vouch for. The version and the digests
    are two separate served keys, so a deploy can move one and leave the other
    behind. The download verifies fail-closed, so the machine would then fetch
    the whole archive, find no digest for it, refuse it, and report an install
    failure with nothing in it to act on. An override no digest describes is
    dropped here instead, and the shipped version, whose digests always ship
    with it, stands.
    """
    try:
        from .install_config import python_versions

        served = python_versions(PYTHON_VERSIONS)
    except Exception:  # noqa: BLE001 -- a bad config must never block an install
        return dict(PYTHON_VERSIONS)
    return _versions_a_digest_describes(served)


def _versions_a_digest_describes(served: dict) -> dict:
    """``served``, minus every override the digest table says nothing about."""
    out = dict(PYTHON_VERSIONS)
    try:
        digests = resolved_python_digests()
        release_tag = resolved_release_tag()
    except Exception:  # noqa: BLE001 -- unreadable digests keep what shipped
        return out
    for minor, version in (served or {}).items():
        if minor not in out or out[minor] == version:
            continue
        names = _asset_names_for(str(version), release_tag)
        if not names:
            # No published build fits this machine, so no download will ask
            # for a digest and there is nothing here to guard.
            out[minor] = version
            continue
        if any(name in digests for name in names):
            out[minor] = version
            continue
        _log(
            f"Ignoring the served Python {version}: no pinned digest for it",
            Qgis.MessageLevel.Warning)
    return out


def _asset_names_for(python_version: str, release_tag: str) -> tuple[str, ...]:
    """The archive names a download would ask for, preferred first.

    The same names the digest table is keyed on, so a version can be held
    against the digests before a single byte is fetched. Empty when no
    published build fits this machine.
    """
    platform_str, ext = _get_platform_info()
    if not platform_str:
        return ()
    prefix = f"cpython-{python_version}+{release_tag}-{platform_str}"
    return (
        f"{prefix}-install_only_stripped{ext}",
        f"{prefix}-install_only{ext}",
    )


def resolved_python_digests() -> dict:
    """The digest table for the release in force.

    Additive while the served release equals the shipped one, so a shipped
    digest always wins. When the server moves the release the shipped digests
    describe different bytes and are dropped rather than checked against the
    wrong file. Fail-closed either way: an asset with no digest is refused.
    """
    try:
        from .install_config import python_digests

        return python_digests(PYTHON_STANDALONE_SHA256, RELEASE_TAG)
    except Exception:  # noqa: BLE001 -- a bad config must never block an install
        return dict(PYTHON_STANDALONE_SHA256)


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
    if (major, minor) in resolved_python_versions():
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
    versions = resolved_python_versions()
    if version_tuple in versions:
        return versions[version_tuple]
    # Fallback: use 3.13 (newest well-tested version) instead of X.Y.0
    # which likely doesn't exist in the release assets
    _log(
        f"Python {version_tuple[0]}.{version_tuple[1]} has no pinned build, falling back to 3.13",
        Qgis.MessageLevel.Warning)
    return versions[(3, 13)]


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
        # The shared helper, so the downloaded interpreter is also shielded
        # from an inherited LD_LIBRARY_PATH: a launcher that points it at
        # QGIS's lib dir makes this probe fail on a missing GLIBCXX.
        env = get_clean_env_for_venv()
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
    """Get platform and architecture info for download URL.

    Returns ("", "") when no published build fits this machine, so the caller
    skips a download that could only fail and goes to a system interpreter.
    """
    system = sys.platform
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64") or IS_ROSETTA:
            return ("aarch64-apple-darwin", ".tar.gz")
        return ("x86_64-apple-darwin", ".tar.gz")
    if system == "win32":
        # Windows on ARM takes the x86_64 build under emulation on purpose:
        # the plugin loads native modules from this environment into the QGIS
        # process itself, so the two architectures have to match, and QGIS
        # ships x86_64 on Windows.
        if machine in ("arm64", "aarch64"):
            _log(
                "Windows on ARM: using the x86_64 Python build, which matches "
                "this QGIS process.", Qgis.MessageLevel.Info)
        return ("x86_64-pc-windows-msvc", ".tar.gz")
    # Linux
    if unsupported_download_platform_reason():
        return ("", "")
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
    release_tag = resolved_release_tag()
    names = _asset_names_for(get_python_full_version(), release_tag)
    if not names:
        return []
    base = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{release_tag}"
    )
    return [f"{base}/{name}" for name in names]


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
    expected = resolved_python_digests().get(asset_name, "")
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

    reason = unsupported_download_platform_reason()
    if reason:
        message = f"No standalone Python build for this system: {reason}"
        _log(message, Qgis.MessageLevel.Warning)
        return False, message

    if standalone_python_exists():
        # An existing standalone is only reusable if it still RUNS. Antivirus
        # quarantine or a partial extraction can leave the executable in place
        # with its stdlib gutted; reusing it blind sent every later install
        # attempt into the same "No module named 'encodings'" wall with no way
        # out. A broken standalone is removed here so the download below
        # rebuilds it in the same run.
        ok, why = verify_standalone_python()
        if ok:
            _log("Python standalone already exists", Qgis.MessageLevel.Info)
            return True, "Python standalone already installed"
        _log(
            f"Existing Python standalone is broken ({why}), re-downloading...",
            Qgis.MessageLevel.Warning)
        remove_standalone_python()

    urls = get_download_urls()
    python_version = get_python_full_version()

    _log(f"Downloading Python {python_version} from: {urls[0]}", Qgis.MessageLevel.Info)

    if progress_callback:
        progress_callback(0, f"Downloading Python {python_version}...")

    # Create temp file for download, contained on the cache volume so a full
    # system drive does not ENOSPC the download (see plugin_cache_tmp_dir).
    fd, temp_path = tempfile.mkstemp(suffix=".tar.gz", dir=plugin_cache_tmp_dir())
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

        def on_bytes(received: int, total: int) -> None:
            """One transfer tick, mapped onto the 5 to 50 band of the install.

            Called from inside the transfer, so it stays to arithmetic and a
            label: anything slower here would pace the download itself.
            """
            if not progress_callback:
                return
            mb_done = received / (1024 * 1024)
            if total > 0:
                pct = 5 + int(45 * min(1.0, received / total))
                progress_callback(pct, tr(
                    "Downloading Python: {done} MB of {total} MB").format(
                        done=f"{mb_done:.1f}",
                        total=f"{total / (1024 * 1024):.1f}"))
            else:
                progress_callback(5, tr(
                    "Downloading Python: {done} MB").format(
                        done=f"{mb_done:.1f}"))

        for url_idx, url in enumerate(urls):
            result = None
            error_msg = ""
            for attempt in range(max_retries):
                if cancel_check and cancel_check():
                    discard_part_file(temp_path)
                    return False, "Download cancelled"

                # Streamed onto the cache volume in bounded chunks. The whole
                # archive is tens of megabytes and used to be held in memory
                # twice over, in the reply and again in the write, on exactly
                # the machines with the least room for it.
                result = stream_url_to_file(
                    url,
                    temp_path,
                    _DOWNLOAD_HARD_TIMEOUT_MS,
                    DOWNLOAD_TIMEOUT_MS,
                    progress_callback=on_bytes,
                    cancel_check=cancel_check,
                )
                if result.cancelled:
                    discard_part_file(temp_path)
                    return False, "Download cancelled"
                if result.ok:
                    break

                discard_part_file(temp_path)
                error_msg = result.error or "Download failed"
                if _asset_unavailable(result.http_status, error_msg):
                    # The asset is not there for this build. Retrying the same
                    # address cannot change that; the next variant might.
                    break

                if attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)  # 5, 10s
                    _log(
                        f"Download failed (attempt {attempt + 1}/{max_retries}): {error_msg}. "
                        f"Retrying in {wait}s...",
                        Qgis.MessageLevel.Warning
                    )
                    if progress_callback:
                        progress_callback(5, tr(
                            "Network error, retrying in {seconds}s...").format(
                                seconds=wait))
                    time.sleep(wait)

            if result is None or not result.ok:
                unavailable = _asset_unavailable(
                    result.http_status if result is not None else None,
                    error_msg,
                )
                if unavailable:
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

            content_size = result.bytes_written
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
                progress_callback(50, tr(
                    "Downloaded {mb} MB, saving...").format(
                        mb=f"{total_mb:.1f}"))

            # Validate archive magic bytes (catch proxy/firewall HTML pages).
            # Read off the file rather than off a buffered body: nothing holds
            # the archive in memory any more.
            head = b""
            try:
                with open(temp_path, "rb") as f:
                    head = f.read(200)
            except OSError as read_err:
                last_error = f"Download failed: cannot read the file ({read_err})"
                _log(last_error, Qgis.MessageLevel.Warning)
                continue
            is_gzip = head[:2] == b"\x1f\x8b"
            is_zip = head[:2] == b"PK"
            if not is_gzip and not is_zip:
                try:
                    preview_text = head.decode(
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
                _remove_standalone_tree(STANDALONE_DIR)

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
        # Test basic execution, through the shared helper for the reason given
        # in standalone_python_is_current above.
        env = get_clean_env_for_venv()
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


def _remove_standalone_tree(path: str) -> None:
    """Delete an interpreter tree the way the rest of the installer does.

    A bare rmtree is not enough on Windows. The extracted interpreter nests
    module paths past the legacy 260-char limit, so the delete needs the
    extended-length path form, and an extracted file can carry the read-only
    bit, which makes the delete refuse. Clear the bit and try the entry again
    before giving up on it. Raises whatever survives both attempts, so the
    caller still reports a tree it could not clear.
    """
    from .venv_manager import _win_extended_path

    def _retry(func, target, _exc_info):
        try:
            os.chmod(target, os.stat(target).st_mode | stat.S_IWRITE)
        except OSError:
            raise
        func(target)

    shutil.rmtree(_win_extended_path(path), onerror=_retry)


def remove_standalone_python() -> tuple[bool, str]:
    """Remove the standalone Python installation."""
    if not os.path.exists(STANDALONE_DIR):
        return True, "Standalone Python not installed"

    try:
        _remove_standalone_tree(STANDALONE_DIR)
        _log("Removed standalone Python installation", Qgis.MessageLevel.Success)
        return True, "Standalone Python removed"
    except Exception as e:
        error_msg = f"Failed to remove: {str(e)}"
        _log(error_msg, Qgis.MessageLevel.Warning)
        return False, error_msg
