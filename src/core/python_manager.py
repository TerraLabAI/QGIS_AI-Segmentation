







from __future__ import annotations

import errno
import hashlib
import os
import platform
import re
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
from .cache_paths import PLUGIN_CACHE_DIR, plugin_cache_tmp_dir, remove_tree_quietly
from .logging_utils import log as _log
from .model_config import IS_ROSETTA
from .python_release_pins import (
    PYTHON_STANDALONE_SHA256,
    PYTHON_VERSIONS,
    RELEASE_TAG,
)
from .streamed_download import (
    discard_part_file,
    sleep_unless_cancelled,
    stream_url_to_file,
)
from .subprocess_utils import (  # nosec B404
    get_clean_env_for_venv,
    get_subprocess_kwargs,
    run_unthrottled,
)
from .uv_manager import (
    DOWNLOAD_TIMEOUT_MS,
    unsupported_download_platform_reason,
)









_DOWNLOAD_HARD_TIMEOUT_MS = 60 * 60 * 1000


def tr(text: str) -> str:





    try:
        from .i18n import tr as translate

        return translate(text)
    except Exception:  # noqa: BLE001
        return text


STANDALONE_DIR = os.path.join(PLUGIN_CACHE_DIR, "python_standalone")






_UNAVAILABLE_STATUSES = (403, 404, 410)




_UNAVAILABLE_IN_TEXT = re.compile(r"\b(?:403|404|410)\b|Not Found|Forbidden|Gone")


def _asset_unavailable(status, error_msg: str) -> bool:







    try:
        code = int(status)
    except (TypeError, ValueError):
        code = 0
    if code in _UNAVAILABLE_STATUSES:
        return True
    if 400 <= code < 500:
        return False
    return bool(_UNAVAILABLE_IN_TEXT.search(error_msg or ""))


def resolved_release_tag() -> str:





    try:
        from .install_config import python_release_tag

        return python_release_tag(RELEASE_TAG)
    except Exception:  # noqa: BLE001
        return RELEASE_TAG


def resolved_python_versions() -> dict:














    try:
        from .install_config import python_versions

        served = python_versions(PYTHON_VERSIONS)
    except Exception:  # noqa: BLE001
        return dict(PYTHON_VERSIONS)
    return _versions_a_digest_describes(served)


def _versions_a_digest_describes(served: dict) -> dict:

    out = dict(PYTHON_VERSIONS)
    try:
        digests = resolved_python_digests()
        release_tag = resolved_release_tag()
    except Exception:  # noqa: BLE001
        return out
    for minor, version in (served or {}).items():
        if minor not in out or out[minor] == version:
            continue
        names = _asset_names_for(str(version), release_tag)
        if not names:


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






    platform_str, ext = _get_platform_info()
    if not platform_str:
        return ()
    prefix = f"cpython-{python_version}+{release_tag}-{platform_str}"
    return (
        f"{prefix}-install_only_stripped{ext}",
        f"{prefix}-install_only{ext}",
    )


def resolved_python_digests() -> dict:







    try:
        from .install_config import python_digests

        return python_digests(PYTHON_STANDALONE_SHA256, RELEASE_TAG)
    except Exception:  # noqa: BLE001
        return dict(PYTHON_STANDALONE_SHA256)


def is_nixos() -> bool:

    if sys.platform != "linux":
        return False
    nix_env = os.environ.get("NIX_PROFILES")
    return os.path.exists("/etc/NIXOS") or bool(nix_env)


def is_flatpak() -> bool:

    if sys.platform != "linux":
        return False
    return os.path.exists("/.flatpak-info") or bool(os.environ.get("FLATPAK_ID"))


def is_snap() -> bool:

    if sys.platform != "linux":
        return False
    return bool(os.environ.get("SNAP")) and bool(os.environ.get("SNAP_NAME"))


def is_sandboxed_linux() -> bool:







    return is_flatpak() or is_snap()


def is_unsupported_windows() -> tuple[bool, str]:










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








    major, minor = get_qgis_python_version()
    if (major, minor) in resolved_python_versions():
        return False, ""
    return True, (
        f"Python {major}.{minor} is not supported by AI Segmentation. "
        "Please use a QGIS build with a supported Python version."
    )


def _get_windows_antivirus_help(plugin_path: str) -> str:



    return (
        "Installation failed - this may be caused by antivirus software blocking the extraction.\n"
        "Please try:\n"
        "  1. Temporarily disable your antivirus (Windows Defender, etc.)\n"
        "  2. Add an exclusion for the QGIS plugins folder\n"
        "  3. Try the installation again\n"
        f"Folder to exclude: {plugin_path}"
    )


def get_qgis_python_version() -> tuple[int, int]:





    if IS_ROSETTA:
        return (3, 10)
    return (sys.version_info.major, sys.version_info.minor)


def get_python_full_version() -> str:

    version_tuple = get_qgis_python_version()
    versions = resolved_python_versions()
    if version_tuple in versions:
        return versions[version_tuple]


    _log(
        f"Python {version_tuple[0]}.{version_tuple[1]} has no pinned build, falling back to 3.13",
        Qgis.MessageLevel.Warning)
    return versions[(3, 13)]


def _create_python_symlinks(python_dir: str) -> None:

    bin_dir = os.path.join(python_dir, "bin")
    python3_path = os.path.join(bin_dir, "python3")
    if os.path.exists(python3_path):
        return

    major, minor = get_qgis_python_version()
    versioned = os.path.join(bin_dir, f"python{major}.{minor}")
    if os.path.exists(versioned):
        os.symlink(f"python{major}.{minor}", python3_path)
        _log(f"Created python3 symlink -> python{major}.{minor}")


def get_standalone_python_path() -> str:

    python_dir = os.path.join(STANDALONE_DIR, "python")

    if sys.platform == "win32":
        return os.path.join(python_dir, "python.exe")
    return os.path.join(python_dir, "bin", "python3")


def standalone_python_exists() -> bool:

    python_path = get_standalone_python_path()
    return os.path.exists(python_path)


def standalone_python_is_current() -> bool:




    python_path = get_standalone_python_path()
    if not os.path.exists(python_path):
        return False

    try:



        env = get_clean_env_for_venv()
        env["PYTHONIOENCODING"] = "utf-8"

        result = run_unthrottled(
            [python_path, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            text=True, encoding="utf-8", errors="replace",
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





    system = sys.platform
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64") or IS_ROSETTA:
            return ("aarch64-apple-darwin", ".tar.gz")
        return ("x86_64-apple-darwin", ".tar.gz")
    if system == "win32":




        if machine in ("arm64", "aarch64"):
            _log(
                "Windows on ARM: using the x86_64 Python build, which matches "
                "this QGIS process.", Qgis.MessageLevel.Info)
        return ("x86_64-pc-windows-msvc", ".tar.gz")

    if unsupported_download_platform_reason():
        return ("", "")
    if machine in ("arm64", "aarch64"):
        return ("aarch64-unknown-linux-gnu", ".tar.gz")
    return ("x86_64-unknown-linux-gnu", ".tar.gz")


def get_download_urls() -> list[str]:







    release_tag = resolved_release_tag()
    names = _asset_names_for(get_python_full_version(), release_tag)
    if not names:
        return []
    base = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{release_tag}"
    )
    return [f"{base}/{name}" for name in names]




_HASH_BLOCK_BYTES = 1024 * 1024


def _sha256_file(filepath: str) -> str:





    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(_HASH_BLOCK_BYTES), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _verify_python_payload(filepath: str, asset_name: str) -> tuple[bool, str]:





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
    if not urls:



        message = (
            f"No standalone Python {python_version} build is published for "
            f"{platform.system()} {platform.machine()}. Automatic mode works "
            "without it; Semi-Auto needs a system Python instead."
        )
        _log(message, Qgis.MessageLevel.Critical)
        return False, message

    _log(f"Downloading Python {python_version} from: {urls[0]}", Qgis.MessageLevel.Info)

    if progress_callback:
        progress_callback(0, tr("Downloading Python {version}...").format(
            version=python_version))



    fd, temp_path = tempfile.mkstemp(suffix=".tar.gz", dir=plugin_cache_tmp_dir())
    os.close(fd)

    try:
        if cancel_check and cancel_check():
            return False, "Download cancelled"

        if progress_callback:
            progress_callback(5, tr("Connecting to download server..."))








        max_retries = 3
        last_error = ""

        def on_bytes(received: int, total: int) -> None:





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





                from .server_dials import dial_in_range
                hard_timeout_ms = dial_in_range(
                    "tuning.install.python_download_hard_timeout_ms",
                    _DOWNLOAD_HARD_TIMEOUT_MS, 60_000, 4 * 60 * 60 * 1000)
                result = stream_url_to_file(
                    url,
                    temp_path,
                    hard_timeout_ms,
                    DOWNLOAD_TIMEOUT_MS,
                    progress_callback=on_bytes,
                    cancel_check=cancel_check,
                )
                if result.cancelled:
                    discard_part_file(temp_path)
                    return False, "Download cancelled"
                if result.ok:
                    break

                error_msg = result.error or "Download failed"
                if _asset_unavailable(result.http_status, error_msg):



                    discard_part_file(temp_path)
                    break




                if attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)
                    _log(
                        f"Download failed (attempt {attempt + 1}/{max_retries}): {error_msg}. "
                        f"Retrying in {wait}s...",
                        Qgis.MessageLevel.Warning
                    )
                    if progress_callback:
                        progress_callback(5, tr(
                            "Network error, retrying in {seconds}s...").format(
                                seconds=wait))


                    if sleep_unless_cancelled(wait, cancel_check):
                        discard_part_file(temp_path)
                        return False, "Download cancelled"

            if result is None or not result.ok:
                unavailable = _asset_unavailable(
                    result.http_status if result is not None else None,
                    error_msg,
                )


                discard_part_file(temp_path)
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


                last_error = "Download failed: received empty file (0 bytes)"
                _log(last_error, Qgis.MessageLevel.Warning)
                continue
            min_expected = 10 * 1024 * 1024
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


            asset_name = url.rsplit("/", 1)[-1]
            ok, verify_msg = _verify_python_payload(temp_path, asset_name)
            if not ok:
                _log(verify_msg, Qgis.MessageLevel.Warning)
                last_error = verify_msg
                continue

            _log(f"Download complete ({content_size} bytes), extracting...", Qgis.MessageLevel.Info)

            if progress_callback:
                progress_callback(55, tr("Extracting Python..."))


            if os.path.exists(STANDALONE_DIR):
                _remove_standalone_tree(STANDALONE_DIR)

            os.makedirs(STANDALONE_DIR, exist_ok=True)


            if temp_path.endswith(".tar.gz") or temp_path.endswith(".tgz"):
                with tarfile.open(temp_path, "r:gz") as tar:
                    _safe_extract_tar(tar, STANDALONE_DIR)
            else:
                with zipfile.ZipFile(temp_path, "r") as z:
                    _safe_extract_zip(z, STANDALONE_DIR)


            if sys.platform != "win32":
                _create_python_symlinks(os.path.join(STANDALONE_DIR, "python"))

            if progress_callback:
                progress_callback(80, tr("Verifying Python installation..."))


            success, verify_msg = verify_standalone_python()

            if success:
                if progress_callback:
                    progress_callback(100, tr("Python {version} installed").format(
                        version=python_version))
                _log("Python standalone installed successfully", Qgis.MessageLevel.Success)
                return True, f"Python {python_version} installed successfully"

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


        if sys.platform == "win32":
            error_lower = str(e).lower()
            if "denied" in error_lower or "access" in error_lower or "permission" in error_lower:
                antivirus_help = _get_windows_antivirus_help(STANDALONE_DIR)
                _log(antivirus_help, Qgis.MessageLevel.Warning)
                error_msg = f"{error_msg}\n\n{antivirus_help}"

        return False, error_msg
    finally:

        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass





_verified_python_key: tuple | None = None


def _python_file_key(python_path: str) -> tuple | None:

    try:
        st = os.stat(python_path)
    except OSError:
        return None
    return (python_path, st.st_mtime_ns, st.st_size, get_python_full_version())


def verify_standalone_python() -> tuple[bool, str]:





    global _verified_python_key
    python_path = get_standalone_python_path()

    if not os.path.exists(python_path):
        return False, f"Python executable not found at {python_path}"

    file_key = _python_file_key(python_path)
    if file_key is not None and file_key == _verified_python_key:
        return True, f"Python {file_key[3]} verified"
    _verified_python_key = None


    if sys.platform != "win32":
        try:
            import stat

            os.chmod(
                python_path,
                stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        except OSError:
            pass

    try:


        env = get_clean_env_for_venv()
        env["PYTHONIOENCODING"] = "utf-8"












        result = None
        for attempt in range(3):
            try:
                result = run_unthrottled(
                    [python_path, "-c", "import subprocess, sys; print(sys.version)"],
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                    env=env,
                    **get_subprocess_kwargs(),
                )
                break
            except subprocess.TimeoutExpired:




                if sys.platform != "win32" or attempt >= 1:
                    raise
                _log("Python check timed out on its first start, retrying once",
                     Qgis.MessageLevel.Info)
            except OSError as err:
                if err.errno != errno.EAGAIN or attempt == 2:
                    raise
                time.sleep(2)

        if result.returncode == 0:
            version_output = result.stdout.strip().split()[0]
            expected_version = get_python_full_version()





            if version_output != expected_version:
                msg = f"Python version mismatch: got {version_output}, expected {expected_version}"
                _log(msg, Qgis.MessageLevel.Warning)
                return False, f"Version mismatch: downloaded {version_output}, expected {expected_version}"

            _log(f"Verified Python standalone: {version_output}", Qgis.MessageLevel.Success)
            _verified_python_key = file_key
            return True, f"Python {version_output} verified"
        error = result.stderr or "Unknown error"
        _log(f"Python verification failed: {error}", Qgis.MessageLevel.Warning)
        return False, f"Verification failed: {error[:100]}"

    except subprocess.TimeoutExpired:
        return False, "Python verification timed out"
    except Exception as e:
        return False, f"Verification error: {str(e)[:100]}"


def _remove_standalone_tree(path: str) -> None:










    global _verified_python_key
    from .venv_manager import _win_extended_path



    _verified_python_key = None
    if sys.platform != "win32":
        def _retry(func, target, _exc_info):
            os.chmod(target, os.stat(target).st_mode | stat.S_IWRITE)
            func(target)

        if sys.version_info >= (3, 12):


            shutil.rmtree(path, onexc=lambda func, target, exc: _retry(func, target, exc))
        else:
            shutil.rmtree(path, onerror=_retry)
        return
    if not remove_tree_quietly(_win_extended_path(path)):
        raise OSError(f"Could not remove every file under {path}; one is still in use")


def remove_standalone_python() -> tuple[bool, str]:

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
