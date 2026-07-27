from __future__ import annotations

import os
import subprocess  # nosec B404
import sys


def get_clean_env_for_venv() -> dict:
    """Get a clean environment for running venv subprocesses."""
    env = os.environ.copy()
    vars_to_remove = [
        "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
        "QGIS_PREFIX_PATH", "QGIS_PLUGINPATH",
        "PROJ_DATA", "PROJ_LIB",
        "GDAL_DATA", "GDAL_DRIVER_PATH",
        # The dynamic loader's own search path. Flatpak, Snap, AppImage, conda
        # and any hand-rolled launcher set it to QGIS's lib dir, and the
        # downloaded interpreter would then resolve libssl, libffi and
        # libstdc++ from there instead of the system: "GLIBCXX not found" on
        # import, or a segfault. macOS strips the DYLD pair itself under SIP,
        # so this only ever bites on Linux; listed for both so the helper says
        # what it guarantees.
        "LD_LIBRARY_PATH", "LD_PRELOAD",
        "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
    ]
    for var in vars_to_remove:
        env.pop(var, None)
    # Remove SSL_CERT_DIR if it points to a non-existent directory.
    # Invalid paths cause tools like uv to emit "SSL_CERT_DIR" warnings that
    # error classifiers would otherwise misread as real SSL errors (#184).
    ssl_cert_dir = env.get("SSL_CERT_DIR", "")
    if ssl_cert_dir and not os.path.isdir(ssl_cert_dir):
        env.pop("SSL_CERT_DIR", None)
    # Same for its single-bundle sibling: a dangling SSL_CERT_FILE (left by an
    # uninstalled Python distro or a rotated corporate CA bundle) breaks
    # uv/rustls the same way.
    ssl_cert_file = env.get("SSL_CERT_FILE", "")
    if ssl_cert_file and not os.path.isfile(ssl_cert_file):
        env.pop("SSL_CERT_FILE", None)
    env["PYTHONIOENCODING"] = "utf-8"
    # Keep the working directory off the child's sys.path. Around twenty
    # `python -c` probes run with cwd set to the cache directory, and without
    # this a torch.py sitting there would be imported instead of the real one.
    # Ignored by Python below 3.11, which is fine: it is the downloaded
    # interpreter that matters and that one is current.
    env["PYTHONSAFEPATH"] = "1"
    return env


def get_subprocess_kwargs() -> dict:
    """Get platform-specific subprocess kwargs (hide window on Windows)."""
    kwargs = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs
