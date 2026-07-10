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
