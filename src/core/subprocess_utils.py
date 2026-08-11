from __future__ import annotations

import os
import subprocess  # nosec B404
import sys


def _sits_inside(path: str, directory: str) -> bool:
    """Whether ``path`` is under ``directory``. False when either is unusable.

    Resolved, not just joined: a directory reached through a link, or written
    on Windows in the short form the launcher uses while the other side holds
    the long one, is the same directory and has to answer as one.
    """
    if not path or not directory:
        return False
    try:
        parent = os.path.normcase(os.path.realpath(directory))
        child = os.path.normcase(os.path.realpath(path))
    except (OSError, ValueError):
        return False
    return child == parent or child.startswith(parent + os.sep)


def _qgis_install_roots() -> list[str]:
    """The directories the running QGIS was installed into.

    The prefix on its own is not enough on Windows: the certificate bundle
    QGIS exports lives in the installation's ``bin`` folder, which is a
    sibling of the prefix rather than a child of it, so a test against the
    prefix alone never matches the file it is meant to catch. The install root
    does match, and where the launcher did not name it, it is the prefix's
    grandparent. That step is Windows only: on the other two the prefix is
    often ``/usr``, whose grandparent is the whole disk.
    """
    roots = []
    for name in ("OSGEO4W_ROOT", "QGIS_PREFIX_PATH"):
        value = os.environ.get(name, "")
        if value:
            roots.append(value)
    if os.name == "nt":
        prefix = os.environ.get("QGIS_PREFIX_PATH", "")
        if prefix:
            grandparent = os.path.dirname(os.path.dirname(prefix))
            # Never a bare drive, which would swallow everything on it.
            if grandparent and len(grandparent.rstrip("\\/")) > 2:
                roots.append(grandparent)
    return roots


def get_clean_env_for_venv() -> dict:
    """Get a clean environment for running venv subprocesses."""
    env = os.environ.copy()
    qgis_roots = _qgis_install_roots()
    vars_to_remove = [
        "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
        # Where pip puts what it installs. A machine-wide PIP_USER=1, which
        # managed IT sets so nothing lands outside the account, makes every
        # install in a venv stop at "Can not perform a '--user' install". The
        # other three redirect the target directory the same way.
        "PIP_USER", "PIP_TARGET", "PIP_PREFIX", "PYTHONUSERBASE",
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
    # Every variable that REPLACES the trust store rather than adding to it.
    # A dangling one (left by an uninstalled Python distro or a rotated
    # corporate bundle) makes uv emit warnings an error classifier reads as a
    # real TLS failure (#184). A LIVE one shipped by QGIS is dropped too: it
    # holds the public roots QGIS travels with, and pointing the installer at
    # it means a company certificate authority the rest of the machine trusts
    # is the one thing the install does not, so every download fails as an
    # unknown issuer. Removed, the installer reads the machine's own store. A
    # bundle the user pointed somewhere else is theirs and is left alone.
    for store_var, exists in (("SSL_CERT_DIR", os.path.isdir),
                              ("SSL_CERT_FILE", os.path.isfile),
                              ("REQUESTS_CA_BUNDLE", os.path.isfile),
                              ("CURL_CA_BUNDLE", os.path.isfile)):
        store = env.get(store_var, "")
        if not store:
            continue
        if not exists(store) or any(_sits_inside(store, r) for r in qgis_roots):
            env.pop(store_var, None)
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
