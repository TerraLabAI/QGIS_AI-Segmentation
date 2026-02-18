import os
import sys
import subprocess


def get_clean_env_for_venv() -> dict:
    """Get a clean environment for running venv subprocesses."""
    env = os.environ.copy()
    vars_to_remove = [
        'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV',
        'QGIS_PREFIX_PATH', 'QGIS_PLUGINPATH',
        'PROJ_DATA', 'PROJ_LIB',
        'GDAL_DATA', 'GDAL_DRIVER_PATH',
    ]
    for var in vars_to_remove:
        env.pop(var, None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def get_subprocess_kwargs() -> dict:
    """Get platform-specific subprocess kwargs (hide window on Windows)."""
    kwargs = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs['startupinfo'] = startupinfo
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    return kwargs
