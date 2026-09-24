






from __future__ import annotations

import shutil
import tempfile

_DIRS: list[str] = []


def make_icon_dir(prefix: str) -> str:

    folder = tempfile.mkdtemp(prefix=prefix)
    _DIRS.append(folder)
    return folder


def remove_icon_dirs() -> int:





    count = 0
    while _DIRS:
        shutil.rmtree(_DIRS.pop(), ignore_errors=True)
        count += 1

    try:
        from . import styles, ui_refresh_instructions
        styles._SPIN_ICON_URLS = None
        ui_refresh_instructions._SIGN_URLS.clear()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return count
