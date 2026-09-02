"""Drop the install's scratch files once nothing can still be reading them.

Every probe and every package install writes its output to a pair of temp
files in the cache directory, and a constraints file beside them. The normal
path deletes them, but a QGIS killed mid-install, or a delete an antivirus
refused, leaves them there and nothing else ever looks again. One pass at
startup clears anything older than a day.
"""
from __future__ import annotations

import os
import time

#: What the install writes into the cache directory. A name has to start with
#: one of these AND end with ".txt" before it is touched.
_TEMP_PREFIXES = ("run_", "pip_", "pip_constraints_")

#: Nothing younger is touched: a live install on a slow link keeps its files
#: open for hours, and a second QGIS window may be running one.
_MAX_AGE_S = 24 * 60 * 60


def sweep_stale_install_temp_files(cache_dir: str) -> int:
    """Delete the install scratch files older than a day. Returns the count.

    Never raises: it runs at startup, where nothing may block the window, and
    a file it cannot delete is simply left for the next start.
    """
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return 0
    cutoff = time.time() - _MAX_AGE_S
    removed = 0
    for name in names:
        if not name.endswith(".txt") or not name.startswith(_TEMP_PREFIXES):
            continue
        path = os.path.join(cache_dir, name)
        try:
            if not os.path.isfile(path) or os.path.getmtime(path) > cutoff:
                continue
            os.unlink(path)
            removed += 1
        except OSError:
            continue
    return removed
