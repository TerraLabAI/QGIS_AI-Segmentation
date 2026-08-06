
from __future__ import annotations

import os
from contextlib import contextmanager

from qgis.PyQt.QtCore import QLockFile


@contextmanager
def gpkg_write_lock(path: str):






    lock_path = os.path.join(os.path.dirname(os.path.abspath(path)), ".aiseg-write.lock")
    lock = QLockFile(lock_path)


    lock.setStaleLockTime(0)
    acquired = False
    try:
        acquired = lock.tryLock(0)
        yield acquired
    finally:
        if acquired:
            lock.unlock()
