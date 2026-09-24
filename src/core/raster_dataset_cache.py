
































from __future__ import annotations

import os
import threading
from collections import OrderedDict
from contextlib import contextmanager

_RASTERIO_LOCK = threading.Lock()
_GDAL_LOCK = threading.Lock()



_HELD_SLOTS = 3


_rasterio_held: OrderedDict = OrderedDict()
_gdal_held: OrderedDict = OrderedDict()


def dataset_identity(path: str) -> tuple:






    raw = path or ""




    if not raw or raw.startswith("/vsi") or "://" in raw:
        return (raw,)
    key = os.path.normcase(raw)
    try:
        stat = os.stat(path.split("|")[0])
    except OSError:
        return (key,)
    return (os.path.normcase(key), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)


def _close_quietly(dataset) -> None:
    try:
        close = getattr(dataset, "close", None)
        if close is not None:
            close()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _drop_superseded(held: OrderedDict, identity: tuple) -> None:





    for key in [k for k in held if k[0] == identity[0] and k != identity]:
        _close_quietly(held.pop(key))


def _held_slots() -> int:

    try:
        from .server_dials import dial_in_range

        return dial_in_range("tuning.network.raster_held_slots", _HELD_SLOTS, 1, 8)
    except Exception:  # noqa: BLE001
        return _HELD_SLOTS


def _evict_extra(held: OrderedDict) -> None:

    while len(held) > _held_slots():
        _close_quietly(held.popitem(last=False)[1])


@contextmanager
def borrow_rasterio_dataset(path: str):






    import rasterio

    identity = dataset_identity(path)
    with _RASTERIO_LOCK:
        dataset = _rasterio_held.get(identity)
        if dataset is not None and not getattr(dataset, "closed", False):
            _rasterio_held.move_to_end(identity)
            yield dataset
            return
        if dataset is not None:
            _rasterio_held.pop(identity, None)
        _drop_superseded(_rasterio_held, identity)
        dataset = rasterio.open(path)
        _rasterio_held[identity] = dataset
        _evict_extra(_rasterio_held)
        yield dataset


def acquire_gdal_dataset(path: str):













    from osgeo import gdal

    identity = dataset_identity(path)
    with _GDAL_LOCK:
        dataset = _gdal_held.get(identity)
        if dataset is not None:
            _gdal_held.move_to_end(identity)
            return dataset
        _drop_superseded(_gdal_held, identity)
        dataset = gdal.Open(path)
        if dataset is None:
            return None
        _gdal_held[identity] = dataset
        _evict_extra(_gdal_held)
        return dataset


def release_raster_datasets() -> None:


    with _RASTERIO_LOCK:
        for dataset in _rasterio_held.values():
            _close_quietly(dataset)
        _rasterio_held.clear()
    with _GDAL_LOCK:
        for dataset in _gdal_held.values():
            _close_quietly(dataset)
        _gdal_held.clear()
