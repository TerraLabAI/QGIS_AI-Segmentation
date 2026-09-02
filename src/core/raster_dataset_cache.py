"""Keep one raster open across the clicks of a session.

A windowed read opens the raster, reads a few hundred kilobytes out of it and
closes it again. On a GeoTIFF that costs almost nothing. On the formats people
actually wait on it is most of the click:

- a COG over http reads its header, its tile index and its overview list on
  every open, which is several round trips before one pixel moves;
- ECW, JPEG2000, HDF and NetCDF build their decoder state on open;
- a VRT re-reads and re-parses every source it points at.

None of that changes between two clicks on the same layer, so this holds the
last dataset and hands it back.

A few rasters are held per backend, not one: a session alternating between two
of them (two orthos compared, an ortho plus a reference mosaic) evicted and
re-opened on every click with a single slot, which is the whole cost this
module exists to remove.

A plain file is re-opened when its size or its modification time moves, so a
raster rewritten under a live session serves its new pixels. A dataset behind
a URL or a GDAL container URI is keyed on the URI alone: there is nothing to
stat, and asking would undo the round trip this exists to save.

The handles are dropped when the session ends and when the plugin unloads
(``release_raster_datasets``), so nothing here keeps a file open once the user
is done with it. That matters on Windows, where an open handle blocks a delete
or a rename.

One lock per backend, never one shared: the rasterio arm falls back to the
GDAL arm on failure, sometimes without leaving its own block, and a single
lock would meet itself there.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from contextlib import contextmanager

_RASTERIO_LOCK = threading.Lock()
_GDAL_LOCK = threading.Lock()

# How many rasters a backend keeps open. Small on purpose: a handle costs
# memory and, on Windows, blocks a delete or a rename.
_HELD_SLOTS = 3

# identity -> dataset per backend, least recently used first.
_rasterio_held: OrderedDict = OrderedDict()
_gdal_held: OrderedDict = OrderedDict()


def dataset_identity(path: str) -> tuple:
    """What has to stay the same for a held dataset to still be the right one.

    For a plain file: the path plus its size and modification time. For a URL
    or a GDAL container URI (``/vsicurl/``, ``NETCDF:"...":var``): the URI
    alone.
    """
    key = os.path.normcase(path or "")
    if not key or key.startswith("/vsi") or "://" in key:
        return (key,)
    try:
        stat = os.stat(path.split("|")[0])
    except OSError:
        return (key,)
    return (key, stat.st_mtime_ns, stat.st_size)


def _close_quietly(dataset) -> None:
    try:
        close = getattr(dataset, "close", None)
        if close is not None:
            close()
    except Exception:  # noqa: BLE001 -- a handle being dropped anyway  # nosec B110
        pass


def _drop_superseded(held: OrderedDict, identity: tuple) -> None:
    """Close a held dataset for the same path whose file has since changed.

    The identity carries the file's size and time, so a rewritten raster gets a
    new key and the stale handle would otherwise sit in the slots.
    """
    for key in [k for k in held if k[0] == identity[0] and k != identity]:
        _close_quietly(held.pop(key))


def _evict_extra(held: OrderedDict) -> None:
    """Close whatever falls off the end of a backend's slots."""
    while len(held) > _HELD_SLOTS:
        _close_quietly(held.popitem(last=False)[1])


@contextmanager
def borrow_rasterio_dataset(path: str):
    """Yield an open rasterio dataset for ``path``. Never close what comes out:
    it belongs to this module and the next click reads it again.

    The lock is held for the whole block, which is the read, so one dataset is
    never read from two threads at once.
    """
    import rasterio

    identity = dataset_identity(path)
    with _RASTERIO_LOCK:
        dataset = _rasterio_held.get(identity)
        if dataset is not None:
            _rasterio_held.move_to_end(identity)
            yield dataset
            return
        _drop_superseded(_rasterio_held, identity)
        dataset = rasterio.open(path)
        _rasterio_held[identity] = dataset
        _evict_extra(_rasterio_held)
        yield dataset


def acquire_gdal_dataset(path: str):
    """The held GDAL dataset for ``path``, opening one if there is none.

    None when it will not open, exactly like ``gdal.Open``. Owned by this
    module: the caller must not close it and must not null it in a finally.

    A plain call rather than the block form above, because the GDAL read runs
    inside a config shadow the caller sets up and unwinds, and wrapping that
    body in another block would mean re-indenting it whole. Nothing is
    serialized here beyond the open itself: the formats that reach this arm are
    read on the GUI thread with nothing else running
    (feature_encoder.crop_read_is_thread_safe), and the crop reader takes one
    raster at a time.
    """
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
    """Drop every held dataset. Called when a session ends and when the plugin
    unloads. Idempotent, and never raises."""
    with _RASTERIO_LOCK:
        for dataset in _rasterio_held.values():
            _close_quietly(dataset)
        _rasterio_held.clear()
    with _GDAL_LOCK:
        for dataset in _gdal_held.values():
            _close_quietly(dataset)
        _gdal_held.clear()
