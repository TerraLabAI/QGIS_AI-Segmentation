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
from contextlib import contextmanager

_RASTERIO_LOCK = threading.Lock()
_GDAL_LOCK = threading.Lock()

# (identity, dataset) per backend, or None.
_rasterio_held: tuple[tuple, object] | None = None
_gdal_held: tuple[tuple, object] | None = None


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


@contextmanager
def borrow_rasterio_dataset(path: str):
    """Yield an open rasterio dataset for ``path``. Never close what comes out:
    it belongs to this module and the next click reads it again.

    The lock is held for the whole block, which is the read, so one dataset is
    never read from two threads at once.
    """
    global _rasterio_held
    import rasterio

    identity = dataset_identity(path)
    with _RASTERIO_LOCK:
        held = _rasterio_held
        if held is not None and held[0] == identity:
            yield held[1]
            return
        if held is not None:
            _close_quietly(held[1])
            _rasterio_held = None
        dataset = rasterio.open(path)
        _rasterio_held = (identity, dataset)
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
    global _gdal_held
    from osgeo import gdal

    identity = dataset_identity(path)
    with _GDAL_LOCK:
        held = _gdal_held
        if held is not None and held[0] == identity:
            return held[1]
        if held is not None:
            _close_quietly(held[1])
            _gdal_held = None
        dataset = gdal.Open(path)
        if dataset is None:
            return None
        _gdal_held = (identity, dataset)
        return dataset


def release_raster_datasets() -> None:
    """Drop every held dataset. Called when a session ends and when the plugin
    unloads. Idempotent, and never raises."""
    global _rasterio_held, _gdal_held
    with _RASTERIO_LOCK:
        if _rasterio_held is not None:
            _close_quietly(_rasterio_held[1])
        _rasterio_held = None
    with _GDAL_LOCK:
        if _gdal_held is not None:
            _close_quietly(_gdal_held[1])
        _gdal_held = None
