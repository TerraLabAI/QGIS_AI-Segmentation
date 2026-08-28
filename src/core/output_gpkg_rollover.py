"""Which output GeoPackage the NEXT run writes into.

One project used to mean one file, for good. Every run appends a table to it,
and every write into it costs more as it grows: on a file of a few hundred
tables, one Export spends seconds writing the run table, reopening it and
storing the style and the metadata, against milliseconds on a fresh file.
Past a ceiling the runs roll over to ``ai_segmentation_2.gpkg``, then ``_3``,
and so on.

Layers already on the map keep the path they were opened from, so a rollover
moves nothing and breaks nothing: it only changes where the next run lands.

Its own module rather than a block inside ``output_store``: it answers one
question, it is the only place in the plugin that reads a GeoPackage through
SQLite instead of through QGIS, and that file is already at the size a file in
this repo may reach.
"""
from __future__ import annotations

import os
from pathlib import Path

#: How many tables one output GeoPackage may hold before the next run rolls
#: over. 64 keeps the file writes on one Export short and still holds months
#: of runs.
GPKG_MAX_TABLES = 64

#: How large one output GeoPackage may grow before the next run rolls over.
#: The table ceiling alone is not enough: it was set when a run wrote about
#: half a megabyte, and a run that returns 20000 objects writes 10.6 MB, so 64
#: tables of those is a 670 MB file. Dropping a table does not shrink it back
#: either, because nothing in the plugin runs VACUUM. Whichever ceiling a file
#: reaches first sends the next run to a new one.
GPKG_MAX_BYTES = 256 * 1024 * 1024

#: Where the numbering stops. A user who reaches this has other problems, and a
#: bounded loop cannot spin on a directory that answers oddly.
_MAX_GPKG_FILES = 999

# Table counts per file, kept for the session and keyed by path. Validated
# against the file's size and mtime, so a run that appends a table is counted
# again on the next call, for the price of one stat.
_TABLE_COUNT_CACHE: dict[str, tuple[tuple[int, int], int]] = {}


def gpkg_table_ceiling() -> int:
    """Tables one output GeoPackage may hold, server-tunable.

    Bounded on both sides: a handful of tables would scatter one project's runs
    over a dozen files, and a very large ceiling brings back the slow click
    this exists to avoid. Cache-only and never raises, so it is safe on the
    write path and offline.
    """
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range(
            "export_policy.gpkg_max_tables", GPKG_MAX_TABLES, 8, 4096))
    except Exception:  # noqa: BLE001 -- the ceiling is best-effort  # nosec B110
        return GPKG_MAX_TABLES


def gpkg_byte_ceiling() -> int:
    """Bytes one output GeoPackage may hold, server-tunable.

    Bounded like the table ceiling: too small scatters one project's runs over
    a dozen files, too large brings back the slow write this exists to avoid.
    Cache-only and never raises, so it is safe on the write path and offline.
    """
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range(
            "export_policy.gpkg_max_bytes", GPKG_MAX_BYTES,
            16 * 1024 * 1024, 4 * 1024 * 1024 * 1024))
    except Exception:  # noqa: BLE001 -- the ceiling is best-effort  # nosec B110
        return GPKG_MAX_BYTES


def read_only_gpkg_uri(path: str) -> str:
    """SQLite read-only URI for a GeoPackage path, a Windows share included.

    ``as_uri()`` percent-encodes what a URI cannot carry raw and handles a
    Windows drive letter, which a hand-built "file:" string does not. It also
    turns a share into ``file://server/name``, and SQLite refuses every
    authority but an empty one, so the host moves back into the path.
    """
    uri = Path(path).as_uri()
    if uri.startswith("file://") and not uri.startswith("file:///"):
        uri = "file:////" + uri[len("file://"):]
    return f"{uri}?mode=ro"


def file_size(path: str) -> int | None:
    """Size of a file in bytes, or None when it cannot be read.

    None means "unknown", never "empty", for the same reason table_count says
    so: a file we cannot stat must not be read as room to keep writing into.
    A missing file is 0, which is room.
    """
    try:
        return os.stat(path).st_size
    except FileNotFoundError:
        return 0
    except OSError:
        return None


def table_count(path: str) -> int | None:
    """Tables in a GeoPackage, or None when the file will not say.

    Read straight from SQLite on purpose. Asking QGIS means querySublayers,
    which opens every table through OGR, and that is the cost this rollover
    exists to avoid paying on a click.

    None means "unknown", never "empty": a file held by another writer answers
    nothing, and reading that as zero would keep pouring runs into it.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return 0
    key = os.path.normcase(path)
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _TABLE_COUNT_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    connection = None
    count = None
    try:
        import sqlite3

        connection = sqlite3.connect(
            read_only_gpkg_uri(path), uri=True, timeout=0.5)
        row = connection.execute("SELECT COUNT(*) FROM gpkg_contents").fetchone()
        count = int(row[0]) if row else None
    except Exception:  # noqa: BLE001 -- an unreadable file keeps today's path
        count = None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:  # nosec B110
                pass
    if count is not None:
        _TABLE_COUNT_CACHE[key] = (stamp, count)
    return count


def next_output_gpkg(directory: str, filename: str) -> str:
    """Path of the file the next run should write into, inside ``directory``.

    The first file in the ``name``, ``name_2``, ``name_3`` sequence that is
    under BOTH ceilings, the table count and the size. A ceiling whose figure
    cannot be read lets the file through on that count alone; the other one
    still holds it back. Two ceilings because a run's table went from about half a
    megabyte to 10.6 MB when the tiling seed was fixed: counting tables alone
    let a file reach 670 MB, and every write into it pays for that.
    """
    stem, extension = os.path.splitext(filename)
    ceiling = gpkg_table_ceiling()
    byte_ceiling = gpkg_byte_ceiling()
    path = os.path.join(directory, filename)
    for index in range(2, _MAX_GPKG_FILES + 1):
        count = table_count(path)
        size = file_size(path)
        # One ceiling per line, and each holds on its own: a figure that
        # cannot be read excuses ITS ceiling, never the other one. Reading a
        # count no one can answer as room for a 670 MB file is how the size
        # ceiling used to be cancelled on any file the plugin cannot open.
        room = ((count is None or count <= ceiling)
                and (size is None or size <= byte_ceiling))
        if room:
            return path
        path = os.path.join(directory, f"{stem}_{index}{extension}")
    return path
