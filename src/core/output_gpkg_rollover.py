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

# Table names per file, kept for the session and keyed by path. Validated
# against the size and mtime of the file AND of its write-ahead log, so a run
# that appends a table is read again on the next call, for the price of two
# stats. The log matters: while a layer from an earlier run holds the file
# open, SQLite leaves every new table in the "-wal" file and the main file
# does not change at all. A stamp on the main file alone then hands the next
# run a table name that is already taken, and the write replaces that table.
_TABLE_NAMES_CACHE: dict[str, tuple[tuple[int, ...], set[str]]] = {}


def _gpkg_stamp(path: str, stat: os.stat_result) -> tuple[int, ...]:
    """Change stamp of a GeoPackage: main file plus its write-ahead log."""
    try:
        wal = os.stat(path + "-wal")
        return (stat.st_mtime_ns, stat.st_size, wal.st_mtime_ns, wal.st_size)
    except OSError:
        return (stat.st_mtime_ns, stat.st_size, 0, 0)


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


def table_names(path: str) -> set[str] | None:
    """Table names in a GeoPackage, or None when the file will not say.

    Read straight from SQLite on purpose. Asking QGIS means querySublayers,
    which opens every table through OGR: on a shared output file holding a few
    hundred runs that is the seconds every Export click used to pay before it
    could pick a free table name.

    None means "unknown", never "empty": a file held by another writer answers
    nothing, and reading that as zero would keep pouring runs into it, or hand
    out a table name that is already in use.
    """
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return set()
    except OSError:
        # The file is there and will not say: that is unknown, not empty.
        return None
    key = os.path.normcase(path)
    stamp = _gpkg_stamp(path, stat)
    cached = _TABLE_NAMES_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    connection = None
    names = None
    try:
        import sqlite3

        connection = sqlite3.connect(
            read_only_gpkg_uri(path), uri=True, timeout=0.5)
        rows = connection.execute(
            "SELECT table_name FROM gpkg_contents").fetchall()
        names = {str(row[0]) for row in rows if row and row[0]}
    except Exception:  # noqa: BLE001 -- an unreadable file keeps today's path
        names = None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:  # nosec B110
                pass
    if names is not None:
        _TABLE_NAMES_CACHE[key] = (stamp, names)
    return names


def layer_identifiers(path: str) -> set[str] | None:
    """The ``identifier`` of every table in a GeoPackage, or None when the
    file will not say.

    A GeoPackage refuses a second table with an identifier already in
    ``gpkg_contents``, so a writer that picks its human title from the project
    tree alone fails on a file that outlives the tree: a result layer removed
    from the project, a second QGIS instance, or a new project writing into
    the same shared file. The dedupe reads the file, like ``table_names``.

    None means "unknown", never "empty", for the same reason as above.
    """
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return set()
    except OSError:
        return None
    if stat.st_size == 0:
        return set()
    connection = None
    try:
        import sqlite3

        connection = sqlite3.connect(
            read_only_gpkg_uri(path), uri=True, timeout=0.5)
        rows = connection.execute(
            "SELECT identifier FROM gpkg_contents").fetchall()
        return {str(row[0]) for row in rows if row and row[0]}
    except Exception:  # noqa: BLE001 -- an unreadable file keeps today's path
        return None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:  # nosec B110
                pass


def table_count(path: str) -> int | None:
    """How many tables a GeoPackage holds, or None when it will not say."""
    names = table_names(path)
    return None if names is None else len(names)


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
    # Start at the newest file in the sequence, not at the first. Every
    # candidate below it is a file already declared full, and asking each one
    # again means one stat and one SQLite read per run per file.
    start = _highest_existing_index(directory, stem, extension)
    path = (os.path.join(directory, filename) if start < 2
            else os.path.join(directory, f"{stem}_{start}{extension}"))
    for index in range(max(start, 1) + 1, _MAX_GPKG_FILES + 1):
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


def _highest_existing_index(directory: str, stem: str, extension: str) -> int:
    """Index of the last file in the ``stem``, ``stem_2``, ``stem_3`` sequence
    that is on disk. 1 when only the base file is there or the folder is
    unreadable, which is where the walk started before.
    """
    highest = 1
    try:
        entries = os.listdir(directory)
    except OSError:
        return highest
    prefix = f"{stem}_"
    for name in entries:
        base, ext = os.path.splitext(name)
        if ext.lower() != extension.lower() or not base.startswith(prefix):
            continue
        tail = base[len(prefix):]
        if tail.isdigit():
            value = int(tail)
            if 2 <= value <= _MAX_GPKG_FILES and value > highest:
                highest = value
    return highest
