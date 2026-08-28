
















from __future__ import annotations

import os
from pathlib import Path




GPKG_MAX_TABLES = 64







GPKG_MAX_BYTES = 256 * 1024 * 1024



_MAX_GPKG_FILES = 999








_TABLE_NAMES_CACHE: dict[str, tuple[tuple[int, ...], set[str]]] = {}


def _gpkg_stamp(path: str, stat: os.stat_result) -> tuple[int, ...]:

    try:
        wal = os.stat(path + "-wal")
        return (stat.st_mtime_ns, stat.st_size, wal.st_mtime_ns, wal.st_size)
    except OSError:
        return (stat.st_mtime_ns, stat.st_size, 0, 0)


def gpkg_table_ceiling() -> int:







    try:
        from .server_dials import dial_in_range

        return int(dial_in_range(
            "export_policy.gpkg_max_tables", GPKG_MAX_TABLES, 8, 4096))
    except Exception:  # noqa: BLE001  # nosec B110
        return GPKG_MAX_TABLES


def gpkg_byte_ceiling() -> int:






    try:
        from .server_dials import dial_in_range

        return int(dial_in_range(
            "export_policy.gpkg_max_bytes", GPKG_MAX_BYTES,
            16 * 1024 * 1024, 4 * 1024 * 1024 * 1024))
    except Exception:  # noqa: BLE001  # nosec B110
        return GPKG_MAX_BYTES


def read_only_gpkg_uri(path: str) -> str:







    uri = Path(path).as_uri()
    if uri.startswith("file://") and not uri.startswith("file:///"):
        uri = "file:////" + uri[len("file://"):]
    return f"{uri}?mode=ro"


def file_size(path: str) -> int | None:






    try:
        return os.stat(path).st_size
    except FileNotFoundError:
        return 0
    except OSError:
        return None


def table_names(path: str) -> set[str] | None:











    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return set()
    except OSError:

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
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        return None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:  # nosec B110
                pass


def table_count(path: str) -> int | None:

    names = table_names(path)
    return None if names is None else len(names)


def next_output_gpkg(directory: str, filename: str) -> str:









    stem, extension = os.path.splitext(filename)
    ceiling = gpkg_table_ceiling()
    byte_ceiling = gpkg_byte_ceiling()



    start = _highest_existing_index(directory, stem, extension)
    path = (os.path.join(directory, filename) if start < 2
            else os.path.join(directory, f"{stem}_{start}{extension}"))
    for index in range(max(start, 1) + 1, _MAX_GPKG_FILES + 1):
        count = table_count(path)
        size = file_size(path)


        journal_size = file_size(path + "-wal")
        if size is not None and journal_size is not None:
            size += journal_size




        room = ((count is None or count < ceiling)
                and (size is None or size <= byte_ceiling))
        if room:
            return path
        path = os.path.join(directory, f"{stem}_{index}{extension}")
    return path


def _highest_existing_index(directory: str, stem: str, extension: str) -> int:




    highest = 1
    try:
        entries = os.listdir(directory)
    except OSError:
        return highest


    prefix = os.path.normcase(f"{stem}_")
    for name in entries:
        base, ext = os.path.splitext(os.path.normcase(name))
        if ext.lower() != extension.lower() or not base.startswith(prefix):
            continue
        tail = base[len(prefix):]
        if tail.isdigit():
            value = int(tail)
            if 2 <= value <= _MAX_GPKG_FILES and value > highest:
                highest = value
    return highest
