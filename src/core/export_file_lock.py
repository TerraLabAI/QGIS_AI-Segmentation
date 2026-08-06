











from __future__ import annotations

import os
import time


_SHAPEFILE_SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix",
                       ".sbn", ".sbx", ".qpj")



_HOLD_CHECKS = 4
_HOLD_CHECK_DELAY_S = 0.25

_GENERIC_READ = 0x80000000
_OPEN_EXISTING = 3
_ERROR_SHARING_VIOLATION = 32
_ERROR_LOCK_VIOLATION = 33


def export_target_files(output_path: str, driver: str) -> list[str]:

    if driver != "ESRI Shapefile":
        return [output_path] if os.path.isfile(output_path) else []
    directory = os.path.dirname(output_path) or "."
    stem = os.path.normcase(os.path.splitext(os.path.basename(output_path))[0])
    wanted = set(_SHAPEFILE_SIDECARS)
    found: list[str] = []
    try:
        names = os.listdir(directory)
    except OSError:
        return found
    for name in names:
        base, ext = os.path.splitext(name)
        if os.path.normcase(base) == stem and ext.lower() in wanted:
            found.append(os.path.join(directory, name))
    return found


def _held_by_another_program(path: str) -> bool:

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        create = kernel32.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                           wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        close = kernel32.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        invalid = wintypes.HANDLE(-1).value
        handle = create(path, _GENERIC_READ, 0, None, _OPEN_EXISTING, 0, None)
        if handle is None or handle == invalid:
            error = ctypes.get_last_error()  # type: ignore[attr-defined]
            return error in (_ERROR_SHARING_VIOLATION, _ERROR_LOCK_VIOLATION)
        close(handle)
        return False
    except Exception:  # noqa: BLE001
        return False


def held_export_files(output_path: str, driver: str) -> list[str]:





    if os.name != "nt" or driver == "GPKG":
        return []
    held: list[str] = []
    for check in range(_HOLD_CHECKS):
        held = [path for path in export_target_files(output_path, driver)
                if _held_by_another_program(path)]
        if not held:
            return held
        if check + 1 < _HOLD_CHECKS:
            time.sleep(_HOLD_CHECK_DELAY_S)
    return held
