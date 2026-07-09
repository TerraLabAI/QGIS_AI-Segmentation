



























from __future__ import annotations

import errno
import os
import sys
import threading
import time
import weakref










_STALE_AGE_S = 5 * 60 * 60


def _stale_age_s() -> float:






    from .server_dials import dial_in_range

    return dial_in_range(
        "tuning.install.lock_stale_age_s", _STALE_AGE_S, 4 * 60 * 60, 24 * 60 * 60)




_LOCK_BASENAME = "install.lock"





_CANNOT_EVER_LOCK_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})



_STILL_ACTIVE = 259
_live_install_locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_live_install_mutex = threading.Lock()
_LOCK_RECORD_BYTES = 4096


class InstallBusyError(RuntimeError):
    pass


def default_install_lock_path() -> str:

    from .cache_paths import PLUGIN_CACHE_DIR

    return os.path.join(PLUGIN_CACHE_DIR, _LOCK_BASENAME)


def _pid_is_alive(pid: int) -> bool | None:





    if pid <= 0:
        return False

    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:

            return True
        except OSError:
            return None
        return True

    if sys.platform == "win32":
        return _pid_is_alive_windows(pid)

    return None


def _pid_is_alive_windows(pid: int) -> bool | None:








    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


        kernel32.OpenProcess.restype = ctypes.c_void_p

        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            try:
                code = ctypes.c_ulong(0)
                ok = kernel32.GetExitCodeProcess(
                    ctypes.c_void_p(handle), ctypes.byref(code))
            finally:
                kernel32.CloseHandle(ctypes.c_void_p(handle))
            if not ok:
                return None
            return code.value == _STILL_ACTIVE
        err = ctypes.get_last_error()
        if err == 87:
            return False
        if err == 5:
            return True
        return None
    except Exception:  # noqa: BLE001
        return None


def _process_start_stamp(pid: int) -> str | None:












    if pid <= 0:
        return None
    if sys.platform == "win32":
        return _process_start_stamp_windows(pid)
    if sys.platform == "darwin":
        return _process_start_stamp_darwin(pid)
    return _process_start_stamp_proc(pid)


def _process_start_stamp_windows(pid: int) -> str | None:

    try:
        import ctypes
        import ctypes.wintypes as wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


        kernel32.OpenProcess.restype = ctypes.c_void_p

        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            created = wintypes.FILETIME()
            exited = wintypes.FILETIME()
            in_kernel = wintypes.FILETIME()
            in_user = wintypes.FILETIME()
            ok = kernel32.GetProcessTimes(
                ctypes.c_void_p(handle),
                ctypes.byref(created), ctypes.byref(exited),
                ctypes.byref(in_kernel), ctypes.byref(in_user))
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
        if not ok:
            return None
        return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
    except Exception:  # noqa: BLE001
        return None


def _process_start_stamp_darwin(pid: int) -> str | None:







    try:
        import subprocess  # nosec B404

        result = subprocess.run(  # nosec B603
            ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=5,
        )
        if result.returncode != 0:
            return None



        stamp = "_".join(result.stdout.split())
        return stamp or None
    except Exception:  # noqa: BLE001
        return None


def _process_start_stamp_proc(pid: int) -> str | None:

    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            data = f.read()



        tail = data[data.rindex(")") + 1:].split()
        return tail[19]
    except Exception:  # noqa: BLE001
        return None


def _parse_pid(content: str) -> int | None:

    try:
        pid = int(content.split()[0])
    except (ValueError, IndexError):
        return None
    return pid if pid > 0 else None


def _parse_start_stamp(content: str) -> str | None:





    parts = content.split()
    return parts[2] if len(parts) >= 3 else None


def _file_older_than(path: str, age_s: float) -> bool:
    try:
        mtime = os.path.getmtime(path)
    except OSError:

        return True
    return (time.time() - mtime) > age_s


def _lock_is_stale(path: str, content: str, self_pid: int | None = None) -> bool:














    pid = _parse_pid(content)
    if pid is None:

        return True
    if self_pid is not None and pid == self_pid:
        return True



    recorded_start = _parse_start_stamp(content)
    if recorded_start is not None:
        live_start = _process_start_stamp(pid)
        if live_start is not None and live_start != recorded_start:
            return True






    if _pid_is_alive(pid) is False:
        return True
    return _file_older_than(path, _stale_age_s())


def lock_age_seconds(lock_path: str | None = None) -> float | None:







    path = lock_path or default_install_lock_path()
    try:
        with open(path, encoding="utf-8") as handle:
            content = handle.read(_LOCK_RECORD_BYTES)
    except (OSError, UnicodeError):
        return None
    parts = content.strip().split()
    if len(parts) >= 2:
        try:
            return max(0.0, time.time() - float(parts[1]))
        except (TypeError, ValueError):
            pass
    try:
        return max(0.0, time.time() - os.path.getmtime(path))
    except OSError:
        return None


def _unlink_lock_file(path: str) -> None:







    attempts = 5 if sys.platform == "win32" else 1
    for attempt in range(attempts):
        try:
            os.unlink(path)
            return
        except FileNotFoundError:
            return
        except OSError as err:
            if getattr(err, "winerror", None) not in (32, 33) or attempt == attempts - 1:
                return
            time.sleep(0.1 * (attempt + 1))


class InstallLock:







    def __init__(self, lock_path: str | None = None) -> None:
        self._path = lock_path or default_install_lock_path()
        self._acquired = False

        self._owns_file = False


        self._record = ""

    @property
    def path(self) -> str:
        return self._path

    def _try_create(self) -> bool:

        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        except OSError:
            pass  # nosec B110
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        except OSError as err:
            if (sys.platform == "win32" and err.errno == errno.EACCES
                    and os.path.lexists(self._path)):




                return False
            if err.errno in _CANNOT_EVER_LOCK_ERRNOS:





                self._acquired = True
                self._owns_file = False
                return True




            return False
        try:
            start_stamp = _process_start_stamp(os.getpid())
            record = f"{os.getpid()} {time.time()}"
            if start_stamp is not None:
                record = f"{record} {start_stamp}"
            try:
                data = f"{record}\n".encode()
                while data:
                    written = os.write(fd, data)
                    if not written:
                        raise OSError(errno.EIO, "Could not write install lock")
                    data = data[written:]
            finally:
                os.close(fd)
        except OSError:







            try:
                os.unlink(self._path)
            except OSError:
                pass  # nosec B110
            return False
        self._acquired = True
        self._owns_file = True
        self._record = record
        return True

    def _break_if_stale(self) -> bool:

        try:
            with open(self._path, encoding="utf-8", errors="replace") as f:
                content = f.read(_LOCK_RECORD_BYTES)
        except OSError:


            return True




        if not _lock_is_stale(self._path, content, os.getpid()):
            return False
        try:
            os.unlink(self._path)
        except OSError:

            return False
        return True

    def acquire(self) -> bool:




        with _live_install_mutex:
            if self._acquired:
                return True
            key = os.path.normcase(os.path.realpath(self._path))
            if key in _live_install_locks:
                return False
            acquired = self._try_create()
            if not acquired and self._break_if_stale():
                acquired = self._try_create()
            if acquired:
                _live_install_locks[key] = self
            return acquired

    def _file_still_ours(self) -> bool:








        try:
            with open(self._path, encoding="utf-8") as f:
                return f.read(_LOCK_RECORD_BYTES).strip() == self._record.strip()
        except (OSError, UnicodeError):
            return False

    def release(self) -> None:

        with _live_install_mutex:
            if self._acquired and self._owns_file and self._file_still_ours():
                _unlink_lock_file(self._path)
            key = os.path.normcase(os.path.realpath(self._path))
            if _live_install_locks.get(key) is self:
                _live_install_locks.pop(key, None)
            self._acquired = False
            self._owns_file = False
            self._record = ""

    def __enter__(self) -> InstallLock:
        if not self.acquire():
            raise InstallBusyError(
                "Another process is installing the AI components."
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def acquire_install_lock(
    lock_path: str | None = None, timeout_s: float = 0.0
) -> InstallLock:






    lock = InstallLock(lock_path)
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if lock.acquire():
            return lock
        if time.monotonic() >= deadline:
            raise InstallBusyError(
                "Another process is installing the AI components."
            )
        time.sleep(0.25)
