"""Cross-process install lock for the isolated AI environment.

The cache directory (``~/.qgis_ai_segmentation`` by default) is shared across
every QGIS instance and profile on the machine, yet the install has no
inter-process mutex. Two QGIS windows that start installing at the same time
race: the second one sees the crash-recovery marker and calls
``shutil.rmtree`` on the venv the first one is still building. This module adds
a real OS-level lock so only one live process can rebuild the venv at a time.

Design notes:
- The lock is an exclusive file created with ``O_CREAT | O_EXCL`` (atomic).
- Acquisition is try-once by default (``timeout_s=0``) so it NEVER blocks the
  GUI thread; the caller reports a friendly busy message on failure.
- A leftover lock from a process that has since died must not wedge every
  future install, so a lock is broken when its recorded PID is provably not
  alive, when the content is corrupted, when the file is older than two hours
  (a recycled PID, or when liveness cannot be probed, e.g. some Windows cases),
  or when it records OUR OWN pid but we do not hold it (an install thread
  hard-killed on unload; only a live DIFFERENT process must ever block us).

Pure standard library on purpose: no QGIS import, so it stays importable and
unit-testable outside a QGIS Python.
"""
from __future__ import annotations

import os
import sys
import time

# When liveness cannot be determined (e.g. Windows OpenProcess returns an
# ambiguous error), fall back to age: an install that has not touched the lock
# in this long is assumed dead. Comfortably above the slowest real install
# (torch download caps at 90 min) so a legitimately slow run is never broken.
_STALE_AGE_S = 2 * 60 * 60  # 2 hours

# Default lock filename, resolved lazily against the same cache dir the rest of
# the plugin uses so callers can simply ``acquire_install_lock()``.
_LOCK_BASENAME = "install.lock"


class InstallBusyError(RuntimeError):
    """Raised when another live process already holds the install lock."""


def default_install_lock_path() -> str:
    """Return the default lock path, mirroring venv_manager's CACHE_DIR logic."""
    cache_dir = os.environ.get("AI_SEGMENTATION_CACHE_DIR") or os.path.expanduser(
        "~/.qgis_ai_segmentation"
    )
    return os.path.join(cache_dir, _LOCK_BASENAME)


def _pid_is_alive(pid: int) -> bool | None:
    """Best-effort liveness probe.

    Returns True (alive), False (dead), or None when it cannot be determined
    (the caller then falls back to the file's age).
    """
    if pid <= 0:
        return False

    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # The process exists but is owned by another user (signal denied).
            return True
        except OSError:
            return None
        return True

    if sys.platform == "win32":
        return _pid_is_alive_windows(pid)

    return None


def _pid_is_alive_windows(pid: int) -> bool | None:
    """Windows liveness probe via OpenProcess; None when ambiguous."""
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # PROCESS_QUERY_LIMITED_INFORMATION (Vista+): enough to prove existence.
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        err = ctypes.get_last_error()
        if err == 87:  # ERROR_INVALID_PARAMETER: no such process
            return False
        if err == 5:  # ERROR_ACCESS_DENIED: exists but not queryable
            return True
        return None
    except Exception:  # noqa: BLE001 - probe is best-effort; fall back to mtime
        return None


def _parse_pid(content: str) -> int | None:
    """Extract the PID from lock content; None if missing or malformed."""
    try:
        pid = int(content.split()[0])
    except (ValueError, IndexError):
        return None
    return pid if pid > 0 else None


def _file_older_than(path: str, age_s: float) -> bool:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        # Vanished under us: treat as breakable so the retry can recreate it.
        return True
    return (time.time() - mtime) > age_s


def _lock_is_stale(path: str, content: str, self_pid: int | None = None) -> bool:
    """Decide whether an existing lock file may be broken and reclaimed.

    ``self_pid`` is the current process id when the caller does NOT itself own
    the file (always the case in _break_if_stale, which only runs after our
    O_EXCL create lost the race). A lock recorded under our OWN pid that we do
    not hold is a dead leftover: the genuine holder short-circuits acquire()
    on its own handle before staleness is ever checked, and same-process
    install concurrency is already single-flighted at the UI layer, so reaching
    here with our own pid means a prior install thread was hard-killed on unload
    (worker.terminate()) and left the still-alive QGIS pid behind. Left unbroken
    it would refuse every future install in this process ("Another QGIS window
    is installing") until a full QGIS restart. A live DIFFERENT process is the
    only thing this lock must exclude.
    """
    pid = _parse_pid(content)
    if pid is None:
        # Corrupted / truncated content: no owner we can respect, so break it.
        return True
    if self_pid is not None and pid == self_pid:
        return True
    # A genuinely live foreign process holds it: respect it, but never past the
    # stale age. OR-ing the age check (rather than short-circuiting on "alive")
    # also breaks a lock whose recorded pid has since been RECYCLED to an
    # unrelated live process, and covers the Windows liveness-unknown case;
    # neither can then wedge the lock forever. _STALE_AGE_S stays comfortably
    # above the slowest real install so a legitimately slow run is not broken.
    if _pid_is_alive(pid) is False:
        return True
    return _file_older_than(path, _STALE_AGE_S)


class InstallLock:
    """Exclusive, cross-process file lock.

    Use ``acquire()`` / ``release()`` for the return-a-status style, or as a
    context manager (``with InstallLock(path):``) which raises
    ``InstallBusyError`` if the lock cannot be taken.
    """

    def __init__(self, lock_path: str | None = None) -> None:
        self._path = lock_path or default_install_lock_path()
        self._acquired = False
        # Whether WE created the lock file (only then may release() remove it).
        self._owns_file = False

    @property
    def path(self) -> str:
        return self._path

    def _try_create(self) -> bool:
        """Atomically create the lock file. True on success, False if it exists."""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
        except OSError:
            pass  # nosec B110 - creation below reports the real problem
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        except OSError:
            # Cannot create the lock (read-only fs, permissions, ...). Do not
            # invent a new failure mode: fail OPEN so the install proceeds and
            # its own writability check reports any real problem. We record
            # that we do not own a file, so release() is a no-op.
            self._acquired = True
            self._owns_file = False
            return True
        try:
            try:
                os.write(fd, f"{os.getpid()} {time.time()}\n".encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            # Write/close failed AFTER O_EXCL created the file (flaky network
            # home, roaming profile): the file now sits on disk with our live
            # pid but we do not own it. Remove it so it cannot self-wedge every
            # future acquire in this process, and report the slot as not taken
            # so acquire() retries cleanly. Belt and braces: a leftover that
            # survives the unlink is now treated as stale by _lock_is_stale's
            # own-pid rule.
            try:
                os.unlink(self._path)
            except OSError:
                pass  # nosec B110 - best-effort; staleness reclaims a leftover
            return False
        self._acquired = True
        self._owns_file = True
        return True

    def _break_if_stale(self) -> bool:
        """Remove the existing lock if it is stale. True if it was broken."""
        try:
            with open(self._path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            # File disappeared between the create attempt and this read: the
            # retry can recreate it cleanly.
            return True
        # We reach here only after our own O_EXCL create lost the race, so this
        # instance does NOT own the file: pass our pid so a leftover recorded
        # under it (a hard-killed install thread on unload) is reclaimable
        # instead of wedging every future install until a QGIS restart.
        if not _lock_is_stale(self._path, content, os.getpid()):
            return False
        try:
            os.unlink(self._path)
        except OSError:
            # Another process reclaimed or removed it first; let acquire retry.
            return False
        return True

    def acquire(self) -> bool:
        """Try once to take the lock, breaking a single stale lock if present.

        Never blocks: returns False immediately if a live process holds it.
        """
        if self._acquired:
            return True
        if self._try_create():
            return True
        if self._break_if_stale():
            return self._try_create()
        return False

    def release(self) -> None:
        """Release the lock, removing the file only if we created it."""
        if self._acquired and self._owns_file:
            try:
                os.unlink(self._path)
            except OSError:
                pass  # nosec B110 - already gone or removed by recovery
        self._acquired = False
        self._owns_file = False

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
    """Acquire the install lock and return the held handle.

    With ``timeout_s=0`` (default) this tries exactly once and never blocks the
    GUI. A positive timeout polls until the deadline. Raises
    ``InstallBusyError`` if the lock cannot be taken in time.
    """
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
