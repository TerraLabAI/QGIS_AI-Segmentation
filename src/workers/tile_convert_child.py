






from __future__ import annotations

import logging
import os

from .tile_convert_threads import _ANY_FAILURE

logger = logging.getLogger(__name__)





STAT_FOLD = {
    "raw_detections_total": "sum",
    "masks_dropped_whole_tile": "sum",
    "masks_whole_tile_armed": "sum",
    "masks_dropped_hard_cover": "sum",
    "masks_dropped_tile_span": "sum",
    "masks_dropped_not_compact": "sum",
    "masks_whole_tile_kept_map": "sum",
    "masks_dropped_map_lowscore": "sum",
    "phase_convert_s": "sum",
    "polygonized_gdal": "sum",
    "polygonized_tracer": "sum",
    "polygonized_fallback": "sum",
    "polygonized_fallback_fast": "sum",
    "map_cover_scores": "extend",
    "observed_mask_gsd": "max",
}




MAX_FRAME_BYTES = 256 * 1024 * 1024


def _write_all(stream, payload) -> None:
    view = memoryview(payload)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise BrokenPipeError("worker pipe stopped accepting bytes")
        view = view[written:]


def _read_exact(stream, size: int):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _send(stream, obj) -> None:

    import pickle  # nosec B403
    import struct

    payload = pickle.dumps(obj, protocol=4)
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds the memory limit")
    _write_all(stream, struct.pack("<I", len(payload)))
    _write_all(stream, payload)
    stream.flush()


def _recv(stream):

    import pickle  # nosec B403
    import struct

    head = _read_exact(stream, 4)
    if head is None:
        return None
    size = struct.unpack("<I", head)[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds the memory limit")
    body = _read_exact(stream, size)
    if body is None:
        return None
    return pickle.loads(body)  # nosec B301


def _fresh(zero):

    return list(zero) if isinstance(zero, list) else zero


def _unfolded_stats(before: dict, after: dict) -> list:










    out = []
    for name, value in after.items():
        if name in STAT_FOLD or not isinstance(value, (int, float)):
            continue
        if isinstance(value, bool):
            continue
        if name in before and before[name] != value:
            out.append(name)
    return out


_UNTHROTTLE_CALL: list = []


def unthrottle_this_process() -> bool:







    import sys

    if sys.platform != "win32":
        return False
    try:
        if not _UNTHROTTLE_CALL:
            import ctypes
            from ctypes import wintypes

            class _ThrottlingState(ctypes.Structure):
                _fields_ = [("Version", wintypes.ULONG),
                            ("ControlMask", wintypes.ULONG),
                            ("StateMask", wintypes.ULONG)]


            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            setter = kernel32.SetProcessInformation
            setter.restype = wintypes.BOOL
            setter.argtypes = [wintypes.HANDLE, ctypes.c_int,
                               ctypes.c_void_p, wintypes.DWORD]

            state = _ThrottlingState(1, 0x1, 0x0)
            _UNTHROTTLE_CALL.append((
                setter, kernel32.GetCurrentProcess(), ctypes.byref(state),
                ctypes.sizeof(state), state))
        setter, handle, ref, size, _state = _UNTHROTTLE_CALL[0]
        return bool(setter(handle, 4, ref, size))
    except Exception:  # noqa: BLE001
        return False







_CHILD_SKIPPED_IMPORTS = ("geopandas",)


def skip_unused_child_imports() -> None:






    import sys

    if sys.platform != "win32":
        return
    for name in _CHILD_SKIPPED_IMPORTS:
        if name not in sys.modules:
            sys.modules[name] = None  # type: ignore[assignment]


def _boot_note(since: float, step: str) -> None:








    import sys
    import time

    try:
        sys.stderr.write(f"boot {step} {time.monotonic() - since:.1f}s\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def child_main() -> None:






    import sys
    import threading
    import time

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer


    sys.stdout = sys.stderr
    booted = time.monotonic()
    _boot_note(booted, "start")
    skip_unused_child_imports()


    try:
        from ..core.macos_activity import promote_current_thread

        promote_current_thread()
    except Exception:  # noqa: BLE001  # nosec B110
        pass

    try:
        from qgis.core import QgsApplication

        _boot_note(booted, "qgis.core")
        app = QgsApplication([], False)
        app.initQgis()
        _boot_note(booted, "initQgis")
        from .auto_detection_worker import AutoDetectionWorker

        _boot_note(booted, "plugin")
    except _ANY_FAILURE as exc:
        try:
            _send(stdout, ("no", repr(exc)))
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return

    worker = None
    stats_reset: dict = {}
    baseline: dict = {}
    _send(stdout, ("ready", None))

    while True:
        frame = _recv(stdin)
        if frame is None:
            break
        kind, payload = frame
        if kind == "init":
            worker = AutoDetectionWorker.__new__(AutoDetectionWorker)
            worker.__dict__.update(payload)




            worker._stat_lock = threading.Lock()
            worker._clip_local = threading.local()


            for name, how in STAT_FOLD.items():
                if how == "extend":
                    stats_reset[name] = []
                elif isinstance(getattr(worker, name, 0), int):
                    stats_reset[name] = 0
                else:
                    stats_reset[name] = 0.0
                setattr(worker, name, _fresh(stats_reset[name]))


            baseline = {k: v for k, v in worker.__dict__.items()
                        if isinstance(v, (int, float))
                        and not isinstance(v, bool)}
            _send(stdout, ("init_ok", None))
            continue
        if kind != "job":
            break
        key, job = payload
        unthrottle_this_process()
        try:
            if worker is None:
                raise RuntimeError("job arrived before init")
            t0 = time.monotonic()
            dets = worker._convert_completed(job)
            stats = {n: getattr(worker, n) for n in STAT_FOLD}
            extra = _unfolded_stats(baseline, worker.__dict__)

            for name, value in stats_reset.items():
                setattr(worker, name, _fresh(value))
            _send(stdout, ("done", (key, dets, stats, extra,
                                    time.monotonic() - t0)))
        except _ANY_FAILURE as exc:
            _send(stdout, ("fail", (key, repr(exc))))


def child_python() -> str | None:

















    import sys

    major, minor = sys.version_info[:2]
    beside_qgis = os.path.dirname(sys.executable or "") or "."
    names = (
        os.path.join(sys.prefix, "python.exe"),
        os.path.join(sys.prefix, "bin", f"python{major}.{minor}"),
        os.path.join(sys.prefix, "bin", "python3"),
        os.path.join(sys.prefix, "bin", "python"),
        os.path.join(beside_qgis, f"python{major}.{minor}"),
        os.path.join(beside_qgis, "python3"),
        os.path.join(beside_qgis, f"python{major}.{minor}.exe"),
    )
    for path in names:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    exe = sys.executable or ""
    if exe and os.path.basename(exe).lower().startswith("python"):
        return exe
    return None


def _stdlib_home() -> str | None:







    import sys
    import sysconfig

    home = sys.base_prefix or sys.prefix
    if not home:
        return None
    candidates = []
    try:
        stdlib = sysconfig.get_paths().get("stdlib")
    except Exception:  # noqa: BLE001
        stdlib = None
    if stdlib:
        candidates.append(stdlib)
    major, minor = sys.version_info[:2]
    candidates.append(os.path.join(home, "Lib"))
    candidates.append(os.path.join(home, "lib", f"python{major}.{minor}"))
    root = os.path.normcase(os.path.abspath(home))
    for path in candidates:



        inside = os.path.normcase(os.path.abspath(path)).startswith(root)
        if inside and os.path.isdir(os.path.join(path, "encodings")):
            return home
    return None


def _child_bytecode_dir() -> str | None:







    import sys

    if sys.platform != "win32":
        return None
    try:
        from ..core.cache_paths import PLUGIN_CACHE_DIR

        path = os.path.join(PLUGIN_CACHE_DIR, "child_bytecode")
        os.makedirs(path, exist_ok=True)
        return path if os.access(path, os.W_OK) else None
    except Exception:  # noqa: BLE001
        return None


def _absolute_path_entries(path_value: str) -> str:








    kept = [p for p in path_value.split(os.pathsep)
            if p and os.path.isabs(p) and not os.path.isfile(p)]
    return os.pathsep.join(kept)




from ..core.subprocess_utils import keep_child_off_power_throttling  # noqa: E402,F401


def _child_stderr_file():







    import tempfile

    try:
        return tempfile.TemporaryFile()
    except OSError:
        pass
    try:
        from ..core.cache_paths import plugin_cache_tmp_dir

        folder = plugin_cache_tmp_dir()
        if folder:
            return tempfile.TemporaryFile(dir=folder)
    except (OSError, ImportError):
        pass
    return None


def child_creation_flags() -> int:







    import subprocess  # nosec B404
    import sys

    if sys.platform != "win32":
        return 0
    return (getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))


def child_cwd() -> str:








    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def spawn_worker_child(exe: str, boot_module: str, env: dict, stderr=None):







    import subprocess  # nosec B404



    proc = subprocess.Popen(  # nosec B603
        [exe, "-s", "-c", f"from {boot_module} import child_main; child_main()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=stderr if stderr is not None else subprocess.DEVNULL,
        env=env, cwd=child_cwd(), close_fds=True,

        creationflags=child_creation_flags())
    keep_child_off_power_throttling(proc)
    return proc


def child_environment() -> dict:



















    import sys
    import sysconfig

    env = dict(os.environ)
    home = _stdlib_home()
    if home:
        env["PYTHONHOME"] = home
    del sysconfig
    parts = [child_cwd()]
    try:
        from qgis.core import QgsApplication

        qgis_python = os.path.join(QgsApplication.prefixPath(), "python")
        if os.path.isdir(qgis_python):
            parts.append(qgis_python)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONNOUSERSITE"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    from ..core.worker_child_env import WORKER_CHILD_ENV
    env[WORKER_CHILD_ENV] = "1"
    env.pop("PYTHONSTARTUP", None)
    bytecode_dir = _child_bytecode_dir()
    if bytecode_dir and not env.get("PYTHONPYCACHEPREFIX"):
        env["PYTHONPYCACHEPREFIX"] = bytecode_dir
    if sys.platform == "win32" and env.get("PATH"):
        env["PATH"] = _absolute_path_entries(env["PATH"])

    logger.debug("TileConvertProcessPool: child sys.path head %s", parts[0])
    del sys
    return env
