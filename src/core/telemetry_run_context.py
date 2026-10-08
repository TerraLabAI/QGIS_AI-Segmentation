

















from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections import OrderedDict

from qgis.PyQt.QtCore import QSettings

from . import telemetry_events as ev
from .telemetry import is_telemetry_enabled, track

_MARKER_KEY = "AI_Segmentation/auto_run_marker"



_LOST_EVENT = getattr(ev, "AUTO_RUN_LOST", "auto_run_lost")



_KEEP_RUNS = 16
_MAX_RUN_ID_CHARS = 64



_RUN_PHASES = frozenset({"imagery", "detecting"})
_PHASE_AT_START = "start"






_STAGES = ("start", "render", "submit", "convert", "stitch", "finalize")
_PHASE_TO_STAGE = {"imagery": "render", "detecting": "submit"}



_ALIGN_RUNNERS = frozenset({"off", "process", "threads", "thread", "gui", "done"})



_PROGRESS_WRITE_EVERY_S = 5.0


_EXIT_CLEAN = "clean"
_EXIT_UNCLEAN = "unclean"
_EXIT_SAME_SESSION = "same_session"


class _RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.attempt_id = ""
        self.active_run_id = ""
        self.phase = _PHASE_AT_START
        self.run_attempts: OrderedDict[str, str] = OrderedDict()

        self.run_headless: OrderedDict[str, bool] = OrderedDict()
        self.review_opened: OrderedDict[str, float] = OrderedDict()


        self.marker_doc: dict | None = None
        self.marker_written_at = 0.0
        self.quit_hooked = False


        self.last_run_id = ""
        self.last_run_stage = ""


_state = _RunState()


def _wall_clock() -> float:
    return time.time()


def _mono_clock() -> float:
    return time.monotonic()


def _remember(table: OrderedDict, key: str, value) -> None:
    table[key] = value
    table.move_to_end(key)
    while len(table) > _KEEP_RUNS:
        table.popitem(last=False)





def begin_run_attempt() -> str:

    attempt = str(uuid.uuid4())
    with _state.lock:
        _state.attempt_id = attempt
    return attempt


def end_run_attempt() -> None:

    with _state.lock:
        _state.attempt_id = ""


def run_attempt_props(run_id: str = "") -> dict:





    with _state.lock:
        if run_id:
            attempt = _state.run_attempts.get(run_id, "")
        else:
            attempt = _state.attempt_id
    return {"attempt_id": attempt} if attempt else {}


def run_headless_props(run_id: str) -> dict:



    with _state.lock:
        headless = _state.run_headless.get(str(run_id or ""))
    return {} if headless is None else {"headless": headless}





def active_run_id() -> str:

    with _state.lock:
        return _state.active_run_id


def note_run_phase(name: str) -> None:

    if name in _RUN_PHASES:
        with _state.lock:
            _state.phase = name
        note_run_stage(_PHASE_TO_STAGE[name])


def run_failure_stage() -> str:

    with _state.lock:
        return _state.phase


def note_run_started(run_id: str, tiles: int, headless: bool = False) -> None:


    run_id = str(run_id or "")
    if not run_id:
        return
    with _state.lock:
        _state.active_run_id = run_id
        _state.phase = _PHASE_AT_START
        _state.marker_doc = None
        _remember(_state.run_attempts, run_id, _state.attempt_id)
        _remember(_state.run_headless, run_id, bool(headless))
    try:
        if not is_telemetry_enabled():
            return
        previous = _read_marker()
        if previous is not None and previous["run_id"] != run_id:

            _send_lost_run(previous)
        _write_marker(run_id, tiles)
        _hook_quit_once()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def note_run_ended(run_id: str) -> None:





    run_id = str(run_id or "")
    with _state.lock:
        target = run_id or _state.active_run_id
        if target and target == _state.active_run_id:
            _state.last_run_id = target
            _state.last_run_stage = _current_stage_locked()
            _state.active_run_id = ""
            _state.marker_doc = None
    if not target:
        return
    try:
        marker = _read_marker()
        if marker is not None and marker["run_id"] == target:
            _drop_marker()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _current_stage_locked() -> str:

    stage = (_state.marker_doc or {}).get("stage")
    if stage in _STAGES:
        return stage
    return _PHASE_TO_STAGE.get(_state.phase, _STAGES[0])


def run_stage_for_report() -> dict:



    with _state.lock:
        if _state.active_run_id:
            return {"run_id": _state.active_run_id,
                    "run_stage": _current_stage_locked(),
                    "run_state": "running"}
        if _state.last_run_id:
            return {"run_id": _state.last_run_id,
                    "run_stage": _state.last_run_stage or _STAGES[0],
                    "run_state": "ended"}
    return {}





def report_lost_run() -> None:

    try:
        marker = _read_marker()
        if marker is None:
            return
        if not is_telemetry_enabled():
            _drop_marker()
            return
        with _state.lock:
            running = _state.active_run_id
        if marker["run_id"] == running:
            return
        _send_lost_run(marker)
        _drop_marker()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _send_lost_run(marker: dict) -> None:
    age_s = max(0, int(_wall_clock() - marker["started_at"]))
    props = {
        "run_id": marker["run_id"],
        "age_s": age_s,
        "tiles": marker["tiles"],
    }
    props.update(_lost_run_detail(marker))
    track(_LOST_EVENT, props, flush_now=True)


def _lost_run_detail(marker: dict) -> dict:



    out: dict = {}
    stage = marker.get("stage")
    if stage:
        out["stage"] = stage
    for key in ("tiles_sent", "tiles_done", "objects"):
        if key in marker:
            out[key] = marker[key]
    if marker.get("align_runner"):
        out["align_runner"] = marker["align_runner"]
    if "updated_at" in marker:


        out["last_progress_s"] = max(0, int(marker["updated_at"] - marker["started_at"]))
    exit_kind = _exit_kind(marker)
    if exit_kind:
        out["qgis_exit"] = exit_kind
    return out


def _exit_kind(marker: dict) -> str:





    if marker.get("quit"):
        return _EXIT_CLEAN
    pid = marker.get("pid")
    if pid is None:
        return ""
    return _EXIT_SAME_SESSION if pid == _process_id() else _EXIT_UNCLEAN


def _process_id() -> int:
    return os.getpid()


def _marker_settings() -> QSettings:
    return QSettings()


def _write_marker(run_id: str, tiles: int) -> None:
    now = round(_wall_clock(), 3)
    doc = {
        "run_id": run_id,
        "started_at": now,
        "updated_at": now,
        "tiles": max(0, int(tiles or 0)),
        "stage": _STAGES[0],
        "pid": _process_id(),
    }
    _store_marker(doc)
    with _state.lock:
        _state.marker_doc = doc


def _store_marker(doc: dict) -> None:
    settings = _marker_settings()
    settings.setValue(_MARKER_KEY, json.dumps(doc))

    settings.sync()
    with _state.lock:
        _state.marker_written_at = _mono_clock()


def _drop_marker() -> None:
    settings = _marker_settings()
    settings.remove(_MARKER_KEY)
    settings.sync()


def _read_marker() -> dict | None:



    raw = _marker_settings().value(_MARKER_KEY, "")
    if raw in ("", None):
        return None
    marker = _parse_marker(raw)
    if marker is None:
        _drop_marker()
    return marker


def _parse_marker(raw) -> dict | None:

    if isinstance(raw, (list, tuple)):
        raw = ",".join(str(piece) for piece in raw)
    try:
        data = json.loads(raw)
        run_id = data["run_id"]
        started_at = float(data["started_at"])
        tiles = int(data.get("tiles") or 0)
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    if not isinstance(run_id, str) or not 0 < len(run_id) <= _MAX_RUN_ID_CHARS:
        return None
    if not math.isfinite(started_at) or started_at <= 0:
        return None
    marker = {"run_id": run_id, "started_at": started_at, "tiles": max(0, tiles)}
    marker.update(_parse_marker_detail(data))
    return marker


def _parse_marker_detail(data: dict) -> dict:


    out: dict = {}
    if data.get("stage") in _STAGES:
        out["stage"] = data["stage"]
    for key in ("tiles_sent", "tiles_done", "objects", "pid"):
        value = data.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            out[key] = value
    if data.get("align_runner") in _ALIGN_RUNNERS:
        out["align_runner"] = data["align_runner"]
    if data.get("quit") is True:
        out["quit"] = True
    updated = data.get("updated_at")
    if isinstance(updated, (int, float)) and not isinstance(updated, bool) \
            and math.isfinite(updated) and updated > 0:
        out["updated_at"] = float(updated)
    return out





def note_run_stage(stage: str) -> None:



    if stage not in _STAGES:
        return
    _update_marker({"stage": stage}, force=True)


def note_run_progress(tiles_sent: int, tiles_done: int, stage: str = "") -> None:




    changes: dict = {
        "tiles_sent": max(0, int(tiles_sent or 0)),
        "tiles_done": max(0, int(tiles_done or 0)),
    }
    if stage in _STAGES:
        changes["stage"] = stage
    _update_marker(changes, force=False)


def progress_write_due() -> bool:


    with _state.lock:
        if _state.marker_doc is None:
            return False
        return _mono_clock() - _state.marker_written_at >= _PROGRESS_WRITE_EVERY_S


def note_run_detail(objects: int | None = None, align_runner: str = "") -> None:


    changes: dict = {"stage": "finalize"}
    if objects is not None:
        changes["objects"] = max(0, int(objects))
    if align_runner in _ALIGN_RUNNERS:
        changes["align_runner"] = align_runner
    _update_marker(changes, force=True)


def _update_marker(changes: dict, force: bool) -> None:





    try:
        with _state.lock:
            doc = _state.marker_doc
            if doc is None:
                return
            merged = dict(doc)
            changed = False
            for key, value in changes.items():
                if key == "stage":
                    if _STAGES.index(value) <= _STAGES.index(merged.get("stage", _STAGES[0])):
                        continue
                if merged.get(key) != value:
                    merged[key] = value
                    changed = True
            if not changed:
                return
            stage_moved = merged.get("stage") != doc.get("stage")
            due = _mono_clock() - _state.marker_written_at >= _PROGRESS_WRITE_EVERY_S
            if not (force or stage_moved or due):
                return
            merged["updated_at"] = round(_wall_clock(), 3)
            _state.marker_doc = merged
        _store_marker(merged)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _hook_quit_once() -> None:



    with _state.lock:
        if _state.quit_hooked:
            return
        _state.quit_hooked = True
    try:
        from qgis.core import QgsApplication

        app = QgsApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(_on_about_to_quit)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _on_about_to_quit() -> None:



    try:
        marker = _read_marker()
        if marker is None or marker.get("quit"):
            return
        raw = _marker_settings().value(_MARKER_KEY, "")
        if isinstance(raw, (list, tuple)):
            raw = ",".join(str(piece) for piece in raw)
        doc = json.loads(raw)
        doc["quit"] = True
        _store_marker(doc)
        with _state.lock:

            if _state.marker_doc is not None:
                _state.marker_doc["quit"] = True
    except Exception:  # noqa: BLE001  # nosec B110
        pass





def note_review_opened(run_id: str) -> None:
    if run_id:
        with _state.lock:
            _remember(_state.review_opened, str(run_id), _mono_clock())


def review_elapsed_props(run_id: str) -> dict:


    with _state.lock:
        opened = _state.review_opened.get(str(run_id or ""))
    if opened is None:
        return {}
    return {"review_ms": max(0, int(round((_mono_clock() - opened) * 1000)))}
