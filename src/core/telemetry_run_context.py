














from __future__ import annotations

import json
import math
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


class _RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.attempt_id = ""
        self.active_run_id = ""
        self.phase = _PHASE_AT_START
        self.run_attempts: OrderedDict[str, str] = OrderedDict()
        self.review_opened: OrderedDict[str, float] = OrderedDict()


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





def active_run_id() -> str:

    with _state.lock:
        return _state.active_run_id


def note_run_phase(name: str) -> None:

    if name in _RUN_PHASES:
        with _state.lock:
            _state.phase = name


def run_failure_stage() -> str:

    with _state.lock:
        return _state.phase


def note_run_started(run_id: str, tiles: int) -> None:

    run_id = str(run_id or "")
    if not run_id:
        return
    with _state.lock:
        _state.active_run_id = run_id
        _state.phase = _PHASE_AT_START
        _remember(_state.run_attempts, run_id, _state.attempt_id)
    try:
        if not is_telemetry_enabled():
            return
        previous = _read_marker()
        if previous is not None and previous["run_id"] != run_id:

            _send_lost_run(previous)
        _write_marker(run_id, tiles)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def note_run_ended(run_id: str) -> None:




    run_id = str(run_id or "")
    with _state.lock:
        target = run_id or _state.active_run_id
        if target and target == _state.active_run_id:
            _state.active_run_id = ""
    if not target:
        return
    try:
        marker = _read_marker()
        if marker is not None and marker["run_id"] == target:
            _drop_marker()
    except Exception:  # noqa: BLE001  # nosec B110
        pass





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
    track(_LOST_EVENT, {
        "run_id": marker["run_id"],
        "age_s": age_s,
        "tiles": marker["tiles"],
    }, flush_now=True)


def _marker_settings() -> QSettings:
    return QSettings()


def _write_marker(run_id: str, tiles: int) -> None:
    settings = _marker_settings()
    settings.setValue(_MARKER_KEY, json.dumps({
        "run_id": run_id,
        "started_at": round(_wall_clock(), 3),
        "tiles": max(0, int(tiles or 0)),
    }))

    settings.sync()


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
    return {"run_id": run_id, "started_at": started_at, "tiles": max(0, tiles)}





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
