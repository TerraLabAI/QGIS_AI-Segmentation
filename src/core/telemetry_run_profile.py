"""The client side of one Automatic run, as flat telemetry properties.

The service keeps a row per tile, so from a run_id alone it can say how fast
it answered and how many requests were in flight. It cannot say what the
plugin did with those answers: which converter pool it got, how long it sat
on renders, uploads or backoff, how long the finalize took, how big the gaps
between two progress updates were. Until now that lived only in the user's
QGIS log, and reading a slow run meant asking them to paste it.

Two shapes come out of here:

- ``client_props(profile)``: the worker's ``client_profile()`` dict plus the
  GUI-side timings, every key under a ``client_`` prefix, merged into the
  run's terminal event (completed, failed, cancelled). All optional.
- ``review_pass_props(profile)``: what the post-run review spent on its
  shape, gap fill and snap passes, on the export and abandon events.

Also the slow-notice event: the first time a run's card says "slow", what the
client was doing at that moment, so a slow card can be told apart from a slow
link.
"""
from __future__ import annotations

from . import telemetry_events as ev
from .telemetry import track

# Phases the slow notice can name. "converting" and "assembling" are work on
# the user's machine; the card must not blame the connection for them.
SLOW_PHASES = ("waiting_service", "converting", "assembling", "unknown")


def _clean(value):
    """A telemetry-safe scalar: floats rounded, bools and ints kept, strings
    cut short, anything else dropped (None)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str):
        return value[:200]
    return None


def client_props(profile: dict | None) -> dict:
    """``profile`` flattened under the ``client_`` prefix. Nested ``loop``
    buckets come out as ``client_loop_<name>_s``. Empty in, empty out."""
    if not profile:
        return {}
    out: dict = {}
    for key, value in profile.items():
        if key == "loop" and isinstance(value, dict):
            for name, secs in value.items():
                cleaned = _clean(secs)
                if cleaned is not None:
                    out[f"client_loop_{name}_s"] = cleaned
            continue
        cleaned = _clean(value)
        if cleaned is not None:
            out[f"client_{key}"] = cleaned
    return out


def review_pass_props(profile: dict | None) -> dict:
    """The review pass profile as event properties: shape_pass_s, gap_fill_s,
    snap_s, objects, on_pool. Only the keys the review filled in."""
    if not profile:
        return {}
    out: dict = {}
    for key in ("shape_pass_s", "gap_fill_s", "snap_s", "objects", "on_pool"):
        if key in profile:
            cleaned = _clean(profile[key])
            if cleaned is not None:
                out[key] = cleaned
    return out


def slow_notice_phase(all_answered: bool, awaiting_conversion: int,
                      inflight: int) -> str:
    """Which side owns a silence: the answers are all in and the shapes are
    being built (assembling), answers sit with the converter (converting),
    requests are out and nothing came back (waiting_service), or nothing the
    worker knows explains it (unknown)."""
    if all_answered:
        return "assembling"
    if awaiting_conversion > 0:
        return "converting"
    if inflight > 0:
        return "waiting_service"
    return "unknown"


def track_auto_run_slow_notice(run_id: str, silent_for_s: float, tiles_answered: int,
                               tiles_awaiting_conversion: int, inflight: int,
                               convert_pool: str, phase: str) -> None:
    """Once per run, the first time the card says the run is slow."""
    if phase not in SLOW_PHASES:
        phase = "unknown"
    track(ev.AUTO_RUN_SLOW_NOTICE, {
        "run_id": run_id,
        "silent_for_s": round(float(silent_for_s), 1),
        "tiles_answered": int(tiles_answered),
        "tiles_awaiting_conversion": int(tiles_awaiting_conversion),
        "inflight": int(inflight),
        "convert_pool": str(convert_pool or "")[:16],
        "phase": phase,
    })


def machine_profile() -> dict:
    """The static half of the client profile: os, versions, cores, memory.
    Never raises; a field that cannot be read is left out."""
    import os as _os
    import platform

    out: dict = {"os": platform.system(), "cpu_count": _os.cpu_count() or 0}
    try:
        from qgis.core import Qgis
        out["qgis_version"] = str(Qgis.QGIS_VERSION)
    except Exception:  # noqa: BLE001 -- outside QGIS  # nosec B110
        pass
    try:
        from .telemetry import _read_plugin_version
        out["plugin_version"] = str(_read_plugin_version())
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    try:
        from ..workers.tile_convert_pool import usable_cores
        out["usable_cores"] = int(usable_cores())
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    try:
        import psutil  # optional; not in the plugin's required packages
        out["rss_mb"] = round(psutil.Process().memory_info().rss / 1048576.0, 1)
    except Exception:  # noqa: BLE001 -- no psutil, or no permission  # nosec B110
        pass
    return out
