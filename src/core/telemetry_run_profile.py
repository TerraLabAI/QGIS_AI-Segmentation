




















from __future__ import annotations

from . import telemetry_events as ev
from .telemetry import track



SLOW_PHASES = ("waiting_service", "converting", "assembling", "unknown")


def _clean(value):


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


    import os as _os
    import platform

    out: dict = {"os": platform.system(), "cpu_count": _os.cpu_count() or 0}
    try:
        from qgis.core import Qgis
        out["qgis_version"] = str(Qgis.QGIS_VERSION)
    except Exception:  # noqa: BLE001  # nosec B110
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
        import psutil
        out["rss_mb"] = round(psutil.Process().memory_info().rss / 1048576.0, 1)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return out
