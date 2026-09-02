










from __future__ import annotations

import time

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr


def geoms_bbox_centre(geoms: list) -> tuple[float, float] | None:



    box = None
    try:
        for geom in geoms:
            if geom is None or geom.isEmpty():
                continue
            bbox = geom.boundingBox()
            if box is None:
                box = bbox
            else:
                box.combineExtentWith(bbox)
        if box is None:
            return None
        centre = box.center()
        return float(centre.x()), float(centre.y())
    except (RuntimeError, AttributeError, TypeError):
        return None


def boundary_snap_is_offered(plugin) -> bool:





    try:
        from ...core.boundary_snap import boundary_snap_offered

        prompt = str((plugin._auto_run_ctx or {}).get("prompt") or "").strip()
        if not prompt:
            prompt = str((plugin._auto_review or {}).get("prompt") or "")
        return boundary_snap_offered(prompt, len(plugin._auto_objects))
    except Exception:  # noqa: BLE001
        return False


def boundary_snap_tolerance_in_units(plugin, geoms: list) -> float:











    from ...core.boundary_snap import boundary_snap_tolerance_units

    ref = geoms_bbox_centre(geoms)
    if ref is None:
        return 0.0
    return boundary_snap_tolerance_units(
        plugin._auto_crs_metres_per_unit(ref[0], ref[1]))


def set_boundary_snap_skip_reason(plugin, reason: str) -> None:





    if getattr(plugin, "_boundary_snap_skip_reason", None) == reason:
        return
    plugin._boundary_snap_skip_reason = reason
    dock = getattr(plugin, "dock_widget", None)
    if dock is None:
        return
    try:
        dock.set_boundary_snap_notice(reason)
    except (RuntimeError, AttributeError):
        pass


def begin_boundary_snap_pass(plugin, geoms: list, params: dict) -> tuple:












    answer, started = _begin_snap_only(plugin, geoms, params)
    from .review_gap_fill import ReviewSetPass, gap_fill_wanted

    if not gap_fill_wanted(params):
        return answer, started
    if started is None:
        return None, (ReviewSetPass(plugin, params, answer, None), None, time.monotonic())
    snap_pass, memo_key, t0 = started
    return None, (ReviewSetPass(plugin, params, geoms, snap_pass), memo_key, t0)


def _begin_snap_only(plugin, geoms: list, params: dict) -> tuple:

    if not params.get("snap_boundaries"):
        set_boundary_snap_skip_reason(plugin, "")
        return geoms, None
    if not isinstance(geoms, list) or len(geoms) < 2:
        return geoms, None
    from ...core.boundary_snap import boundary_snap_max_objects

    cap = boundary_snap_max_objects()
    if cap > 0 and len(geoms) > cap:



        QgsMessageLog.logMessage(
            f"Auto review: shared borders skipped, {len(geoms)} shapes "
            f"over the {cap} limit",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        set_boundary_snap_skip_reason(plugin, tr(
            "Shared borders is off above {cap} shapes. This result has "
            "{count}.").format(cap=cap, count=len(geoms)))
        return geoms, None
    tolerance = boundary_snap_tolerance_in_units(plugin, geoms)
    if tolerance <= 0:
        set_boundary_snap_skip_reason(plugin, tr(
            "Shared borders needs a position for the shapes and this result "
            "carries none."))
        return geoms, None
    set_boundary_snap_skip_reason(plugin, "")




    memo = getattr(plugin, "_boundary_snap_memo", None)
    memo_key = (tuple(id(g) for g in geoms), round(float(tolerance), 9))
    if memo is not None and memo[0] == memo_key:


        return list(memo[1]), None
    from ...core.boundary_snap_pass import BoundarySnapPass

    try:
        pass_ = BoundarySnapPass(
            geoms, tolerance, plugin._auto_crs_authid or None)
    except Exception as exc:  # noqa: BLE001
        QgsMessageLog.logMessage(
            f"Auto review: shared borders failed, keeping the unsnapped "
            f"shapes ({exc})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return geoms, None
    return None, (pass_, memo_key, time.monotonic())


def finish_boundary_snap_pass(plugin, geoms: list, started: tuple) -> list:






    pass_, memo_key, t0 = started
    try:
        result = pass_.result()
        out = result.geometries
    except Exception as exc:  # noqa: BLE001


        QgsMessageLog.logMessage(
            f"Auto review: shared borders failed, keeping the unsnapped "
            f"shapes ({exc})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        try:
            from ...core.telemetry_errors import track_plugin_error
            track_plugin_error(stage="segment", error_code="boundary_snap_failed",
                               message=type(exc).__name__)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return geoms
    if not isinstance(out, list) or len(out) != len(geoms):
        return geoms
    if memo_key is None:

        return out


    snapped = getattr(result, "snapped", out)
    if not isinstance(snapped, list) or len(snapped) != len(geoms):
        return out
    took_ms = int((time.monotonic() - t0) * 1000)
    from .auto_client_profile import add_review_pass_seconds
    add_review_pass_seconds(plugin, "snap_s", took_ms / 1000.0)



    if took_ms > 50:
        QgsMessageLog.logMessage(
            f"Auto review: shared borders over {len(geoms)} shape(s) "
            f"in {took_ms} ms",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
    plugin._boundary_snap_memo = (memo_key, list(snapped), geoms)
    return out


def apply_boundary_snap_to_set(plugin, geoms: list, params: dict) -> list:








    out, started = begin_boundary_snap_pass(plugin, geoms, params)
    if started is None:
        return out
    while not started[0].step(4096):
        pass
    return finish_boundary_snap_pass(plugin, geoms, started)
