"""Shared borders: the whole-set half of the review's shape work.

Every other review control shapes ONE object, so it lives in the per-object
refine. Sharing a border needs every neighbour at once, so it runs on the
assembled visible set instead, right before the review adopts it.

Free functions taking the plugin controller, not a mixin: the review geometry
mixin was over its size band, and this is one self-contained concern with three
entry points. ``AutoReviewGeometryMixin`` keeps the method names and delegates
here, so every caller and every test keeps the name it already uses.
"""
from __future__ import annotations

import time

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr


def geoms_bbox_centre(geoms: list) -> tuple[float, float] | None:
    """Centre of the combined bounding box of ``geoms``, or None when it cannot
    be read. The position a per-CRS ground measure is taken at, so the whole set
    converts on one factor."""
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
    """Whether this run may show the shared-borders control: a land-cover
    object, and few enough shapes for one pass. The prompt comes from the run
    context (the review's own prompt as a fallback), so a restored run answers
    the same way a fresh one does. Fail-closed on any error: the control simply
    does not appear."""
    try:
        from ...core.boundary_snap import boundary_snap_offered

        prompt = str((plugin._auto_run_ctx or {}).get("prompt") or "").strip()
        if not prompt:
            prompt = str((plugin._auto_review or {}).get("prompt") or "")
        return boundary_snap_offered(prompt, len(plugin._auto_objects))
    except Exception:  # noqa: BLE001 -- a gate failure hides the control
        return False


def boundary_snap_tolerance_in_units(plugin, geoms: list) -> float:
    """The shared-borders tolerance for this set, in RUN CRS UNITS.

    The dial is a ground distance (see core.boundary_snap) and a run often works
    in Web Mercator, where one unit is well under a metre: measure the metres
    one unit spans in the middle of the set, then convert.

    A set whose position cannot be read returns 0, which skips the snap. The
    metre value is not a safe stand-in: in a geographic run one unit is a
    degree, so the dial would be applied about a hundred thousand times too wide
    and dissolve the set.
    """
    from ...core.boundary_snap import boundary_snap_tolerance_units

    ref = geoms_bbox_centre(geoms)
    if ref is None:
        return 0.0
    return boundary_snap_tolerance_units(
        plugin._auto_crs_metres_per_unit(ref[0], ref[1]))


def set_boundary_snap_skip_reason(plugin, reason: str) -> None:
    """Tell the panel why shared borders did nothing, once.

    The pass could refuse a set silently (too many shapes, no position to
    measure at), which leaves a ticked control with no effect and nothing on
    screen to explain it. Repeats are dropped: this runs on every reslice."""
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
    """``(answer, started)`` for the whole-set passes over ``geoms``: shared
    borders, then the gap half of Fill holes (review_gap_fill).

    Exactly one of the two is meaningful. An ``answer`` list means there is
    nothing to run: both controls are off, a gate refused the set, or the memo
    already holds this answer. A ``started`` tuple ``(pass, memo_key, t0)``
    means the caller owns a stepped pass it must step to the end and then hand
    to ``finish_boundary_snap_pass``.

    Split from the run so the finalize can slice a pass that costs seconds on a
    set at the offered ceiling, instead of holding the map for it.
    """
    answer, started = _begin_snap_only(plugin, geoms, params)
    from .review_gap_fill import ReviewSetPass, gap_fill_wanted

    if not gap_fill_wanted(params):
        return answer, started
    if started is None:
        return None, (ReviewSetPass(plugin, params, answer, None), None, time.monotonic())
    snap_pass, memo_key, t0 = started
    return None, (ReviewSetPass(plugin, params, geoms, snap_pass), memo_key, t0)


def _begin_snap_only(plugin, geoms: list, params: dict) -> tuple:
    """The shared-borders half of ``begin_boundary_snap_pass``."""
    if not params.get("snap_boundaries"):
        set_boundary_snap_skip_reason(plugin, "")
        return geoms, None
    if not isinstance(geoms, list) or len(geoms) < 2:
        return geoms, None
    from ...core.boundary_snap import boundary_snap_max_objects

    cap = boundary_snap_max_objects()
    if cap > 0 and len(geoms) > cap:
        # The control is not offered above the cap, so this is a safety net (a
        # set that grew after a batch fold). Said out loud: a ticked box that
        # changes nothing reads as a broken control.
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
    # The whole-set snap runs on the interface thread on every reslice, so a
    # filter-only pass over the same shapes paid for it again. The key holds the
    # input list too: that keeps every input geometry alive, so no other object
    # can land on one of their identities.
    memo = getattr(plugin, "_boundary_snap_memo", None)
    memo_key = (tuple(id(g) for g in geoms), round(float(tolerance), 9))
    if memo is not None and memo[0] == memo_key:
        # A copy: the answer becomes the review's visible set, and the review
        # owns that list.
        return list(memo[1]), None
    from ...core.boundary_snap_pass import BoundarySnapPass

    try:
        pass_ = BoundarySnapPass(
            geoms, tolerance, plugin._auto_crs_authid or None)
    except Exception as exc:  # noqa: BLE001 -- keep the unsnapped shapes
        QgsMessageLog.logMessage(
            f"Auto review: shared borders failed, keeping the unsnapped "
            f"shapes ({exc})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return geoms, None
    return None, (pass_, memo_key, time.monotonic())


def finish_boundary_snap_pass(plugin, geoms: list, started: tuple) -> list:
    """The snapped set from a pass the caller has stepped to the end.

    Falls back to the input list whenever the pass refused the set, so the
    length and the order always match the input, which is what keeps the
    parallel score and id lists aligned.
    """
    pass_, memo_key, t0 = started
    try:
        result = pass_.result()
        out = result.geometries
    except Exception as exc:  # noqa: BLE001 -- keep the unsnapped shapes
        # A whole-set GEOS failure here must degrade to "no shared borders",
        # never end the finalize/reslice chain that called it.
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
        # Only the gap fill ran (review_gap_fill logs and memoizes it).
        return out
    # A chained pass carries the shared-borders answer on its own, and that
    # is what the memo must hold: the next reslice fills gaps on it again.
    snapped = getattr(result, "snapped", out)
    if not isinstance(snapped, list) or len(snapped) != len(geoms):
        return out
    took_ms = int((time.monotonic() - t0) * 1000)
    from .auto_client_profile import add_review_pass_seconds
    add_review_pass_seconds(plugin, "snap_s", took_ms / 1000.0)
    # Log only a slow snap: this runs on every reslice while Shared borders is
    # on, so logging each cheap pass would spam the main-thread message log
    # during a confidence drag.
    if took_ms > 50:
        QgsMessageLog.logMessage(
            f"Auto review: shared borders over {len(geoms)} shape(s) "
            f"in {took_ms} ms",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
    plugin._boundary_snap_memo = (memo_key, list(snapped), geoms)
    return out


def apply_boundary_snap_to_set(plugin, geoms: list, params: dict) -> list:
    """Give the VISIBLE set exact shared borders, once, as a whole set, in one
    blocking call.

    Off unless the user ticked the control, and the control itself is only
    offered on a land-cover run small enough for one pass. Returns the input
    list when anything is off, so a caller can always use the result. The
    finalize slices the same pass instead (see ``begin_boundary_snap_pass``).
    """
    out, started = begin_boundary_snap_pass(plugin, geoms, params)
    if started is None:
        return out
    while not started[0].step(4096):
        pass
    return finish_boundary_snap_pass(plugin, geoms, started)
