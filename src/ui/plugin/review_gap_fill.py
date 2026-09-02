"""Fill holes, the whole-set half: gaps enclosed between neighbours.

The per-object refine fills an object's own interior rings. A gap several
touching objects enclose between them is nobody's ring, so it needs every
neighbour at once and runs here, on the assembled visible set, chained after
shared borders under the one step protocol the finalize pump drives. Geometry
in ``core.gap_fill``; this module only decides the cutoff for this run, keeps
the memo, and logs.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from qgis.core import Qgis, QgsMessageLog

from ...core.gap_fill import GapFillPass, gap_fill_summary, gap_fill_thread_min_objects
from ...core.server_dials import feature_enabled
from .review_boundary_snap import geoms_bbox_centre


def gap_fill_wanted(params: dict) -> bool:
    """Whether this reslice fills gaps at all: Fill holes ticked, and the
    fleet switch (``features.review_gap_fill``) not turned off."""
    return bool(params.get("fill_holes")) and feature_enabled("review_gap_fill")


def gap_fill_cutoff_units2(plugin, geoms: list, params: dict) -> float | None:
    """The Fill-holes cutoff for this set in RUN CRS UNITS SQUARED.

    The dial is true ground m2 and one CRS unit is not one metre outside a
    metric projection, so it crosses the same conversion the per-object refine
    uses (core.live_refine), measured in the middle of the set. 0 means every
    enclosed gap, which is what "No limit" means for an object's own rings.
    None when the set carries no position to measure at: a metre value read as
    CRS units would fill a degree-wide region in a geographic run.
    """
    hole_m2 = float(params.get("fill_max_m2") or 0.0)
    if hole_m2 <= 0.0:
        return 0.0
    ref = geoms_bbox_centre(geoms)
    if ref is None:
        return None
    try:
        metres_per_unit = float(plugin._auto_crs_metres_per_unit(ref[0], ref[1]) or 1.0)
        aspect = float(plugin._auto_crs_unit_aspect(ref[0], ref[1]) or 1.0)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None
    if metres_per_unit <= 0.0 or aspect <= 0.0:
        return None
    return hole_m2 / (metres_per_unit * metres_per_unit * aspect)


def begin_gap_fill_pass(plugin, geoms: list, params: dict) -> tuple:
    """``(answer, started)``: an ``answer`` list when nothing runs (the fill is
    off, the set has no position, or the memo holds it), else a ``started``
    tuple ``(pass, memo_key, t0)`` the caller steps and hands to
    ``finish_gap_fill_pass``. Same shape as the shared-borders pair."""
    if not gap_fill_wanted(params) or not isinstance(geoms, list) or len(geoms) < 1:
        return geoms, None
    cutoff = gap_fill_cutoff_units2(plugin, geoms, params)
    if cutoff is None:
        return geoms, None
    memo = getattr(plugin, "_gap_fill_memo", None)
    memo_key = (tuple(id(g) for g in geoms), round(float(cutoff), 9))
    if memo is not None and memo[0] == memo_key:
        return list(memo[1]), None
    return None, (_gap_fill_pass_for(plugin, geoms, cutoff), memo_key, time.monotonic())


def _gap_fill_pass_for(plugin, geoms: list, cutoff: float):
    """The pass over ``geoms``: on its own thread past the served floor, on
    the interface thread under it (or when a thread cannot start).

    The thread gets deep copies. The originals are the review's visible
    shapes and stay in use on the interface thread while the pass runs; the
    runner hands each original back wherever its copy came out untouched, so
    the answer carries the same objects the interface-thread pass would."""
    stop_review_set_thread(plugin)
    floor = gap_fill_thread_min_objects()
    if len(geoms) < max(1, floor):
        return GapFillPass(geoms, cutoff)
    try:
        from qgis.core import QgsGeometry

        from ...core.stepped_pass_thread import SteppedPassThread

        copies = []
        for geom in geoms:
            part = None if geom is None or geom.isEmpty() else geom.constGet()
            copies.append(geom if part is None else QgsGeometry(part.clone()))
        runner = SteppedPassThread(
            GapFillPass(copies, cutoff), inputs=copies, originals=geoms).start()
    except Exception:  # noqa: BLE001 -- the interface-thread pass is still there
        return GapFillPass(geoms, cutoff)
    plugin._review_set_thread = runner
    return runner


def stop_review_set_thread(plugin) -> None:
    """End a gap fill still running on its thread, if any. Called before a
    new pass starts and from every path that abandons the review's shapes
    (through ``_stop_review_refine_thread``). Never raises."""
    runner = getattr(plugin, "_review_set_thread", None)
    plugin._review_set_thread = None
    if runner is None:
        return
    try:
        runner.stop()
    except Exception:  # noqa: BLE001 -- teardown must never propagate  # nosec B110
        pass


def finish_gap_fill_pass(plugin, geoms: list, started: tuple) -> list:
    """The filled set from a pass stepped to the end, or the input list when
    the pass declined or failed, so length and order always match."""
    pass_, memo_key, t0 = started
    out = pass_.result()
    took_ms = int((time.monotonic() - t0) * 1000)
    from .auto_client_profile import add_review_pass_seconds
    add_review_pass_seconds(plugin, "gap_fill_s", took_ms / 1000.0)
    if not isinstance(out, list) or len(out) != len(geoms):
        QgsMessageLog.logMessage(
            f"Auto review: {gap_fill_summary(pass_)}; keeping the shapes as "
            f"they were ({len(geoms)} shape(s), {took_ms} ms)",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        # A pass that failed (not one that declined) left the gaps open in
        # what the user exports; the reason's first word is its class.
        reason = str(getattr(pass_, "skip_reason", "") or "")
        if reason.startswith("failed"):
            try:
                from ...core.telemetry_errors import track_plugin_error
                track_plugin_error(stage="segment",
                                   error_code="gap_fill_pass_failed",
                                   message=reason.split("(", 1)[-1].split(":", 1)[0][:80])
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        return geoms
    if pass_.objects_changed or took_ms > 50:
        QgsMessageLog.logMessage(
            f"Auto review: {gap_fill_summary(pass_)} over {len(geoms)} "
            f"shape(s) in {took_ms} ms",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
    # The key holds the input list too, so no other object can land on one of
    # those identities while the memo is alive.
    plugin._gap_fill_memo = (memo_key, list(out), geoms)
    return out


class ReviewSetPass:
    """Shared borders, then the gap fill, as ONE stepped pass.

    The finalize pump steps whatever ``_begin_boundary_snap`` hands it and
    reads ``result().geometries`` at the end; this keeps that contract while
    two whole-set passes share it. ``result().snapped`` is the shared-borders
    answer alone (None when that pass did not run or failed), for its memo.
    """

    def __init__(self, plugin, params: dict, geoms: list, snap_pass) -> None:
        self._plugin = plugin
        self._params = params
        self._geoms = geoms
        self._snap = snap_pass
        self._snapped: list | None = None
        self._gap_input: list | None = None
        self._gap_started: tuple | None = None
        self._final: list | None = None

    def step(self, count: int = 32) -> bool:
        if self._final is not None:
            return True
        if self._snap is not None:
            if not self._snap.step(count):
                return False
            snapped = None
            try:
                snapped = self._snap.result().geometries
            except Exception:  # noqa: BLE001 -- the gap fill runs on the unsnapped set
                snapped = None
            if isinstance(snapped, list) and len(snapped) == len(self._geoms):
                self._snapped = snapped
            self._snap = None
        if self._gap_input is None:
            base = self._snapped if self._snapped is not None else self._geoms
            answer, started = begin_gap_fill_pass(self._plugin, base, self._params)
            self._gap_input = base
            if started is None:
                self._final = answer
                return True
            self._gap_started = started
        assert self._gap_started is not None  # nosec B101
        if not self._gap_started[0].step(count):
            return False
        self._final = finish_gap_fill_pass(
            self._plugin, self._gap_input, self._gap_started)
        return True

    def result(self) -> SimpleNamespace:
        return SimpleNamespace(geometries=self._final, snapped=self._snapped)
