"""The run-wide footprint alignment stepped off the interface thread.

The alignment is a whole-set pass at finalize, between the redundancy sweep
and the object build (`auto_finalize_steps`). It is fine-grained coordinate
work per object, so on the interface thread a large run spends its whole
budget and stops with most of its shapes untouched. Past a served object
floor the pass moves to its own thread, over copies, and gets the longer
budget: what a budget guards there is a wait screen, not a frozen interface.

Same shape as `review_gap_fill`: `begin_align_pass` hands the caller a pass
it steps exactly as before, and `stop_align_thread` rides every abandon path.
"""
from __future__ import annotations

from ...core.shape_policy_dials import (
    align_offload_budget_s,
    align_phase_budget_s,
    align_thread_min_objects,
)

# Seconds the pass may take on the interface thread. One object costs about a
# millisecond, so this is the point past which the wait between the last tile
# and the review stops being worth a few squared corners.
ALIGN_PHASE_BUDGET_S = 10.0

# Seconds it may take on its own thread. Far longer, because the interface is
# drawing throughout: the cost is a wait screen the user is already watching,
# and the alternative is a run whose shapes are mostly unaligned.
ALIGN_OFFLOAD_BUDGET_S = 150.0

# Fewest objects worth a thread. Under it the pass ends between two drawn
# frames and the copies would cost more than the thread saves.
ALIGN_THREAD_MIN_OBJECTS = 1500


def begin_align_pass(plugin, rows: list) -> tuple:
    """``(pass, budget_seconds)`` for the run-wide alignment over ``rows``.

    ``(None, 0.0)`` when the pass is off for this run (the server has not
    opted the prompt family in, or the run has no metre frame). The caller
    steps the pass and reads its counters exactly as it does the sweep, on
    the thread or off it.
    """
    floor = max(1, int(align_thread_min_objects(ALIGN_THREAD_MIN_OBJECTS)))
    if len(rows) < floor:
        return plugin._auto_footprint_align_sweep(rows), align_phase_budget_s(
            ALIGN_PHASE_BUDGET_S)
    copies = _row_copies(rows)
    sweep = plugin._auto_footprint_align_sweep(copies)
    if sweep is None:
        return None, 0.0
    try:
        from ...core.stepped_pass_thread import SteppedPassThread

        runner = SteppedPassThread(
            sweep, step_count=8, inputs=copies, originals=rows).start()
    except Exception:  # noqa: BLE001 -- the interface-thread pass is still there
        return sweep, align_phase_budget_s(ALIGN_PHASE_BUDGET_S)
    plugin._auto_align_thread = runner
    return runner, align_offload_budget_s(ALIGN_OFFLOAD_BUDGET_S)


def _row_copies(rows: list) -> list:
    """``(fid, geometry, score)`` rows with their own geometry instances.

    The originals stay in reach of the interface thread while the pass runs
    (the finalize keeps them as its fallback), and a geometry read from two
    threads at once is not something to reason about per call site.
    """
    from qgis.core import QgsGeometry

    out = []
    for fid, geom, score in rows:
        part = None if geom is None or geom.isEmpty() else geom.constGet()
        out.append((fid, geom if part is None else QgsGeometry(part.clone()),
                    score))
    return out


def finish_align_pass(plugin, pass_, rows: list, timed_out: bool) -> list:
    """The aligned rows, or ``rows`` unchanged when the pass answered nothing.

    ``timed_out`` ends a threaded pass where it stands and takes what it has:
    the objects it reached keep their aligned shape, the rest keep the one
    they came in with, which is what the interface-thread pass always did.
    """
    out = None
    try:
        if timed_out and hasattr(pass_, "stop_and_take"):
            out = pass_.stop_and_take()
        else:
            out = pass_.result()
    except Exception:  # noqa: BLE001 -- alignment must never block finalize
        out = None
    stop_align_thread(plugin)
    if not isinstance(out, list) or len(out) != len(rows):
        return rows
    return out


def stop_align_thread(plugin) -> None:
    """End an alignment still running on its thread, if any. Called when the
    pass ends and from every path that abandons the run's shapes (through
    ``_stop_review_refine_thread``). Never raises."""
    runner = getattr(plugin, "_auto_align_thread", None)
    plugin._auto_align_thread = None
    if runner is None:
        return
    try:
        runner.stop()
    except Exception:  # noqa: BLE001 -- teardown must never propagate  # nosec B110
        pass
