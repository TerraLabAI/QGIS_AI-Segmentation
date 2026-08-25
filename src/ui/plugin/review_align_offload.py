












from __future__ import annotations

from ...core.shape_policy_dials import (
    align_gui_max_objects,
    align_offload_budget_s,
    align_phase_budget_s,
    align_process_min_objects,
)




ALIGN_PHASE_BUDGET_S = 10.0




ALIGN_OFFLOAD_BUDGET_S = 150.0






ALIGN_GUI_MAX_OBJECTS = 0




ALIGN_PROCESS_MIN_OBJECTS = 800


def begin_align_pass(plugin, rows: list) -> tuple:







    pass_, budget = _choose_align_pass(plugin, rows)
    _note_align_started(rows, pass_)
    return pass_, budget


def _note_align_started(rows: list, pass_) -> None:



    try:
        from ...core.telemetry_run_context import note_run_detail

        kind = type(pass_).__name__
        if pass_ is None:
            runner = "off"
        elif kind == "ProcessAlignPass":
            runner = "process" if getattr(pass_, "_workers", 0) > 0 else "threads"
        elif kind == "SteppedPassThread":
            runner = "thread"
        else:
            runner = "gui"
        note_run_detail(objects=len(rows), align_runner=runner)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _choose_align_pass(plugin, rows: list) -> tuple:
    runner = _begin_process_pass(plugin, rows)
    if runner is not None:
        return runner, align_offload_budget_s(ALIGN_OFFLOAD_BUDGET_S)
    if len(rows) <= int(align_gui_max_objects(ALIGN_GUI_MAX_OBJECTS)):
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
    except Exception:  # noqa: BLE001
        return sweep, align_phase_budget_s(ALIGN_PHASE_BUDGET_S)
    plugin._auto_align_thread = runner
    return runner, align_offload_budget_s(ALIGN_OFFLOAD_BUDGET_S)


def _begin_process_pass(plugin, rows: list):






    try:
        from ...workers.align_process_pool import (
            ProcessAlignPass,
            align_children,
            align_threads,
        )

        floor = align_process_min_objects(ALIGN_PROCESS_MIN_OBJECTS)
        children = align_children(len(rows), floor)


        threads = 0 if children > 0 else align_threads(len(rows), floor)
        if children <= 0 and threads <= 0:
            return None
        sweep = plugin._auto_footprint_align_sweep(rows)
        if sweep is None:
            return None
        runner = ProcessAlignPass(
            rows, sweep._params, (sweep._kx, sweep._ky), children,
            threads=threads).start()
    except Exception:  # noqa: BLE001
        return None
    plugin._auto_align_thread = runner
    try:
        from qgis.core import Qgis, QgsMessageLog

        QgsMessageLog.logMessage(
            f"Auto detection: aligning {len(rows)} shape(s) on "
            + (f"{children} process(es)" if children > 0 else f"{threads} thread(s)"),
            "AI Segmentation", level=Qgis.MessageLevel.Info)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return runner


def _row_copies(rows: list) -> list:






    from qgis.core import QgsGeometry

    out = []
    for fid, geom, score in rows:
        part = None if geom is None or geom.isEmpty() else geom.constGet()
        out.append((fid, geom if part is None else QgsGeometry(part.clone()),
                    score))
    return out


def finish_align_pass(plugin, pass_, rows: list, timed_out: bool) -> list:






    out = None
    try:
        if timed_out and hasattr(pass_, "stop_and_take"):
            out = pass_.stop_and_take()
        else:
            out = pass_.result()
    except Exception:  # noqa: BLE001
        out = None
    stop_align_thread(plugin)
    try:
        from ...core.telemetry_run_context import note_run_detail

        note_run_detail(align_runner="done")
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    if not isinstance(out, list) or len(out) != len(rows):
        return rows
    return out


def stop_align_thread(plugin) -> None:



    runner = getattr(plugin, "_auto_align_thread", None)
    plugin._auto_align_thread = None
    if runner is None:
        return
    try:
        runner.stop()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
