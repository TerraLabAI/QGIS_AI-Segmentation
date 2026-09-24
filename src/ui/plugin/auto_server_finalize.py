















from __future__ import annotations

import time as _t

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr



_APPLY_CHUNK = 256


class AutoServerFinalizeMixin:




    def _arm_server_finalize_record(self, stitcher) -> None:


        self._auto_finalize_fragments = None
        try:
            from ...core.server_finalize import server_finalize_active

            if (not server_finalize_active()
                    or getattr(self, "_auto_is_exemplar_only", False)
                    or not self._server_finalize_prompt()):
                return
            stitcher.fragment_log = []
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _take_server_finalize_record(self, stitcher) -> None:

        self._auto_finalize_fragments = getattr(stitcher, "fragment_log", None)

    def _server_finalize_prompt(self) -> str:
        return str((getattr(self, "_auto_run_ctx", None) or {}).get("prompt") or "").strip()



    def _server_finalize_context(self) -> dict | None:


        merger = getattr(self, "_auto_merger", None)
        merge = getattr(merger, "init_kwargs", None)
        prompt = self._server_finalize_prompt()
        if merger is None or not isinstance(merge, dict) or not prompt:
            return None
        zone_wkt = ""
        clip = getattr(self, "_auto_clip_polygon", None)
        if clip is not None:
            try:
                zone_wkt = clip.asWkt()
            except (RuntimeError, AttributeError):
                zone_wkt = ""
        align = None
        try:
            from ...core.detection_policy import auto_regularize_settings
            from ...core.review_presets import shape_class_for

            settings = auto_regularize_settings(shape_class_for(prompt))
            if settings is not None:
                from qgis.core import QgsProject




                align = dict(settings)
                align["pixel_size"] = float(self._auto_refine_pixel_size())
                align["ellipsoid"] = str(QgsProject.instance().ellipsoid() or "")
                align.update(self._server_finalize_align_dials())
        except Exception:  # noqa: BLE001
            align = None
        return {
            "run_id": self._auto_run_id or "",
            "crs_authid": self._auto_crs_authid or "",
            "zone_wkt": zone_wkt,
            "prompt": prompt,
            "merge": {**merge, **self._server_finalize_merge_dials()},
            "align": align,
        }

    @staticmethod
    def _server_finalize_merge_dials() -> dict:



        from ...core import merger as _merger
        from ...core import transport_dials as _td
        from ...core.detection_policy import merge_scalar
        from ...core.polygon_geometry import COVER_THRESHOLD_DEFAULT

        return {
            "cover_threshold": float(merge_scalar(
                "cover_threshold", COVER_THRESHOLD_DEFAULT)),
            "compact_min_live": int(_td.merge_compact_min_live(
                _merger._COMPACT_MIN_LIVE)),
            "absorbed_pool_mult": int(_td.merge_absorbed_pool_mult(
                _merger._ABSORBED_POOL_MULT)),
        }

    @staticmethod
    def _server_finalize_align_dials() -> dict:


        from ...core import footprint_alignment as _fa
        from ...core.shape_policy_dials import (
            circle_segments,
            consensus_neighbour_cap,
            min_angle_split_deg,
        )

        return {
            "consensus_neighbour_cap": int(consensus_neighbour_cap(
                _fa._CONSENSUS_NEIGHBOUR_CAP)),
            "min_angle_split_deg": float(min_angle_split_deg(
                _fa._MIN_ANGLE_SPLIT_DEG)),
            "circle_segments": int(circle_segments(_fa._CIRCLE_SEGMENTS)),
        }

    def _start_server_finalize(self):


        fragments = getattr(self, "_auto_finalize_fragments", None)
        self._auto_finalize_fragments = None
        if not fragments or getattr(self, "_auto_is_exemplar_only", False):
            return None
        try:
            from ...core.activation_manager import get_auth_header
            from ...core.server_finalize import (
                finalize_timeout_ms,
                finalize_url,
                server_finalize_enabled,
                start_finalize_task,
            )

            url = finalize_url()
            if not server_finalize_enabled() or not url:
                return None
            context = self._server_finalize_context()
            if context is None:
                return None
            auth = get_auth_header()
            if not auth:
                return None
            timeout_ms = finalize_timeout_ms(len(fragments))
            task = start_finalize_task(url, context, fragments, auth, timeout_ms)
            task.pump_until = _t.monotonic() + timeout_ms / 1000.0 + 5.0
            return task
        except Exception as exc:  # noqa: BLE001
            self._note_server_finalize_failure("CLIENT", 0, repr(exc)[:80])
            return None

    def _note_server_finalize_failure(self, reason: str, fragments: int,
                                      detail: str = "") -> None:

        try:
            QgsMessageLog.logMessage(
                f"Auto detection: server finalize failed ({reason}) on "
                f"{fragments} fragment(s); finishing the run locally",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        try:
            from ...core.telemetry_errors import track_plugin_error

            track_plugin_error(
                "segment", f"SERVER_FINALIZE_{reason}",
                f"run {self._auto_run_id or ''} fragments {fragments} {detail}".strip())
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _log_server_finalize_done(self, task, objects: int) -> None:
        stats = getattr(task, "stats", None) or {}
        try:
            QgsMessageLog.logMessage(
                f"Auto detection: server finalize answered {objects} object(s) "
                f"from {task.fragment_count} fragment(s) in {task.elapsed_ms} ms "
                f"(request {task.request_bytes // 1024} KB, "
                f"{int(stats.get('covered_dropped', 0) or 0)} covered dropped, "
                f"{int(stats.get('aligned', 0) or 0)} aligned)",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
        except Exception:  # noqa: BLE001  # nosec B110
            pass



    def _begin_server_finalize_phase(self, state: dict) -> bool:




        task = self._start_server_finalize()
        if task is None:
            return False
        state["phase"] = "server"
        state["server_task"] = task
        if self.dock_widget is not None and state.get("mode") != "reslice":
            try:
                self.dock_widget.set_auto_finalize_phase(tr("Joining the detected parts"))
                state["announced_phase"] = "server"
            except (RuntimeError, AttributeError):
                pass  # nosec B110
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(self._auto_offgui_poll_ms(), self._step_auto_finalize_refine)
        return True

    def _step_server_finalize(self, state: dict, deadline: float) -> None:

        from qgis.PyQt.QtCore import QTimer

        task = state["server_task"]
        if state.get("phase") == "server":
            if not task.done:
                if _t.monotonic() < getattr(task, "pump_until", 0.0):
                    QTimer.singleShot(self._auto_offgui_poll_ms(),
                                      self._step_auto_finalize_refine)
                    return
                task.failure = task.failure or "TIMEOUT"
                self._server_finalize_fallback(state, task)
                return
            if task.rows is None:
                self._server_finalize_fallback(state, task)
                return
            state.update({
                "phase": "server_apply",
                "server_local": self._local_rows_by_fid(),
                "server_cursor": 0,
                "server_out": [],
                "server_dirty": set(),
            })
        if not self._apply_server_rows(state, task.rows, deadline):
            QTimer.singleShot(0, self._step_auto_finalize_refine)
            return
        out = state.pop("server_out")
        dirty = state.pop("server_dirty")
        for key in ("server_local", "server_cursor", "server_task"):
            state.pop(key, None)
        if not out:
            task.failure = "EMPTY"
            self._server_finalize_fallback(state, task)
            return
        self._adopt_server_rows(task, out, dirty)
        self._mark_finalize_phase(state, "autosave")
        self._autosave_billed_results(out)


        state.update({
            "measurer": self._make_auto_area_measurer(),
            "params": self._fresh_review_params(),
            "pixel_size": self._auto_refine_pixel_size(),
        })
        self._seed_finalize_build_phase(state, out)

    def _apply_server_rows(self, state: dict, rows: list, deadline: float) -> bool:



        from qgis.core import QgsGeometry

        local = state["server_local"]
        out = state["server_out"]
        dirty = state["server_dirty"]
        i = int(state["server_cursor"])
        n = len(rows)
        while i < n:
            stop = min(n, i + _APPLY_CHUNK)
            for fid, wkb, score in rows[i:stop]:
                geom = QgsGeometry()
                geom.fromWkb(wkb)
                if geom.isEmpty():
                    continue
                before = local.get(fid)
                try:
                    same = before is not None and bytes(before.asWkb()) == wkb
                except (RuntimeError, AttributeError):
                    same = False
                if not same:
                    dirty.add(fid)
                out.append((fid, geom, score))
            i = stop
            if _t.monotonic() >= deadline:
                break
        state["server_cursor"] = i
        return i >= n

    def _adopt_server_rows(self, task, out: list, dirty: set) -> None:


        self._auto_merger = None
        self._note_stitch_shapes_dirty(dirty)
        self._auto_stitch_shapes_stale = False
        self._log_server_finalize_done(task, len(out))

    def _local_rows_by_fid(self) -> dict:


        local = {}
        try:
            for fid, geom, _score in self._auto_merger.result_scored_ided():
                local[fid] = geom
        except (AttributeError, RuntimeError):
            return {}
        return local

    def _server_finalize_fallback(self, state: dict, task) -> None:


        self._note_server_finalize_failure(
            task.failure or "CLIENT", task.fragment_count)
        for key in ("server_task", "server_local", "server_cursor",
                    "server_out", "server_dirty"):
            state.pop(key, None)
        self._mark_finalize_phase(state, "restore")
        merged_ided, remerge = self._begin_exemplar_finalize_merge()
        self._auto_merger = None
        if remerge is not None:

            from qgis.PyQt.QtCore import QTimer
            state["phase"] = "remerge"
            state["remerge"] = remerge
            state["remerge_t0"] = None
            QTimer.singleShot(0, self._step_auto_finalize_refine)
            return
        self._seed_finalize_sweep_phase(state, merged_ided)



    def _server_finalize_rows_now(self) -> list | None:



        task = self._start_server_finalize()
        if task is None:
            return None
        from qgis.PyQt.QtCore import QEventLoop, QTimer

        poll_ms = max(10, int(self._auto_offgui_poll_ms()))
        while not task.done and _t.monotonic() < task.pump_until:
            wait = QEventLoop()
            QTimer.singleShot(poll_ms, wait.quit)
            wait.exec()
        if not task.done:
            task.failure = task.failure or "TIMEOUT"
        if not task.done or task.rows is None:
            self._note_server_finalize_failure(
                task.failure or "CLIENT", task.fragment_count)
            return None
        state = {"server_local": self._local_rows_by_fid(), "server_cursor": 0,
                 "server_out": [], "server_dirty": set()}
        self._apply_server_rows(state, task.rows, float("inf"))
        out = state["server_out"]
        if not out:
            self._note_server_finalize_failure("EMPTY", task.fragment_count)
            return None
        self._adopt_server_rows(task, out, state["server_dirty"])
        return out
