





from __future__ import annotations

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
)

from ...core.i18n import tr
from ...core.interaction_dials import reslice_screen_first_min_objects
from ...core.server_dials import dial_in_range
from ...core.telemetry_errors import slot_guard
from ..dialogs.confirm_dialog import (
    DANGER,
    DISCARD,
    PRIMARY,
    SECONDARY,
    ChoiceButton,
    ask_choice,
)
from ..error_report_dialog import show_error_report
from .shared import _debounce_timer






_MERGE_SETS_CACHE: dict[str, object] = {"id": None, "sets": None, "merge": None}



_RESLICE_SCREEN_FIRST_MIN_OBJECTS = 400


def _merge_token_sets(merge: dict) -> tuple[frozenset, frozenset, frozenset]:






    key = id(merge)
    if _MERGE_SETS_CACHE["id"] == key and _MERGE_SETS_CACHE["sets"] is not None:
        return _MERGE_SETS_CACHE["sets"]  # type: ignore[return-value]

    def _tokens(vals: object) -> frozenset:
        return frozenset(
            str(v).strip().lower().replace("_", " ")
            for v in (vals or []) if isinstance(v, str)) if isinstance(vals, list) else frozenset()

    def _cats(vals: object) -> frozenset:
        return frozenset(
            str(v).strip().lower()
            for v in (vals or []) if isinstance(v, str)) if isinstance(vals, list) else frozenset()

    result = (
        _tokens(merge.get("continuous_tokens")),
        _tokens(merge.get("discrete_tokens")),
        _cats(merge.get("continuous_categories")),
    )
    _MERGE_SETS_CACHE["id"] = key
    _MERGE_SETS_CACHE["sets"] = result
    _MERGE_SETS_CACHE["merge"] = merge
    return result


def _union_review_sets(geoms: list, scores: list | None, ids: object,
                       full_geoms: list, full_scores: list,
                       full_ids: list) -> tuple[list, list | None, int] | None:









    vis_ids = list(ids) if isinstance(ids, (list, tuple)) else None
    if vis_ids is None or len(vis_ids) != len(geoms):
        return None
    if len(full_ids) != len(full_geoms):
        return None
    out_geoms = list(geoms)
    out_scores = list(scores) if scores is not None and len(scores) == len(geoms) else None
    seen = {det_id for det_id in vis_ids if det_id is not None}
    added = 0
    for i, det_id in enumerate(full_ids):
        if det_id is not None and det_id in seen:
            continue
        g = full_geoms[i]
        if g is None or g.isEmpty():
            continue
        out_geoms.append(g)
        if out_scores is not None:
            out_scores.append(full_scores[i] if i < len(full_scores) else None)
        if det_id is not None:
            seen.add(det_id)
        added += 1
    return out_geoms, out_scores, added


class AutoReviewMixin:




    def _auto_seam_min_dim(self) -> float:


















        from ...core.tile_manager import OVERLAP_FRACTION, TILE_SIZE

        if self._auto_gsd <= 0:
            return float("inf") if self._auto_merge_separate else 0.0
        return OVERLAP_FRACTION * TILE_SIZE * self._auto_gsd

    def _default_merge_separate(self, prompt: str) -> bool:














        token = (prompt or "").strip().lower()
        if not token:
            return True

        norm = token.replace("_", " ")
        from ...core.detection_policy import merge_policy
        continuous_tokens, discrete_tokens, continuous_categories = _merge_token_sets(
            merge_policy())

        if norm in discrete_tokens or token in discrete_tokens:
            return True
        if norm in continuous_tokens or token in continuous_tokens:
            return False
        try:
            from ...core.review_presets import live_catalog_categories
            for category in live_catalog_categories():
                for preset in category.get("presets") or []:
                    if str(preset.get("prompt", "")).strip().lower().replace("_", " ") != norm:
                        continue
                    if preset.get("weak"):
                        return False
                    cat = str(preset.get("category") or category.get("key") or "").lower()
                    return cat not in continuous_categories
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return True

    def _on_auto_review_refine_debounced(self) -> None:



        if self._auto_review is not None:
            self._review_tel_refined = True
        self._start_auto_reslice()

    def _start_auto_reslice(self) -> None:








        if not self._auto_review:
            return
        self._auto_finalize_gen += 1
        self._review_push_err_logged = False
        self._review_push_editable_logged = False
        self._auto_finalize_state = {
            "mode": "reslice",
            "phase": "filter",


            "filter_pending": self._reslice_pending_screen_last(),
            "total_filter": len(self._auto_objects),
            "visible": [],
            "visible_scores": [],
            "visible_ids": [],
            "visible_order": [],
            "params": self._widget_review_params(),
            "pixel_size": (self._auto_review or {}).get("pixel_size", 1.0),
            "gen": self._auto_finalize_gen,
        }




        self._review_geoms_stale = True



        self._set_review_busy(
            len(self._auto_objects)
            >= reslice_screen_first_min_objects(_RESLICE_SCREEN_FIRST_MIN_OBJECTS))
        self._step_auto_finalize_refine()

    def _set_review_busy(self, busy: bool) -> None:

        dock = self.dock_widget
        if dock is None:
            return
        try:
            if busy and not getattr(self, "_review_stop_wired", False):
                btn = getattr(dock, "auto_review_stop_btn", None)
                if btn is not None:
                    btn.clicked.connect(self._on_review_stop_clicked)
                    self._review_stop_wired = True
            dock.set_review_busy(bool(busy))
        except (RuntimeError, AttributeError):
            pass

    def _on_review_stop_clicked(self) -> None:








        if not self._auto_review:
            return
        self._auto_finalize_gen += 1
        self._auto_finalize_state = None
        stop = getattr(self, "_stop_review_refine_thread", None)
        if stop is not None:
            stop()
        self._set_review_busy(False)

    def _reslice_pending_screen_last(self) -> list:













        pending = list(enumerate(self._auto_objects))
        if len(pending) < reslice_screen_first_min_objects(_RESLICE_SCREEN_FIRST_MIN_OBJECTS):
            return pending
        boxes = self._reslice_object_boxes()
        try:
            canvas = self.iface.mapCanvas()
            extent = canvas.extent()
            canvas_crs = canvas.mapSettings().destinationCrs()
            run_crs = QgsCoordinateReferenceSystem(
                self._auto_crs_authid or "EPSG:4326")
            if not run_crs.isValid() or extent.isEmpty():
                return pending
            if canvas_crs.isValid() and canvas_crs != run_crs:
                extent = QgsCoordinateTransform(
                    canvas_crs, run_crs,
                    QgsProject.instance()).transformBoundingBox(extent)
            offscreen, onscreen = [], []
            for row in pending:
                box = boxes[row[0]]
                if box is not None and extent.intersects(box):
                    onscreen.append(row)
                else:
                    offscreen.append(row)
        except Exception:  # noqa: BLE001
            return pending
        if not onscreen or not offscreen:
            return pending
        return offscreen + onscreen

    def _reslice_object_boxes(self) -> list:








        objects = self._auto_objects or []
        key = (id(objects), len(objects))
        memo = getattr(self, "_reslice_box_memo", None)
        if memo is not None and memo[0] == key:
            return memo[1]
        boxes = [row[0].boundingBox() if row[0] is not None else None
                 for row in objects]
        self._reslice_box_memo = (key, boxes)
        return boxes

    def _on_auto_show_tiles_toggled(self, show: bool) -> None:




        if not show:
            self._clear_zone_tile_grid()
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            grid = self._compute_auto_grid(layer)
        except (RuntimeError, AttributeError):
            grid = None
        if grid is not None:


            self._show_zone_tile_grid(layer, grid, force=True)

    def _on_auto_review_confidence_preview(self, percent: int) -> None:











        if not self._auto_review:
            return
        conf = max(0.0, min(1.0, percent / 100.0))









        self._auto_confidence = conf
        preview = []
        pscores = []
        pids = []




        removed = self._review_removed_fids()


        params = self._widget_review_params()
        if self._auto_preview_geoms:





            stamp = ("prev", self._auto_preview_build_gen)
            for geom, score, area, det_idx, exempt in self._auto_preview_geoms:
                if not exempt:
                    if score < conf:
                        break
                    if not self._passes_size_filters(area, params):
                        continue
                if det_idx in removed:
                    continue
                preview.append(geom)
                pscores.append(score)
                pids.append(self._object_fid_for(det_idx))
        else:




            stamp = ("base", 0)
            for det_idx, (g, s, area) in enumerate(self._auto_objects):
                if det_idx in removed or g is None:
                    continue
                if not self._object_is_manual(det_idx):
                    if s < conf or not self._passes_size_filters(area, params):
                        continue
                preview.append(g)
                pscores.append(s)
                pids.append(self._object_fid_for(det_idx))
        self._push_review_geoms(preview, repair=False, scores=pscores, ids=pids,
                                stamp=stamp)




        self._review_geoms_stale = True

    def _on_auto_review_confidence_changed(self, percent: int) -> None:



        if not self._auto_review:
            return
        self._auto_confidence = max(0.0, min(1.0, percent / 100.0))
        self._start_auto_reslice()


        self._review_tel_conf_changed = True
        self._review_conf_moves = getattr(self, "_review_conf_moves", 0) + 1
        settle_ms = dial_in_range("tuning.review.confidence_settle_ms", 2000, 500, 10000)
        _debounce_timer(self, "_review_conf_timer", self.dock_widget, settle_ms,
                        self._emit_review_confidence_final)

    def _emit_review_confidence_final(self) -> None:
        if not self._auto_review:
            return
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_review_confidence_final(
                run_id=self._auto_run_id or "",
                final_pct=int(round((self._auto_confidence or 0.0) * 100)),
                visible_count=len(self._auto_review.get("geoms", [])),
                moves=getattr(self, "_review_conf_moves", 0),
            )
        except Exception:
            pass  # nosec B110

    def _refresh_auto_review_preview(self) -> None:







        if self._auto_review is None:
            return





        self._push_review_geoms(
            self._auto_review["geoms"], repair=False, update_extents=True,
            scores=self._auto_review.get("scores"),
            ids=self._auto_review.get("ids"),
            stamp=self._auto_review.get("stamp"))


        self._review_geoms_stale = False

    def _push_review_geoms(self, geoms: list, repair: bool = True,
                           scores: list | None = None,
                           ids: list | None = None,
                           stamp: tuple | None = None,
                           partial: bool = False,
                           update_extents: bool | None = None) -> None:



        from .review_layer_push import push_review_geoms

        push_review_geoms(self, geoms, repair=repair, scores=scores, ids=ids,
                          stamp=stamp, partial=partial,
                          update_extents=update_extents)

    def _on_auto_refine_changed_debounced(self) -> None:






        if not self.dock_widget:
            return
        try:
            from .auto_shape_overrides import shape_debounce_ms
            self.dock_widget._auto_review_debounce_timer.start(shape_debounce_ms())
        except (RuntimeError, AttributeError):
            pass

    def _update_review_header(self, visible: int) -> None:


        if not self.dock_widget:
            return



        self._set_review_busy(False)
        try:




            removed = self._review_removed_fids()


            n_objects = len(self._auto_objects)
            total = n_objects - sum(
                1 for det_idx in removed if 0 <= det_idx < n_objects)
            pct = int(round((self._auto_confidence or 0.0) * 100))



            bound = ("confidence" if visible or total <= 0
                     else self._review_zero_binding_gate())
            self.dock_widget.update_auto_review_count(
                visible, total, pct, bound=bound)
        except (RuntimeError, AttributeError):
            pass

    def _review_zero_binding_gate(self) -> str:







        conf = self._auto_confidence or 0.0
        params = self._widget_review_params()
        min_a = float(params.get("min_a") or 0.0)
        max_a = float(params.get("max_a") or 0.0)
        removed = self._review_removed_fids()
        below_min = 0
        above_max = 0
        for det_idx, (base, score, area) in enumerate(self._auto_objects):
            if det_idx in removed or base is None or base.isEmpty():
                continue
            if score < conf:
                continue
            if min_a > 0 and area < min_a:
                below_min += 1
            elif max_a > 0 and area > max_a:
                above_max += 1
        if not below_min and not above_max:
            return "confidence"
        return "min" if below_min >= above_max else "max"

    def _drop_preview_geom_cache(self) -> None:













        self._auto_preview_geoms = []
        self._auto_preview_build_state = None
        self._auto_preview_build_gen = (
            getattr(self, "_auto_preview_build_gen", 0) + 1)

    def _current_visible_review_count(self) -> int:

        review = self._auto_review or {}
        return len(review.get("geoms", []))

    def _full_found_review_count(self) -> int:






        params = dict(self._widget_review_params())
        params["conf"] = 0.0
        removed = self._review_removed_fids()
        n = 0
        for det_idx, (base, score, area) in enumerate(self._auto_objects):
            if det_idx in removed or base is None or base.isEmpty():
                continue



            if (self._object_is_manual(det_idx)
                    or self._passes_review_filters(score, area, params)):
                n += 1
        return n

    def _full_found_review_geoms(self) -> tuple[list, list, list]:









        review = self._auto_review or {}
        pixel_size = review.get("pixel_size", 1.0) or 1.0
        params = dict(self._widget_review_params())
        params["conf"] = 0.0



        from .auto_review_geometry import rescue_refine_budget
        return self._compute_visible_objects(
            params, pixel_size, with_scores=True,
            refine_budget_s=rescue_refine_budget(), with_ids=True)

    def _settle_review_geoms_for_export(self) -> None:




















        if not self._auto_review:
            return
        if not getattr(self, "_review_geoms_stale", False):
            return


        self._auto_finalize_gen += 1
        self._auto_finalize_state = None
        stop = getattr(self, "_stop_review_refine_thread", None)
        if stop is not None:
            stop()
        self._set_review_busy(False)
        params = self._widget_review_params()
        pixel_size = (self._auto_review or {}).get("pixel_size", 1.0) or 1.0
        geoms, scores, ids = self._compute_visible_objects(
            params, pixel_size, with_scores=True, with_ids=True)
        self._apply_auto_reslice_result(geoms, scores, ids)

    def _export_auto_review(self, include_hidden: bool = False,
                            autosave: bool = False
                            ) -> tuple[str | None, int] | None:























        review = self._auto_review
        if not review:
            return None
        if not autosave:




            self._settle_review_geoms_for_export()
            review = self._auto_review or review






        geoms = review["geoms"]
        scores = review.get("scores")







        conf_applied = float(getattr(self, "_auto_confidence", 0.0) or 0.0)
        if include_hidden:
            visible_n = sum(1 for g in geoms if g is not None and not g.isEmpty())
            full_geoms, full_scores, full_ids = self._full_found_review_geoms()
            merged = _union_review_sets(
                geoms, scores, review.get("ids"), full_geoms, full_scores, full_ids)
            if merged is not None:
                geoms, scores, extra = merged
                if extra:
                    conf_applied = 0.0
            elif len(full_geoms) > visible_n:

                geoms, scores = full_geoms, full_scores
                conf_applied = 0.0
        if scores is not None and len(scores) != len(geoms):
            scores = None
        refined, refined_scores = [], []
        for index, g in enumerate(geoms):
            if g is None or g.isEmpty():
                continue
            refined.append(QgsGeometry(g))
            refined_scores.append(scores[index] if scores else None)




        self._drop_preview_geom_cache()
        name = self._export_auto_detections(
            refined, review["crs"], review["source_layer_name"], review["prompt"],
            scores=refined_scores, confidence_applied=conf_applied)
        if name:






            run_id = self._auto_run_id or None
            exported_layer_id = self._auto_export_layer_id or ""

            def _drop_autosave_copy():
                self._pending_autosave_drop = None
                try:
                    from ...core.run_autosave import clear_pending
                    clear_pending(run_id, drop_table=True)
                except Exception:  # nosec B110
                    pass






                try:
                    layer = QgsProject.instance().mapLayer(exported_layer_id)
                    if layer is not None:
                        layer.triggerRepaint()
                except (RuntimeError, AttributeError):  # nosec B110
                    pass





            self._pending_autosave_drop = _drop_autosave_copy

            try:
                from ...core.qt_compat import safe_single_shot

                safe_single_shot(0, self.dock_widget, _drop_autosave_copy)
            except Exception:  # noqa: BLE001  # nosec B110
                _drop_autosave_copy()
        else:









            try:
                self._start_build_preview_cache(
                    (self._auto_review or {}).get("pixel_size", 1.0) or 1.0)
            except (RuntimeError, AttributeError):
                pass
            return None, len(refined)




        try:
            from .run_export_upload import queue_run_export_upload
            queue_run_export_upload(
                self, review, refined, refined_scores,
                export_path=("autosave" if autosave
                             else "exit_save" if include_hidden else "finish"),
                confidence_applied=conf_applied)
        except Exception:  # noqa: BLE001
            pass  # nosec B110



        try:
            from ...core.presets import segment_history
            segment_history.add_recent(
                review.get("prompt", ""),
                detections=len(refined),
                detail=(self._auto_run_ctx or {}).get("detail"),
            )
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        try:
            from ...core import telemetry_run_events, telemetry_session_events
            from .auto_client_profile import review_pass_profile
            found = len(self._auto_objects)
            pass_profile = dict(review_pass_profile(self))
            pass_profile["objects"] = len(refined)
            pass_profile.setdefault("on_pool", False)
            telemetry_run_events.track_auto_export_done(
                run_id=self._auto_run_id or "",
                exported_count=len(refined),
                visible_pct_of_found=int(round(len(refined) / found * 100)) if found else 0,
                final_confidence=int(round((self._auto_confidence or 0.0) * 100)),
                display_mode=self._auto_display_mode,
                refined_in_manual=getattr(self, "_auto_refined_in_manual", False),
                autosave=autosave,
                pass_profile=pass_profile,
            )
            if refined:
                telemetry_session_events.track_first_generation_milestone(mode="auto")
        except Exception:
            pass  # nosec B110
        self._auto_review = None



        self._set_review_busy(False)
        self._clear_free_zone_review_outline()
        self._auto_objects = []
        self._auto_object_fids = []
        self._drop_preview_geom_cache()
        self._reset_review_refine_cache()
        self._remove_auto_selection_layer()
        self._auto_manual_removed = set()
        self._auto_refined_in_manual = False
        self._clear_auto_raw_fragments()



        self._release_local_ai_install()
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_active(False)
            except (RuntimeError, AttributeError):
                pass



        written = int(getattr(self, "_auto_export_feature_count", 0) or 0)
        return name, (written or len(refined))

    @slot_guard(stage="export", user_message=tr(
        "Something went wrong saving your detections. Please try again."))
    def _on_auto_export_clicked(self) -> None:




        include_hidden = False



        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication
        try:
            self.dock_widget.set_auto_export_saving(True)
        except (RuntimeError, AttributeError):
            pass
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            exported = self._export_auto_review(include_hidden=include_hidden)
        finally:
            QApplication.restoreOverrideCursor()
            try:
                self.dock_widget.set_auto_export_saving(False)
            except (RuntimeError, AttributeError):
                pass
        if exported is None:
            return
        name, count = exported






        if not name:
            reason = getattr(self, "_auto_export_failure", "") or "file_refused"
            if reason == "nothing_visible":


                detail = tr("Nothing is visible to save. Lower Confidence, or "
                            "widen the size range, then try Export again.")
            elif reason == "no_shapes":
                detail = tr("None of the objects came out as a shape the file "
                            "could take. Turn the cleanup settings down and "
                            "try Export again.")
            else:
                detail = tr("The file may be open in QGIS or in another "
                            "program. Close it and try Export again.")
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                "{}\n\n{}".format(
                    tr("Could not save your detections to a file."), detail),
                error_code="export_failed",
            )
            return



        try:
            recap_layer_id = getattr(self, "_auto_export_layer_id", "")
        except Exception:  # nosec B110
            recap_layer_id = ""
        self._reset_auto_for_new_run()








        try:
            if self.dock_widget:




                self.dock_widget.set_auto_export_success(
                    count, name or "",
                    object_word=None,
                    layer_id=recap_layer_id,
                )
        except Exception:  # nosec B110
            pass

    def _reset_auto_for_new_run(self) -> None:




        self._restore_maptool_after_zone()


        self._auto_run_plan = None
        self._auto_attribute_filters = []
        self._cancel_task("_auto_run_plan_task")
        self._cancel_task("_auto_token_task")
        self._auto_zone = None
        self._auto_zone_polygon = None
        self._clear_auto_canvas()
        if self.dock_widget:
            try:


                self.dock_widget.reset_auto_to_start()
            except (RuntimeError, AttributeError):
                pass

    def _track_review_abandoned(self, exit_path: str) -> None:




        if self._auto_review is None or not self._auto_objects:
            return
        if getattr(self, "_review_abandon_tracked", False):
            return
        self._review_abandon_tracked = True
        try:
            from ...core import telemetry_run_events
            from .auto_client_profile import review_pass_profile
            instances = len((self._auto_review or {}).get("geoms", []))
            pass_profile = dict(review_pass_profile(self))
            pass_profile["objects"] = instances
            pass_profile.setdefault("on_pool", False)
            telemetry_run_events.track_review_abandoned(
                run_id=self._auto_run_id or "",
                instances_at_exit=instances,
                refined=bool(getattr(self, "_review_tel_refined", False)),
                confidence_changed=bool(
                    getattr(self, "_review_tel_conf_changed", False)),
                exit_path=exit_path,
                pass_profile=pass_profile,
            )
        except Exception:
            pass  # nosec B110

    def _discard_review_without_autosave(self, exit_path: str = "other") -> None:





        self._track_review_abandoned(exit_path)



        self._abandon_fix_session_for_discard()



        try:
            from ...core.run_autosave import clear_pending
            clear_pending(self._auto_run_id or None)
        except Exception:  # nosec B110
            pass
        self._auto_review = None
        self._clear_free_zone_review_outline()
        self._auto_objects = []
        self._auto_object_fids = []
        self._reset_review_refine_cache()
        self._remove_auto_selection_layer()
        self._auto_manual_removed = set()
        self._auto_refined_in_manual = False
        self._auto_finalize_gen += 1
        self._auto_finalize_state = None


        self._set_review_busy(False)
        self._drop_preview_geom_cache()
        self._clear_auto_raw_fragments()

        self._release_local_ai_install()



        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_active(False)
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_review_exit_clicked(self) -> None:








        try:
            if getattr(self, "_refine_add_mode_active", False):
                self._exit_ai_add_mode()
            if (getattr(self, "_refine_handoff_active", False)
                    or getattr(self, "_qgis_bridge_active", False)):
                self._fold_active_correct_session()
            self._disarm_shape_tool()
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        visible = self._current_visible_review_count()










        rescue = visible <= 0
        save_count = self._full_found_review_count() if rescue else visible
        if save_count > 0 and not self._auto_headless_run:
            hidden = save_count if rescue else 0
            if hidden > 0:
                label = tr(
                    "Save {save} detections ({hidden} currently hidden by "
                    "Confidence) to a layer before leaving?").format(
                        save=save_count, hidden=hidden)
            else:
                label = tr(
                    "Save {save} detections to a layer before leaving?").format(
                        save=save_count)
            clicked = ask_choice(
                self.iface.mainWindow(), tr("Keep your detections?"), label,
                [ChoiceButton("discard", tr("Discard && exit"), DISCARD),
                 ChoiceButton("cancel", tr("Cancel"), SECONDARY),
                 ChoiceButton("save", tr("Save && exit"), PRIMARY)],
                default="save", escape="cancel")
            if clicked == "save":



                saved = self._export_auto_review(include_hidden=rescue)
                if not saved or not saved[0]:




                    return
                self._reset_auto_for_new_run()
                self._signal_gpu_session_end("review_exit")
                return
            if clicked != "discard":
                return
        self._discard_review_without_autosave(
            exit_path="exit_button")
        self._reset_auto_for_new_run()
        self._signal_gpu_session_end("review_exit")

    def _on_auto_retry_clicked(self) -> bool:

















        total = len(self._auto_objects)
        discarded = len((self._auto_review or {}).get("geoms", [])) or total
        confirmed = True
        if total > 0 and not self._auto_headless_run:
            confirmed = ask_choice(
                self.iface.mainWindow(), tr("Discard these detections?"),
                tr(
                    "Your {total} detections will be discarded. You keep your zone, "
                    "object and settings. Running Detect again spends new cloud "
                    "detections."
                ).format(total=total),
                [ChoiceButton("cancel", tr("Cancel"), SECONDARY),
                 ChoiceButton("discard", tr("Discard && adjust"), DANGER)],
                default="discard", escape="cancel") == "discard"
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_auto_retry_clicked(
                run_id=self._auto_run_id or "",
                discarded_count=discarded,
                confirmed=confirmed,
            )
        except Exception:
            pass  # nosec B110
        if not confirmed:
            return False

        self._discard_review_without_autosave(exit_path="new_run")
        if not self.dock_widget:
            return True
        try:
            self.dock_widget.set_auto_review_active(False)


            self._set_zone_band_fill_visible(True)


            self.dock_widget.set_auto_zone_state("zone_set")



            self._restore_tile_grid_after_run()
            self._refresh_exemplar_chips()
            self.dock_widget.set_auto_status("idle")
        except (RuntimeError, AttributeError):
            pass
        return True

    def _on_auto_exit_clicked(self) -> None:








        try:
            from ...core import telemetry_run_events
            from_step = None
            try:
                from_step = int(self.dock_widget.auto_steps.currentIndex())
            except (RuntimeError, AttributeError):
                pass
            autosaved = len((self._auto_review or {}).get("geoms", []))
            telemetry_run_events.track_auto_exit_clicked(
                from_step=from_step if from_step is not None else -1,
                autosaved_count=autosaved,
            )
        except Exception:
            pass  # nosec B110

        self._restore_maptool_after_zone()
        self._discard_auto_review()


        self._set_review_busy(False)
        self._auto_zone = None
        self._auto_zone_polygon = None
        self._clear_auto_canvas()
        if self.dock_widget:
            try:
                self.dock_widget.reset_auto_to_start()
            except (RuntimeError, AttributeError):
                pass
        self._signal_gpu_session_end("auto_exit")
