


















from __future__ import annotations

from qgis.core import QgsGeometry, QgsPointXY, QgsRectangle

from ...core.i18n import tr
from ...core.review_corrections import JournalEntry
from ...core.shape_edits import (
    KIND_MERGE,
    KIND_REFINE,
    KIND_REMOVE,
    MIN_MERGE_PICKS,
    PickSet,
    align_ids,
    apply_merge,
    merge_plan,
    revert_shape_edit,
)
from ..canvas_palette import (
    SHAPE_EDIT_STROKE,
    SHAPE_EDIT_WIDTH,
    SHAPE_HOVER_STROKE,
    SHAPE_HOVER_WIDTH,
)




KIND_SELECT = "select"




_NAVIGATION_TOOL_CLASSES = ("QgsMapToolPan",)





_MERGE_SEAM_TOLERANCE_PX = 2.0



_UNDO_TOKEN_ATTR = "shape_undo_token"  # nosec B105


class AutoShapeEditMixin:






    _PICK_TOLERANCE_PX = 8





    def _init_auto_shape_edit_state(self) -> None:


        self._shape_maptool = None
        self._maptool_before_shape = None

        _init_overrides = getattr(self, "_init_shape_override_state", None)
        if _init_overrides is not None:
            _init_overrides()

        self._shape_edit_mode: str | None = None
        self._shape_picks = PickSet()
        self._shape_pick_bands: dict[int, object] = {}



        self._correct_selected_idx: int | None = None
        self._correct_selection_band = None
        self._shape_hover_idx: int | None = None
        self._shape_hover_band = None




        self._shape_review_params = None
        self._shape_hit_index = None


        self._shape_hit_built_from = None




        self._shape_hit_geoms: dict[int, object] = {}




        self._shape_edit_undo: dict[int, object] = {}
        self._shape_undo_token_seq = 0



        self._fold_manual_removed_undo: dict[int, set] = {}

    def _connect_auto_shape_edit_signals(self) -> None:


        dock = self.dock_widget
        for sig, slot in (
            (dock.auto_shape_edit_requested, self._on_auto_shape_edit_requested),
            (dock.auto_remove_requested, self._on_remove_requested),
            (dock.auto_shape_only_changed, self._on_shape_only_changed),
            (dock.auto_shape_only_reset_requested, self._on_shape_only_reset),
        ):
            try:
                sig.connect(slot)
            except (RuntimeError, AttributeError):
                pass





    def _shape_edit_allowed(self) -> bool:




        return (self._auto_review is not None and self._auto_worker is None)

    def _on_auto_shape_edit_requested(self, kind: str) -> None:




        kind = str(kind)
        if kind != KIND_MERGE:
            return
        if self._shape_edit_mode == kind:
            self._disarm_shape_tool()
            self._arm_correct_select()
            return
        if not self._shape_edit_allowed():
            return




        if getattr(self, "_refine_handoff_active", False):
            det_id = self._det_id_for_object_index(
                getattr(self, "_correct_selected_idx", None))
            self._fold_active_correct_session()
            self._correct_selected_idx = self._object_index_for_det_id(det_id)



        seed = self._correct_selected_idx
        self._arm_shape_tool(kind)
        if seed is not None and 0 <= seed < len(self._auto_objects):
            self._shape_picks.toggle(seed)
            self._refresh_pick_highlight()
            self._refresh_merge_pick_status()

    def _arm_shape_tool(self, kind: str) -> None:



        self._disarm_shape_tool()
        canvas = self.iface.mapCanvas()
        self._maptool_before_shape = canvas.mapTool()
        from ..pick_object_maptool import PickObjectMapTool
        tool = PickObjectMapTool(canvas)
        tool.point_clicked.connect(self._on_shape_point_clicked)
        tool.cursor_moved.connect(self._on_shape_cursor_moved)
        if kind == KIND_MERGE:
            tool.confirmed.connect(self._on_shape_merge_confirmed)
            tool.cancelled.connect(self._on_shape_draw_cancelled)
        else:
            tool.cancelled.connect(self._on_select_cancelled)



            tool.point_right_clicked.connect(self._on_shape_point_right_clicked)



        try:
            tool.tool_deactivated.connect(self._on_shape_tool_deactivated)
        except (RuntimeError, AttributeError):
            pass
        self._shape_maptool = tool
        self._shape_edit_mode = kind
        self._shape_picks.clear()
        self._build_shape_hit_index()


        if kind == KIND_MERGE:
            try:
                self.dock_widget.set_auto_shortcuts_enabled(False)
            except (RuntimeError, AttributeError):
                pass



        self._suspend_shape_deactivate = True
        try:
            canvas.setMapTool(tool)
        finally:
            self._suspend_shape_deactivate = False



        try:
            canvas.setFocus()
        except (RuntimeError, AttributeError):
            pass

        try:
            self.dock_widget.set_correct_armed(kind if kind == KIND_MERGE else None)
        except (RuntimeError, AttributeError):
            pass
        if kind == KIND_MERGE:
            self._refresh_merge_pick_status()

    def _disarm_shape_tool(self) -> None:











        self._set_correct_status("neutral", "")
        tool = self._shape_maptool
        if tool is None and self._shape_edit_mode is None:
            self._clear_correct_selection()
            return
        self._shape_maptool = None
        self._shape_edit_mode = None
        self._shape_picks.clear()
        self._clear_pick_highlight()
        self._clear_correct_selection()
        self._set_shape_hover(None)
        self._shape_review_params = None
        self._shape_hit_index = None
        self._shape_hit_geoms = {}
        if tool is not None:



            try:
                tool.suppress_deactivate_signal = True
            except (RuntimeError, AttributeError):
                pass
            try:
                canvas = self.iface.mapCanvas()
                if canvas.mapTool() is tool:
                    previous = self._maptool_before_shape
                    if previous is not None:
                        canvas.setMapTool(previous)
                    else:
                        canvas.unsetMapTool(tool)
            except (RuntimeError, AttributeError):
                pass
            try:
                tool.deleteLater()
            except (RuntimeError, AttributeError):
                pass
        self._maptool_before_shape = None
        try:
            self.dock_widget.set_auto_shortcuts_enabled(True)
            self.dock_widget.set_correct_armed(None)
        except (RuntimeError, AttributeError):
            pass

    def _on_shape_draw_cancelled(self) -> None:


        self._disarm_shape_tool()
        self._arm_correct_select()

    def _on_shape_tool_deactivated(self) -> None:




        if getattr(self, "_suspend_shape_deactivate", False):
            return
        if self._shape_maptool is None and self._shape_edit_mode is None:
            return
        self._shape_maptool = None
        self._shape_edit_mode = None
        self._maptool_before_shape = None
        self._shape_picks.clear()
        self._clear_pick_highlight()
        self._clear_correct_selection()
        self._set_shape_hover(None)
        self._shape_review_params = None
        self._shape_hit_index = None
        self._shape_hit_geoms = {}
        try:
            self.dock_widget.set_auto_shortcuts_enabled(True)
            self.dock_widget.set_correct_armed(None)
            self._set_correct_status("neutral", "")
        except (RuntimeError, AttributeError):
            pass



        try:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._rearm_correct_select_on_navigation)
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _rearm_correct_select_on_navigation(self) -> None:







        if self._shape_maptool is not None or self._shape_edit_mode is not None:
            return


        if getattr(self, "_qgis_bridge_active", False) or getattr(
                self, "_refine_handoff_active", False):
            return
        try:
            tool = self.iface.mapCanvas().mapTool()
            if tool is None:
                return
            meta = getattr(tool, "metaObject", None)
            name = meta().className() if meta is not None else type(tool).__name__
        except (RuntimeError, AttributeError, TypeError):
            return
        if name not in _NAVIGATION_TOOL_CLASSES:
            return
        self._arm_correct_select()





    def _arm_correct_select(self) -> None:




        if getattr(self, "_auto_review_step", 0) != 1:
            return
        if not self._shape_edit_allowed():
            return
        if self._shape_edit_mode is not None:
            return
        if getattr(self, "_refine_handoff_active", False):
            return
        self._arm_shape_tool(KIND_SELECT)

    def _on_select_cancelled(self) -> None:


        self._set_correct_selection(None)

    def _set_correct_selection(self, idx, *, enter_session: bool = False) -> None:














        if self._local_ai_install_pending():
            return
        if idx is not None and (
                idx < 0 or idx >= len(self._auto_objects) or idx in self._review_removed_fids()):
            idx = None
        self._correct_selected_idx = idx
        self._refresh_correct_selection_band()
        try:
            self.dock_widget.set_correct_selection(1 if idx is not None else 0)
        except (RuntimeError, AttributeError):
            pass


        try:
            self.dock_widget.set_merge_available(
                idx is not None and self._selected_has_mergeable_neighbor(idx))
        except (RuntimeError, AttributeError):
            pass


        _push = getattr(self, "_push_shape_only_state", None)
        if _push is not None:
            _push()
        if enter_session and idx is not None:
            self._enter_correct_method_session()

    def _enter_correct_method_session(self) -> None:





        idx = getattr(self, "_correct_selected_idx", None)
        if idx is None or self._auto_review is None:
            return



        if idx < 0 or idx >= len(self._auto_objects):
            return
        if idx in self._review_removed_fids():
            return
        if getattr(self, "_auto_review_step", 0) != 1:
            return
        if getattr(self, "_headless", False):
            return
        if getattr(self, "_refine_handoff_active", False) or getattr(
                self, "_qgis_bridge_active", False):
            return
        method = getattr(self, "_correct_method", "ai")





        self._begin_correct_focus(self._det_id_for_object_index(idx))
        try:
            if method == "manual":


                self.enter_qgis_edit_bridge()
            else:

                self._on_reshape_ai_requested()
        finally:
            if not (getattr(self, "_refine_handoff_active", False)
                    or getattr(self, "_qgis_bridge_active", False)):
                self._end_correct_focus()

    def _merge_seam_tolerance(self) -> float:





        try:
            px = float((self._auto_review or {}).get("pixel_size", 0.0) or 0.0)
        except (TypeError, ValueError, AttributeError):
            return 0.0
        from ...core.server_dials import dial_in_range
        tolerance_px = dial_in_range(
            "tuning.review.merge_seam_tolerance_px", _MERGE_SEAM_TOLERANCE_PX, 0.0, 10.0)
        return max(0.0, px) * tolerance_px

    def _selected_has_mergeable_neighbor(self, det_idx) -> bool:




        if det_idx is None or det_idx < 0 or det_idx >= len(self._auto_objects):
            return False
        sel = self._shape_hit_geoms.get(det_idx)
        if sel is None:
            sel = self._auto_objects[det_idx][0]
        if sel is None or sel.isEmpty():
            return False








        tol = self._merge_seam_tolerance()
        bbox = sel.boundingBox()
        rect = QgsRectangle(bbox.xMinimum() - tol, bbox.yMinimum() - tol,
                            bbox.xMaximum() + tol, bbox.yMaximum() + tol)
        removed = self._review_removed_fids()
        for cand in self._shape_hit_candidates(rect):
            if cand == det_idx or cand in removed or cand >= len(self._auto_objects):
                continue
            geom = self._shape_hit_geoms.get(cand)
            if geom is None:
                geom = self._auto_objects[cand][0]
            if geom is None or geom.isEmpty():
                continue
            try:
                if geom.intersects(sel) or (tol > 0 and geom.distance(sel) <= tol):
                    return True
            except (RuntimeError, TypeError, ValueError):
                continue
        return False

    def _clear_correct_selection(self) -> None:

        self._correct_selected_idx = None
        band = self._correct_selection_band
        self._correct_selection_band = None
        if band is not None:
            self._drop_geom_bands([band])
        try:
            self.dock_widget.set_correct_selection(0)
        except (RuntimeError, AttributeError):
            pass

    def _refresh_correct_selection_after_reslice(self) -> None:









        if self._auto_review is None:
            return
        if getattr(self, "_shape_edit_mode", None) is not None:
            self._build_shape_hit_index()
        else:



            self._shape_hit_geoms = {}
            self._shape_hit_index = None
            self._shape_hit_built_from = None
        if getattr(self, "_correct_selected_idx", None) is None:
            return
        self._refresh_correct_selection_band()


        try:
            self.dock_widget.set_merge_available(
                self._selected_has_mergeable_neighbor(self._correct_selected_idx))
        except (RuntimeError, AttributeError):
            pass
        _push = getattr(self, "_push_shape_only_state", None)
        if _push is not None:
            _push()

    def _drawn_geom_for_index(self, det_idx: int):









        geom = (getattr(self, "_shape_hit_geoms", None) or {}).get(det_idx)
        if geom is not None:
            return geom
        review = self._auto_review or {}
        geoms = review.get("geoms") or []
        ids = review.get("ids")
        if isinstance(ids, list) and len(ids) == len(geoms):
            try:
                det_id = self._object_fid_for(det_idx)
                for visible, vis_id in zip(geoms, ids):
                    if vis_id == det_id:
                        return visible
            except (RuntimeError, AttributeError, TypeError):
                pass
        try:
            return self._auto_objects[det_idx][0]
        except (IndexError, TypeError):
            return None

    def _refresh_correct_selection_band(self) -> None:


        band = self._correct_selection_band
        self._correct_selection_band = None
        if band is not None:
            self._drop_geom_bands([band])
        idx = self._correct_selected_idx
        if idx is None or idx >= len(self._auto_objects):
            return
        geom = self._drawn_geom_for_index(idx)
        if geom is None or geom.isEmpty():
            return
        from qgis.PyQt.QtGui import QColor

        self._correct_selection_band = self._make_geom_band(
            geom, QColor(255, 255, 0, 255), 3)

    def _on_remove_requested(self) -> None:








        if getattr(self, "_refine_handoff_active", False):
            det_id = self._det_id_for_object_index(
                getattr(self, "_correct_selected_idx", None))
            self._fold_active_correct_session()
            self._correct_selected_idx = self._object_index_for_det_id(det_id)
        self._remove_detection_index(self._correct_selected_idx)

    def _remove_detection_index(self, idx) -> None:





        if idx is None or self._auto_review is None:
            return
        if idx < 0 or idx >= len(self._auto_objects):
            return
        if idx in self._review_removed_fids():
            return
        self._auto_correction_removed.add(int(idx))
        self._record_remove_entry(idx)
        if self._correct_selected_idx == idx:




            self._clear_correct_selection()
        if getattr(self, "_shape_hover_idx", None) == idx:


            self._set_shape_hover(None)
        self._after_shape_edit(changed=())
        self._track_shape_edit(KIND_REMOVE, "removed", 1)

    def _record_remove_entry(self, idx: int) -> None:


        self._push_correct_entry(JournalEntry(kind=KIND_REMOVE, fids=(int(idx),)))





    def _on_shape_point_clicked(self, point) -> None:





        if self._auto_review is None or self._auto_review_export_busy():
            return
        if self._shape_edit_mode == KIND_SELECT:


            det_idx = self._hit_object_at(point)



            if self._correct_focus_blocks_det_id(
                    self._det_id_for_object_index(det_idx)):
                return




            self._set_correct_selection(det_idx, enter_session=True)
            return
        if self._shape_edit_mode != KIND_MERGE:
            return
        det_idx = self._hit_object_at(point)
        if det_idx is None:
            self._set_correct_status("neutral", tr(
                "No detection under that click."))
            return
        self._shape_picks.toggle(det_idx)
        self._refresh_pick_highlight()
        self._refresh_merge_pick_status()

    def _on_shape_point_right_clicked(self, point) -> None:









        if self._auto_review is None or self._shape_edit_mode != KIND_SELECT:
            return


        if getattr(self, "_refine_handoff_active", False) or getattr(
                self, "_qgis_bridge_active", False):
            return
        det_idx = self._hit_object_at(point)
        if det_idx is None:
            return
        self._remove_detection_index(det_idx)

    def _on_shape_cursor_moved(self, point) -> None:







        if self._shape_edit_mode not in (KIND_SELECT, KIND_MERGE):
            return
        idx = self._hit_object_at(point)
        self._set_shape_hover(idx)
        if self._shape_edit_mode == KIND_SELECT:


            self._schedule_correct_hover_warm(idx)

    def _set_shape_hover(self, idx) -> None:

        if idx == self._shape_hover_idx:
            return
        self._shape_hover_idx = idx
        band = self._shape_hover_band
        self._shape_hover_band = None
        if band is not None:
            self._drop_geom_bands([band])
        tool = self._shape_maptool
        try:
            from qgis.PyQt.QtCore import Qt
            from qgis.PyQt.QtGui import QCursor
            cursor = (Qt.CursorShape.PointingHandCursor if idx is not None
                      else Qt.CursorShape.OpenHandCursor)
            if tool is not None:
                tool.setCursor(QCursor(cursor))
        except (RuntimeError, AttributeError):
            pass
        if idx is None:
            return
        geom = self._shape_hit_geoms.get(idx)
        if geom is None and 0 <= idx < len(self._auto_objects):
            geom = self._auto_objects[idx][0]
        if geom is not None and not geom.isEmpty():
            self._shape_hover_band = self._make_geom_band(
                geom, SHAPE_HOVER_STROKE, SHAPE_HOVER_WIDTH)

    def _refresh_merge_pick_status(self) -> None:


        picked = len(self._shape_picks)
        action = ""
        if picked == 0:


            text = tr("Click each piece of the object you want to merge.")
        elif picked < MIN_MERGE_PICKS:


            text = tr("Now click the other pieces of this object.")
        else:


            text = tr("{n} shapes picked. Press Enter to merge.").format(
                n=picked)
            action = tr("Merge {n} shapes · Free").format(n=picked)
        self._set_correct_status("armed", text, action=action)

    def _on_shape_merge_confirmed(self) -> None:


        if self._shape_edit_mode != KIND_MERGE:
            return
        if len(self._shape_picks) >= MIN_MERGE_PICKS:
            self._commit_merge_picks()

    def _commit_merge_picks(self) -> None:


        if self._shape_edit_mode != KIND_MERGE or self._auto_review is None:
            return
        picks = [(idx, float(self._auto_objects[idx][1]))
                 for idx in self._shape_picks.indices
                 if 0 <= idx < len(self._auto_objects)]
        plan = merge_plan(picks, frozenset(self._review_removed_fids()))
        if plan is None:
            self._set_correct_status("neutral", tr(
                "Pick at least two shapes to merge them."))
            return
        from ...core.geometry_ops import merge_geometries




        geoms = [self._auto_objects[idx][0]
                 for idx in (plan.target, *plan.absorbed)
                 if self._auto_objects[idx][0] is not None]
        merged = merge_geometries(geoms)
        if merged is None or merged.isEmpty():
            self._set_correct_status("warning", tr(
                "Those shapes could not be merged. Nothing was changed."))
            return





        from ...core.geometry_ops import bridge_seam_gap, polygon_part_count
        if polygon_part_count(merged) > 1:
            bridged = bridge_seam_gap(merged, self._merge_seam_tolerance())
            if bridged is None:
                self._set_correct_status("warning", tr(
                    "Those shapes could not be merged. Nothing was changed."))
                return
            merged = bridged
        joined = len(plan.absorbed) + 1
        edit = apply_merge(self._auto_objects, plan,
                           self._object_row(merged, plan.score))
        self._auto_correction_removed.update(plan.absorbed)
        self._record_shape_edit(edit, fids=plan.absorbed)
        self._disarm_shape_tool()


        self._after_shape_edit(changed=(plan.target,))
        self._track_shape_edit(KIND_MERGE, "merged", joined)
        self._arm_correct_select()






        self._correct_reenter_det_id = self._det_id_for_object_index(plan.target)
        self._correct_reenter_looks = 0
        self._reenter_correct_session_after_fold()





    def _shape_undo_token_for(self, entry) -> int:





        token = int(getattr(self, "_shape_undo_token_seq", 0)) + 1
        self._shape_undo_token_seq = token
        try:
            setattr(entry, _UNDO_TOKEN_ATTR, token)
        except (AttributeError, TypeError):
            pass
        return token

    def _shape_undo_token_of(self, entry):


        return getattr(entry, _UNDO_TOKEN_ATTR, None)

    def _record_shape_edit(self, edit, fids: tuple) -> None:



        entry = JournalEntry(kind=edit.kind, fids=tuple(fids))
        self._push_correct_entry(entry)
        self._shape_edit_undo[self._shape_undo_token_for(entry)] = edit
        self._trim_shape_edit_undo()

    def _trim_shape_edit_undo(self, keep: int | None = None) -> None:


        if keep is None:
            try:
                from ...core.server_dials import dial_in_range
                keep = dial_in_range("tuning.review.undo_journal_keep", 30, 5, 200)
            except Exception:  # noqa: BLE001
                keep = 30
        while len(self._shape_edit_undo) > keep:
            oldest = next(iter(self._shape_edit_undo))
            self._shape_edit_undo.pop(oldest, None)
            self._fold_manual_removed_undo.pop(oldest, None)

    def _record_fold_edit(self, edit, fids: tuple,
                          manual_removed_before=None):










        if not (edit.restored or edit.appended or edit.unremoved or edit.exempted or manual_removed_before is not None):
            return None
        entry = JournalEntry(kind=KIND_REFINE, fids=tuple(fids))
        self._push_correct_entry(entry)
        token = self._shape_undo_token_for(entry)
        self._shape_edit_undo[token] = edit
        if manual_removed_before is not None:
            self._fold_manual_removed_undo[token] = set(manual_removed_before)
        self._trim_shape_edit_undo()
        self._refresh_correction_summary()
        return entry

    def _revert_shape_edit_entry(self, entry,
                                 defer_refresh: bool = False) -> None:









        token = self._shape_undo_token_of(entry)
        edit = None if token is None else self._shape_edit_undo.pop(token, None)
        if edit is None:
            if token is not None:
                self._fold_manual_removed_undo.pop(token, None)
            return
        ids = self._shape_edit_ids()
        restored_indices = tuple(idx for idx, _row in edit.restored)
        unremoved = revert_shape_edit(self._auto_objects, ids, edit)
        self._auto_object_fids = ids
        self._auto_correction_removed.difference_update(unremoved)



        exempted = getattr(edit, "exempted", ()) or ()
        if exempted:
            manual_ids = getattr(self, "_auto_manual_object_ids", None)
            if manual_ids is not None:
                manual_ids.difference_update(int(i) for i in exempted)


        snap = self._fold_manual_removed_undo.pop(token, None)
        if snap is not None:
            self._auto_manual_removed = set(snap)




        overrides = getattr(self, "_auto_shape_overrides", None)
        if overrides:
            ceiling = len(self._auto_objects)
            for stale in [key for key in overrides if key >= ceiling]:
                overrides.pop(stale, None)



        if not defer_refresh:
            self._after_shape_edit(changed=restored_indices)

    def _after_shape_edit(self, changed=None) -> None:











        if changed is None:
            self._reset_review_refine_cache()
        else:
            self._invalidate_review_refine(changed)
        pixel_size = (self._auto_review or {}).get("pixel_size", 1.0)
        self._start_build_preview_cache(pixel_size)
        self._start_auto_reslice()
        self._refresh_correction_summary()

    def _shape_edit_ids(self) -> list:

        ids = list(getattr(self, "_auto_object_fids", None) or [])
        align_ids(self._auto_objects, ids)
        return ids

    def _object_row(self, geom, score: float, measurer=None) -> tuple:





        if measurer is None:
            measurer = self._make_auto_area_measurer()
        return (geom, float(score), self._object_area_m2(geom, measurer))

    def _track_shape_edit(self, kind: str, outcome: str, objects: int) -> None:
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_review_correct_box(
                run_id=self._auto_run_id or "", label=1, outcome=outcome,
                objects=int(objects), gesture=kind)
        except Exception:
            pass  # nosec B110





    def _visible_object_indices(self) -> list[int]:




        params = self._shape_review_params or self._widget_review_params()
        removed = self._review_removed_fids()
        out: list[int] = []
        for det_idx, (geom, score, area) in enumerate(self._auto_objects):
            if det_idx in removed or geom is None or geom.isEmpty():
                continue
            if self._object_is_manual(det_idx) or self._passes_review_filters(score, area, params):
                out.append(det_idx)
        return out

    def _drawn_set_published(self):







        geoms = (self._auto_review or {}).get("geoms")
        return geoms if isinstance(geoms, list) else None

    def _build_shape_hit_index(self) -> None:











        params = self._widget_review_params()
        published = self._drawn_set_published()
        if (published is not None and self._shape_hit_index is not None
                and published is getattr(self, "_shape_hit_built_from", None)
                and params == self._shape_review_params):
            return
        self._shape_review_params = params
        self._shape_hit_index = None
        self._shape_hit_built_from = None
        self._shape_hit_geoms = {}
        try:
            from qgis.core import QgsSpatialIndex
            index = QgsSpatialIndex()
            review = self._auto_review or {}
            review_geoms = review.get("geoms") or []
            review_ids = review.get("ids")
            n_objects = len(self._auto_objects)
            fids = getattr(self, "_auto_object_fids", None)
            if fids is not None and len(fids) >= n_objects:



                by_display_id = {fids[i]: i for i in range(n_objects)}
            else:
                by_display_id = {
                    self._object_fid_for(det_idx): det_idx
                    for det_idx in range(n_objects)
                }
            visible = []
            built_from = published
            if isinstance(review_ids, list) and len(review_ids) == len(review_geoms):
                for geom, det_id in zip(review_geoms, review_ids):
                    det_idx = by_display_id.get(det_id)
                    if det_idx is not None:
                        visible.append((det_idx, geom))
            else:


                built_from = None
                visible = [(det_idx, self._auto_objects[det_idx][0])
                           for det_idx in self._visible_object_indices()]
            for det_idx, geom in visible:
                if geom is None or geom.isEmpty():
                    continue




                held = QgsGeometry(geom)
                index.addFeature(int(det_idx), held.boundingBox())
                self._shape_hit_geoms[det_idx] = held
            self._shape_hit_index = index
            self._shape_hit_built_from = built_from
        except (RuntimeError, AttributeError, TypeError, ImportError):
            self._shape_hit_index = None
            self._shape_hit_built_from = None
            self._shape_hit_geoms = {}

    def _shape_hit_candidates(self, rect) -> list[int]:


        index = self._shape_hit_index
        if index is not None:
            try:
                return [int(i) for i in index.intersects(rect)]
            except (RuntimeError, AttributeError, TypeError):
                pass
        return self._visible_object_indices()

    def _hit_object_at(self, point) -> int | None:









        tol = 0.0
        try:
            from ...core.server_dials import dial_in_range
            pick_tolerance_px = dial_in_range(
                "tuning.review.shape_pick_tolerance_px", self._PICK_TOLERANCE_PX, 2, 20)
        except Exception:  # noqa: BLE001
            pick_tolerance_px = self._PICK_TOLERANCE_PX
        try:
            mupp = self.iface.mapCanvas().mapSettings().mapUnitsPerPixel()
            tol_canvas = mupp * pick_tolerance_px
            off = QgsPointXY(point.x() + tol_canvas, point.y())
            run_pts = self._points_in_run_crs([point, off])
            if len(run_pts) == 2:
                tol = float(run_pts[0].distance(run_pts[1]))
        except (RuntimeError, AttributeError, TypeError):
            run_pts = self._points_in_run_crs([point])
        if not run_pts:
            return None
        pt = run_pts[0]
        probe = QgsGeometry.fromPointXY(pt)
        rect = QgsRectangle(pt.x() - tol, pt.y() - tol, pt.x() + tol, pt.y() + tol)
        params = self._shape_review_params or self._widget_review_params()
        removed = self._review_removed_fids()
        best = None
        best_area = 0.0
        near = None
        near_dist = tol if tol > 0 else -1.0
        for det_idx in self._shape_hit_candidates(rect):
            if det_idx in removed or det_idx >= len(self._auto_objects):
                continue
            base_geom, score, area_m2 = self._auto_objects[det_idx]
            geom = self._shape_hit_geoms.get(det_idx, base_geom)
            if geom is None or geom.isEmpty():
                continue




            if (det_idx not in self._shape_hit_geoms
                    and not self._object_is_manual(det_idx)
                    and not self._passes_review_filters(score, area_m2, params)):
                continue
            try:
                if geom.contains(probe):
                    area = float(geom.area())
                    if best is None or area < best_area:
                        best, best_area = det_idx, area
                    continue


                if best is None and tol > 0:
                    dist = float(geom.distance(probe))
                    if dist <= near_dist:
                        near, near_dist = det_idx, dist
            except (RuntimeError, TypeError, ValueError):
                continue
        return best if best is not None else near

    def _points_in_run_crs(self, points) -> list:


        pts = [QgsPointXY(p) for p in points]
        try:
            run_authid = (self._auto_run_ctx or {}).get(
                "crs_authid") or self._auto_crs_authid
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            if run_authid and canvas_crs.authid() != run_authid:
                from qgis.core import (
                    QgsCoordinateReferenceSystem,
                    QgsCoordinateTransform,
                    QgsProject,
                )
                xform = QgsCoordinateTransform(
                    canvas_crs, QgsCoordinateReferenceSystem(run_authid),
                    QgsProject.instance())
                pts = [QgsPointXY(xform.transform(p)) for p in pts]
        except (RuntimeError, AttributeError, TypeError):
            return pts
        return pts

    def _refresh_pick_highlight(self) -> None:



        self._clear_pick_highlight()
        for det_idx in self._shape_picks.indices:
            try:
                geom = self._shape_hit_geoms.get(det_idx, self._auto_objects[det_idx][0])
            except (IndexError, TypeError):
                continue
            if geom is None or geom.isEmpty():
                continue
            band = self._make_geom_band(geom, SHAPE_EDIT_STROKE, SHAPE_EDIT_WIDTH)
            if band is not None:
                self._shape_pick_bands[det_idx] = band

    def _clear_pick_highlight(self) -> None:
        bands = list(self._shape_pick_bands.values())
        self._shape_pick_bands = {}
        if bands:
            self._drop_geom_bands(bands)

    def _make_geom_band(self, geom, color, width: int):



        try:
            from qgis.core import QgsCoordinateReferenceSystem
            from qgis.gui import QgsRubberBand

            from ...core.qt_compat import PolygonGeometry
            band = QgsRubberBand(self.iface.mapCanvas(), PolygonGeometry)





            authid = getattr(self, "_auto_crs_authid", None)
            band_crs = QgsCoordinateReferenceSystem(authid) if authid else None
            if band_crs is not None and not band_crs.isValid():
                band_crs = None
            band.setToGeometry(QgsGeometry(geom), band_crs)
            band.setStrokeColor(color)
            band.setWidth(width)
            fill = type(color)(color)
            fill.setAlpha(0)
            band.setFillColor(fill)
            return band
        except (RuntimeError, AttributeError, TypeError):
            return None

    def _drop_geom_bands(self, bands: list) -> None:
        from ...core.qt_compat import PolygonGeometry
        for band in bands:
            try:
                band.reset(PolygonGeometry)
                self.iface.mapCanvas().scene().removeItem(band)
            except (RuntimeError, AttributeError):
                pass
