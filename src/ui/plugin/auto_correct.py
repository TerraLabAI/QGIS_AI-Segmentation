








from __future__ import annotations

from qgis.core import QgsGeometry, QgsProject
from qgis.PyQt.QtCore import QTimer

from ...core.interaction_dials import (
    confirm_reset_ms,
    correct_fold_look_ms,
    correct_fold_max_looks,
)
from ...core.review_corrections import CorrectionJournal, JournalEntry, RetryLinkState
from ...core.shape_edits import KIND_MERGE, KIND_REFINE, KIND_REMOVE, KIND_SPLIT, ShapeEdit
from .auto_shape_edit import KIND_SELECT


_RETRY_CONFIRM_RESET_MS = 6000




_FOLD_RESLICE_LOOK_MS = 50
_FOLD_RESLICE_MAX_LOOKS = 120


class AutoCorrectMixin:






    def _init_auto_correct_state(self) -> None:

        self._auto_correction_removed: set[int] = set()




        self._auto_manual_object_ids: set[int] = set()
        self._auto_correct_journal = CorrectionJournal()
        self._auto_review_step = 0
        self._auto_review_steps_seen: set[int] = set()
        self._auto_retry_guard = RetryLinkState()





        self._correct_method = "manual"



        self._refine_add_mode_active = False
        self._ai_add_install_pending = False


        self._correct_reenter_det_id = None
        self._correct_reenter_looks = 0


        self._init_auto_shape_edit_state()



        self._init_correct_focus_state()

    def _connect_auto_correct_signals(self) -> None:

        dock = self.dock_widget
        dock.auto_correction_undo_requested.connect(
            self._on_auto_correction_undo_requested)
        dock.auto_correction_clear_requested.connect(
            self._on_auto_correction_clear_requested)
        dock.auto_review_step_requested.connect(self._on_auto_review_step_requested)
        dock.auto_correct_status_action_requested.connect(
            self._on_correct_status_action_requested)
        self._connect_auto_shape_edit_signals()

    def _reset_auto_corrections(self) -> None:


        self._disarm_shape_tool()
        self._init_auto_correct_state()
        dock = self.dock_widget
        if dock is not None:
            try:
                dock.reset_review_steps()
            except (RuntimeError, AttributeError):
                pass




        if dock is not None:
            try:
                self._correct_method = dock.get_correct_method()
            except (RuntimeError, AttributeError):
                pass

        self._apply_correct_class_label()

    def _apply_correct_class_label(self) -> None:




        dock = self.dock_widget
        if dock is None:
            return
        prompt = str((self._auto_review or {}).get("prompt") or "").strip()
        label = (prompt[:1].upper() + prompt[1:]) if prompt else ""
        try:
            dock.set_correct_class_label(label)
        except (RuntimeError, AttributeError):
            pass

    def _on_correct_method_changed(self, method: str) -> None:





        method = "manual" if str(method) == "manual" else "ai"




        if self._local_ai_install_pending():
            try:
                self.dock_widget.set_correct_method(self._correct_method)
            except (RuntimeError, AttributeError):
                pass
            return
        self._correct_method = method
        if method == "ai":
            self._warm_local_ai_for_correct()


        if getattr(self, "_refine_add_mode_active", False):
            self._exit_ai_add_mode()


        target_det_id = self._active_session_target_det_id()
        folded = self._fold_active_correct_session()
        if not folded or target_det_id is None:
            return



        self._correct_reenter_det_id = target_det_id
        self._correct_reenter_looks = 0
        self._reenter_correct_session_after_fold()

    def _reenter_correct_session_after_fold(self) -> None:









        det_id = self._correct_reenter_det_id
        if det_id is None:
            return
        if self._auto_review is None or self._auto_review_step != 1:
            self._correct_reenter_det_id = None
            return





        if (getattr(self, "_refine_handoff_active", False)
                or getattr(self, "_qgis_bridge_active", False)
                or self._shape_edit_mode not in (None, KIND_SELECT)):
            self._correct_reenter_det_id = None
            return
        if (getattr(self, "_auto_finalize_state", None) is not None
                and self._correct_reenter_looks
                < correct_fold_max_looks(_FOLD_RESLICE_MAX_LOOKS)):
            self._correct_reenter_looks += 1
            QTimer.singleShot(
                correct_fold_look_ms(_FOLD_RESLICE_LOOK_MS),
                self._reenter_correct_session_after_fold)
            return
        self._correct_reenter_det_id = None
        idx = self._object_index_for_det_id(det_id)
        if idx is None:
            return
        self._set_correct_selection(idx, enter_session=True)

    def _active_session_target_det_id(self):


        if getattr(self, "_qgis_bridge_active", False):
            return getattr(self, "_qgis_bridge_target_det_id", None)
        if getattr(self, "_refine_handoff_active", False):
            idx = getattr(self, "_correct_selected_idx", None)
            return self._det_id_for_object_index(idx)
        return None

    def _det_id_for_object_index(self, idx):


        if idx is None:
            return None
        ids = getattr(self, "_auto_object_fids", None) or []
        if 0 <= idx < len(ids):
            return ids[idx]
        return None

    def _object_index_for_det_id(self, det_id):


        if det_id is None:
            return None
        ids = getattr(self, "_auto_object_fids", None) or []
        for i, fid in enumerate(ids):
            if fid == det_id:
                if i in self._review_removed_fids():
                    return None
                return i
        return None

    def _fold_active_correct_session(self) -> bool:



        if getattr(self, "_qgis_bridge_active", False):
            self.finish_qgis_edit_bridge()
            return True
        if getattr(self, "_refine_handoff_active", False):
            self._on_reshape_done()
            return True
        return False





    def _on_auto_review_step_requested(self, step: int) -> None:



        if self._auto_review is None:
            return



        if self._local_ai_install_pending():
            return
        step = max(0, min(2, int(step)))
        if step != 1:



            if getattr(self, "_refine_add_mode_active", False):
                self._exit_ai_add_mode()
            if (getattr(self, "_refine_handoff_active", False) or getattr(self, "_qgis_bridge_active", False)):
                self._fold_active_correct_session()
            self._disarm_shape_tool()
        self._auto_review_step = step
        self._auto_retry_guard.reset()
        dock = self.dock_widget
        if dock is not None:
            try:
                dock.set_retry_confirm_pending(False)
                dock.set_auto_review_step(step)
            except (RuntimeError, AttributeError):
                pass
        if step == 1:


            self._arm_correct_select()
            self._warm_local_ai_for_correct()
            if self._correct_ai_route_is_remote():




                self._maybe_warmup_auto()
        if step not in self._auto_review_steps_seen:
            self._auto_review_steps_seen.add(step)
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_review_step(
                    run_id=self._auto_run_id or "", step=step)
            except Exception:
                pass  # nosec B110

    def _zero_review_crs(self, layer):





        from qgis.core import QgsCoordinateReferenceSystem

        authid = getattr(self, "_auto_crs_authid", None)
        if authid:
            crs = QgsCoordinateReferenceSystem(authid)
            if crs.isValid():
                return crs
        try:
            return layer.crs() if layer is not None else None
        except (RuntimeError, AttributeError):
            return None

    def _enter_zero_detection_review(self, tiles_succeeded: int) -> None:




        ctx = self._auto_run_ctx or {}
        layer = QgsProject.instance().mapLayer(ctx.get("layer_id") or "")
        prompt_text = ""
        try:
            prompt_text = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            pass
        self._auto_review = {
            "geoms": [],
            "scores": [],
            "ids": [],



            "crs": self._zero_review_crs(layer),
            "source_layer_name": layer.name() if layer is not None else "",
            "prompt": prompt_text,
            "pixel_size": self._auto_refine_pixel_size(),
            "stamp": None,
        }
        self._reset_auto_corrections()
        dock = self.dock_widget
        if dock is not None:
            try:
                dock.set_auto_review_active(True)
                dock.set_zero_detection_entry(True)






                dock.set_auto_review_score_useful(True)
            except (RuntimeError, AttributeError):
                pass


        self._auto_review_step = 1
        self._auto_review_steps_seen.add(1)
        self._arm_correct_select()
        self._warm_local_ai_for_correct()
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_review_step(run_id=self._auto_run_id or "", step=1)
        except Exception:
            pass  # nosec B110

    def _disarm_after_handoff(self) -> None:





        self._disarm_shape_tool()





    def _on_correct_status_action_requested(self) -> None:


        if self._shape_edit_mode == KIND_MERGE:
            self._commit_merge_picks()

    def _set_correct_status(self, kind: str, text: str,
                            action: str = "") -> None:
        try:
            self.dock_widget.set_correct_status(
                kind, text, action_text=action)
        except (RuntimeError, AttributeError, TypeError):
            pass





    def _push_correct_entry(self, entry: JournalEntry) -> None:
        self._auto_correct_journal.push(entry)

    def _ai_session_has_undo(self) -> bool:



        if not getattr(self, "_refine_handoff_active", False):
            return False
        try:
            if any(getattr(self, "prompts", None).point_count):
                return True
        except (AttributeError, TypeError):
            pass
        for name in ("_mask_state_history", "_refine_geom_history",
                     "_deleted_objects_stack", "_frozen_sessions"):
            if getattr(self, name, None):
                return True
        return False

    def _on_auto_correction_undo_requested(self) -> None:



        if getattr(self, "_qgis_bridge_active", False):
            self.undo_qgis_bridge_edit()
            return


        if self._ai_session_has_undo():
            self._on_undo()
            return




        if getattr(self, "_refine_handoff_active", False):
            return
        entry = self._auto_correct_journal.undo()
        if entry is None:
            return
        self._undo_entry(entry)
        self._refresh_correction_summary()
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_review_correct_undo(
                run_id=self._auto_run_id or "", kind=entry.kind)
        except Exception:
            pass  # nosec B110

    def _on_auto_correction_clear_requested(self) -> None:
        entries = self._auto_correct_journal.clear()
        if not entries:
            return




        for entry in entries:
            self._undo_entry(entry, defer_refresh=True)
        self._after_shape_edit()
        self._refresh_correction_summary()
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_review_correct_undo(
                run_id=self._auto_run_id or "", kind="clear_all")
        except Exception:
            pass  # nosec B110

    def _undo_entry(self, entry: JournalEntry,
                    defer_refresh: bool = False) -> None:








        if entry.kind == KIND_REMOVE:


            for fid in entry.fids:
                self._auto_correction_removed.discard(int(fid))
            if not defer_refresh:
                self._after_shape_edit(changed=())
            return
        if entry.kind in (KIND_MERGE, KIND_SPLIT, KIND_REFINE):
            self._revert_shape_edit_entry(entry, defer_refresh=defer_refresh)

    def _refresh_correction_summary(self) -> None:
        try:
            self.dock_widget.set_correction_summary(
                self._auto_correct_journal.count)
        except (RuntimeError, AttributeError):
            pass





        try:
            self.dock_widget.set_auto_review_score_useful(
                self._run_scores_rank_objects())
        except (RuntimeError, AttributeError):
            pass





    def _on_auto_retry_guarded(self) -> None:







        self._auto_retry_guard.reset()
        self._on_auto_retry_clicked()


        try:
            self.dock_widget.set_retry_confirm_pending(False)
        except (RuntimeError, AttributeError):
            pass
        QTimer.singleShot(confirm_reset_ms(_RETRY_CONFIRM_RESET_MS), self._reset_retry_guard)

    def _reset_retry_guard(self) -> None:






        if self._auto_retry_guard.armed:
            self._auto_retry_guard.reset()
            try:
                self.dock_widget.set_retry_confirm_pending(False)
            except (RuntimeError, AttributeError):
                pass











    def _fold_qgis_edits_back(self, layer) -> int:








        if layer is None:
            return 0
        try:
            fields = [f.name() for f in layer.fields()]
        except (RuntimeError, AttributeError):
            fields = []
        has_score = "score" in fields
        has_det_id = "det_id" in fields
        features = []
        try:
            for feat in layer.getFeatures():
                geom = feat.geometry()
                if geom is None or geom.isEmpty():
                    continue
                score = 1.0
                if has_score:
                    try:
                        raw = feat["score"]
                        if raw is not None:
                            score = float(raw)
                    except (KeyError, TypeError, ValueError):
                        score = 1.0
                det_id = None
                if has_det_id:
                    try:
                        raw_id = feat["det_id"]
                        det_id = int(raw_id) if raw_id is not None else None
                    except (KeyError, TypeError, ValueError):
                        det_id = None
                from ...core.layer_conventions import repair_polygon




                repaired = repair_polygon(QgsGeometry(geom))
                if repaired is None or repaired.isEmpty():
                    continue
                features.append((repaired, score, det_id))
        except (RuntimeError, AttributeError):
            return 0





        clear_isolation = getattr(self, "_clear_bridge_isolation", None)
        if callable(clear_isolation):
            clear_isolation()
        from ...core.review_corrections import unique_object_ids

        existing_objects = list(getattr(self, "_auto_objects", None) or [])
        existing_ids = [self._object_fid_for(i) for i in range(len(existing_objects))]
        by_id = {det_id: index for index, det_id in enumerate(existing_ids)}
        final_ids = unique_object_ids(
            [det_id for _geom, _score, det_id in features], prior=existing_ids)
        snapshot = getattr(self, "_qgis_bridge_snapshot", {}) or {}
        present_snapshot_ids: set[int] = set()
        measurer = self._make_auto_area_measurer()


        restored: list[tuple[int, object]] = []
        pre_len = len(existing_objects)
        exempted: list[int] = []
        manual_ids = self._auto_manual_object_ids

        for (geom, score, raw_id), det_id in zip(features, final_ids):
            old_wkb = snapshot.get(raw_id) if raw_id is not None else None
            if raw_id in snapshot:
                present_snapshot_ids.add(raw_id)
            unchanged = old_wkb is not None and old_wkb == bytes(geom.asWkb())
            index = by_id.get(det_id)
            if index is None:
                existing_objects.append((geom, score, self._object_area_m2(geom, measurer)))
                existing_ids.append(det_id)
                by_id[det_id] = len(existing_objects) - 1




                if det_id is not None and det_id not in manual_ids:
                    manual_ids.add(det_id)
                    exempted.append(det_id)
            elif not unchanged:
                restored.append((index, existing_objects[index]))


                existing_objects[index] = (
                    geom, score, self._object_area_m2(geom, measurer))




        removed = set(getattr(self, "_auto_correction_removed", set()) or set())
        added_removed: list[int] = []
        for det_id in set(snapshot) - present_snapshot_ids:
            index = by_id.get(det_id)
            if index is not None and index not in removed:
                removed.add(index)
                added_removed.append(index)
        self._auto_objects = existing_objects
        self._auto_object_fids = existing_ids
        self._auto_correction_removed = removed




        edit = ShapeEdit(
            kind=KIND_REFINE,
            restored=tuple(restored),
            appended=len(existing_objects) - pre_len,
            unremoved=tuple(added_removed),
            exempted=tuple(exempted),
        )
        self._record_fold_edit(edit, fids=tuple(added_removed))



        self._shape_hit_geoms = {}
        self._reset_review_refine_cache()
        pixel_size = (self._auto_review or {}).get("pixel_size", 1.0)
        try:
            self._start_build_preview_cache(pixel_size)
        except (RuntimeError, AttributeError):
            pass





        try:
            self._start_auto_reslice()
        except Exception as exc:  # noqa: BLE001
            from ...core import telemetry_errors
            telemetry_errors.report_exception(
                exc, stage="auto_reslice_after_fold", module="auto_correct")
        try:
            self._refresh_correction_summary()
        except (RuntimeError, AttributeError):
            pass
        return len(features)
