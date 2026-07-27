"""Review ladder plumbing: the shared edit journal and step navigation.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); a mixin like its
siblings: state lives on the plugin instance, widgets on the dock. The pure
logic (the journal itself) lives in core/review_corrections.py; the two hand
edits that fill the journal live in auto_shape_edit.py. This file is what they
share: the journal, undo and clear, the step ladder, the status line and the
two-stage confirm on "Adjust and run again".
"""
from __future__ import annotations

from qgis.core import QgsGeometry, QgsProject
from qgis.PyQt.QtCore import QTimer

from ...core.i18n import tr
from ...core.review_corrections import CorrectionJournal, JournalEntry, RetryLinkState
from ...core.shape_edits import KIND_MERGE, KIND_REFINE, KIND_REMOVE, KIND_SPLIT, ShapeEdit

# The retry two-stage confirm disarms itself after this idle window.
_RETRY_CONFIRM_RESET_MS = 6000


class AutoCorrectMixin:
    """The review's edit journal, its step ladder and its locks."""

    # ------------------------------------------------------------------
    # State + wiring
    # ------------------------------------------------------------------

    def _init_auto_correct_state(self) -> None:
        """Fresh correction state; called at plugin init and per review."""
        self._auto_correction_removed: set[int] = set()
        # det_ids of objects the user drew by hand (Add a polygon) or made by a
        # native split. They carry score 1.0 and skip the review's confidence
        # and size gates, so a hand-drawn shape can never be filtered away by a
        # slider the user forgot. Keyed by det_id so it survives a reslice.
        self._auto_manual_object_ids: set[int] = set()
        # det_ids whose outline the user drew or edited by hand in the Manual
        # method. The shared shape refine is skipped for them, so the polygon
        # keeps the exact corners the user placed: squaring, simplifying or
        # opening a hand-traced outline moves it off the measurements they
        # took, and a small one can be erased outright. The AI method keeps
        # following the shared settings, which is what it is for.
        self._auto_shape_frozen_ids: set[int] = set()
        self._auto_correct_journal = CorrectionJournal()
        self._auto_review_step = 0
        self._auto_review_steps_seen: set[int] = set()
        self._auto_retry_guard = RetryLinkState()
        # The Correct step's fix method, mirrored from the dock's AI | Manual
        # switch: "ai" (on-device point refine) or "manual" (QGIS vertices).
        # The dock defaults a fresh review to "ai"; the plugin tracks it here so
        # a selection routes to the right session without asking the dock.
        self._correct_method = "ai"
        # AI-assisted Add lane state: True while a click on empty ground starts a
        # fresh outline, and True while a background install for Add is running.
        # Cleared on every session exit so it never leaks into the next one.
        self._refine_add_mode_active = False
        self._ai_add_install_pending = False
        # The free hand edits (merge / split) live in auto_shape_edit.py and
        # share this file's journal, so their state resets with this one.
        self._init_auto_shape_edit_state()

    def _connect_auto_correct_signals(self) -> None:
        """Wire the dock's review signals; called once from setup."""
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
        """Clear every correction surface (review entry, export, discard,
        retry)."""
        self._disarm_shape_tool()
        self._init_auto_correct_state()
        dock = self.dock_widget
        if dock is not None:
            try:
                dock.reset_review_steps()
            except (RuntimeError, AttributeError):
                pass
        # reset_review_steps() puts the dock switch back to "ai" (no signal), so
        # the plugin mirror already matches; name the polygon by the run class.
        self._apply_correct_class_label()

    def _apply_correct_class_label(self) -> None:
        """Name the selected polygon by the run's class (single class per run):
        the prompt, first letter capitalised. Falls back to the dock's neutral
        label when there is no prompt. tr-safe: the class name is user data, not
        a translatable string, so it is shown verbatim (only capitalised)."""
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
        """The user toggled the AI | Manual switch. Track it plugin-side and,
        when a fix session is live on a selected polygon, fold that session and
        re-enter the OTHER method on the same polygon (keeping the selection).
        Without a live session there is nothing to fold: the dock already
        re-labelled its own controls, so this only records the choice."""
        method = "manual" if str(method) == "manual" else "ai"
        self._correct_method = method
        if method == "ai":
            self._warm_local_ai_for_correct()
        # A live Add session belongs to no single polygon: just drop the lane so
        # the next click starts the other method's flow cleanly.
        if getattr(self, "_refine_add_mode_active", False):
            self._exit_ai_add_mode()
        # Remember the polygon under the live session (by stable det_id, since a
        # fold can reorder rows), fold the session, then re-open on it.
        target_det_id = self._active_session_target_det_id()
        folded = self._fold_active_correct_session()
        if not folded or target_det_id is None:
            return
        idx = self._object_index_for_det_id(target_det_id)
        if idx is None:
            return
        # Re-select on the new method; enter_session opens the matching session.
        self._set_correct_selection(idx, enter_session=True)

    def _active_session_target_det_id(self):
        """The stable det_id of the polygon the live fix session is working on,
        or None when no session is live (or its polygon cannot be named)."""
        if getattr(self, "_qgis_bridge_active", False):
            return getattr(self, "_qgis_bridge_target_det_id", None)
        if getattr(self, "_refine_handoff_active", False):
            idx = getattr(self, "_correct_selected_idx", None)
            return self._det_id_for_object_index(idx)
        return None

    def _det_id_for_object_index(self, idx):
        """Stable det_id for a canonical object index, read from the id list
        directly (never the index-fallback helper). None when unresolvable."""
        if idx is None:
            return None
        ids = getattr(self, "_auto_object_fids", None) or []
        if 0 <= idx < len(ids):
            return ids[idx]
        return None

    def _object_index_for_det_id(self, det_id):
        """Canonical object index for a stable det_id, or None. Skips a removed
        object (a folded delete): re-selecting it would be a no-op."""
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
        """Silently fold whatever fix session is live (the Done equivalent) so a
        method toggle or a merge/delete can run on committed geometry. Returns
        True when a session was folded, False when none was live."""
        if getattr(self, "_qgis_bridge_active", False):
            self.finish_qgis_edit_bridge()
            return True
        if getattr(self, "_refine_handoff_active", False):
            self._on_reshape_done()
            return True
        return False

    # ------------------------------------------------------------------
    # Ladder navigation
    # ------------------------------------------------------------------

    def _on_auto_review_step_requested(self, step: int) -> None:
        """Navigate the review ladder (0 Keep / 1 Correct / 2 Shapes).
        Every step is reachable both ways: the Manual lock that blocked
        backward moves is gone."""
        if self._auto_review is None:
            return
        step = max(0, min(2, int(step)))
        if step != 1:
            # Leaving Correct (step 1): fold any live fix session (so its edits
            # are not stranded), drop the Add lane, then drop the armed map tool
            # (select or the merge pick). The summary and journal remain the trace.
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
                if step == 1:
                    dock.set_review_recap(self._review_recap_text())
            except (RuntimeError, AttributeError):
                pass
        if step == 1:
            # Entering Correct: arm the resting select tool (click a detection
            # to select it), unless a reshape is open or a worker is running.
            self._arm_correct_select()
            self._warm_local_ai_for_correct()
        if step not in self._auto_review_steps_seen:
            self._auto_review_steps_seen.add(step)
            try:
                from ...core import telemetry
                telemetry.track_review_step(
                    run_id=self._auto_run_id or "", step=step)
            except Exception:
                pass  # nosec B110

    def _review_recap_text(self) -> str:
        """Last-step recap: what is kept, and what the hand edits changed."""
        kept = self._current_visible_review_count()
        edits = self._auto_correct_journal.count
        text = tr("{n} kept").format(n=kept)
        if edits:
            text += " · " + tr("{n} shape(s) edited this session").format(n=edits)
        return text

    def _enter_zero_detection_review(self, tiles_succeeded: int) -> None:
        """A run that found NOTHING still opens the review, on its own quiet
        line: the user who got zero results needs the prompt and the run-again
        link, not an empty export button. The selection layer stays alive so a
        later edit has somewhere to land."""
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
            "crs": layer.crs() if layer is not None else None,
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
            except (RuntimeError, AttributeError):
                pass
        # Land on Correct (step 1): a run that found nothing needs the "add
        # what is missing" surface, not the outline-polish step.
        self._auto_review_step = 1
        self._auto_review_steps_seen.add(1)
        self._arm_correct_select()
        self._warm_local_ai_for_correct()
        try:
            from ...core import telemetry
            telemetry.track_review_step(run_id=self._auto_run_id or "", step=1)
        except Exception:
            pass  # nosec B110

    def _disarm_after_handoff(self) -> None:
        """Handoff-return hook: disarm any stray Merge/Split gesture. It does
        NOT lock the steps. Hand edits are folded into the canonical rows and
        the touched det_ids skip the gates, so they survive every later
        recompute without a lock, and the paid batch a lock once guarded is
        gone (5706f2c), so a lock only broke Merge/Split and free navigation."""
        self._disarm_shape_tool()

    # ------------------------------------------------------------------
    # Status line
    # ------------------------------------------------------------------

    def _on_correct_status_action_requested(self) -> None:
        """The status line's secondary action: today the merge confirm, armed
        as soon as enough shapes are picked."""
        if self._shape_edit_mode == KIND_MERGE:
            self._commit_merge_picks()

    def _set_correct_status(self, kind: str, text: str, undo: bool = False,
                            action: str = "") -> None:
        try:
            self.dock_widget.set_correct_status(
                kind, text, undo_visible=undo, action_text=action)
        except (RuntimeError, AttributeError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Journal: push, undo, clear
    # ------------------------------------------------------------------

    def _push_correct_entry(self, entry: JournalEntry) -> None:
        self._auto_correct_journal.push(entry)

    def _ai_session_has_undo(self) -> bool:
        """True while a live AI fix session still has a gesture to unwind (a
        placed point, a mask/geometry step, a frozen part or a deleted object).
        With none, Undo falls through to the review's own edit journal."""
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
        # A live manual session owns Undo: the last point placed, else the last
        # native edit. Falling through to the journal here would undo a
        # PREVIOUS polygon's correction while this one is still open.
        if getattr(self, "_qgis_bridge_active", False):
            self.undo_qgis_bridge_edit()
            return
        # During a live AI fix session, Undo unwinds the last session gesture
        # (point, reshape, delete); only once the session has nothing left to
        # undo does it fall through to the review's edit journal.
        if self._ai_session_has_undo():
            self._on_undo()
            return
        entry = self._auto_correct_journal.undo()
        if entry is None:
            return
        self._undo_entry(entry)
        self._refresh_correction_summary()
        try:
            from ...core import telemetry
            telemetry.track_review_correct_undo(
                run_id=self._auto_run_id or "", kind=entry.kind)
        except Exception:
            pass  # nosec B110

    def _on_auto_correction_clear_requested(self) -> None:
        entries = self._auto_correct_journal.clear()
        if not entries:
            return
        for entry in entries:
            self._undo_entry(entry)
        self._refresh_correction_summary()
        try:
            from ...core import telemetry
            telemetry.track_review_correct_undo(
                run_id=self._auto_run_id or "", kind="clear_all")
        except Exception:
            pass  # nosec B110

    def _undo_entry(self, entry: JournalEntry) -> None:
        """Reverse ONE journal entry's effect."""
        if entry.kind == KIND_REMOVE:
            # Remove only marked indices in _auto_correction_removed; undo drops
            # them back and reslices (no geometry was rewritten).
            for fid in entry.fids:
                self._auto_correction_removed.discard(int(fid))
            self._after_shape_edit(changed=())
            return
        if entry.kind in (KIND_MERGE, KIND_SPLIT, KIND_REFINE):
            self._revert_shape_edit_entry(entry)

    def _refresh_correction_summary(self) -> None:
        try:
            self.dock_widget.set_correction_summary(
                self._auto_correct_journal.count)
        except (RuntimeError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Retry two-stage guard
    # ------------------------------------------------------------------

    def _on_auto_retry_guarded(self) -> None:
        """Adjust-and-run-again with edits in flight asks one inline confirm
        (the link itself relabels; no modal). Empty journal = the original
        one-click behavior."""
        if (self._auto_review is None or self._auto_correct_journal.count == 0):
            self._auto_retry_guard.reset()
            self._on_auto_retry_clicked()
            return
        if self._auto_retry_guard.activate():
            try:
                self.dock_widget.set_retry_confirm_pending(False)
            except (RuntimeError, AttributeError):
                pass
            self._on_auto_retry_clicked()
            return
        try:
            self.dock_widget.set_retry_confirm_pending(True)
        except (RuntimeError, AttributeError):
            pass
        QTimer.singleShot(_RETRY_CONFIRM_RESET_MS, self._reset_retry_guard)

    def _reset_retry_guard(self) -> None:
        if self._auto_retry_guard.armed:
            self._auto_retry_guard.reset()

    # ------------------------------------------------------------------
    # QGIS Advanced Digitizing bridge fold-back (seam owned by plan 1)
    # ------------------------------------------------------------------
    # The bridge itself (arm editing/snapping/topology, reveal toolbars, commit
    # + restore) lives in QgisEditBridgeMixin (qgis_edit_bridge.py). Plan 1 owns
    # only two halves of the seam: the dock anchor banner
    # (enter/leave_qgis_bridge_state, dock/qgis_bridge.py) and this fold-back,
    # which the bridge calls after a commit to bring the edited geometry back
    # into the review's object model.

    def _fold_qgis_edits_back(self, layer) -> int:
        """Synchronise native edits without replacing the entire review model.

        The editable layer only contains features visible at the current
        confidence/size filters. Replacing ``_auto_objects`` from it used to
        silently discard hidden detections and to use temporary provider fids as
        display identities. ``det_id`` plus the bridge's pre-edit snapshot lets
        us apply only the actual QGIS changes and retain every untouched object.
        """
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
                features.append((QgsGeometry(geom), score, det_id))
        except (RuntimeError, AttributeError):
            return 0
        from ...core.review_corrections import unique_object_ids

        existing_objects = list(getattr(self, "_auto_objects", None) or [])
        existing_ids = [self._object_fid_for(i) for i in range(len(existing_objects))]
        by_id = {det_id: index for index, det_id in enumerate(existing_ids)}
        final_ids = unique_object_ids(
            [det_id for _geom, _score, det_id in features], prior=existing_ids)
        snapshot = getattr(self, "_qgis_bridge_snapshot", {}) or {}
        present_snapshot_ids: set[int] = set()
        measurer = self._make_auto_area_measurer()
        # Undo bookkeeping for the fold journal: the pre-edit row of every row
        # this fold overwrites, and the det_ids it newly exempts from the gates.
        restored: list[tuple[int, object]] = []
        pre_len = len(existing_objects)
        exempted: list[int] = []
        manual_ids = self._auto_manual_object_ids
        # Every det_id this fold freezes, so one Undo can thaw exactly those.
        froze: list[int] = []
        frozen_ids = getattr(self, "_auto_shape_frozen_ids", None)
        if frozen_ids is None:
            frozen_ids = set()
            self._auto_shape_frozen_ids = frozen_ids

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
                # A feature with no match in the pre-edit set is deliberate user
                # geometry: a polygon drawn with Add a polygon, or a new piece
                # from a native split. Both are hand work, so both skip the
                # confidence/size gates and stay on the map as drawn.
                if det_id is not None and det_id not in manual_ids:
                    manual_ids.add(det_id)
                    exempted.append(det_id)
                if det_id is not None and det_id not in frozen_ids:
                    frozen_ids.add(det_id)
                    froze.append(det_id)
            elif not unchanged:
                restored.append((index, existing_objects[index]))
                existing_objects[index] = (
                    geom, score, self._object_area_m2(geom, measurer))
                # The user moved these corners themselves. Freeze the outline so
                # the shared shape refine stops rewriting it on the next reslice.
                if det_id is not None and det_id not in frozen_ids:
                    frozen_ids.add(det_id)
                    froze.append(det_id)

        # A feature that was visible when editing opened but is gone now was
        # explicitly deleted or absorbed by a native merge. Hidden detections
        # never occur in the snapshot and therefore remain untouched.
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
        # Every edit is now a canonical base, so a native edit is the same class
        # of mutation as an AI reshape: journal it so one Undo restores the
        # pre-edit bases, pops the appended objects, and clears the exemptions
        # and removals this fold added, strictly LIFO with merge/remove.
        edit = ShapeEdit(
            kind=KIND_REFINE,
            restored=tuple(restored),
            appended=len(existing_objects) - pre_len,
            unremoved=tuple(added_removed),
            exempted=tuple(exempted),
            frozen=tuple(froze),
        )
        self._record_fold_edit(edit, fids=tuple(added_removed))
        # The hit snapshot describes the shapes as they were BEFORE these
        # edits. Dropping it here means the click test and the yellow highlight
        # fall back to the folded geometry until the reslice republishes.
        self._shape_hit_geoms = {}
        self._reset_review_refine_cache()
        pixel_size = (self._auto_review or {}).get("pixel_size", 1.0)
        try:
            self._start_build_preview_cache(pixel_size)
        except (RuntimeError, AttributeError):
            pass
        # The object model is already corrected above (distinct det_ids), so the
        # reslice is what republishes the layer with the new per-object colours.
        # If it raises, the layer keeps the raw committed geometry (after a split
        # both halves share a colour): log it rather than let it bubble into the
        # bridge's swallow, and still run the summary refresh.
        try:
            self._start_auto_reslice()
        except Exception as exc:  # noqa: BLE001
            from ...core import telemetry
            telemetry.report_exception(
                exc, stage="auto_reslice_after_fold", module="auto_correct")
        try:
            self._refresh_correction_summary()
        except (RuntimeError, AttributeError):
            pass
        return len(features)
