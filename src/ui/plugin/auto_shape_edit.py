"""Review hand edit: merge several detections into one.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); a mixin like its
siblings. It sits beside auto_correct.py, which owns the review ladder, and it
shares that file's journal, so "Undo last" and "Clear all" reach it too.

Merge is FREE: pure geometry on detections the run already paid for, no server
call, no credit.

The geometry is core/geometry_ops.py, the index bookkeeping is
core/shape_edits.py, and both are pure and tested on their own. What is left
here is the part that needs a canvas: arming the pick tool, deciding which
detection the user clicked, and pushing the result back through the same
reslice every other correction uses.

Split is gone: cutting a detection now goes through the QGIS Advanced
Digitizing bridge (qgis_edit_bridge.py), which runs QGIS's own, mature split
tool on the review layer instead of our home-grown one.
"""
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

# Correct step's default map tool: click a detection to select it (then Reshape
# with AI or Remove). It shares the one shape-tool slot, so select and merge can
# never both own the canvas tool. UI-only, so it lives here, not in shape_edits.
KIND_SELECT = "select"


class AutoShapeEditMixin:
    """The review's Correct-step map tools: select a detection, or merge
    pieces. Both run on the ONE shape-tool slot (never two owners at once)."""

    # Click tolerance for a merge pick, in screen pixels: a click this far from
    # a shape still selects it when nothing strictly contains the point, so
    # thin or small detections are reachable.
    _PICK_TOLERANCE_PX = 8

    # ------------------------------------------------------------------
    # State + wiring
    # ------------------------------------------------------------------

    def _init_auto_shape_edit_state(self) -> None:
        """Fresh hand-edit state; called from _init_auto_correct_state, so a
        new review starts with nothing armed and nothing picked."""
        self._shape_maptool = None
        self._maptool_before_shape = None
        # Per-shape settings live and die with the review they belong to.
        _init_overrides = getattr(self, "_init_shape_override_state", None)
        if _init_overrides is not None:
            _init_overrides()
        # "select" | "merge" | None: which map tool owns the canvas right now.
        self._shape_edit_mode: str | None = None
        self._shape_picks = PickSet()
        self._shape_pick_bands: dict[int, object] = {}
        # The Correct step's single selected detection (canonical _auto_objects
        # index), and its yellow highlight band. Selection drives the Reshape /
        # Remove sub-card; None means nothing selected.
        self._correct_selected_idx: int | None = None
        self._correct_selection_band = None
        self._shape_hover_idx: int | None = None
        self._shape_hover_band = None
        # Per-gesture hit-test cache (built on arm, dropped on disarm): the
        # review params (a preset regex rebuild) and a QgsSpatialIndex over the
        # object bboxes, so a merge pick hit-test does not scan every object
        # with a full GEOS op on each click.
        self._shape_review_params = None
        self._shape_hit_index = None
        # Canonical object index -> geometry currently drawn in the review.
        # After a Manual reshape the visible geometry can differ from the
        # original detection; hit-testing the rendered shape keeps Correct's
        # actions anchored to what the user actually clicked.
        self._shape_hit_geoms: dict[int, object] = {}
        # id(JournalEntry) -> the ShapeEdit that reverses it. Entries are plain
        # dataclasses (unhashable), so the payload is keyed on their id.
        self._shape_edit_undo: dict[int, object] = {}
        # id(JournalEntry) -> the pre-fold _auto_manual_removed snapshot, for an
        # AI-refine fold whose deletions live in that set (not the correction
        # removal set the ShapeEdit's unremoved covers). Restored on undo.
        self._fold_manual_removed_undo: dict[int, set] = {}

    def _connect_auto_shape_edit_signals(self) -> None:
        """Wire the dock's Correct-step map-tool signals; called from the
        correction wiring."""
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

    # ------------------------------------------------------------------
    # Arming
    # ------------------------------------------------------------------

    def _shape_edit_allowed(self) -> bool:
        """Whether a hand edit may be armed at all: an open review and no
        worker in flight. Merge/Split stay available after a Refine-in-Manual
        return (the old Manual-lock block is gone; hand edits are protected by
        geometry, so there is nothing left to guard)."""
        return (self._auto_review is not None and self._auto_worker is None)

    def _on_auto_shape_edit_requested(self, kind: str) -> None:
        """Arm (or disarm) the Merge pick. Clicking it while armed turns it off
        and returns to the resting select tool. Only 'merge' comes through this
        signal now: split moved to the QGIS bridge, select is armed by the step
        navigation, not a button."""
        kind = str(kind)
        if kind != KIND_MERGE:
            return
        if self._shape_edit_mode == kind:
            self._disarm_shape_tool()
            self._arm_correct_select()  # back to the resting select tool
            return
        if not self._shape_edit_allowed():
            return
        # Merge from inside a live AI fix session: fold that session first so the
        # merge runs on committed geometry. Carry the selection across the fold
        # by stable det_id (a fold can reorder rows) so the merge still seeds
        # from the polygon the user was on.
        if getattr(self, "_refine_handoff_active", False):
            det_id = self._det_id_for_object_index(
                getattr(self, "_correct_selected_idx", None))
            self._fold_active_correct_session()
            self._correct_selected_idx = self._object_index_for_det_id(det_id)
        # Join starts FROM the selected detection: it is the first piece, so the
        # user only has to click the others. Capture it before _arm_shape_tool,
        # whose disarm clears the selection.
        seed = self._correct_selected_idx
        self._arm_shape_tool(kind)
        if seed is not None and 0 <= seed < len(self._auto_objects):
            self._shape_picks.toggle(seed)
            self._refresh_pick_highlight()
            self._refresh_merge_pick_status()

    def _arm_shape_tool(self, kind: str) -> None:
        """Put the select or merge pick tool on the canvas (one at a time)."""
        # Drops any armed pick first, so exactly one tool is ever live and the
        # map tool it replaced is the one restored later.
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
        else:  # KIND_SELECT: Escape clears the selection.
            tool.cancelled.connect(self._on_select_cancelled)
            # Right-click deletes the shape under the cursor. Select only: in a
            # merge pick a context-click must stay inert, or it would drop a
            # piece the user is still assembling.
            tool.point_right_clicked.connect(self._on_shape_point_right_clicked)
        # The user picking any OTHER QGIS map tool deactivates ours; reset the
        # armed UI (shortcuts, indicator, picks) WITHOUT restoring the previous
        # tool, since QGIS has already switched to the user's new choice.
        try:
            tool.tool_deactivated.connect(self._on_shape_tool_deactivated)
        except (RuntimeError, AttributeError):
            pass
        self._shape_maptool = tool
        self._shape_edit_mode = kind
        self._shape_picks.clear()
        self._build_shape_hit_index()
        # Merge owns Enter/Escape (confirm / cancel), so it silences the review
        # shortcuts; select is passive and leaves them on.
        if kind == KIND_MERGE:
            try:
                self.dock_widget.set_auto_shortcuts_enabled(False)
            except (RuntimeError, AttributeError):
                pass
        # setMapTool deactivates the outgoing tool; if that is a leftover draw
        # tool whose signal is still live, ignore its deactivation so it cannot
        # null the tool we just set.
        self._suspend_shape_deactivate = True
        try:
            canvas.setMapTool(tool)
        finally:
            self._suspend_shape_deactivate = False
        # The canvas must hold keyboard focus for the tool's Enter/Escape to
        # arrive before the first click; arming from a dock button leaves focus
        # in the dock otherwise.
        try:
            canvas.setFocus()
        except (RuntimeError, AttributeError):
            pass
        # The Merge toggle only lights for merge; select is the resting default.
        try:
            self.dock_widget.set_correct_armed(kind if kind == KIND_MERGE else None)
        except (RuntimeError, AttributeError):
            pass
        if kind == KIND_MERGE:
            self._refresh_merge_pick_status()

    def _disarm_shape_tool(self) -> None:
        """Drop the armed select/merge tool and restore the previous map tool.
        Also clears the selection highlight, so leaving the Correct step (which
        calls this) leaves no stray yellow outline on the canvas."""
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
            # We are switching the tool ourselves, so silence the tool's own
            # deactivate signal (it would re-enter this cleanup) and free the
            # C++ object once the previous tool is back.
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
        """Escape or right-click during a merge: disarm, change nothing, and
        fall back to the resting select tool."""
        self._disarm_shape_tool()
        self._arm_correct_select()

    def _on_shape_tool_deactivated(self) -> None:
        """The armed tool was deactivated by QGIS because the user picked
        ANOTHER map tool. Reset the armed UI (shortcuts, indicator, picks and
        the hit cache) but do NOT restore the previous tool: QGIS already
        switched to the user's new choice, so re-setting it would fight them."""
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

    # ------------------------------------------------------------------
    # Select a detection (the Correct step's resting tool) + Remove
    # ------------------------------------------------------------------

    def _arm_correct_select(self) -> None:
        """Arm the resting select tool on the Correct step: click a detection
        to select it, empty ground to deselect. No-op unless the review is open
        on the Correct step, nothing is already armed, and no reshape session
        or worker is running (so it never fights the merge pick or a reshape)."""
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
        """Escape while selecting: clear the selection but keep the tool armed
        (select is the resting default, not a one-shot gesture)."""
        self._set_correct_selection(None)

    def _set_correct_selection(self, idx, *, enter_session: bool = False) -> None:
        """Select the detection at canonical index ``idx`` (or None to clear).
        A removed or out-of-range index reads as no selection. Drives the
        selected-object panel via the dock.

        ``enter_session`` is the round-3 hook: a genuine map pick (the canvas
        click, or a merge re-select) opens the current method's fix session on
        the picked polygon. Every other caller (deselect, programmatic
        re-select, the bridge return) leaves it False so it stays a pure state
        setter. Keyword-only so the positional signature the MCP path relies on
        never changes."""
        if idx is not None and (
                idx < 0 or idx >= len(self._auto_objects) or idx in self._review_removed_fids()):
            idx = None
        self._correct_selected_idx = idx
        self._refresh_correct_selection_band()
        try:
            self.dock_widget.set_correct_selection(1 if idx is not None else 0)
        except (RuntimeError, AttributeError):
            pass
        # Merge only helps when there is another shape to join this one to, so
        # its tile shows only when a neighbour overlaps or touches the selection.
        try:
            self.dock_widget.set_merge_available(
                idx is not None and self._selected_has_mergeable_neighbor(idx))
        except (RuntimeError, AttributeError):
            pass
        # The card shows this object's measurements and its own shape settings,
        # so both are pushed from the ONE place the selection changes.
        _push = getattr(self, "_push_shape_only_state", None)
        if _push is not None:
            _push()
        if enter_session and idx is not None:
            self._enter_correct_method_session()

    def _enter_correct_method_session(self) -> None:
        """Open the selected polygon's fix session for the current method: AI =
        the on-device point refine (reuses the reshape handoff opened on this
        polygon), Manual = the native QGIS vertex bridge. No-op when a session
        already owns the canvas, off the Correct step, or headless (the MCP path
        never opens an interactive session)."""
        idx = getattr(self, "_correct_selected_idx", None)
        if idx is None or self._auto_review is None:
            return
        if getattr(self, "_auto_review_step", 0) != 1:
            return
        if getattr(self, "_headless", False):
            return
        if getattr(self, "_refine_handoff_active", False) or getattr(
                self, "_qgis_bridge_active", False):
            return
        method = getattr(self, "_correct_method", "ai")
        if method == "manual":
            # The bridge reads _correct_selected_idx as its target; it keeps its
            # own banner rather than the panel (kept treatment, see report).
            self.enter_qgis_edit_bridge()
        else:
            # _on_reshape_ai_requested reads _correct_selected_idx too.
            self._on_reshape_ai_requested()

    def _selected_has_mergeable_neighbor(self, det_idx) -> bool:
        """True when another VISIBLE detection overlaps or touches the object at
        ``det_idx`` (so Merge has something to join). Uses the gesture hit index
        when armed, else the visible set, and a small ground tolerance so a
        seam-half that touches but does not strictly overlap still qualifies."""
        if det_idx is None or det_idx < 0 or det_idx >= len(self._auto_objects):
            return False
        sel = self._shape_hit_geoms.get(det_idx)
        if sel is None:
            sel = self._auto_objects[det_idx][0]
        if sel is None or sel.isEmpty():
            return False
        # Carry a few screen pixels into the run CRS, the same way _hit_object_at
        # does, so "touching" tolerates a hairline gap between seam halves.
        tol = 0.0
        try:
            mupp = self.iface.mapCanvas().mapSettings().mapUnitsPerPixel()
            centre = sel.boundingBox().center()
            off = QgsPointXY(centre.x() + mupp * self._PICK_TOLERANCE_PX, centre.y())
            run_pts = self._points_in_run_crs([QgsPointXY(centre), off])
            if len(run_pts) == 2:
                tol = float(run_pts[0].distance(run_pts[1]))
        except (RuntimeError, AttributeError, TypeError):
            tol = 0.0
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
        """Drop the selection index, highlight band and matching dock state."""
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
        """Re-anchor the Correct step on the visible set just published.

        Two caches describe the shapes as drawn: the hit index (built when a
        gesture arms) and the yellow selection band. A native QGIS edit, a
        per-polygon setting or any reslice rewrites geometry AFTER both were
        taken, so they are re-taken here, from the ONE place the visible set
        changes. Cheap: the index build is the same work an arm already does,
        and it only runs while a gesture owns the canvas.
        """
        if self._auto_review is None:
            return
        if getattr(self, "_shape_edit_mode", None) is not None:
            self._build_shape_hit_index()
        else:
            # Nothing armed: a stale snapshot would only mislead the band.
            self._shape_hit_geoms = {}
        if getattr(self, "_correct_selected_idx", None) is None:
            return
        self._refresh_correct_selection_band()
        # A filter reslice can hide or reveal a neighbour, so re-evaluate whether
        # Merge still has something to join.
        try:
            self.dock_widget.set_merge_available(
                self._selected_has_mergeable_neighbor(self._correct_selected_idx))
        except (RuntimeError, AttributeError):
            pass
        _push = getattr(self, "_push_shape_only_state", None)
        if _push is not None:
            _push()

    def _drawn_geom_for_index(self, det_idx: int):
        """The geometry currently DRAWN for one object, or None.

        Three sources, best first: the armed gesture's hit snapshot, the
        review's published visible set (matched by display id), and last the
        canonical detection. The order matters after a native QGIS edit or a
        per-polygon setting change: the canonical row and the drawn shape are
        the same object seen before and after the refine, and the highlight
        has to trace what the user sees.
        """
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
        """Redraw the yellow selected-detection outline from the current index.
        Rebuilt each time so a deselect clears it and nothing drifts."""
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
        # QGIS-native selection yellow, the reflex every QGIS user already has.
        self._correct_selection_band = self._make_geom_band(
            geom, QColor(255, 255, 0, 255), 3)

    def _on_remove_requested(self) -> None:
        """Remove the selected detection (the panel's Delete row and the Delete
        key).

        Also the Delete-key path DURING an AI fix session: fold the session
        first (by stable det_id, since a fold can reorder rows), then remove the
        polygon through this same review-side flow. The dock's Delete shortcut
        stands down while a session runs, so the panel's Delete row is the
        session entry here."""
        if getattr(self, "_refine_handoff_active", False):
            det_id = self._det_id_for_object_index(
                getattr(self, "_correct_selected_idx", None))
            self._fold_active_correct_session()
            self._correct_selected_idx = self._object_index_for_det_id(det_id)
        self._remove_detection_index(self._correct_selected_idx)

    def _remove_detection_index(self, idx) -> None:
        """Remove ONE detection by canonical index: mark it removed (no geometry
        rewrite), journal it so Undo last / Clear all reach it, drop the
        selection and the hover outline when they were on it, and reslice the
        visible set. Shared by the panel's Delete row and the canvas
        right-click, so both land on the same journal entry and the same undo."""
        if idx is None or self._auto_review is None:
            return
        if idx < 0 or idx >= len(self._auto_objects):
            return
        self._auto_correction_removed.add(int(idx))
        self._record_remove_entry(idx)
        if self._correct_selected_idx == idx:
            self._set_correct_selection(None)
        if getattr(self, "_shape_hover_idx", None) == idx:
            # The shape is gone; its blue outline would otherwise sit on the
            # canvas until the next cursor move.
            self._set_shape_hover(None)
        self._after_shape_edit(changed=())
        self._set_correct_status(
            "success", tr("1 detection removed"), undo=True)
        self._track_shape_edit(KIND_REMOVE, "removed", 1)

    def _record_remove_entry(self, idx: int) -> None:
        """Push a Remove onto the correction journal (undo discards the index
        from _auto_correction_removed; see AutoCorrectMixin._undo_entry)."""
        self._push_correct_entry(JournalEntry(kind=KIND_REMOVE, fids=(int(idx),)))

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _on_shape_point_clicked(self, point) -> None:
        """A click on the canvas while a Correct map tool is armed. In select
        mode it picks the detection under the cursor (empty ground deselects);
        in merge mode it toggles that detection in and out of the pick set."""
        if self._auto_review is None:
            return
        if self._shape_edit_mode == KIND_SELECT:
            # A genuine map pick opens the current method's fix session on the
            # polygon under the cursor; empty ground clears the selection.
            self._set_correct_selection(
                self._hit_object_at(point), enter_session=True)
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
        """Right-click on the resting select tool: delete the shape under the
        cursor on the spot.

        The left click on this tool OPENS the polygon in the current method's
        fix session, so throwing a bad detection away used to cost a session
        entry and an exit, whichever method was live. The right click never
        selects and never opens: it removes what it lands on, through the same
        journalled path as the panel's Delete row, so Undo brings it back.
        Empty ground does nothing (no message: a miss is not an error)."""
        if self._auto_review is None or self._shape_edit_mode != KIND_SELECT:
            return
        # Belt and braces: the select tool is off the canvas while either fix
        # session owns it, and those sessions have their own right-click.
        if getattr(self, "_refine_handoff_active", False) or getattr(
                self, "_qgis_bridge_active", False):
            return
        det_idx = self._hit_object_at(point)
        if det_idx is None:
            return
        self._remove_detection_index(det_idx)

    def _on_shape_cursor_moved(self, point) -> None:
        """Show a lightweight outline before a Correct-step click.

        The active tool remains navigable: a drag pans, while a stationary
        cursor only asks the spatial index for the one visible object under it.
        Rebuilding the rubber band only when that object changes keeps the
        canvas responsive over dense detection sets.
        """
        if self._shape_edit_mode not in (KIND_SELECT, KIND_MERGE):
            return
        idx = self._hit_object_at(point)
        self._set_shape_hover(idx)
        if self._shape_edit_mode == KIND_SELECT:
            # Resting on a polygon is the announcement of the pick: start
            # reading its imagery now, so the click does not have to wait for it.
            self._schedule_correct_hover_warm(idx)

    def _set_shape_hover(self, idx) -> None:
        """Set the transient blue hover outline for one selectable shape."""
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
        """The armed merge line: how many are picked, and the confirm link as
        soon as there are enough to join."""
        picked = len(self._shape_picks)
        action = ""
        if picked == 0:
            text = tr("Click each piece of the object you want to join.")
        elif picked < MIN_MERGE_PICKS:
            # One pick is the seeded selection, so name what is missing rather
            # than counting: the user has not "picked one", they picked THIS one.
            text = tr("Now click the other pieces of this object.")
        else:
            # Same confirm rule as Split: press Enter to commit. The link
            # stays as a mouse alternative, so both paths work.
            text = tr("{n} shapes picked. Press Enter to join.").format(
                n=picked)
            action = tr("Merge {n} shapes · Free").format(n=picked)
        self._set_correct_status("armed", text, action=action)

    def _on_shape_merge_confirmed(self) -> None:
        """Enter while the merge pick is armed: commit if enough are picked,
        otherwise leave the status line as the hint (never an error)."""
        if self._shape_edit_mode != KIND_MERGE:
            return
        if len(self._shape_picks) >= MIN_MERGE_PICKS:
            self._commit_merge_picks()

    def _commit_merge_picks(self) -> None:
        """Join the picked detections into one object. Too few picks is a
        no-op with a line saying so, never an error."""
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
        geoms = [self._shape_hit_geoms.get(idx, self._auto_objects[idx][0])
                 for idx in (plan.target, *plan.absorbed)
                 if self._shape_hit_geoms.get(idx, self._auto_objects[idx][0]) is not None]
        merged = merge_geometries(geoms)
        if merged is None or merged.isEmpty():
            self._set_correct_status("warning", tr(
                "Those shapes could not be joined. Nothing was changed."))
            return
        joined = len(plan.absorbed) + 1
        edit = apply_merge(self._auto_objects, plan,
                           self._object_row(merged, plan.score))
        self._auto_correction_removed.update(plan.absorbed)
        self._record_shape_edit(edit, fids=plan.absorbed)
        text = tr("{n} shapes merged into one · Free").format(n=joined)
        if merged.isMultipart() and len(merged.asGeometryCollection()) > 1:
            # Honest, not blocking: the user may well want one object made of
            # separate parts (an object cut by a road, a courtyard building).
            text += " " + tr(
                "The pieces do not touch, so the result is one object in "
                "several parts.")
        self._disarm_shape_tool()
        self._set_correct_status("success", text, undo=True)
        # Only the target row's geometry changed (the absorbed rows are just
        # marked removed); the rest keep their cached squared shape.
        self._after_shape_edit(changed=(plan.target,))
        self._track_shape_edit(KIND_MERGE, "merged", joined)
        self._arm_correct_select()  # back to the resting select tool
        # The merged shape becomes the polygon's new base: re-select it (as a
        # genuine pick) so the current method's fix session opens on it.
        self._set_correct_selection(plan.target, enter_session=True)

    # ------------------------------------------------------------------
    # Journal, undo and the shared refresh
    # ------------------------------------------------------------------

    def _record_shape_edit(self, edit, fids: tuple) -> None:
        """Record one hand edit on the review's journal, so Undo last and Clear
        all reach it. The reversal payload rides beside it, keyed on the
        entry's id."""
        entry = JournalEntry(kind=edit.kind, fids=tuple(fids))
        self._push_correct_entry(entry)
        self._shape_edit_undo[id(entry)] = edit

    def _record_fold_edit(self, edit, fids: tuple,
                          manual_removed_before=None):
        """Journal one batch fold (an AI-refine session or a native-edit
        commit): every hand edit becomes a canonical base, so a single Undo
        restores the pre-fold bases, pops the appended objects, drops the gate
        exemptions the fold added, and puts back the removal set it changed.
        Strictly LIFO with merge/remove, so an older edit never clobbers it.

        ``manual_removed_before`` is the pre-fold ``_auto_manual_removed`` set
        for the AI path (whose deletions live there); the native path leaves it
        None and rides the ShapeEdit's ``unremoved`` for the correction set.
        Returns the journal entry, or None when the fold changed nothing."""
        if not (edit.restored or edit.appended or edit.unremoved or edit.exempted or manual_removed_before is not None):
            return None
        entry = JournalEntry(kind=KIND_REFINE, fids=tuple(fids))
        self._push_correct_entry(entry)
        self._shape_edit_undo[id(entry)] = edit
        if manual_removed_before is not None:
            self._fold_manual_removed_undo[id(entry)] = set(manual_removed_before)
        self._refresh_correction_summary()
        return entry

    def _revert_shape_edit_entry(self, entry) -> None:
        """Undo one recorded hand edit or fold (called by the journal's undo
        branch).

        A payload that is gone means the object list was rebuilt under it: do
        nothing rather than write rows at indices that moved.
        """
        edit = self._shape_edit_undo.pop(id(entry), None)
        if edit is None:
            self._fold_manual_removed_undo.pop(id(entry), None)
            return
        ids = self._shape_edit_ids()
        restored_indices = tuple(idx for idx, _row in edit.restored)
        unremoved = revert_shape_edit(self._auto_objects, ids, edit)
        self._auto_object_fids = ids
        self._auto_correction_removed.difference_update(unremoved)
        # A fold marks the objects it touched exempt from the gates; undo drops
        # only the det_ids THIS fold added, so an object touched by an earlier
        # fold too keeps its exemption.
        exempted = getattr(edit, "exempted", ()) or ()
        if exempted:
            manual_ids = getattr(self, "_auto_manual_object_ids", None)
            if manual_ids is not None:
                manual_ids.difference_update(int(i) for i in exempted)
        # Same rule for the shape freeze: undoing the fold that froze an
        # outline hands it back to the shared settings, while an outline an
        # earlier fold froze stays frozen.
        froze = getattr(edit, "frozen", ()) or ()
        if froze:
            frozen_ids = getattr(self, "_auto_shape_frozen_ids", None)
            if frozen_ids is not None:
                frozen_ids.difference_update(int(i) for i in froze)
        # The AI-refine fold records deletions in _auto_manual_removed wholesale,
        # so its undo restores the pre-fold snapshot rather than a per-index diff.
        snap = self._fold_manual_removed_undo.pop(id(entry), None)
        if snap is not None:
            self._auto_manual_removed = set(snap)
        # An undo restores the overwritten rows (and drops any appended tail);
        # only the restored rows changed shape. The unremoved (merge-absorbed)
        # rows kept their original geometry, so their cache is still valid.
        self._after_shape_edit(changed=restored_indices)

    def _after_shape_edit(self, changed=None) -> None:
        """One refresh for every hand edit and every undo of one: the changed
        objects' cached geometry is dropped, the confidence-drag cache is
        rebuilt and the visible set (with the Export count) is resliced. Same
        path the drawn corrections use, so nothing else has to know a shape
        changed.

        ``changed`` is the object indices this edit rewrote (merge target,
        split target + appended pieces, or the rows an undo restored). Only
        those lose their cached squared shape; the rest of the set keeps it, so
        a single edit no longer re-squares every object. ``None`` (an unknown
        caller) falls back to dropping the whole cache."""
        if changed is None:
            self._reset_review_refine_cache()
        else:
            self._invalidate_review_refine(changed)
        pixel_size = (self._auto_review or {}).get("pixel_size", 1.0)
        self._start_build_preview_cache(pixel_size)
        self._start_auto_reslice()
        self._refresh_correction_summary()

    def _shape_edit_ids(self) -> list:
        """The stable colour ids, padded to run parallel to _auto_objects."""
        ids = list(getattr(self, "_auto_object_fids", None) or [])
        align_ids(self._auto_objects, ids)
        return ids

    def _object_row(self, geom, score: float, measurer=None) -> tuple:
        """One canonical object row: (geometry, score, ground area m2). The
        area is measured the same way the run measured every other object, so
        the Min/Max size filters treat an edited shape like any other. Pass a
        shared ``measurer`` when building several rows: building one per row
        reloads the ellipsoid each time."""
        if measurer is None:
            measurer = self._make_auto_area_measurer()
        return (geom, float(score), self._object_area_m2(geom, measurer))

    def _track_shape_edit(self, kind: str, outcome: str, objects: int) -> None:
        try:
            from ...core import telemetry
            telemetry.track_review_correct_box(
                run_id=self._auto_run_id or "", label=1, outcome=outcome,
                objects=int(objects), gesture=kind)
        except Exception:
            pass  # nosec B110

    # ------------------------------------------------------------------
    # Canvas helpers
    # ------------------------------------------------------------------

    def _visible_object_indices(self) -> list[int]:
        """Indices of the objects currently SHOWN in the review: not removed,
        and passing the confidence + size filters. While a gesture is armed the
        review params are cached (they do not change during picking), so this
        skips the per-call preset rebuild."""
        params = self._shape_review_params or self._widget_review_params()
        removed = self._review_removed_fids()
        out: list[int] = []
        for det_idx, (geom, score, area) in enumerate(self._auto_objects):
            if det_idx in removed or geom is None or geom.isEmpty():
                continue
            if self._passes_review_filters(score, area, params):
                out.append(det_idx)
        return out

    def _build_shape_hit_index(self) -> None:
        """Build the per-gesture hit-test cache: snapshot the review params
        (a preset regex rebuild) and a QgsSpatialIndex over every object's
        bbox, so a click narrows to a few candidates instead of scanning all
        objects with a full GEOS op. Best-effort: on any failure the hit tests
        fall back to the full scan."""
        self._shape_review_params = self._widget_review_params()
        self._shape_hit_index = None
        self._shape_hit_geoms = {}
        try:
            from qgis.core import QgsFeature, QgsSpatialIndex
            index = QgsSpatialIndex()
            review = self._auto_review or {}
            review_geoms = review.get("geoms") or []
            review_ids = review.get("ids")
            by_display_id = {
                self._object_fid_for(det_idx): det_idx
                for det_idx in range(len(self._auto_objects))
            }
            visible = []
            if isinstance(review_ids, list) and len(review_ids) == len(review_geoms):
                for geom, det_id in zip(review_geoms, review_ids):
                    det_idx = by_display_id.get(det_id)
                    if det_idx is not None:
                        visible.append((det_idx, geom))
            if not visible:
                visible = [(det_idx, row[0])
                           for det_idx, row in enumerate(self._auto_objects)]
            for det_idx, geom in visible:
                if geom is None or geom.isEmpty():
                    continue
                feat = QgsFeature(det_idx)
                feat.setGeometry(QgsGeometry(geom))
                index.addFeature(feat)
                self._shape_hit_geoms[det_idx] = QgsGeometry(geom)
            self._shape_hit_index = index
        except (RuntimeError, AttributeError, TypeError, ImportError):
            self._shape_hit_index = None
            self._shape_hit_geoms = {}

    def _shape_hit_candidates(self, rect) -> list[int]:
        """Object indices whose bbox intersects ``rect`` (a QgsRectangle), via
        the spatial index when it built, else every visible index."""
        index = self._shape_hit_index
        if index is not None:
            try:
                return [int(i) for i in index.intersects(rect)]
            except (RuntimeError, AttributeError, TypeError):
                pass
        return self._visible_object_indices()

    def _hit_object_at(self, point) -> int | None:
        """The smallest visible detection containing ``point``, or None.

        Smallest wins so a shape sitting inside another (a roof section on a
        block) is still reachable by clicking it.
        """
        # A few-pixel tolerance around the click, carried into the run CRS by
        # transforming the click AND a point one tolerance to its east, so a
        # thin or small detection is reachable even when the click lands just
        # off it. Zero tolerance (no canvas) still works via strict contains.
        tol = 0.0
        try:
            mupp = self.iface.mapCanvas().mapSettings().mapUnitsPerPixel()
            tol_canvas = mupp * self._PICK_TOLERANCE_PX
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
            # A geometry cached from review["geoms"] is already the exact
            # visible result after all current filters. Its canonical source may
            # have a different Manual-edited area, so re-filtering it here can
            # make a visibly rendered shape impossible to click.
            if det_idx not in self._shape_hit_geoms and not self._passes_review_filters(
                    score, area_m2, params):
                continue
            try:
                if geom.contains(probe):
                    area = float(geom.area())
                    if best is None or area < best_area:
                        best, best_area = det_idx, area
                    continue
                # Not inside: keep the nearest one within the tolerance, used
                # only when nothing strictly contains the click.
                if best is None and tol > 0:
                    dist = float(geom.distance(probe))
                    if dist <= near_dist:
                        near, near_dist = det_idx, dist
            except (RuntimeError, TypeError, ValueError):
                continue
        return best if best is not None else near

    def _points_in_run_crs(self, points) -> list:
        """Canvas map points -> QgsPointXY in the run CRS (the CRS every stored
        object lives in). Mirrors _correct_rect_in_run_crs for points."""
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
        """Outline the picked detections in the amber "being edited now"
        colour. Rebuilt from the pick set each time, so an un-pick clears its
        own outline and nothing can drift."""
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
        """An outline-only rubber band over one detection. It sits ON TOP of
        whatever renderer the review uses, so the highlight reads the same in
        every display mode."""
        try:
            from qgis.gui import QgsRubberBand

            from ...core.qt_compat import PolygonGeometry
            band = QgsRubberBand(self.iface.mapCanvas(), PolygonGeometry)
            band.setToGeometry(QgsGeometry(geom), None)
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
