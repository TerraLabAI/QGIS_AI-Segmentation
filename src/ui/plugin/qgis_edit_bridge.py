"""The native geometry bridge behind Correct's "Edit by hand" action.

It runs on the detection the user SELECTED on Correct. One click turns the
review's selection layer into an editable QGIS layer, holds the session to that
one object (see plugin/bridge_isolation.py), selects it so native Reshape and
Split scope to it, frames it when it is off-screen or too small to work on, and
arms the vertex tool. The dock exposes Move points, Redraw edge and Split as
focused buttons backed by QGIS's native actions, each with the exact gesture
spelled out. "Done editing" commits, restores the user's prior editing aids,
folds the committed geometry back into the review, and re-selects the same
object by its stable det_id.

Snapping is ON; topological editing and avoid-overlap are deliberately OFF.
On a dense layer of touching detections avoid-overlap silently carves an edit
that grazes a neighbour (GEOS difference; an edit fully covered by neighbours is
rejected outright, and a split result keeps only the largest part), which reads
as "editing is broken". Topological editing goes the other way and writes the
edit THROUGH the shared border: moving or deleting a vertex takes the
coincident vertices of the touching polygons with it. A coherent shared border
is not worth a silent change to a polygon the user did not pick, so the bridge
neutralises both and restores the user's own settings on exit. Every editing aid
the bridge changes is snapshotted on entry and restored on every exit path,
including cancel and error.

This borrows QGIS's mature geometry engine instead of rebuilding vertex / reshape
/ split ourselves, and it retires our own failing split map tool: the review
selection layer is already a normal editable ``QgsVectorLayer``, so every native
digitizing tool runs on it directly.

Seam with plan 1 (the Correct-page owner). This mixin only CALLS what plan 1
provides, always guarded, so the engine runs and tests standalone before plan 1
lands:

  - ``dock.enter_qgis_bridge_state()`` / ``dock.leave_qgis_bridge_state()`` --
    the anchor banner ("Editing in QGIS - Done editing").
  - ``self._fold_qgis_edits_back(layer)`` -- rebuild ``_auto_objects`` and refresh
    the review from the committed layer. Until it exists, a local fallback here
    keeps the review consistent.
  - dock signals ``auto_edit_in_qgis_requested`` (entry) and
    ``auto_qgis_bridge_done_requested`` (Done). Connected in the plugin; every
    entry/finish is guarded so a double connect (plan 1 also wiring them) is a
    harmless no-op.

The #1 risk is a leaked global editing aid: the user's project left with our
snapping / topology / avoid-overlap forced on. ``_restore_bridge_editing_aids``
runs on EVERY exit path (Done, editing toggled off in QGIS, Exit review, mode
switch, unload) and must never raise.
"""
from __future__ import annotations

import time

from ...core.i18n import tr
from ...core.qt_compat import QAction

# Snap tolerance in screen pixels, so it feels right at any imagery zoom (map
# units would drift as the user zooms the raster). Verified live at QGIS 3.44.
_SNAP_TOLERANCE_PX = 12

# How often the in-flight gesture is sampled. QGIS exposes NO signal per capture
# point (verified against the tool metaobjects on 3.44), so a short poll is the
# only way to tell the user how many points they have placed.
_BRIDGE_POLL_MS = 200

# Capture tools by the C++ class name, used only to WORD the result of a
# finished gesture (each one fails for its own reason, so each gets its own
# sentence). type() is useless here: PyQGIS downcasts them all to
# QgsMapToolCapture, so only metaObject().className() tells them apart.
# Whether a tool is capturing AT ALL is decided by _bridge_capture_points
# below, which asks the tool instead of matching a name.
_SPLIT_TOOL_CLASS = "QgsMapToolSplitFeatures"
_ADD_TOOL_CLASSES = ("QgsMapToolAddFeature", "QgsMapToolDigitizeFeature")

# The QGIS vertex ("node") tool, by C++ class name across versions. Not in the
# public PyQGIS API, so it is only ever recognised by className(), never
# imported or subclassed.
_VERTEX_TOOL_CLASSES = ("QgsVertexTool", "QgsVertexToolV2", "QgsMapToolVertexEdit")

# The three tools that edit the shape under the cursor. Each needs a polygon the
# session is held to; Add does not, because what it draws is new.
_BRIDGE_SHAPE_TOOLS = ("vertex", "reshape", "split")


def _bridge_capture_points(tool, class_name: str):
    """Points placed so far in the tool's open capture line, or None when the
    tool is not capturing one.

    Asks the capability rather than matching a class name. A capture tool
    exposes ``size()``; the vertex tool does not, and it is excluded by name
    as well so a future QGIS growing that method there cannot turn a corner
    drag into a phantom open line. Matching names alone is what let Add Feature
    slip through: it landed after the list was written, so its line never
    counted as open, "Finish the line" never replaced Save, and Save committed
    an empty buffer over the corners the user had just placed.
    """
    if tool is None or class_name in _VERTEX_TOOL_CLASSES:
        return None
    size = getattr(tool, "size", None)
    if not callable(size):
        return None
    try:
        return int(size())
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None


def _snap_mode_all_layers():
    """QgsSnappingConfig "snap to all layers" mode, across QGIS 3.22 -> 4.

    QGIS >= 3.26 scopes it as ``SnappingMode.AllLayers``; older builds expose it
    flat as ``QgsSnappingConfig.AllLayers``. Resolved by string so the static
    Qt6 checker sees no flat enum access."""
    from qgis.core import QgsSnappingConfig
    scope = getattr(QgsSnappingConfig, "SnappingMode", None)
    if scope is not None:
        val = getattr(scope, "AllLayers", None)
        if val is not None:
            return val
    return getattr(QgsSnappingConfig, "AllLayers", None)


def _snap_type_flags():
    """Vertex|Segment snapping flags, across QGIS 3.22 -> 4.

    QGIS >= 3.26: ``SnappingTypes.VertexFlag | SnappingTypes.SegmentFlag``.
    QGIS 3.22: the flat ``QgsSnappingConfig.Vertex | .Segment`` (different member
    names, so the qt_compat ``resolve_qt_enum`` -- which reuses one name -- cannot
    bridge them). Returns None if neither shape resolves."""
    from qgis.core import QgsSnappingConfig
    scope = getattr(QgsSnappingConfig, "SnappingTypes", None)
    if scope is not None:
        vtx = getattr(scope, "VertexFlag", None)
        seg = getattr(scope, "SegmentFlag", None)
        if vtx is not None and seg is not None:
            return vtx | seg
    vtx = getattr(QgsSnappingConfig, "Vertex", None)
    seg = getattr(QgsSnappingConfig, "Segment", None)
    if vtx is not None and seg is not None:
        return vtx | seg
    return None


def _tolerance_pixels_unit():
    """QgsTolerance pixel unit, across QGIS 3.22 -> 4.

    QGIS >= 3.26 scopes it as ``UnitType.Pixels``; older builds expose the flat
    ``QgsTolerance.Pixels``."""
    from qgis.core import QgsTolerance
    scope = getattr(QgsTolerance, "UnitType", None)
    if scope is not None:
        val = getattr(scope, "Pixels", None)
        if val is not None:
            return val
    return getattr(QgsTolerance, "Pixels", None)


def flags_without(flags, flag):
    """``flags`` with ``flag`` cleared, rebuilt as the type the setter accepts.

    On the Qt6 bindings a layer flag is a real enum member: ``~flag`` is a plain
    int, so ``flags & ~flag`` degrades to an int and ``setFlags`` refuses it with
    a TypeError. Masking the two as ints and rebuilding the flags object keeps
    one expression working on QGIS 3 (where they already are ints) and QGIS 4.
    """
    try:
        masked = int(flags) & ~int(flag)
    except (TypeError, ValueError):
        return flags
    try:
        return type(flags)(masked)
    except (TypeError, ValueError):
        return masked


def _avoid_mode(name: str):
    """A ``Qgis.AvoidIntersectionsMode`` member by name, or None.

    ``name`` is "AvoidIntersectionsLayers" (on) or "AllowIntersections" (off).
    Resolved by string; the enum is scoped on every QGIS that has it."""
    from qgis.core import Qgis
    scope = getattr(Qgis, "AvoidIntersectionsMode", None)
    if scope is None:
        return None
    return getattr(scope, name, None)


class QgisEditBridgeMixin:
    """Arms QGIS's native digitizing on the review layer and folds edits back.

    Part of AISegmentationPlugin (see ai_segmentation_plugin.py); a mixin like
    its siblings. State lives on the plugin instance; the banner widget lives on
    the dock (built by plan 1). All method names here are unique to this mixin
    (``enter_qgis_edit_bridge`` / ``finish_qgis_edit_bridge`` / ``_bridge_*`` /
    ``_qgis_bridge_*``), so it never shadows another mixin in the MRO."""

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _init_qgis_bridge_state(self) -> None:
        """Fresh bridge state; called once from the plugin __init__."""
        self._qgis_bridge_active = False
        # A reentrancy guard: commitChanges()/rollBack() fire editingStopped,
        # which would re-enter teardown without it.
        self._qgis_bridge_finishing = False
        # True while a teardown queued by QGIS's own editingStopped is waiting
        # for its turn of the event loop. Cleared ONLY by the deferred call
        # itself, so at most one is ever in flight; clearing it from a teardown
        # would let a second one land on a session that opened meanwhile.
        self._qgis_bridge_stop_queued = False
        # True only while the synthetic right-click closes an open capture line.
        # It keeps a deferred identity write out of the capture tool's own
        # release handler, without silencing the featureAdded that same call
        # produces.
        self._qgis_bridge_closing_capture = False
        self._qgis_bridge_layer = None
        # Prior project editing aids, saved on enter and restored on every exit.
        self._qgis_bridge_saved_aids: dict | None = None
        # The map tool active before we armed the vertex tool.
        self._qgis_bridge_prev_maptool = None
        # True once layer.editingStopped is connected, so teardown knows to drop
        # it before touching the edit buffer.
        self._qgis_bridge_editing_conn = False
        self._qgis_bridge_t0 = 0.0
        # The review layer is deliberately Private during ordinary review, but
        # QGIS refuses to make a Private layer the active digitizing layer.
        # Keep its original flags so the bridge can temporarily expose it and
        # restore the quiet review presentation on every exit path.
        self._qgis_bridge_layer_flags = None
        # Exact geometry by det_id when native editing begins. The fold-back
        # uses this to distinguish an untouched visible feature from a vertex
        # edit, split, merge or deletion, while retaining detections hidden by
        # the current review filters.
        self._qgis_bridge_snapshot: dict[int, bytes] = {}
        # The vertex tool may reveal QGIS's Vertex Editor dock. It is redundant
        # with the focused controls in our own dock, so preserve its prior
        # visibility and hide only a panel our bridge opened.
        self._qgis_bridge_vertex_dock_visibility: dict[int, tuple[object, bool]] = {}
        # The detection the user selected on Correct when they opened the bridge
        # (canonical _auto_objects index). We select + frame + highlight it so
        # native editing has a clear target, and re-select it on the way back.
        self._qgis_bridge_target_idx: int | None = None
        # The target's stable display identity (det_id), captured on enter. Used
        # to re-select the SAME object after fold-back, which reorders indices.
        self._qgis_bridge_target_det_id: int | None = None
        # The prior project "search radius for vertex edits". A value of 0 makes
        # the native vertex tool feel dead (it cannot decide which vertex a click
        # means), so we guarantee a workable radius during the bridge and restore
        # whatever the user had on exit.
        self._qgis_bridge_saved_search_radius: tuple | None = None
        # The canvas selection colour before the bridge. The target has to be
        # SELECTED for the native tools to scope to it, but the flat selection
        # fill hides the review render the user is editing against, so the
        # colour goes transparent for the session and comes back on exit.
        self._qgis_bridge_saved_selection_colour = None
        # True once the layer edit-buffer signals are connected, so the live
        # feedback line can be driven and dropped cleanly on teardown.
        self._qgis_bridge_feedback_conn = False
        # True while the project's layer-removal signal is watched. The session
        # strips the review layer's Private flag, so the user can see it in the
        # Layers panel and remove it; without this watch the bridge would stay
        # open on a layer that no longer exists.
        self._qgis_bridge_layer_watch = False
        # True when the subset holding the session to ONE polygon went on. False
        # means the session runs on the whole layer, which is only allowed when
        # it has no polygon of its own.
        self._qgis_bridge_isolated = False
        # The edited layer's QUndoStack. It names the operation QGIS actually
        # recorded ("Moved vertex", "Split features"), which is the only honest
        # way to report a native edit, and it backs the panel's Undo.
        self._qgis_bridge_undo_stack = None
        # In-flight gesture sampling: points placed at the last tick, and the
        # undo depth when the gesture began (to spot a gesture that did nothing).
        self._qgis_bridge_poll_timer = None
        self._qgis_bridge_prev_points = 0
        self._qgis_bridge_gesture_undo_count = None
        # The Points dial thins the target from its pre-edit snapshot, so it has
        # to retire once the user edits the shape by hand (or it would wipe their
        # corners). _dial_edit guards the dial's own changeGeometry so its
        # signal is not mistaken for a hand edit.
        self._qgis_bridge_hand_edited = False
        self._qgis_bridge_dial_edit = False
        # Features born inside this session (a polygon drawn by hand, the half a
        # native split creates) still waiting for their final det_id.
        self._qgis_bridge_pending_fids: list[int] = []
        # det_ids this session has already handed out. They are not in the review
        # model yet (the fold runs on Save), so nothing else knows they are taken.
        self._qgis_bridge_born_det_ids: set[int] = set()
        # Undo-stack positions of our own identity writes. Assigning an id is not
        # a step the user took, so the panel's Undo steps over it.
        self._qgis_bridge_id_write_marks: set[int] = set()
        # True while an identity is being written, so the panel's "last change"
        # line keeps naming the edit the user made.
        self._qgis_bridge_id_edit = False

    # ------------------------------------------------------------------
    # Public API (plan 1's slot delegates here)
    # ------------------------------------------------------------------

    def enter_qgis_edit_bridge(self) -> None:
        """Turn the review selection layer into an editable QGIS layer with the
        digitizing aids on, reveal the toolbars, arm the vertex tool, and show
        the dock banner. A no-op if the bridge is already open or there is no
        editable review layer."""
        if getattr(self, "_qgis_bridge_active", False):
            return
        layer = self._resolve_bridge_layer()
        if layer is None:
            return
        # Capture the polygon the user picked on Correct BEFORE
        # _disarm_shape_tool clears it. It decides where the map looks, what the
        # banner is called, which object the session may touch, and which one
        # the review re-selects on the way back. Entering with nothing picked is
        # a supported, ordinary case: there is no polygon to hold the session
        # to, so the whole layer stays reachable.
        self._qgis_bridge_target_idx = getattr(self, "_correct_selected_idx", None)
        # A hand edit (Merge / Split pick) still armed would fight the QGIS tool
        # we are about to set; drop it first (no-op if nothing is armed).
        try:
            self._disarm_shape_tool()
        except (RuntimeError, AttributeError):
            pass
        # _disarm_shape_tool correctly drops the plugin-side selection before
        # native editing, but the review card is about to be hidden. Clear its
        # visible selected state too, so returning from QGIS never leaves a
        # clickable-looking Reshape/Remove card with no selected object behind
        # it.
        try:
            self.dock_widget.set_correct_selection(0)
        except (RuntimeError, AttributeError):
            pass

        canvas = self.iface.mapCanvas()
        self._qgis_bridge_prev_maptool = canvas.mapTool()
        self._save_bridge_editing_aids()
        if not self._expose_and_activate_bridge_layer(layer):
            self._restore_bridge_layer_presentation(layer)
            self._qgis_bridge_layer_flags = None
            self._restore_bridge_setting(self._restore_bridge_editing_aids)
            self._qgis_bridge_saved_aids = None
            self._qgis_bridge_prev_maptool = None
            self._show_bridge_unavailable()
            self._rearm_correct_select_after_bridge_bail()
            return
        # Reset BEFORE the isolate call below. The ceiling that call computes
        # reads the ids this session has handed out, so ids left by the previous
        # one would push it above every id this one can mint, and a shape drawn
        # here would fall outside the subset and be lost at Save.
        self._qgis_bridge_pending_fids = []
        self._qgis_bridge_born_det_ids = set()
        self._qgis_bridge_id_write_marks = set()
        # The session's own polygon, resolved HERE rather than in
        # _select_and_frame_bridge_target below: the subset that holds the
        # session to it has to be on the layer before startEditing(), which is
        # the next call, and a layer already in edit mode refuses one.
        self._qgis_bridge_target_det_id = self._bridge_target_det_id()
        self._qgis_bridge_isolated = bool(
            self._isolate_bridge_target(layer, self._qgis_bridge_target_det_id))
        if (self._qgis_bridge_target_det_id is not None
                and not self._qgis_bridge_isolated):
            # Nothing holds the session to its polygon. The vertex tool scopes to
            # the LAYER, so a click could take a neighbour's vertex and the fold
            # would write that neighbour back as a real edit, with nothing on
            # screen saying so. Refuse the session instead.
            self._clear_bridge_isolation()
            self._restore_bridge_layer_presentation(layer)
            self._qgis_bridge_layer_flags = None
            self._restore_bridge_setting(self._restore_bridge_editing_aids)
            self._qgis_bridge_saved_aids = None
            self._qgis_bridge_prev_maptool = None
            self._qgis_bridge_target_det_id = None
            self._show_bridge_isolation_failed()
            self._rearm_correct_select_after_bridge_bail()
            return
        if not self._apply_bridge_editing_config(layer):
            # Could not start editing: restore whatever we saved and bail, so we
            # never leave half the aids on, nor a layer showing one polygon.
            self._clear_bridge_isolation()
            self._restore_bridge_layer_presentation(layer)
            self._qgis_bridge_layer_flags = None
            self._restore_bridge_setting(self._restore_bridge_editing_aids)
            self._qgis_bridge_saved_aids = None
            self._qgis_bridge_prev_maptool = None
            # No session, so no target: the Delete row reads this id and would
            # otherwise act on a polygon whose session never opened.
            self._qgis_bridge_target_det_id = None
            self._rearm_correct_select_after_bridge_bail()
            return

        # Taken UNDER the subset, so it holds the session's own polygon and
        # nothing else. That is what lets the fold treat an untouched neighbour
        # as untouched instead of re-basing it.
        self._qgis_bridge_snapshot = self._snapshot_bridge_layer(layer)

        self._qgis_bridge_layer = layer
        self._qgis_bridge_active = True
        self._qgis_bridge_finishing = False
        self._qgis_bridge_id_edit = False
        self._qgis_bridge_t0 = time.monotonic()

        self._remember_bridge_vertex_editor_visibility()
        # Move points, Redraw edge and Split each edit the shape under the
        # cursor. A session with no polygon of its own has nothing holding them
        # to one, so they leave the panel and Add is the only lane.
        has_target = self._qgis_bridge_target_det_id is not None
        self._set_bridge_shape_tools_visible(has_target)
        if has_target:
            self.activate_qgis_bridge_tool("vertex")
        # Sync our banner if the user ends editing directly from QGIS's own
        # toggle (which commits or rolls back on its own prompt).
        try:
            layer.editingStopped.connect(self._on_qgis_bridge_editing_stopped)
            self._qgis_bridge_editing_conn = True
        except (RuntimeError, AttributeError, TypeError):
            self._qgis_bridge_editing_conn = False

        self._enter_bridge_banner()
        # Select + frame the target so native editing has an unmistakable anchor,
        # then wire the live feedback line. Both are best-effort: a failure here
        # never leaves the bridge half-open (editing is already armed above).
        self._select_and_frame_bridge_target(layer)
        # A fresh session has not been hand-edited yet, so the Points dial is
        # live. Reveal it only when a single target polygon resolved: a
        # whole-layer session has nothing to thin.
        self._qgis_bridge_hand_edited = False
        self._qgis_bridge_dial_edit = False
        dock = getattr(self, "dock_widget", None)
        try:
            reset = getattr(dock, "reset_qgis_bridge_points", None)
            if callable(reset):
                reset()
            show = getattr(dock, "set_qgis_bridge_points_visible", None)
            if callable(show):
                show(self._qgis_bridge_target_det_id is not None)
            # Delete needs the same target, and unlike Points it stays up for
            # the whole session: a hand edit is where a user finds out the
            # shape is not worth keeping.
            show_delete = getattr(dock, "set_qgis_bridge_delete_visible", None)
            if callable(show_delete):
                show_delete(self._qgis_bridge_target_det_id is not None)
        except (RuntimeError, AttributeError, TypeError):
            pass
        self._connect_bridge_feedback(layer)
        self._connect_bridge_layer_watch()
        self._connect_bridge_tool_messages()
        self._start_bridge_gesture_poll()
        self._track_bridge("opened")

    def _bridge_target_det_id(self):
        """The stable identity of the polygon the session opened on, or None.

        Read from the id list DIRECTLY. ``_object_fid_for`` falls back to
        returning the index when that list is missing or short, and indices and
        det_ids are different id spaces (det_id is the merger's keeper fid), so
        the fallback could silently name a DIFFERENT detection than the user
        picked. No usable id list means no target, which is honest.
        """
        idx = getattr(self, "_qgis_bridge_target_idx", None)
        if idx is None:
            return None
        try:
            ids = getattr(self, "_auto_object_fids", None) or []
            if idx < 0 or idx >= len(ids):
                return None
            return ids[idx]
        except (RuntimeError, AttributeError, TypeError):
            return None

    def _select_and_frame_bridge_target(self, layer) -> None:
        """Select and frame the polygon the user came in with.

        The session works on ONE polygon. The subset applied on entry is what
        makes that true for the vertex tool, which scopes to the active LAYER
        and would otherwise let a click grab a neighbour's vertex; the selection
        set here is what makes it true for native Reshape and Split, which act
        only on selected features. Any selection already on the layer is
        dropped, so a stale one cannot widen the session.

        The entry polygon also decides where the map looks and what the banner
        is called, and its ``det_id`` is kept so the review can re-select it on
        the way back. Opened with nothing picked, nothing is selected and
        nothing is isolated: there is no polygon to hold the session to.
        """
        det_id = self._bridge_target_det_id()
        self._qgis_bridge_target_det_id = det_id
        try:
            layer.removeSelection()
        except (RuntimeError, AttributeError):
            pass
        if det_id is None:
            # Opened with nothing picked: an ordinary case, and the banner says
            # the tools work on the layer rather than naming a polygon.
            self._set_bridge_target_label(None)
            return
        feature = self._bridge_feature_for_det_id(
            layer, det_id, with_geometry=True)
        if feature is None:
            return
        try:
            layer.selectByIds([feature.id()])
        except (RuntimeError, AttributeError, TypeError):
            pass
        target_geom = feature.geometry()
        if target_geom is not None and not target_geom.isEmpty():
            self._frame_bridge_target(layer, target_geom)
        # Name the target in the banner so the user reads which object is live.
        self._set_bridge_target_label(self._qgis_bridge_target_idx)

    def _frame_bridge_target(self, layer, geom) -> None:
        """Bring the entry polygon into view ONLY when none of it is on screen.

        A session opens from a click on the polygon itself, so the user is
        already looking at it. Moving the map then slides the object out from
        under the cursor for no gain, which is what the old "inside the view
        AND at least 90 px across" rule did on every ordinary click. A view
        change is only justified when the target is nowhere on screen (the dock
        button can open a session on a polygon the user has since panned away
        from). Zooming for a small object is the user's call, not ours.

        Best-effort; a transform or extent failure leaves the view as it is.
        """
        try:
            from qgis.core import (
                QgsCoordinateTransform,
                QgsProject,
                QgsRectangle,
            )
            canvas = self.iface.mapCanvas()
            settings = canvas.mapSettings()
            bbox = geom.boundingBox()
            layer_crs = layer.crs()
            canvas_crs = settings.destinationCrs()
            if layer_crs.authid() and canvas_crs.authid() != layer_crs.authid():
                xform = QgsCoordinateTransform(
                    layer_crs, canvas_crs, QgsProject.instance())
                bbox = xform.transformBoundingBox(bbox)
            if bbox.isEmpty():
                return
            if canvas.extent().intersects(bbox):
                return
            pad = max(bbox.width(), bbox.height()) * 1.5 or 1.0
            framed = QgsRectangle(
                bbox.xMinimum() - pad, bbox.yMinimum() - pad,
                bbox.xMaximum() + pad, bbox.yMaximum() + pad)
            canvas.setExtent(framed)
            canvas.refresh()
        except (RuntimeError, AttributeError, TypeError, ImportError):
            pass

    def _set_bridge_target_label(self, idx: int | None) -> None:
        """Tell the dock which object the manual edit is working on, by class."""
        label = ""
        try:
            if idx is not None:
                review = getattr(self, "_auto_review", None) or {}
                label = str(review.get("prompt") or "").strip()
        except (RuntimeError, AttributeError, TypeError):
            label = ""
        dock = getattr(self, "dock_widget", None)
        fn = getattr(dock, "set_qgis_bridge_target", None)
        if callable(fn):
            try:
                fn(label)
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _set_bridge_shape_tools_visible(self, visible: bool) -> None:
        """Show or hide Move points, Redraw edge and Split on the session panel.

        Hidden, never greyed: a session with no polygon of its own cannot hold
        them to one shape, so they are not a choice it offers. Goes through the
        dock setter when the build has one, and drives the panel's own buttons
        otherwise. Never raises."""
        dock = getattr(self, "dock_widget", None)
        if dock is None:
            return
        setter = getattr(dock, "set_qgis_bridge_tools_visible", None)
        if callable(setter):
            try:
                setter(bool(visible))
            except (RuntimeError, AttributeError, TypeError):
                pass
            return
        buttons = getattr(dock, "_qgis_bridge_tool_buttons", None) or {}
        for key in _BRIDGE_SHAPE_TOOLS:
            button = buttons.get(key)
            if button is None:
                continue
            try:
                button.setVisible(bool(visible))
            except (RuntimeError, AttributeError):
                pass

    def _rearm_correct_select_after_bridge_bail(self) -> None:
        """Put Correct's resting select tool back after a failed entry.

        Entry disarms it before QGIS takes the canvas. Without this the step
        still reads "Click a polygon" while no click selects one, and only
        leaving the step and coming back arms it again."""
        try:
            self._arm_correct_select()
        except (RuntimeError, AttributeError):
            pass

    def activate_qgis_bridge_tool(self, tool: str) -> None:
        """Activate one native QGIS geometry tool from the focused dock.

        Vertex and Split are public QgisInterface actions. Reshape has no
        QgisInterface getter, so it is resolved by QGIS's stable QAction object
        name. Missing actions fail quietly and leave the current tool active.
        """
        if not getattr(self, "_qgis_bridge_active", False):
            return
        layer = self._qgis_bridge_layer
        if layer is None or not self._is_layer_valid(layer):
            return
        tool = str(tool).lower()
        if (tool in _BRIDGE_SHAPE_TOOLS
                and getattr(self, "_qgis_bridge_target_det_id", None) is None):
            # A session with no polygon of its own runs on the whole layer, so
            # these three would edit any shape the cursor lands on. Refused here
            # as well as hidden, because the dock is not the only caller.
            return
        try:
            self.iface.setActiveLayer(layer)
        except (RuntimeError, AttributeError):
            pass

        action = None
        if tool == "vertex":
            action = self._bridge_iface_action(
                "actionVertexToolActiveLayer", "actionVertexTool")
        elif tool == "split":
            action = self._bridge_iface_action("actionSplitFeatures")
        elif tool == "reshape":
            action = self._bridge_named_action("mActionReshapeFeatures")
        elif tool == "add":
            # Native Add Feature capture: click each corner, right-click to
            # finish. It stays armed for the next polygon, so several missed
            # objects can be drawn before Done. QgisInterface exposes it
            # directly; the named-action fallback covers releases that omit the
            # getter.
            action = (self._bridge_iface_action("actionAddFeature") or self._bridge_named_action("mActionAddFeature"))
        if action is None:
            return
        try:
            action.trigger()
        except (RuntimeError, AttributeError):
            return
        # QGIS keeps ONE instance of each capture tool for the whole session,
        # and neither deactivate() nor a fresh trigger() empties its point
        # buffer. A line the user abandoned by picking another tool button is
        # still in there, so the next line is APPENDED to it: the two strokes
        # cross, GEOS refuses a split line that is not simple and a reshape line
        # that meets the outline four times, and both tools report that nothing
        # happened for the rest of the QGIS session. Arm on an empty buffer.
        self._bridge_cancel_capture()
        try:
            self.dock_widget.set_qgis_bridge_tool(tool)
        except (RuntimeError, AttributeError):
            pass
        if tool == "vertex":
            self._hide_bridge_opened_vertex_editors()
            try:
                from qgis.PyQt.QtCore import QTimer
                QTimer.singleShot(0, self._hide_bridge_opened_vertex_editors)
            except (RuntimeError, AttributeError):
                pass

    def _on_add_polygon_requested(self) -> None:
        """Correct step 'Add a polygon': arm QGIS's native Add Feature capture on
        the review layer so the user can draw an object the AI missed. Enters the
        digitizing bridge first if it is not already open (Add a polygon is a
        zone-level act, exactly like Edit manually), then arms the capture tool.
        The drawing is given its det_id the moment it lands, then folds back
        through the normal bridge fold (score 1.0, run class at export), and is
        flagged manual so no filter can hide it. A no-op when there is no
        editable review layer."""
        if not getattr(self, "_qgis_bridge_active", False):
            self.enter_qgis_edit_bridge()
        if not getattr(self, "_qgis_bridge_active", False):
            return  # no editable layer / entry failed; enter already reported it
        self.activate_qgis_bridge_tool("add")

    def _bridge_iface_action(self, *method_names):
        """Return the first available QAction exposed by QgisInterface."""
        for method_name in method_names:
            getter = getattr(self.iface, method_name, None)
            if getter is None:
                continue
            try:
                action = getter()
                if action is not None:
                    return action
            except (RuntimeError, AttributeError, TypeError):
                pass
        return None

    def _bridge_named_action(self, object_name: str):
        """Resolve a native QGIS QAction by its stable object name."""
        try:
            return self.iface.mainWindow().findChild(QAction, object_name)
        except (RuntimeError, AttributeError, TypeError):
            return None

    def finish_qgis_edit_bridge(self, commit: bool = True) -> None:
        """Done editing: commit (or roll back), restore the user's prior editing
        aids, and fold the committed geometry back into the review. On a commit
        failure the edit session is KEPT open with an error banner, so the user
        can fix and retry (the aids stay on)."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        if getattr(self, "_qgis_bridge_finishing", False):
            return
        if commit:
            # The synthetic right-click inside _close_open_bridge_capture runs
            # the capture tool's own release handler, and a deferred identity
            # write queued a moment earlier can land in the middle of it. Hold
            # this guard across the call so that write waits for the settled
            # pass right below, which runs on our own stack. NOT
            # _qgis_bridge_finishing: that one also silences featureAdded, and
            # the polygon Save itself closes is born inside this very call and
            # has to reach the queue.
            self._qgis_bridge_closing_capture = True
            try:
                self._close_open_bridge_capture()
            finally:
                self._qgis_bridge_closing_capture = False
            # A polygon that Save itself closes is born in this same turn of the
            # event loop, so its deferred identity write has not run yet. Settle
            # every pending id before the commit, or the fold mints a different
            # one and the shape changes colour the moment it is saved.
            self._assign_bridge_born_det_ids()
        layer = self._qgis_bridge_layer
        if commit and layer is not None and self._is_layer_valid(layer):
            try:
                editable = layer.isEditable()
            except (RuntimeError, AttributeError):
                editable = False
            # ``editingStopped`` is emitted synchronously by many QGIS
            # providers. Keep our bridge signal handler out of that nested
            # callback: this method owns the failure message and then performs
            # the single, orderly teardown below.
            committed = True
            commit_raised = False
            if editable:
                self._qgis_bridge_finishing = True
                try:
                    committed = bool(layer.commitChanges())
                except (RuntimeError, AttributeError):
                    commit_raised = True
                finally:
                    self._qgis_bridge_finishing = False
            if commit_raised:
                # The layer died between the validity check above and the call
                # (a dead sip wrapper raises RuntimeError). Never let that
                # escape into the Qt slot: the teardown below is the only thing
                # that gives the user back their snapping, topological editing,
                # avoid-overlap, vertex search radius and selection colour, and
                # skipping it leaves the project with all of them forced on.
                self._teardown_qgis_edit_bridge(commit=False, external=True)
                return
            if not committed:
                errors = []
                try:
                    errors = list(layer.commitErrors() or [])
                except (RuntimeError, AttributeError):
                    pass
                # Keep editing: the aids stay on and the banner reports the
                # problem so the user can fix the geometry and click Done again.
                self._show_bridge_commit_error(errors)
                return
        self._teardown_qgis_edit_bridge(commit=commit, external=False)

    def delete_bridge_target_polygon(self) -> None:
        """The Manual session's Delete row: drop the polygon under edit.

        Deleting beats the edit in progress, so the session ends WITHOUT saving
        it: the ordinary rollback exit, which restores every editing aid and
        skips the fold-back. The removal itself then runs through
        ``_remove_detection_index``, the same primitive the AI panel's Delete
        row uses, so the correction journal, Undo last and the status line
        behave identically whichever tab the user was on.

        The target's stable det_id is read BEFORE the teardown, which clears it
        and the selection. Indices are untouched here (no fold-back on a
        rollback), but the id is resolved back to an index anyway, because that
        is the only mapping that stays honest if the model moves."""
        det_id = getattr(self, "_qgis_bridge_target_det_id", None)
        if getattr(self, "_qgis_bridge_active", False):
            self.finish_qgis_edit_bridge(commit=False)
        if det_id is None:
            return
        idx = self._object_index_for_det_id(det_id)
        if idx is None:
            return
        self._remove_detection_index(idx)

    # ------------------------------------------------------------------
    # Exit-path safety (the #1 risk: a leaked global editing aid)
    # ------------------------------------------------------------------

    def _abort_qgis_edit_bridge_if_active(self) -> None:
        """Roll back and restore, guaranteed. Called from every teardown path
        (Exit review, mode switch, new run, unload) BEFORE the selection layer
        is removed. Idempotent and must never raise into unload / reset."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        try:
            self._teardown_qgis_edit_bridge(commit=False, external=False)
        except Exception as exc:  # noqa: BLE001 -- teardown must never break unload/reset
            # Say it once, in the log panel a bug report carries. Without this
            # the root failure is invisible and only its symptoms are reported.
            self._log_bridge_failure("qgis_bridge_teardown", exc)
            # Last-resort: force the aids back even if the structured teardown
            # threw, so the user's project is never left with our aids on.
            try:
                # Before the aids, because this is the one leak the user cannot
                # work around: a subset left on shows the review a single
                # polygon, with no control anywhere to take it off.
                self._clear_bridge_isolation()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_editing_aids()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_vertex_search_radius()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_selection_colour()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_attribute_form(self._qgis_bridge_layer)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_vertex_editor_visibility()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                # The poll would otherwise run until its next tick noticed the
                # bridge was gone. Self-healing, but this emergency path exists
                # precisely to leave nothing running.
                self._stop_bridge_gesture_poll()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._disconnect_bridge_layer_watch()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._set_bridge_shape_tools_visible(True)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            self._qgis_bridge_active = False
            self._qgis_bridge_finishing = False
            self._qgis_bridge_isolated = False
            self._qgis_bridge_layer = None
            self._qgis_bridge_saved_aids = None
            self._qgis_bridge_prev_maptool = None
            self._qgis_bridge_editing_conn = False
            self._qgis_bridge_feedback_conn = False
            self._qgis_bridge_layer_flags = None
            self._qgis_bridge_snapshot = {}
            self._qgis_bridge_vertex_dock_visibility = {}
            self._qgis_bridge_saved_search_radius = None
            self._qgis_bridge_saved_selection_colour = None
            self._qgis_bridge_saved_form_suppress = None
            self._qgis_bridge_target_idx = None
            self._qgis_bridge_target_det_id = None
            self._qgis_bridge_undo_stack = None
            self._qgis_bridge_pending_fids = []
            self._qgis_bridge_born_det_ids = set()
            self._qgis_bridge_id_write_marks = set()
            self._qgis_bridge_id_edit = False
            try:
                self._end_correct_focus()
            except Exception:  # noqa: BLE001
                pass  # nosec B110

    def _on_qgis_bridge_editing_stopped(self) -> None:
        """The user ended editing directly in QGIS (its own toggle already
        committed or rolled back). Do not touch the edit buffer; just restore
        the aids, fold back what is now on the layer, and drop the banner.

        The teardown is DEFERRED by one turn of the event loop, never run here.
        QGIS emits ``editingStopped`` from inside ``commitChanges()``, so on
        this stack frame the teardown would swap the map tool out from under a
        live vertex tool, fold edits back through a provider delete-and-add on
        the same layer, and reset the layer flags, all while the commit is
        still running. Same class as the ``featureAdded`` trap below, one level
        up."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        if getattr(self, "_qgis_bridge_finishing", False):
            return
        if getattr(self, "_qgis_bridge_stop_queued", False):
            return
        self._qgis_bridge_stop_queued = True
        try:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._teardown_bridge_after_external_stop)
        except (RuntimeError, AttributeError, ImportError):
            # No timer, no deferral. Tearing down on the emitting frame is the
            # risk this method exists to avoid, but leaving the user's project
            # with our editing aids forced on is the worse of the two.
            self._qgis_bridge_stop_queued = False
            self._teardown_qgis_edit_bridge(commit=True, external=True)

    def _teardown_bridge_after_external_stop(self) -> None:
        """The deferred half of ``_on_qgis_bridge_editing_stopped``: run the
        teardown one turn later, off the commit's own C++ stack frame.

        Everything is re-checked here, because a turn is long enough for the
        session to have ended by another path (Save, Exit review, mode switch)
        or for the layer to have been removed from the project."""
        try:
            if not getattr(self, "_qgis_bridge_active", False):
                return
            if getattr(self, "_qgis_bridge_finishing", False):
                return
            layer = getattr(self, "_qgis_bridge_layer", None)
            if layer is not None and not self._is_layer_valid(layer):
                # Gone with the session: nothing left to fold or to restore on
                # the layer itself. Drop the dead reference so the teardown does
                # not carry it around, and let it run for the aids.
                self._qgis_bridge_layer = None
            self._teardown_qgis_edit_bridge(commit=True, external=True)
        finally:
            self._qgis_bridge_stop_queued = False

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _teardown_qgis_edit_bridge(self, commit: bool, external: bool) -> None:
        """Single teardown: (optionally) commit/rollback, restore aids, reset the
        map tool, drop the banner, fold back. ``external`` = QGIS already ended
        the edit session, so leave the buffer alone."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        self._qgis_bridge_finishing = True
        layer = self._qgis_bridge_layer
        committed = external or commit
        try:
            # Drop editingStopped FIRST so the commit/rollback below cannot
            # re-enter this teardown.
            self._disconnect_bridge_editing_signal(layer)
            self._disconnect_bridge_feedback(layer)
            self._disconnect_bridge_layer_watch()
            self._disconnect_bridge_tool_messages()
            self._stop_bridge_gesture_poll()
            if (not external and layer is not None and self._is_layer_valid(layer)):
                try:
                    if layer.isEditable():
                        if commit:
                            layer.commitChanges()
                        else:
                            layer.rollBack()
                except (RuntimeError, AttributeError):
                    pass
            # Clear the target selection we set on enter, so no stray selected
            # feature lingers under the resting review renderer.
            if layer is not None and self._is_layer_valid(layer):
                try:
                    layer.removeSelection()
                except (RuntimeError, AttributeError):
                    pass
            # Aids come back on EVERY path, before anything that could fail.
            # Each one under its own guard: a raise partway used to skip the
            # ones after it, and the finally below then dropped every saved
            # value, so the user lost that setting for the session.
            self._restore_bridge_setting(self._restore_bridge_editing_aids)
            self._restore_bridge_setting(self._restore_bridge_vertex_search_radius)
            self._restore_bridge_setting(self._restore_bridge_selection_colour)
            self._restore_bridge_setting(self._restore_bridge_attribute_form, layer)
            self._restore_bridge_setting(self._restore_bridge_map_tool)
            self._restore_bridge_setting(self._restore_bridge_vertex_editor_visibility)
            self._leave_bridge_banner()
            # Fold back ONLY when edits committed (Done, or QGIS ended the
            # session itself). On an explicit rollback (Exit review, mode
            # switch, unload) the buffer is discarded and the review is being
            # torn down, so a fold-back is wasted work and would repaint a layer
            # about to be removed.
            features = 0
            if committed and layer is not None and self._is_layer_valid(layer):
                features = self._bridge_fold_back(layer)
            # leave_qgis_bridge_state restores the always-present pieces; the
            # per-step primary is the plugin's call, so re-drive the current
            # review step (Correct) to bring its button back.
            self._restore_review_step_after_bridge()
            duration_ms = int(
                (time.monotonic() - (self._qgis_bridge_t0 or time.monotonic())) * 1000)
            self._track_bridge(
                "committed" if committed else "rolled_back",
                duration_ms=duration_ms, features=features)
        finally:
            # Every detection goes back on the review layer here, AFTER the
            # fold above has read it. The fold tells an edited shape from an
            # untouched one against a snapshot taken under the subset, so a
            # layer handed back whole first would re-base every visible object.
            # Reached even when the fold raised, and a no-op when the fold
            # already cleared it on its own way through.
            self._clear_bridge_isolation()
            self._restore_bridge_layer_presentation(layer)
            self._disconnect_bridge_layer_watch()
            # The next session may own a polygon, so the three shape tools come
            # back on the panel whatever this one was.
            self._set_bridge_shape_tools_visible(True)
            self._qgis_bridge_active = False
            self._qgis_bridge_finishing = False
            self._qgis_bridge_isolated = False
            self._qgis_bridge_layer = None
            self._qgis_bridge_saved_aids = None
            self._qgis_bridge_prev_maptool = None
            self._qgis_bridge_editing_conn = False
            self._qgis_bridge_feedback_conn = False
            self._qgis_bridge_layer_flags = None
            self._qgis_bridge_snapshot = {}
            self._qgis_bridge_vertex_dock_visibility = {}
            self._qgis_bridge_saved_search_radius = None
            self._qgis_bridge_saved_selection_colour = None
            self._qgis_bridge_saved_form_suppress = None
            self._qgis_bridge_target_idx = None
            self._qgis_bridge_target_det_id = None
            self._qgis_bridge_pending_fids = []
            self._qgis_bridge_born_det_ids = set()
            self._qgis_bridge_id_write_marks = set()
            self._qgis_bridge_id_edit = False
        # Give every polygon its colour back. After the finally, so the flag it
        # reads is already down; the repaint is a no-op when the review layer
        # went with the teardown.
        try:
            self._end_correct_focus()
        except (RuntimeError, AttributeError):
            pass
        # A finished Manual edit ends at REST, with no polygon selected, exactly
        # where a finished AI refine ends. The session used to re-select the
        # polygon it had just saved, which reopened the per-polygon panel; in
        # Manual that panel has no shape controls, so the only thing left on it
        # was the line pointing at the other method, with no way on to another
        # edit or to Add. leave_qgis_bridge_state has already brought the two
        # branch cards back and _restore_review_step_after_bridge re-armed the
        # pick tool, so the next click on a polygon opens the next edit. Do not
        # re-add a re-selection here.

    def _disconnect_bridge_editing_signal(self, layer) -> None:
        if not self._qgis_bridge_editing_conn or layer is None:
            self._qgis_bridge_editing_conn = False
            return
        try:
            layer.editingStopped.disconnect(self._on_qgis_bridge_editing_stopped)
        except (TypeError, RuntimeError, AttributeError):
            pass
        self._qgis_bridge_editing_conn = False

    # ------------------------------------------------------------------
    # The session's layer leaving the project
    # ------------------------------------------------------------------

    def _connect_bridge_layer_watch(self) -> None:
        """Watch for the session's own layer leaving the project.

        The session strips the review layer's Private flag, which is what puts
        it in the Layers panel, so the user can remove it there. Nothing else
        would notice: the bridge would stay open, its poll ticking, with
        snapping, topological editing and the selection colour still forced on
        the project until the user pressed Save."""
        self._qgis_bridge_layer_watch = False
        try:
            from qgis.core import QgsProject

            QgsProject.instance().layersWillBeRemoved.connect(
                self._on_bridge_layer_will_be_removed)
            self._qgis_bridge_layer_watch = True
        except (RuntimeError, AttributeError, TypeError, ImportError):
            self._qgis_bridge_layer_watch = False

    def _disconnect_bridge_layer_watch(self) -> None:
        """Drop the layer-removal watch. Idempotent; never raises."""
        if not getattr(self, "_qgis_bridge_layer_watch", False):
            return
        self._qgis_bridge_layer_watch = False
        try:
            from qgis.core import QgsProject

            QgsProject.instance().layersWillBeRemoved.disconnect(
                self._on_bridge_layer_will_be_removed)
        except (RuntimeError, AttributeError, TypeError, ImportError):
            pass

    def _on_bridge_layer_will_be_removed(self, layer_ids) -> None:
        """The session's layer is about to leave the project: end the bridge.

        Run on THIS stack, unlike the editingStopped teardown: the layer is
        still alive here, and the rollback, the subset and the layer flags all
        need it. One turn later it is gone and every one of them fails."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        layer = getattr(self, "_qgis_bridge_layer", None)
        if layer is None:
            return
        try:
            ids = set(layer_ids or [])
            if not ids or layer.id() not in ids:
                return
        except (RuntimeError, AttributeError, TypeError):
            return
        # Dropped first, so the context layer the teardown removes cannot come
        # back through here.
        self._disconnect_bridge_layer_watch()
        self._abort_qgis_edit_bridge_if_active()

    # ------------------------------------------------------------------
    # Live feedback (echo each native edit as a plain line in the banner)
    # ------------------------------------------------------------------

    def _connect_bridge_feedback(self, layer) -> None:
        """Echo the edit buffer's changes as a running feedback line, since QGIS
        reports them only in the easily-missed status bar. Best-effort: a build
        that lacks a signal simply gives less feedback, never an error."""
        if layer is None:
            return
        connected = False
        for signal_name, handler in (
            ("geometryChanged", self._on_bridge_geometry_changed),
            ("featureAdded", self._on_bridge_feature_added),
            ("featuresDeleted", self._on_bridge_features_deleted),
        ):
            signal = getattr(layer, signal_name, None)
            if signal is None:
                continue
            try:
                signal.connect(handler)
                connected = True
            except (RuntimeError, AttributeError, TypeError):
                pass
        stack = None
        try:
            stack = layer.undoStack()
        except (RuntimeError, AttributeError):
            stack = None
        self._qgis_bridge_undo_stack = stack
        if stack is not None:
            try:
                stack.indexChanged.connect(self._on_bridge_undo_index_changed)
                connected = True
            except (RuntimeError, AttributeError, TypeError):
                pass
        self._qgis_bridge_feedback_conn = connected

    def _disconnect_bridge_feedback(self, layer) -> None:
        if not self._qgis_bridge_feedback_conn or layer is None:
            # Drop the stack reference on this path too. A build where every
            # connect failed still stored it, and holding a layer's undo stack
            # after teardown is exactly the kind of reference that outlives the
            # layer it belongs to.
            self._qgis_bridge_undo_stack = None
            self._qgis_bridge_feedback_conn = False
            return
        for signal_name, handler in (
            ("geometryChanged", self._on_bridge_geometry_changed),
            ("featureAdded", self._on_bridge_feature_added),
            ("featuresDeleted", self._on_bridge_features_deleted),
        ):
            signal = getattr(layer, signal_name, None)
            if signal is None:
                continue
            try:
                signal.disconnect(handler)
            except (RuntimeError, AttributeError, TypeError):
                pass
        stack = getattr(self, "_qgis_bridge_undo_stack", None)
        self._qgis_bridge_undo_stack = None
        if stack is not None:
            try:
                stack.indexChanged.disconnect(
                    self._on_bridge_undo_index_changed)
            except (RuntimeError, AttributeError, TypeError):
                pass
        self._qgis_bridge_feedback_conn = False

    def _bridge_feedback(self, text: str, kind: str = "armed") -> None:
        """Write the panel's one line. ``kind`` is the message taxonomy: the
        running gesture is "armed", a recorded edit "success", a gesture that
        did nothing "warning"."""
        dock = getattr(self, "dock_widget", None)
        fn = getattr(dock, "set_qgis_bridge_feedback", None)
        if callable(fn):
            try:
                fn(text, kind)
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _bridge_line_open(self, open_: bool) -> None:
        """Tell the panel a capture line is (or is no longer) in progress, so
        Finish takes Save's slot for exactly as long as there is a line to
        close."""
        dock = getattr(self, "dock_widget", None)
        fn = getattr(dock, "set_qgis_bridge_line_open", None)
        if callable(fn):
            try:
                fn(bool(open_))
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _on_bridge_geometry_changed(self, *_args) -> None:
        if getattr(self, "_qgis_bridge_finishing", False):
            return
        # The Points dial drives its own changeGeometry; that is not a hand edit
        # and reports its own line, so skip both the latch and the feedback.
        if getattr(self, "_qgis_bridge_dial_edit", False):
            return
        self._mark_bridge_hand_edited()
        self._bridge_feedback(
            tr("Shape updated. Keep editing, or click Save."), "success")

    def _on_bridge_feature_added(self, *args) -> None:
        # A native split adds one feature per new piece, and Add Feature adds the
        # polygon the user drew; one neutral line covers both.
        if getattr(self, "_qgis_bridge_finishing", False):
            return
        if getattr(self, "_qgis_bridge_dial_edit", False):
            return
        self._mark_bridge_hand_edited()
        self._bridge_feedback(tr("New shape added. Click Save to keep it."),
                              "success")
        self._queue_bridge_born_det_id(args[0] if args else None)
        # The Distinct render keys on det_id, and this shape has none yet, so
        # repaint as soon as the queued write gives it one.
        layer = getattr(self, "_qgis_bridge_layer", None)
        if layer is not None:
            try:
                layer.triggerRepaint()
            except (RuntimeError, AttributeError):
                pass

    # ------------------------------------------------------------------
    # Identity of a feature born inside the session
    # ------------------------------------------------------------------

    def _queue_bridge_born_det_id(self, fid) -> None:
        """Remember a just-added feature and settle its det_id next turn.

        The write cannot happen here. QGIS emits ``featureAdded`` from inside
        the undo command that adds the feature, so an attribute write made from
        this handler is pushed onto the undo stack BEFORE the add it belongs to;
        undoing the pair then runs the attribute undo against a feature the add
        undo has already erased. One turn of the event loop puts the write after
        the add, where undo can step over it cleanly.
        """
        if not isinstance(fid, int):
            return
        pending = getattr(self, "_qgis_bridge_pending_fids", None)
        if pending is None:
            pending = []
            self._qgis_bridge_pending_fids = pending
        if fid not in pending:
            pending.append(fid)
        try:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._assign_bridge_born_det_ids)
        except (RuntimeError, AttributeError, ImportError):
            # No timer means no early identity; Save still folds the shape back
            # and the fold mints an id, exactly as it did before.
            pass

    def _bridge_det_id_field_index(self, layer) -> int:
        """Position of the layer's ``det_id`` field, or -1 when it has none."""
        try:
            return int(layer.fields().indexFromName("det_id"))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return -1

    def _bridge_used_det_ids(self, layer, skip_fids=()) -> set[int]:
        """Every det_id that is already spoken for, from BOTH sources.

        The editable layer carries only the detections the review filters
        currently show. ``_auto_objects`` is the whole model, hidden objects
        included, and the fold uses exactly that list as its prior: an id taken
        from an object a confidence or size filter is hiding would make the fold
        read the new drawing as an edit of that object and overwrite it.
        """
        used: set[int] = set()
        try:
            objects = getattr(self, "_auto_objects", None) or []
            for index in range(len(objects)):
                value = self._object_fid_for(index)
                if isinstance(value, int) and value >= 0:
                    used.add(int(value))
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            from qgis.core import QgsFeatureRequest
            request = QgsFeatureRequest()
            request.setSubsetOfAttributes(["det_id"], layer.fields())
            scope = getattr(QgsFeatureRequest, "Flag", QgsFeatureRequest)
            no_geometry = getattr(scope, "NoGeometry", None)
            if no_geometry is not None:
                request.setFlags(no_geometry)
            for feature in layer.getFeatures(request):
                if feature.id() in skip_fids:
                    continue
                value = feature["det_id"]
                if isinstance(value, int) and value >= 0:
                    used.add(int(value))
        except (RuntimeError, AttributeError, TypeError, KeyError, ValueError,
                ImportError):
            pass
        used.update(getattr(self, "_qgis_bridge_born_det_ids", None) or ())
        return used

    def _assign_bridge_born_det_ids(self) -> None:
        """Give every feature born in this session its FINAL det_id, now.

        A polygon drawn by hand has no det_id, and the half a native split
        creates carries a copy of the parent's. Both used to be identified only
        when Save folded them back, so the colour a shape wore while it was
        being drawn came from one number and the colour it kept came from
        another, and it changed under the user at Save. The id written here is
        the id the object keeps: the fold re-mints only an id that is missing,
        negative or already taken, so one that is none of those survives it
        unchanged. A split half is a duplicate and gets a fresh id here, which
        is what makes the cut visible the moment it lands.

        Best-effort at every step. A layer with no det_id field, or a provider
        that refuses the write, leaves the identity to the fold as before.
        """
        # A commit or a teardown is mid-flight and owns the layer, so an edit
        # macro opened here would land inside it. LEAVE THE QUEUE ALONE (this
        # check comes before the queue is read, never after): the commit path
        # settles it a moment later on its own stack, and draining it here would
        # hand the fold a shape with no id and change its colour at Save.
        if (getattr(self, "_qgis_bridge_finishing", False) or getattr(self, "_qgis_bridge_closing_capture", False)):
            return
        pending = list(getattr(self, "_qgis_bridge_pending_fids", None) or ())
        self._qgis_bridge_pending_fids = []
        if not pending or not getattr(self, "_qgis_bridge_active", False):
            return
        layer = getattr(self, "_qgis_bridge_layer", None)
        if layer is None or not self._is_layer_valid(layer):
            return
        # The session may have ended between the queue and this turn. An edit
        # macro opened on a layer that is no longer editable writes nothing and
        # lands an empty command on the user's own undo stack.
        try:
            if not layer.isEditable():
                return
        except (RuntimeError, AttributeError):
            return
        field_index = self._bridge_det_id_field_index(layer)
        if field_index < 0:
            return
        used = self._bridge_used_det_ids(layer, skip_fids=set(pending))
        next_id = max(used, default=-1) + 1
        assignments: list[tuple[int, int]] = []
        for fid in pending:
            try:
                feature = layer.getFeature(fid)
                if feature is None or not feature.isValid():
                    continue
                current = feature["det_id"]
            except (RuntimeError, AttributeError, KeyError, TypeError, ValueError):
                continue
            if isinstance(current, int) and current >= 0 and current not in used:
                used.add(int(current))
                continue
            while next_id in used:
                next_id += 1
            assignments.append((fid, next_id))
            used.add(next_id)
            next_id += 1
        if not assignments:
            return
        self._write_bridge_born_det_ids(layer, field_index, assignments)

    def _write_bridge_born_det_ids(self, layer, field_index, assignments) -> None:
        """Write the settled ids as ONE undo step, and mark where it landed.

        One step, because a split can produce several pieces at once and the
        panel's Undo has to step over the whole assignment to reach the edit the
        user actually made."""
        born = getattr(self, "_qgis_bridge_born_det_ids", None)
        if born is None:
            born = set()
            self._qgis_bridge_born_det_ids = born
        self._qgis_bridge_id_edit = True
        wrote = False
        opened = False
        try:
            layer.beginEditCommand(tr("Identify new shape"))
            opened = True
            for fid, det_id in assignments:
                try:
                    if layer.changeAttributeValue(fid, field_index, det_id):
                        born.add(det_id)
                        wrote = True
                except (RuntimeError, AttributeError, TypeError, ValueError):
                    pass
        except (RuntimeError, AttributeError, TypeError):
            wrote = False
        finally:
            # An edit command left open would break every later undo, so it is
            # closed on the failure path too.
            if opened:
                try:
                    layer.endEditCommand()
                except (RuntimeError, AttributeError):
                    wrote = False
            self._qgis_bridge_id_edit = False
        if not wrote:
            return
        stack = getattr(self, "_qgis_bridge_undo_stack", None)
        marks = getattr(self, "_qgis_bridge_id_write_marks", None)
        if stack is not None and marks is not None:
            try:
                marks.add(int(stack.index()))
            except (RuntimeError, AttributeError, TypeError):
                pass
        try:
            layer.triggerRepaint()
        except (RuntimeError, AttributeError):
            pass

    def _bridge_pop_identity_mark(self, stack) -> bool:
        """True when our own identity write sits on top of the undo stack.

        Marks above the current position belong to a redo branch QGIS has
        already dropped, so they go with it."""
        marks = getattr(self, "_qgis_bridge_id_write_marks", None)
        if not marks:
            return False
        try:
            index = int(stack.index())
        except (RuntimeError, AttributeError, TypeError):
            return False
        for mark in [value for value in marks if value > index]:
            marks.discard(mark)
        if index not in marks:
            return False
        marks.discard(index)
        return True

    def _on_bridge_features_deleted(self, *_args) -> None:
        if getattr(self, "_qgis_bridge_finishing", False):
            return
        self._bridge_feedback(
            tr("A shape was removed. Click Save to confirm."), "success")

    def _on_bridge_undo_index_changed(self, *_args) -> None:
        """Report the edit QGIS just recorded, by its own name, and offer Undo.

        Fires after the edit-buffer signals above, so its more precise text wins
        over their generic line."""
        if getattr(self, "_qgis_bridge_finishing", False):
            return
        # Settling a new shape's det_id is our bookkeeping, not a change the
        # user made, so it must not take over the line naming their last edit.
        if getattr(self, "_qgis_bridge_id_edit", False):
            return
        stack = getattr(self, "_qgis_bridge_undo_stack", None)
        text = ""
        can_undo = False
        if stack is not None:
            try:
                can_undo = bool(stack.canUndo())
                if can_undo:
                    # QGIS names the operation itself ("Moved vertex", "Split
                    # features"). The line's success glyph already says it
                    # landed, so "Last change:" in front of it was one label
                    # too many.
                    text = (str(stack.undoText() or "").strip() or tr("Change recorded."))
            except (RuntimeError, AttributeError, TypeError):
                can_undo = False
        dock = getattr(self, "dock_widget", None)
        fn = getattr(dock, "set_qgis_bridge_last_change", None)
        if callable(fn):
            try:
                fn(text, can_undo)
            except (RuntimeError, AttributeError, TypeError):
                pass

    def undo_qgis_bridge_edit(self) -> None:
        """Step back ONE thing, without leaving the bridge.

        The panel has a single Undo because the user has a single mental
        "back": the last thing they did. While a line is being traced that is
        the point just placed (nothing is committed yet, so the undo stack
        knows nothing about it); otherwise it is the last recorded edit. Native
        tools commit a gesture the moment it ends, with no confirmation, so
        this is the way back from a mis-drawn split or a dragged-away corner."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        if int(getattr(self, "_qgis_bridge_prev_points", 0) or 0) > 0:
            self._bridge_undo_capture_vertex()
            return
        stack = getattr(self, "_qgis_bridge_undo_stack", None)
        if stack is None:
            return
        try:
            # A new shape's det_id is written as its own undo step, right on top
            # of the add it belongs to. The user never asked for it, so step over
            # it and take the edit underneath, or their first Undo would only
            # change the shape's colour and leave it on the map.
            paired = self._bridge_pop_identity_mark(stack)
            if stack.canUndo():
                stack.undo()
                if paired and stack.canUndo():
                    stack.undo()
        except (RuntimeError, AttributeError):
            pass

    def _route_escape_qgis_bridge(self) -> bool:
        """Escape while a manual edit session owns the canvas.

        A line being traced is dropped and the session stays open; with no line
        in progress, Escape ends the session the way Save does. It NEVER
        reaches the review-exit dialog: leaving the whole review on a reflex
        Escape, mid-edit, was the old behaviour and it asked to save 77
        detections when the user only wanted to drop a two-point line.
        Returns True (Escape consumed)."""
        if int(getattr(self, "_qgis_bridge_prev_points", 0) or 0) > 0:
            self._bridge_cancel_capture()
            return True
        self.finish_qgis_edit_bridge()
        return True

    def _bridge_cancel_capture(self) -> None:
        """Drop the capture line in progress, keeping the tool armed.

        Calls the tool's own reset rather than sending an Escape key: a
        synthetic key press goes through the shortcut map first, where the
        dock's window-level Escape claims it and exits the review."""
        try:
            canvas = self.iface.mapCanvas()
            tool = canvas.mapTool() if canvas is not None else None
            if tool is None:
                return
            reset = getattr(tool, "stopCapturing", None)
            if not callable(reset):
                return
            reset()
            self._qgis_bridge_prev_points = 0
            self._bridge_line_open(False)
            self._bridge_feedback("")
            canvas.refresh()
        except (RuntimeError, AttributeError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Banner gesture buttons (QGIS's native shortcuts as buttons)
    # ------------------------------------------------------------------

    def _on_bridge_gesture_requested(self, kind: str) -> None:
        """Panel gesture button pressed: run the matching native QGIS gesture.

        "finish" ends a capture line the way a right-click on the map does;
        "delete_corner" sends a real Delete key. A synthetic mouse event goes
        ONLY through the right-click finish, and never while the vertex tool
        owns the canvas (that crashes QGIS)."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        kind = str(kind)
        if kind == "finish":
            self._bridge_send_right_click()
            return
        if kind == "cancel":
            # No button emits this any more (Escape and Undo cover it), but the
            # signal is public, so the safe route stays: never a synthetic
            # Escape, which the dock's window-level shortcut would claim.
            self._bridge_cancel_capture()
            return
        if kind == "undo_point":
            # A synthetic Backspace sent to the canvas never reached the capture
            # tool's keyPressEvent, so Undo did nothing while tracing a reshape
            # or split line. Call the capture tool's own last-vertex undo
            # directly, the same way "finish" acts on the tool instead of faking
            # a key.
            self._bridge_undo_capture_vertex()
            return
        from qgis.PyQt.QtCore import Qt
        key = {"delete_corner": Qt.Key.Key_Delete}.get(kind)
        if key is not None:
            self._bridge_send_key(key)

    def _bridge_undo_capture_vertex(self) -> None:
        """Remove the last vertex from the in-progress capture trace (reshape /
        split / add). QgsMapToolCapture.undo() is the native last-vertex undo;
        it is reachable on the live tool object exactly like tool.size() is in
        the gesture poll. No-op when the active tool is not a capture tool."""
        try:
            canvas = self.iface.mapCanvas()
            tool = canvas.mapTool() if canvas is not None else None
            if tool is None:
                return
            undo = getattr(tool, "undo", None)
            # size() is the QgsMapToolCapture marker also used by the poll: its
            # presence confirms this is a capture tool with a vertex buffer.
            if not callable(undo) or not callable(getattr(tool, "size", None)):
                return
            undo()
            canvas.refresh()
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _bridge_send_key(self, key) -> None:
        """Deliver one key press then release to the map canvas.

        Keys only: a synthetic MOUSE event on the vertex tool crashes QGIS, so
        every vertex gesture without a button (undo a point, cancel, delete a
        corner) goes through a real key here instead."""
        try:
            from qgis.PyQt.QtCore import QEvent, Qt
            from qgis.PyQt.QtGui import QKeyEvent
            from qgis.PyQt.QtWidgets import QApplication
            canvas = self.iface.mapCanvas()
            if canvas is None:
                return
            no_mod = Qt.KeyboardModifier.NoModifier
            for phase in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                QApplication.sendEvent(
                    canvas, QKeyEvent(phase, int(key), no_mod))
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError):
            pass

    def _bridge_send_right_click(self) -> None:
        """Finish the active capture line with a synthetic right-click on the
        canvas viewport, which QGIS's capture tools treat as "finish".

        Guarded against the vertex tool: a synthetic mouse event delivered while
        a vertex tool owns the canvas crashes QGIS. The Finish button is never
        shown in vertex mode (dock-side), and this is the belt-and-braces check.
        The 6-argument QMouseEvent constructor exists on both Qt5 and Qt6."""
        try:
            from qgis.PyQt.QtCore import QEvent, QPointF, Qt
            from qgis.PyQt.QtGui import QMouseEvent
            from qgis.PyQt.QtWidgets import QApplication
            canvas = self.iface.mapCanvas()
            if canvas is None:
                return
            tool = canvas.mapTool()
            if tool is not None:
                try:
                    cls = str(tool.metaObject().className())
                except (RuntimeError, AttributeError):
                    cls = ""
                if cls in _VERTEX_TOOL_CLASSES:
                    return
            viewport = canvas.viewport()
            if viewport is None:
                return
            centre = viewport.rect().center()
            local = QPointF(centre)
            global_pt = QPointF(viewport.mapToGlobal(centre))
            right = Qt.MouseButton.RightButton
            no_mod = Qt.KeyboardModifier.NoModifier
            for phase in (QEvent.Type.MouseButtonPress,
                          QEvent.Type.MouseButtonRelease):
                event = QMouseEvent(
                    phase, local, global_pt, right, right, no_mod)
                QApplication.sendEvent(viewport, event)
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError):
            pass

    # ------------------------------------------------------------------
    # Banner Points dial (thin the target before hand edits)
    # ------------------------------------------------------------------

    def _mark_bridge_hand_edited(self) -> None:
        """Latch the session as hand-edited and retire the Points dial.

        The dial rewrites the target from its pre-edit snapshot, so once the
        user has moved a corner themselves it must stop (or it would wipe those
        corners). Hides the row too."""
        self._qgis_bridge_hand_edited = True
        dock = getattr(self, "dock_widget", None)
        fn = getattr(dock, "set_qgis_bridge_points_visible", None)
        if callable(fn):
            try:
                fn(False)
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _on_bridge_points_changed(self, pct: int) -> None:
        """Banner Points dial: thin the target polygon before hand editing.

        No-op if the bridge is inactive, if the shape has already been
        hand-edited, or if there is no resolved target. Rewrites from the
        pre-edit snapshot with the review's Points engine and commits it as one
        QGIS edit command, so the native undo steps over it cleanly."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        if getattr(self, "_qgis_bridge_hand_edited", False):
            return
        layer = self._qgis_bridge_layer
        det_id = getattr(self, "_qgis_bridge_target_det_id", None)
        if layer is None or det_id is None or not self._is_layer_valid(layer):
            return
        base_wkb = (getattr(self, "_qgis_bridge_snapshot", None) or {}).get(det_id)
        if not base_wkb:
            return
        from qgis.core import QgsGeometry
        base = QgsGeometry()
        try:
            base.fromWkb(base_wkb)
        except (RuntimeError, AttributeError, TypeError):
            return
        if base is None or base.isEmpty():
            return
        try:
            pct = max(10, min(100, int(pct)))
        except (TypeError, ValueError):
            return
        geom = QgsGeometry(base) if pct >= 100 else self._bridge_thin_geometry(
            base, pct)
        if geom is None or geom.isEmpty():
            geom = QgsGeometry(base)
        feature = self._bridge_feature_for_det_id(layer, det_id)
        if feature is None:
            return
        fid = feature.id()
        self._qgis_bridge_dial_edit = True
        try:
            layer.beginEditCommand(tr("Fewer points"))
            layer.changeGeometry(fid, geom)
            layer.endEditCommand()
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, TypeError):
            try:
                layer.destroyEditCommand()
            except (RuntimeError, AttributeError):
                pass
        finally:
            self._qgis_bridge_dial_edit = False
        try:
            count = self._bridge_ring_vertex_count(geom)
            if count:
                self._bridge_feedback(tr("Points: {n}").format(n=count))
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _apply_shape_only_to_session(self, det_idx: int) -> bool:
        """Push this polygon's own shape dials onto the LIVE bridge geometry.

        Returns True when a session took the change, so the caller knows not to
        rebuild the review layer for it. During a session the editable copy is
        what the user sees; an override written only to the review moves a
        number and nothing on screen follows it, which is exactly where the
        user reaches for these dials (thin a dense outline, then drag it).

        Mirrors _on_bridge_points_changed: rewrite from the pre-edit snapshot
        in ONE QGIS edit command so native undo steps over it cleanly, and
        leave a shape the user has already hand-edited alone, because
        rewriting it would undo their work under them.
        """
        if not getattr(self, "_qgis_bridge_active", False):
            return False
        if getattr(self, "_qgis_bridge_hand_edited", False):
            return False
        layer = self._qgis_bridge_layer
        det_id = getattr(self, "_qgis_bridge_target_det_id", None)
        if layer is None or det_id is None or not self._is_layer_valid(layer):
            return False
        if self._det_id_for_object_index(det_idx) != det_id:
            return False
        base_wkb = (getattr(self, "_qgis_bridge_snapshot", None) or {}).get(det_id)
        if not base_wkb:
            return False
        from qgis.core import QgsGeometry
        base = QgsGeometry()
        try:
            base.fromWkb(base_wkb)
        except (RuntimeError, AttributeError, TypeError):
            return False
        if base.isEmpty():
            return False
        review = self._auto_review or {}
        try:
            params = self._shape_params_for_object(
                int(det_idx), dict(review.get("params") or {}))
            geom = self._refine_geom_for_review(
                base, params, float(review.get("pixel_size", 1.0) or 1.0))
        except Exception:  # noqa: BLE001 -- a bad refine never breaks a session
            return False
        if geom is None or geom.isEmpty():
            return False
        feature = self._bridge_feature_for_det_id(layer, det_id)
        if feature is None:
            return False
        self._qgis_bridge_dial_edit = True
        try:
            layer.beginEditCommand(tr("Outline settings"))
            layer.changeGeometry(feature.id(), geom)
            layer.endEditCommand()
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, TypeError):
            try:
                layer.destroyEditCommand()
            except (RuntimeError, AttributeError):
                pass
            return False
        finally:
            self._qgis_bridge_dial_edit = False
        try:
            count = self._bridge_ring_vertex_count(geom)
            if count:
                self._bridge_feedback(tr("Points: {n}").format(n=count))
        except (RuntimeError, AttributeError, TypeError):
            pass
        return True

    def _bridge_thin_geometry(self, base, pct: int):
        """Thin one polygon with the review's Points engine at the given share.

        Mirrors the point budget core.live_refine.LiveRefiner resolves for the
        review: the min-points floor and the deviation cap are ground metres,
        crossed by the run's metres-per-unit factor at the object, so one
        setting behaves the same across CRSs.
        Returns a copy of the base when thinning yields nothing usable."""
        from qgis.core import QgsGeometry

        from ...core.detection_policy import vertex_budget_settings
        from ...core.vertex_budget import simplify_to_budget
        try:
            settings = vertex_budget_settings()
        except (RuntimeError, AttributeError, TypeError, KeyError):
            return QgsGeometry(base)
        factor = None
        aspect = 1.0
        try:
            centre = base.boundingBox().center()
            factor = self._auto_crs_metres_per_unit(centre.x(), centre.y())
            aspect = self._auto_crs_unit_aspect(centre.x(), centre.y())
        except (RuntimeError, AttributeError, TypeError):
            factor = None
        # No metres-per-unit means the ground dial cannot cross into this CRS.
        # Standing in 1.0 does not fail safe, it changes the unit: a few ground
        # metres read as a few degrees drops every ring to its floor. Hand the
        # shape back untouched instead.
        if not factor or factor <= 0:
            return QgsGeometry(base)
        try:
            aspect = float(aspect)
        except (TypeError, ValueError):
            aspect = 1.0
        if aspect <= 0:
            aspect = 1.0
        try:
            result = simplify_to_budget(
                base,
                spacing=0.0,
                min_vertices=int(settings["min_vertices"]),
                max_deviation=float(settings["max_deviation_m"]) / factor,
                max_deviation_fraction=float(
                    settings["max_deviation_fraction"]),
                # The y axis measures differently in a geographic CRS, and the
                # budget has to square against ground distance, not raw units.
                unit_aspect=aspect,
                keep_fraction=pct / 100.0,
                # A low dial loosens the boundary-movement cap; this ceiling
                # keeps the loosened cap inside the object. Server dial, so it
                # travels with the settings rather than the engine default.
                dial_max_cap_fraction=float(
                    settings["dial_max_cap_fraction"]),
            )
        except (RuntimeError, AttributeError, TypeError, KeyError, ValueError):
            return QgsGeometry(base)
        if result is None or result.isEmpty():
            return QgsGeometry(base)
        return result

    def _bridge_ring_vertex_count(self, geom) -> int:
        """Points in the outer ring of a (multi)polygon, for the dial feedback.
        Best-effort; 0 when it cannot be read."""
        try:
            if geom is None or geom.isEmpty():
                return 0
            if geom.isMultipart():
                polys = geom.asMultiPolygon()
                if polys and polys[0]:
                    return len(polys[0][0])
                return 0
            poly = geom.asPolygon()
            if poly:
                return len(poly[0])
            return 0
        except (RuntimeError, AttributeError, TypeError, IndexError):
            return 0

    # ------------------------------------------------------------------
    # In-flight gesture: point count, and the silent-failure case
    # ------------------------------------------------------------------

    def _start_bridge_gesture_poll(self) -> None:
        """Sample the armed capture tool so the panel can count points as they
        are placed. The timer is parented to the DOCK: the plugin controller is
        not a QObject, so it can never own Qt children."""
        try:
            from qgis.PyQt.QtCore import QTimer
            timer = self._qgis_bridge_poll_timer
            if timer is None:
                timer = QTimer(self.dock_widget)
                timer.setInterval(_BRIDGE_POLL_MS)
                timer.timeout.connect(self._on_bridge_gesture_tick)
                self._qgis_bridge_poll_timer = timer
            self._qgis_bridge_prev_points = 0
            self._qgis_bridge_gesture_undo_count = None
            timer.start()
        except (RuntimeError, AttributeError, TypeError, ImportError):
            self._qgis_bridge_poll_timer = None

    def _stop_bridge_gesture_poll(self) -> None:
        timer = getattr(self, "_qgis_bridge_poll_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):
                pass
        self._qgis_bridge_prev_points = 0
        self._qgis_bridge_gesture_undo_count = None

    def _bridge_undo_depth(self):
        """Position in the layer's undo stack, or None. index(), not count():
        a command pushed after an Undo truncates the redo tail, so the count
        can stay put or drop on a gesture that worked, and the caller would
        call it a failure."""
        stack = getattr(self, "_qgis_bridge_undo_stack", None)
        if stack is None:
            return None
        try:
            return int(stack.index())
        except (RuntimeError, AttributeError, TypeError):
            return None

    def _on_bridge_gesture_tick(self) -> None:
        """Report points as the user places them, and catch a gesture that did
        nothing. Re-reads the active tool every tick: a kept reference keeps
        reporting the last gesture forever, because deactivate() does not clear
        a capture tool's buffer."""
        if not getattr(self, "_qgis_bridge_active", False):
            self._stop_bridge_gesture_poll()
            return
        tool = None
        class_name = ""
        try:
            tool = self.iface.mapCanvas().mapTool()
            if tool is not None:
                class_name = str(tool.metaObject().className())
        except (RuntimeError, AttributeError, TypeError):
            return
        points = _bridge_capture_points(tool, class_name)
        if points is None:
            if int(getattr(self, "_qgis_bridge_prev_points", 0) or 0) > 0:
                self._bridge_line_open(False)
            self._qgis_bridge_prev_points = 0
            # Vertex tool: no point buffer, but Delete corner must appear only
            # once a corner is picked.
            self._sync_bridge_delete_corner(class_name)
            return
        # A capture tool owns the canvas: no picked corner, keep Delete hidden.
        self._sync_bridge_delete_corner("")
        prev = int(getattr(self, "_qgis_bridge_prev_points", 0))
        if points == prev:
            return
        if prev == 0 and points > 0:
            self._qgis_bridge_gesture_undo_count = self._bridge_undo_depth()
        self._qgis_bridge_prev_points = points
        # Finish only exists while there is a line to close, and it takes the
        # primary slot from Save for exactly that long.
        self._bridge_line_open(points > 0)
        if points > 0:
            # No "right-click to finish" here: Finish is the button right
            # under this line, and its tooltip carries the mouse gesture.
            self._bridge_feedback(
                tr("{n} point placed.").format(n=points) if points == 1
                else tr("{n} points placed.").format(n=points))
        elif prev > 0:
            self._report_bridge_gesture_result(class_name)

    def _report_bridge_gesture_result(self, tool_class: str) -> None:
        """A finished gesture that recorded NOTHING is the worst case: QGIS says
        nothing at all for a failed reshape (no signal, no message bar, no undo
        entry), so the user redraws the same wrong line forever. Compare the undo
        depth against the gesture's start and name the rule that was missed."""
        start = getattr(self, "_qgis_bridge_gesture_undo_count", None)
        self._qgis_bridge_gesture_undo_count = None
        depth = self._bridge_undo_depth()
        if start is None or depth is None or depth > start:
            return  # something was recorded; the undo-stack handler reports it
        if tool_class == _SPLIT_TOOL_CLASS:
            self._bridge_feedback(tr(
                "Nothing was split. The line has to cross the shape completely, "
                "starting and ending outside it."), "warning")
        elif tool_class in _ADD_TOOL_CLASSES:
            self._bridge_feedback(tr(
                "Nothing was added. A polygon needs at least three corners."),
                "warning")
        else:
            self._bridge_feedback(tr(
                "Nothing changed. The line has to cross the outline twice, "
                "starting and ending outside the shape."), "warning")

    def _close_open_bridge_capture(self) -> None:
        """Close a capture line the user left open, before Save commits.

        A capture tool holds its points in its own buffer, not in the layer's
        edit buffer, so committing while a line is open commits nothing and
        still reports success: the corners are simply thrown away, with no
        error to show. Right-click is what closes the line, and the tool
        rejects it below three corners, which leaves the caller free to commit
        either way. Runs on every commit path, so this holds even if a future
        QGIS class name slips past the poll again."""
        try:
            tool = self.iface.mapCanvas().mapTool()
            class_name = str(tool.metaObject().className()) if tool else ""
        except (RuntimeError, AttributeError, TypeError):
            return
        points = _bridge_capture_points(tool, class_name)
        if not points:
            return
        self._bridge_send_right_click()
        self._qgis_bridge_prev_points = 0
        self._bridge_line_open(False)

    # ------------------------------------------------------------------
    # Layer resolution
    # ------------------------------------------------------------------

    def _resolve_bridge_layer(self):
        """The review's selection layer, if it is a live editable vector layer,
        else None. Asserted here so a future read-only or non-vector overlay
        fails the bridge cleanly instead of half-arming it."""
        layer = getattr(self, "_auto_selection_layer", None)
        if layer is None or not self._is_layer_valid(layer):
            return None
        from qgis.core import QgsVectorLayer
        if not isinstance(layer, QgsVectorLayer):
            return None
        return layer

    def _bridge_feature_for_det_id(self, layer, det_id, with_geometry=False):
        """The layer feature carrying ``det_id``, or None.

        Filtered by the provider instead of scanned in Python: the Points dial
        resolves its target on every tick of the spin box, and walking a review
        layer that holds thousands of detections (each with its geometry) on
        every step made the dial lag behind the pointer. Falls back to the plain
        scan for an id the expression cannot carry."""
        try:
            from qgis.core import QgsFeatureRequest
            request = QgsFeatureRequest()
            request.setFilterExpression(f'"det_id" = {int(det_id):d}')
            request.setLimit(1)
            if not with_geometry:
                scope = getattr(QgsFeatureRequest, "Flag", QgsFeatureRequest)
                no_geometry = getattr(scope, "NoGeometry", None)
                if no_geometry is not None:
                    request.setFlags(no_geometry)
            for feature in layer.getFeatures(request):
                return feature
            return None
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError):
            pass
        try:
            for feature in layer.getFeatures():
                if feature["det_id"] == det_id:
                    return feature
        except (RuntimeError, AttributeError, KeyError, TypeError):
            return None
        return None

    def _expose_and_activate_bridge_layer(self, layer) -> bool:
        """Make the temporary review layer a real active QGIS edit target.

        A Private map layer is ideal while detections are only a review overlay,
        but QGIS intentionally disables every digitizing action for it.  The
        bridge is the one moment that layer must participate in the standard
        layer-tree/active-layer workflow, otherwise the Vertex Editor opens but
        all of its tools remain disabled.
        """
        try:
            from qgis.core import QgsMapLayer

            original = layer.flags()
            self._qgis_bridge_layer_flags = original
            private = getattr(QgsMapLayer.LayerFlag, "Private", None)
            if private is not None and original & private:
                layer.setFlags(flags_without(original, private))
            # The layer-tree view is the authority behind QGIS's digitizing
            # actions.  Set it first, then the interface's active layer; either
            # call alone can be ignored by a particular QGIS release.
            view = self.iface.layerTreeView()
            if view is not None:
                view.setCurrentLayer(layer)
            self.iface.setActiveLayer(layer)
            active = self.iface.activeLayer()
            return active is not None and active.id() == layer.id()
        except (RuntimeError, AttributeError, TypeError):
            return False

    def _restore_bridge_layer_presentation(self, layer) -> None:
        """Put the review layer back into its non-editing private state."""
        flags = getattr(self, "_qgis_bridge_layer_flags", None)
        if layer is None or flags is None:
            return
        try:
            if self._is_layer_valid(layer):
                layer.setFlags(flags)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _snapshot_bridge_layer(self, layer) -> dict[int, bytes]:
        """Return the pre-edit geometry for every uniquely identified feature."""
        snapshot: dict[int, bytes] = {}
        try:
            fields = {field.name() for field in layer.fields()}
            if "det_id" not in fields:
                return snapshot
            for feature in layer.getFeatures():
                det_id = feature["det_id"]
                geom = feature.geometry()
                if (not isinstance(det_id, int) or det_id < 0 or geom is None or geom.isEmpty() or det_id in snapshot):
                    continue
                snapshot[det_id] = bytes(geom.asWkb())
        except (RuntimeError, AttributeError, TypeError, KeyError):
            return {}
        return snapshot

    def _show_bridge_unavailable(self) -> None:
        """Explain a failed active-layer handoff instead of opening dead tools."""
        message = tr(
            "QGIS could not activate the temporary review layer. Close any "
            "other editing session, then try Edit manually again.")
        try:
            from qgis.core import Qgis

            self.iface.messageBar().pushMessage(
                "AI Segmentation", message,
                level=Qgis.MessageLevel.Warning, duration=7)
        except (RuntimeError, AttributeError):
            pass

    def _show_bridge_isolation_failed(self) -> None:
        """Explain a session refused because the other polygons could not be
        held out of it.

        Refusing beats opening: the vertex tool works on the whole active layer,
        so a click could take a neighbour's vertex and Save would write that
        neighbour back as a real edit."""
        message = tr(
            "Editing by hand could not open on this polygon on its own. Try "
            "again, or fix it with the AI.")
        try:
            from qgis.core import Qgis, QgsMessageLog

            QgsMessageLog.logMessage(
                "Manual edit refused: the session could not be held to one "
                "polygon", "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.iface.messageBar().pushMessage(
                "AI Segmentation", message,
                level=Qgis.MessageLevel.Warning, duration=7)
        except (RuntimeError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Editing aids: snapshot on entry, restore on every exit path
    # ------------------------------------------------------------------

    def _save_bridge_editing_aids(self) -> None:
        """Snapshot the project's snapping / topology / avoid-overlap so they can
        be restored exactly on leave.

        The snapping config is saved as the FIVE SCALARS this bridge changes,
        never as the config object. The object is a value copy, but its
        per-layer settings are keyed on raw layer pointers: when a layer is
        removed during the session QGIS scrubs its own copy and cannot scrub
        ours, so putting the whole thing back resurrects a destroyed layer and
        the next project save or quit dereferences it. That is the upstream
        crash `output_store` documents. The bridge never touches a per-layer
        setting, so it has no business restoring one.
        """
        from qgis.core import QgsProject
        proj = QgsProject.instance()
        aids: dict = {}
        try:
            cfg = proj.snappingConfig()
            getter = (getattr(cfg, "typeFlag", None) or getattr(cfg, "type", None))
            aids["snap"] = {
                "enabled": bool(cfg.enabled()),
                "mode": cfg.mode(),
                "type": getter() if getter is not None else None,
                "tolerance": cfg.tolerance(),
                "units": cfg.units(),
            }
        except (RuntimeError, AttributeError, TypeError):
            aids["snap"] = None
        try:
            aids["topo"] = bool(proj.topologicalEditing())
        except (RuntimeError, AttributeError):
            aids["topo"] = False
        try:
            aids["avoid_mode"] = proj.avoidIntersectionsMode()
        except (RuntimeError, AttributeError):
            aids["avoid_mode"] = None
        try:
            aids["avoid_layers"] = list(proj.avoidIntersectionsLayers() or [])
        except (RuntimeError, AttributeError):
            aids["avoid_layers"] = []
        self._qgis_bridge_saved_aids = aids

    def _apply_bridge_editing_config(self, layer) -> bool:
        """Start editing on ``layer`` with snapping (vertex|segment, 12 px) on
        and topological editing plus avoid-overlap off, each for the reason
        given below. Returns False if editing could not start (then the caller
        restores and bails)."""
        from qgis.core import QgsProject
        proj = QgsProject.instance()
        # The review rewrites this layer through the PROVIDER on every filter
        # change, which leaves QGIS's snapping index and the vertex tool's cache
        # describing the features from before that change: the locator then
        # answers a hover with a feature id that no longer exists, and the click
        # does nothing. Each rewrite site invalidates on its own (see
        # shared._notify_provider_write, which carries the full explanation);
        # repeat it on entry because startEditing() demonstrably does NOT rebuild
        # the index, so any provider write we have not tracked down would strand
        # the tools again. Emitted inline rather than imported, to keep this
        # module's import surface small enough to test without PyQGIS.
        try:
            layer.dataChanged.emit()
        except (RuntimeError, AttributeError):
            pass
        try:
            if not layer.isEditable() and not layer.startEditing():
                return False
        except (RuntimeError, AttributeError):
            return False
        self._suppress_bridge_attribute_form(layer)
        try:
            cfg = proj.snappingConfig()
            cfg.setEnabled(True)
            mode = _snap_mode_all_layers()
            if mode is not None:
                cfg.setMode(mode)
            flags = _snap_type_flags()
            if flags is not None:
                # setTypeFlag (QGIS >= 3.26) vs setType (older); resolve by name.
                setter = (getattr(cfg, "setTypeFlag", None) or getattr(cfg, "setType", None))
                if setter is not None:
                    setter(flags)
            cfg.setTolerance(_SNAP_TOLERANCE_PX)
            unit = _tolerance_pixels_unit()
            if unit is not None:
                cfg.setUnits(unit)
            proj.setSnappingConfig(cfg)
        except (RuntimeError, AttributeError, TypeError):
            pass
        # Topological editing stays OFF during hand correction. With it on, a
        # vertex move or delete is written through every border that touches
        # the one under the cursor, so the neighbours change shape too. The
        # session owns ONE polygon, and a coherent shared border is not worth a
        # silent edit on a polygon the user did not pick. Restored to whatever
        # the user had on exit, like every other aid here.
        try:
            proj.setTopologicalEditing(False)
        except (RuntimeError, AttributeError):
            pass
        # Avoid-overlap stays OFF during hand correction. On a dense layer of
        # touching detections it silently carves an edit that grazes a neighbour
        # (clipping it, keeping only the largest part, or rejecting it outright),
        # which reads as "editing is broken". The session is already held to one
        # polygon, so we neutralise avoid here and restore whatever the user had
        # on exit.
        try:
            off_mode = _avoid_mode("AllowIntersections")
            if off_mode is not None and hasattr(proj, "setAvoidIntersectionsMode"):
                proj.setAvoidIntersectionsMode(off_mode)
        except (RuntimeError, AttributeError, TypeError):
            pass
        self._apply_bridge_vertex_search_radius()
        self._apply_bridge_selection_colour()
        return True

    def _suppress_bridge_attribute_form(self, layer) -> None:
        """Stop QGIS popping the attribute-entry dialog every time a feature is
        drawn (Add a polygon) or a piece is cut (Split). During a review the
        user is drawing geometry, not filling a form; the fold assigns the
        class + score. Saves the layer's prior setting and restores it on exit.
        Best-effort: an unknown enum name leaves the default behaviour."""
        try:
            cfg = layer.editFormConfig()
            self._qgis_bridge_saved_form_suppress = cfg.suppress()
            suppress_on = None
            # QGIS >= 3.32 nests the flag under FeatureFormSuppress; older
            # releases expose SuppressOn on the config class directly. Resolve
            # by name so the static Qt6 checker never sees a flat enum access.
            enum = getattr(
                type(cfg), "FeatureFormSuppress", None)
            if enum is not None:
                suppress_on = getattr(enum, "SuppressOn", None)
            if suppress_on is None:
                suppress_on = getattr(type(cfg), "SuppressOn", None)
            if suppress_on is not None:
                cfg.setSuppress(suppress_on)
                layer.setEditFormConfig(cfg)
        except (RuntimeError, AttributeError, TypeError):
            self._qgis_bridge_saved_form_suppress = None

    def _restore_bridge_attribute_form(self, layer) -> None:
        """Put the layer's attribute-form suppression back. Never raises."""
        saved = getattr(self, "_qgis_bridge_saved_form_suppress", None)
        self._qgis_bridge_saved_form_suppress = None
        if saved is None or layer is None:
            return
        try:
            cfg = layer.editFormConfig()
            cfg.setSuppress(saved)
            layer.setEditFormConfig(cfg)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _apply_bridge_selection_colour(self) -> None:
        """Stop the selected polygon from turning solid yellow while editing.

        Scoping the native tools needs the feature SELECTED, but QGIS paints a
        selected feature in the flat selection colour: the review render the
        user is editing against (the Distinct colour, the outline-only view)
        and the imagery under it both disappear. A faint tint keeps QGIS's own
        "this one is selected" reflex while the shape keeps its own look, and
        it tracks the geometry as it is edited, which a drawn outline of our
        own could not. The user's colour comes back on exit.
        """
        canvas = self.iface.mapCanvas()
        getter = getattr(canvas, "selectionColor", None)
        setter = getattr(canvas, "setSelectionColor", None)
        # No getter, no copy to put back, so the colour is left as the user set
        # it. Writing it here would tint every selection in QGIS until restart.
        if setter is None or getter is None:
            return
        try:
            from qgis.PyQt.QtGui import QColor
            saved = QColor(getter())
        except (RuntimeError, AttributeError, TypeError, ImportError):
            self._qgis_bridge_saved_selection_colour = None
            return
        # Held BEFORE the write, and kept through a failed one: the restore
        # reads this field, and clearing it is what loses the user's colour.
        self._qgis_bridge_saved_selection_colour = saved
        try:
            setter(QColor(255, 255, 0, 60))
            canvas.refresh()
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _restore_bridge_selection_colour(self) -> None:
        """Put the user's selection colour back. Never raises."""
        saved = getattr(self, "_qgis_bridge_saved_selection_colour", None)
        if saved is None:
            return
        self._qgis_bridge_saved_selection_colour = None
        try:
            canvas = self.iface.mapCanvas()
            setter = getattr(canvas, "setSelectionColor", None)
            if setter is not None:
                setter(saved)
                canvas.refresh()
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _apply_bridge_vertex_search_radius(self) -> None:
        """Guarantee a non-zero "search radius for vertex edits" so the native
        vertex tool can decide which vertex a click means. A user who set it to
        0 gets a vertex tool that feels dead; we save their value and restore it
        on exit. Best-effort; a read/write failure just leaves it as it was."""
        try:
            from qgis.core import QgsSettings
            settings = QgsSettings()
            key = "qgis/digitizing/search_radius_vertex_edit"
            unit_key = "qgis/digitizing/search_radius_vertex_edit_unit"
            prior = settings.value(key, None)
            prior_unit = settings.value(unit_key, None)
            self._qgis_bridge_saved_search_radius = (prior, prior_unit)
            try:
                current = float(prior) if prior is not None else 0.0
            except (TypeError, ValueError):
                current = 0.0
            if current <= 0.0:
                # 10 pixels is QGIS's own default and matches our snap feel.
                settings.setValue(key, 10)
                # Write the unit the way QGIS itself stores it (an enum NAME),
                # so we never depend on its int-parsing fallback.
                settings.setValue(unit_key, "Pixels")
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError):
            self._qgis_bridge_saved_search_radius = None

    def _restore_bridge_vertex_search_radius(self) -> None:
        """Put the user's vertex-edit search radius back exactly. Never raises."""
        saved = getattr(self, "_qgis_bridge_saved_search_radius", None)
        if saved is None:
            return
        prior, prior_unit = saved
        self._qgis_bridge_saved_search_radius = None
        key = "qgis/digitizing/search_radius_vertex_edit"
        unit_key = "qgis/digitizing/search_radius_vertex_edit_unit"
        try:
            from qgis.core import QgsSettings
            settings = QgsSettings()
            if prior is None:
                settings.remove(key)
            else:
                settings.setValue(key, prior)
            if prior_unit is None:
                settings.remove(unit_key)
            else:
                settings.setValue(unit_key, prior_unit)
        except (RuntimeError, AttributeError, TypeError, ImportError):
            pass

    def _restore_bridge_setting(self, restore, *args) -> None:
        """Run ONE restore step under its own guard, and log a failure.

        The steps are independent, and the teardown drops every saved value once
        it has run them, so a raise in one used to cost the user the settings the
        steps after it would have put back."""
        try:
            restore(*args)
        except Exception as exc:  # noqa: BLE001 -- one failure must not cost the others
            self._log_bridge_failure("qgis_bridge_restore", exc)

    def _restore_bridge_editing_aids(self) -> None:
        """Put the project's snapping / topology / avoid-overlap back exactly as
        they were before enter. Each step is isolated so one failure cannot leave
        a later one unrestored. Never raises."""
        saved = getattr(self, "_qgis_bridge_saved_aids", None)
        if not saved:
            return
        from qgis.core import QgsProject
        proj = QgsProject.instance()
        try:
            proj.setTopologicalEditing(bool(saved.get("topo")))
        except (RuntimeError, AttributeError):
            pass
        try:
            avoid_mode = saved.get("avoid_mode")
            if avoid_mode is not None and hasattr(proj, "setAvoidIntersectionsMode"):
                proj.setAvoidIntersectionsMode(avoid_mode)
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            layers = [ly for ly in (saved.get("avoid_layers") or [])
                      if self._is_layer_valid(ly)]
            if hasattr(proj, "setAvoidIntersectionsLayers"):
                proj.setAvoidIntersectionsLayers(layers)
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            snap = saved.get("snap")
            if snap:
                # Onto the project's LIVE config, so the per-layer settings
                # stay the ones QGIS is holding now. Restoring a saved config
                # object here put back layers the user had deleted meanwhile.
                cfg = proj.snappingConfig()
                cfg.setEnabled(snap["enabled"])
                cfg.setMode(snap["mode"])
                if snap["type"] is not None:
                    setter = (getattr(cfg, "setTypeFlag", None)
                              or getattr(cfg, "setType", None))
                    if setter is not None:
                        setter(snap["type"])
                cfg.setTolerance(snap["tolerance"])
                cfg.setUnits(snap["units"])
                proj.setSnappingConfig(cfg)
        except (RuntimeError, AttributeError, TypeError, KeyError):
            pass

    # ------------------------------------------------------------------
    # Native geometry UI
    # ------------------------------------------------------------------

    def _remember_bridge_vertex_editor_visibility(self) -> None:
        """Snapshot any existing Vertex Editor dock before a native tool runs."""
        self._qgis_bridge_vertex_dock_visibility = {}
        for dock in self._bridge_vertex_editor_docks():
            try:
                self._qgis_bridge_vertex_dock_visibility[id(dock)] = (
                    dock, bool(dock.isVisible()))
            except (RuntimeError, AttributeError):
                pass

    def _hide_bridge_opened_vertex_editors(self) -> None:
        """Hide Vertex Editor only when it was not visible before this bridge."""
        if not getattr(self, "_qgis_bridge_active", False):
            return
        saved = self._qgis_bridge_vertex_dock_visibility
        for dock in self._bridge_vertex_editor_docks():
            prior = saved.get(id(dock))
            if prior is None:
                saved[id(dock)] = (dock, False)
                was_visible = False
            else:
                was_visible = prior[1]
            if was_visible:
                continue
            try:
                dock.setVisible(False)
            except (RuntimeError, AttributeError):
                pass

    def _restore_bridge_vertex_editor_visibility(self) -> None:
        """Restore the exact Vertex Editor visibility from before the bridge."""
        for dock, visible in self._qgis_bridge_vertex_dock_visibility.values():
            try:
                dock.setVisible(bool(visible))
            except (RuntimeError, AttributeError):
                pass

    def _sync_bridge_delete_corner(self, class_name: str) -> None:
        """Show Delete corner only while the vertex tool has a corner picked.

        QgsVertexTool is not in the PyQGIS API, so there is no selection signal
        to connect to; the reliable proxy is the QGIS Vertex Editor, which QGIS
        fills with the locked feature's vertices the moment a corner is picked.
        Read its row count each poll tick and drive the button from it."""
        dock = getattr(self, "dock_widget", None)
        setter = getattr(dock, "set_qgis_bridge_delete_corner_visible", None)
        if setter is None:
            return
        picked = class_name in _VERTEX_TOOL_CLASSES and self._bridge_vertex_is_locked()
        try:
            setter(picked)
        except (RuntimeError, AttributeError):
            pass

    def _bridge_vertex_is_locked(self) -> bool:
        """True when the Vertex Editor holds a locked feature's vertices (the
        user clicked a corner). The bridge force-hides that dock, but its table
        model still reflects the lock, so read the model row count, not
        isVisible()."""
        try:
            from qgis.PyQt.QtWidgets import QTableView
        except ImportError:
            return False
        for dock in self._bridge_vertex_editor_docks():
            try:
                for view in dock.findChildren(QTableView):
                    model = view.model()
                    if model is not None and model.rowCount() > 0:
                        return True
            except (RuntimeError, AttributeError, TypeError):
                continue
        return False

    def _bridge_vertex_editor_docks(self) -> list:
        """Find the optional QGIS Vertex Editor dock without relying on locale."""
        try:
            from qgis.PyQt.QtWidgets import QDockWidget
            docks = self.iface.mainWindow().findChildren(QDockWidget)
        except (RuntimeError, AttributeError, TypeError):
            return []
        found = []
        for dock in docks:
            try:
                key = f"{dock.objectName()} {dock.windowTitle()}".lower()
            except (RuntimeError, AttributeError):
                continue
            if "vertex" in key and ("editor" in key or "dock" in key):
                found.append(dock)
        return found

    def _arm_bridge_vertex_tool(self, layer) -> None:
        """Compatibility wrapper retained for older tests/controller paths."""
        try:
            self.iface.setActiveLayer(layer)
        except (RuntimeError, AttributeError):
            pass
        action = self._bridge_iface_action(
            "actionVertexToolActiveLayer", "actionVertexTool")
        if action is not None:
            try:
                action.trigger()
            except (RuntimeError, AttributeError):
                pass

    def _restore_bridge_map_tool(self) -> None:
        """Return the map tool the user had before we armed the vertex tool."""
        prev = self._qgis_bridge_prev_maptool
        if prev is None:
            return
        try:
            self.iface.mapCanvas().setMapTool(prev)
        except (RuntimeError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Dock banner (plan 1 owns the widget; call it guarded)
    # ------------------------------------------------------------------

    def _enter_bridge_banner(self) -> None:
        self._call_dock_bridge_banner("enter_qgis_bridge_state")

    def _leave_bridge_banner(self) -> None:
        self._call_dock_bridge_banner("leave_qgis_bridge_state")

    def _call_dock_bridge_banner(self, method: str) -> None:
        dock = getattr(self, "dock_widget", None)
        if dock is None:
            return
        fn = getattr(dock, method, None)
        if not callable(fn):
            return
        try:
            fn()
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _log_bridge_failure(self, stage: str, exc: BaseException) -> None:
        """Write one warning line for a swallowed bridge failure.

        The shared reporter writes the exception class and a traceback hash to
        the QGIS log panel, which is what a bug report carries, and nothing that
        identifies the machine. Never raises: every caller is already on a
        failure path where a second failure must stay invisible."""
        try:
            from ...core import telemetry_errors
            telemetry_errors.report_exception(
                exc, stage=stage, module="qgis_edit_bridge")
        except Exception:  # noqa: BLE001 - a second failure stays invisible
            pass  # nosec B110

    def _show_bridge_commit_error(self, errors: list) -> None:
        """Surface a commit failure without leaving the bridge: log it and warn
        on the message bar. The edit session stays open so the user can retry."""
        detail = "; ".join(str(e) for e in (errors or [])[:3])
        message = tr("QGIS could not save these edits. Fix the geometry and "
                     "click Done again.")
        try:
            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                f"QGIS edit bridge commit failed: {detail}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        try:
            from qgis.core import Qgis
            self.iface.messageBar().pushMessage(
                "AI Segmentation", message,
                level=Qgis.MessageLevel.Warning, duration=6)
        except (RuntimeError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Fold-back
    # ------------------------------------------------------------------

    def _bridge_fold_back(self, layer) -> int:
        """Fold the committed geometry back into the review via plan 1's
        ``_fold_qgis_edits_back`` (it owns the review's object model: it rebuilds
        ``_auto_objects`` from the layer's features, carrying class/score on the
        attributes, and reslices). Resolved defensively so a missing hook cannot
        break teardown; returns the object count after the fold, for telemetry."""
        hook = getattr(self, "_fold_qgis_edits_back", None)
        if callable(hook):
            try:
                hook(layer)
            except Exception as exc:  # noqa: BLE001 -- fold-back must not break teardown
                # It must not fail SILENTLY, though: a swallowed fold leaves the
                # raw committed edit on the layer. After a native split that
                # means both halves keep QGIS's copied det_id, so they paint the
                # same colour in Distinct mode with no trace at all. Log it (and
                # count it in telemetry) so this failure is diagnosable, then
                # still finish teardown.
                try:
                    from ...core import telemetry_errors
                    telemetry_errors.report_exception(
                        exc, stage="qgis_bridge_fold_back",
                        module="qgis_edit_bridge")
                except Exception:  # noqa: BLE001 - a second failure stays invisible
                    pass  # nosec B110
        try:
            return len(getattr(self, "_auto_objects", None) or [])
        except (TypeError, AttributeError):
            return 0

    def _restore_review_step_after_bridge(self) -> None:
        """Re-drive the current review step so its per-step primary reappears
        after the banner is dismissed, then re-arm Correct's resting selection
        tool. The bridge deliberately disarms that tool before QGIS owns the
        canvas; without this final re-arm, the restored Correct page looks
        ready but its polygons cannot be selected."""
        dock = getattr(self, "dock_widget", None)
        if dock is None:
            return
        fn = getattr(dock, "set_auto_review_step", None)
        if not callable(fn):
            return
        try:
            fn(int(getattr(self, "_auto_review_step", 1)))
        except (RuntimeError, AttributeError, TypeError):
            pass
        if getattr(self, "_auto_review_step", 1) == 1:
            try:
                self._arm_correct_select()
            except (RuntimeError, AttributeError):
                pass

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _track_bridge(self, outcome: str, duration_ms: int | None = None,
                      features: int | None = None) -> None:
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_qgis_edit_bridge(
                run_id=getattr(self, "_auto_run_id", "") or "",
                outcome=outcome, duration_ms=duration_ms, features=features)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
