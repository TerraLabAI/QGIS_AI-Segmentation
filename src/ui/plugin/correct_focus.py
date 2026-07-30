"""One polygon at a time in the review's Correct step: the canvas half.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); a mixin like its
siblings, so state lives on the plugin instance.

`set_correct_session_active` is the panel half of the same idea: while a fix
session runs, the dock shows only what belongs to the polygon under edit. This
file is the canvas half. While a session owns one polygon:

- every other polygon drops to a very pale grey (`SESSION_DIM_*`);
- every other polygon stops answering a click or a hover, so a stray click
  cannot move the session onto a shape the user was not editing;
- while the model is working, the polygon under edit wears a dashed outline
  and the cursor turns busy, so the wait is visible instead of guessed.

The focused polygon is DERIVED from the two session flags on every read
(`_active_correct_focus_det_id`), never trusted from the stored id alone. A
grey map or a busy cursor that outlives its session is worse than neither, and
the ways out are many (Save, Undo, Delete, Escape, a step change, a method
switch, the review closing, unload), so no exit is asked to remember this.
"""
from __future__ import annotations

from qgis.PyQt.QtWidgets import QApplication

from ...core.qt_compat import WaitCursor
from ..canvas_palette import SESSION_DIM_FILL_STR, SESSION_DIM_STROKE_STR


def dimmed_others_renderer(base_renderer, det_id: int):
    """Wrap a layer renderer so only ``det_id`` keeps its colours.

    The focused polygon keeps the EXACT look its display mode gave it: the
    mode's renderer is converted to rules and hung under one filter, rather
    than rebuilt as a single symbol, so Distinct keeps its per-object hue and
    Confidence its ramp. Everything else falls through to the pale grey rule,
    which is a flat symbol and therefore cheaper per feature than the
    data-defined fill it replaces.

    Returns None when this build cannot convert the renderer, which every
    caller reads as "leave the plain mode renderer alone".
    """
    if base_renderer is None:
        return None
    try:
        from qgis.core import QgsFillSymbol, QgsRuleBasedRenderer
    except ImportError:
        return None
    try:
        target = int(det_id)
    except (TypeError, ValueError):
        return None
    try:
        converted = QgsRuleBasedRenderer.convertFromRenderer(base_renderer)
        if converted is None:
            return None
        root = QgsRuleBasedRenderer.Rule(None)
        focused = QgsRuleBasedRenderer.Rule(None, 0, 0, f'"det_id" = {target}')
        for child in converted.rootRule().children():
            focused.appendChild(child.clone())
        root.appendChild(focused)
        dim = QgsFillSymbol.createSimple({
            "color": SESSION_DIM_FILL_STR,
            "outline_color": SESSION_DIM_STROKE_STR,
            # The dim fill alone cannot hold a small polygon on dark imagery,
            # so the outline carries it and is wider than the review's own.
            "outline_width": "0.4",
        })
        # A row with no det_id is greyed too: it cannot be the polygon the
        # session opened on, since that one was named by its det_id.
        root.appendChild(QgsRuleBasedRenderer.Rule(
            dim, 0, 0, f'"det_id" IS NULL OR "det_id" <> {target}'))
        return QgsRuleBasedRenderer(root)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None


class CorrectFocusMixin:
    """The Correct step's canvas focus: dim the others, gate their clicks,
    and show the model wait on the polygon under edit."""

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _init_correct_focus_state(self) -> None:
        """Fresh focus and wait state; called with the rest of the review's."""
        self._end_correct_wait()
        self._correct_focus_det_id = None
        self._correct_wait_active = False
        self._correct_wait_cursor_set = False

    def _active_correct_focus_det_id(self):
        """The det_id a live fix session owns, or None.

        Read off the session flags every time. The stored id is a cache, so an
        exit that forgets to clear it cannot leave the map grey or the other
        polygons deaf to a click."""
        if not (getattr(self, "_refine_handoff_active", False)
                or getattr(self, "_qgis_bridge_active", False)):
            return None
        det_id = getattr(self, "_correct_focus_det_id", None)
        if det_id is None:
            return None
        try:
            return int(det_id)
        except (TypeError, ValueError):
            return None

    def _begin_correct_focus(self, det_id) -> None:
        """Put the canvas on ONE polygon for the fix session about to open."""
        try:
            self._correct_focus_det_id = None if det_id is None else int(det_id)
        except (TypeError, ValueError):
            self._correct_focus_det_id = None
        self._repaint_correct_focus()

    def _end_correct_focus(self) -> None:
        """Give every polygon its colour and its clicks back, and take any
        waiting state down with it. Idempotent, so every exit may call it."""
        self._end_correct_wait()
        if getattr(self, "_correct_focus_det_id", None) is None:
            return
        self._correct_focus_det_id = None
        self._repaint_correct_focus()

    def _correct_focus_blocks_det_id(self, det_id) -> bool:
        """True when a click or a hover on this polygon must be ignored:
        a session is open on a different one. The polygon under edit always
        answers, and with no session open nothing is blocked."""
        focus = self._active_correct_focus_det_id()
        if focus is None:
            return False
        if det_id is None:
            return True
        try:
            return int(det_id) != focus
        except (TypeError, ValueError):
            return True

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _repaint_correct_focus(self) -> None:
        """Re-run the two renderer funnels so the dim goes on or comes off.

        Both funnels end in `_apply_correct_focus_dim`, so a reslice, a display
        mode switch or an edit-state swap keeps whatever this decided without
        knowing the focus exists."""
        layer = getattr(self, "_auto_selection_layer", None)
        if layer is not None:
            try:
                if layer.isValid():
                    self._apply_review_display_mode(layer)
            except (RuntimeError, AttributeError):
                pass
        refresh = getattr(self, "_refresh_handoff_display_renderers", None)
        if refresh is not None:
            try:
                refresh()
            except (RuntimeError, AttributeError):
                pass

    def _apply_correct_focus_dim(self, layer) -> None:
        """Grey out every polygon but the one a fix session works on.

        Called at the tail of the display-mode apply on the review layer and on
        the two handoff seed layers, which are the only places their renderers
        are set. No-op with no session open, so it is safe to call from there
        unconditionally."""
        det_id = self._active_correct_focus_det_id()
        if det_id is None or layer is None:
            return
        try:
            base = layer.renderer()
            if base is None:
                return
            wrapped = dimmed_others_renderer(base.clone(), det_id)
            if wrapped is None:
                return
            layer.setRenderer(wrapped)
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, TypeError):
            pass

    # ------------------------------------------------------------------
    # The model wait
    # ------------------------------------------------------------------

    def _correct_wait_showing(self) -> bool:
        """True while the polygon under edit is waiting on the model."""
        return bool(getattr(self, "_correct_wait_active", False))

    def _begin_correct_wait(self) -> bool:
        """The model is working on the polygon under edit: dash its outline and
        turn the cursor busy. ONE push, guarded by the flag, so the pop below
        is always its match.

        Returns True only when THIS call started the wait, which is what a
        caller wrapping a blocking predict checks before ending it: a predict
        that runs under a read the session already announced must not take the
        read's cursor down with it."""
        if getattr(self, "_correct_wait_active", False):
            return False
        if not getattr(self, "_refine_handoff_active", False):
            return False
        self._correct_wait_active = True
        if not getattr(self, "_headless", False):
            try:
                QApplication.setOverrideCursor(WaitCursor)
                self._correct_wait_cursor_set = True
            except (RuntimeError, AttributeError):
                self._correct_wait_cursor_set = False
        self._refresh_correct_wait_band()
        return True

    def _end_correct_wait(self) -> None:
        """Take the waiting outline and the busy cursor back down.

        Never gated on the session still being live: a teardown in the middle
        of a read has to give the cursor back, and that is exactly the path
        that would otherwise leave QGIS busy for the rest of the session."""
        if not getattr(self, "_correct_wait_active", False):
            return
        self._correct_wait_active = False
        if getattr(self, "_correct_wait_cursor_set", False):
            self._correct_wait_cursor_set = False
            try:
                QApplication.restoreOverrideCursor()
            except (RuntimeError, AttributeError):
                pass
        self._refresh_correct_wait_band()

    def _refresh_correct_wait_band(self) -> None:
        """Repaint the active polygon through its own style funnel, so the
        dashed waiting outline goes on and comes off in one place."""
        style = getattr(self, "_apply_mask_band_style", None)
        if style is None:
            return
        try:
            style()
            band = getattr(self, "mask_rubber_band", None)
            if band is not None:
                band.update()
        except (RuntimeError, AttributeError):
            pass
