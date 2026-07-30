"""Display colour modes of the Automatic selection layer: the four renderers,
the mode switch and its legend line.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from ...core.qt_compat import symbol_fill_color_property
from ..canvas_palette import OUTLINE_MODE_STROKE_STR

# Random display mode: how many distinct hues one symbol per bucket buys.
# Enough that a same-colour neighbour is rare, few enough that the categorized
# renderer stays cheaper than the per-feature expression it replaced (the frame
# grows with the category count, and past a few dozen the win erodes).
RANDOM_MODE_BUCKETS = 48
RANDOM_MODE_SATURATION = 0.78
RANDOM_MODE_LIGHTNESS = 0.55
# Degrees per bucket. Both Random renderers hue on the SAME bucket wheel, so a
# layer that enters or leaves an edit session keeps every colour it had.
RANDOM_MODE_HUE_STEP = 360.0 / RANDOM_MODE_BUCKETS


def random_mode_fill_expression() -> str:
    """The Distinct-mode fill colour as one QGIS expression, for every layer
    that paints it.

    The review layer and the handoff seed layers show the SAME objects one
    after the other. A second formula over the same det_id repaints the whole
    map the moment the user opens a polygon to edit it, which reads as the
    colours being reshuffled. So there is one expression, and it lands on the
    same bucket wheel as the categorized renderer.
    """
    bucket = ('(to_int(abs(coalesce("det_id", $id))) * 67)'
              f" % {RANDOM_MODE_BUCKETS}")
    return (f"color_hsla(floor(({bucket}) * {RANDOM_MODE_HUE_STEP}),"
            f" {round(RANDOM_MODE_SATURATION * 100)},"
            f" {round(RANDOM_MODE_LIGHTNESS * 100)}, 205)")


class AutoReviewDisplayMixin:
    """The Normal / Outline / Confidence / Distinct renderers and the switch between them."""

    def _apply_review_heatmap_renderer(self, layer) -> None:
        """Color the review selection layer as a confidence heatmap: high score
        (confident) = yellow, low score (uncertain) = purple, so dragging the
        confidence slider visibly shows what gets cut. Viridis is used instead of
        a red->green ramp because red-green is the most common colorblind
        confusion; Viridis is perceptually uniform and colorblind-safe (the legend
        states the yellow/purple meaning). One translucent fill symbol with a
        data-defined ramp over the per-object 'score' field (stable 0..1 domain).
        Best-effort: a render failure must never break the review."""
        try:
            from qgis.core import QgsFillSymbol, QgsProperty, QgsSingleSymbolRenderer, QgsStyle
            symbol = QgsFillSymbol.createSimple({
                "color": "0,150,80,110",
                "outline_color": "40,40,40,180",
                "outline_width": "0.2",
            })
            sl = symbol.symbolLayer(0)
            # Viridis ships with QGIS, but guard so an unusual install without it
            # still gets a colorblind-safe (non red-green) ramp.
            ramp_name = ("Viridis"
                         if QgsStyle.defaultStyle().colorRamp("Viridis")
                         else "Spectral")
            expr = f"ramp_color('{ramp_name}', coalesce(\"score\", 0))"
            # FillColor property key spelling moved across QGIS 3.x -> 4; the
            # shim resolves whichever this build exposes.
            prop_key = symbol_fill_color_property()
            sl.setDataDefinedProperty(prop_key, QgsProperty.fromExpression(expr))
            symbol.setOpacity(0.55)
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _random_hue_symbol(self, hue: int):
        """One Random-mode fill at ``hue`` degrees.

        Saturation 78 + lightness 55 + alpha 205, combined with the symbol
        opacity, lands ~0.60 effective: readable over dark imagery (rooftops)
        instead of washed out, and still translucent enough to see through.
        """
        from qgis.core import QgsFillSymbol
        from qgis.PyQt.QtGui import QColor
        c = QColor.fromHslF(
            (hue % 360) / 360.0, RANDOM_MODE_SATURATION, RANDOM_MODE_LIGHTNESS)
        symbol = QgsFillSymbol.createSimple({
            "color": f"{c.red()},{c.green()},{c.blue()},205",
            "outline_color": "20,20,20,200",
            "outline_width": "0.2",
        })
        symbol.setOpacity(0.75)
        return symbol

    def _apply_review_random_renderer(self, layer) -> None:
        """Colour each detection a distinct pseudo-random colour (stable per
        detection id) so touching or merged objects are easy to tell apart: a
        visual debug aid. ONE identical style from the first live tile through
        the review and up to export: any style switch at review-open reads as
        the results "changing" once the run ends. Best-effort: a render failure
        must never break the review.

        Two renderers for one look, picked on whether the layer is mid-edit.

        At rest the colour comes from a CATEGORIZED renderer over
        ``"det_id" % N``. The equivalent data-defined fill colour is evaluated
        per feature on every repaint and QGIS caches none of it, which on a
        dense run is the single largest per-frame cost the review carries.
        Classifying on a plain field expression skips the whole data-defined
        symbol path (measured: the frame drops by a third at review scale).

        The classifier must NOT mention ``$id``: referencing it costs more than
        the data-defined colour it replaces. A categorized renderer also has no
        answer for a feature that has no det_id yet, and one born inside a
        native edit session has none for the moment it takes the bridge to write
        it. So while the layer is editable the per-feature expression stays in
        force, with ``$id`` as its stand-in: the user is zoomed onto one object
        then, where the frame is a few milliseconds either way.
        ``_watch_review_edit_state`` swaps them back on QGIS's own
        editingStarted/editingStopped, so no caller has to remember.
        """
        try:
            editable = bool(layer.isEditable())
        except (RuntimeError, AttributeError):
            editable = False
        try:
            if editable:
                self._apply_random_renderer_editing(layer)
            else:
                self._apply_random_renderer_categorized(layer)
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, ImportError, ValueError):
            pass

    def _apply_random_renderer_categorized(self, layer) -> None:
        """Random mode at rest: one symbol per ``"det_id" % N`` bucket.

        ``* 67`` spreads adjacent detection ids across the wheel, so two objects
        that appeared one after the other never land on neighbouring hues. The
        catch-all category covers a row with no det_id (the hand-edited dissolve
        path) so nothing can ever render invisible.

        ``to_int`` is load-bearing, not decoration: a categorized renderer keys
        its categories on the value's STRING form, and a bare ``abs()`` returns
        a double, so every row evaluated to "37.0", matched no category named
        "37", and the whole layer fell through to the catch-all in one colour.
        """
        from qgis.core import QgsCategorizedSymbolRenderer, QgsRendererCategory
        cats = [
            QgsRendererCategory(
                i, self._random_hue_symbol(int(i * RANDOM_MODE_HUE_STEP)), str(i))
            for i in range(RANDOM_MODE_BUCKETS)
        ]
        cats.append(QgsRendererCategory(
            None, self._random_hue_symbol(0), "", True))
        layer.setRenderer(QgsCategorizedSymbolRenderer(
            f'(to_int(abs("det_id")) * 67) % {RANDOM_MODE_BUCKETS}', cats))

    def _apply_random_renderer_editing(self, layer) -> None:
        """Random mode while the layer is in a native edit session: the same
        colour as at rest, read per feature instead of from categories.

        The formula lands on the SAME bucket wheel as the resting renderer, and
        that is the whole point: the two swap on every
        editingStarted/editingStopped, so a formula that hued on a free 0..359
        circle repainted every polygon on the map the moment one of them was
        clicked, and repainted them all back on Save.

        Nothing here treats a new shape specially any more. The bridge gives a
        feature its final det_id as it is born, so a hand-drawn polygon and each
        half of a split each arrive with their own identity and the plain
        formula already tells them apart. ``$id`` stays only as the stand-in for
        the moment between the add and that write, and for a layer that carries
        no det_id field at all.
        """
        from qgis.core import QgsFillSymbol, QgsProperty, QgsSingleSymbolRenderer
        symbol = QgsFillSymbol.createSimple({
            "color": "120,120,120,120",
            "outline_color": "20,20,20,200",
            "outline_width": "0.2",
        })
        symbol.symbolLayer(0).setDataDefinedProperty(
            symbol_fill_color_property(),
            QgsProperty.fromExpression(random_mode_fill_expression()))
        symbol.setOpacity(0.75)
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    def _watch_review_edit_state(self, layer) -> None:
        """Re-apply the display mode whenever THIS layer enters or leaves a
        native edit session, so Random mode can run its cheap renderer at rest
        and its split-aware one while editing.

        Wired to the layer's own signals rather than to the edit bridge's entry
        and exit points: the bridge has several ways out (commit, rollback,
        tool change, teardown) and a missed one would leave the wrong renderer
        on the map. Connected once per layer; the id guard is what stops a
        second call from stacking a duplicate connection.
        """
        try:
            layer_id = layer.id()
        except (RuntimeError, AttributeError):
            return
        if getattr(self, "_review_edit_style_layer_id", None) == layer_id:
            return
        try:
            layer.editingStarted.connect(self._on_review_layer_edit_state)
            layer.editingStopped.connect(self._on_review_layer_edit_state)
        except (RuntimeError, AttributeError):
            return
        self._review_edit_style_layer_id = layer_id

    def _on_review_layer_edit_state(self) -> None:
        """The review layer entered or left an edit session: re-pick the
        renderer for the current display mode.

        The swap is DEFERRED by one turn of the event loop, never run here.
        QGIS emits editingStarted and editingStopped from inside its own
        commitChanges() and rollBack(), so a setRenderer plus triggerRepaint on
        that frame reenters the layer while the commit is still unwinding. The
        edit bridge drops ITS editingStopped before committing, but it cannot
        know about this second subscriber. A renderer has no deadline, so it
        waits.
        """
        if getattr(self, "_review_edit_style_pending", False):
            return
        self._review_edit_style_pending = True
        try:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._reapply_review_display_after_edit_state)
        except (RuntimeError, AttributeError, ImportError):
            # No timer, no deferral: keeping the colour right is worth less than
            # not reentering the commit, so drop this swap. The next mode switch
            # or review rebuild re-applies it.
            self._review_edit_style_pending = False

    def _reapply_review_display_after_edit_state(self) -> None:
        """The deferred half of ``_on_review_layer_edit_state``: re-read the
        selection layer (a turn is long enough for it to have been removed with
        the session) and re-apply the display mode.

        The flag is cleared in a ``finally`` because ``_apply_review_display_mode``
        calls ``_watch_review_edit_state``, the setup for the very signal whose
        handler queued this call."""
        try:
            layer = getattr(self, "_auto_selection_layer", None)
            if layer is None:
                return
            try:
                if not layer.isValid():
                    return
            except (RuntimeError, AttributeError):
                return
            self._apply_review_display_mode(layer)
        finally:
            self._review_edit_style_pending = False

    def _apply_review_outline_renderer(self, layer) -> None:
        """Red outline only, NO fill, so the imagery INSIDE each detection stays
        visible: the filled modes (Normal blue, Confidence heatmap, Random) tint
        the interior and hide what is underneath. Red matches the saved-polygon
        convention. Best-effort: a render failure must never break the review."""
        try:
            from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer
            symbol = QgsFillSymbol.createSimple({
                "style": "no",                       # no fill brush = see through
                "outline_color": OUTLINE_MODE_STROKE_STR,  # red outline
                "outline_width": "0.5",
                "outline_style": "solid",
            })
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _apply_review_display_mode(self, layer) -> None:
        """Apply the current review DISPLAY colour mode to the selection layer:
        'normal' (blue review fill), 'outline' (red outline only, see-through),
        'confidence' (green->red heatmap on the per-object score) or 'random' (one
        colour per object). The SAME renderer serves the live run and the review
        (one visual identity end to end). Visual only: never touches geometry,
        filters or export."""
        if layer is None:
            return
        # Random mode picks its renderer on the layer's edit state, so it needs
        # to hear about a session opening or closing. Idempotent, so calling it
        # from the one place every mode switch already goes through is enough.
        self._watch_review_edit_state(layer)
        mode = getattr(self, "_auto_display_mode", "normal")
        if mode == "random":
            self._apply_review_random_renderer(layer)
        elif mode == "outline":
            self._apply_review_outline_renderer(layer)
        elif mode == "confidence":
            self._apply_review_heatmap_renderer(layer)
        else:
            # Normal: colour the detections with the SAME per-prompt hue the
            # export uses (committed_color_for_prompt), so the review preview
            # and the exported layer match instead of the review being a fixed
            # blue that has nothing to do with the object. Promptless runs fall
            # back to the legacy colour inside committed_color_for_prompt.
            try:
                from ...core.layer_conventions import make_committed_renderer
                from ...core.output_store import committed_color_for_prompt
                raw_prompt = (getattr(self, "_auto_run_ctx", None) or {}).get("prompt")
                raw_prompt = raw_prompt or (getattr(self, "_auto_review", None) or {}).get("prompt")
                prompt = str(raw_prompt or "").strip()
                layer.setRenderer(
                    make_committed_renderer(color=committed_color_for_prompt(prompt)))
                layer.triggerRepaint()
            except (RuntimeError, AttributeError, ImportError):
                pass
        # One polygon at a time: while a fix session is open on one of them,
        # every other polygon greys out. Applied LAST and here only, because
        # this method is the single funnel the review layer's renderer goes
        # through, so a reslice or an edit-state swap keeps the dim for free.
        self._apply_correct_focus_dim(layer)

    def _apply_review_display_to_session(self) -> None:
        """Push the current display mode onto what a fix session put on the
        canvas: the two handoff seed layers and the active mask band.

        The review's own selection layer is out of the project for as long as an
        AI fix or an Add lane runs, so a mode switch that only reached that
        layer would do nothing while the user is correcting. The native QGIS
        bridge needs nothing here: it edits the selection layer itself, which
        keeps its renderer, and QGIS paints the one selected feature over it."""
        try:
            self._refresh_handoff_display_renderers()
        except (RuntimeError, AttributeError):
            pass
        try:
            self._apply_mask_band_style()
            band = getattr(self, "mask_rubber_band", None)
            if band is not None:
                band.update()
        except (RuntimeError, AttributeError):
            pass

    def _display_legend_text(self, mode: str) -> str:
        """Legend line for a display mode: swatch dots in the real renderer
        colours + what they mean (single source: the dock-side helper, so the
        build-time seed and every mode switch render the same line)."""
        from ..dock.auto_review_build import display_legend_html
        return display_legend_html(mode)

    def _seed_review_display_mode(self) -> None:
        """Default the display colour to Random (one distinct colour per object)
        so touching instances read apart at a glance. Called at RUN START (so the
        live results stream in Random, not blue) and again when review opens.
        Syncs the dock combo + legend signal-free so control and renderer never
        desync; the user's later switches then work normally. The committed
        export keeps the red-outline convention (make_committed_renderer)."""
        self._auto_display_mode = "random"
        if self.dock_widget is not None:
            try:
                self.dock_widget.set_auto_display_mode("random")
                self.dock_widget.set_display_legend(
                    self._display_legend_text("random"))
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_display_mode_changed(self, mode: str) -> None:
        """Review display colour mode changed (Normal / Outline / Confidence /
        Random): store it and re-render whatever is drawing the detections right
        now. Visual only, no re-detection. The switch works at ANY moment,
        including in the middle of a correction, so it also reaches the layers a
        fix session put on the canvas in the review layer's place."""
        self._auto_display_mode = (
            mode if mode in ("normal", "outline", "confidence", "random")
            else "normal")
        if self._auto_selection_layer is not None:
            self._apply_review_display_mode(self._auto_selection_layer)
        self._apply_review_display_to_session()
        # Muted legend line under the combo: what the colours MEAN for this mode.
        if self.dock_widget is not None:
            try:
                self.dock_widget.set_display_legend(
                    self._display_legend_text(self._auto_display_mode))
            except (RuntimeError, AttributeError):
                pass
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_review_display_mode(
                mode=self._auto_display_mode,
                run_id=getattr(self, "_auto_run_id", "") or "")
        except Exception:
            pass  # nosec B110
