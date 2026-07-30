






from __future__ import annotations

from ...core.qt_compat import symbol_fill_color_property
from ..canvas_palette import OUTLINE_MODE_STROKE_STR





RANDOM_MODE_BUCKETS = 48
RANDOM_MODE_SATURATION = 0.78
RANDOM_MODE_LIGHTNESS = 0.55


RANDOM_MODE_HUE_STEP = 360.0 / RANDOM_MODE_BUCKETS


def _random_mode_style() -> tuple[int, float, float, float]:



    try:
        from ...core.server_dials import dial_in_range
        buckets = int(dial_in_range(
            "tuning.review.random_mode_buckets", RANDOM_MODE_BUCKETS, 8, 120))
        saturation = dial_in_range(
            "tuning.review.random_mode_saturation", RANDOM_MODE_SATURATION, 0.0, 1.0)
        lightness = dial_in_range(
            "tuning.review.random_mode_lightness", RANDOM_MODE_LIGHTNESS, 0.0, 1.0)
    except Exception:  # noqa: BLE001
        buckets, saturation, lightness = (
            RANDOM_MODE_BUCKETS, RANDOM_MODE_SATURATION, RANDOM_MODE_LIGHTNESS)
    return buckets, 360.0 / buckets, saturation, lightness


def random_mode_fill_expression() -> str:









    buckets, hue_step, saturation, lightness = _random_mode_style()
    bucket = ('(to_int(abs(coalesce("det_id", $id))) * 67)'
              f" % {buckets}")
    return (f"color_hsla(floor(({bucket}) * {hue_step}),"
            f" {round(saturation * 100)},"
            f" {round(lightness * 100)}, 205)")


class AutoReviewDisplayMixin:


    def _apply_review_heatmap_renderer(self, layer) -> None:








        try:
            from qgis.core import QgsFillSymbol, QgsProperty, QgsSingleSymbolRenderer, QgsStyle
            symbol = QgsFillSymbol.createSimple({
                "color": "0,150,80,110",
                "outline_color": "40,40,40,180",
                "outline_width": "0.2",
            })
            sl = symbol.symbolLayer(0)


            ramp_name = ("Viridis"
                         if QgsStyle.defaultStyle().colorRamp("Viridis")
                         else "Spectral")
            expr = f"ramp_color('{ramp_name}', coalesce(\"score\", 0))"


            prop_key = symbol_fill_color_property()
            sl.setDataDefinedProperty(prop_key, QgsProperty.fromExpression(expr))
            symbol.setOpacity(0.55)
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _random_hue_symbol(self, hue: int, saturation: float = RANDOM_MODE_SATURATION,
                           lightness: float = RANDOM_MODE_LIGHTNESS):






        from qgis.core import QgsFillSymbol
        from qgis.PyQt.QtGui import QColor
        c = QColor.fromHslF((hue % 360) / 360.0, saturation, lightness)
        symbol = QgsFillSymbol.createSimple({
            "color": f"{c.red()},{c.green()},{c.blue()},205",
            "outline_color": "20,20,20,200",
            "outline_width": "0.2",
        })
        symbol.setOpacity(0.75)
        return symbol

    def _apply_review_random_renderer(self, layer) -> None:


























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












        from qgis.core import QgsCategorizedSymbolRenderer, QgsRendererCategory
        buckets, hue_step, saturation, lightness = _random_mode_style()
        cats = [
            QgsRendererCategory(
                i, self._random_hue_symbol(int(i * hue_step), saturation, lightness), str(i))
            for i in range(buckets)
        ]
        cats.append(QgsRendererCategory(
            None, self._random_hue_symbol(0, saturation, lightness), "", True))
        layer.setRenderer(QgsCategorizedSymbolRenderer(
            f'(to_int(abs("det_id")) * 67) % {buckets}', cats))

    def _apply_random_renderer_editing(self, layer) -> None:
















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



        self._review_edit_style_pending = False

    def _on_review_layer_edit_state(self) -> None:











        if getattr(self, "_review_edit_style_pending", False):
            return
        owner = getattr(self, "dock_widget", None)
        if owner is None:
            return
        self._review_edit_style_pending = True
        try:
            from ...core.qt_compat import safe_single_shot



            safe_single_shot(
                0, owner, self._reapply_review_display_after_edit_state)
        except (RuntimeError, AttributeError, ImportError):



            self._review_edit_style_pending = False

    def _reapply_review_display_after_edit_state(self) -> None:







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




        try:
            from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer
            symbol = QgsFillSymbol.createSimple({
                "style": "no",
                "outline_color": OUTLINE_MODE_STROKE_STR,
                "outline_width": "0.5",
                "outline_style": "solid",
            })
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _apply_review_display_mode(self, layer) -> None:






        if layer is None:
            return



        self._watch_review_edit_state(layer)
        mode = getattr(self, "_auto_display_mode", "normal")
        if mode == "confidence":




            try:
                flat = not self._run_scores_rank_objects()
            except (RuntimeError, AttributeError, TypeError):
                flat = False
            if flat:
                mode = "random"
        if mode == "random":
            self._apply_review_random_renderer(layer)
        elif mode == "outline":
            self._apply_review_outline_renderer(layer)
        elif mode == "confidence":
            self._apply_review_heatmap_renderer(layer)
        else:





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




        self._apply_correct_focus_dim(layer)

    def _apply_review_display_to_session(self) -> None:








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



        from ..dock.auto_review_build import display_legend_html
        return display_legend_html(mode)

    def _seed_review_display_mode(self) -> None:






        self._auto_display_mode = "random"
        if self.dock_widget is not None:
            try:
                self.dock_widget.set_auto_display_mode("random")
                self.dock_widget.set_display_legend(
                    self._display_legend_text("random"))
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_display_mode_changed(self, mode: str) -> None:





        self._auto_display_mode = (
            mode if mode in ("normal", "outline", "confidence", "random")
            else "normal")
        if self._auto_selection_layer is not None:
            self._apply_review_display_mode(self._auto_selection_layer)
        self._apply_review_display_to_session()

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
