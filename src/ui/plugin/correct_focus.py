




















from __future__ import annotations

from qgis.PyQt.QtWidgets import QApplication

from ...core.qt_compat import WaitCursor
from ..canvas_palette import SESSION_DIM_FILL_STR, SESSION_DIM_STROKE_STR






FOCUS_ID_NEW_OBJECT = -1


def dimmed_others_renderer(base_renderer, det_id: int):












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


            "outline_width": "0.4",
        })


        root.appendChild(QgsRuleBasedRenderer.Rule(
            dim, 0, 0, f'"det_id" IS NULL OR "det_id" <> {target}'))
        return QgsRuleBasedRenderer(root)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None


class CorrectFocusMixin:







    def _init_correct_focus_state(self) -> None:

        self._end_correct_wait()
        self._correct_focus_det_id = None
        self._correct_wait_active = False
        self._correct_wait_cursor_set = False

    def _active_correct_focus_det_id(self):





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

        try:
            self._correct_focus_det_id = None if det_id is None else int(det_id)
        except (TypeError, ValueError):
            self._correct_focus_det_id = None
        self._repaint_correct_focus()

    def _end_correct_focus(self) -> None:


        self._end_correct_wait()
        if getattr(self, "_correct_focus_det_id", None) is None:
            return
        self._correct_focus_det_id = None
        self._repaint_correct_focus()

    def _correct_focus_blocks_det_id(self, det_id) -> bool:



        focus = self._active_correct_focus_det_id()
        if focus is None:
            return False
        if det_id is None:
            return True
        try:
            return int(det_id) != focus
        except (TypeError, ValueError):
            return True





    def _repaint_correct_focus(self) -> None:





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





    def _correct_wait_showing(self) -> bool:

        return bool(getattr(self, "_correct_wait_active", False))

    def _begin_correct_wait(self) -> bool:








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
