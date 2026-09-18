






from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog


class ManualWorkflowRefineMixin:





    def _safe_restore_canvas_focus(self):

        try:
            from qgis.PyQt.QtWidgets import (
                QApplication,
                QDoubleSpinBox,
                QLineEdit,
                QPlainTextEdit,
                QSpinBox,
                QTextEdit,
            )
            focused = QApplication.instance().focusWidget()
            if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit,
                                    QSpinBox, QDoubleSpinBox)):
                return
            self.iface.mapCanvas().setFocus()
        except (RuntimeError, AttributeError):
            pass

    def _on_size_filter_changed(self, min_m2: float, max_m2: float) -> None:



        del max_m2
        self._refine_min_size_m2 = max(0.0, float(min_m2 or 0.0))
        self._refine_max_size_m2 = 0.0

    def _on_fill_holes_size_changed(self, max_m2: float) -> None:



        self._refine_fill_holes_max_m2 = max(0.0, float(max_m2 or 0.0))

    def _on_clean_edges_changed(self, clean_px: float) -> None:



        self._refine_clean = max(0.0, float(clean_px or 0.0))

    def _on_outline_budget_changed(self, simplify_px: float, points_pct: int) -> None:




        self._refine_simplify = max(0.0, float(simplify_px or 0.0))
        self._refine_points_pct = max(1, min(100, int(points_pct or 100)))

    def _on_refine_settings_changed(self, simplify: int, smooth: int, expand: int,
                                    fill_holes: bool, right_angles: bool = False):





        QgsMessageLog.logMessage(
            f"Refine settings: simplify={self._refine_simplify}, "
            f"points_pct={self._refine_points_pct}, smooth={smooth}, "
            f"expand={expand}, fill_holes={fill_holes}, "
            f"right_angles={right_angles}, "
            f"min_area={self._refine_min_area} (auto)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        self._refine_smooth = smooth
        self._refine_expand = expand
        self._refine_fill_holes = fill_holes
        self._refine_ortho = right_angles





        self._reshape_hover_preview()




        if self._apply_handoff_refine_settings():
            self._safe_restore_canvas_focus()
            return



        self._update_mask_visualization()

        self._safe_restore_canvas_focus()
