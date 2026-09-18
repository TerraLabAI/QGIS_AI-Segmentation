







from __future__ import annotations

from qgis.core import (
    QgsGeometry,
    QgsPointXY,
    QgsRectangle,
)
from qgis.gui import QgsRubberBand

from ...core.qt_compat import PolygonGeometry
from ...core.shape_edits import KIND_MERGE
from ..canvas_palette import ZONE_FILL, ZONE_STROKE


class AutoZoneCanvasMixin:





    def _show_zone_polygon_band(self, geom: QgsGeometry) -> None:





        self._clear_zone_rubber_band()
        try:
            canvas = self.iface.mapCanvas()
            rb = QgsRubberBand(canvas, PolygonGeometry)


            rb.setColor(ZONE_FILL)
            rb.setStrokeColor(ZONE_STROKE)
            rb.setWidth(2)
            rb.setToGeometry(geom, None)
            self._zone_rubber_band = rb


            self._show_zone_delete_badge(canvas, anchor=self._polygon_badge_anchor(geom))
        except (RuntimeError, AttributeError):
            pass

    def _polygon_badge_anchor(self, geom: QgsGeometry) -> QgsPointXY:



        try:
            ring = geom.asPolygon()[0]
        except (IndexError, TypeError):
            bb = geom.boundingBox()
            return QgsPointXY(bb.xMaximum(), bb.yMaximum())
        best = ring[0]
        for p in ring[1:]:
            if p.y() > best.y() or (p.y() == best.y() and p.x() > best.x()):
                best = p
        return QgsPointXY(best)

    def _show_zone_delete_badge(self, canvas, rect: QgsRectangle = None,
                                anchor: QgsPointXY = None) -> None:


        from ..zone_selection_maptool import ZoneBadgeClickFilter, ZoneDeleteBadge, ZoneEscapeFilter
        self._remove_zone_delete_badge()
        if anchor is None:
            anchor = QgsPointXY(rect.xMaximum(), rect.yMaximum())
        badge = ZoneDeleteBadge(canvas)
        badge.set_anchor(anchor)
        self._zone_delete_badge = badge


        self._zone_badge_filter = ZoneBadgeClickFilter(
            badge, self._on_zone_badge_clicked, parent=canvas)
        canvas.viewport().installEventFilter(self._zone_badge_filter)
        self._zone_escape_filter = ZoneEscapeFilter(
            self._on_zone_escape, parent=canvas)
        canvas.installEventFilter(self._zone_escape_filter)

    def _remove_zone_delete_badge(self) -> None:




        if self._zone_escape_filter is not None:
            try:
                self.iface.mapCanvas().removeEventFilter(self._zone_escape_filter)
            except (RuntimeError, AttributeError):
                pass
            try:
                self._zone_escape_filter.deleteLater()
            except (RuntimeError, AttributeError):
                pass
            self._zone_escape_filter = None
        if self._zone_badge_filter is not None:
            try:
                self.iface.mapCanvas().viewport().removeEventFilter(
                    self._zone_badge_filter)
            except (RuntimeError, AttributeError):
                pass
            try:
                self._zone_badge_filter.deleteLater()
            except (RuntimeError, AttributeError):
                pass
            self._zone_badge_filter = None
        if self._zone_delete_badge is not None:
            try:
                scene = self.iface.mapCanvas().scene()
                scene.removeItem(self._zone_delete_badge)
            except (RuntimeError, AttributeError):
                pass
            self._zone_delete_badge = None

    def _on_zone_badge_clicked(self) -> None:

        if self._auto_worker is not None:
            return


        if self._auto_review is not None:
            self._on_auto_review_exit_clicked()
            return
        self._discard_auto_review()
        self._on_zone_cleared()
        if self.dock_widget:
            self.dock_widget.on_zone_deleted_from_canvas()

    def _route_escape(self) -> bool:















        if self._exemplar_maptool is not None:


            self._restore_maptool_after_exemplar()
            return True
        if self._auto_worker is not None:






            dock = self.dock_widget
            if dock is not None:
                try:
                    dock.arm_auto_cancel_confirm()
                except (RuntimeError, AttributeError):
                    pass
            self._on_auto_cancel_clicked()
            return True
        if getattr(self, "_auto_finalize_state", None) is not None:







            return True
        if self._auto_review is not None:



            if getattr(self, "_shape_edit_mode", None) == KIND_MERGE:
                self._on_shape_draw_cancelled()
                return True




            if getattr(self, "_qgis_bridge_active", False):
                return self._route_escape_qgis_bridge()


            if getattr(self, "_refine_add_mode_active", False):
                return self._route_escape_add_mode()


            if getattr(self, "_refine_handoff_active", False):
                self._on_reshape_done()
                return True



            on_correct = getattr(self, "_auto_review_step", 0) == 1
            if on_correct and getattr(self, "_correct_selected_idx", None) is not None:
                self._set_correct_selection(None)
                return True
            self._on_auto_review_exit_clicked()
            return True





        tool = self._zone_selection_tool
        try:
            mid_draw = tool is not None and self.iface.mapCanvas().mapTool() is tool and tool.has_points()
        except (RuntimeError, AttributeError):
            mid_draw = False
        if mid_draw:
            tool.clear_selection()
            return True
        self._on_auto_exit_clicked()
        return True

    def _route_enter(self) -> bool:



        dock = self.dock_widget
        if dock is None or self._auto_worker is not None:
            return False
        if self._auto_review is not None:
            try:
                if dock.auto_export_btn.isVisible() and dock.auto_export_btn.isEnabled():
                    self._on_auto_export_clicked()
                    return True
            except (RuntimeError, AttributeError):
                pass
            return False
        try:
            if dock.auto_detect_btn.isVisible() and dock.auto_detect_btn.isEnabled():
                self._on_auto_detect_requested()
                return True
        except (RuntimeError, AttributeError):
            pass
        return False

    def _on_zone_escape(self) -> bool:






        if self._exemplar_maptool is not None:


            return False
        return self._route_escape()

    def _on_auto_escape_shortcut(self) -> None:




        self._route_escape()

    def _set_zone_badge_enabled(self, enabled: bool) -> None:

        if self._zone_delete_badge is not None:
            try:
                self._zone_delete_badge.set_enabled(enabled)
            except (RuntimeError, AttributeError):
                pass

    def _set_zone_band_fill_visible(self, visible: bool) -> None:





        rb = self._zone_rubber_band
        if rb is None:
            return
        try:
            from qgis.PyQt.QtGui import QColor
            rb.setFillColor(
                ZONE_FILL if visible else QColor(0, 0, 0, 0))
            rb.update()
        except (RuntimeError, AttributeError):
            pass

    def _clear_zone_rubber_band(self) -> None:

        self._remove_zone_delete_badge()
        self._clear_zone_tile_grid()
        if self._zone_rubber_band is not None:
            self._safe_remove_rubber_band(self._zone_rubber_band)
            self._zone_rubber_band = None

    def _set_auto_zone_overlays_visible(self, visible: bool) -> None:









        for attr in ("_zone_grid_rubber_band", "_zone_delete_badge"):
            item = getattr(self, attr, None)
            if item is None:
                continue
            try:
                item.setVisible(visible)
            except (RuntimeError, AttributeError):
                pass

    def _clear_auto_canvas(self) -> None:








        self._clear_exemplars()
        self._clear_zone_rubber_band()

        self._clear_free_zone_review_outline()
        self._remove_auto_selection_layer()


        self._auto_grid_suppressed = False
