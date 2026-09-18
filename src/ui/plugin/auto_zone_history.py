







from __future__ import annotations

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
    QgsRectangle,
)

from ...core.i18n import tr
from ...core.qt_compat import geometry_op_succeeded


class AutoZoneHistoryMixin:







    def _on_history_rerun_requested(self, entry: dict) -> None:






        from ...core.qt_compat import safe_single_shot
        payload = dict(entry) if isinstance(entry, dict) else {}
        safe_single_shot(0, self.dock_widget or self.iface.mainWindow(),
                         lambda: self._history_rerun_here(payload))

    def _on_history_reuse_prompt_requested(self, prompt: str) -> None:


        from ...core.qt_compat import safe_single_shot
        text = str(prompt or "")
        safe_single_shot(0, self.dock_widget or self.iface.mainWindow(),
                         lambda: self._history_reuse_prompt(text))

    def _history_rerun_here(self, entry: dict) -> None:





        if self._auto_worker is not None or self._auto_review is not None:
            self._history_rerun_busy_notice()
            return
        prompt = (entry.get("prompt") or "").strip()
        authid = str(entry.get("crs") or "")
        geom = self._zone_geom_from_wkt(entry.get("zone_wkt"), authid)
        if geom is None:
            geom = self._zone_geom_from_extent(entry.get("extent"), authid)
        self._enter_auto_flow_for_history(prompt)
        if geom is None:


            self._track_history_rerun("new_zone")
            return


        self._on_zone_polygon_drawn(geom)
        self._match_stored_tile_count(entry.get("tiles"))
        self._track_history_rerun("same_zone")

    def _match_stored_tile_count(self, wanted) -> None:













        try:
            target = int(wanted or 0)
        except (TypeError, ValueError):
            return
        if target <= 0:
            return
        dock = self.dock_widget
        layer = self._get_active_raster_layer()
        if dock is None or layer is None or self._auto_zone is None:
            return
        try:
            slider = dock.auto_detail_slider
            zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
            best, best_gap = None, None
            for level in range(slider.minimum(), slider.maximum() + 1):
                grid = self._grid_for_detail(layer, zone_in_layer, level)
                if not grid or grid[3] <= 0:
                    continue
                gap = abs(grid[3] - target)
                if best_gap is None or gap < best_gap:
                    best, best_gap = level, gap
                if gap == 0:
                    break
            if best is None or best == slider.value():
                return
            dock.set_auto_detail_value(best)



            self._auto_detail_user_locked = True
            self._auto_detail_lock_prompt = (
                self._resolved_auto_object_class() or "").lower()
            self._update_credit_estimate()
        except (RuntimeError, AttributeError):
            return

    def _history_reuse_prompt(self, prompt: str) -> None:


        if self._auto_worker is not None or self._auto_review is not None:
            self._history_rerun_busy_notice()
            return
        self._enter_auto_flow_for_history((prompt or "").strip())
        self._track_history_rerun("new_zone")

    def _enter_auto_flow_for_history(self, prompt: str) -> None:



        dock = self.dock_widget
        if dock is None:
            return
        from ..ai_segmentation_dockwidget import Mode
        if dock._mode != Mode.AUTOMATIC:
            try:
                dock._on_mode_selected(Mode.AUTOMATIC)
            except (RuntimeError, AttributeError):
                pass
        if self._tile_manager is None:
            self._setup_auto_mode()
        try:
            self._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass



        self._reset_auto_flow_to_start(exit_path="new_run")
        try:
            dock._on_auto_start_clicked()
        except (RuntimeError, AttributeError):
            pass


        if prompt:
            try:
                dock.set_prompt_text(prompt)
            except (RuntimeError, AttributeError):
                pass

    def _zone_geom_from_wkt(self, wkt, authid: str) -> QgsGeometry | None:













        if not wkt or not isinstance(wkt, str) or not authid:
            return None
        geom = QgsGeometry.fromWkt(wkt)
        if geom is None or geom.isEmpty():
            return None
        src = QgsCoordinateReferenceSystem(authid)
        if not src.isValid():
            return None
        try:
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        except (RuntimeError, AttributeError):
            return None
        if canvas_crs.isValid() and src != canvas_crs:
            try:
                xform = QgsCoordinateTransform(
                    src, canvas_crs, QgsProject.instance())



                if not geometry_op_succeeded(geom.transform(xform)):
                    return None
            except Exception:  # nosec B110
                return None
        return None if geom.isEmpty() else geom

    def _zone_geom_from_extent(self, extent, authid: str) -> QgsGeometry | None:



        if not extent or len(extent) != 4 or not authid:
            return None
        try:
            xmin, ymin, xmax, ymax = (float(v) for v in extent)
        except (TypeError, ValueError):
            return None
        rect = QgsRectangle(xmin, ymin, xmax, ymax)
        if rect.isEmpty() or rect.width() <= 0 or rect.height() <= 0:
            return None
        geom = QgsGeometry.fromRect(rect)
        src = QgsCoordinateReferenceSystem(authid)
        if not src.isValid():
            return None
        try:
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        except (RuntimeError, AttributeError):
            return None
        if canvas_crs.isValid() and src != canvas_crs:
            try:
                xform = QgsCoordinateTransform(
                    src, canvas_crs, QgsProject.instance())


                if not geometry_op_succeeded(geom.transform(xform)):
                    return None
            except Exception:  # nosec B110
                return None
            if geom.isEmpty():
                return None
        return geom

    def _history_rerun_busy_notice(self) -> None:


        try:
            self.iface.messageBar().pushInfo(
                "AI Segmentation",
                tr("Finish or cancel the current detection before "
                   "re-running a past one."),
            )
        except (RuntimeError, AttributeError):
            pass

    def _track_history_rerun(self, kind: str) -> None:

        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_history_rerun(kind)
        except Exception:
            pass  # nosec B110
