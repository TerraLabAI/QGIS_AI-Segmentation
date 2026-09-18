













from __future__ import annotations

from dataclasses import dataclass

from qgis.core import Qgis, QgsCoordinateTransform, QgsGeometry, QgsMessageLog, QgsProject

from .shared import free_zone_cap_km2



FREE_FIT_MIN_KM2 = 0.05



FREE_FIT_MARGIN = 0.99


@dataclass
class FreeZoneFit:


    geom: QgsGeometry | None
    requested_km2: float
    processed_km2: float
    left_km2: float | None

    drawn: QgsGeometry | None = None
    crs: object = None


class AutoZoneFreeFitMixin:


    def _free_zone_budget_km2(self) -> tuple[float, float | None] | None:




        try:
            usage_known = bool(self._last_usage)
            _credits, is_free = self._auto_credit_snapshot()
        except (RuntimeError, AttributeError):
            return None
        if not usage_known or not is_free:
            return None
        budget = free_zone_cap_km2()
        left = None
        try:
            if self.dock_widget is not None:
                left = self.dock_widget._auto_km2_left()
        except (RuntimeError, AttributeError):
            left = None
        if left is not None:
            budget = min(budget, float(left))
        return budget, left

    def _fit_zone_to_free_budget(self, geom, crs=None) -> FreeZoneFit | None:







        budget = self._free_zone_budget_km2()
        if budget is None:
            return None
        budget_km2, left = budget

        def measure(shape) -> float:
            return self._zone_geodesic_area_km2(
                self._zone_billable_shape(shape, crs), crs)

        requested = measure(geom)
        if requested <= budget_km2 or budget_km2 < FREE_FIT_MIN_KM2:
            return None
        from ...core.zone_fit import fit_zone_to_area
        fitted = fit_zone_to_area(
            QgsGeometry(geom), budget_km2 * FREE_FIT_MARGIN, measure)
        processed = measure(fitted) if fitted is not None else 0.0
        if fitted is None or processed <= 0:
            return FreeZoneFit(None, requested, 0.0, left)
        return FreeZoneFit(fitted, requested, processed, left,
                           drawn=QgsGeometry(geom), crs=crs)

    def _record_free_zone_fit(self, fit: FreeZoneFit, notify: bool = True) -> None:




        self._auto_free_zone_fit = (fit.requested_km2, fit.processed_km2)
        self._auto_free_zone_fit_info = fit
        QgsMessageLog.logMessage(
            f"Auto detection: free zone of {fit.requested_km2:.2f} km2 kept to "
            f"{fit.processed_km2:.2f} km2 (what the month has left)",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_auto_zone_free_clipped(
                km2_requested=fit.requested_km2,
                km2_processed=fit.processed_km2,
                km2_left=fit.left_km2,
            )
        except Exception:
            pass  # nosec B110

    def _refit_stored_zone_for_free(self) -> FreeZoneFit | None:








        if self._auto_zone is None:
            return None
        zone_crs = self._zone_source_crs(self._auto_zone)
        shape = (QgsGeometry(self._auto_zone_polygon)
                 if self._auto_zone_polygon is not None
                 else QgsGeometry.fromRect(self._auto_zone))


        prior = self._current_free_zone_fit()
        if (prior is not None and prior.drawn is not None
                and prior.crs == zone_crs):
            shape = QgsGeometry(prior.drawn)
        fit = self._fit_zone_to_free_budget(shape, zone_crs)
        if fit is None or fit.geom is None:
            return fit
        from qgis.core import QgsRectangle
        self._store_auto_zone(QgsRectangle(fit.geom.boundingBox()), crs=zone_crs)
        self._auto_zone_polygon = QgsGeometry(fit.geom)
        band = QgsGeometry(fit.geom)
        try:
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            if (zone_crs is not None and zone_crs.isValid()
                    and canvas_crs.isValid() and zone_crs != canvas_crs):
                band.transform(QgsCoordinateTransform(
                    zone_crs, canvas_crs, QgsProject.instance()))
            self._show_zone_polygon_band(band)
        except (RuntimeError, AttributeError):
            pass
        self._record_free_zone_fit(fit)
        return fit

    def _offer_free_zone_fit_upsell(self) -> None:








        fit = getattr(self, "_auto_free_zone_fit", None)
        info = self._current_free_zone_fit()
        if info is not None and info.geom is not None:
            self._show_free_zone_rest_band(info)
        else:
            self._hide_free_zone_rest_band()
        if not self.dock_widget:
            return
        try:
            if fit is None:
                self.dock_widget.set_free_zone_fit_offer(None, None)
            else:
                self.dock_widget.set_free_zone_fit_offer(fit[0], fit[1])
        except (RuntimeError, AttributeError):
            pass



    def _current_free_zone_fit(self) -> FreeZoneFit | None:



        if getattr(self, "_auto_free_zone_fit", None) is None:
            return None
        return getattr(self, "_auto_free_zone_fit_info", None)

    def _free_clip_question_passes(self, on_detect) -> bool:



        if getattr(self, "_free_clip_confirmed", False):
            self._free_clip_confirmed = False
            return True
        fit = self._current_free_zone_fit()
        if fit is None or fit.geom is None:
            return True
        if getattr(self, "_free_clip_dialog", None) is not None:
            try:
                self._free_clip_dialog.raise_()
            except RuntimeError:
                self._free_clip_dialog = None
            return False
        from ..dialogs.free_zone_clip_dialog import FreeZoneClipDialog
        from ..dock.ui_refresh_credits import format_km2_surface
        left = fit.left_km2 if fit.left_km2 is not None else free_zone_cap_km2()
        dialog = FreeZoneClipDialog(
            zone_km2=format_km2_surface(fit.requested_km2),
            left_km2=format_km2_surface(left),
            done_km2=format_km2_surface(fit.processed_km2),
            on_answer=lambda choice: self._on_free_clip_answered(
                fit, choice, on_detect),
            parent=self.iface.mainWindow(),
        )
        self._free_clip_dialog = dialog
        self._show_free_zone_rest_band(fit)
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(
                trigger="free_zone_clip_prompt",
                cta_source="free_zone_clip_prompt")
        except Exception:
            pass  # nosec B110
        dialog.open()
        return False

    def _on_free_clip_answered(self, fit: FreeZoneFit, choice: str, on_detect) -> None:
        from ...core.free_clip_choice import plan_for_choice
        dialog = getattr(self, "_free_clip_dialog", None)
        self._free_clip_dialog = None
        if dialog is not None:
            try:
                dialog.deleteLater()
            except RuntimeError:
                pass  # nosec B110
        self._hide_free_zone_rest_band()
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_auto_zone_free_clip_choice(
                choice=choice,
                km2_requested=fit.requested_km2,
                km2_processed=fit.processed_km2,
                km2_left=fit.left_km2,
            )
        except Exception:
            pass  # nosec B110
        plan = plan_for_choice(choice)
        if plan.upgrade:
            from ...core.pro_page_link import open_pro_page
            open_pro_page("plugin_free_zone_clip_prompt", "free_zone_clip_prompt",
                          parent=self.iface.mainWindow())
        elif plan.redraw:

            if self._auto_worker is None and self._auto_review is None:
                self._on_zone_cleared()
                if self.dock_widget:

                    self.dock_widget.on_zone_deleted_from_canvas()
                self._activate_zone_drawing()
        elif plan.run:

            if (self._current_free_zone_fit() is fit
                    and self._auto_worker is None):
                self._free_clip_confirmed = True
                try:
                    on_detect()
                finally:
                    self._free_clip_confirmed = False

    def _show_free_zone_rest_band(self, fit: FreeZoneFit) -> None:


        self._hide_free_zone_rest_band()
        if fit.drawn is None:
            return
        try:
            from qgis.gui import QgsRubberBand
            from qgis.PyQt.QtCore import Qt
            from qgis.PyQt.QtGui import QColor

            from ...core.qt_compat import PolygonGeometry
            from ..canvas_palette import ZONE_REST_STROKE, ZONE_REST_WIDTH
            canvas = self.iface.mapCanvas()
            shape = QgsGeometry(fit.drawn)
            canvas_crs = canvas.mapSettings().destinationCrs()
            if (fit.crs is not None and fit.crs.isValid()
                    and canvas_crs.isValid() and fit.crs != canvas_crs):
                shape.transform(QgsCoordinateTransform(
                    fit.crs, canvas_crs, QgsProject.instance()))
            rb = QgsRubberBand(canvas, PolygonGeometry)
            rb.setFillColor(QColor(0, 0, 0, 0))
            rb.setStrokeColor(ZONE_REST_STROKE)
            rb.setWidth(ZONE_REST_WIDTH)
            rb.setLineStyle(Qt.PenStyle.DashLine)
            rb.setToGeometry(shape, None)
            self._free_zone_rest_band = rb
        except (RuntimeError, AttributeError, TypeError):
            self._free_zone_rest_band = None

    def _clear_free_zone_review_outline(self) -> None:



        try:
            self._hide_free_zone_rest_band()
        except Exception:  # noqa: BLE001
            self._free_zone_rest_band = None

    def _hide_free_zone_rest_band(self) -> None:
        rb = getattr(self, "_free_zone_rest_band", None)
        self._free_zone_rest_band = None
        if rb is not None:
            self._safe_remove_rubber_band(rb)
