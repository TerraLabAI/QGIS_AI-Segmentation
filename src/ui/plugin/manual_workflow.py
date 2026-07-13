"""Manual session: start/stop segmentation, map tools, save polygon, export.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import math
import os

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import (
    QMessageBox,
)

from ...core.i18n import tr
from ...core.qt_compat import PolygonGeometry
from ..canvas_palette import KEPT_FILL, KEPT_STROKE
from ..error_report_dialog import show_error_report
from ..shortcut_filter import ShortcutFilter
from .shared import (
    SETTINGS_KEY_LAST_MANUAL_SESSION_TS,
    SETTINGS_KEY_TUTORIAL_SHOWN,
    _FIELD_TYPE_DOUBLE,
    _FIELD_TYPE_STRING,
    _add_features_fast,
    _apply_fast_render,
)


class ManualWorkflowMixin:
    """Manual session: start/stop segmentation, map tools, save polygon, export."""

    def _on_start_segmentation(self, layer: QgsRasterLayer):
        if self.predictor is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Not Ready"),
                tr("Please wait for the SAM model to load.")
            )
            return

        # Validate layer BEFORE resetting session to avoid leaving broken state
        if not self._is_layer_valid(layer):
            QgsMessageLog.logMessage(
                "Layer was deleted before segmentation could start",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return

        try:
            layer_name = layer.name().replace(" ", "_")
            raster_path = os.path.normcase(layer.source())
        except RuntimeError:
            QgsMessageLog.logMessage(
                "Layer deleted during segmentation start",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return

        self._reset_session()

        self._current_layer = layer
        self._current_layer_name = layer_name

        # Detect online layer (WMS, XYZ, WMTS, WCS, ArcGIS)
        self._is_online_layer = self._is_online_provider(layer)

        # Detect if layer is non-georeferenced (pixel coordinate mode)
        self._is_non_georeferenced_mode = (
            not self._is_online_layer and not self._is_layer_georeferenced(layer)
        )
        if self._is_non_georeferenced_mode:
            QgsMessageLog.logMessage(
                "Non-georeferenced image detected - using pixel coordinate mode. "
                "Polygons will be created in pixel coordinates.",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )

        if self._is_online_layer:
            QgsMessageLog.logMessage(
                f"Online layer detected ({layer.dataProvider().name()})",
                "AI Segmentation", level=Qgis.MessageLevel.Info
            )

        # CRS guard: a GEOREFERENCED raster with no valid CRS would segment in an
        # identity transform and land polygons in the wrong place. Non-georef
        # pixel mode (which QGIS runs without a CRS) and online layers are exempt,
        # so this only blocks the genuinely broken case.
        if not self._is_online_layer and not self._is_non_georeferenced_mode and not layer.crs().isValid():
            show_error_report(
                self.iface.mainWindow(),
                tr("Invalid CRS"),
                tr("This layer has no valid coordinate reference system. "
                   "Set one in Layer Properties before segmenting."),
                error_code="invalid_crs",
            )
            return

        # Validate layer extent
        if not self._is_online_layer:
            try:
                ext = layer.extent()
                if ext and not ext.isEmpty():
                    coords = (ext.xMinimum(), ext.yMinimum(),
                              ext.xMaximum(), ext.yMaximum())
                    if any(math.isnan(c) or math.isinf(c) for c in coords):
                        show_error_report(
                            self.iface.mainWindow(),
                            tr("Invalid Layer"),
                            tr("Layer extent contains invalid coordinates "
                               "(NaN/Inf). Check the raster file."),
                            error_code="invalid_layer",
                        )
                        return
            except RuntimeError:
                pass

        # Rotation guard: the crop-to-ground mapping assumes an axis-aligned
        # affine, exactly like Automatic (which blocks rotated rasters and
        # steers users HERE). A rotated/sheared local raster renders fine in
        # QGIS but every exported polygon would land rotated/offset, silently.
        # Reuses the Auto path's fail-open detector (local files only).
        if not self._is_online_layer and not self._is_non_georeferenced_mode and self._raster_is_rotated(layer):
            show_error_report(
                self.iface.mainWindow(),
                tr("Rotated raster"),
                tr("This raster is rotated. Convert it to an axis-aligned "
                   "GeoTIFF before segmenting."),
                error_code="rotated_raster",
            )
            return

        # Store raster path for on-demand crop extraction
        self._current_raster_path = raster_path

        # Set up CRS transforms (canvas CRS <-> raster CRS)
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        raster_crs = layer.crs() if layer else None
        self._canvas_to_raster_xform = None
        self._raster_to_canvas_xform = None
        if raster_crs and canvas_crs.isValid() and raster_crs.isValid():
            if canvas_crs != raster_crs:
                self._canvas_to_raster_xform = QgsCoordinateTransform(
                    canvas_crs, raster_crs, QgsProject.instance())
                self._raster_to_canvas_xform = QgsCoordinateTransform(
                    raster_crs, canvas_crs, QgsProject.instance())
                QgsMessageLog.logMessage(
                    f"CRS transform enabled: {canvas_crs.authid()} -> {raster_crs.authid()}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info
                )

        # Pre-warm the worker subprocess so SAM model loads while the
        # user positions their first click (reduces first-click latency)
        self.predictor.warm_up()

        # Mark the start time so segmentation_run telemetry can report duration.
        import time as _time
        self._segmentation_start_ts = _time.time()

        # Remember that this machine uses Manual mode: the next QGIS launch
        # pre-warms the model as soon as the predictor loads (_manual_used
        # _recently in env_setup), so a returning user's first click never
        # waits out the model load again.
        try:
            QSettings().setValue(
                SETTINGS_KEY_LAST_MANUAL_SESSION_TS,
                int(self._segmentation_start_ts))
        except Exception:  # noqa: BLE001 - heuristic only
            pass  # nosec B110

        self._activate_segmentation_tool()

        # Pre-encode the visible view while the user aims their first click:
        # warm_up() above only pre-starts the subprocess, so the first click
        # still paid the model-load tail + the first encode. singleShot lets
        # the Start UI paint before the GUI-thread crop extraction runs. The
        # Refine handoff (which reaches here with _refine_handoff_active set)
        # seeds its own whole-object encode instead.
        if not self._headless and not self._refine_handoff_active:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._prewarm_manual_encode)

    def _active_space_pan_tool(self):
        """Return the plugin-owned map tool currently active on the canvas
        that supports temporary space panning, or None."""
        try:
            current = self.iface.mapCanvas().mapTool()
        except (RuntimeError, AttributeError):
            return None
        for tool in (self.map_tool, self._zone_selection_tool):
            if tool is not None and current == tool:
                return tool
        return None

    def _activate_segmentation_tool(self):
        # Save the current map tool to restore it later
        current_tool = self.iface.mapCanvas().mapTool()
        if current_tool and current_tool != self.map_tool:
            self._previous_map_tool = current_tool

        self.iface.mapCanvas().setMapTool(self.map_tool)
        # Snapshot the session layer from the authoritative _current_layer (set by
        # _on_start_segmentation), not the combo: the handoff starts on the run's
        # raster, which the locked combo may not yet reflect.
        self.dock_widget.set_segmentation_active(True, layer=self._current_layer)

        # Install keyboard shortcut filter on both mainWindow and the canvas
        # viewport.  mainWindow catches keys when focus is elsewhere (e.g.
        # after dock widget updates steal focus).  The canvas viewport is
        # needed to intercept ShortcutOverride for Space *before* QGIS
        # activates its built-in pan-tool shortcut.
        if self._shortcut_filter is None:
            self._shortcut_filter = ShortcutFilter(self)
        self.iface.mainWindow().installEventFilter(self._shortcut_filter)
        canvas = self.iface.mapCanvas()
        canvas.viewport().installEventFilter(self._shortcut_filter)
        canvas.installEventFilter(self._shortcut_filter)

        # Show tutorial notification for first-time users
        self._show_tutorial_notification()

    def _show_tutorial_notification(self):
        """Show YouTube tutorial notification (once ever, persisted in QSettings)."""
        settings = QSettings()
        if settings.value(SETTINGS_KEY_TUTORIAL_SHOWN, False, type=bool):
            return
        settings.setValue(SETTINGS_KEY_TUTORIAL_SHOWN, True)

        from ...core.activation_manager import get_tutorial_url
        tutorial_url = get_tutorial_url()
        message = '{} <a href="{}">{}</a>'.format(
            tr("New here?"),
            tutorial_url,
            tr("Watch the tutorial"))

        self.iface.messageBar().pushMessage(
            "AI Segmentation",
            message,
            level=Qgis.MessageLevel.Info,
            duration=10
        )

    def _on_layer_combo_changed(self, layer):
        """Handle layer selection change in the combo box."""
        # Only care about segmentation reset if we're currently segmenting
        if not self._current_layer:
            return

        # Check if it's actually a different layer
        # Handle case where the C++ layer object was deleted
        try:
            new_layer_id = layer.id() if layer else None
            current_layer_id = self._current_layer.id() if self._current_layer else None
        except RuntimeError:
            # Layer was deleted, reset our reference
            self._current_layer = None
            return

        if new_layer_id == current_layer_id:
            return

        # Different layer selected while segmenting
        if self.iface.mapCanvas().mapTool() == self.map_tool:
            has_unsaved_mask = self.current_mask is not None
            has_unsaved_mask = has_unsaved_mask or bool(self._frozen_sessions)
            has_unsaved_mask = has_unsaved_mask or self._unfrozen_display_polygon is not None
            has_saved_polygons = len(self.saved_polygons) > 0

            if has_unsaved_mask or has_saved_polygons:
                polygon_count = len(self.saved_polygons)
                if has_unsaved_mask:
                    polygon_count += 1
                message = "{}\n\n{}".format(
                    tr("You have {count} unsaved polygon(s).").format(
                        count=polygon_count),
                    tr("Changing layer will discard your current segmentation. Continue?"))

                reply = QMessageBox.warning(
                    self.iface.mainWindow(),
                    tr("Change Layer?"),
                    message,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )

                if reply != QMessageBox.StandardButton.Yes:
                    self.dock_widget.layer_combo.blockSignals(True)
                    self.dock_widget.layer_combo.setLayer(self._current_layer)
                    self.dock_widget.layer_combo.blockSignals(False)
                    return
                try:
                    from ...core import telemetry
                    telemetry.track_manual_abandoned(
                        context="change_layer", polygon_count=polygon_count)
                except Exception:
                    pass  # nosec B110

            self._stopping_segmentation = True
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self._restore_previous_map_tool()
            self._stopping_segmentation = False
            self._reset_session()
            self.dock_widget.reset_session()

    def _on_save_polygon(self):
        """Save current mask as polygon (including any frozen crop sessions)."""
        if self._encoding_in_progress:
            return
        # Allow save if we have frozen sessions even without active mask
        has_active = self.current_mask is not None and self.current_transform_info is not None
        if not has_active and not self._frozen_sessions and self._unfrozen_display_polygon is None:
            return

        self._ensure_polygon_rubberband_sync()

        # Collect all geometry parts: frozen polygons + active mask.
        # An unfrozen session polygon (no numpy mask yet) counts as active.
        all_geoms = [s.polygon for s in self._frozen_sessions]
        if not has_active and self._unfrozen_display_polygon is not None:
            all_geoms.append(self._unfrozen_display_polygon)

        if has_active:
            # Shared refine tail (fill holes, expand, min region, simplify,
            # right angles, rounding, size window): saves exactly the preview.
            active_combined = self._refined_active_mask_geometry()
            if active_combined is not None and not active_combined.isEmpty():
                all_geoms.append(active_combined)

        if all_geoms:
            combined = QgsGeometry.unaryUnion(all_geoms)
        else:
            combined = None

        if combined and not combined.isEmpty():
            # Refine handoff: if this save overlaps an existing detection, fold
            # them into ONE polygon (complete-don't-stack). No-op in base Manual.
            combined = self._absorb_overlapping_saved(combined)
            # Per-instance identity: an object re-opened for editing keeps its
            # original det_id (its Random colour survives the edit); a brand-new
            # hand save gets a synthetic one. Score follows the same rule.
            origin = self._active_refine_origin_entry or {}
            origin_id = origin.get("det_id")
            # Store WKT (with effects), transform info, raw mask, points, and refine settings
            self.saved_polygons.append({
                "det_id": int(origin_id) if origin_id is not None
                else self._next_handoff_det_id(),
                "score": origin.get("score"),
                "manual_touched": self._refine_handoff_active,
                "geometry_wkt": combined.asWkt(),
                # Cache the parsed geometry (absorb/click/collect reuse it without
                # re-parsing WKT over a big handoff set).
                "geom_obj": combined,
                "transform_info": self.current_transform_info.copy() if self.current_transform_info else None,
                "raw_mask": self.current_mask.copy() if self.current_mask is not None else None,
                "points_positive": list(self.prompts.positive_points),
                "points_negative": list(self.prompts.negative_points),
                "refine_simplify": self._refine_simplify,
                "refine_smooth": self._refine_smooth,
                "refine_expand": self._refine_expand,
                "refine_fill_holes": self._refine_fill_holes,
                "refine_ortho": self._refine_ortho,
                "refine_min_area": self._refine_min_area,
                "refine_min_size_m2": self._refine_min_size_m2,
                "refine_max_size_m2": self._refine_max_size_m2,
                # No keep concept in a handoff: every detection on the page is
                # already part of the result, so a saved edit returns to the
                # same pending style as its neighbours. Base Manual keeps True
                # (its real bands are green by construction).
                "validated": not self._refine_handoff_active,
            })

            if self._refine_handoff_active:
                # Handoff: no per-object band; the seed layer draws it. One
                # incremental add (the absorb above already dropped anything it
                # merged), not a full rebuild of both seed layers per Save.
                self.saved_rubber_bands.append(None)
                if not self._handoff_add_entry_feature(self.saved_polygons[-1]):
                    self._rebuild_handoff_layers()
            else:
                saved_rb = QgsRubberBand(self.iface.mapCanvas(), PolygonGeometry)
                saved_rb.setColor(KEPT_STROKE)
                saved_rb.setFillColor(KEPT_FILL)
                saved_rb.setWidth(2)
                # Geometry is in raster CRS; transform to canvas CRS for display
                display_geom = QgsGeometry(combined)
                self._transform_geometry_to_canvas_crs(display_geom)
                saved_rb.setToGeometry(display_geom, None)
                self.saved_rubber_bands.append(saved_rb)

            QgsMessageLog.logMessage(
                f"Saved mask #{len(self.saved_polygons)}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )

            self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            self._update_handoff_progress()

            # Minimal telemetry: one event per successful segmentation run.
            try:
                import time as _time

                from ...core.telemetry import track_segmentation_run
                start_ts = getattr(self, "_segmentation_start_ts", None)
                duration_ms = int((_time.time() - start_ts) * 1000) if start_ts else None
                track_segmentation_run(success=True, duration_ms=duration_ms)
                self._segmentation_start_ts = None
                # Per-session counters for the manual_session_summary event.
                self._manual_saves_session = getattr(self, "_manual_saves_session", 0) + 1
                if getattr(self, "_manual_session_t0", None) is None:
                    self._manual_session_t0 = _time.time()
            except Exception:
                pass  # nosec B110

            # Note: We keep refinement settings in batch mode so the user can
            # apply the same expand/simplify to multiple masks

        # The saved object is committed (green): no longer the active editable
        # one. Delete-undo history is kept (the stack restores prior removals).
        self._is_refining_saved_object = False
        self._active_refine_origin_entry = None
        self._refine_geom_history = []
        if self.dock_widget:
            try:
                self.dock_widget.set_handoff_editing(False)
            except (RuntimeError, AttributeError):
                pass
        # Clear current state for next polygon (including frozen sessions)
        self.prompts.clear()
        self._mask_state_history = []
        self._frozen_sessions = []
        self._unfrozen_display_polygon = None
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        if self.map_tool:
            self.map_tool.clear_markers()
        self._clear_mask_visualization()
        self.current_mask = None
        self.current_score = 0.0
        self.current_low_res_mask = None
        self.dock_widget.set_point_count(0, 0)

        # Keep crop info so clicks in the same area reuse the encoding.

    def _on_export_layer(self):
        """Export all saved polygons + current unsaved mask to a new layer."""
        # Refine handoff: committing goes through the review's Finish, never a
        # direct export (it would dump the imported detections to a layer the
        # review would then commit AGAIN). Enter/Export = Back to review.
        if self._refine_handoff_active:
            self._on_back_to_review_clicked()
            return
        if self._exporting_in_progress:
            return
        self._exporting_in_progress = True
        try:
            self._on_export_layer_impl()
        except Exception:
            import traceback
            QgsMessageLog.logMessage(
                traceback.format_exc(),
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                tr("An unexpected error occurred during export. Please check the logs."),
                error_code="export_failed",
            )
        finally:
            self._exporting_in_progress = False

    def _on_export_layer_impl(self):
        """Internal export implementation."""
        self._ensure_polygon_rubberband_sync()

        has_active = self.current_mask is not None and self.current_transform_info is not None
        should_skip_export = not self.saved_polygons and not has_active
        should_skip_export = should_skip_export and not self._frozen_sessions
        should_skip_export = should_skip_export and self._unfrozen_display_polygon is None
        if should_skip_export:
            return  # Nothing to export

        polygons_to_export = list(self.saved_polygons)

        # Build current unsaved geometry: frozen sessions + active mask.
        # An unfrozen session polygon (no numpy mask yet) counts as active.
        current_geoms = [s.polygon for s in self._frozen_sessions]
        if not has_active and self._unfrozen_display_polygon is not None:
            current_geoms.append(self._unfrozen_display_polygon)

        if has_active:
            # Shared refine tail: exports exactly what the preview shows.
            active_combined = self._refined_active_mask_geometry()
            if active_combined is not None and not active_combined.isEmpty():
                current_geoms.append(active_combined)

        if current_geoms:
            combined = QgsGeometry.unaryUnion(current_geoms)
            if combined and not combined.isEmpty():
                polygons_to_export.append({
                    "geometry_wkt": combined.asWkt(),
                    "transform_info": self.current_transform_info.copy() if self.current_transform_info else None,
                })

        self._stopping_segmentation = True
        self.iface.mapCanvas().unsetMapTool(self.map_tool)
        self._restore_previous_map_tool()
        self._stopping_segmentation = False

        from ...core import output_store

        # Friendly display name ("Segmentation (3 Jul)", deduped); the table
        # name inside the shared GeoPackage is derived by the output store.
        layer_name = output_store.friendly_layer_name("")

        # Determine CRS
        # For non-georeferenced images, use a local pixel-based CRS
        if self._is_non_georeferenced_mode:
            # Use EPSG:3857 (Web Mercator) with pixel coordinates
            # This allows visualization while being clear it's not true geographic data
            crs = QgsCoordinateReferenceSystem("EPSG:3857")
            QgsMessageLog.logMessage(
                "Non-georeferenced mode: Using EPSG:3857 with pixel coordinates",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
        else:
            # Normal georeferenced mode
            crs_str = None
            for pg in polygons_to_export:
                ti = pg.get("transform_info")
                if ti:
                    crs_str = ti.get("crs", None)
                    if isinstance(crs_str, str) and crs_str.strip():
                        break
                    crs_str = None
            if crs_str is None and self.current_transform_info:
                val = self.current_transform_info.get("crs", None)
                if isinstance(val, str) and val.strip():
                    crs_str = val
            if crs_str is None:
                try:
                    if self._is_layer_valid() and self._current_layer.crs().isValid():
                        crs_str = self._current_layer.crs().authid()
                except RuntimeError:
                    pass
            if isinstance(crs_str, str) and crs_str.strip():
                crs = QgsCoordinateReferenceSystem(crs_str)
            else:
                crs = QgsCoordinateReferenceSystem("EPSG:4326")
                QgsMessageLog.logMessage(
                    "CRS could not be determined, falling back to EPSG:4326",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)

        # Create a temporary memory layer to build features
        temp_layer = QgsVectorLayer("MultiPolygon", layer_name, "memory")
        if not temp_layer.isValid():
            show_error_report(
                self.iface.mainWindow(),
                tr("Layer Creation Failed"),
                tr("Could not create the output layer."),
                error_code="layer_creation_failed",
            )
            return

        temp_layer.setCrs(crs)

        from ...core.layer_conventions import (
            apply_output_conventions,
            make_area_measurer,
            make_committed_renderer,
            repair_polygon,
            to_multipolygon,
        )

        # Minimal per-feature schema: an editable label plus the geodesic
        # measure. Run-level provenance (source raster, date) lives in the
        # layer metadata instead of being repeated on every row.
        pr = temp_layer.dataProvider()
        pr.addAttributes([
            QgsField("label", _FIELD_TYPE_STRING),
            QgsField("area_m2", _FIELD_TYPE_DOUBLE),
        ])
        temp_layer.updateFields()

        raster_name = ""
        try:
            if self._is_layer_valid() and self._current_layer:
                raster_name = self._current_layer.name()
        except RuntimeError:
            pass

        # Add features to temp layer. One measurer for the whole batch (setEllipsoid
        # loads from the SRS DB, so rebuilding it per feature is slow on big runs).
        measurer = make_area_measurer(crs)
        features_to_add = []
        for i, polygon_data in enumerate(polygons_to_export):
            feature = QgsFeature(temp_layer.fields())

            # Reconstruct geometry from WKT
            geom_wkt = polygon_data.get("geometry_wkt")
            if not geom_wkt:
                QgsMessageLog.logMessage(
                    f"Polygon {i + 1} has no WKT data",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning
                )
                continue

            geom = QgsGeometry.fromWkt(geom_wkt)

            if geom and not geom.isEmpty():
                # Repair instead of silently dropping invalid rings, then coerce
                # to a polygon-only MultiPolygon (a collection would be rejected).
                geom = to_multipolygon(repair_polygon(geom) or geom)
                if geom is None or geom.isEmpty():
                    continue
                feature.setGeometry(geom)
                feature.setAttributes(["", float(measurer.measureArea(geom))])
                features_to_add.append(feature)

        if not features_to_add:
            QgsMessageLog.logMessage(
                "Export aborted: no valid geometries produced from mask",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                tr("No valid polygons could be created from the selection. "
                   "Try adjusting the refine settings or making a new selection."),
                error_code="export_failed",
            )
            return

        _add_features_fast(pr, features_to_add)
        temp_layer.updateExtents()

        try:
            source_layer = self._current_layer if self._is_layer_valid() else None
        except RuntimeError:
            source_layer = None
        # Name of the raster this run was made on: drives the per-raster
        # sub-group under "AI Segmentation" so outputs group by source layer.
        try:
            source_name = source_layer.name() if source_layer is not None else ""
        except RuntimeError:
            source_name = ""

        # Destination chosen by the user: append to an existing layer (the
        # incremental-digitization path, e.g. adding shapes to an Automatic
        # detections layer) or, by default, create a new layer below. A stale
        # or failed append falls through to the new-layer path so work is never
        # lost.
        if self._append_manual_run_if_requested(
                temp_layer, source_name, features_to_add):
            return

        # Write into the shared per-project GeoPackage (one table per run).
        # The store handles directory priority (project, raster dir, home)
        # and falls back to a standalone per-run file if the shared file is
        # locked or unwritable.
        result = output_store.write_run_table(
            temp_layer,
            prompt="",
            source_layer=source_layer,
            fallback_stem="segmentation",
        )

        if result is None:
            # Keep the user's work on screen: the features already live in
            # the memory layer, so show that instead of dead-ending.
            temp_layer.setRenderer(make_committed_renderer(
                color=output_store.committed_color_for_prompt("")))
            output_store.add_committed_layer(temp_layer, source_name=source_name)
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                "{}\n\n{}".format(
                    tr("Could not save layer to file:"),
                    tr("Your polygons were added as a temporary layer so "
                       "nothing is lost.")),
                error_code="export_failed",
            )
            return

        result_layer = result.layer
        gpkg_path = result.gpkg_path
        layer_name = result_layer.name()

        # Add under the source raster's sub-group in the "AI Segmentation" group
        output_store.add_committed_layer(result_layer, source_name=source_name)

        if result.used_fallback:
            msg = tr(
                "Could not write to {name}. Saved to a separate file instead."
            ).format(name=output_store.GPKG_FILENAME)
            self.iface.messageBar().pushMessage(
                "AI Segmentation", msg,
                level=Qgis.MessageLevel.Warning, duration=8)

        # Log the layer extent for debugging
        layer_extent = result_layer.extent()
        QgsMessageLog.logMessage(
            f"Exported layer extent: xmin={layer_extent.xMinimum():.2f}, ymin={layer_extent.yMinimum():.2f}, "
            f"xmax={layer_extent.xMaximum():.2f}, ymax={layer_extent.yMaximum():.2f}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        QgsMessageLog.logMessage(
            f"Layer CRS: {result_layer.crs().authid()}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        QgsMessageLog.logMessage(
            f"Saved to: {gpkg_path}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        # Committed look: solid outline + light same-hue fill (legacy red hue
        # for Manual runs, which carry no object prompt).
        result_layer.setRenderer(make_committed_renderer(
            color=output_store.committed_color_for_prompt("")))
        # Smooth pan/zoom on a dense result: render-time simplification (the GPKG
        # already ships a spatial index from the OGR writer).
        _apply_fast_render(result_layer)
        # Style + provenance stored with the .gpkg: the file opens styled and
        # documented in any QGIS, with or without the plugin.
        from datetime import datetime
        try:
            plugin_version = self._read_plugin_version()
        except Exception:  # nosec B110
            plugin_version = ""
        apply_output_conventions(
            result_layer, raster_name,
            created_iso=datetime.now().astimezone().isoformat(timespec="seconds"),
            plugin_version=plugin_version,
        )
        result_layer.triggerRepaint()
        self.iface.mapCanvas().refresh()

        QgsMessageLog.logMessage(
            f"Created segmentation layer: {layer_name} with {len(features_to_add)} polygons",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        try:
            from ...core import telemetry
            from ...core.review_defaults import REFINE_SIMPLIFY_DEFAULT
            refine_shape_changed = self._refine_simplify != REFINE_SIMPLIFY_DEFAULT
            refine_shape_changed = refine_shape_changed or self._refine_smooth or self._refine_expand
            refine_fill_or_ortho_changed = (
                not self._refine_fill_holes or self._refine_ortho)
            refine_used = bool(refine_shape_changed or refine_fill_or_ortho_changed)
            telemetry.track_manual_export_done(
                polygon_count=len(features_to_add),
                refine_used=refine_used,
                destination="new",
            )
            telemetry.track_first_generation_milestone(mode="manual")
        except Exception:
            pass  # nosec B110

        # Value recap on the Manual Start view (session only, mirrors the
        # Automatic Finish recap). The per-feature geodesic area_m2 was just
        # computed above (attribute 1), so the total is a free sum. Entirely
        # best-effort: the export already succeeded, never raise here.
        try:
            if self.dock_widget is not None:
                total_m2 = sum(
                    float(f.attributes()[1] or 0.0) for f in features_to_add)
                self.dock_widget.set_manual_last_run_recap(
                    count=len(features_to_add),
                    area_km2=total_m2 / 1e6,
                )
        except Exception:  # nosec B110 -- never break export on the recap
            pass

        self._reset_session()
        self.dock_widget.reset_session()

    def _append_manual_run_if_requested(self, temp_layer, source_name, features_to_add):
        """Append the built features to an existing layer if the user picked one.

        Returns True when the append is handled (success or the whole flow is
        done); False when the caller should create a new layer instead (no
        destination chosen, target gone, or the append failed). Never raises:
        every failure degrades to the new-layer path with a clear message so a
        paid/hand-made polygon is never lost."""
        try:
            dest_id = self.dock_widget.selected_export_destination()
        except (RuntimeError, AttributeError):
            dest_id = None
        if not dest_id:
            return False

        from ...core import output_store

        target = QgsProject.instance().mapLayer(dest_id)
        if target is None or not output_store.is_appendable_polygon_layer(target):
            # Target deleted or made unwritable mid-flow: keep the work by
            # falling back to a new layer, and say so.
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("That layer is no longer available. Created a new layer instead."),
                level=Qgis.MessageLevel.Warning, duration=8)
            return False

        target_name = target.name()
        res = output_store.append_run_to_layer(
            target, temp_layer, source_name=source_name)
        if res.added <= 0:
            QgsMessageLog.logMessage(
                "Append to '{}' failed ({}); creating a new layer".format(
                    target_name, res.error_message or "unknown"),
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Could not add to that layer. Created a new layer instead."),
                level=Qgis.MessageLevel.Warning, duration=8)
            return False

        self.iface.mapCanvas().refresh()
        self.iface.messageBar().pushMessage(
            "AI Segmentation",
            tr("Added {count} polygon(s) to {name}.").format(
                count=res.added, name=target_name),
            level=Qgis.MessageLevel.Success, duration=6)

        try:
            from ...core import telemetry
            from ...core.review_defaults import REFINE_SIMPLIFY_DEFAULT
            refine_shape_changed = self._refine_simplify != REFINE_SIMPLIFY_DEFAULT
            refine_shape_changed = refine_shape_changed or self._refine_smooth or self._refine_expand
            refine_fill_or_ortho_changed = (
                not self._refine_fill_holes or self._refine_ortho)
            refine_used = bool(refine_shape_changed or refine_fill_or_ortho_changed)
            telemetry.track_manual_export_done(
                polygon_count=res.added,
                refine_used=refine_used,
                destination="append",
            )
            telemetry.track_first_generation_milestone(mode="manual")
        except Exception:
            pass  # nosec B110

        # Value recap, mirroring the new-layer path (best-effort, never raise).
        try:
            if self.dock_widget is not None:
                total_m2 = sum(
                    float(f.attributes()[1] or 0.0) for f in features_to_add)
                self.dock_widget.set_manual_last_run_recap(
                    count=res.added,
                    area_km2=total_m2 / 1e6,
                )
        except Exception:  # nosec B110 -- recap must never break the append
            pass

        self._reset_session()
        self.dock_widget.reset_session()
        return True

    def _on_tool_deactivated(self):
        # Remove keyboard shortcut filter from all targets
        try:
            if self._shortcut_filter is not None:
                self.iface.mainWindow().removeEventFilter(self._shortcut_filter)
                canvas = self.iface.mapCanvas()
                canvas.viewport().removeEventFilter(self._shortcut_filter)
                canvas.removeEventFilter(self._shortcut_filter)
        except (RuntimeError, AttributeError):
            pass

        if self._stopping_segmentation:
            if self.dock_widget:
                self.dock_widget.set_segmentation_active(False)
            return

        # User switched to another tool (pan, etc.) while segmenting.
        # Re-activate segmentation tool silently to prevent accidental exits.
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, self._return_to_segmentation)

    def _return_to_segmentation(self):
        if self._stopping_segmentation:
            return
        if not self.map_tool or not self.dock_widget:
            return
        self._activate_segmentation_tool()

    def _restore_previous_map_tool(self):
        """Restore the map tool that was active before segmentation started."""
        if self._previous_map_tool:
            try:
                self.iface.mapCanvas().setMapTool(self._previous_map_tool)
            except RuntimeError:
                # The previous tool may have been deleted
                pass
        self._previous_map_tool = None

    def _on_stop_segmentation(self):
        """Exit segmentation mode without saving."""
        # Refine handoff: Esc/stop must NEVER offer to discard the whole
        # imported review. An open edit closes back to pending; otherwise the
        # gesture means "leave the refine", which is Back to review (harvests
        # the edits, non-destructive; Finish stays on the review page).
        if self._refine_handoff_active:
            if self._refine_edit_session_active():
                self._close_active_edit_to_pending()
            else:
                self._on_back_to_review_clicked()
            return
        polygon_count = len(self.saved_polygons)
        # Frozen/unfrozen polygons are unsaved work too: without counting
        # them, stopping discards them with no confirmation at all.
        if self.current_mask is not None or self._frozen_sessions or self._unfrozen_display_polygon is not None:
            polygon_count += 1

        if polygon_count > 0:
            reply = QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Stop Segmentation?"),
                "{}\n\n{}".format(
                    tr("This will discard {count} polygon(s).").format(count=polygon_count),
                    tr("Use 'Export to layer' to keep them.")),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            try:
                from ...core import telemetry
                telemetry.track_manual_abandoned(
                    context="stop", polygon_count=polygon_count)
            except Exception:
                pass  # nosec B110

        self._stop_manual_session(keep_saves=False)

    def _stop_manual_session(self, keep_saves: bool) -> None:
        """The 'actually stop' body of the Manual session, without any dialog.

        keep_saves=False discards the session work (the Stop path, after the
        user confirmed the discard dialog). keep_saves=True first commits any
        work to a layer via the normal export (T17: the session raster was
        removed from the project, so the session cannot continue, but
        hand-made work is never lost, invariant I2); the export path performs
        the same session reset itself, and the teardown below is idempotent.
        """
        has_unsaved_work = False
        if keep_saves:
            has_unsaved_work = self.saved_polygons or self.current_mask is not None
            has_unsaved_work = has_unsaved_work or self._frozen_sessions
            has_unsaved_work = has_unsaved_work or self._unfrozen_display_polygon is not None
        if has_unsaved_work:
            # _on_export_layer never raises (it reports its own failures) and
            # resets the session on success; a failed export leaves the work
            # in place and the teardown below still ends the session cleanly.
            self._on_export_layer()
        if self._shortcut_filter is not None:
            try:
                self.iface.mainWindow().removeEventFilter(self._shortcut_filter)
                canvas = self.iface.mapCanvas()
                canvas.viewport().removeEventFilter(self._shortcut_filter)
                canvas.removeEventFilter(self._shortcut_filter)
            except RuntimeError:
                pass
        self._stopping_segmentation = True
        try:
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self._restore_previous_map_tool()
        finally:
            self._stopping_segmentation = False
        self._reset_session()
        if self.dock_widget:
            self.dock_widget.reset_session()

    def _safe_restore_canvas_focus(self):
        """Restore keyboard focus to canvas unless the user is typing in a widget."""
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
        """Store the Min/Max size window (ground m2, 0 = off). Store-only: the
        dock emits this right before refine_settings_changed on the same
        debounce tick, and THAT handler repaints once with everything fresh."""
        self._refine_min_size_m2 = max(0.0, float(min_m2 or 0.0))
        self._refine_max_size_m2 = max(0.0, float(max_m2 or 0.0))

    def _on_refine_settings_changed(self, simplify: int, smooth: int, expand: int,
                                    fill_holes: bool, right_angles: bool = False):
        """Handle refinement control changes.

        min_area is no longer UI-controlled: it is auto-computed per crop in
        _compute_auto_min_area() and never overwritten from the refine panel.
        """
        QgsMessageLog.logMessage(
            f"Refine settings: simplify={simplify}, smooth={smooth}, "
            f"expand={expand}, fill_holes={fill_holes}, "
            f"right_angles={right_angles}, "
            f"min_area={self._refine_min_area} (auto)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        self._refine_simplify = simplify
        self._refine_smooth = smooth
        self._refine_expand = expand
        self._refine_fill_holes = fill_holes
        self._refine_ortho = right_angles

        # Refine handoff: the panel is per-polygon Shape settings, applied in
        # geometry space to the open edit or the selected detections (the
        # entries have no source mask). Consumed there; base Manual continues.
        if self._apply_handoff_refine_settings():
            self._safe_restore_canvas_focus()
            return

        # In both modes: update current mask preview only
        # Saved masks (green) keep their own refine settings from when they were saved
        self._update_mask_visualization()

        self._safe_restore_canvas_focus()
