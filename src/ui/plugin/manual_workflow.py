"""Manual session: start/stop segmentation, map tools, save polygon, export.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import math

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
    _FIELD_TYPE_DOUBLE,
    _FIELD_TYPE_STRING,
    SETTINGS_KEY_LAST_MANUAL_SESSION_TS,
    SETTINGS_KEY_TUTORIAL_SHOWN,
    _add_features_fast,
    looks_like_pixel_image,
    pixel_grid_crs,
)


class ManualWorkflowMixin:
    """Manual session: start/stop segmentation, map tools, save polygon, export."""

    def _on_manual_engine_changed(self, cloud: bool) -> None:
        """The Semi-Auto engine card was picked. Ask for a machine on cloud.

        Start used to be the first thing that asked, and it measured as too
        late: in nearly every Semi-Auto session the user's own first click was
        what started the machine, and it waited out the whole start. Picking
        the card happens while they are still choosing a layer and framing an
        object, which is the runway this needs.
        """
        if not cloud:
            return
        self._maybe_warmup_auto()

    def _warmup_if_manual_cloud(self) -> None:
        """Ask for a machine when Semi-Auto opens already on the cloud engine.

        The card is only picked once. Every session after that arrives with
        the choice already made and nothing to emit, so the mode switch is the
        other place the intent shows. Silent, and debounced with every other
        caller of the ping, so the two triggers together still cost one.
        """
        try:
            from ...core.manual_cloud_route import (
                manual_cloud_route_enabled,
                manual_cloud_route_offered,
            )

            if not (manual_cloud_route_enabled() and manual_cloud_route_offered()):
                return
        except Exception:  # noqa: BLE001 -- an unreadable route asks for nothing
            return
        self._maybe_warmup_auto()

    def _on_start_segmentation(self, layer: QgsRasterLayer):
        # A remote predictor left by an Automatic review is handed back first,
        # so this session starts from a clean slot and decides its own route.
        # Then the mode's own option, which is off unless the user turned it on
        # and keeps the on-device predictor behind it as the fallback.
        if not getattr(self, "_refine_handoff_active", False):
            self._drop_cloud_correct_predictor()
            if self._ensure_manual_cloud_predictor():
                # Ask for a machine NOW, not at the first click. Nothing is
                # running between sessions, and starting one takes about half a
                # minute, so whoever clicks first waits it out. Started here it
                # overlaps the user framing their object, and the click that
                # follows lands on a service that is already up.
                #
                # Shared with the Automatic flow on purpose: one warm instance
                # answers both, and the ping is the same open probe on the same
                # host. It is rate limited and silent, so a session opened and
                # closed repeatedly costs one ping.
                self._maybe_warmup_auto()
        if self.predictor is None:
            # A user who turns the cloud option back off after a session has no
            # on-device predictor in the slot, and the loader stands down while
            # a remote one holds it. Ask for the load here so the sentence below
            # is true and the next Start finds a model.
            self._load_predictor()
            QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Not Ready"),
                tr("The AI is still loading. Try again in a few seconds.")
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
            # RAW source: it is opened as-is by the crop reader. Normalising it
            # here destroys a GDAL URI source on Windows (/vsicurl/, /vsizip/,
            # GPKG:...:layer, NETCDF:"...":var), where normcase lowercases and
            # flips the separators. The one consumer that needs a comparable
            # form (the crop-error dedup key) normalises it itself.
            raster_path = layer.source()
        except RuntimeError:
            QgsMessageLog.logMessage(
                "Layer deleted during segmentation start",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return

        self._reset_session()
        # No warm-up of the previous session survives into this one, and a flag
        # left standing would let a Save here abandon a crop this session asked
        # for by name.
        self._speculative_manual_crop = False

        self._current_layer = layer
        self._current_layer_name = layer_name

        # Rendered rather than read from a file: the online services, plus the
        # local providers that hold no file (PostGIS raster, virtual raster).
        self._is_online_layer = self._needs_canvas_render(layer)

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
            # A PNG/JPG in pixel mode is what the user expects. A GeoTIFF-style
            # raster landing here means its CRS is missing, and silently
            # exporting pixel coordinates would look like a georeferencing bug.
            # Say it once, without blocking: Manual still runs either way.
            if not looks_like_pixel_image(layer):
                self.iface.messageBar().pushInfo(
                    "AI Segmentation",
                    tr("This raster has no coordinate reference system, so "
                       "polygons will use pixel coordinates. Set a CRS in "
                       "Layer Properties for georeferenced output."))

        if self._is_online_layer:
            QgsMessageLog.logMessage(
                f"Layer read through the QGIS renderer ({layer.dataProvider().name()})",
                "AI Segmentation", level=Qgis.MessageLevel.Info
            )

        # No CRS guard here: a missing CRS IS the pixel-mode condition above, so
        # Manual runs the layer in pixel coordinates instead of turning the user
        # away. Only Automatic (which maps masks back to the ground) still
        # requires a CRS.

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
                            tr("This layer has no usable position on the map. "
                               "Open it in QGIS and check its extent."),
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
                tr("This raster is rotated. Run Warp (Reproject) on it to "
                   "straighten it before segmenting."),
                error_code="rotated_raster",
            )
            return

        # Store raster path for on-demand crop extraction
        self._current_raster_path = raster_path

        # Opened here, past every guard above: a Start that turns back leaves no
        # ledger behind for a session that never opened. The review's Correct
        # step comes through here too and opens one on the same terms, because
        # its AI lane is answered off the machine and costs a credit per fix.
        # Whether a ledger opens at all is the predictor's answer, not the
        # mode's, so a session on the machine still gets none.
        self._start_manual_credit_session()

        # Canvas CRS <-> raster CRS, and the watch that rebuilds the pair when
        # the project CRS changes under the session.
        self._rebuild_manual_crs_transforms()
        self._start_canvas_crs_watch()

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

    def _rebuild_manual_crs_transforms(self) -> None:
        """Build the canvas <-> raster transform pair for the session raster.

        Both stay None when the two sides share a CRS, which is the common case
        and costs the click path nothing.
        """
        self._canvas_to_raster_xform = None
        self._raster_to_canvas_xform = None
        layer = self._current_layer
        if layer is None:
            return
        try:
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            raster_crs = layer.crs()
            if not (canvas_crs.isValid() and raster_crs.isValid()):
                return
            if canvas_crs == raster_crs:
                return
            project = QgsProject.instance()
            self._canvas_to_raster_xform = QgsCoordinateTransform(
                canvas_crs, raster_crs, project)
            self._raster_to_canvas_xform = QgsCoordinateTransform(
                raster_crs, canvas_crs, project)
            QgsMessageLog.logMessage(
                f"CRS transform enabled: {canvas_crs.authid()} -> {raster_crs.authid()}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
        except (RuntimeError, AttributeError):
            self._canvas_to_raster_xform = None
            self._raster_to_canvas_xform = None

    def _start_canvas_crs_watch(self) -> None:
        """Follow the canvas CRS for the rest of the session. Idempotent: Start
        runs again on every session over the same dock."""
        if getattr(self, "_canvas_crs_watch_on", False):
            return
        try:
            self.iface.mapCanvas().destinationCrsChanged.connect(
                self._on_canvas_crs_changed)
        except (RuntimeError, AttributeError):
            return
        self._canvas_crs_watch_on = True

    def _stop_canvas_crs_watch(self) -> None:
        """Let go of the canvas CRS. Never raises: teardown paths call it."""
        if not getattr(self, "_canvas_crs_watch_on", False):
            return
        self._canvas_crs_watch_on = False
        try:
            self.iface.mapCanvas().destinationCrsChanged.disconnect(
                self._on_canvas_crs_changed)
        except (TypeError, RuntimeError, AttributeError):
            pass

    def _on_canvas_crs_changed(self) -> None:
        """The project changed CRS mid-session: rebuild the pair every click is
        converted with. Without it the session keeps converting through the CRS
        the canvas had at Start, so clicks read the wrong pixels and saved
        polygons draw in the wrong place.

        Also how the watch ends when no session teardown ran: with no session
        raster there is nothing to convert, so it lets go of the canvas.
        """
        if self._current_layer is None:
            self._stop_canvas_crs_watch()
            return
        self._rebuild_manual_crs_transforms()

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
        """Show the tutorial notification (once ever, persisted in QSettings).

        The address is server-supplied and lands inside an href, so it goes
        through the same guard as the footer button: https only, a real host,
        and none of the characters that would let it break out of the
        attribute. Anything else becomes the built-in address.
        """
        settings = QSettings()
        if settings.value(SETTINGS_KEY_TUTORIAL_SHOWN, False, type=bool):
            return
        settings.setValue(SETTINGS_KEY_TUTORIAL_SHOWN, True)

        from ...core.activation_manager import TUTORIAL_URL_FALLBACK, get_tutorial_url
        from ...core.server_dials import safe_web_url
        tutorial_url = safe_web_url(get_tutorial_url(), TUTORIAL_URL_FALLBACK)
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
                    from ...core import telemetry_session_events
                    telemetry_session_events.track_manual_abandoned(
                        context="change_layer", polygon_count=polygon_count)
                except Exception:
                    pass  # nosec B110

            self._stopping_segmentation = True
            try:
                self.iface.mapCanvas().unsetMapTool(self.map_tool)
                self._restore_previous_map_tool()
            finally:
                # A stuck-True flag makes _on_tool_deactivated refuse to ever
                # re-arm the segmentation tool for the rest of the session.
                self._stopping_segmentation = False
            self._reset_session()
            self.dock_widget.reset_session()

    def _on_save_polygon(self):
        """Save current mask as polygon (including any frozen crop sessions)."""
        # Allow save if we have frozen sessions even without active mask
        has_active = self.current_mask is not None and self.current_transform_info is not None
        if not has_active and not self._frozen_sessions and self._unfrozen_display_polygon is None:
            return
        # An object whose clicks TerraLab's servers answered costs one credit,
        # and this is where it is spent. Asked before anything is committed, so
        # a refusal leaves the shape on screen exactly as the user traced it.
        # The identity is the one this object already had (a re-opened object is
        # paid for); a brand-new one has none yet and is billable by definition.
        billing_id = (self._active_refine_origin_entry or {}).get("det_id")
        if self._manual_save_refused_for_credits(billing_id):
            return
        # There IS a shape to commit. A crop read owning the predictor pipe is
        # no reason to drop it: saving never touches that pipe. What the read
        # carries is a click that has not landed yet, and that click is given up
        # here so a Save is never swallowed by a wait with nothing on screen to
        # explain it. Order matters: a Save with nothing to commit returns
        # above, so it never costs a read that a waiting click still needs.
        if self._encoding_in_progress:
            self._drop_inflight_crop_for_gesture()

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
            # Refine handoff: count the edit. The shape is saved as drawn, and
            # the detections around it are not touched. No-op in base Manual.
            combined = self._note_handoff_shape_saved(combined)
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
                "refine_points_pct": self._refine_points_pct,
                "refine_smooth": self._refine_smooth,
                "refine_clean": self._refine_clean,
                "refine_expand": self._refine_expand,
                "refine_fill_holes": self._refine_fill_holes,
                "refine_fill_holes_max_m2": self._refine_fill_holes_max_m2,
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

            # The object is committed, so this is the moment it is paid for.
            # Sent in the background: the shape is already on the canvas and
            # nothing about it waits on the account.
            self._charge_manual_saved_object(self.saved_polygons[-1]["det_id"])

            # Minimal telemetry: one event per successful segmentation run.
            try:
                import time as _time

                from ...core.telemetry_session_events import track_segmentation_run
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
        # Clear current state for next polygon (including frozen sessions).
        # The ledger goes with it: what the NEXT object costs depends on where
        # its own clicks are answered, not on this one's.
        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is not None:
            ledger.start_next_object()
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
        # Nothing is on screen any more, so the Add lane's Keep goes with it.
        refresh = getattr(self, "_refresh_ai_add_keep_button", None)
        if refresh is not None:
            refresh()

        # Keep crop info so clicks in the same area reuse the encoding.

    def _manual_saved_crs_definition(self) -> str:
        """What CRS the saved polygons are in, as a string QGIS can rebuild.

        The pixel grid carries no authority code, so it travels as its own
        definition. Empty when nothing answers, and the caller's default takes
        over from there.
        """
        if self._is_non_georeferenced_mode:
            return pixel_grid_crs().toWkt()
        for entry in self.saved_polygons:
            info = entry.get("transform_info") or {}
            value = info.get("crs")
            if isinstance(value, str) and value.strip():
                return value.strip()
        # The object still on screen answers when nothing is saved yet, and it
        # answers even after the session raster left the project.
        live = (self.current_transform_info or {}).get("crs")
        if isinstance(live, str) and live.strip():
            return live.strip()
        try:
            if self._is_layer_valid() and self._current_layer.crs().isValid():
                return self._current_layer.crs().authid()
        except RuntimeError:
            pass
        return ""

    def _live_manual_shape_geometry(self):
        """The object still on screen, as one geometry in the raster CRS, or
        None. Same parts the export commits: the frozen crops, the active mask
        through the shared refine tail, or the display polygon when no mask has
        come back yet. Never raises."""
        try:
            parts = [s.polygon for s in self._frozen_sessions
                     if s.polygon is not None and not s.polygon.isEmpty()]
            active = None
            if self.current_mask is not None and self.current_transform_info is not None:
                active = self._refined_active_mask_geometry()
            if active is None and self._unfrozen_display_polygon is not None:
                active = self._unfrozen_display_polygon
            if active is not None and not active.isEmpty():
                parts.append(active)
            if not parts:
                return None
            combined = parts[0] if len(parts) == 1 else QgsGeometry.unaryUnion(parts)
            if combined is None or combined.isEmpty():
                return None
            return combined
        except Exception:  # noqa: BLE001 -- the net keeps the saved polygons either way
            return None

    def _autosave_manual_saved_polygons(self, include_live: bool = False) -> None:
        """Write the saved polygons to disk when the session ends without an
        Export.

        Save keeps them in memory, and a docked panel gets no close event when
        QGIS quits, so the work a user believes they saved would otherwise leave
        with the session. Same net as a finished Automatic run: one GeoPackage
        table, no dialog, the path written to the log.

        ``include_live`` adds the object still on screen. Only for a caller
        about to tear the session down after a failed export: at quit, a
        half-traced selection is not something the user believes they saved.

        No-op when nothing is saved, so it costs a teardown nothing. Never
        raises: it runs where a failure must not stop the rest of the cleanup.
        """
        try:
            if not self.saved_polygons and not include_live:
                return
            # A handoff's saves belong to the Automatic review, which folds them
            # back and writes its own net over the whole set.
            if self._refine_handoff_active:
                return
            import time as _time

            from ...core import run_autosave

            saved = []
            for index, entry in enumerate(self.saved_polygons):
                # One unreadable entry costs its own polygon, never the others.
                try:
                    geom = entry.get("geom_obj")
                    if geom is None:
                        geom = QgsGeometry.fromWkt(entry.get("geometry_wkt") or "")
                    if geom is None or geom.isEmpty():
                        continue
                    det_id = entry.get("det_id")
                    score = entry.get("score")
                    saved.append((
                        int(det_id) if det_id is not None else index,
                        geom,
                        # A hand save carries no model score, and the file must
                        # not claim one for it.
                        float(score) if score is not None else None,
                    ))
                except (TypeError, ValueError, AttributeError, RuntimeError):
                    continue
            if include_live:
                live = self._live_manual_shape_geometry()
                if live is not None:
                    saved.append((len(self.saved_polygons), live, None))
            if not saved:
                return
            try:
                source_layer = self._current_layer if self._is_layer_valid() else None
            except RuntimeError:
                source_layer = None
            info = run_autosave.write_autosave(
                saved, self._manual_saved_crs_definition(), "",
                f"manual-{int(_time.time() * 1000)}", source_layer=source_layer)
            if not info:
                return
            run_autosave.record_pending(info)
            QgsMessageLog.logMessage(
                "Semi-Auto: {n} saved polygon(s) written to {path}".format(
                    n=info.get("count", 0), path=info.get("path", "")),
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
        except Exception:  # noqa: BLE001 -- the net never breaks a teardown
            try:
                QgsMessageLog.logMessage(
                    "Semi-Auto: the session autosave failed",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            except Exception:  # nosec B110
                pass

    def _on_export_layer(self):
        """Export all saved polygons + current unsaved mask to a new layer."""
        # Refine handoff: committing goes through the review's Finish, never a
        # direct export (it would dump the imported detections to a layer the
        # review would then commit AGAIN). Enter/Export = Back to review.
        if self._refine_handoff_active:
            self._on_reshape_done()
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
                tr("The export did not finish. Your polygons are still on the "
                   "map, so you can try again."),
                error_code="export_failed",
            )
        finally:
            self._exporting_in_progress = False

    def _on_export_layer_impl(self):
        """Internal export implementation."""
        import time as _time
        _t_start = _time.perf_counter()
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
            # Shared refine tail: exports exactly what the preview shows, and
            # reads it from the memo the preview filled (manual_shape_cache).
            active_combined = self._refined_active_mask_geometry()
            if active_combined is not None and not active_combined.isEmpty():
                current_geoms.append(active_combined)
        _t_shape = _time.perf_counter()

        # An object still on screen leaves the session through here, so this is
        # its Save: same gate, and the same charge once it is written. Without
        # it, one object per session travels to TerraLab's servers and out to a
        # file for nothing.
        live_billing_id = None
        if current_geoms:
            combined = QgsGeometry.unaryUnion(current_geoms)
            if combined and not combined.isEmpty():
                origin = self._active_refine_origin_entry or {}
                origin_id = origin.get("det_id")
                if self._manual_save_refused_for_credits(origin_id):
                    # An object nobody can pay for is not committed, exactly as
                    # a Save would answer. Nothing else goes out either: the
                    # export ends the session, so writing the saved polygons
                    # here would take the refused shape down with them. They
                    # stay in the session and leave on the next try, once the
                    # user has credits or has put the clicks back on their own
                    # computer.
                    return
                live_billing_id = (int(origin_id) if origin_id is not None
                                   else self._next_handoff_det_id())
                polygons_to_export.append({
                    "geometry_wkt": combined.asWkt(),
                    "score": origin.get("score"),
                    "transform_info": self.current_transform_info.copy() if self.current_transform_info else None,
                })

        self._stopping_segmentation = True
        try:
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self._restore_previous_map_tool()
        finally:
            # A stuck-True flag makes _on_tool_deactivated refuse to ever
            # re-arm the segmentation tool for the rest of the session.
            self._stopping_segmentation = False

        from ...core import output_store

        # Friendly display name ("Segmentation (3 Jul)", deduped); the table
        # name inside the shared GeoPackage is derived by the output store.
        layer_name = output_store.friendly_layer_name("")

        # Determine CRS
        # For non-georeferenced images, use a local pixel grid: a CRS the writer
        # leaves alone, so the coordinates stay the pixel coordinates this mode
        # promises and the polygons draw on top of the image.
        if self._is_non_georeferenced_mode:
            crs = pixel_grid_crs()
            QgsMessageLog.logMessage(
                "Non-georeferenced mode: writing pixel coordinates on a local grid",
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
            crs = None
            if isinstance(crs_str, str) and crs_str.strip():
                crs = QgsCoordinateReferenceSystem(crs_str)
            if crs is None or not crs.isValid():
                # A custom or WKT-only raster CRS has no authid, so every lookup
                # above comes back empty. Take the layer's own CRS object rather
                # than stamping EPSG:4326 on projected coordinates, which lands
                # the polygons thousands of km away with no visible error.
                try:
                    if self._is_layer_valid() and self._current_layer.crs().isValid():
                        crs = self._current_layer.crs()
                except RuntimeError:
                    crs = None
            if crs is None or not crs.isValid():
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
            round_measure,
            to_multipolygon,
        )

        # Per-feature schema: an editable label, the model's confidence when the
        # polygon came from one, and both geodesic measures. Same measure columns
        # as an Automatic export, so the two modes stack in one table without a
        # hole. Run-level provenance (source raster, date) lives in the layer
        # metadata instead of being repeated on every row.
        pr = temp_layer.dataProvider()
        pr.addAttributes([
            QgsField("label", _FIELD_TYPE_STRING),
            QgsField("score", _FIELD_TYPE_DOUBLE),
            QgsField("area_m2", _FIELD_TYPE_DOUBLE),
            QgsField("perimeter_m", _FIELD_TYPE_DOUBLE),
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
        # None on the pixel grid: that CRS sits on no ellipsoid, so a geodesic
        # measure of it answers NaN and the two columns are measured flat, in the
        # pixels this mode works in.
        measurer = None if self._is_non_georeferenced_mode else make_area_measurer(crs)
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
                # A hand-drawn save carries no model score: NULL, not 0.0.
                score = polygon_data.get("score")
                if measurer is None:
                    area, perimeter = geom.area(), geom.length()
                else:
                    area = measurer.measureArea(geom)
                    perimeter = measurer.measurePerimeter(geom)
                feature.setAttributes([
                    "",
                    round(float(score), 3) if score is not None else None,
                    round_measure(area),
                    round_measure(perimeter),
                ])
                features_to_add.append(feature)
            else:
                QgsMessageLog.logMessage(
                    f"Polygon {i + 1} could not be read back from its saved "
                    "outline and was left out",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning
                )

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

        if not _add_features_fast(pr, features_to_add):
            # A refused batch leaves an empty layer, and every step after this
            # would report a file the polygons never reached.
            QgsMessageLog.logMessage(
                f"Export aborted: the layer refused {len(features_to_add)} polygon(s)",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                tr("The polygons could not be put into the new layer, so "
                   "nothing was saved. They are still on the map, so you can "
                   "try again."),
                error_code="export_failed",
            )
            return
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
        _t_write = _time.perf_counter()

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

        # The object that was still on screen is written to the file now, so
        # this is the moment it is paid for. Sent in the background, like a
        # Save. After the write, never before: a failed write leaves the object
        # in the session, and the next try mints it a new id, so charging early
        # bills twice for a file that was never written once.
        if live_billing_id is not None:
            self._charge_manual_saved_object(live_billing_id)

        result_layer = result.layer
        gpkg_path = result.gpkg_path
        layer_name = result_layer.name()

        # Style and provenance BEFORE the layer reaches the project. Added
        # first, it paints once with the provider default and again with the
        # committed look; dressed first, it paints once, correctly.
        #
        # Committed look: solid outline + light same-hue fill (legacy red hue
        # for Manual runs, which carry no object prompt).
        result_layer.setRenderer(make_committed_renderer(
            color=output_store.committed_color_for_prompt("")))
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

        # Add under the source raster's sub-group in the "AI Segmentation"
        # group. This also turns on render-time simplification and builds the
        # provider spatial index, so neither is repeated below.
        output_store.add_committed_layer(result_layer, source_name=source_name)

        if result.used_fallback:
            msg = tr(
                "Could not write to {name}. Saved to a separate file instead."
            ).format(name=output_store.GPKG_FILENAME)
            self.iface.messageBar().pushMessage(
                "AI Segmentation", msg,
                level=Qgis.MessageLevel.Warning, duration=8)

        # One repaint. The layer tree bridge already schedules one when the
        # layer joins the project, and a triggerRepaint on top of it made the
        # canvas draw the whole map twice for the same result.
        self.iface.mapCanvas().refresh()

        # One line, with where the wait went. Four separate lines used to
        # report the extent, the CRS and the path and said nothing about what
        # the user was waiting for; the next report of a slow export should be
        # answerable from the log alone.
        _t_end = _time.perf_counter()
        _ms_shape = int((_t_shape - _t_start) * 1000)
        _ms_write = int((_t_write - _t_shape) * 1000)
        _ms_layer = int((_t_end - _t_write) * 1000)
        _ms_total = int((_t_end - _t_start) * 1000)
        _extent = result_layer.extent()
        QgsMessageLog.logMessage(
            f"Export: {len(features_to_add)} polygon(s) to {layer_name} "
            f"[{result_layer.crs().authid()}] in {_ms_total} ms "
            f"(shape {_ms_shape}, write {_ms_write}, layer {_ms_layer}); "
            f"extent {_extent.xMinimum():.1f},{_extent.yMinimum():.1f} to "
            f"{_extent.xMaximum():.1f},{_extent.yMaximum():.1f}; file {gpkg_path}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        try:
            from ...core import telemetry_session_events
            from ...core.review_defaults import (
                REFINE_POINTS_PCT_DEFAULT,
                REFINE_SIMPLIFY_DEFAULT,
            )
            # Simplify is a float, so compare with a tolerance well under the
            # spinbox step rather than on equality.
            refine_shape_changed = abs(
                float(self._refine_simplify) - REFINE_SIMPLIFY_DEFAULT) > 1e-6
            refine_shape_changed = refine_shape_changed or (
                int(self._refine_points_pct) != REFINE_POINTS_PCT_DEFAULT)
            refine_shape_changed = refine_shape_changed or self._refine_smooth or self._refine_expand
            refine_fill_or_ortho_changed = (
                not self._refine_fill_holes or self._refine_ortho)
            refine_used = bool(refine_shape_changed or refine_fill_or_ortho_changed)
            telemetry_session_events.track_manual_export_done(
                polygon_count=len(features_to_add),
                refine_used=refine_used,
                destination="new",
            )
            telemetry_session_events.track_first_generation_milestone(mode="manual")
        except Exception:
            pass  # nosec B110

        # No value recap here any more. A card counting what the last session
        # produced sat on the Start view for the rest of the session, and the
        # saved layer in the legend says the same thing without taking a
        # quarter of the panel. Removed 2026-08-11.

        self._reset_session()
        self.dock_widget.reset_session()

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
                self._on_reshape_done()
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
                from ...core import telemetry_session_events
                telemetry_session_events.track_manual_abandoned(
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
            # resets the session on success.
            self._on_export_layer()
            # An export that did not go through leaves the work in the session,
            # and the teardown below drops it. The user was told it was still on
            # the map, so write it to disk before it is not.
            self._autosave_manual_saved_polygons(include_live=True)
        self._teardown_manual_session()

    def _teardown_manual_session(self) -> None:
        """End the Manual session: drop the shortcut filter, restore the map
        tool, reset plugin + dock state. No confirm dialog and no export here;
        callers harvest or export unsaved work first. Idempotent; shared by
        the stop button, the refine handoff and the zone teardown paths."""
        # The billing ledger belongs to the session, so it ends with it. A
        # charge already in flight keeps its own copy of what it needs.
        self._end_manual_credit_session()
        # The session raster goes with it, so the canvas CRS stops being ours to
        # follow.
        self._stop_canvas_crs_watch()
        # The AI-assisted Add flag rides the handoff session; a leaked True would
        # flip the resting-click gate in the NEXT session, so drop it here (the
        # single choke point every session end passes through) and disarm the lane.
        if getattr(self, "_refine_add_mode_active", False):
            exit_add = getattr(self, "_exit_ai_add_mode", None)
            if exit_add is not None:
                exit_add()
            else:
                self._refine_add_mode_active = False
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
            try:
                self.dock_widget.reset_session()
            except (RuntimeError, AttributeError):
                pass

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

    def _on_fill_holes_size_changed(self, max_m2: float) -> None:
        """Store the fill-holes size threshold (ground m2, 0 = fill every
        hole). Store-only for the same reason as the size window above: the
        refine handler that follows on the same tick does the one repaint."""
        self._refine_fill_holes_max_m2 = max(0.0, float(max_m2 or 0.0))

    def _on_clean_edges_changed(self, clean_px: float) -> None:
        """Store the Clean-edges opening distance (px, 0 = off). Store-only:
        the dock emits it right before refine_settings_changed on the same
        debounce tick, and THAT handler does the one repaint."""
        self._refine_clean = max(0.0, float(clean_px or 0.0))

    def _on_outline_budget_changed(self, simplify_px: float, points_pct: int) -> None:
        """Store Simplify (px, 0 = off) and Points (share of an outline's own
        points, 100 = off). Store-only: the dock emits it right before
        refine_settings_changed on the same debounce tick, and THAT handler does
        the one repaint."""
        self._refine_simplify = max(0.0, float(simplify_px or 0.0))
        self._refine_points_pct = max(1, min(100, int(points_pct or 100)))

    def _on_refine_settings_changed(self, simplify: int, smooth: int, expand: int,
                                    fill_holes: bool, right_angles: bool = False):
        """Handle refinement control changes.

        min_area is no longer UI-controlled: it is auto-computed per crop in
        _compute_auto_min_area() and never overwritten from the refine panel.
        """
        QgsMessageLog.logMessage(
            f"Refine settings: simplify={self._refine_simplify}, "
            f"points_pct={self._refine_points_pct}, smooth={smooth}, "
            f"expand={expand}, fill_holes={fill_holes}, "
            f"right_angles={right_angles}, "
            f"min_area={self._refine_min_area} (auto)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        # `simplify` is legacy: the float arrives on outline_budget_changed.
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
