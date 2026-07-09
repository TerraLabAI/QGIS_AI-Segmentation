"""Zone drawing, tile grid overlay, canvas badges, project-cleared handling.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations


from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

from ...core.i18n import tr
from ...core.qt_compat import PolygonGeometry
from ..canvas_palette import CHROME_BLUE, GRID_LINE, ZONE_FILL, ZONE_STROKE
from ..shortcut_filter import ShortcutFilter
from .shared import FREE_TRIAL_MAX_ZONE_KM2


class AutoZoneMixin:
    """Zone drawing, tile grid overlay, canvas badges, project-cleared handling."""

    # ---- Pro Auto mode: zone selection, tile grid overlay -------------------

    def _get_active_raster_layer(self) -> QgsRasterLayer | None:
        """Return the raster layer selected in the dock combo of the CURRENT
        mode. Interactive and Automatic each have their own combo; reading
        the wrong one silently runs detection on a stale layer."""
        if not self.dock_widget:
            return None
        try:
            from ..ai_segmentation_dockwidget import Mode
            if self.dock_widget._mode == Mode.AUTOMATIC:
                layer = self.dock_widget.auto_layer_combo.currentLayer()
            else:
                layer = self.dock_widget.layer_combo.currentLayer()
        except (RuntimeError, AttributeError):
            return None
        if layer is None:
            return None
        # isinstance is the version-stable raster check: the layer-type enum
        # spelling moved across QGIS releases, and QgsRasterLayer.LayerType is
        # the raster RENDER type enum, which has no RasterLayer member.
        if not isinstance(layer, QgsRasterLayer):
            return None
        return layer

    def _setup_auto_mode(self) -> None:
        """Create the zone selection tool, tile manager, and tile grid overlay.

        Idempotent: returns immediately if already set up.
        """
        if self._zone_selection_tool is not None:
            return

        from ...core.tile_manager import MAX_TILES, OVERLAP_FRACTION, TILE_SIZE, TileManager
        from ..polygon_zone_maptool import PolygonZoneMapTool

        canvas = self.iface.mapCanvas()
        self._tile_manager = TileManager(
            tile_size=TILE_SIZE,
            overlap_fraction=OVERLAP_FRACTION,
            max_tiles=MAX_TILES,
        )
        # Point-by-point polygon tool so the user outlines the exact area to
        # scan; tiles outside the polygon are never rendered/billed.
        self._zone_selection_tool = PolygonZoneMapTool(canvas)
        self._zone_selection_tool.zone_selected.connect(self._on_zone_polygon_drawn)
        self._zone_selection_tool.zone_cleared.connect(self._on_zone_cleared)
        self._zone_selection_tool.vertices_changed.connect(self._on_zone_vertices_changed)
        # Ctrl+Z / Backspace on the empty draw canvas leaves the draw step.
        self._zone_selection_tool.back_requested.connect(self._on_auto_exit_clicked)

        # Hide overlay when project is replaced or cleared
        QgsProject.instance().cleared.connect(self._on_project_cleared_auto)
        QgsProject.instance().readProject.connect(self._on_project_cleared_auto)

    def _teardown_auto_mode(self) -> None:
        """Clean up zone selection tool, rubber bands, and overlay layers.

        Called from unload() and whenever the Pro auto mode is deactivated.
        """
        if self._zone_selection_tool is not None:
            # Restore the tool the user had before our zone draw (pan by
            # default) instead of leaving a bare cursor once the plugin is
            # gone; only acts when our zone tool is still the active tool.
            self._restore_maptool_after_zone()
            try:
                self.iface.mapCanvas().unsetMapTool(self._zone_selection_tool)
            except (RuntimeError, AttributeError):
                pass
            try:
                self._zone_selection_tool.zone_selected.disconnect(self._on_zone_polygon_drawn)
                self._zone_selection_tool.zone_cleared.disconnect(self._on_zone_cleared)
                self._zone_selection_tool.vertices_changed.disconnect(
                    self._on_zone_vertices_changed)
                self._zone_selection_tool.back_requested.disconnect(self._on_auto_exit_clicked)
            except (TypeError, RuntimeError):
                pass
            self._zone_selection_tool = None

        self._clear_auto_canvas()

        try:
            QgsProject.instance().cleared.disconnect(self._on_project_cleared_auto)
            QgsProject.instance().readProject.disconnect(self._on_project_cleared_auto)
        except TypeError:
            pass  # nosec B110 -- not connected is fine

        self._auto_zone = None
        self._auto_zone_polygon = None
        self._tile_manager = None
        self._auto_run_ctx = None

        # Stop any running detection worker and remove the live selection layer.
        self._stop_auto_detection()
        # Save a still-pending review before tearing down (unload / mode reset):
        # a billed detection must not be lost just because the user quit QGIS or
        # switched away without clicking Finish.
        self._autosave_pending_auto_review()
        self._auto_review = None  # discard review without UI update (widget may be gone)
        self._remove_auto_selection_layer()

    def _activate_zone_drawing(self) -> None:
        """Activate the zone selection map tool on the canvas."""
        # The zone belongs to a run in flight or a pending (billed) review:
        # arming a redraw now would invalidate work mid-use, so no code path
        # (button, badge, stray re-entry) may start a draw here.
        if self._auto_worker is not None or self._auto_review is not None:
            return
        if self._zone_selection_tool is None:
            self._setup_auto_mode()

        layer = self._get_active_raster_layer()
        if layer is not None:
            try:
                raster_w = layer.width()
                raster_h = layer.height()
                ext = layer.extent()
                if raster_w > 0 and raster_h > 0 and ext.width() > 0:
                    self._zone_selection_tool.set_snap_context(
                        tile_manager=self._tile_manager,
                        pixel_size_x=ext.width() / raster_w,
                        pixel_size_y=ext.height() / raster_h,
                    )
            except (RuntimeError, AttributeError):
                pass

        self._clear_auto_canvas()
        try:
            canvas = self.iface.mapCanvas()
            # Remember the tool active before arming (the pan tool by default) so
            # exit/finish/zone-drawn can restore it instead of leaving a bare
            # cursor. Do not capture the zone tool itself on a re-arm.
            current = canvas.mapTool()
            if current is not self._zone_selection_tool:
                self._maptool_before_zone = current
            canvas.setMapTool(self._zone_selection_tool)
            # Focus the canvas so the polygon tool receives key events (Ctrl+Z to
            # undo a point / go back, Backspace, Enter) without a first click.
            canvas.setFocus()
        except (RuntimeError, AttributeError):
            pass
        # Space-pan needs the shortcut filter; installing the same filter
        # object twice is a Qt no-op, so this is safe alongside the
        # interactive-mode install.
        if self._shortcut_filter is None:
            self._shortcut_filter = ShortcutFilter(self)
        self.iface.mainWindow().installEventFilter(self._shortcut_filter)
        canvas = self.iface.mapCanvas()
        canvas.viewport().installEventFilter(self._shortcut_filter)
        canvas.installEventFilter(self._shortcut_filter)
        if self.dock_widget:
            self.dock_widget.set_auto_zone_state("drawing")
            self.dock_widget.set_zone_draw_progress(0)

    # ---- Re-run from the Recent detection history (library) -----------------

    def _on_history_rerun_requested(self, entry: dict) -> None:
        """Recent card "Run again here": rebuild the stored run's exact zone
        and prompt, land on step 2 ready to Detect (never auto-launch).

        Deferred a tick so the library modal has fully closed before the flow
        switch runs (the signal fires while the dialog loop is still
        unwinding)."""
        from qgis.PyQt.QtCore import QTimer
        payload = dict(entry) if isinstance(entry, dict) else {}
        QTimer.singleShot(0, lambda: self._history_rerun_here(payload))

    def _on_history_reuse_prompt_requested(self, prompt: str) -> None:
        """Recent card "Same object, new zone": start the Automatic flow on the
        draw-zone step with the prompt prefilled for step 2."""
        from qgis.PyQt.QtCore import QTimer
        text = str(prompt or "")
        QTimer.singleShot(0, lambda: self._history_reuse_prompt(text))

    def _history_rerun_here(self, entry: dict) -> None:
        """Same zone + same object. The zone is rebuilt as a rectangle from the
        stored extent, transformed from its CRS to the canvas CRS, then fed
        through the normal draw path so the wrong-layer guard, the free-trial
        cap, the badge, grid and credit estimate all apply for free."""
        if self._auto_worker is not None or self._auto_review is not None:
            self._history_rerun_busy_notice()
            return
        prompt = (entry.get("prompt") or "").strip()
        geom = self._zone_geom_from_extent(
            entry.get("extent"), str(entry.get("crs") or ""))
        self._enter_auto_flow_for_history(prompt)
        if geom is None:
            # No usable stored extent: degrade to the new-zone path (the flow is
            # already armed on the draw step, so the user just draws).
            self._track_history_rerun("new_zone")
            return
        # Feed the rebuilt zone through the normal commit path (wrong-layer +
        # free-cap guards, badge, grid, credit estimate, and the jump to step 2).
        self._on_zone_polygon_drawn(geom)
        self._track_history_rerun("same_zone")

    def _history_reuse_prompt(self, prompt: str) -> None:
        """Same object, fresh zone: land on the draw-zone step with the prompt
        prefilled for step 2."""
        if self._auto_worker is not None or self._auto_review is not None:
            self._history_rerun_busy_notice()
            return
        self._enter_auto_flow_for_history((prompt or "").strip())
        self._track_history_rerun("new_zone")

    def _enter_auto_flow_for_history(self, prompt: str) -> None:
        """Bring the Automatic flow to a clean started state (step 1, draw tool
        armed) with ``prompt`` prefilled, reusing the dock's Start path so no
        grid/credit logic is duplicated. Both re-run actions share this base."""
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
        # Clean slate: drop any zone/canvas left from a mid-flow library open so
        # both actions start from the same predictable base (idempotent; a fresh
        # mode switch has already reset the flow).
        self._reset_auto_flow_to_start()
        try:
            dock._on_auto_start_clicked()  # locks the selected raster, opens step 1
        except (RuntimeError, AttributeError):
            pass
        # Prefill after Start: the start/reset path clears the prompt box, so the
        # object must be set last for it to stick.
        if prompt:
            try:
                dock.set_prompt_text(prompt)
            except (RuntimeError, AttributeError):
                pass

    def _zone_geom_from_extent(self, extent, authid: str) -> QgsGeometry | None:
        """Rectangle geometry in the canvas CRS from a stored run's extent +
        CRS authid, or None when the entry has no usable zone (old format,
        empty rectangle, or an unresolvable CRS/transform)."""
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
                geom.transform(xform)
            except Exception:  # nosec B110 -- antimeridian, invalid CRS, etc.
                return None
            if geom.isEmpty():
                return None
        return geom

    def _history_rerun_busy_notice(self) -> None:
        """Discreet no-op message when a re-run is asked while a detection or
        review is still in flight (mirrors the exemplar-draw guards)."""
        try:
            self.iface.messageBar().pushInfo(
                "AI Segmentation",
                tr("Finish or cancel the current detection before "
                   "re-running a past one."),
            )
        except (RuntimeError, AttributeError):
            pass

    def _track_history_rerun(self, kind: str) -> None:
        """Additive telemetry: kind = 'same_zone' | 'new_zone'. No coordinates."""
        try:
            from ...core import telemetry
            from ...core import telemetry_events as tev
            telemetry.track(tev.HISTORY_RERUN, {"kind": kind})
        except Exception:
            pass  # nosec B110

    def _on_zone_polygon_drawn(self, geom: QgsGeometry) -> None:
        """Store the drawn polygon zone (+ its bounding box) and show the
        persistent outline and tile grid.

        The bounding box becomes ``_auto_zone`` so the whole bbox-based grid and
        render pipeline is unchanged; the polygon is kept separately to cull
        tiles that fall outside the drawn shape (no render, no credits, no
        false positives on empty ground).
        """
        # Wrong-layer guard: a zone with no raster data under it would bill
        # credits for blank tiles, so the commit is checked BEFORE any state
        # is stored (interactive path only; the MCP path sets _auto_zone
        # directly and never comes through here).
        overlap = self._zone_layer_overlap_verdict(geom)
        if overlap == "outside":
            self._reject_zone_outside_layer()
            return
        # Free-trial zone cap: a free-tier zone above FREE_TRIAL_MAX_ZONE_KM2
        # is refused BEFORE any state is stored (interactive path only; the
        # MCP/headless paths run their own guard). Subscribers never hit this.
        cap_area = self._free_zone_cap_exceeded_km2(geom)
        if cap_area is not None:
            self._reject_zone_over_free_cap(cap_area)
            return
        self._auto_zone_polygon = QgsGeometry(geom)  # canvas CRS
        rect = QgsRectangle(geom.boundingBox())
        self._auto_zone = rect  # stored in canvas CRS
        # Zone set: the user is now picking prompt/settings, so this is a second
        # natural moment to ensure the backend is warm before Detect (debounced).
        self._maybe_warmup_auto()
        self._show_zone_polygon_band(geom)
        # Seed a good detail level for this zone BEFORE estimating, so the user
        # lands on a resolution that detects well instead of a coarse 1x1.
        self._apply_default_detail(rect)
        self._update_credit_estimate()
        # Zone drawn: disarm the polygon tool and restore the pan tool so the
        # user can move around the map (the QGIS default) instead of being left
        # with a bare cursor while they pick a prompt.
        self._restore_maptool_after_zone()
        if self.dock_widget:
            self.dock_widget.set_auto_zone_state("zone_set")
        try:
            from ...core import telemetry
            try:
                vtx = int(geom.constGet().vertexCount())
            except Exception:
                vtx = 0
            telemetry.track_zone_drawn(
                vertices=vtx,
                area_km2=self._zone_geodesic_area_km2(geom),
            )
        except Exception:
            pass  # nosec B110
        if overlap == "partial":
            layer = self._get_active_raster_layer()
            if layer is not None:
                self.iface.messageBar().pushInfo(
                    "AI Segmentation",
                    tr(
                        "Part of your zone is outside \"{layer}\" - only the "
                        "overlapping area will return objects."
                    ).format(layer=layer.name()),
                )

    # More than half the zone outside the raster: worth telling the user.
    _ZONE_OUTSIDE_INFO_FRACTION = 0.5

    def _zone_layer_overlap_verdict(self, geom: QgsGeometry) -> str:
        """Classify a freshly drawn zone against the selected raster's data
        extent, so a wrong-layer pick is caught at draw time instead of after
        the run bills blank tiles.

        Returns "outside" (no intersection at all: block the commit),
        "partial" (more than half of the zone lies outside the extent:
        proceed with an info message) or "ok". Online XYZ/WMS providers
        report a whole-world extent so the guard naturally never fires for
        them (intended: the footgun is local rasters). Any transform or
        geometry failure skips the check ("ok") rather than blocking the
        user on a false negative.
        """
        layer = self._get_active_raster_layer()
        if layer is None:
            return "ok"
        try:
            if self._is_online_provider(layer):
                return "ok"
            extent = layer.extent()
            if extent.isEmpty() or extent.width() <= 0 or extent.height() <= 0:
                return "ok"
            zone = QgsGeometry(geom)  # canvas CRS copy: transform() mutates
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            layer_crs = layer.crs()
            if canvas_crs != layer_crs:
                if not canvas_crs.isValid() or not layer_crs.isValid():
                    return "ok"
                xform = QgsCoordinateTransform(
                    canvas_crs, layer_crs, QgsProject.instance())
                zone.transform(xform)
                if zone.isEmpty():
                    return "ok"
            extent_geom = QgsGeometry.fromRect(extent)
            if not zone.intersects(extent_geom):
                return "outside"
            zone_area = zone.area()
            if zone_area <= 0:
                return "ok"
            inside_fraction = zone.intersection(extent_geom).area() / zone_area
            if inside_fraction < self._ZONE_OUTSIDE_INFO_FRACTION:
                return "partial"
        except Exception:  # nosec B110 -- guard rail only, never block on failure
            return "ok"
        return "ok"

    def _reject_zone_outside_layer(self) -> None:
        """Block a zone that misses the raster entirely (wrong layer in the
        combo, e.g. drawn over a basemap while a local raster is selected).

        Clears the sketch the same way the delete badge path does and keeps
        the user on the draw step, armed, so they can redraw or Exit to pick
        another layer."""
        layer = self._get_active_raster_layer()
        name = layer.name() if layer is not None else ""
        # Nothing was committed yet and the draw tool is still active, so the
        # badge-style zone_cleared reset lands the dock back on "drawing".
        self._on_zone_cleared()
        self.iface.messageBar().pushWarning(
            "AI Segmentation",
            tr(
                "Your zone is outside \"{layer}\". Pick the right layer "
                "or draw inside it."
            ).format(layer=name),
        )

    def _zone_geodesic_area_km2(self, geom: QgsGeometry, crs=None) -> float:
        """Geodesic area (km2, WGS84 ellipsoid) of a zone geometry expressed
        in ``crs`` (default: the canvas CRS). Returns 0.0 on any failure so
        callers fail open (a zone that cannot be measured is never blocked)."""
        try:
            from qgis.core import QgsDistanceArea
            da = QgsDistanceArea()
            if crs is None:
                crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            if crs is not None and crs.isValid():
                da.setSourceCrs(crs, QgsProject.instance().transformContext())
            da.setEllipsoid(QgsProject.instance().ellipsoid() or "WGS84")
            return max(0.0, da.measureArea(geom) / 1_000_000.0)
        except Exception:  # nosec B110 -- measurement guard, never blocks
            return 0.0

    def _free_zone_cap_exceeded_km2(self, geom: QgsGeometry, crs=None) -> float | None:
        """Free-trial zone cap check: return the zone's geodesic area (km2)
        when a free-tier user's zone exceeds FREE_TRIAL_MAX_ZONE_KM2, else
        None (zone allowed). Subscribers are never capped. Usage never
        fetched (startup fetch still in flight, or it failed) means the tier
        is UNKNOWN, not free: capping then would reject a paying subscriber's
        valid zone with a free-trial upsell, so the check fails open and the
        server enforces the real quota. ``crs`` is the CRS the geometry is
        expressed in (default: canvas CRS)."""
        try:
            usage_known = bool(self._last_usage)
            _credits, is_free = self._auto_credit_snapshot()
        except (RuntimeError, AttributeError):
            usage_known, is_free = False, True
        if not usage_known or not is_free:
            return None
        area = self._zone_geodesic_area_km2(geom, crs)
        if area > FREE_TRIAL_MAX_ZONE_KM2:
            return area
        return None

    def _reject_zone_over_free_cap(self, area_km2: float) -> None:
        """Refuse a free-tier zone above FREE_TRIAL_MAX_ZONE_KM2 (contextual
        upsell touchpoint 1): the sketch is cleared the badge way, the user
        stays on the draw step, armed, and the step-1 hero shows the
        subscribe message until a valid zone lands or the flow is exited."""
        self._on_zone_cleared()
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_zone_rejected(area_km2)
            except (RuntimeError, AttributeError):
                pass
        try:
            from ...core import telemetry
            telemetry.track_auto_zone_too_large(area_km2=area_km2)
        except Exception:
            pass  # nosec B110

    def _on_zone_vertices_changed(self, count: int) -> None:
        """Live point count from the polygon tool: refresh the draw-zone hint so
        the user always sees what to do next and how to finish."""
        if self.dock_widget:
            try:
                self.dock_widget.set_zone_draw_progress(count)
            except (RuntimeError, AttributeError):
                pass

    def _on_zone_cleared(self) -> None:
        """Clear the zone state and hide all auto mode visuals.

        When the drawing tool is still active (Escape pressed mid-draw), the
        user stays in drawing mode: the next drag starts a fresh zone.
        """
        self._auto_zone = None
        self._auto_zone_polygon = None
        # Exemplars are positioned inside the old zone; a new zone invalidates
        # them, so every zone-bound canvas artifact goes with the zone.
        self._clear_auto_canvas()
        self._update_credit_estimate()
        still_drawing = False
        try:
            still_drawing = (
                self._zone_selection_tool is not None
                and self.iface.mapCanvas().mapTool() == self._zone_selection_tool  # noqa: W503
            )
        except (RuntimeError, AttributeError):
            pass
        if self.dock_widget:
            self.dock_widget.set_auto_zone_state(
                "drawing" if still_drawing else "idle")

    def _update_credit_estimate(self) -> None:
        """Compute credit estimate for current zone + layer and update the grid preview."""
        if self._tile_manager is None:
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            self._clear_zone_tile_grid()
            if self.dock_widget:
                self.dock_widget.set_auto_detail_visible(False)
                self._hide_auto_cost_label()
            return

        # Tiling is a user choice for every layer type now; the slider only
        # needs a zone to be meaningful.
        if self.dock_widget:
            self.dock_widget.set_auto_detail_visible(self._auto_zone is not None)
            # Cap the slider at the useful level so the cursor never moves in
            # the void. Done before the grid compute so the estimate below
            # reflects the (possibly) clamped detail value.
            if self._auto_zone is not None:
                zone_in_layer = self._reproject_zone_to_layer_crs(self._auto_zone, layer)
                self.dock_widget.set_auto_detail_max(
                    self._max_useful_detail(layer, zone_in_layer))

        grid = self._compute_auto_grid(layer)
        if grid is None:
            # Online layer with no zone yet (or unresolvable layer): hide the preview quietly.
            self._clear_zone_tile_grid()
            self._hide_auto_cost_label()
            return

        pixel_w = grid["pixel_w"]
        pixel_h = grid["pixel_h"]

        # Compute the grid once: the credit count AND the actual cols x rows
        # both come from it. The grid is NOT always square (a non-square zone
        # gives n x m), so the detail label must show the real shape, not the
        # slider value squared, or it contradicts the credit count.
        tiles_list = self._tile_manager.compute_grid(pixel_w, pixel_h)
        if tiles_list is not None:
            # Count only tiles inside the drawn polygon, so "N credits" matches
            # what the run actually bills (no-op for the rectangle/MCP path).
            tiles_list = self._tiles_in_polygon(
                tiles_list, grid["bbox"], pixel_w, pixel_h, layer)
        credit_count = len(tiles_list) if tiles_list is not None else -1
        # credit_count == -1 means > MAX_TILES
        self._auto_est_tiles = credit_count  # cached for detail_changed telemetry

        QgsMessageLog.logMessage(
            "Credit estimate: {}x{}px -> {} tile(s)".format(pixel_w, pixel_h, credit_count),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        # Only reflect the estimate in the dock while a zone is set; the
        # zone-less path estimates the full raster for the overlay only.
        if self.dock_widget and self._auto_zone is not None:
            self.dock_widget.set_auto_credit_estimate(credit_count)
            # GSD guard: warn when the chosen detail leaves the imagery coarser
            # than the cloud model's quality floor (~0.5 m/px ground resolution).
            try:
                zone_in_layer = self._reproject_zone_to_layer_crs(self._auto_zone, layer)
                sized = self._grid_for_detail(
                    layer, zone_in_layer, self._get_auto_detail_level())
                if sized is not None:
                    ground_mupp = self._mupp_to_meters(layer, zone_in_layer, sized[2])
                    self.dock_widget.set_auto_detail_gsd_warning(ground_mupp >= 0.5)
            except (RuntimeError, AttributeError):
                pass

        if credit_count == -1:
            # Zone exceeds tile cap: hide the preview, show warning
            self._clear_zone_tile_grid()
            QgsMessageLog.logMessage(
                tr("Zone too large. Reduce the area to {max} tiles or fewer.").format(
                    max=self._tile_manager.max_tiles),
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return

        if credit_count > 0 and self._auto_zone is not None:
            self._show_zone_tile_grid(layer, grid)

    def _hide_auto_cost_label(self) -> None:
        """Blank the 'N tile(s) = N credit(s)' label when there is nothing to
        estimate (layer removed, zone dropped); it used to go stale."""
        if not self.dock_widget:
            return
        try:
            self.dock_widget.auto_credit_cost_label.setVisible(False)
        except (RuntimeError, AttributeError):
            pass

    def _show_zone_tile_grid(self, layer, grid: dict) -> None:
        """Draw the tile grid inside the drawn zone as a canvas rubber band.

        Follows the detail slider live, so the user sees how the zone will
        be split (n tiles = n credits). Display simplification: the real
        tiles overlap by OVERLAP_FRACTION, which reads on screen as a
        blurry double grid. The preview shows clean equal cells instead:
        same row and column counts, honest cost, readable layout. A rubber
        band, not a memory layer: nothing lands in the project's layer tree.
        """
        self._clear_zone_tile_grid()
        if self._tile_manager is None or self._auto_zone is None:
            return
        tiles = self._tile_manager.compute_grid(grid["pixel_w"], grid["pixel_h"])
        if not tiles or len(tiles) <= 1:
            return  # a single tile needs no inner grid
        minx, miny, maxx, maxy = grid["bbox"]
        try:
            cols = len({tx for tx, _ty, _tw, _th in tiles})
            rows = len({ty for _tx, ty, _tw, _th in tiles})
            if cols < 1 or rows < 1:
                return

            poly = self._polygon_in_layer_crs(layer)  # None for rectangle/MCP path
            # Rectangle zones (or the MCP path with no poly) never need a
            # per-cell clip: the grid already sits flush inside the bbox, so
            # skip the GEOS intersection call entirely for every cell.
            is_rect_zone = poly is None or poly.isGeosEqual(
                QgsGeometry.fromRect(poly.boundingBox()))
            # For a real polygon, prepare a geometry engine once so most
            # cells (the ones fully inside the shape) can be drawn raw via a
            # cheap `contains` check instead of a GEOS intersection; only
            # boundary cells still pay for the intersection.
            engine = None
            if not is_rect_zone:
                engine = QgsGeometry.createGeometryEngine(poly.constGet())
                engine.prepareGeometry()
            canvas = self.iface.mapCanvas()
            rb = QgsRubberBand(canvas, PolygonGeometry)
            # Lines only, no cell fill: the grid must NOT tint the zone interior
            # (a coloured fill doubled the blue and read as a background change).
            # DASHED + low opacity, no dark casing: a solid cased grid read as
            # heavy furniture over the imagery; a faint dashed line shows the
            # split without competing with the detections. Same brand blue as the
            # zone outline so the grid stays in one colour family. Each cell is
            # clipped to the polygon so the grid stays inside the drawn shape.
            rb.setColor(QColor(0, 0, 0, 0))  # transparent fill: no interior tint
            # Clearly visible while adjusting detail (the old alpha-110 width-1
            # line was nearly invisible over imagery), but still a DASHED line in
            # the same brand-blue family as the zone outline so it stays coherent
            # and never reads as the heavy solid cased grid we had before.
            rb.setStrokeColor(GRID_LINE)  # brand blue, clearly visible
            rb.setSecondaryStrokeColor(QColor(0, 0, 0, 0))  # no casing: lighter footprint
            rb.setLineStyle(Qt.PenStyle.DashLine)  # dashed: a guide, not furniture
            rb.setWidth(2)
            step_x = (maxx - minx) / cols
            step_y = (maxy - miny) / rows
            for i in range(cols):
                for j in range(rows):
                    cx0 = minx + i * step_x
                    cy0 = miny + j * step_y
                    cell = QgsGeometry.fromRect(
                        QgsRectangle(cx0, cy0, cx0 + step_x, cy0 + step_y))
                    if not is_rect_zone:
                        if not engine.contains(cell.constGet()):
                            cell = cell.intersection(poly)  # clip to the shape
                            if cell.isEmpty():
                                continue
                    # addGeometry(geom, layer) reprojects layer CRS -> canvas CRS.
                    rb.addGeometry(cell, layer)
            self._zone_grid_rubber_band = rb
        except (RuntimeError, AttributeError, ZeroDivisionError):
            pass

    def _clear_zone_tile_grid(self) -> None:
        """Remove the tile grid preview rubber band from the canvas."""
        if self._zone_grid_rubber_band is not None:
            self._safe_remove_rubber_band(self._zone_grid_rubber_band)
            self._zone_grid_rubber_band = None

    def _reproject_zone_to_layer_crs(
        self, zone: QgsRectangle, layer: QgsRasterLayer
    ) -> QgsRectangle:
        """Reproject a zone rectangle from canvas CRS to layer CRS.

        Returns the original zone if CRS are identical or either is invalid.
        Returns the original zone (graceful fallback) if the transform fails.
        """
        try:
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            layer_crs = layer.crs()
        except (RuntimeError, AttributeError):
            return zone

        if canvas_crs == layer_crs or not canvas_crs.isValid() or not layer_crs.isValid():
            return zone

        try:
            xform = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
            result = xform.transformBoundingBox(zone)
        except Exception:  # nosec B110 -- antimeridian, invalid CRS, etc.
            return zone

        if result.width() <= 0 or result.height() <= 0:
            return zone

        return result

    @staticmethod
    def _prepare_clip_engine(clip_geom):
        """Build a prepared GEOS engine for the zone clip polygon.

        Every detection in the pump is tested against this single fixed polygon,
        so preparing it once (GEOS builds an edge index) makes the per-detection
        contains() check cheap. Returns None when there is no clip (rectangle/MCP
        path) or on any failure, so the caller falls back to plain
        geom.intersection(clip)."""
        if clip_geom is None or clip_geom.isEmpty():
            return None
        try:
            engine = QgsGeometry.createGeometryEngine(clip_geom.constGet())
            engine.prepareGeometry()
            return engine
        except Exception:  # noqa: BLE001 - optimisation only; fall back on failure
            return None

    def _polygon_in_layer_crs(self, layer):
        """The drawn polygon zone reprojected to the layer CRS, or None when no
        polygon was drawn (e.g. the MCP/headless path sets only the bbox)."""
        if self._auto_zone_polygon is None:
            return None
        geom = QgsGeometry(self._auto_zone_polygon)  # canvas CRS copy
        try:
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            layer_crs = layer.crs()
        except (RuntimeError, AttributeError):
            return None
        if canvas_crs == layer_crs or not canvas_crs.isValid() or not layer_crs.isValid():
            return geom
        try:
            xform = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
            geom.transform(xform)
        except Exception:  # nosec B110 -- antimeridian, invalid CRS, etc.
            return None
        return geom

    def _tiles_in_polygon(self, tiles, bbox, pixel_w, pixel_h, layer):
        """Keep only tiles whose geographic footprint intersects the drawn
        polygon, so ground outside the shape is never rendered or billed.

        tiles: list of (tx, ty, tw, th) pixel rects in the rendered image space.
        bbox:  (minx, miny, maxx, maxy) of that image in LAYER CRS (image row 0
               maps to maxy; pixel Y grows downward).
        Returns tiles unchanged when no polygon was drawn (rectangle/MCP path)
        or on any error, and never culls to an empty list (safety fallback)."""
        poly = self._polygon_in_layer_crs(layer)
        if poly is None or poly.isEmpty() or not tiles:
            return tiles
        if pixel_w <= 0 or pixel_h <= 0:
            return tiles
        minx, _miny, maxx, maxy = bbox
        span_x = maxx - minx
        span_y = maxy - bbox[1]
        # Prepare the (fixed) clip polygon once so the per-tile test uses GEOS's
        # edge index instead of an unprepared intersects each call; this runs at
        # run start AND on every credit-estimate refresh, so it is worth it. Also
        # bbox-cull cheaply first: tiles in the image corners outside the drawn
        # shape's extent skip the GEOS call entirely.
        engine = self._prepare_clip_engine(poly)
        pbb = poly.boundingBox()
        kept = []
        for tile in tiles:
            tx, ty, tw, th = tile
            gx0 = minx + (tx / pixel_w) * span_x
            gx1 = minx + ((tx + tw) / pixel_w) * span_x
            gy1 = maxy - (ty / pixel_h) * span_y          # top edge (row ty)
            gy0 = maxy - ((ty + th) / pixel_h) * span_y   # bottom edge
            if gx1 < pbb.xMinimum() or gx0 > pbb.xMaximum() or gy1 < pbb.yMinimum() or gy0 > pbb.yMaximum():
                continue  # tile extent does not even overlap the polygon bbox
            cell = QgsGeometry.fromRect(QgsRectangle(gx0, gy0, gx1, gy1))
            if engine is not None:
                if engine.intersects(cell.constGet()):
                    kept.append(tile)
            elif cell.intersects(poly):
                kept.append(tile)
        return kept or tiles

    def _show_zone_rubber_band(self, rect: QgsRectangle) -> None:
        """Display a persistent blue rectangle on the canvas for the drawn zone."""
        self._clear_zone_rubber_band()
        try:
            canvas = self.iface.mapCanvas()
            rb = QgsRubberBand(canvas, PolygonGeometry)
            # Legacy rectangle zone: same chrome blue as the polygon zone, at its
            # own (heavier) alpha. Derived from the palette CHROME_BLUE so no raw
            # literal lives here.
            _rect_fill = QColor(CHROME_BLUE)
            _rect_fill.setAlpha(40)         # blue, 16% opacity
            _rect_stroke = QColor(CHROME_BLUE)
            _rect_stroke.setAlpha(180)      # blue, 71% opacity
            rb.setColor(_rect_fill)
            rb.setStrokeColor(_rect_stroke)
            rb.setWidth(2)
            rb.addPoint(QgsPointXY(rect.xMinimum(), rect.yMinimum()), False)
            rb.addPoint(QgsPointXY(rect.xMaximum(), rect.yMinimum()), False)
            rb.addPoint(QgsPointXY(rect.xMaximum(), rect.yMaximum()), False)
            rb.addPoint(QgsPointXY(rect.xMinimum(), rect.yMaximum()), True)
            self._zone_rubber_band = rb
            self._show_zone_delete_badge(canvas, rect)
        except (RuntimeError, AttributeError):
            pass

    def _show_zone_polygon_band(self, geom: QgsGeometry) -> None:
        """Persistent branded outline of the drawn polygon zone + delete badge.

        Mirrors _show_zone_rubber_band but follows the polygon shape instead of
        a rectangle. The geometry is already in canvas CRS, so setToGeometry is
        called with layer=None (no reprojection)."""
        self._clear_zone_rubber_band()
        try:
            canvas = self.iface.mapCanvas()
            rb = QgsRubberBand(canvas, PolygonGeometry)
            # Very light fill so the imagery underneath stays readable; the
            # zone reads from its outline, not from a heavy tint.
            rb.setColor(ZONE_FILL)         # blue, ~7% opacity
            rb.setStrokeColor(ZONE_STROKE)  # blue outline, kept prominent
            rb.setWidth(2)
            rb.setToGeometry(geom, None)
            self._zone_rubber_band = rb
            # Anchor the x on the shape's topmost vertex: the bbox top-right
            # corner often lands in empty space outside an irregular polygon.
            self._show_zone_delete_badge(canvas, anchor=self._polygon_badge_anchor(geom))
        except (RuntimeError, AttributeError):
            pass

    def _polygon_badge_anchor(self, geom: QgsGeometry) -> QgsPointXY:
        """Anchor point for the x badge on a polygon zone: the topmost vertex,
        tie-broken to the right. Keeps the handle attached to the drawn shape
        instead of floating at the bounding-box corner (often empty space)."""
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
        """Anchor a x badge to the zone (top-right corner for a rectangle, or an
        explicit anchor point for a polygon). Mirrors AI Edit."""
        from ..zone_selection_maptool import ZoneBadgeClickFilter, ZoneDeleteBadge, ZoneEscapeFilter
        self._remove_zone_delete_badge()
        if anchor is None:
            anchor = QgsPointXY(rect.xMaximum(), rect.yMaximum())
        badge = ZoneDeleteBadge(canvas)
        badge.set_anchor(anchor)
        self._zone_delete_badge = badge
        # The active map tool eats canvas mouse events before scene items,
        # so badge clicks are caught at the viewport level instead.
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
            self._zone_escape_filter = None
        if self._zone_badge_filter is not None:
            try:
                self.iface.mapCanvas().viewport().removeEventFilter(
                    self._zone_badge_filter)
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
        """Canvas x badge: drop the zone and send the dock back to step 2."""
        if self._auto_worker is not None:
            return  # the zone belongs to the running job; cancel that first
        # A pending review is billed work: never silently autosave it here.
        # Route through the Save/Discard/Cancel dialog instead.
        if self._auto_review is not None:
            self._on_auto_review_exit_clicked()
            return
        self._discard_auto_review()
        self._on_zone_cleared()
        if self.dock_widget:
            self.dock_widget.on_zone_deleted_from_canvas()

    def _route_escape(self) -> bool:
        """Single decision point for Escape in the Automatic flow (K2). Every
        Escape channel (dock shortcut, canvas filter) delegates here; no
        sibling escape decision logic may exist elsewhere.

        Precedence:
          - an armed example-box draw: disarm it (cancel the draw);
          - a run in flight: soft Cancel (partials salvaged into the review);
          - a pending review: the Save/Discard/Cancel exit dialog, so a billed
            result is never silently dropped nor silently autosaved;
          - a zone draw with points placed: clear the points, stay drawing
            (first Escape stage);
          - otherwise (empty draw canvas or the prompt step): exit the flow
            (second Escape stage, same as the Exit button).
        Returns True when consumed.
        """
        if self._exemplar_maptool is not None:
            # Exemplar draw stays SINGLE-stage: the sketch is throwaway, so
            # Escape disarms immediately regardless of points placed.
            self._restore_maptool_after_exemplar()
            return True
        if self._auto_worker is not None:
            self._on_auto_cancel_clicked()  # running: Escape = soft Cancel
            return True
        if self._auto_review is not None:
            self._on_auto_review_exit_clicked()
            return True
        # Two-stage Escape on the draw step. Single-fire by construction: per
        # press exactly ONE channel runs (the dock QShortcut consumes the key
        # before delivery, the canvas ZoneEscapeFilter consumes the KeyPress
        # before the tool, and the tool's own keyPressEvent runs only when
        # neither did), and the maptool mirrors this same two-stage decision.
        tool = self._zone_selection_tool
        try:
            mid_draw = (
                tool is not None
                and self.iface.mapCanvas().mapTool() is tool  # noqa: W503
                and tool.has_points()  # noqa: W503
            )
        except (RuntimeError, AttributeError):
            mid_draw = False
        if mid_draw:
            tool.clear_selection()  # emits zone_cleared; the user stays drawing
            return True
        self._on_auto_exit_clicked()
        return True

    def _route_enter(self) -> bool:
        """Single decision point for Enter in the Automatic flow (K2): review
        -> Export the visible polygons; prompt step -> Detect; no-op while a
        run is in flight. Returns True when consumed."""
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
        """Escape from the canvas filter: delegate to the single dispatcher.

        Only the armed example draw is special-cased: the draw map tool
        cancels its own draw, so the event must fall through (return False)
        instead of being consumed here.
        """
        if self._exemplar_maptool is not None:
            # The example draw tool handles Escape itself (disarm via its
            # zone_cleared / back_requested signals).
            return False
        return self._route_escape()

    def _on_auto_escape_shortcut(self) -> None:
        """Dock-level Escape (works when the canvas is not focused / on step 1).

        Pure delegate: the decision lives in _route_escape.
        """
        self._route_escape()

    def _set_zone_badge_enabled(self, enabled: bool) -> None:
        """Grey the canvas x badge out while a detection run is in flight."""
        if self._zone_delete_badge is not None:
            try:
                self._zone_delete_badge.set_enabled(enabled)
            except (RuntimeError, AttributeError):
                pass

    def _set_zone_band_fill_visible(self, visible: bool) -> None:
        """Toggle the zone polygon's light blue fill. Hidden during the post-run
        review so the detections read cleanly with no blue wash over them; the
        blue outline stays, so the zone boundary is still clear. The fill is
        restored to its light tint whenever a new zone is drawn (the band is
        rebuilt then), so this only needs to turn it off."""
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
        """Remove the persistent zone rubber band from the canvas scene."""
        self._remove_zone_delete_badge()
        self._clear_zone_tile_grid()
        if self._zone_rubber_band is not None:
            self._safe_remove_rubber_band(self._zone_rubber_band)
            self._zone_rubber_band = None

    def _clear_auto_canvas(self) -> None:
        """Single owner for removing every Automatic-flow canvas artifact
        (invariant I6): exemplar boxes + their bands, the zone outline band
        (with its x badge and tile grid), and the live selection layer.

        Each removal is guarded and idempotent, so EVERY exit of the flow
        (Exit, Finish, mode switch, project clear, teardown, zone redraw)
        shares this one cleanup instead of repeating the list inline.
        """
        self._clear_exemplars()
        self._clear_zone_rubber_band()  # also removes the x badge + tile grid
        self._remove_auto_selection_layer()

    def _on_project_cleared_auto(self) -> None:
        """Full teardown when the project is cleared or replaced (T14).

        Order matters: fold live handoff edits into the held review (hand
        work), then drop the review WITHOUT autosave (the target project is
        gone, so an export would write into the void), stop the worker, clear
        the handoff flags on both sides, end any manual session, and remove
        every canvas artifact so nothing strays into the fresh project.
        """
        try:
            if getattr(self, "_refine_handoff_active", False) and self.saved_polygons:
                self._collect_manual_refine_into_review()
        except Exception:
            pass  # nosec B110 -- teardown must never raise mid-signal
        self._auto_review = None  # NO autosave: the target project is gone
        self._stop_auto_detection()  # hard teardown; joined later via cancelled
        self._refine_handoff_active = False
        if self.dock_widget and getattr(self.dock_widget, "_refine_handoff", False):
            try:
                from ..ai_segmentation_dockwidget import Mode
                self.dock_widget.end_refine_handoff(Mode.INTERACTIVE)
            except (RuntimeError, AttributeError):
                pass
        try:
            self._teardown_manual_session()
        except Exception:
            pass  # nosec B110 -- teardown must never raise mid-signal
        self._auto_zone = None
        self._auto_zone_polygon = None
        self._clear_auto_canvas()
        if self.dock_widget:
            try:
                self.dock_widget.reset_auto_to_start()
            except (RuntimeError, AttributeError):
                pass
        QgsMessageLog.logMessage(
            "Project closed: automatic run stopped.",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )
