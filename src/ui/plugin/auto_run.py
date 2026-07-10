"""Automatic run lifecycle: start/launch/cancel/stop and the headless MCP path.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations


from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)

from ...core.i18n import tr
from ...core.telemetry import slot_guard
from .shared import (
    _RECALL_FLOOR,
    _RECALL_FLOOR_EXEMPLAR_ONLY,
    _provider_name_for_log,
)

# A cooperative stop normally confirms within ~3s (the poll loop breaks within
# one 250ms slice, then drains its in-flight tiles for up to _STOP_DRAIN_BUDGET_S).
# If a wedged reply (a GPU cold-starting past our submit deadline) keeps the
# worker from confirming, this watchdog forces the UI out of the run so Cancel
# can never read as ignored. Comfortably past the normal path, short enough that
# a genuinely stuck cancel still feels handled.
_CANCEL_WATCHDOG_MS = 5000


class AutoRunMixin:
    """Automatic run lifecycle: start/launch/cancel/stop and the headless MCP path."""

    def _auto_raster_guard_message(self, layer) -> str | None:
        """Return a blocking message when ``layer`` cannot run in Automatic, else
        None. Blocks an invalid CRS (any layer), a non-georeferenced raster, or a
        rotated/sheared affine (local rasters only). Online providers are
        georeferenced by construction and skip the georef/rotation checks.
        Fail-open on any classification error so a legitimate run is never
        blocked by a false positive."""
        try:
            online = self._is_online_provider(layer)
        except (RuntimeError, AttributeError):
            online = False
        if not online:
            try:
                georef = self._is_layer_georeferenced(layer)
            except (RuntimeError, AttributeError):
                georef = True
            if not georef:
                return tr(
                    "Automatic detection needs a georeferenced raster. "
                    "Use Manual mode for this image."
                )
            if self._raster_is_rotated(layer):
                return tr(
                    "This raster is rotated. Convert it to an axis-aligned "
                    "GeoTIFF, or use Manual mode."
                )
        try:
            crs_valid = layer.crs().isValid()
        except (RuntimeError, AttributeError):
            crs_valid = False
        if not crs_valid:
            return tr(
                "This layer has no valid coordinate reference system. "
                "Set one in Layer Properties before detecting."
            )
        return None

    def _warn_local_raster_quality(self, layer) -> None:
        """Non-blocking, once-per-run quality/perf advice for LOCAL rasters.
        Two gaps vs online:

        - A geographic-CRS raster (EPSG:4326 and friends) renders with pixels
          square in DEGREES: at latitude 45 the image reaches the model
          stretched ~1.4x horizontally, which degrades masks.
          Reprojecting to a projected CRS fixes it at the source.
        - A large local file with no overviews forces GDAL to read full-res
          pixels for every downsampled tile: minutes of disk IO on multi-GB
          orthos. Building pyramids makes detection tile reads near-instant.

        Advice only (message bar, auto-dismisses): both cases still run.
        Fail-open on any probe error."""
        try:
            if self._is_online_provider(layer):
                return
        except (RuntimeError, AttributeError):
            return
        try:
            if layer.crs().isValid() and layer.crs().isGeographic():
                self.iface.messageBar().pushInfo(
                    "AI Segmentation",
                    tr("This raster uses a geographic CRS (degrees), which "
                       "distorts the imagery sent to the AI. For best "
                       "results, reproject it to a projected CRS (e.g. UTM)."))
                return  # one hint per run is enough
        except (RuntimeError, AttributeError):
            pass
        # Overview probe: local plain files only (same source gating as the
        # rotation guard: never trigger a network open for a remote source).
        try:
            source = layer.source() or ""
            low = source.lower()
            if low.startswith("/vsi") or "://" in low:
                return
            import os
            if not os.path.isfile(source):
                return
            if os.path.getsize(source) < 512 * 1024 * 1024:
                return  # small file: full-res reads are cheap enough
            from osgeo import gdal
            ds = gdal.Open(source)
            if ds is None or ds.RasterCount < 1:
                return
            try:
                has_overviews = ds.GetRasterBand(1).GetOverviewCount() > 0
            finally:
                ds = None
            if not has_overviews:
                self.iface.messageBar().pushInfo(
                    "AI Segmentation",
                    tr("Tip: this raster has no overviews (pyramids). "
                       "Build them (Raster menu, Miscellaneous, Build "
                       "Overviews) to make detection much faster."))
        except Exception:  # noqa: BLE001 - advice must never block a run  # nosec B110
            pass

    @staticmethod
    def _raster_is_rotated(layer) -> bool:
        """True only when a LOCAL raster's affine has a non-negligible rotation /
        shear (geotransform terms gt[2]/gt[4]). Such a raster renders correctly
        through QGIS but Automatic maps masks with an axis-aligned transform, so
        the output would land rotated/offset. Fail-open: a remote/unreadable
        source (never opened here to avoid a network round-trip) returns False so
        a legitimate run is never blocked."""
        try:
            source = layer.source()
        except (RuntimeError, AttributeError):
            return False
        if not source:
            return False
        # Skip anything not a plain local file (remote/vsi sources would trigger
        # a network open just to read a geotransform).
        low = source.lower()
        if low.startswith("/vsi") or "://" in low:
            return False
        try:
            from osgeo import gdal
        except ImportError:
            return False
        ds = None
        try:
            ds = gdal.Open(source)
            if ds is None:
                return False
            gt = ds.GetGeoTransform()
            if gt is None:
                return False
            scale = max(abs(gt[1]), abs(gt[5]), 1e-9)
            return (abs(gt[2]) / scale > 1e-3) or (abs(gt[4]) / scale > 1e-3)
        except Exception:  # noqa: BLE001 - never let a guard crash the run
            return False
        finally:
            ds = None

    def _start_auto_detection(self) -> None:
        """Start an automatic cloud detection run for the current zone + layer.

        The whole zone is rendered to one QImage here on the main thread (QGIS
        renderers are main-thread only), then handed to the worker, which slices
        and PNG-encodes each tile lazily on its own thread. Only the render runs
        on the GUI thread; the per-tile encode does not, so pressing Detect no
        longer freezes the UI while many tiles are prepared.
        """
        import uuid as _uuid

        # Reset MCP result so callers waiting on QEventLoop get a fresh status.
        self._last_auto_result = None

        from ...core.activation_manager import get_auth_header, is_plugin_activated
        from ...core.cloud_detection import visible_extent_for
        from ...core.tile_manager import MAX_TILES

        if not self.dock_widget:
            return

        # Guard: a worker is still alive (running, or a cancelled one winding
        # down its last in-flight network call). Tell the user instead of
        # silently swallowing the click.
        if self._auto_worker is not None and self._auto_worker.isRunning():
            self._tel_detect_blocked("worker_busy")
            if self.dock_widget:
                try:
                    self.dock_widget.auto_status_banner.setText(
                        tr("Finishing the previous run, please wait a moment...")
                    )
                    self.dock_widget.auto_status_banner.setVisible(True)
                except (RuntimeError, AttributeError):
                    pass
            return

        # Discard any pending post-run review before starting a fresh run.
        self._discard_auto_review()
        # Disarm an armed example-box draw so it cannot fire mid-run. The draw
        # tool stays the active map tool until a box is drawn, so a Detect click
        # while it is armed would otherwise leave the user able to draw an
        # exclude box during the run (or after leaving the page).
        self._restore_maptool_after_exemplar()
        # Also hand back the tool the user had before our zone draw (pan by
        # default) so they can navigate the map while the run is in flight and
        # during the post-run review. The zone tool is already restored when the
        # zone is committed; this guarantees a Detect never leaves our draw tool
        # armed. No-op when the zone tool is not the active tool.
        self._restore_maptool_after_zone()

        layer = self._get_active_raster_layer()
        if layer is None:
            self._tel_detect_blocked("no_layer")
            QgsMessageLog.logMessage(
                "Auto detection: no raster layer selected",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return

        # Raster shape guard: block a run whose CRS is invalid, that is not
        # georeferenced, or whose affine is rotated/sheared. Automatic renders
        # through the layer but maps masks back with an axis-aligned CRS
        # transform, so these shapes would silently misalign or bill blank tiles.
        # Runs BEFORE any billable work; mirrors the zone-outside-layer abort.
        guard_msg = self._auto_raster_guard_message(layer)
        if guard_msg is not None:
            self._tel_detect_blocked("raster_shape")
            try:
                self.dock_widget.set_auto_status("error", guard_msg)
            except (RuntimeError, AttributeError):
                pass
            self.iface.messageBar().pushWarning("AI Segmentation", guard_msg)
            QgsMessageLog.logMessage(
                "Auto detection: raster shape guard blocked the run",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return
        # Non-blocking local-raster advice (geographic CRS distortion, missing
        # overviews on big files). Interactive only: headless has no user to
        # advise and its caller expects a clean message-free run.
        if not self._auto_headless_run:
            self._warn_local_raster_quality(layer)

        # Auth check.
        if not is_plugin_activated():
            self._tel_detect_blocked("not_activated")
            QgsMessageLog.logMessage(
                "Auto detection: plugin not activated",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return

        # Server-side kill switch (fails open if config is unreachable).
        from ...core.activation_manager import is_automatic_mode_enabled
        if not is_automatic_mode_enabled():
            self._tel_detect_blocked("kill_switch")
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("Automatic detection is temporarily unavailable. Please try again later."),
            )
            return

        auth = get_auth_header()
        if not auth:
            self._tel_detect_blocked("no_auth")
            QgsMessageLog.logMessage(
                "Auto detection: no auth token available",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return

        if self._tile_manager is None:
            self._setup_auto_mode()

        grid = self._compute_auto_grid(layer)
        if grid is None:
            is_online = self._is_online_provider(layer)
            try:
                layer_w = layer.width()
                layer_h = layer.height()
            except (RuntimeError, AttributeError):
                layer_w = 0
                layer_h = 0
            if is_online or layer_w <= 0 or layer_h <= 0:
                # Online layer with no zone drawn yet.
                msg = tr(
                    "Draw a zone first. Automatic detection on online layers needs a zone."
                )
                if self.dock_widget:
                    try:
                        self.dock_widget.set_auto_status("info", msg)
                    except (RuntimeError, AttributeError):
                        pass
                self.iface.messageBar().pushWarning("AI Segmentation", msg)
                QgsMessageLog.logMessage(
                    "Auto detection: online layer requires a zone; aborting",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
            else:
                QgsMessageLog.logMessage(
                    "Auto detection: could not compute pixel grid for layer",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
            return

        pixel_w = grid["pixel_w"]
        pixel_h = grid["pixel_h"]
        geo_bbox = grid["bbox"]

        # Visual-exemplar runs tile at FULL detail resolution, exactly like a
        # text run. The cloud model's box exemplars are in-image only, so the
        # worker STAMPS each example's full-res crop into every tile
        # (composite-per-tile) and sends it as that tile's box. This keeps the
        # detection resolution high: squeezing the whole zone into a single
        # small image shrinks small objects below the model's useful size, so
        # references find nothing.
        has_exemplars = self._auto_exemplar_store.count() > 0

        # A zone outside the selected raster renders blank tiles: the model
        # sees empty images, returns nothing, and credits are still consumed.
        # Online providers cover the whole world so the check only applies to
        # local rasters.
        if not self._is_online_provider(layer) and self._auto_zone is not None:
            zone_rect = QgsRectangle(geo_bbox[0], geo_bbox[1], geo_bbox[2], geo_bbox[3])
            if not zone_rect.intersects(layer.extent()):
                msg = tr(
                    "The zone is outside the selected raster layer. "
                    "Pick the right layer or redraw the zone."
                )
                # The headless/MCP caller only sees a generic "did not start"
                # unless the precise reason is handed back through this field.
                self._headless_error = msg
                try:
                    self.dock_widget.set_auto_status("error", msg)
                except (RuntimeError, AttributeError):
                    pass
                self.iface.messageBar().pushWarning("AI Segmentation", msg)
                QgsMessageLog.logMessage(
                    "Auto detection: zone does not intersect layer extent; aborting",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
                return

        # Both text and exemplar runs tile the full-resolution zone the same way
        # (exemplar crops are stamped per tile by the worker, see _launch).
        tiles = self._tile_manager.compute_grid(pixel_w, pixel_h)
        if tiles is not None:
            # Cull tiles outside the drawn polygon: empty ground is never
            # rendered, sent, or billed (no-op for the rectangle/MCP path).
            before = len(tiles)
            tiles = self._tiles_in_polygon(tiles, geo_bbox, pixel_w, pixel_h, layer)
            if len(tiles) != before:
                QgsMessageLog.logMessage(
                    "Auto detection: polygon culled {} of {} tiles".format(
                        before - len(tiles), before),
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
        if tiles is None:
            QgsMessageLog.logMessage(
                "Auto detection: zone too large (exceeds {} tiles)".format(MAX_TILES),
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return

        # Hard credit re-gate: the UI's Detect gate runs on a ~130ms cost
        # debounce, so a quick zone/detail change can leave Detect live for a
        # grid that now over-spends. Re-check the REAL culled tile count against
        # the same balance the UI gate uses (set_auto_credit_estimate) and abort
        # before any billable setup, re-showing that gate's red cost line. The
        # headless/MCP path has no dock gate and manages its own budget.
        if not self._auto_headless_run:
            balance, _is_free = self._auto_credit_snapshot()
            if balance is not None and len(tiles) > int(balance):
                self._tel_detect_blocked("cost_over_balance")
                if self.dock_widget:
                    try:
                        self.dock_widget.set_auto_credit_estimate(len(tiles))
                    except (RuntimeError, AttributeError):
                        pass
                QgsMessageLog.logMessage(
                    "Auto detection: {} tiles exceed the {} credit balance; "
                    "aborting before billing".format(len(tiles), int(balance)),
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
                return

        crs_authid = layer.crs().authid()

        # Per-tile JIT render: DO NOT render the whole zone up front. The old
        # path rendered one big QImage (for a 25-tile run, a ~4838x4838 px image
        # fetching ~400 basemap tiles) and blocked 10-40s before the FIRST
        # /predict. Instead the worker renders ONLY each tile's ground sub-extent
        # on demand (via a main-thread bridge), overlapping render with detection,
        # so the first tile submits in ~1s. The global geo_transform is still
        # computed here, identically: QgsMapSettings expands the requested extent
        # to the output aspect ratio (visibleExtent), so we read that EXPANDED
        # extent the same way the full render used to - just without starting a
        # render job (no basemap fetch). The automatic grid makes square pixels
        # (bbox aspect == pixel_w:pixel_h), so visibleExtent equals the requested
        # extent here; computing it explicitly keeps the rectangle/MCP path exact
        # too. Every tile's bbox_native is derived from this one geo_transform, so
        # detections land exactly on the imagery and overlap strips are
        # geometrically identical across seams (the merger uses IoS/containment,
        # not pixel equality, so independent per-tile renders are fine).
        import time as _time
        zone_extent = QgsRectangle(geo_bbox[0], geo_bbox[1], geo_bbox[2], geo_bbox[3])
        # The extent QGIS would actually render for (zone_extent, pixel_w,
        # pixel_h), aspect-corrected, computed WITHOUT a render. Masks map back to
        # ground through this, exactly as the old full render's actual_extent did.
        actual_extent = visible_extent_for(zone_extent, pixel_w, pixel_h)

        # Use the aspect-corrected extent so masks align with pixels.
        geo_bbox = (
            actual_extent.xMinimum(), actual_extent.yMinimum(),
            actual_extent.xMaximum(), actual_extent.yMaximum(),
        )
        geo_transform = {
            "bbox": geo_bbox,
            "img_shape": (pixel_h, pixel_w),
            "crs": crs_authid,
        }
        # Ground sample distance (layer map units per pixel) for GSD-relative
        # geometry refinement at export time.
        self._auto_gsd = (geo_bbox[2] - geo_bbox[0]) / pixel_w if pixel_w > 0 else 0.0
        # Same resolution in METERS per pixel (geographic CRSes measure _auto_gsd
        # in degrees): the review's prompt-aware Min size floor is ground m2, so
        # it needs a real meter figure regardless of the run CRS.
        self._auto_gsd_m = self._mupp_to_meters(layer, zone_extent, self._auto_gsd)
        # Fresh run: forget the previous run's observed mask resolution (the
        # worker re-measures it from this run's responses).
        self._auto_mask_gsd = 0.0

        # The basemap fetch now happens per tile inside the run (overlapped with
        # detection), not in a single up-front render, so there is no separate
        # "render phase" to time. Keep the fields for the run summary (0 render
        # ms) so _finalize's split still reads cleanly.
        self._auto_render_ms = 0
        self._auto_detect_t0 = _time.monotonic()
        QgsMessageLog.logMessage(
            "Auto detection: per-tile JIT render, zone {}x{}px, {} tile(s) "
            "(provider={})".format(
                pixel_w, pixel_h, len(tiles),
                _provider_name_for_log(layer)),
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

        # Get prompt text from dock widget.
        try:
            prompt = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            prompt = ""

        # Pick the merge policy default from the object type (overridable in the
        # review): discrete countable objects stay SEPARATE; continuous features
        # (roads, rivers, water, land cover) merge tile-split pieces. Set before
        # the merger is built below so the live preview already uses it.
        self._auto_merge_separate = self._default_merge_separate(prompt)

        # Exemplar boxes in xyxy PIXEL coords on the full-resolution zone grid.
        # Kept for the degenerate-exemplar guard + size logging below. The actual
        # stamps the worker composites come from _build_exemplar_stamps(layer)
        # (pre-rendered crisp crops), NOT from a whole-zone crop: no zone image is
        # rendered up front any more (tiles render just-in-time), so the worker's
        # legacy zone-crop fallback is inert. Empty/None for text-only runs.
        exemplar_payload = (
            self._compute_exemplar_pixel_boxes(layer, geo_bbox, pixel_w, pixel_h)
            if has_exemplars else None
        )

        # Empty-query guard (CRITICAL): a run with NEITHER a text prompt NOR an
        # exemplar sends an empty query to the backend, which returns 0 instances
        # AND still consumes credits. This happens when Detect is pressed with an
        # empty object box (e.g. a selection that never reached the prompt input,
        # or the MCP/headless path forwarding an empty object_class). Abort here,
        # BEFORE the run id / worker / any billable call, so an empty selection
        # can never silently burn credits and return nothing.
        if not prompt and not has_exemplars:
            msg = tr("Pick an object to detect first (nothing was selected).")
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("error", msg)
            except (RuntimeError, AttributeError):
                pass
            self.iface.messageBar().pushWarning("AI Segmentation", msg)
            QgsMessageLog.logMessage(
                "Auto detection: empty prompt and no exemplars; aborting before "
                "any credit is spent",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            self._auto_gsd = 0.0
            self._auto_run_id = None
            return

        # Text-only runs stamp nothing; exemplar runs fill this in below (and
        # abort here if every example fails to render).
        exemplar_stamps = None

        if has_exemplars:
            # Degenerate exemplar guard: every box failed to reproject or came
            # out sub-pixel, so the payload is empty. Sending that means an
            # empty prompt with NO exemplars, which the server rejects (or
            # which returns noise). Abort with a clear message instead of
            # silently degrading into a "detected anything" run.
            if not exemplar_payload:
                msg = tr(
                    "Could not place the example on the image. Redraw the "
                    "example box inside the zone and try again."
                )
                try:
                    self.dock_widget.set_auto_run_active(False)
                    self.dock_widget.set_auto_status("error", msg)
                except (RuntimeError, AttributeError):
                    pass
                self.iface.messageBar().pushWarning("AI Segmentation", msg)
                self._auto_gsd = 0.0
                self._auto_run_id = None
                return
            # Visibility (production-safe: dimensions only). With composite-per-
            # tile the example keeps its FULL-resolution size (its crop is stamped
            # into each tile), so the smallest example side in the full image is
            # the size the cloud model actually sees - the best predictor of a usable
            # reference. Log it for every exemplar run.
            smallest_px = min(
                min(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1])
                for b in exemplar_payload
            )
            QgsMessageLog.logMessage(
                "Auto detection (exemplar): composite-per-tile, full image "
                "{}x{}px, {} example(s), smallest example {:.0f}px".format(
                    pixel_w, pixel_h, len(exemplar_payload), smallest_px),
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )

            # Pre-render each example crop from the LAYER now, BEFORE any run
            # setup. The JIT per-tile path renders no whole-zone image, so the
            # worker has no coarse fallback to crop from: if every example fails
            # to render, the user's drawn reference would be silently dropped
            # and the run would bill tiles as a misleading text-only detection.
            # Guard it exactly like the empty-payload case above.
            exemplar_stamps = self._build_exemplar_stamps(layer)
            if not exemplar_stamps:
                msg = tr(
                    "Could not place the example on the image. Redraw the "
                    "example box inside the zone and try again."
                )
                try:
                    self.dock_widget.set_auto_run_active(False)
                    self.dock_widget.set_auto_status("error", msg)
                except (RuntimeError, AttributeError):
                    pass
                self.iface.messageBar().pushWarning("AI Segmentation", msg)
                QgsMessageLog.logMessage(
                    "Auto detection (exemplar): all example renders failed; "
                    "aborting before any credit is spent",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
                self._auto_gsd = 0.0
                self._auto_run_id = None
                return

        # Every abort guard has passed (empty prompt, unplaceable example): flip
        # to the in-run layout ONLY now, not before, so a start that bails never
        # flashes the Detect/Exit row hidden-then-restored nor wipes the tile
        # grid preview. Hide the grid here (the user watches segmentations appear
        # cleanly during the run; it is debug-only, re-shown from the review).
        from qgis.PyQt.QtWidgets import QApplication

        self._clear_zone_tile_grid()
        try:
            self.dock_widget.set_auto_run_active(True)
            self.dock_widget.auto_status_banner.setText(tr("Preparing tiles..."))
            self.dock_widget.auto_status_banner.setVisible(True)
        except (RuntimeError, AttributeError):
            pass
        QApplication.processEvents()

        from ...core.polygon_exporter import IncrementalMerger

        # Fresh run: clear any leftover live queue / repaint timer / pending
        # finalize from a prior run so a stale step can never feed this one.
        self._reset_auto_live_pipeline()
        self._auto_run_id = str(_uuid.uuid4())
        # Remember it for the error report (the server archives this run under
        # the same id, so a user can quote it to support). Best-effort.
        try:
            from ...core import telemetry
            telemetry.set_last_run_id(self._auto_run_id)
        except Exception:
            pass  # nosec B110
        # SEPARATE/count runs resolve duplicate conflicts by ADDITIVE UNION:
        # redundant readings are skipped (no outline dilation, no rebuilt
        # mega-blobs) while a seam complement that adds real area is stitched
        # in, so big buildings never ship truncated flat along the tile grid.
        # The run gsd sizes the one-pixel erosion that tells the two apart.
        # Merge/dedup scalars come from the server policy (generic client
        # fallbacks when absent); resolved once here and reused for the worker's
        # per-tile merge helpers so both share one policy read.
        from ...core import detection_policy
        self._auto_merge_scalars = detection_policy.merge_scalars()
        self._auto_merger = IncrementalMerger(
            seam_min_dim=self._auto_seam_min_dim(),
            select_duplicates=self._auto_merge_separate,
            gsd=self._auto_gsd,
            merge_ios=self._auto_merge_scalars["merge_ios"],
            dedup_ios=self._auto_merge_scalars["dedup_ios"],
            dup_ios_floor=self._auto_merge_scalars["dup_ios_floor"],
            dup_centroid_frac=self._auto_merge_scalars["dup_centroid_frac"],
            seam_span_ios=self._auto_merge_scalars["seam_span_ios"],
        )
        # Confidence becomes a live post-run filter: capture the pre-run cutoff
        # for the live merge, and reset the scored-geom store this run fills.
        # The cutoff is the PROMPT's shape-class default when the server policy
        # carries one (natural objects score lower than built ones, so a flat
        # cutoff would hide live detections the review then reveals), else the
        # generic spin value.
        from ...core.review_presets import class_confidence_for

        class_conf = class_confidence_for(prompt)
        self._auto_confidence = (
            class_conf if class_conf is not None
            else self.dock_widget.get_auto_confidence())
        self._auto_raw_count = 0
        self._auto_dense_tiles = 0
        self._auto_objects = []
        self._auto_preview_geoms = []
        self._auto_protected_geoms = []  # fresh run: no hand-edited objects yet
        self._auto_manual_removed = set()
        # CRS/GSD are known here from the render; do not reset them (a stale
        # reset to 0.0 here used to disable GSD-relative edge refinement).
        self._auto_crs_authid = crs_authid
        # Polygon (in the detection CRS) every result is clipped to, so objects
        # the cloud model finds in the rectangular overflow of a boundary tile never reach
        # the output. None on the rectangle/MCP path leaves results unclipped.
        self._auto_clip_polygon = self._polygon_in_layer_crs(layer)
        # Also confine to the raster's own data extent: where the drawn zone
        # overhangs a local raster, tiles render hard black nodata bands, and
        # the model can hallucinate detections along that fake edge. For online
        # basemaps the layer extent is world-sized, so this is a no-op there.
        if self._auto_clip_polygon is not None:
            try:
                from qgis.core import QgsGeometry as _QgsGeometry
                data_rect = _QgsGeometry.fromRect(layer.extent())
                clipped = self._auto_clip_polygon.intersection(data_rect)
                if clipped is not None and not clipped.isEmpty() and clipped.area() > 0:
                    self._auto_clip_polygon = clipped
            except Exception:  # noqa: BLE001 -- keep the plain zone clip  # nosec B110
                pass
        # Prepared GEOS engine: every detection is tested against this ONE clip
        # polygon, so preparing it once (GEOS indexes its edges) makes the
        # per-detection contains()/intersection in the pump much faster.
        self._auto_clip_engine = self._prepare_clip_engine(self._auto_clip_polygon)

        # Live results (and the post-run review) default to Random colours so
        # each detected instance streams in with its own hue: the fastest way to
        # watch building footprints separate in real time instead of one flat
        # blue mask. Seed BEFORE the layer is built so its very first repaint is
        # already Random, and sync the dock combo/legend signal-free.
        self._seed_review_display_mode()
        # Create a live selection layer to show in-progress results.
        self._remove_auto_selection_layer()
        self._auto_selection_layer = self._create_auto_selection_layer(layer)

        # Remember the run inputs for layer resolution in the post-run Manual
        # refine handoff (resume was removed). No whole-zone image is rendered
        # any more (tiles render just-in-time), so there is none to keep here.
        self._auto_run_ctx = {
            "tiles": tiles,
            "geo_transform": geo_transform,
            "crs_authid": crs_authid,
            "prompt": prompt,
            "layer_id": layer.id(),
            "zone": QgsRectangle(self._auto_zone) if self._auto_zone is not None else None,
            "detail": self._get_auto_detail_level(),
            "detection_threshold": self.dock_widget.get_auto_confidence(),
            "exemplars": exemplar_payload,
            "total": len(tiles),
        }

        # exemplar_stamps was pre-rendered above (before run setup) so a render
        # failure aborts cleanly instead of billing a misleading text-only run;
        # it stays None for text-only runs.

        # Main-thread render bridge: the worker asks it (via a queued signal +
        # wait-condition handshake) to render each tile JUST IN TIME on the GUI
        # thread, so only ~max_concurrent tiles render ahead and the first tile
        # submits in ~1s instead of after the whole zone renders. Held on self so
        # it is not garbage-collected mid-run; cleared when the run winds down.
        from ...workers.auto_detection_worker import TileRenderBridge
        self._auto_tile_bridge = TileRenderBridge(layer, geo_transform)

        # Recall floor sent to the server: resolved AT RUN START from the policy
        # getter (the module constants are the generic fallback), overridden by
        # the stored run plan when it was fetched for THIS prompt. Text runs use
        # the text floor; exemplar-only runs use the higher one.
        recall_text = detection_policy.recall_floor(_RECALL_FLOOR)
        recall_exemplar = detection_policy.recall_floor_exemplar_only(
            _RECALL_FLOOR_EXEMPLAR_ONLY)
        plan = self._active_run_plan(prompt)
        if plan is not None:
            pv = plan.get("recall_floor")
            if isinstance(pv, (int, float)) and not isinstance(pv, bool):
                recall_text = float(pv)
            pv = plan.get("recall_floor_exemplar_only")
            if isinstance(pv, (int, float)) and not isinstance(pv, bool):
                recall_exemplar = float(pv)
        detection_threshold = (
            recall_text if (prompt or "").strip() else recall_exemplar)

        # Build and start the worker via the shared launcher. The worker renders
        # + encodes each tile on demand on its own thread (render hops to the main
        # thread via the bridge), interleaved with the network waits, so pressing
        # Detect no longer freezes the UI nor blocks on a whole-zone render.
        self._launch_auto_worker(
            tile_renderer=self._auto_tile_bridge.render_tile,
            tiles=tiles,
            geo_transform=geo_transform,
            crs_authid=crs_authid,
            prompt=prompt,
            auth=auth,
            run_id=self._auto_run_id,
            # Cap concurrent in-flight tiles to the server's per-instance
            # concurrency: the per-tile cost is compute-bound, so sending more
            # than the server processes at once just queues the extra tiles
            # there. Keep in sync with the server's concurrency. The worker
            # backs off on RATE_LIMITED, so a transient cap degrades gracefully.
            max_concurrent=detection_policy.max_concurrent(),
            # Send the low recall floor so the server returns every plausible
            # mask; the client keeps them all (scored) and filters at
            # _auto_confidence, so the review slider re-filters with no re-run.
            # Exemplar-only runs (no text prior) use a higher floor: at 0.10
            # they flooded the run with low-confidence context blobs. Resolved
            # above from the policy getter / run plan.
            detection_threshold=detection_threshold,
            exemplar_stamps=exemplar_stamps,
            # Resolved once with the run merger above; the worker's per-tile
            # merge helpers reuse the same scalars.
            merge_scalars=self._auto_merge_scalars,
            subdivide_budget=self._auto_subdivide_budget(
                len(tiles), bool(exemplar_stamps)),
        )

        # Telemetry: the paid run is now committed (all guards passed, worker
        # started). Reset per-run counters and emit the started milestone.
        self._auto_tel_stop_reason = None
        self._auto_skipped_tiles = 0
        self._auto_timeout_tiles = 0
        # Pre-submit, uncharged tile drops (captured from the worker at its
        # terminal); reset per run so a stale count never leaks across runs.
        self._auto_skipped_blank_tiles = 0
        self._auto_render_failed_tiles = 0
        # Waiting-room (cold start / queue) wall time, accumulated by
        # _on_auto_queue_state and reported as warming_ms at the terminal.
        self._auto_warming_t0 = None
        self._auto_warming_ms = 0
        try:
            from ...core import telemetry
            credits_before, is_free_tier = self._auto_credit_snapshot()
            telemetry.track_auto_detect_started(
                run_id=self._auto_run_id,
                tiles=len(tiles),
                zone_km2=self._auto_zone_area_km2(),
                # Example-only run: label it so analytics can tell it apart from a
                # blank prompt. English literal (not localized) keeps aggregation
                # clean across locales; the payload still carries only the object.
                object_class=prompt or "Example match",
                detail=self._get_auto_detail_level(),
                exemplar_count=self._auto_exemplar_store.count(),
                est_credits=len(tiles),
                credits_before=credits_before,
                is_free_tier=is_free_tier,
            )
        except Exception:
            pass  # nosec B110

    def _tel_detect_blocked(self, reason: str) -> None:
        """Best-effort detect_blocked telemetry for the hard Detect guards."""
        try:
            from ...core import telemetry
            telemetry.track_detect_blocked(reason)
        except Exception:
            pass  # nosec B110

    def _auto_subdivide_budget(self, base_tiles: int, has_stamps: bool) -> int:
        """Extra tiles this run may spend re-splitting saturated tiles.

        Exemplar runs get 0 (their stamps are rendered for the run scale, a 2x
        quadrant would mismatch). Otherwise the budget is bounded by the run
        size and by the credits left AFTER the base grid, so a re-split can
        never be the thing that trips mid-run exhaustion on its own."""
        if has_stamps:
            return 0
        # The floor lets a SMALL dense zone walk the whole ladder: 4 base
        # tiles that all saturate need 16 quadrants at depth 1 and up to 48
        # more at depth 2 before every inference is under the ceiling; depth 1
        # alone is often not enough on very dense scenes. Large runs scale at
        # 2x the base grid, bounded by the same absolute cap.
        cap = min(96, max(64, 2 * base_tiles))
        from ...core.detection_policy import resplit_charge_every

        every = resplit_charge_every()
        if every <= 0:
            # The server serves quadrants on the parent tile's charge: a
            # re-split can never drain the credit allowance, no clamp needed.
            return cap
        credits, _is_free = self._auto_credit_snapshot()
        if credits is None:
            return cap
        # The server bills 1 credit per `every` quadrants, so the credits left
        # after the base grid stretch that much further.
        return max(0, min(cap, (int(credits) - base_tiles) * every))

    def _launch_auto_worker(
        self,
        *,
        tile_renderer=None,
        tiles: list,
        geo_transform: dict,
        crs_authid: str,
        prompt: str,
        auth: dict,
        run_id: str,
        max_concurrent: int = 6,  # match the inference service max concurrency
        detection_threshold: float = 0.30,
        progress_offset: int = 0,
        progress_total: int | None = None,
        exemplar_stamps: list | None = None,
        merge_scalars: dict | None = None,
        subdivide_budget: int = 0,
    ) -> None:
        """Build, wire and start the detection worker; flip the dock to its
        running state. The worker renders + encodes each tile on its own thread
        (render hops to the main thread via tile_renderer), so neither the encode
        nor the basemap fetch freezes the GUI."""
        from ...workers.auto_detection_worker import AutoDetectionWorker

        # The mask -> geometry pipeline runs on the worker now, so it needs the
        # zone clip polygon (as WKB, run CRS) and the run GSD. Both are set on
        # self by _start_auto_detection before this launcher is called. WKB is a
        # plain immutable bytes payload, safe to hand to the worker thread.
        clip_polygon_wkb = None
        if self._auto_clip_polygon is not None:
            try:
                clip_polygon_wkb = bytes(self._auto_clip_polygon.asWkb())
            except (RuntimeError, AttributeError):
                clip_polygon_wkb = None

        self._auto_worker = AutoDetectionWorker(
            tile_renderer=tile_renderer,
            tiles=tiles,
            geo_transform=geo_transform,
            crs_authid=crs_authid,
            prompt=prompt,
            auth=auth,
            run_id=run_id,
            max_concurrent=max_concurrent,
            detection_threshold=detection_threshold,
            exemplar_stamps=exemplar_stamps,
            progress_offset=progress_offset,
            progress_total=progress_total,
            clip_polygon_wkb=clip_polygon_wkb,
            gsd=self._auto_gsd,
            merge_separate=self._auto_merge_separate,
            seam_min_dim=self._auto_seam_min_dim(),
            merge_scalars=merge_scalars,
            subdivide_budget=subdivide_budget,
        )
        # Explicit QueuedConnection: these slots touch QGIS objects on the main
        # thread and the worker emits from its own thread. AutoConnection would
        # marshal correctly today (connect() runs on the main thread), but the
        # pump/repaint contract depends on queued delivery, so pin it so a future
        # refactor that wires these from another thread cannot silently break it.
        from qgis.PyQt.QtCore import Qt
        _queued = Qt.ConnectionType.QueuedConnection
        self._auto_worker.tile_completed.connect(self._on_auto_tile_completed, _queued)
        self._auto_worker.all_tiles_finished.connect(self._on_auto_all_finished, _queued)
        self._auto_worker.progress.connect(self._on_auto_progress, _queued)
        self._auto_worker.warning.connect(self._on_auto_warning, _queued)
        self._auto_worker.error.connect(self._on_auto_error, _queued)
        self._auto_worker.credits_exhausted.connect(self._on_auto_credits_exhausted, _queued)
        self._auto_worker.cancelled.connect(self._on_auto_cancelled, _queued)
        self._auto_worker.queue_state.connect(self._on_auto_queue_state, _queued)
        self._auto_worker.start()

        total_display = progress_total if progress_total else len(tiles)
        self.dock_widget.set_auto_run_active(True)
        self._set_zone_badge_enabled(False)
        self.dock_widget.set_auto_tile_progress(progress_offset, total_display)
        # No "running" banner: the tile progress bar already says it.
        try:
            self.dock_widget.auto_detect_btn.setText(tr("Detect objects"))
            self.dock_widget.set_auto_status("progress")
        except (RuntimeError, AttributeError):
            pass

    @slot_guard(stage="segment")
    def _on_auto_cancel_clicked(self) -> None:
        """Dock Cancel button: ask the worker to stop but keep its reference.

        Cleanup happens in _on_auto_cancelled once the worker confirms, which
        drops the user into the review of whatever was found so far;
        _stop_auto_detection stays the hard teardown path (unload, MCP cancel)
        where nothing is kept.
        """
        worker = self._auto_worker
        if worker is None:
            # No live worker, but the user still sees a run in progress: the
            # thread wound down without the UI resolving. Force it back to a
            # usable state so Cancel is never a no-op.
            if self.dock_widget is not None:
                try:
                    self.dock_widget.set_auto_run_active(False)
                    self.dock_widget.set_auto_status("idle")
                except (RuntimeError, AttributeError):
                    pass
            self._reset_auto_live_pipeline()
            return
        # Paint the "Stopping…" state on THIS click, before request_stop: the
        # stop is cooperative (the worker drains its in-flight tiles first), so
        # the review can be up to a few slices away. Without instant feedback
        # the click reads as ignored (the reported "cancel does nothing").
        if self.dock_widget is not None:
            try:
                self.dock_widget.set_auto_cancelling()
            except (RuntimeError, AttributeError):
                pass
        try:
            worker.request_stop()
        except (RuntimeError, AttributeError):
            pass
        # If the worker is blocked on a JIT tile render, cancel that too: it
        # makes Cancel instant on a slow basemap, and it is the use-after-free
        # guard when this cancel was triggered re-entrantly from INSIDE that
        # render's nested event loop by a layer removal or project clear (the
        # raster must not be freed while the render job still reads it).
        try:
            from ...core.cloud_detection import cancel_active_tile_render
            cancel_active_tile_render()
        except Exception:  # noqa: BLE001 - cancel is best-effort
            pass  # nosec B110
        # Watchdog: guarantee the UI leaves the run even if the worker never
        # confirms (a wedged cold-start reply, a lost terminal signal). If this
        # same worker is still active after the grace window, force the normal
        # cancelled wind-down (salvages billed partials into review, or lands
        # back on the prompt step at zero tiles).
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(
            _CANCEL_WATCHDOG_MS, lambda w=worker: self._auto_cancel_watchdog(w))

    def _auto_cancel_watchdog(self, worker) -> None:
        """Fallback wind-down if a cooperative cancel never confirms. No-ops if
        the run already resolved (worker nulled) or a new run replaced it."""
        if worker is None or self._auto_worker is not worker:
            return
        QgsMessageLog.logMessage(
            "Auto detection: cancel watchdog forcing wind-down "
            "(worker did not confirm within {}ms)".format(_CANCEL_WATCHDOG_MS),
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        # Detach live-paint + result signals so the still-winding thread cannot
        # repaint after we finalize; keep 'cancelled' connected so its real
        # emission later still releases the worker ref (mirrors _stop path).
        for sig, slot in (
            (worker.tile_completed, self._on_auto_tile_completed),
            (worker.progress, self._on_auto_progress),
            (worker.all_tiles_finished, self._on_auto_all_finished),
            (worker.warning, self._on_auto_warning),
            (worker.error, self._on_auto_error),
            (worker.credits_exhausted, self._on_auto_credits_exhausted),
            (worker.queue_state, self._on_auto_queue_state),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        # Route through the normal cancelled handler: it nulls the worker,
        # salvages any billed partials into review, and restores step-2.
        self._on_auto_cancelled()

    def _stop_auto_detection(self) -> None:
        """Cancel the running worker instantly, without freezing the UI.

        The worker thread is blocked in a network call when Cancel is clicked,
        so it cannot stop on the same instant. We therefore make the UI respond
        immediately and let the thread wind down in the background:
          1. Detach the live-draw signals so no tile still in flight can paint
             onto the canvas after the user cancels (the "cancel did nothing"
             symptom was the orphaned thread still emitting tile_completed).
          2. Drop the partial preview + merged set so the canvas clears at once.
          3. Request the stop flag (honored between the worker's network calls).
        We keep the worker reference alive: _on_auto_cancelled nulls it when the
        thread truly exits. Nulling it here would orphan a live QThread and risk
        a crash on shutdown, and blocking on wait() here would freeze the UI for
        the length of the in-flight request (the freeze the user reported).
        """
        worker = self._auto_worker
        if worker is None:
            # No live worker, but a cooperative finalize/reslice/preview chain
            # may still be pending (QTimer.singleShot turns). Invalidate it so
            # a late step can never fire after teardown (plugin reload, mode
            # switch, project clear) and rebuild review state on a dead flow.
            self._reset_auto_live_pipeline()
            return

        # Detach every result/terminal signal EXCEPT cancelled: an in-flight
        # tile that already passed its stop-check can still emit completed,
        # error, or credits_exhausted after the user cancels, and any of those
        # would repaint the canvas or overwrite _last_auto_result/the banner.
        # cancelled stays connected so _on_auto_cancelled can null the worker
        # when the thread truly winds down.
        for sig, slot in (
            (worker.tile_completed, self._on_auto_tile_completed),
            (worker.progress, self._on_auto_progress),
            (worker.all_tiles_finished, self._on_auto_all_finished),
            (worker.warning, self._on_auto_warning),
            (worker.error, self._on_auto_error),
            (worker.credits_exhausted, self._on_auto_credits_exhausted),
            # A late queue_state from the winding-down worker would otherwise
            # repaint the bar / hide the "Cancelling..." banner post-teardown.
            (worker.queue_state, self._on_auto_queue_state),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

        try:
            worker.request_stop()
        except (RuntimeError, AttributeError):
            pass
        # Cancel any in-flight JIT tile render synchronously: on the unload /
        # project-clear paths this teardown can be running re-entrantly inside
        # the render's nested event loop, and the rendered layer may be about
        # to be freed (use-after-free guard, mirrors _on_auto_cancel_clicked).
        try:
            from ...core.cloud_detection import cancel_active_tile_render
            cancel_active_tile_render()
        except Exception:  # noqa: BLE001 - cancel is best-effort
            pass  # nosec B110

        # Clear the in-progress results now; a late queued tile_completed will
        # no-op because the merger is gone. We deliberately do NOT null
        # self._auto_worker here: this is the hard path, the thread is still
        # winding down, and _on_auto_cancelled nulls it once it truly exits
        # (unload joins it). Nulling a live QThread risks a shutdown crash.
        self._auto_merger = None
        # Drop any queued tiles + stop the repaint timer so a late pump turn
        # cannot rebuild a layer we are tearing down.
        self._reset_auto_live_pipeline()
        self._remove_auto_selection_layer()
        self._set_zone_badge_enabled(True)
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.auto_status_banner.setText(tr("Cancelling..."))
                self.dock_widget.auto_status_banner.setVisible(True)
            except (RuntimeError, AttributeError):
                pass

    def _run_auto_detect_headless(
        self,
        zone_wkt: str,
        object_class: str,
        layer_name: str | None = None,
        timeout_s: int = 280,
        exemplars: list[dict] | None = None,
        detail: int | None = None,
    ) -> dict:
        """Run automatic cloud detection without user interaction.

        Bridges the async AutoDetectionWorker into a synchronous call via a
        QEventLoop. The caller (MCP API) blocks here until the worker emits
        one of its terminal signals (all_tiles_finished, error,
        credits_exhausted, cancelled) or the timeout fires.

        Parameters
        ----------
        zone_wkt:
            WKT geometry in the raster layer's CRS. Empty string = full raster.
        object_class:
            Detection prompt, e.g. "Building".
        layer_name:
            Optional raster layer name. None uses the current dock selection.
        timeout_s:
            Hard timeout in seconds. Default 280 s (well under 5 min MCP limit).
        """
        from qgis.PyQt.QtCore import QEventLoop, QTimer

        from ..ai_segmentation_dockwidget import Mode

        # Ensure the dock exists and auto-mode infrastructure is ready.
        self._ensure_dock_widget()
        if self._tile_manager is None:
            self._setup_auto_mode()

        # Switch dock to Automatic mode so _start_auto_detection can run.
        try:
            dock = self.dock_widget
            if dock and dock._mode != Mode.AUTOMATIC:
                dock._on_mode_selected(Mode.AUTOMATIC)
        except (RuntimeError, AttributeError):
            pass

        # Resolve and activate the target raster layer.
        if layer_name:
            target_layer = None
            for lyr in QgsProject.instance().mapLayers().values():
                if isinstance(lyr, QgsRasterLayer) and lyr.name() == layer_name:
                    target_layer = lyr
                    break
            if target_layer is None:
                return {"_error": "Raster layer '{}' not found".format(layer_name)}
            # _get_active_raster_layer reads the combo of the current mode;
            # set both so the target sticks regardless of mode.
            try:
                if self.dock_widget and hasattr(self.dock_widget, "layer_combo"):
                    self.dock_widget.layer_combo.setLayer(target_layer)
                if self.dock_widget and hasattr(self.dock_widget, "auto_layer_combo"):
                    self.dock_widget.auto_layer_combo.setLayer(target_layer)
            except (RuntimeError, AttributeError):
                pass
            # setLayer is a no-op when the combo does not list the target (it
            # only lists layer-tree-VISIBLE rasters, and it refreshes on a
            # debounce). A silent miss used to run detection on whatever layer
            # the combo held, billing the wrong raster. Make the target
            # selectable (visibility + immediate rebuild), then verify; if it
            # still does not stick, fail loudly instead of running.
            active = self._get_active_raster_layer()
            if active is None or active.id() != target_layer.id():
                try:
                    node = QgsProject.instance().layerTreeRoot().findLayer(
                        target_layer.id())
                    if node is not None and not node.itemVisibilityChecked():
                        node.setItemVisibilityChecked(True)
                    for combo_name in ("layer_combo", "auto_layer_combo"):
                        combo = getattr(self.dock_widget, combo_name, None)
                        if combo is None:
                            continue
                        refresh = getattr(combo, "_refresh", None)
                        if callable(refresh):
                            refresh()  # skip the 100 ms debounce
                        combo.setLayer(target_layer)
                except (RuntimeError, AttributeError):
                    pass
                active = self._get_active_raster_layer()
                if active is None or active.id() != target_layer.id():
                    return {"_error": (
                        "Raster layer '{}' exists but could not be selected "
                        "(hidden in the layer tree or filtered out). Make it "
                        "visible and retry.".format(layer_name))}

        # Parse and transform zone WKT (layer CRS) to canvas CRS.
        if zone_wkt and zone_wkt.strip():
            geom = QgsGeometry.fromWkt(zone_wkt)
            if geom is None or geom.isEmpty():
                return {"_error": "Invalid zone WKT"}
            bbox = geom.boundingBox()

            # _start_auto_detection stores _auto_zone in canvas CRS and
            # reprojects it to layer CRS internally. Convert here.
            active_layer = self._get_active_raster_layer()

            # Free-trial zone cap: mirrors the interactive draw guard so the
            # headless path can never start a run over an oversized free-tier
            # zone. The WKT is in the layer CRS (canvas CRS when no layer).
            zone_crs = active_layer.crs() if active_layer is not None else None
            cap_area = self._free_zone_cap_exceeded_km2(geom, crs=zone_crs)
            if cap_area is not None:
                try:
                    from ...core import telemetry
                    telemetry.track_auto_zone_too_large(area_km2=cap_area)
                except Exception:
                    pass  # nosec B110
                from .shared import FREE_TRIAL_MAX_ZONE_KM2
                return {"_error": (
                    "Zone is {:.1f} km2; free trial zones go up to {:g} km2. "
                    "Use a smaller zone, or subscribe to segment areas of "
                    "any size.".format(cap_area, FREE_TRIAL_MAX_ZONE_KM2)
                )}
            if active_layer is not None:
                try:
                    layer_crs = active_layer.crs()
                    canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
                    if layer_crs.isValid() and canvas_crs.isValid() and layer_crs != canvas_crs:
                        xform = QgsCoordinateTransform(layer_crs, canvas_crs, QgsProject.instance())
                        bbox = xform.transformBoundingBox(bbox)
                except Exception:  # nosec B110 -- antimeridian, invalid CRS
                    pass
            self._auto_zone = bbox
            try:
                if self.dock_widget:
                    self.dock_widget.set_auto_zone_state("zone_set")
            except (RuntimeError, AttributeError):
                pass
        else:
            self._auto_zone = None
            self._auto_zone_polygon = None
            try:
                if self.dock_widget:
                    self.dock_widget.set_auto_zone_state("idle")
            except (RuntimeError, AttributeError):
                pass

        # Set the prompt text for _start_auto_detection to pick up.
        try:
            if self.dock_widget:
                self.dock_widget.set_prompt_text(object_class)
        except (RuntimeError, AttributeError):
            pass

        # Populate visual exemplars (MCP gives bboxes in the layer CRS, same as
        # zone_wkt). The store holds canvas-CRS rects, so transform each bbox the
        # same way the zone bbox is transformed above. Single-image mode then
        # picks them up automatically in _start_auto_detection.
        self._clear_exemplars()
        if exemplars:
            ex_layer = self._get_active_raster_layer()
            ex_xform = None
            if ex_layer is not None:
                try:
                    l_crs = ex_layer.crs()
                    c_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
                    if l_crs.isValid() and c_crs.isValid() and l_crs != c_crs:
                        ex_xform = QgsCoordinateTransform(l_crs, c_crs, QgsProject.instance())
                except (RuntimeError, AttributeError):
                    ex_xform = None
            for ex in exemplars:
                try:
                    bb = ex.get("bbox") or ex.get("box")
                    if not bb or len(bb) < 4:
                        continue
                    rect = QgsRectangle(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                    if ex_xform is not None:
                        rect = ex_xform.transformBoundingBox(rect)
                    self._auto_exemplar_store.add(rect, int(ex.get("label", 1)))
                except (RuntimeError, AttributeError, ValueError, TypeError):
                    continue
            self._refresh_exemplar_chips()

        # Optional explicit detail (tile-grid density) override. Headless runs
        # otherwise seed the same OBJECT-AWARE default the interactive prompt
        # commit picks, so an API run for "tree" gets the same grid the UI
        # would (it used to keep whatever the slider held, often 1 coarse
        # tile). Clamp to >=1; the slider widget caps the upper bound to the
        # useful max for the zone.
        if detail is not None:
            try:
                if self.dock_widget:
                    self.dock_widget.set_auto_detail_value(max(1, int(detail)))
            except (RuntimeError, AttributeError, ValueError, TypeError):
                pass
        elif object_class and self._auto_zone is not None:
            try:
                # A stale interactive slider lock must not pin an API run.
                self._auto_detail_user_locked = False
                self._auto_detail_lock_prompt = ""
                self._reseed_auto_detail_from_blob(object_class)
            except (RuntimeError, AttributeError):
                pass

        # Mark this as a headless/MCP run so _on_auto_all_finished skips the
        # interactive review and exports directly (preserving MCP behavior).
        self._auto_headless_run = True
        try:
            # Start detection. _last_auto_result is reset inside _start_auto_detection.
            self._headless_error = None
            self._start_auto_detection()

            if self._auto_worker is None:
                # Surface the precise abort reason when the start path recorded
                # one; the generic catch-all is a last resort.
                reason = getattr(self, "_headless_error", None)
                return {
                    "_error": reason or (
                        "Detection did not start. Check the AI Segmentation log: "
                        "missing raster, not signed in, zone too large, or feature disabled."
                    )
                }

            # Block on QEventLoop until a terminal signal fires or timeout expires.
            # Connect AFTER _start_auto_detection so the plugin's own slots (wired
            # first) run before loop.quit; QueuedConnection ensures ordering.
            loop = QEventLoop()

            worker = self._auto_worker

            def _on_finished(_results):
                loop.quit()

            def _on_error(_msg):
                loop.quit()

            def _on_exhausted(_remaining):
                loop.quit()

            def _on_cancelled():
                loop.quit()

            worker.all_tiles_finished.connect(_on_finished)
            worker.error.connect(_on_error)
            worker.credits_exhausted.connect(_on_exhausted)
            worker.cancelled.connect(_on_cancelled)

            QTimer.singleShot(timeout_s * 1000, loop.quit)
            loop.exec()

            # Detach the local loop slots: the worker may still be alive (timeout
            # path) and a late emission must not call quit() on this finished loop.
            for sig, slot in (
                (worker.all_tiles_finished, _on_finished),
                (worker.error, _on_error),
                (worker.credits_exhausted, _on_exhausted),
                (worker.cancelled, _on_cancelled),
            ):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass

            if self._last_auto_result is None:
                # Timeout path: worker is still running (or silently died).
                self._stop_auto_detection()
                return {"_error": "Detection timed out after {}s".format(timeout_s)}

            result = self._last_auto_result
            status = result.get("status")

            if status == "completed":
                # Headless callers (MCP) have no human to click "Export to
                # layer", so commit the post-run review automatically. This
                # keeps the old API contract intact: ai_segment_detect_auto
                # still returns a saved layer name, not None.
                if self._auto_review is not None:
                    exported = self._export_auto_review()
                    if exported is not None:
                        result["layer_name"] = exported[0]
                return {
                    "instances": result.get("instances", 0),
                    "credits_used": result.get("tiles_processed", 0),
                    "layer_name": result.get("layer_name"),
                }
            if status == "error":
                return {"_error": result.get("message", "Unknown error")}
            if status == "credits_exhausted":
                out = {
                    "_error": "Credits exhausted",
                    "credits_remaining": result.get("credits_remaining", 0),
                }
                # The headless finalize already exported the billed partials
                # to a layer; surface it so the caller neither orphans nor
                # blindly retries work that was charged and kept.
                if result.get("layer_name"):
                    out["layer_name"] = result.get("layer_name")
                    out["instances"] = result.get("instances", 0)
                return out
            if status == "cancelled":
                return {"_error": "Cancelled"}

            return {"_error": "Unexpected result status: {}".format(status)}
        finally:
            self._auto_headless_run = False
