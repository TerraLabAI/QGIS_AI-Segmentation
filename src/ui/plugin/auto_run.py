"""Automatic run lifecycle: start/launch/cancel/stop and the headless MCP path.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from typing import Callable

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)

from ...core.i18n import tr
from ...core.telemetry_errors import slot_guard
from .shared import (
    _RECALL_FLOOR,
    _RECALL_FLOOR_EXEMPLAR_ONLY,
    _provider_name_for_log,
    park_orphaned_worker,
)

# A cooperative stop normally confirms within ~3s (the poll loop breaks within
# one 250ms slice, then drains its in-flight tiles for up to _STOP_DRAIN_BUDGET_S).
# If a wedged reply (a GPU cold-starting past our submit deadline) keeps the
# worker from confirming, this watchdog forces the UI out of the run so Cancel
# can never read as ignored. Comfortably past the normal path, short enough that
# a genuinely stuck cancel still feels handled.
_CANCEL_WATCHDOG_MS = 5000

# Stall watchdog. The worker's poll loop gives every in-flight tile a per-tile
# deadline, so no tile stays "pending" forever at the protocol level. What that
# cannot catch is the network CALL itself blocking forever (a half-open socket,
# a reply that never signals finished): the loop is stuck INSIDE the call and
# cannot self-check, so no terminal signal ever fires and the run shows forever
# progress. This main-thread timer watches the age of the last progress; if a run makes no
# progress for the timeout it forces a terminal, salvaging billed partials into
# the review. The timeout is server-tunable (network.stall_timeout_s); this is
# the generic client fallback.
_STALL_TIMEOUT_S = 300.0           # 5 min of zero progress = wedged worker
_STALL_CHECK_INTERVAL_MS = 5000    # main-thread poll cadence
# Seconds of zero progress, mid-run, after which the card says the connection is
# slow instead of showing a bar that simply stopped moving. Far below the stall
# timeout on purpose: five minutes of silence is a hang to the person watching
# it, whatever the code knows. Server-tunable (network.slow_notice_s).
_SLOW_NOTICE_S = 20.0

# How often the headless path asks its caller whether the user cancelled. The
# caller sits inside a blocking event loop, so this timer is the only thing that
# can hear a Cancel pressed outside the panel (a Processing dialog, a script).
# Short enough that the click feels answered, long enough to cost nothing.
_HEADLESS_CANCEL_POLL_MS = 250
# After the cancel is asked for, how long the headless path waits for the run to
# wind down and hand back what it salvaged. The soft cancel is cooperative and
# has its own watchdog (_CANCEL_WATCHDOG_MS), so this window sits past it.
_HEADLESS_CANCEL_GRACE_MS = _CANCEL_WATCHDOG_MS * 3


class AutoRunMixin:
    """Automatic run lifecycle: start/launch/cancel/stop and the headless MCP path."""

    def _auto_raster_guard_message(self, layer) -> str | None:
        """Return a blocking message when ``layer`` cannot run in Automatic, else
        None. Blocks an invalid CRS (any layer), a non-georeferenced raster, or a
        rotated/sheared affine (local rasters only). Online providers are
        georeferenced by construction and skip the georef/rotation checks.
        Fail-open on any classification error so a legitimate run is never
        blocked by a false positive."""
        # CRS first: a missing CRS also makes the georef test below fail, and
        # "set one in Layer Properties" is the more actionable advice of the
        # two for a raster that is otherwise fine (a CRS-less GeoTIFF keeps its
        # geotransform, so naming its CRS is all it takes to unblock Automatic).
        try:
            crs_valid = layer.crs().isValid()
        except (RuntimeError, AttributeError):
            crs_valid = False
        if not crs_valid:
            return tr(
                "This layer has no valid coordinate reference system. "
                "Set one in Layer Properties before detecting."
            )
        try:
            online = self._needs_canvas_render(layer)
        except (RuntimeError, AttributeError):
            online = False
        if not online:
            try:
                georef = self._is_layer_georeferenced(layer)
            except (RuntimeError, AttributeError):
                georef = True
            if not georef:
                # Name the tool that fixes it, not just the fallback: this
                # image has no position on Earth, and QGIS ships the thing
                # that gives it one. Named without a menu path, which moved
                # between the QGIS versions this plugin supports.
                return tr(
                    "This image has no position on the map, so Automatic "
                    "cannot place what it finds. Give it one with the QGIS "
                    "Georeferencer, or use Semi-Auto mode on it as is."
                )
            if self._raster_is_rotated(layer):
                # Never offer the other mode here: manual_workflow refuses a
                # rotated raster on the same detector, so the old "or use the
                # other mode" sent the user to a second refusal. "Warp
                # (Reproject)" is the Processing algorithm name and has been
                # stable across every supported QGIS.
                return tr(
                    "This raster is rotated. Run Warp (Reproject) on it to "
                    "straighten it first. Semi-Auto mode cannot read it "
                    "either."
                )
        return None

    def _warn_local_raster_quality(self, layer) -> None:
        """Non-blocking, once-per-run quality/perf advice for LOCAL rasters.
        Two gaps vs online:

        - A geographic-CRS raster (EPSG:4326 and friends) renders with pixels
          square in DEGREES, which on the ground are rectangles, so the image
          reaches the model squashed and masks suffer. The run moves itself to
          a ground-metre CRS where it can, and this only speaks up where it
          could not, because there the advice is still the user's to act on.
        - A large local file with no overviews forces GDAL to read full-res
          pixels for every downsampled tile: minutes of disk IO on multi-GB
          orthos. Building pyramids makes detection tile reads near-instant.

        Advice only (message bar, auto-dismisses): both cases still run.
        Fail-open on any probe error."""
        try:
            if self._needs_canvas_render(layer):
                return
        except (RuntimeError, AttributeError):
            return
        try:
            if layer.crs().isValid() and layer.crs().isGeographic():
                zone = self._auto_zone
                if zone is None:
                    # No zone is the native-grid path, which keeps the layer's
                    # own CRS whatever it is, so the hint applies. Asking for a
                    # run CRS over the whole raster would answer for a run that
                    # cannot happen AND memoize that answer, which is then read
                    # back for the real grid and measures degrees through a
                    # metre measurer.
                    run_crs = layer.crs()
                else:
                    run_crs = self._run_crs_for_layer(
                        layer, self._zone_in_layer_crs(zone, layer))
                if run_crs is None or run_crs == layer.crs():
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

    def _offer_automatic_setup(self, reason: str) -> None:
        """Offer the light setup right where Automatic hits the wall.

        The dock hides the install card in Automatic, so pointing at a panel
        button was pointing at nothing. This is the door. What it starts is the
        short install: Automatic needs numpy and rasterio to read the imagery
        and nothing else, because the model answers off the machine.
        """
        from qgis.PyQt.QtWidgets import QMessageBox

        box = QMessageBox(self.iface.mainWindow())
        box.setWindowTitle(tr("One-time setup"))
        box.setText(reason)
        setup_btn = box.addButton(tr("Set up now"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(tr("Not now"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(setup_btn)
        box.exec()
        if box.clickedButton() is not setup_btn:
            return
        # Spelled out rather than inferred: the dock's mode is what
        # _install_wants_local_model would read, and it is right here today,
        # but this call site knows the answer and should not depend on that.
        self._on_install_requested(include_local_model=False)

    def _push_auto_warning(self, message: str) -> None:
        """Message bar warning for an interactive run only.

        A headless caller has nobody watching the bar and its contract is a run
        that leaves no UI behind, so it reads the same sentence back from
        ``_headless_error`` instead.
        """
        if self._auto_headless_run:
            return
        try:
            self.iface.messageBar().pushWarning("AI Segmentation", message)
        except (RuntimeError, AttributeError):
            pass

    def _probe_imagery_behind_banner(self, layer, grid) -> str | None:
        """The imagery probe, with its wait explained on screen.

        The probe renders a window and waits on the answer, so on a slow link
        it is several seconds where the panel says nothing and the click reads
        as ignored. Raise the preparing banner for the wait and take it back
        down, so no path below inherits a banner it did not put up. The verdict
        is the probe's own, unchanged.
        """
        banner = None
        if not self._auto_headless_run and self.dock_widget is not None:
            try:
                banner = self.dock_widget.auto_status_banner
                banner.setText(tr("Preparing your zone..."))
                banner.setVisible(True)
            except (RuntimeError, AttributeError):
                banner = None
        if banner is not None:
            from qgis.PyQt.QtWidgets import QApplication
            QApplication.processEvents()
        try:
            return self._online_imagery_probe_message(layer, grid)
        finally:
            if banner is not None:
                try:
                    banner.setVisible(False)
                except RuntimeError:
                    pass

    def _start_auto_detection(self) -> None:
        """Start an automatic cloud detection run for the current zone + layer.

        The whole zone is rendered to one QImage here on the main thread (QGIS
        renderers are main-thread only), then handed to the worker, which slices
        and PNG-encodes each tile lazily on its own thread. Only the render runs
        on the GUI thread; the per-tile encode does not, so pressing Detect no
        longer freezes the UI while many tiles are prepared.
        """
        import uuid as _uuid

        from ...core.activation_manager import get_auth_header, is_plugin_activated
        from .shared import max_tiles_per_run_cap

        # The last result is NOT cleared here. Every guard below returns
        # without starting anything, and a caller polling auto_detect_status()
        # would then read a blank where the previous run's result still stands.
        # _launch_auto_worker clears it, because that is the one point past
        # which a new run really is starting.

        if not self.dock_widget:
            # The panel holds the prompt, the detail and the layer, so there is
            # nothing to run from. Name it: a headless caller reads this field
            # and otherwise gets only a generic "did not start".
            self._headless_error = tr(
                "The AI Segmentation panel is closed, so there is nothing to "
                "detect from. Open it and try again.")
            return

        # The identical-re-run note belongs to the run that finished, not to
        # the one being asked for. Dropped here so a start that a guard below
        # refuses cannot leave the previous run's signature standing.
        self._auto_last_run_sig = None

        # Automatic reads the raster and decodes masks in process, so it needs
        # the venv's numpy even though inference is remote. A missing or broken
        # venv used to surface here as a bare ImportError, which the slot guard
        # turned into "please try again": advice that can never work, because
        # retrying re-runs the same failing import. Name the real cause and
        # send the user to the install instead. Runs before any billable work.
        try:
            from ...core.cloud_detection import visible_extent_for
        except (ImportError, OSError) as err:
            # OSError too: a half-installed native package fails its library
            # load rather than its import on some platforms.
            self._tel_detect_blocked("deps_missing")
            deps_msg = tr(
                "Automatic mode needs a small one-time setup before it can "
                "read your imagery. It takes about a minute."
            )
            # The headless/MCP caller only sees a generic "did not start"
            # unless the precise reason is handed back through this field.
            self._headless_error = deps_msg
            try:
                self.dock_widget.set_auto_status("error", deps_msg)
            except (RuntimeError, AttributeError):
                pass
            QgsMessageLog.logMessage(
                f"Auto detection: local packages unavailable ({err})",
                "AI Segmentation", level=Qgis.MessageLevel.Critical,
            )
            # The offer is a modal window: a headless caller has nobody to
            # answer it and would block on it until its own timeout.
            if not self._auto_headless_run:
                self._offer_automatic_setup(deps_msg)
            return

        # Guard: a worker is still alive (running, or a cancelled one winding
        # down its last in-flight network call). Tell the user instead of
        # silently swallowing the click.
        if self._auto_worker is not None and self._auto_worker.isRunning():
            self._tel_detect_blocked("worker_busy")
            # A headless caller reads this field for the reason. Without it the
            # second call used to fall through to the worker of the run already
            # in flight, block on it, and report ITS instances, credits and
            # layer as its own.
            self._headless_error = tr(
                "A zone detection is already running. Wait for it to finish, "
                "or stop it, before starting another."
            )
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
        self._discard_auto_review(exit_path="new_run")
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
            self._headless_error = tr(
                "Pick a raster layer at the top of the panel first.")
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
            self._headless_error = guard_msg
            self._push_auto_warning(guard_msg)
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
            self._headless_error = tr("Sign in to run Automatic.")
            QgsMessageLog.logMessage(
                "Auto detection: plugin not activated",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return

        # Server-side kill switch (fails open if config is unreachable).
        from ...core.activation_manager import is_automatic_mode_enabled
        if not is_automatic_mode_enabled():
            self._tel_detect_blocked("kill_switch")
            kill_msg = tr(
                "Automatic detection is temporarily unavailable. Please try again later.")
            self._headless_error = kill_msg
            self._push_auto_warning(kill_msg)
            return

        auth = get_auth_header()
        if not auth:
            self._tel_detect_blocked("no_auth")
            self._headless_error = tr("Sign in to run Automatic.")
            QgsMessageLog.logMessage(
                "Auto detection: no auth token available",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return

        if self._tile_manager is None:
            self._setup_auto_mode()

        grid = self._compute_auto_grid(layer)
        if grid is None:
            is_online = self._needs_canvas_render(layer)
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
                self._headless_error = msg
                self._push_auto_warning(msg)
                QgsMessageLog.logMessage(
                    "Auto detection: online layer requires a zone; aborting",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
            else:
                self._headless_error = tr(
                    "Could not read the pixel grid of this raster. Check the "
                    "layer opens and shows in QGIS, then try again.")
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
        if not self._needs_canvas_render(layer) and self._auto_zone is not None:
            zone_rect = QgsRectangle(geo_bbox[0], geo_bbox[1], geo_bbox[2], geo_bbox[3])
            # The bbox is in the run CRS, which stops being the layer's own CRS
            # as soon as a geographic raster is moved to ground metres. Bring
            # the extent across first: metres tested against degrees never
            # intersect, so a zone drawn well inside the raster reads as
            # outside it. Fail OPEN on a refused transform, because this guard
            # only saves credits and must never be the thing that blocks a run.
            layer_extent = self._layer_extent_in_run_crs(layer, grid["crs"])
            if layer_extent is not None and not zone_rect.intersects(layer_extent):
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
                self._push_auto_warning(msg)
                QgsMessageLog.logMessage(
                    "Auto detection: zone does not intersect layer extent; aborting",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
                return

        # Which kind of source this run reads. Held for the worker (only an
        # online source answers a missing tile with a placeholder card) and for
        # the probe just below.
        self._auto_source_is_online = self._needs_canvas_render(layer)

        # The project's datum-transform table, read HERE on the main thread and
        # handed to the worker with the rest of the run state. QgsProject is not
        # thread-safe and the worker must never touch it, so reading it from the
        # worker thread was a real race on a singleton the user can change from
        # the GUI mid-run.
        self._auto_transform_context = self._read_project_transform_context()

        # Imagery probe: an online source that holds no picture of this ground
        # at this detail does not fail, it answers with a grey "no image here"
        # card. Rendered, that card is real pixels, so without this the run
        # would send a zone of grey cards and the user would pay to be told
        # nothing was found. One small window at the RUN's resolution settles
        # it before any billable work, and warms the provider cache for the
        # first real tile. Fail OPEN: only a positively recognised card stops
        # the run.
        probe_msg = self._probe_imagery_behind_banner(layer, grid)
        if probe_msg is not None:
            # No detect_blocked telemetry yet: its reason enum is owned by the
            # website registry, and a value has to land there before the plugin
            # sends it.
            self._headless_error = probe_msg
            try:
                self.dock_widget.set_auto_status("error", probe_msg)
            except (RuntimeError, AttributeError):
                pass
            self._push_auto_warning(probe_msg)
            QgsMessageLog.logMessage(
                "Auto detection: the layer serves no imagery at this detail over "
                "the zone; aborting before billing",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return

        # Both text and exemplar runs tile the full-resolution zone the same way
        # (exemplar crops are stamped per tile by the worker, see _launch).
        tiles = self._tile_manager.compute_grid(pixel_w, pixel_h)
        if tiles is not None:
            # Cull tiles outside the drawn polygon or off the raster's own
            # extent: empty ground is never rendered, sent, or billed.
            before = len(tiles)
            tiles = self._tiles_in_polygon(
                tiles, geo_bbox, pixel_w, pixel_h, layer, grid.get("crs"))
            if len(tiles) != before:
                QgsMessageLog.logMessage(
                    f"Auto detection: zone cull kept {len(tiles)} of {before} tiles",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
        if tiles is None:
            from .shared import zone_too_large_message
            self._headless_error = zone_too_large_message(max_tiles_per_run_cap())
            QgsMessageLog.logMessage(
                f"Auto detection: zone too large (exceeds {max_tiles_per_run_cap()} tiles)",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return

        # Hard surface re-gate: the UI's Detect gate runs on a ~130ms cost
        # debounce, so a quick zone change can leave Detect live over a zone
        # the month no longer covers. Re-measure the billable surface against
        # the envelope the panel reads and abort before any billable setup,
        # re-showing the panel's own wall. It gates in km2, not in tiles: the
        # bill follows the surface and never the grid, so a tile count judged
        # against a credit balance refused runs that fitted the month easily.
        # The headless/MCP path has no dock gate and manages its own budget.
        if not self._auto_headless_run and self.dock_widget is not None:
            try:
                left_km2 = self.dock_widget._auto_km2_left()
            except (RuntimeError, AttributeError):
                left_km2 = None
            zone_km2 = self._auto_zone_area_km2()
            # Unknown envelope fails open: the server enforces it, and a gate
            # that fires on a figure we never received refuses a run the
            # account could pay for.
            if left_km2 is not None and zone_km2 > 0 and zone_km2 > left_km2:
                self._tel_detect_blocked("cost_over_balance")
                try:
                    self.dock_widget.set_auto_zone_surface(zone_km2)
                except (RuntimeError, AttributeError):
                    pass
                QgsMessageLog.logMessage(
                    f"Auto detection: zone of {zone_km2:.2f} km2 over the "
                    f"{left_km2:.2f} km2 left this month; aborting before billing",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
                return

        # The CRS the grid was built in, which every tile, detection and review
        # geometry of this run is expressed in. It is the layer's own CRS except
        # where that one measures a different ground distance along each axis.
        crs_authid = grid.get("crs") or layer.crs().authid()

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
            f"Auto detection: per-tile JIT render, zone {pixel_w}x{pixel_h}px, {len(tiles)} tile(s) "
            f"(provider={_provider_name_for_log(layer)})",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

        # Get prompt text from dock widget.
        try:
            prompt = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            prompt = ""

        # Pick the merge policy from the object type: discrete countable objects
        # stay SEPARATE; continuous features (roads, rivers, water, land cover)
        # merge tile-split pieces. Set before the merger is built below so the
        # live preview already uses it. An EMPTY prompt with drawn exemplars
        # carries no token signal at all: it streams the live preview as
        # continuous cover (MAP) and the client decides count-vs-map
        # automatically at the end of the run from the run's own masks (see
        # _finalize_drain_done).
        if prompt:
            self._auto_merge_separate = self._default_merge_separate(prompt)
            self._auto_merge_mode_source = "prompt"
        else:
            self._auto_merge_separate = False
            self._auto_merge_mode_source = "signal"

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
            self._headless_error = msg
            self._push_auto_warning(msg)
            QgsMessageLog.logMessage(
                "Auto detection: empty prompt and no exemplars; aborting before "
                "any credit is spent",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            self._auto_gsd = 0.0
            self._auto_run_id = None
            return

        # Floor guard (single source of truth: detect_gate.can_detect, a
        # TYPED PROMPT): belt-and-braces behind the empty-query guard
        # above, and the one gate the headless/MCP path reads (via
        # _headless_error). Blocks BEFORE any billable call.
        from ...core.detect_gate import can_detect
        positives = self._auto_exemplar_store.positives()
        if not can_detect(bool(prompt), positives):
            msg = tr("Type what to find first. An example is optional.")
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("error", msg)
            except (RuntimeError, AttributeError):
                pass
            self._headless_error = msg
            self._push_auto_warning(msg)
            QgsMessageLog.logMessage(
                "Auto detection: no prompt; aborting before any credit "
                "is spent",
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
                self._headless_error = msg
                self._push_auto_warning(msg)
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
                f"{pixel_w}x{pixel_h}px, {len(exemplar_payload)} example(s), smallest example {smallest_px:.0f}px",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )

            # Pre-render each example crop from the LAYER now, BEFORE any run
            # setup. The JIT per-tile path renders no whole-zone image, so the
            # worker has no coarse fallback to crop from: if every example fails
            # to render, the user's drawn reference would be silently dropped
            # and the run would bill tiles as a misleading text-only detection.
            # Guard it exactly like the empty-payload case above. The zone grid
            # (geo_bbox is the aspect-corrected extent) rides along so each
            # stamp carries its full-image pixel box: on the one tile that
            # fully contains it, the worker sends the box at the REAL in-situ
            # object instead of pasting the crop there.
            exemplar_stamps = self._build_exemplar_stamps(
                layer, geo_bbox, pixel_w, pixel_h, has_prompt=bool(prompt))
            if not exemplar_stamps and prompt:
                # With a text prompt the query still stands on its own: an
                # unusable example (render failed AND no in-situ box) degrades
                # to a text run with a log line, never an abort.
                QgsMessageLog.logMessage(
                    "Auto detection: no usable example (render failed, no "
                    "in-situ box); continuing on the text prompt alone",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
                exemplar_stamps = None
            elif not exemplar_stamps:
                msg = tr(
                    "Could not place the example on the image. Redraw the "
                    "example box inside the zone and try again."
                )
                try:
                    self.dock_widget.set_auto_run_active(False)
                    self.dock_widget.set_auto_status("error", msg)
                except (RuntimeError, AttributeError):
                    pass
                self._headless_error = msg
                self._push_auto_warning(msg)
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
        if not self._auto_headless_run:
            # Clearing it is not enough: latch it off until the flow is back on
            # the setup screen. Every other "a run owns the canvas" signal is
            # transient, so an async reply landing after the last tile could
            # redraw the grid over the results (see _tile_grid_allowed).
            self._auto_grid_suppressed = True
        # Drop the zone's light blue fill for the WHOLE run, not just the
        # review: the wash over the streaming detections made them read darker
        # and more opaque live than the same objects after the run (the review
        # already hides it). The zone outline stays; error/zero paths and
        # "Re-run the whole zone" restore the fill.
        self._set_zone_band_fill_visible(False)
        try:
            self.dock_widget.set_auto_run_active(True)
            self.dock_widget.auto_status_banner.setText(tr("Preparing your zone..."))
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
        # Give back the objects a coarse reading swallows, on the classes the
        # server lists (empty by default, so this is off until it is turned
        # on). Resolved ONCE for the run and kept, because the exemplar
        # re-merge and the review's mode override build their own mergers from
        # the retained fragments: reading the policy again there would be fine,
        # but reading the PROMPT again would not (an exemplar-only run has
        # none), and the three mergers of one run must never disagree.
        # _auto_is_exemplar_only is set further down, so derive it here from the
        # same two values rather than reading it before it exists.
        self._auto_restore_partitions = detection_policy.restore_partitions_for(
            prompt, exemplar_only=bool(has_exemplars) and not prompt)
        self._auto_merger = IncrementalMerger(
            seam_min_dim=self._auto_seam_min_dim(),
            select_duplicates=self._auto_merge_separate,
            gsd=self._auto_gsd,
            # Only meaningful in count mode: union mode has no skipped members
            # to give back.
            restore_partitions=(self._auto_merge_separate and self._auto_restore_partitions),
            # Every shared scalar the merger takes, picked off its own
            # signature: a key listed by hand here would silently keep the
            # GUI merger on a stale default the moment a scalar is added.
            **detection_policy.merge_scalar_kwargs(
                IncrementalMerger, self._auto_merge_scalars),
        )
        # Confidence becomes a live post-run filter: capture the pre-run cutoff
        # for the live merge, and reset the scored-geom store this run fills.
        # ONE decision for the whole run, resolved here and read back by the
        # review through _effective_confidence_default. It used to be resolved
        # twice, and the two answers could differ: the live seed read the prompt
        # class and the hidden spin, the review read the run plan and the served
        # default. When the server carried a higher value, every object between
        # the two cutoffs appeared live and vanished the moment the review
        # opened, which reads as a run that threw half its work away.
        # The spin is hidden before a run (confidence is a post-run control), so
        # a value still at the client default means nobody picked one. MCP is
        # the only caller that sets it, and an explicit choice is honored.
        from ...core.review_defaults import AUTO_DEFAULT_CONFIDENCE
        try:
            spin_conf = (self.dock_widget.get_auto_confidence()
                         if self.dock_widget is not None else None)
            if spin_conf is not None and abs(
                    float(spin_conf) - AUTO_DEFAULT_CONFIDENCE) > 1e-9:
                self._auto_confidence = float(spin_conf)
            else:
                # Snapped to the review slider's 5% grid here, because the
                # finalize snaps the same number (_snap_review_start_confidence)
                # and an unsnapped live cutoff would sit up to half a step below
                # it: the objects in that sliver appear live and go at the end.
                self._auto_confidence = self._snap_review_start_confidence(
                    self._confidence_default_for(
                        prompt, bool(has_exemplars) and not prompt))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            self._auto_confidence = AUTO_DEFAULT_CONFIDENCE
        self._auto_raw_count = 0
        self._auto_dense_tiles = 0
        self._auto_objects = []
        self._auto_preview_geoms = []
        self._reset_review_refine_cache()
        # Count-vs-map plumbing, for exemplar-only runs alone: with no prompt
        # token to read, they decide MAP vs SEPARATE from the run's own masks
        # (the area-weighted mean tile coverage, accumulated below) and re-merge
        # the retained fragments that way at finalize. collect_raw makes the
        # worker emit NMS-only, un-gated, un-pre-stitched fragments, the neutral
        # form that decision needs. A prompted run reads its grouping from the
        # prompt, so it keeps the worker's per-tile pre-stitch and retains
        # nothing.
        from ...core.tile_manager import TILE_SIZE
        self._auto_is_exemplar_only = bool(has_exemplars) and not prompt
        self._auto_collect_raw = self._auto_is_exemplar_only
        self._auto_retain_raw = self._auto_collect_raw
        self._auto_raw_fragments = [] if self._auto_retain_raw else None
        self._auto_raw_n_total = 0
        self._auto_raw_cov_sum = 0.0
        self._auto_raw_cov_sq_sum = 0.0
        # Tile ground area the count-vs-map coverage + the separate-mode gate are
        # measured against; constant per run. Needed whenever fragments are kept.
        self._auto_tile_ground_area = (
            (TILE_SIZE * self._auto_gsd) ** 2
            if self._auto_retain_raw and self._auto_gsd > 0 else 0.0)
        self._auto_manual_removed = set()
        # Both other per-object carry-overs go with it: a removal made in the
        # last review, and the ids of the objects that review drew by hand.
        # Left standing they filter THIS run's objects, whose indices mean
        # something else entirely.
        self._auto_correction_removed = set()
        self._auto_manual_object_ids = set()
        # CRS/GSD are known here from the render; do not reset them (a stale
        # reset to 0.0 here used to disable GSD-relative edge refinement).
        self._auto_crs_authid = crs_authid
        # Polygon (in the detection CRS) every result is clipped to, so objects
        # the cloud model finds in the rectangular overflow of a boundary tile never reach
        # the output. None on the rectangle/MCP path leaves results unclipped.
        self._auto_clip_polygon = self._polygon_in_run_crs(layer)
        # Also confine to the raster's own data extent: where the drawn zone
        # overhangs a local raster, tiles render hard black nodata bands, and
        # the model can hallucinate detections along that fake edge. For online
        # basemaps the layer extent is world-sized, so this is a no-op there.
        if self._auto_clip_polygon is not None:
            try:
                from qgis.core import QgsCoordinateReferenceSystem as _QgsCrs
                from qgis.core import QgsCoordinateTransform as _QgsTransform
                from qgis.core import QgsGeometry as _QgsGeometry
                from qgis.core import QgsProject as _QgsProject
                # The clip polygon is in the run CRS, so the raster's extent has
                # to reach the same CRS before the two can be intersected.
                data_extent = layer.extent()
                run_crs = _QgsCrs(crs_authid)
                if run_crs.isValid() and run_crs != layer.crs():
                    data_extent = _QgsTransform(
                        layer.crs(), run_crs, _QgsProject.instance()
                    ).transformBoundingBox(data_extent)
                data_rect = _QgsGeometry.fromRect(data_extent)
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

        # Coarse mask-grid routing: the server may let specific classes return
        # masks on a coarser grid at fine run resolutions (faster, a validated
        # no-op on the final polygons for those classes). Resolved ONCE here
        # from the run's seeded ground resolution (meters/px) so the whole run
        # shares one decision and any re-detect over the same grid inherits it;
        # fail-closed to the full grid without a server table.
        self._auto_mask_scale = detection_policy.mask_scale_for_run(
            prompt, getattr(self, "_auto_gsd_m", 0.0))

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
            "mask_scale": self._auto_mask_scale,
            "total": len(tiles),
        }

        # Signature of THIS run's inputs. A later Detect with the same prompt,
        # detail and example count would return the same masks and only spend
        # credits, so _refresh_rerun_guard flags it before launch. Advisory
        # only, never blocks (the iterate-until-right path is a real one).
        from .shared import AutoRerunSignature, auto_rerun_scope
        self._auto_last_run_sig = AutoRerunSignature(
            self,
            (prompt, self._get_auto_detail_level(),
             self._auto_exemplar_store.count()),
            auto_rerun_scope(self))

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

        # Coverage-map zero-instance rescue: request the coverage map only for a
        # map-like TEXT prompt when the server dial is on (the same count-vs-map
        # policy the review uses decides map-like: not self._auto_merge_separate).
        # Fail-closed and additive, so nothing changes until the blob enables it.
        from ...core.cloud_detection import should_request_semantic
        return_semantic = should_request_semantic(
            detection_policy.semantic_rescue_enabled(),
            bool(prompt),
            self._auto_merge_separate,
        )

        # Additive, optional per-run provenance + benchmark fields for the
        # worker (which client + policy produced the run, the map-vs-count read,
        # and the drawn zone). None-safe end to end: absent values are omitted
        # and the request stays byte-identical to before.
        client_meta = self._build_auto_client_meta()

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
            # A run with no word to go on uses a higher floor, so the context
            # around the example does not come back as objects. Resolved above
            # from the policy getter / run plan.
            detection_threshold=detection_threshold,
            exemplar_stamps=exemplar_stamps,
            # Resolved once with the run merger above; the worker's per-tile
            # merge helpers reuse the same scalars.
            merge_scalars=self._auto_merge_scalars,
            subdivide_budget=self._auto_subdivide_budget(
                len(tiles), bool(exemplar_stamps)),
            # Collect raw (NMS-only, un-gated, un-pre-stitched) fragments so the
            # client can flip count-vs-map after the run: exemplar-only runs
            # (their own auto decision) and small continuous-default runs (so the
            # Keep step's toggle can un-split to separate). Resolved above.
            collect_raw=self._auto_collect_raw,
            # Coverage-map zero-instance rescue (map-like text prompts only,
            # server-gated); off until the blob enables it.
            return_semantic=return_semantic,
            # Empty-tile scan gate (sparse text prompts only, server-gated);
            # None until the blob enables it for this run's shape class.
            gate_config=self._auto_gate_config(
                prompt, bool(exemplar_stamps), len(tiles)),
            client_meta=client_meta,
        )

        # Telemetry: the paid run is now committed (all guards passed, worker
        # started). Reset per-run counters and emit the started milestone.
        self._auto_tel_stop_reason = None
        self._auto_skipped_tiles = 0
        self._auto_timeout_tiles = 0
        # One auto-opened error report per run (a run can hit more than one
        # terminal surface); reset here so the next run can surface its own.
        self._auto_error_dialog_shown = False
        # Pre-submit, uncharged tile drops (captured from the worker at its
        # terminal); reset per run so a stale count never leaks across runs.
        self._auto_skipped_blank_tiles = 0
        self._auto_render_failed_tiles = 0
        self._auto_unavailable_tiles = 0
        # Waiting-room (cold start / queue) wall time, accumulated by
        # _on_auto_queue_state and reported as warming_ms at the terminal.
        self._auto_warming_t0 = None
        self._auto_warming_ms = 0
        try:
            from ...core import telemetry_run_events
            credits_before, is_free_tier = self._auto_credit_snapshot()
            telemetry_run_events.track_auto_detect_started(
                run_id=self._auto_run_id,
                tiles=len(tiles),
                zone_km2=self._auto_zone_area_km2(),
                # Example-only run: label it so analytics can tell it apart from
                # a blank prompt. English literal (not localized) keeps
                # aggregation clean across locales.
                object_class=prompt or "Example match",
                detail=self._get_auto_detail_level(),
                detail_seeded=getattr(self, "_auto_detail_seeded", None),
                exemplar_count=self._auto_exemplar_store.count(),
                est_credits=len(tiles),
                credits_before=credits_before,
                is_free_tier=bool(is_free_tier),
                merge_mode="separate" if self._auto_merge_separate else "map",
                merge_mode_source=getattr(self, "_auto_merge_mode_source", "prompt"),
            )
        except Exception:
            pass  # nosec B110

    @staticmethod
    def _layer_extent_in_run_crs(layer, crs_authid: str):
        """The raster's own extent, expressed in the CRS the run works in.

        Returns None when the two CRSes cannot be related, which tells the
        caller to skip whatever test it wanted the extent for rather than
        answer it on mismatched units.
        """
        try:
            extent = layer.extent()
            run_crs = QgsCoordinateReferenceSystem(crs_authid)
            if not run_crs.isValid():
                # An authid this QGIS cannot resolve (a user-defined CRS from
                # another machine) is only safe to ignore when it names the
                # layer's own CRS, where no transform was due anyway.
                return extent if crs_authid == layer.crs().authid() else None
            if run_crs == layer.crs():
                return extent
            return QgsCoordinateTransform(
                layer.crs(), run_crs, QgsProject.instance()
            ).transformBoundingBox(extent)
        except Exception:  # noqa: BLE001 -- unrelatable CRSes, invalid layer
            return None

    def _auto_gate_config(
        self, prompt: str, has_exemplars: bool, n_tiles: int
    ) -> dict | None:
        """Empty-tile scan gate settings for THIS run, or None (today's
        behaviour). Fail-CLOSED end to end: the gate arms only for a text-only
        run of a shape class the SERVER policy explicitly gates (the client
        ships no class table), on a grid big enough to be worth a scan phase.
        The worker applies one more guard at run time (the packed scan's
        ground resolution vs the class's validated cap)."""
        try:
            from ...core import detection_policy
            if has_exemplars or not (prompt or "").strip():
                return None
            if not detection_policy.gate_enabled():
                return None
            if n_tiles < detection_policy.gate_min_tiles():
                return None
            from ...core.review_presets import shape_class_for
            # The gate keys on its OWN class taxonomy when the policy ships
            # one (gate.class_map), because the review shape classes group
            # prompts by geometry and can force classes with very different
            # scan behaviour onto one rule; the shape class stays the
            # fallback so a partial map extends the table.
            rule = detection_policy.gate_class_rule(
                detection_policy.gate_class_for_prompt(
                    prompt, shape_class_for(prompt)))
            if not rule:
                return None
            config = {
                "group": detection_policy.gate_group(),
                # Adaptive packing ceiling (the worker deepens the block side
                # up to it when the class's resolution cap allows); the
                # per-class override wins over the top-level dial.
                "max_group": rule.get(
                    "max_group", detection_policy.gate_max_group()),
                "min_score": rule["min_score"],
                "min_pixels": detection_policy.gate_min_pixels(),
            }
            if "max_scan_mupp" in rule:
                config["max_scan_mupp"] = rule["max_scan_mupp"]
            return config
        except Exception:  # noqa: BLE001 - policy doubt = gate off  # nosec B110
            return None

    def _tel_detect_blocked(self, reason: str) -> None:
        """Best-effort detect_blocked telemetry for the hard Detect guards."""
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_detect_blocked(reason)
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
        # credit_gate.subdivide_budget owns the cap (min(96, max(64, 2*base)))
        # and the credit clamp; subdivide_cap is the same ceiling before any
        # clamp. The every<=0 short-circuit stays here so the credit snapshot is
        # not read when quadrants ride the parent's charge (no clamp needed).
        from ...core.credit_gate import subdivide_budget, subdivide_cap
        from ...core.detection_policy import resplit_charge_every

        every = resplit_charge_every()
        if every <= 0:
            return subdivide_cap(base_tiles)
        credits, _is_free = self._auto_credit_snapshot()
        return subdivide_budget(credits, base_tiles, every)

    def _build_auto_client_meta(self) -> dict:
        """The run's optional per-run provenance + benchmark fields for the
        worker: which client build and policy revision produced the run, the
        client's map-vs-count read of the prompt, the drawn zone as a GeoJSON
        geometry in the run CRS, its geodesic area in km2, and the run's ground
        resolution in meters per pixel. Every value is best-effort; a missing
        one is simply omitted, so the worker leaves the payload byte-identical
        to before."""
        import math

        from ...core import detection_policy

        # The map-vs-count read is the same signal the review uses: map = the
        # continuous-cover (not counting/SEPARATE) merge policy for this prompt.
        prompt_mode = (
            "count" if getattr(self, "_auto_merge_separate", True) else "map")
        meta: dict = {
            "plugin_version": self._read_plugin_version(),
            "policy_rev": detection_policy.policy_rev(),
            "prompt_mode": prompt_mode,
        }
        zone = self._auto_zone_geojson()
        if zone is not None:
            meta["zone_geojson"] = zone
        # The same zone as WGS84 WKT, the shape the account service stores.
        zone_wkt = self._auto_zone_wkt_wgs84()
        if zone_wkt is not None:
            meta["zone_wkt"] = zone_wkt
        # Same geodesic measurement the run telemetry uses; 0.0 means unknown.
        zone_km2 = self._auto_zone_area_km2()
        if math.isfinite(zone_km2) and zone_km2 > 0:
            meta["zone_km2"] = round(zone_km2, 4)
        # Same ground resolution the run-plan request reads.
        try:
            _zone_area_m2, native_mupp = self._auto_run_plan_inputs()
        except Exception:  # noqa: BLE001 - best-effort provenance field
            native_mupp = None
        if (native_mupp is not None and math.isfinite(native_mupp)
                and native_mupp > 0):
            meta["native_mupp"] = round(float(native_mupp), 4)
        return meta

    def _billable_zone_layer(self):
        """The raster the billable zone is measured against: the layer of the
        run in flight, the dock's current pick otherwise. None when neither
        answers.

        The run context is preferred only WHILE a worker is live. It outlives
        its run, so a user who ran on raster A, exited, picked raster B and
        drew over B was quoted a zone clipped to A's extent, and then billed
        the larger true surface the claim rebuilt at run start.
        """
        try:
            if getattr(self, "_auto_worker", None) is not None:
                layer_id = (getattr(self, "_auto_run_ctx", None) or {}).get("layer_id")
                if layer_id:
                    layer = QgsProject.instance().mapLayer(layer_id)
                    if layer is not None:
                        return layer
            return self._get_active_raster_layer()
        except (RuntimeError, AttributeError):
            return None

    def _read_project_transform_context(self):
        """The project's datum-transform table, copied on the main thread.

        A copy, not the project's own object: the worker keeps it for the whole
        run and the user can edit the project's while it runs. None when it
        cannot be read, which leaves the worker on a bare context.
        """
        try:
            from qgis.core import QgsCoordinateTransformContext, QgsProject
            return QgsCoordinateTransformContext(
                QgsProject.instance().transformContext())
        except (RuntimeError, AttributeError, TypeError):
            return None

    def _zone_clipped_to_data(self, geom, crs):
        """``geom`` confined to the raster's own extent, expressed in ``crs``.

        Ground the image does not cover carries no data, so a zone drawn past
        the edge of the raster must not be billed on its empty part. Returns
        ``geom`` untouched when no layer answers, when the extent cannot be
        brought into ``crs``, or when the intersection is empty or invalid:
        this may only ever shrink the billed surface, never grow it.
        """
        try:
            layer = self._billable_zone_layer()
            authid = crs.authid() if crs is not None else ""
            if layer is None or not authid:
                return geom
            extent = self._layer_extent_in_run_crs(layer, authid)
            if extent is None or extent.isEmpty():
                return geom
            clipped = geom.intersection(QgsGeometry.fromRect(extent))
            if clipped is None or clipped.isEmpty() or clipped.area() <= 0:
                return geom
            return clipped
        except Exception:  # noqa: BLE001 - the unclipped zone is the safe answer
            return geom

    def _auto_billable_zone_geometry(self):
        """The zone the run is billed on, as ``(QgsGeometry, crs)``.

        Two shapes answer, in order. The clip polygon, in the run CRS, is the
        exact ground a drawn run covers. When there is none, the zone rectangle
        answers in the CRS it was stored in: the MCP and headless paths keep
        only a rectangle, and a run with no surface at all is billed on the
        server's own floor instead of on what the user asked for.

        Whichever answers is confined to the raster's own extent, so the three
        paths (drawn clip, drawn polygon, rectangle) all bill the part of the
        zone that has imagery under it.

        ``(None, None)`` when neither exists or anything fails.
        """
        try:
            from qgis.core import QgsCoordinateReferenceSystem, QgsGeometry

            poly = getattr(self, "_auto_clip_polygon", None)
            authid = getattr(self, "_auto_crs_authid", None)
            if poly is not None and not poly.isEmpty() and authid:
                run_crs = QgsCoordinateReferenceSystem(authid)
                return self._zone_clipped_to_data(QgsGeometry(poly), run_crs), run_crs
            rect = getattr(self, "_auto_zone", None)
            if rect is None or rect.isEmpty():
                return None, None
            crs = self._zone_source_crs(rect)
            if crs is None or not crs.isValid():
                crs = (QgsCoordinateReferenceSystem(authid) if authid else None)
            if crs is None or not crs.isValid():
                return None, None
            # Before Detect the clip polygon does not exist yet, but the drawn
            # polygon does, in the zone's draw CRS. The card must show the
            # polygon's surface, not its box, or it disagrees with the bill.
            drawn = getattr(self, "_auto_zone_polygon", None)
            if drawn is not None and not drawn.isEmpty():
                return self._zone_clipped_to_data(QgsGeometry(drawn), crs), crs
            return self._zone_clipped_to_data(QgsGeometry.fromRect(rect), crs), crs
        except Exception:  # noqa: BLE001 - best-effort provenance field
            return None, None

    def _auto_zone_wkt_wgs84(self) -> str | None:
        """The run's billable zone as a WGS84 WKT string, or None when there is
        no zone, its CRS is unknown, the WKT is oversized, or anything fails.
        Same source geometry as the zone surface."""
        geom, src = self._auto_billable_zone_geometry()
        if geom is None or src is None:
            return None
        try:
            from qgis.core import (
                QgsCoordinateReferenceSystem,
                QgsCoordinateTransform,
                QgsProject,
            )

            from ...core.zone_crs_check import zone_fits_declared_crs

            dst = QgsCoordinateReferenceSystem("EPSG:4326")
            if not src.isValid() or not dst.isValid():
                return None
            # Same refusal as the surface: a polygon that cannot be in the
            # declared CRS would reproject into a shape nowhere near the ground
            # the user drew on.
            if not zone_fits_declared_crs(geom, src):
                return None
            copy = geom
            if src != dst:
                transform = QgsCoordinateTransform(
                    src, dst, QgsProject.instance().transformContext())
                if copy.transform(transform) != 0:
                    return None
            wkt = copy.asWkt(7)
            if not wkt or len(wkt) > 100_000:
                return None
            return wkt
        except Exception:  # noqa: BLE001 - best-effort provenance field
            return None

    def _auto_zone_geojson(self) -> dict | None:
        """The drawn zone polygon as a GeoJSON geometry dict in the run CRS, or
        None on the rectangle/MCP path (no polygon) or any failure. Derived from
        the same geometry that seeds the run's clip polygon."""
        poly = getattr(self, "_auto_clip_polygon", None)
        if poly is None:
            return None
        try:
            import json

            return json.loads(poly.asJson(6))
        except Exception:  # noqa: BLE001 - best-effort provenance field
            return None

    def _build_auto_worker(
        self,
        *,
        tile_renderer=None,
        tiles: list,
        geo_transform: dict,
        crs_authid: str,
        prompt: str,
        auth: dict,
        run_id: str,
        max_concurrent: int = 6,
        detection_threshold: float = 0.30,
        progress_offset: int = 0,
        progress_total: int | None = None,
        exemplar_stamps: list | None = None,
        merge_scalars: dict | None = None,
        subdivide_budget: int = 0,
        collect_raw: bool = False,
        return_semantic: bool = False,
        gate_config: dict | None = None,
        mask_scale: int | None = None,
        client_meta: dict | None = None,
    ):
        """Construct (without wiring or starting) a detection worker from the
        current run state.

        gate_config and mask_scale ride to the worker so the scan gate and the
        per-run mask grid apply. mask_scale defaults to the run's stored value,
        so a caller that omits it still shares the run's grid. client_meta rides
        as the run's optional per-run provenance + benchmark fields (None = the
        request stays byte-identical to before)."""
        from ...workers.auto_detection_worker import AutoDetectionWorker

        # The mask -> geometry pipeline runs on the worker, so it needs the
        # zone clip polygon (as WKB, run CRS) and the run GSD. Both are set on
        # self by _start_auto_detection before any worker is built. WKB is a
        # plain immutable bytes payload, safe to hand to the worker thread.
        clip_polygon_wkb = None
        if self._auto_clip_polygon is not None:
            try:
                clip_polygon_wkb = bytes(self._auto_clip_polygon.asWkb())
            except (RuntimeError, AttributeError):
                clip_polygon_wkb = None

        return AutoDetectionWorker(
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
            collect_raw=collect_raw,
            return_semantic=return_semantic,
            gate_config=gate_config,
            mask_scale=(getattr(self, "_auto_mask_scale", 1)
                        if mask_scale is None else mask_scale),
            client_meta=client_meta,
            # Only an online source answers a missing tile with a placeholder
            # card, so only there may the worker read a flat grey render as a
            # load failure. Set with the rest of the run state in
            # _start_auto_detection.
            source_is_online=bool(getattr(self, "_auto_source_is_online", False)),
            # Read on the main thread with the rest of the run state, so the
            # worker never reaches into QgsProject for it.
            transform_context=getattr(self, "_auto_transform_context", None),
        )

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
        collect_raw: bool = False,
        return_semantic: bool = False,
        gate_config: dict | None = None,
        client_meta: dict | None = None,
    ) -> None:
        """Build, wire and start the detection worker; flip the dock to its
        running state. The worker renders + encodes each tile on its own thread
        (render hops to the main thread via tile_renderer), so neither the encode
        nor the basemap fetch freezes the GUI. client_meta rides to the worker as
        the run's optional per-run provenance + benchmark fields."""
        # The one point past which a new run really is starting, so the one
        # place the previous run's result may be dropped. Clearing it earlier
        # blanks auto_detect_status() for anyone polling while a guard refuses.
        self._last_auto_result = None
        self._auto_quota_stop_banner = None
        self._auto_worker = self._build_auto_worker(
            tile_renderer=tile_renderer,
            tiles=tiles,
            geo_transform=geo_transform,
            crs_authid=crs_authid,
            prompt=prompt,
            auth=auth,
            run_id=run_id,
            max_concurrent=max_concurrent,
            detection_threshold=detection_threshold,
            progress_offset=progress_offset,
            progress_total=progress_total,
            exemplar_stamps=exemplar_stamps,
            merge_scalars=merge_scalars,
            subdivide_budget=subdivide_budget,
            collect_raw=collect_raw,
            return_semantic=return_semantic,
            gate_config=gate_config,
            # Resolved once per run in _start_auto_detection and stored on the
            # run context, so every worker built for this run (and any re-detect
            # over the same grid) shares the same mask grid.
            mask_scale=getattr(self, "_auto_mask_scale", 1),
            client_meta=client_meta,
        )
        # Start the live stitcher BEFORE the worker: it owns the merger for the
        # length of the run, and the first tile must already have somewhere to
        # go. Without it every tile_completed would be dropped on the floor.
        self._start_auto_stitcher(geo_transform)
        # Explicit QueuedConnection: these slots touch QGIS objects on the main
        # thread and the worker emits from its own thread. AutoConnection would
        # marshal correctly today (connect() runs on the main thread), but the
        # hand-off to the stitcher depends on queued delivery, so pin it so a
        # future refactor that wires these from another thread cannot silently
        # break it.
        from qgis.PyQt.QtCore import Qt
        _queued = Qt.ConnectionType.QueuedConnection
        self._auto_worker.tile_completed.connect(self._on_auto_tile_completed, _queued)
        self._auto_worker.all_tiles_finished.connect(self._on_auto_all_finished, _queued)
        self._auto_worker.progress.connect(self._on_auto_progress, _queued)
        self._auto_worker.warning.connect(self._on_auto_warning, _queued)
        self._auto_worker.error.connect(self._on_auto_error, _queued)
        self._auto_worker.credits_exhausted.connect(self._on_auto_credits_exhausted, _queued)
        # Bound to THIS worker. _stop_auto_detection leaves 'cancelled' wired
        # on purpose, so a queued one from run N can land after run N+1 has
        # started, and an unbound handler would tear the live run down.
        self._auto_cancelled_slot = (
            lambda w=self._auto_worker: self._on_auto_cancelled(worker=w))
        self._auto_worker.cancelled.connect(self._auto_cancelled_slot, _queued)
        self._auto_worker.queue_state.connect(self._on_auto_queue_state, _queued)
        self._auto_worker.rescan_state.connect(self._on_auto_rescan_state, _queued)
        self._auto_worker.run_phase.connect(self._on_auto_run_phase, _queued)

        # The dock flips to its running state BEFORE the worker starts. The
        # imagery probe above runs a nested event loop, so the user can close
        # the panel during it: touching a deleted dock then raises, and a raise
        # after start() would leave a paid run going with no Cancel button.
        # Aborting here costs nothing, because nothing billable has begun.
        total_display = progress_total or len(tiles)
        # A run with no dock at all is the headless surface, which has no UI to
        # protect and must still run.
        if self.dock_widget is not None:
            try:
                self.dock_widget.set_auto_run_active(True)
                self._set_zone_badge_enabled(False)
                # The grid the user was quoted and is charged for. A dense run
                # re-splits saturated tiles into free quadrants, which grow the
                # worker's total, so the count row would otherwise read a number
                # nobody agreed to pay.
                self.dock_widget.set_auto_billed_tile_total(total_display)
                self.dock_widget.set_auto_tile_progress(progress_offset, total_display)
                # No "running" banner: the tile progress bar already says it.
                self.dock_widget.auto_detect_btn.setText(tr("Detect objects"))
                self.dock_widget.set_auto_status("progress")
            except (RuntimeError, AttributeError):
                QgsMessageLog.logMessage(
                    "Auto detection: the panel went away before the run "
                    "started; aborting before any tile is sent",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
                # The flip is several calls: the first can land and a later one
                # raise, which left the panel painted as a run with no worker
                # behind it and no way back. Undo what took.
                try:
                    self.dock_widget.set_auto_run_active(False)
                    self._set_zone_badge_enabled(True)
                except (RuntimeError, AttributeError):
                    pass
                self._auto_worker = None
                self._auto_cancelled_slot = None
                self._reset_auto_live_pipeline()
                self._drop_auto_tile_bridge()
                return

        self._auto_worker.start()
        # Arm the stall watchdog for this run (see _on_auto_stall_check).
        self._start_auto_stall_watchdog()
        # Canvas preview jobs re-render (and on online basemaps re-FETCH) the
        # out-of-view margin after every repaint; during live tile streaming
        # that is pure waste stacked on our own coalesced repaints. Paused for
        # the run, restored by _stop_auto_live_pump at every terminal.
        self._pause_preview_jobs()

    # ------------------------------------------------------------------
    # Stall watchdog (main thread): guarantee a hung run always terminates
    # ------------------------------------------------------------------

    def _start_auto_stall_watchdog(self) -> None:
        """Arm the per-run stall watchdog: a repeating main-thread timer that
        forces a terminal if the worker stops making progress (a network call
        wedged with no timeout, which the worker's own loop cannot catch because
        it is blocked inside it). Idempotent; the timestamp is reset here so a
        fresh run always gets the full window."""
        import time as _t
        self._auto_last_progress_ts = _t.monotonic()
        timer = getattr(self, "_auto_stall_timer", None)
        # A cached timer whose parent dock is gone (panel closed, plugin
        # reloaded) is a deleted C++ object: every call on it raises. So it is
        # rebuilt whenever its parent is not the dock in front of the user.
        if timer is not None and not self._stall_timer_owns_dock(timer):
            timer = None
            self._auto_stall_timer = None
        if timer is None:
            from qgis.PyQt.QtCore import QTimer
            timer = QTimer(self.dock_widget)  # parented to the dock (never to
            # the controller, which is not a QObject), so it dies with the dock
            timer.setInterval(_STALL_CHECK_INTERVAL_MS)
            timer.timeout.connect(self._on_auto_stall_check)
            self._auto_stall_timer = timer
        try:
            timer.start()
        except (RuntimeError, AttributeError):
            self._auto_stall_timer = None

    def _stall_timer_owns_dock(self, timer) -> bool:
        """Whether ``timer`` is still parented to the dock in front of the user.

        False also when reading the parent raises, which is what a timer whose
        C++ half was deleted with its dock does."""
        try:
            return timer.parent() is self.dock_widget
        except (RuntimeError, AttributeError):
            return False

    def _stop_auto_stall_watchdog(self) -> None:
        """Disarm the stall watchdog. Called at every terminal (via
        _finalize_auto_results and the hard teardown), so the timer never ticks
        into a review or an idle dock. Best-effort and idempotent."""
        timer = getattr(self, "_auto_stall_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):
                pass

    def _note_auto_progress(self) -> None:
        """Record that the run advanced, so the stall watchdog's window resets.
        Called from the progress and tile-completed handlers."""
        import time as _t
        self._auto_last_progress_ts = _t.monotonic()
        # Clear the slow-link note on the answer itself, not on the next
        # watchdog tick: the tile the user was waiting for has landed, and a
        # sentence saying the link is slow must not outlive it by seconds.
        if getattr(self, "_auto_link_slow_shown", False) and self.dock_widget is not None:
            self._auto_link_slow_shown = False
            try:
                self.dock_widget.set_auto_link_slow(False)
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_run_phase(self, name: str) -> None:
        """Slot: the worker says which work the pre-first-tile wait belongs to
        ("imagery" or "detecting"). Passed straight to the card, which picks the
        copy. Best-effort: a phase that never arrives leaves the old wording."""
        if self.dock_widget is None:
            return
        try:
            self.dock_widget.set_auto_wait_phase(name)
        except (RuntimeError, AttributeError):
            pass

    def _note_auto_link_slow(self, silent_for_s: float) -> None:
        """Tell the card whether the run has gone quiet long enough to say so.

        A run whose tiles have stopped landing looks identical to a hung one,
        and the only honest difference is that we still expect it to finish. The
        card shows a line saying so instead of a bar that stopped moving."""
        if self.dock_widget is None:
            return
        from ...core.detection_policy import slow_notice_s
        slow = silent_for_s >= slow_notice_s(_SLOW_NOTICE_S)
        self._auto_link_slow_shown = slow
        try:
            self.dock_widget.set_auto_link_slow(slow)
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_stall_check(self) -> None:
        """Watchdog tick. Says so on the card once a live run has been quiet for
        the slow-notice window, and forces a terminal when it has made no
        progress for the full stall timeout: the worker is wedged inside a
        network call, so the run would otherwise show forever progress with no
        signal. Self-stops when there is no running worker (a terminal already
        resolved the run)."""
        worker = self._auto_worker
        if worker is None:
            self._stop_auto_stall_watchdog()
            return
        try:
            running = worker.isRunning()
        except RuntimeError:
            return
        import time as _t

        from ...core.detection_policy import stall_timeout_s
        from ...core.run_watchdog import run_is_stalled
        last = getattr(self, "_auto_last_progress_ts", None)
        if running and last is not None:
            self._note_auto_link_slow(_t.monotonic() - last)
        timeout = stall_timeout_s(_STALL_TIMEOUT_S)
        if not run_is_stalled(running, last, _t.monotonic(), timeout):
            return
        # Stalled. One-shot: disarm before handling so we never double-fire.
        self._stop_auto_stall_watchdog()
        self._handle_auto_stall(worker, int(timeout))

    def _handle_auto_stall(self, worker, timeout_s: int) -> None:
        """A run made no progress for the timeout: wake the wedged worker, then
        salvage its billed partials into the review as a TIMEOUT (never a silent
        forever-progress). Mirrors the cancel watchdog's safe wind-down."""
        if worker is None or self._auto_worker is not worker:
            return
        QgsMessageLog.logMessage(
            "Auto detection: stall watchdog forcing wind-down "
            f"(no progress for {timeout_s}s)",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        # Mark the worker stalled so its own terminal (if it ever unwinds) stays
        # silent, then wake it out of the blocked network call.
        try:
            worker._stop_reason = "stalled"
            worker.request_stop()
        except (RuntimeError, AttributeError):
            pass
        self._cancel_active_tile_render()
        # Detach live-paint + result signals so the still-winding thread cannot
        # repaint or double-finalize after we salvage; keep 'cancelled'
        # connected (it never fires for a stalled worker, but this mirrors the
        # cancel watchdog and stays safe if the reason is ever cleared).
        for sig, slot in (
            (worker.tile_completed, self._on_auto_tile_completed),
            (worker.progress, self._on_auto_progress),
            (worker.all_tiles_finished, self._on_auto_all_finished),
            (worker.warning, self._on_auto_warning),
            (worker.error, self._on_auto_error),
            (worker.credits_exhausted, self._on_auto_credits_exhausted),
            (worker.queue_state, self._on_auto_queue_state),
            (worker.run_phase, self._on_auto_run_phase),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._on_auto_cancelled(reason="stalled")

    @slot_guard(stage="segment")
    def _on_auto_cancel_clicked(self) -> None:
        """Dock Cancel button: ask the worker to stop but keep its reference.

        Cleanup happens in _on_auto_cancelled once the worker confirms, which
        drops the user into the review of whatever was found so far;
        _stop_auto_detection stays the hard teardown path (unload, MCP cancel)
        where nothing is kept.
        """
        if getattr(self, "_auto_finalize_state", None) is not None:
            # Between the last tile and the review, with the "building the
            # shapes" card up. The worker is already gone, so Cancel used to
            # fall into the no-worker branch below, reset the live pipeline and
            # leave a paid run with nowhere to land. Swallow it: the window is
            # short, it is announced on screen, and Cancel is not on offer in
            # it (Escape is swallowed there for the same reason).
            return
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
        # If the worker is blocked on a JIT tile render, cancel that too.
        self._cancel_active_tile_render()
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
            f"(worker did not confirm within {_CANCEL_WATCHDOG_MS}ms)",
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
            (worker.run_phase, self._on_auto_run_phase),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        # If the thread is still executing (the very reason this watchdog
        # fired), _on_auto_cancelled must NOT be allowed to drop the last
        # reference to it: garbage-collecting a running QThread hard-aborts
        # QGIS, and freeing the busy guard would let a second run start over
        # the wedged thread. Park a strong reference (released when the thread
        # finishes) first, exactly as the unload path does.
        try:
            still_running = worker.isRunning()
        except RuntimeError:
            still_running = False
        if still_running:
            park_orphaned_worker(worker)
        # Route through the normal cancelled handler: it nulls the worker,
        # salvages any billed partials into review, and restores step-2.
        self._on_auto_cancelled()

    def _cancel_active_tile_render(self) -> None:
        """Tear down any in-flight just-in-time tile render. Best-effort.

        The render runs a nested event loop on the GUI thread, so a teardown
        can re-enter from inside it (a layer removal, a project clear). The
        raster must not be freed while the render job still reads it, and a
        cancel also makes the user's Cancel instant on a slow basemap.
        """
        try:
            from ...core.cloud_detection import cancel_active_tile_render
            cancel_active_tile_render()
        except Exception:  # noqa: BLE001 - cancel is best-effort
            pass  # nosec B110

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
        # Hard teardown does not route through _finalize_auto_results, so disarm
        # the stall watchdog here too.
        self._stop_auto_stall_watchdog()
        # Cancel any in-flight JIT tile render FIRST, before the no-worker exit
        # below: this teardown can be running re-entrantly inside a render's
        # nested event loop (a layer removed while an exemplar stamp renders),
        # and the rendered layer may be about to be freed. Every teardown path
        # has to cancel, worker or not (use-after-free guard).
        self._cancel_active_tile_render()
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
            (worker.run_phase, self._on_auto_run_phase),
            # Same for a late rescan mark: it would draw on a canvas the
            # teardown has just cleared, with no run left to remove it.
            (worker.rescan_state, self._on_auto_rescan_state),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

        try:
            worker.request_stop()
        except (RuntimeError, AttributeError):
            pass

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
                # A hard teardown mid-hand-over kills the finalize chain, so no
                # review will ever claim the screen: release the run-card hold
                # or the dock stays frozen on it (see set_auto_finalizing).
                self.dock_widget.set_auto_finalizing(False)
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.auto_status_banner.setText(tr("Cancelling..."))
                self.dock_widget.auto_status_banner.setVisible(True)
            except (RuntimeError, AttributeError):
                pass

    def _store_auto_zone_from_geometry(self, geom, active_layer) -> QgsRectangle:
        """Store an API-supplied zone geometry as the Automatic zone.

        ``geom`` is in ``active_layer``'s CRS, or in canvas numbers when no
        layer resolved. Mirrors the interactive draw: a polygon is kept beside
        the rectangle, so tiles outside it are culled and the run is billed on
        the shape the caller asked for rather than on its bounding box. A
        non-polygon geometry leaves the rectangle alone. Returns the rectangle
        that was stored.
        """
        from ...core.qt_compat import PolygonGeometry

        shape = QgsGeometry(geom)
        # None means canvas numbers, and it only stays None while the
        # conversion below actually reaches the canvas.
        zone_crs = None
        if active_layer is not None:
            try:
                layer_crs = active_layer.crs()
                canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
                if (layer_crs.isValid() and canvas_crs.isValid()
                        and layer_crs != canvas_crs):
                    xform = QgsCoordinateTransform(
                        layer_crs, canvas_crs, QgsProject.instance())
                    if shape.transform(xform) != 0:
                        raise ValueError("zone transform failed")
            except Exception:  # noqa: BLE001 -- antimeridian, invalid CRS
                # The shape never left the layer's CRS. Calling it canvas
                # numbers would tile, bill and clip ground nobody asked for.
                shape = QgsGeometry(geom)
                zone_crs = active_layer.crs()
        rect = QgsRectangle(shape.boundingBox())
        # Stored with the CRS its numbers are actually in, so a project CRS
        # change before the run cannot reinterpret them.
        self._store_auto_zone(rect, crs=zone_crs)
        self._auto_zone_polygon = None
        if not shape.isEmpty() and shape.type() == PolygonGeometry:
            # Same repair as the draw: a self-intersecting outline breaks area
            # math and tile culling. Fail-open to the raw polygon.
            from ...core.layer_conventions import repair_polygon
            repaired = repair_polygon(QgsGeometry(shape))
            self._auto_zone_polygon = (
                repaired if repaired is not None and not repaired.isEmpty()
                else shape)
        return rect

    def _arm_headless_cancel_poll(self, loop, should_cancel, state: dict):
        """Start the timer that lets a blocked headless run hear its caller's Cancel.

        Returns the running QTimer, or None when the caller gave no
        ``should_cancel``. The first True takes the panel's own soft-cancel
        route, which keeps the tiles the account already paid for, and the wait
        goes on until the run hands them back or the grace window closes.
        ``state`` carries "asked" back out, because the caller has to tell a
        cancel from a timeout.
        """
        state["asked"] = False
        state["deadline"] = 0.0
        if should_cancel is None:
            return None

        import time as _t

        from qgis.PyQt.QtCore import QTimer

        timer = QTimer(self.dock_widget)
        timer.setInterval(_HEADLESS_CANCEL_POLL_MS)

        def _tick():
            if state["asked"]:
                if (self._last_auto_result is not None
                        or _t.monotonic() >= state["deadline"]):
                    loop.quit()
                return
            try:
                wants_stop = bool(should_cancel())
            except Exception:  # noqa: BLE001 -- a caller's callback never reaches the loop
                # Unreadable answer: read it as "not cancelled" and stop asking.
                timer.stop()
                return
            if not wants_stop:
                return
            state["asked"] = True
            state["deadline"] = _t.monotonic() + _HEADLESS_CANCEL_GRACE_MS / 1000.0
            self._on_auto_cancel_clicked()

        timer.timeout.connect(_tick)
        timer.start()
        return timer

    @staticmethod
    def _disarm_headless_cancel_poll(timer) -> None:
        """Stop and drop the headless cancel timer, on every exit path.

        A timer still ticking against a finished event loop is the crash class
        this path has to stay clear of, so the caller stops it in a finally.
        """
        if timer is None:
            return
        try:
            timer.stop()
            timer.timeout.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            timer.deleteLater()
        except RuntimeError:
            pass

    @staticmethod
    def _resolve_headless_raster(name_or_id: str):
        """The raster a headless caller named, as ``(layer, ambiguous)``.

        Tried as a layer id first: an id is unique, and it is what every other
        answer of the API hands back, so a caller feeding one back used to get
        "no such layer" while a name typed by a person still works. Two rasters
        sharing a name have no right answer, so that case is reported rather
        than billed against whichever the project listed first.
        """
        project = QgsProject.instance()
        by_id = project.mapLayer(name_or_id)
        if isinstance(by_id, QgsRasterLayer):
            return by_id, False
        matches = [
            lyr for lyr in project.mapLayers().values()
            if isinstance(lyr, QgsRasterLayer) and lyr.name() == name_or_id
        ]
        if len(matches) > 1:
            return None, True
        return (matches[0] if matches else None), False

    def _restore_dock_after_headless(self, mode_before, shown_layer_id: str) -> None:
        """Give back what a headless run borrowed from the panel: the mode it
        switched to Automatic, and the layer-tree visibility it turned on to
        reach the target raster.

        The mode goes back only when nothing is left running and no review is
        open: a run still winding down (a cancel inside its grace window) would
        be torn down by the switch, and the tiles it already paid for with it.
        """
        if shown_layer_id:
            try:
                node = QgsProject.instance().layerTreeRoot().findLayer(shown_layer_id)
                if node is not None:
                    node.setItemVisibilityChecked(False)
            except (RuntimeError, AttributeError):
                pass
        if mode_before is None:
            return
        worker = getattr(self, "_auto_worker", None)
        if worker is not None and worker.isRunning():
            return
        if getattr(self, "_auto_review", None) is not None:
            return
        try:
            dock = self.dock_widget
            if dock is not None and dock._mode != mode_before:
                dock._on_mode_selected(mode_before)
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
        confidence: float | None = None,
        refine: dict | None = None,
        should_cancel: Callable[[], bool] | None = None,
        instance_colors: bool = False,
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
        confidence:
            Explicit cutoff in [0, 1] for this run. None keeps the value the
            run plan resolves for the prompt.
        refine:
            Shape-cleanup overrides merged onto the run's review preset for the
            duration of this call. Keys are those of the preset dict
            (``simplify_px``, ``smooth``, ``expand_px``, ``fill_holes``,
            ``fill_holes_max_m2``, ``clean_px``, ``ortho``, ``min_size_m2``,
            ``points_pct``). Unknown keys are ignored.
        should_cancel:
            Asked every few hundred milliseconds while the run blocks here. The
            first True takes the same soft-cancel route as the panel's Cancel
            button, so the tiles the account already paid for go on into the
            review and come back in this call's result. None (the default)
            polls nothing and behaves exactly as before. A callable that raises
            is read as "not cancelled" and is never asked again.
        instance_colors:
            Colour the saved layer one hue per object instead of one hue for
            the whole run, so objects that touch read apart. Off by default,
            and cosmetic: it runs once the objects are written, so it can never
            turn a run that found them into a failure. The result then carries
            ``instance_colors``, plus ``instance_colors_note`` saying why when
            it is False.

        A run stopped this way returns ``cancelled: True`` beside its usual
        keys, so a caller can tell a deliberate stop from a failure.
        """
        from qgis.PyQt.QtCore import QEventLoop, QTimer

        from ..ai_segmentation_dockwidget import Mode

        # One run at a time. Without this, a second call fell through to the
        # worker of the run already in flight, blocked on it, and reported that
        # run's paid instances, credits and layer as its own.
        live = getattr(self, "_auto_worker", None)
        if live is not None and live.isRunning():
            return {
                "_error": (
                    "A zone detection is already running. Poll it with "
                    "auto_detect_status(), or stop it with cancel_auto(), "
                    "before starting another."
                ),
                "busy": True,
            }

        # Ensure the dock exists and auto-mode infrastructure is ready.
        self._ensure_dock_widget()
        if self._tile_manager is None:
            self._setup_auto_mode()

        # Switch dock to Automatic mode so _start_auto_detection can run. The
        # mode and any layer visibility this borrows go back at the end: an
        # agent call must not leave the panel on a mode the person did not pick.
        mode_before = None
        shown_layer_id = ""
        # True once the hard teardown has run on this call's worker. That path
        # leaves no terminal behind to drop the headless flag: the cancelled
        # slot finds a run already torn down and only releases the references.
        hard_stopped = False
        # Setup and validation run INSIDE the try, so a refusal gives back
        # what they borrowed. An unknown raster, an unusable zone or a zone
        # over the free cap returned straight to the caller and left the
        # panel switched to Automatic with a raster re-ticked in the layer
        # tree. The restore only puts back what was changed, so a refusal
        # that borrowed nothing costs nothing.
        try:
            try:
                dock = self.dock_widget
                if dock and dock._mode != Mode.AUTOMATIC:
                    mode_before = dock._mode
                    dock._on_mode_selected(Mode.AUTOMATIC)
            except (RuntimeError, AttributeError):
                pass

            # Resolve and activate the target raster layer.
            if layer_name:
                target_layer, ambiguous = self._resolve_headless_raster(layer_name)
                if ambiguous:
                    return {"_error": (
                        f"More than one raster layer is called '{layer_name}'. "
                        "Pass the layer id instead, so the run reads the one you "
                        "mean.")}
                if target_layer is None:
                    # A caller reading the argument as "name the layer you want
                    # created" passes the output name here and gets told a layer is
                    # missing, which points at the wrong thing. Say what the
                    # argument means, and list what it can actually take.
                    from ...mcp_api import (
                        LAYER_NAME_ARGUMENT_NOTE,
                        not_found_error,
                    )
                    available = sorted(
                        lyr.name()
                        for lyr in QgsProject.instance().mapLayers().values()
                        if isinstance(lyr, QgsRasterLayer)
                    )
                    return not_found_error(
                        "raster layer", layer_name, available,
                        note=LAYER_NAME_ARGUMENT_NOTE,
                    )
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
                            shown_layer_id = target_layer.id()
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
                            f"Raster layer '{layer_name}' exists but could not be selected "
                            "(hidden in the layer tree or filtered out). Make it "
                            "visible and retry.")}

            # Parse and transform zone WKT (layer CRS) to canvas CRS.
            if zone_wkt and zone_wkt.strip():
                geom = QgsGeometry.fromWkt(zone_wkt)
                if geom is None or geom.isEmpty():
                    return {"_error": "Invalid zone WKT"}

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
                        from ...core import telemetry_run_events
                        telemetry_run_events.track_auto_zone_too_large(area_km2=cap_area)
                    except Exception:
                        pass  # nosec B110
                    from .shared import zone_over_free_cap_message
                    return {"_error": zone_over_free_cap_message(cap_area)}
                # Keeps the polygon beside the rectangle, so the cap that measured
                # the polygon and the run that tiles the zone agree on one shape.
                self._store_auto_zone_from_geometry(geom, active_layer)
                try:
                    if self.dock_widget:
                        self.dock_widget.set_auto_zone_state("zone_set")
                except (RuntimeError, AttributeError):
                    pass
            else:
                self._store_auto_zone(None)
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
                    # The caller named the level, so no plugin seed stands for this
                    # run: an earlier interactive seed must not be reported as one.
                    self._auto_detail_seeded = None
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

            # Fetch the server run plan inline (bounded, best-effort, fail-open)
            # BEFORE the run. The interactive path fires this on the debounced
            # prompt-commit; a headless/MCP run has no such commit, so it used to
            # skip the plan and fall back to the neutral cached-policy preset. With
            # the plan in hand the finalize applies the SAME per-prompt review shape
            # (right angles / fill / simplify) and tuned confidence the product
            # uses, so an agent result is refined like the product. Any failure
            # leaves _auto_run_plan None and keeps today's cached-policy path.
            # Stored exactly like _on_auto_run_plan_ready, keyed on object_class
            # (which set_prompt_text put in the box, so _auto_run_ctx["prompt"]
            # matches it and _active_run_plan resolves during finalize).
            self._auto_run_plan = None
            if object_class:
                try:
                    from ...core.activation_manager import get_auth_header
                    plan_auth = get_auth_header()
                    if plan_auth:
                        from ...api.terralab_client import TerraLabClient
                        zone_area_m2, native_mupp = self._auto_run_plan_inputs()
                        plan = TerraLabClient().get_seg_run_plan(
                            object_class, zone_area_m2, native_mupp, auth=plan_auth)
                        if isinstance(plan, dict) and not plan.get("error"):
                            self._auto_run_plan = {"prompt": object_class, "plan": plan}
                except Exception:  # noqa: BLE001 -- planning is best-effort
                    pass  # nosec B110 -- fail-open to the cached-policy preset

            # An explicit cutoff for this run. _start_auto_detection honours a spin
            # value that differs from the client default, so this is the same route
            # a person's choice takes.
            if confidence is not None:
                try:
                    spin = getattr(self.dock_widget, "auto_confidence_spin", None)
                    if spin is not None:
                        spin.setValue(max(0.05, min(0.95, float(confidence))))
                except (RuntimeError, AttributeError, TypeError, ValueError):
                    pass

            # Shape-cleanup overrides for this run only. _auto_review_preset merges
            # them, so the finalize refines with them without anything patching a
            # method at runtime.
            self._auto_review_preset_overrides = (
                dict(refine) if isinstance(refine, dict) and refine else None)

            # Mark this as a headless/MCP run so _on_auto_all_finished skips the
            # interactive review and exports directly (preserving MCP behavior).
            self._auto_headless_run = True
            # Start detection. _last_auto_result is cleared in
            # _launch_auto_worker, past every guard that could refuse.
            self._headless_error = None
            worker_before = self._auto_worker
            self._start_auto_detection()

            if self._auto_worker is None or self._auto_worker is worker_before:
                # Either nothing started, or a guard refused and left the
                # previous run's worker in the slot. Blocking on that one would
                # report its paid results as this call's own.
                reason = getattr(self, "_headless_error", None)
                refused = {
                    "_error": reason or (
                        "Detection did not start. Check the AI Segmentation log: "
                        "missing raster, not signed in, zone too large, or feature disabled."
                    )
                }
                if worker_before is not None and self._auto_worker is worker_before:
                    refused["busy"] = True
                return refused

            # An agent run opens at the product's cutoff with nothing to do
            # here: _start_auto_detection resolves it from the run plan fetched
            # above, honouring an explicit MCP confidence in the spin. This
            # block used to re-resolve it, which is the drift the run-start
            # comment describes.

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

            # The caller is blocked inside loop.exec(), so a Cancel it owns (a
            # Processing dialog, a script) can only be heard by a timer running
            # in this loop. The first True asks for the soft cancel, then the
            # wait continues so the salvaged tiles reach _last_auto_result.
            cancel_state = {"asked": False, "deadline": 0.0}
            poll = self._arm_headless_cancel_poll(loop, should_cancel, cancel_state)
            try:
                QTimer.singleShot(timeout_s * 1000, loop.quit)
                loop.exec()
            finally:
                self._disarm_headless_cancel_poll(poll)

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
                if cancel_state["asked"]:
                    # The user cancelled and the run had not finished winding
                    # down when the grace window closed. The soft path and its
                    # own watchdog own the rest, so the hard teardown must not
                    # run here: it would throw away the tiles already paid for.
                    return {"_error": "Cancelled", "cancelled": True}
                # Timeout path: worker is still running (or silently died).
                self._stop_auto_detection()
                hard_stopped = True
                return {"_error": f"Detection timed out after {timeout_s}s"}

            result = self._last_auto_result
            status = result.get("status")

            if status == "completed":
                # Headless callers (MCP) have no human to click "Export to
                # layer", so commit the post-run review automatically. This
                # keeps the old API contract intact: ai_segment_detect_auto
                # still returns a saved layer name, not None.
                if self._auto_review is not None:
                    exported = self._export_auto_review()
                    if exported and exported[0]:
                        result["layer_name"] = exported[0]
                    else:
                        # Every write target refused. The contract promises a
                        # saved layer name, so reporting "completed" with none
                        # would read as a run whose results are safe somewhere.
                        return {"_error": "The detections could not be written "
                                          "to a layer. Check the project folder "
                                          "is writable and run Finish again."}
                # tiles_processed counts tiles that answered, never a charge: a
                # run is charged for the surface of its zone whatever it finds,
                # and the plugin never learns the final figure, so no key here
                # may claim one.
                return self._color_saved_objects_apart({
                    "instances": result.get("instances", 0),
                    "tiles_processed": result.get("tiles_processed", 0),
                    "layer_name": result.get("layer_name"),
                }, instance_colors)
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
                    out["tiles_processed"] = result.get("tiles_processed", 0)
                return self._color_saved_objects_apart(out, instance_colors)
            if status in ("cancelled", "stalled"):
                out = {"_error": (
                    "Detection stopped responding" if status == "stalled"
                    else "Cancelled")}
                # A deliberate stop is not a failure. The flag lets a caller
                # report what was kept instead of raising on the error text.
                if status == "cancelled":
                    out["cancelled"] = True
                # A stopped run still exports the tiles it got answers for;
                # surface that layer so the caller neither orphans nor blindly
                # retries work that was kept. Same rule as the completed
                # return: tiles, never a charge the plugin cannot verify.
                if result.get("layer_name"):
                    out["layer_name"] = result.get("layer_name")
                    out["instances"] = result.get("instances", 0)
                    out["tiles_processed"] = result.get("tiles_processed", 0)
                return self._color_saved_objects_apart(out, instance_colors)

            return {"_error": f"Unexpected result status: {status}"}
        finally:
            # The flag says "this run has no human", and the wait above can
            # return while the run is still winding down (its own timeout, a
            # cancel inside its grace window). A finalize that then reads it as
            # False takes the interactive route: it opens a review nobody asked
            # for and records no result for the caller. So it is dropped here
            # only when nothing is still running; otherwise the terminal that
            # really ends the run drops it. A hard teardown is the exception:
            # it tore the run down here and left no terminal to drop anything,
            # so its thread winding down in the background must not keep the
            # next interactive run on the headless route, nor an unload waiting
            # on a run that has already ended.
            live = getattr(self, "_auto_worker", None)
            if hard_stopped or live is None or not live.isRunning():
                self._auto_headless_run = False
            self._auto_review_preset_overrides = None
            self._restore_dock_after_headless(mode_before, shown_layer_id)
            # A teardown asked for while this call was blocking was made to
            # wait: unloading under a live nested loop frees objects this very
            # stack holds. The wait is over, so finish it, and last, because
            # nothing above survives it.
            if getattr(self, "_unload_deferred", False):
                self._unload_deferred = False
                try:
                    self.unload()
                except Exception:  # noqa: BLE001 -- teardown must never raise here
                    pass  # nosec B110

    # Runs last, on a result that already holds every object the run paid for.
    # Nothing in here may raise or turn `out` into a failure: a run that found
    # objects and could not colour them is a run that worked, with a note.
    def _color_saved_objects_apart(self, out: dict, wanted: bool) -> dict:
        """Give each object on the saved layer its own colour, and say so.

        A caller that did not ask gets the dict back untouched, so an existing
        caller sees exactly the keys it saw before.
        """
        if not wanted or not out.get("layer_name"):
            return out
        try:
            from ...core.instance_symbology import report_instance_colors
            layer_id = getattr(self, "_auto_export_layer_id", "") or ""
            layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
            return report_instance_colors(out, layer)
        except Exception:  # noqa: BLE001 -- colouring never fails a run
            out["instance_colors"] = False
            out["instance_colors_note"] = (
                "The objects are saved. Colouring them one by one did not work, "
                "so the layer keeps its export style.")
            return out
