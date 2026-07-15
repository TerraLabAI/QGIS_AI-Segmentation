"""Automatic mode flow: mode switching, credits, warmup, detail/grid math.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import (
    QgsProject,
)

from ...core.i18n import tr
from ...core.qt_compat import DistanceMeters
from ...core.telemetry import slot_guard
from .shared import (
    _debounce_timer,
    _WEBMERC_MUPP_Z0,
)


class AutoFlowMixin:
    """Automatic mode flow: mode switching, credits, warmup, detail/grid math."""

    def _on_mode_changed(self, mode) -> None:
        from ..ai_segmentation_dockwidget import Mode
        # Refine-in-Manual handoff: preserve both the held _auto_review and the
        # manual session across the Manual<->Auto hop, so the usual destructive
        # reset below MUST be skipped. The flag stays set for the whole round trip
        # (it is cleared only on Finish / Exit / discard).
        if self._refine_handoff_active:
            if mode == Mode.INTERACTIVE:
                self._ensure_interactive_setup()
                self._enter_manual_refine_session()
            else:  # back to Automatic review
                self._restore_auto_review_after_handoff()
            return
        # Capture the pre-reset state for telemetry (the resets below clear it).
        to_mode = "auto" if mode == Mode.AUTOMATIC else "manual"
        has_mask = getattr(self, "current_mask", None) is not None
        has_frozen_sessions = getattr(self, "_frozen_sessions", None)
        has_unfrozen_display = getattr(self, "_unfrozen_display_polygon", None) is not None
        has_saved_polygons = getattr(self, "saved_polygons", None)
        had_unsaved_manual = bool(has_mask or has_frozen_sessions or has_unfrozen_display or has_saved_polygons)
        auto_step = None
        try:
            auto_step = int(self.dock_widget.auto_steps.currentIndex())
        except (RuntimeError, AttributeError):
            pass
        # Switching modes returns each flow to its pre-Start base state, so no
        # mid-flow state (a drawn zone, a pending review, a running detection)
        # leaks across the switch. The Automatic reset is always safe (its
        # results are transient or already exported); the Manual reset is skipped
        # when there is unsaved manual work, to never silently discard it.
        self._reset_auto_flow_to_start(exit_path="mode_switch")
        self._reset_manual_flow_to_start()
        if mode == Mode.AUTOMATIC and self.dock_widget:
            self._refresh_auto_credits()
        elif mode == Mode.INTERACTIVE and self.dock_widget:
            self._ensure_interactive_setup()
        try:
            from ...core import telemetry
            telemetry.track_mode_switched(
                to_mode=to_mode,
                had_unsaved_manual=had_unsaved_manual,
                auto_step=auto_step,
            )
        except Exception:
            pass  # nosec B110

    def _reset_auto_flow_to_start(self, exit_path: str = "other") -> None:
        """Return the Automatic flow to its pre-Start base: stop any run, drop
        the zone / review / exemplars / canvas visuals, and land on the Start
        step. Idempotent, so it is safe to call when nothing is active.
        ``exit_path`` names the leave path for the abandonment telemetry."""
        self._stop_auto_detection()  # hard teardown of a live worker; no-op if idle
        self._discard_auto_review(exit_path=exit_path)
        self._restore_maptool_after_zone()
        # Drop any pending/stored run plan so it cannot leak across the switch.
        self._auto_run_plan = None
        self._cancel_task("_auto_run_plan_task")
        self._cancel_task("_auto_token_task")
        self._auto_zone = None
        self._auto_zone_polygon = None
        self._clear_auto_canvas()
        if self.dock_widget:
            try:
                self.dock_widget.reset_auto_to_start()
            except (RuntimeError, AttributeError):
                pass

    def _reset_manual_flow_to_start(self) -> None:
        """Return the Manual flow to its pre-Start base when switching modes.

        Skips the reset when there is unsaved manual work (a live mask, frozen
        sessions, an unfrozen polygon, or saved-but-unexported polygons), so a
        mode toggle never silently discards it; the user stops Manual explicitly
        in that case. Exported layers are always untouched."""
        # getattr defaults: the frozen/unfrozen fields only exist once a manual
        # session has started, and a mode switch can fire before that.
        has_unsaved = getattr(self, "current_mask", None) is not None
        has_unsaved = has_unsaved or getattr(self, "_frozen_sessions", None)
        has_unsaved = has_unsaved or getattr(self, "_unfrozen_display_polygon", None) is not None
        has_unsaved = has_unsaved or getattr(self, "saved_polygons", None)
        if has_unsaved:
            return
        try:
            canvas = self.iface.mapCanvas()
            tool = getattr(self, "map_tool", None)
            if tool is not None and canvas.mapTool() == tool:
                canvas.unsetMapTool(tool)
        except (RuntimeError, AttributeError):
            pass
        self._reset_session()
        if self.dock_widget:
            try:
                self.dock_widget.reset_session()
            except (RuntimeError, AttributeError):
                pass

    def _refresh_auto_credits(self) -> None:
        """Fetch usage/credits off the main thread and reflect them in the dock."""
        from ...core.activation_manager import get_auth_header, is_plugin_activated
        if not self.dock_widget or not is_plugin_activated():
            return
        if self._usage_fetch_task is not None and self._usage_fetch_task.is_active():
            return
        auth = get_auth_header()
        if not auth:
            return
        from qgis.core import QgsApplication

        from ...api.terralab_client import TerraLabClient
        from ...workers.generic_request_task import GenericRequestTask
        client = TerraLabClient()
        self._usage_fetch_task = GenericRequestTask(
            tr("Refreshing credits"),
            lambda: client.get_usage(auth=auth),
            hidden=True,
        )
        self._usage_fetch_task.succeeded.connect(self._on_usage_fetched)
        # A usage error (network or server) keeps the dock's last figures rather
        # than blanking the ring; matches the old "empty dict on error" contract.
        # A lapsed subscription (failed payment) is called out clearly instead.
        self._usage_fetch_task.failed.connect(self._on_usage_failed)
        QgsApplication.taskManager().addTask(self._usage_fetch_task)

    def _on_usage_failed(self, message: str = "", code: str = "") -> None:
        """Credits refresh failed. Keep the last-known figures (never blank the
        ring on a transient error). If the server reports the subscription is no
        longer active (a failed renewal payment: the key still authenticates
        locally but every call is rejected), surface a clear one-time notice
        pointing at the account page, so the user is not left with stale credits
        and a mysterious failed run. Throttled to once per session."""
        self._usage_fetch_task = None
        if code == "SUBSCRIPTION_INACTIVE" and not self._billing_warning_shown:
            self._billing_warning_shown = True
            try:
                from qgis.core import Qgis
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("There's a problem with your subscription. Open Settings "
                       "to update your payment method or review your plan."),
                    level=Qgis.MessageLevel.Warning,
                )
            except (RuntimeError, AttributeError):
                pass

    def _on_usage_fetched(self, usage: dict) -> None:
        """Main thread: push fetched usage data into the dock's credit display."""
        self._usage_fetch_task = None
        if not usage or not self.dock_widget:
            return
        # Cache the last-known usage so run telemetry can report credits_before /
        # is_free_tier without an extra fetch on the critical path. Capture the
        # previous free balance first so a drop can emit free_taste_consumed.
        prev_free = (self._last_usage or {}).get("free_detections_remaining")
        self._last_usage = dict(usage)
        is_free = usage.get("is_free_tier", True)
        free_left = usage.get("free_detections_remaining")
        if is_free and prev_free is not None and free_left is not None and free_left < prev_free:
            try:
                from ...core import telemetry
                telemetry.track_free_taste_consumed(
                    remaining=free_left,
                    # Best effort: the refresh follows the run that consumed it.
                    run_id=self._auto_run_id or "",
                )
            except Exception:
                pass  # nosec B110
        if is_free:
            credits = free_left if free_left is not None else 0
            # Lifetime free-taste allowance. Newer servers echo the total; older
            # deploys omit it, so fall back to the known lifetime cap (300) for
            # the ring gauge. Keep in sync with the server cap.
            total = int(usage.get("free_detections_total") or 300)
        else:
            used = usage.get("images_used", 0) or 0
            limit = usage.get("images_limit", 0) or 0
            credits = max(0, limit - used)
            total = limit or None
        reset_date = usage.get("period_end") or ""
        self.dock_widget.set_auto_credits(credits, reset_date,
                                          is_subscriber=not is_free,
                                          total=total)
        # The plan gates the detail slider (free-run cap): re-derive the cap
        # now that the plan is known, so a subscriber's slider uncaps (and a
        # free user's caps) without waiting for the next slider move. Idle
        # zone-set state only; a live run or review owns the cost label.
        if (self._auto_zone is not None and self._auto_worker is None and self._auto_review is None):
            self._update_credit_estimate()

    def _auto_credit_snapshot(self) -> tuple[int | None, bool]:
        """Return (credits_before, is_free_tier) from the last fetched usage.

        Best-effort for telemetry: (None, True) when usage was never fetched."""
        # credit_gate.credit_snapshot owns the usage -> (credits, is_free) rule.
        from ...core.credit_gate import credit_snapshot

        return credit_snapshot(self._last_usage or {})

    def _auto_zone_area_km2(self) -> float:
        """Geodesic area (km2) of the active run's clip polygon; 0.0 if unknown."""
        try:
            from qgis.core import QgsCoordinateReferenceSystem, QgsDistanceArea
            poly = getattr(self, "_auto_clip_polygon", None)
            if poly is None or poly.isEmpty():
                return 0.0
            da = QgsDistanceArea()
            ctx = QgsProject.instance().transformContext()
            crs = (QgsCoordinateReferenceSystem(self._auto_crs_authid)
                   if self._auto_crs_authid else None)
            if crs is not None and crs.isValid():
                da.setSourceCrs(crs, ctx)
            da.setEllipsoid(QgsProject.instance().ellipsoid() or "WGS84")
            return max(0.0, da.measureArea(poly) / 1_000_000.0)
        except Exception:
            return 0.0

    def _auto_duration_ms(self) -> int:
        """Elapsed ms since the detection phase started (0 if not started)."""
        try:
            import time as _time
            if not self._auto_detect_t0:
                return 0
            return int((_time.monotonic() - self._auto_detect_t0) * 1000)
        except Exception:
            return 0

    @staticmethod
    def _classify_auto_error(msg: str) -> str:
        """Map an auto-detection error message to a telemetry error_class enum:
        NETWORK / AUTH / CREDITS_EXHAUSTED / SERVER / CANCELLED / TIMEOUT / UNKNOWN."""
        low = (msg or "").lower()
        if "credit" in low or "quota" in low or "402" in low:
            return "CREDITS_EXHAUSTED"
        if "auth" in low or "401" in low or "403" in low or "sign in" in low:
            return "AUTH"
        if "timeout" in low or "timed out" in low:
            return "TIMEOUT"
        if "cancel" in low:
            return "CANCELLED"
        if any(t in low for t in ("network", "connection", "connect", "ssl", "dns",
                                  "unreachable", "offline")):
            return "NETWORK"
        if any(t in low for t in ("server", "500", "502", "503", "504", "bad gateway")):
            return "SERVER"
        return "UNKNOWN"

    def _track_manual_run_failed(self) -> None:
        """Emit an unsampled segmentation_run(success=False) on a manual predict error."""
        try:
            import time as _time
            from ...core import telemetry
            start_ts = getattr(self, "_segmentation_start_ts", None)
            duration_ms = int((_time.time() - start_ts) * 1000) if start_ts else None
            telemetry.track_segmentation_run(success=False, duration_ms=duration_ms)
        except Exception:
            pass  # nosec B110

    # Minimum gap between cold-start pings: one warmup keeps the backend hot
    # well past this, so re-firing on every step/zone event is wasteful.
    _WARMUP_MIN_INTERVAL_S = 30.0

    def _maybe_warmup_auto(self) -> None:
        """Best-effort cloud cold-start ping, off the GUI thread.

        Fired at each rising-intent step of the Automatic flow so the idle
        backend overlaps the user's setup and is warm by the time they hit
        Detect: entering the draw-zone step (Start clicked), the zone drawn,
        the prompt committed, and a detail-slider move. The prompt-commit
        trigger is the universal one (every run passes through it); the others
        buy extra runway when they fire first.
        Debounced to at most once per ~30s and gated on cloud access (an
        activated key); every failure is silent. Never blocks the main thread.
        """
        import time

        from ...core.activation_manager import get_auth_header, is_plugin_activated

        # Only warmup for users who can actually use the cloud Automatic flow.
        if not is_plugin_activated():
            return
        # A run already keeps the backend hot (the worker hammers it); skip
        # redundant pings while one is in flight.
        if self._auto_worker is not None:
            return
        auth = get_auth_header()
        if not auth:
            return

        now = time.monotonic()
        if now - self._last_warmup_monotonic < self._WARMUP_MIN_INTERVAL_S:
            return
        # A ping already in flight covers the cold start; don't pile on.
        if self._warmup_task is not None and self._warmup_task.is_active():
            return

        self._last_warmup_monotonic = now
        try:
            from qgis.core import QgsApplication

            from ...api.terralab_client import TerraLabClient
            from ...workers.generic_request_task import GenericRequestTask
            client = TerraLabClient()
            self._warmup_task = GenericRequestTask(
                tr("Warming up AI Segmentation"),
                lambda: client.warmup(auth=auth),
                hidden=True,
            )
            self._warmup_task.succeeded.connect(self._on_warmup_finished)
            self._warmup_task.failed.connect(lambda *_a: self._on_warmup_finished(False))
            QgsApplication.taskManager().addTask(self._warmup_task)
        except Exception:
            # Warmup is purely cosmetic; never let it disrupt the flow.
            self._warmup_task = None

    def _on_warmup_finished(self, _ok: object = None) -> None:
        """Main thread: release the finished warmup task. Result is ignored
        (warmup is best-effort; the real detection reports its own outcome)."""
        self._warmup_task = None

    @slot_guard(stage="segment", user_message=tr(
        "Something went wrong starting the detection. Please try again."))
    def _on_auto_detect_requested(self) -> None:
        # Commit-time guard rail: both entry points (Detect click and Enter)
        # funnel here, so this is the single gate where an off-rails prompt
        # blocks the run and shows its guidance under the prompt box.
        dock = self.dock_widget
        if dock is not None and not dock.confirm_prompt_for_detect():
            return
        # First committed Detect seals Terms + Privacy consent (the checkbox
        # sits right above the button; Detect stays disabled until it is
        # ticked, so reaching this line means consent was given).
        if dock is not None:
            try:
                dock.seal_tos_consent()
            except (RuntimeError, AttributeError):
                pass
        # Every Detect is a fresh run. The resume/re-detect-the-same-zone flow
        # was removed (it confused more than it helped): a stopped run keeps its
        # partial results in the review instead.
        self._start_auto_detection()

    def _on_auto_library_clicked(self) -> None:
        """Open the segment library gallery; drop the chosen English token into
        the prompt box. The token is the literal cloud-model prompt (labels are
        localized, tokens are not)."""
        try:
            from ..dialogs.segment_library_dialog import SegmentLibraryDialog
        except Exception as err:  # noqa: BLE001
            from qgis.core import Qgis
            from ...core.logging_utils import log
            log(f"Segment library unavailable: {err}", Qgis.MessageLevel.Warning)
            return
        from ...core.presets import segment_history
        # While a run (or review) is in flight the library opens view-only: the
        # user can still browse, but picking / re-running is disabled (mirrors AI
        # Edit). A picked token can never overwrite the locked in-run prompt.
        run_active = getattr(self.dock_widget, "_auto_run_active", False)
        review_active = getattr(self.dock_widget, "_auto_review_active", False)
        view_only = bool(run_active or review_active)
        dlg = SegmentLibraryDialog(
            self.dock_widget, recent=segment_history.get_recent(), plugin=self,
            view_only=view_only)
        if dlg.exec() and not view_only:
            token = dlg.get_selected_prompt()
            if token:
                self.dock_widget.set_prompt_text(token)

    def _on_zone_draw_requested(self) -> None:
        """Handle the dock's 'Draw zone' button (wired by plan #76)."""
        self._activate_zone_drawing()

    def _on_auto_detail_changed(self, _value: int) -> None:
        """Detail slider moved: the grid, overlay and credit estimate change.

        This fires only on a genuine USER move: programmatic seeds go through
        set_auto_detail_value / set_auto_detail_max, which block the slider's
        signal. So a real move here means the user is overriding the auto pick;
        latch that so the object-aware re-seed stops fighting them for this zone.
        The prompt the override was made for is recorded with it: a manual
        tweak stands for THAT object only, a different prompt re-seeds.
        """
        self._auto_detail_user_locked = True
        self._auto_detail_lock_prompt = self._resolved_auto_object_class().lower()
        # Debounce the cost/grid recompute: each tick runs compute_grid + up to
        # MAX_TILES polygon-clip GEOS calls on the GUI thread, so dragging the
        # slider across many levels would stutter. Coalesce to ~130ms of
        # inactivity (other callers still update immediately).
        self._schedule_credit_estimate()
        # Active setup = the user is about to Detect: warm the backend (debounced
        # to ~30s, gated on the Automatic flow + no active run). This is the
        # activity-driven keep-warm: pings only while the user is actually working,
        # so the backend goes idle on its own once they stop - a dock left open
        # and idle (or a Manual-mode user) never holds the backend up.
        self._maybe_warmup_auto()
        self._schedule_detail_telemetry("user")

    def _schedule_credit_estimate(self) -> None:
        """Debounce _update_credit_estimate for rapid slider moves (~130ms of
        inactivity), so a drag recomputes the grid/cost once it settles instead
        of on every intermediate tick."""
        _debounce_timer(self, "_credit_est_timer", self.dock_widget, 130,
                        self._update_credit_estimate)

    def _schedule_detail_telemetry(self, source: str) -> None:
        """Debounce detail_changed telemetry to ~1s of slider inactivity."""
        self._detail_tel_source = source
        _debounce_timer(self, "_detail_tel_timer", self.dock_widget, 1000,
                        self._emit_detail_telemetry)

    def _emit_detail_telemetry(self) -> None:
        try:
            from ...core import telemetry
            tiles = getattr(self, "_auto_est_tiles", -1)
            telemetry.track_detail_changed(
                detail=self._get_auto_detail_level(),
                tiles=tiles if tiles is not None else -1,
                source=getattr(self, "_detail_tel_source", "user"),
            )
        except Exception:
            pass  # nosec B110

    def _on_auto_layer_combo_changed(self, _layer) -> None:
        """Raster picked (or gone) in the Automatic combo: the cost, the
        slider visibility and the grid preview all depend on the layer."""
        self._update_credit_estimate()

    def _on_auto_step_changed(self, index: int) -> None:
        """Arm the zone drawing tool whenever the zone step opens bare.

        Drawing is the only way to pick the detection area (there is no Full
        image button), so the cross-hair tool activates by itself instead of
        waiting for a button click. Leaving the step disarms the tool unless
        a zone was actually drawn.
        """
        if self._auto_worker is not None:
            return
        if index == 1:
            # Entering the draw-zone step (Start clicked, or re-draw): warm the
            # backend now so the cold start overlaps the user picking a zone.
            self._maybe_warmup_auto()
            # Re-entering the zone step means re-choosing the zone: drop any
            # previous rectangle so the next drag starts clean.
            if self._auto_zone is not None:
                self._on_zone_cleared()
            self._activate_zone_drawing()
        else:
            # Leaving the draw step without (re)drawing: disarm the zone tool and
            # hand the pan tool back rather than leaving a bare cursor.
            self._restore_maptool_after_zone()

    # ---- Pro Auto mode: detection worker (plan #78) -------------------------

    def _get_auto_detail_level(self) -> int:
        """Current detail slider value (grid side n, 1-7). Defaults to 1."""
        if self.dock_widget is None:
            return 1
        try:
            return max(1, int(self.dock_widget.auto_detail_slider.value()))
        except (RuntimeError, AttributeError):
            return 1

    def _online_native_mupp(self, layer) -> float:
        """Finest native resolution (layer-CRS units per pixel) of an online tiled
        source, so the detail slider never renders finer than what the source
        actually serves (no upsampled blurry tiles, no wasted credits). Returns
        0.0 (no clamp) when the native resolution is genuinely unknown, which
        preserves the prior behaviour for non-tiled sources and, by failing OPEN,
        can never over-restrict legitimate detail or make the plugin crash.

        Covered:
          - XYZ basemaps (Google, OSM, ...), WITH or WITHOUT an explicit zmax.
          - WMTS (its tile matrix set defines the finest level's resolution).
          - Any raster provider that reports native resolutions, any CRS.
        Returns 0.0 for non-tiled WMS / WCS / anything with no fixed native grid.
        """
        if layer is None:
            return 0.0
        # Primary path: ask the provider for its native resolutions. QGIS builds
        # this from the tile pyramid for XYZ (derived from zmin/zmax, so it works
        # even when the URI carries no explicit zmax) and from the tile matrix set
        # for WMTS. The values are already in the provider/layer CRS units per
        # pixel - exactly what `_grid_for_detail` compares against - so no CRS math
        # is needed here (a Web Mercator zoom's resolution is latitude-independent
        # in projected units, and any other matrix set reports its own CRS units).
        # The finest level = the smallest value; clamping the render to it only
        # blocks upsampling, never real detail.
        try:
            provider = layer.dataProvider()
            if provider is not None:
                # nativeResolutions(): QGIS >= 3.8 (our floor is 3.22).
                native = provider.nativeResolutions()
                finest = self._finest_native_resolution(native)
                if finest > 0:
                    return finest
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass  # provider does not expose it; fall through to the URI fallback

        # Fallback: derive the deepest-zoom resolution from an XYZ URI's zmax using
        # the Web Mercator slippy-map pyramid. Only valid when the tiling is Web
        # Mercator (EPSG:3857); for any other CRS we cannot assume that formula, so
        # we fail open (0.0) rather than risk an over- or under-clamp.
        try:
            if self._layer_is_web_mercator(layer):
                return self._xyz_zmax_mupp_from_source(layer.source() or "")
        except (RuntimeError, AttributeError, ValueError, TypeError):
            pass
        return 0.0

    @staticmethod
    def _finest_native_resolution(resolutions) -> float:
        """Finest (smallest positive) resolution, in layer-CRS units per pixel,
        from a provider's ``nativeResolutions()`` list. Skips non-numeric or
        non-positive entries individually and returns 0.0 for an empty/invalid
        list (fail open: no clamp)."""
        best = 0.0
        try:
            items = list(resolutions or [])
        except TypeError:
            return 0.0
        for r in items:
            try:
                v = float(r)
            except (TypeError, ValueError):
                continue
            if v > 0 and (best == 0.0 or v < best):
                best = v
        return best

    @staticmethod
    def _layer_is_web_mercator(layer) -> bool:
        """True when the layer's CRS is Web Mercator, so the slippy-map zoom
        formula (``_WEBMERC_MUPP_Z0 / 2**z``) yields correct EPSG:3857 units/px.
        Covers the common historical aliases for spherical Web Mercator."""
        try:
            authid = (layer.crs().authid() or "").upper()
        except (RuntimeError, AttributeError):
            return False
        return authid in (
            "EPSG:3857", "EPSG:900913", "EPSG:102100", "EPSG:102113",
        )

    @staticmethod
    def _xyz_zmax_mupp_from_source(source: str) -> float:
        """Deepest-zoom resolution (EPSG:3857 units per pixel) parsed from an XYZ
        URI's ``zmax``, via the Web Mercator tile pyramid. Returns 0.0 when there
        is no usable zmax: a plain WMS/WMTS URI has none, and an XYZ without an
        explicit zmax is left unclamped on purpose (QGIS's default max zoom is
        provider-internal and version-dependent, so guessing one would over- or
        under-clamp; the runtime nativeResolutions() path already covers it).
        Never raises."""
        try:
            import re  # noqa: PLC0415 - cheap, cached; keeps the top imports lean
            # QGIS raster URIs are '&'-delimited key=value pairs; the tile 'url'
            # value carries its own ampersands URL-encoded (%26), so a top-level
            # '&'/'?' split never cuts through it. Match zmax as a standalone key.
            m = re.search(r"(?:^|[?&])zmax=(\d+)", source or "")
            if not m:
                return 0.0
            zmax = int(m.group(1))
            if zmax <= 0:
                return 0.0
            return _WEBMERC_MUPP_Z0 / float(2 ** zmax)
        except (ValueError, TypeError):
            return 0.0

    def _grid_for_detail(self, layer, zone_in_layer, detail_n: int):
        """Pixel grid for one detail level. Single source of truth for the
        detail-slider math, shared by `_compute_auto_grid` (the real run),
        `_max_useful_detail` (the slider cap) and the m/px hint.

        The zone's longer side is rendered as exactly ``detail_n`` tiles
        (snap formula: TILE_SIZE + (n-1) * stride pixels). Local rasters AND
        online Web Mercator XYZ basemaps are clamped near their native
        resolution: the render may oversample up to NATIVE_OVERSAMPLE_MAX
        (2x linear) past it, because the cloud model's fixed processing window
        means a finer grid still enlarges small objects in model space, but
        never further (pure interpolation, credits for nothing).

        ``zone_in_layer`` is the zone already reprojected to the layer CRS.

        Returns ``(pixel_w, pixel_h, mupp, tile_count)`` where ``mupp`` is the
        rendered ground resolution in layer-CRS units per pixel (identical on
        both axes) and ``tile_count`` is the credit cost (``-1`` when the grid
        exceeds MAX_TILES). Returns None when the zone has no extent or the
        layer cannot be read.
        """
        from ...core.tile_manager import (
            NATIVE_OVERSAMPLE_MAX,
            OVERLAP_FRACTION,
            TILE_SIZE,
        )

        longer_side = max(zone_in_layer.width(), zone_in_layer.height())
        if longer_side <= 0:
            return None
        try:
            layer_w = layer.width()
            layer_h = layer.height()
            ext = layer.extent()
        except (RuntimeError, AttributeError):
            return None

        use_online = layer_w <= 0 or layer_h <= 0 or self._is_online_provider(layer)
        # Same integer stride as TileManager. compute_grid() emits one tile at
        # offset 0 then one per full stride, so a span of TILE_SIZE + k * stride
        # pixels yields exactly k + 1 tiles per axis.
        stride = int(TILE_SIZE * (1.0 - OVERLAP_FRACTION))
        target_px = TILE_SIZE + (max(1, detail_n) - 1) * stride
        mupp = longer_side / target_px
        # Both clamps allow up to NATIVE_OVERSAMPLE_MAX past the source's
        # native resolution: upsampling adds no pixels, but a finer grid
        # enlarges small objects in the cloud model's fixed processing window,
        # so trees/cars on a coarse source gain real recall. Past 2x it is pure
        # interpolation, so the clamp holds there.
        if not use_online and ext.width() > 0 and ext.height() > 0:
            native_mupp = max(ext.width() / layer_w, ext.height() / layer_h)
            mupp = max(mupp, native_mupp / NATIVE_OVERSAMPLE_MAX)
        elif use_online:
            # Online XYZ basemap: clamp near its deepest-zoom native resolution
            # so a finer detail level cannot burn unbounded credits on ever
            # blurrier upsamples. No-op for WMS / non-Mercator.
            online_mupp = self._online_native_mupp(layer)
            if online_mupp > 0:
                mupp = max(mupp, online_mupp / NATIVE_OVERSAMPLE_MAX)

        pixel_w = max(1, int(zone_in_layer.width() / mupp))
        pixel_h = max(1, int(zone_in_layer.height() / mupp))
        # Do NOT snap up to clean tile boundaries: snapping grew the grid past
        # the drawn zone, so the preview and the tiles spilled into whatever lay
        # outside the selection (e.g. desert). Tile the zone exactly instead;
        # compute_grid's last row/column is a partial edge tile that stays fully
        # inside the zone, and the 20% overlap means the adjacent full tile
        # already covers most of that strip.
        tile_count = self._tile_manager.estimate_credits(pixel_w, pixel_h)
        return pixel_w, pixel_h, mupp, tile_count

    def _max_useful_detail(self, layer, zone_in_layer) -> int:
        """Largest detail level (1-MAX_DETAIL_LEVEL) that still changes the grid.

        Two regimes bound the slider, and crossing either makes higher levels
        inert (the cursor moves with no visible effect):
          - the MAX_TILES cap (a finer grid would exceed it), and
          - the native-resolution clamp: on a local raster, past some level
            the render is pinned to native pixels, so a higher level repeats
            the exact same mupp and the same tiles.
        Returns the largest n whose grid fits MAX_TILES and whose mupp is
        still strictly finer than n-1's. Always >= 1.
        """
        from ...core.tile_manager import MAX_DETAIL_LEVEL, MAX_TILES

        best = 1
        prev_mupp = None
        for n in range(1, MAX_DETAIL_LEVEL + 1):
            sized = self._grid_for_detail(layer, zone_in_layer, n)
            if sized is None:
                break
            _pw, _ph, mupp, tiles = sized
            if tiles == -1 or tiles > MAX_TILES:
                break  # grids only grow with n; nothing finer will fit
            if prev_mupp is not None and mupp >= prev_mupp:
                break  # clamped to native: this level renders no finer
            best = n
            prev_mupp = mupp
        return best

    def _free_run_tile_cap(self) -> int | None:
        """Per-run tile (credit) cap for free-tier users; None for subscribers.

        A single free run may only cost a fraction of the lifetime free
        allowance (server policy `seed.free_run_fraction`), so one Detect can
        never drain the trial: the remaining runs are where the product proves
        itself. The cap gates DETECT, never the slider: the user drags through
        the full Pro range and sees an upgrade message past the cap (the locked
        capability stays visible). The auto seeds also respect it, so the
        suggested default never lands in the gated range. Before the usage
        fetch lands the user is treated as free (the safe default); the
        estimate re-runs when the fetch arrives, so a subscriber ungates the
        moment their plan is known.
        """
        from ...core.credit_gate import free_run_tile_cap
        from ...core.detection_policy import free_run_fraction

        usage = self._last_usage or {}
        if not usage.get("is_free_tier", True):
            return None
        # credit_gate.free_run_tile_cap owns the cap arithmetic (the round(),
        # the older-server total fallback, and the >= 1 floor). The subscriber
        # None stays here: it is a tier decision, not cap arithmetic.
        return free_run_tile_cap(usage.get("free_detections_total"), free_run_fraction())

    def _seed_tile_cap_for_plan(self) -> int:
        """The hard seed tile cap, tightened to the free-run cap on free tier,
        so the auto-picked default never proposes a level the plan gates."""
        from ...core.detection_policy import seed_tile_cap

        cap = seed_tile_cap()
        free_cap = self._free_run_tile_cap()
        return cap if free_cap is None else min(cap, free_cap)

    def _default_detail_for_zone(self, layer, zone_in_layer) -> int:
        """Cheapest detail level whose ground resolution reaches the seed target.

        Walks detail levels coarse -> fine and returns the first level reaching
        the seed resolution within the soft tile budget. Past that budget it
        returns the cheapest level still inside the sweet-spot band (the credit
        cost is displayed live and Detect is the user's confirmation), and never
        spends past the hard seed cap. Falls back to the cheapest in-budget
        level when even the slider maximum stays coarser. Always >= 1, so a
        fresh zone never sits at a too-coarse 1x1 grid when a finer level is the
        better default. The object-aware seed refines this per prompt.
        """
        from ...core.detection_policy import (
            soft_tile_budget,
            sweet_spot_max_mupp,
            zone_seed_mupp,
        )

        seed_mupp = zone_seed_mupp()
        tile_cap = self._seed_tile_cap_for_plan()
        soft_budget = soft_tile_budget()
        sweet_max = sweet_spot_max_mupp()
        best_within_budget = 1
        for n in range(1, self._max_useful_detail(layer, zone_in_layer) + 1):
            sized = self._grid_for_detail(layer, zone_in_layer, n)
            if sized is None:
                break
            _pw, _ph, mupp, tiles = sized
            if tiles == -1:
                break
            if tiles > tile_cap:
                break  # the default never spends past the hard seed cap
            ground_mupp = self._mupp_to_meters(layer, zone_in_layer, mupp)
            if tiles > soft_budget:
                if 0 < ground_mupp <= sweet_max:
                    return n  # cheapest ADEQUATE level beyond the soft budget
                continue
            best_within_budget = n
            if 0 < ground_mupp <= seed_mupp:
                return n  # cheapest level reaching the seed target
        return best_within_budget

    def _object_detail_profile(self, object_class: str) -> tuple[float, float]:
        """(typical ground size m, target ground resolution m/px) for the
        object named by ``object_class``.

        The per-object targets are provided by the server configuration;
        without it a generic object at the default seed resolution applies. See
        ``core.detection_policy.object_profile``.
        """
        from ...core.detection_policy import object_profile

        return object_profile(object_class)

    def _auto_detail_for_object(self, layer, zone_in_layer, object_class) -> int:
        """Cheapest detail level whose ground resolution suits the OBJECT.

        The object maps to a target ground resolution (`_object_detail_profile`)
        so distinct objects land on distinctly different levels. Walks detail
        levels coarse -> fine and returns the cheapest reaching the target, past
        the soft tile budget when needed (the credit cost is displayed live and
        Detect is the user's confirmation) but never past the hard seed cap.

        Fallbacks, in order, when no level reaches the target: the cheapest
        level where the object still renders at the object minimum pixel size
        (resolvable, if not ideal), else the finest level within the soft tile
        budget; on any layer read error, the resolution-only zone default.
        Always >= 1.
        """
        try:
            obj_m, target_mupp = self._object_detail_profile(object_class)
            return self._auto_detail_for_target(
                layer, zone_in_layer, obj_m, target_mupp)
        except (RuntimeError, AttributeError, ValueError):
            return self._default_detail_for_zone(layer, zone_in_layer)

    def _auto_detail_for_target(
        self, layer, zone_in_layer, obj_m: float, target_mupp: float
    ) -> int:
        """Cheapest detail level whose ground resolution reaches ``target_mupp``.

        The shared coarse -> fine walk behind `_auto_detail_for_object` (blob
        tier target) and the async run-plan re-seed (server target), so the two
        paths pick levels identically. ``obj_m`` is the object's typical ground
        size, used only for the resolvable fallback. Fallbacks, in order, when
        no level reaches the target: the cheapest level where the object still
        renders at the object minimum pixel size, else the finest level within
        the soft tile budget. Always >= 1.
        """
        from ...core.detection_policy import (
            object_min_px,
            soft_tile_budget,
        )

        cap = self._max_useful_detail(layer, zone_in_layer)
        min_px = object_min_px()
        tile_cap = self._seed_tile_cap_for_plan()
        soft_budget = soft_tile_budget()
        resolvable_n = None
        finest_in_budget = 1
        for n in range(1, cap + 1):
            sized = self._grid_for_detail(layer, zone_in_layer, n)
            if sized is None:
                break
            _pw, _ph, mupp, tiles = sized
            if tiles == -1:
                break
            if tiles > tile_cap:
                break  # tiles grow with n: the seed never spends past the cap
            ground_mupp = self._mupp_to_meters(layer, zone_in_layer, mupp)
            if ground_mupp <= 0:
                continue
            if ground_mupp <= target_mupp:
                return n
            if tiles <= soft_budget:
                finest_in_budget = n
            if resolvable_n is None and obj_m / ground_mupp >= min_px:
                resolvable_n = n
        if resolvable_n is not None:
            return resolvable_n
        return max(1, finest_in_budget)

    def _current_auto_object_class(self) -> str:
        """The object class currently entered in the Automatic prompt box."""
        if self.dock_widget is None:
            return ""
        try:
            return self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            return ""

    def _resolve_object_token(self, raw: str) -> str:
        """English cloud-model token for a possibly-localized prompt.

        The offline lexicon resolves a localized object word to its English
        token synchronously, so the detail seed and every prompt-keyed policy
        lookup key on the SAME token the run will send ("jardin" seeds like
        "garden"). An English or lexicon-missing word passes through unchanged;
        a miss beyond the offline lexicon is resolved asynchronously by the
        server fallback. Cached per prompt so the debounced commit never
        re-queries the catalogue on every keystroke settle.
        """
        raw = (raw or "").strip()
        if not raw:
            return ""
        cache = getattr(self, "_auto_token_cache", None)
        if cache is None:
            cache = {}
            self._auto_token_cache = cache
        key = raw.lower()
        if key in cache:
            return cache[key]
        try:
            from ..dock.prompt_guard import resolve_object_token
            token = resolve_object_token(raw)
        except Exception:  # noqa: BLE001 -- resolution is best-effort
            token = raw
        cache[key] = token
        return token

    def _resolved_auto_object_class(self) -> str:
        """The current prompt resolved to its English token. The box keeps
        showing the user's own words; the seed and policy lookups key on this
        so a localized prompt behaves like its English equivalent."""
        return self._resolve_object_token(self._current_auto_object_class())

    def _apply_default_detail(self, zone_rect) -> None:
        """Seed the detail slider with a good default for a freshly drawn zone.

        A freshly drawn zone is a fresh start, so the manual-override lock is
        cleared: the object-aware re-seed may drive the slider again until the
        user next moves it themselves. When an object class is already entered,
        seed with the object-aware pick; otherwise fall back to the
        resolution-only default.
        """
        # New zone = fresh default: let the auto picker own the slider again.
        self._auto_detail_user_locked = False
        self._auto_detail_lock_prompt = ""
        if not self.dock_widget:
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            zone_in_layer = self._reproject_zone_to_layer_crs(zone_rect, layer)
            object_class = self._resolved_auto_object_class()
            if object_class:
                detail = self._auto_detail_for_object(
                    layer, zone_in_layer, object_class)
            else:
                detail = self._default_detail_for_zone(layer, zone_in_layer)
            self.dock_widget.set_auto_detail_value(detail)
        except (RuntimeError, AttributeError):
            pass

    def _reseed_auto_detail_for_object(self, object_class: str = "") -> None:
        """Prompt settled (debounced): re-seed the detail default from the blob
        NOW (must land instantly), then fire the async server run-plan fetch.

        The synchronous blob seed is the source of truth the moment the prompt
        settles; the run plan, when it lands and still matches this prompt,
        refines it (see _on_auto_run_plan_ready).

        A localized prompt is resolved to its English token first (offline
        lexicon), so the object tiers match ("jardin" seeds like "garden"). The
        box keeps the user's own words; only the lookups key on the token."""
        raw = (object_class or "").strip()
        token = self._resolve_object_token(raw)
        self._reseed_auto_detail_from_blob(token)
        self._fetch_auto_run_plan(token)
        # A localized word the offline lexicon does not cover (a rare language)
        # cannot seed the object tiers yet: ask the server for its English token
        # and re-seed when it lands, if this prompt is still the one shown.
        if raw and token == raw:
            self._fetch_auto_token(raw)
        # A committed prompt is the highest-intent, most universal pre-Detect
        # signal: it is the ONE step every run passes through (the zone-draw and
        # slider triggers miss a user who lands on the prompt step with a zone
        # already set and runs without touching the slider). Warm here so the
        # backend is spinning up while they read the estimate and reach Detect.
        # Self-limited: debounced ~30s, no-op mid-run and when already warm.
        self._maybe_warmup_auto()

    def _reseed_auto_detail_from_blob(self, object_class: str = "") -> None:
        """Re-pick the detail level when the object class changes (debounced).

        Respects a manual override for the SAME prompt: once the user has
        moved the slider, their value stands for the object it was tuned for.
        A different prompt is a new sizing problem, so it releases the
        override and re-seeds (the user can still re-adjust afterwards).
        Requires a drawn zone and a non-empty object class. Sets the slider
        programmatically (signal free via set_auto_detail_value's
        blockSignals), so it never trips the user-lock itself.
        """
        if not self.dock_widget or self._auto_zone is None:
            return
        # Never fight a run/review or override while one is in flight.
        if self._auto_worker is not None or self._auto_review is not None:
            return
        object_class = (object_class or "").strip()
        if not object_class:
            return
        if self._auto_detail_user_locked:
            locked_for = getattr(self, "_auto_detail_lock_prompt", "")
            if object_class.lower() == locked_for:
                return
            self._auto_detail_user_locked = False
            self._auto_detail_lock_prompt = ""
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            zone_in_layer = self._reproject_zone_to_layer_crs(self._auto_zone, layer)
            detail = self._auto_detail_for_object(layer, zone_in_layer, object_class)
            self.dock_widget.set_auto_detail_value(detail)
            self._update_credit_estimate()
        except (RuntimeError, AttributeError):
            pass

    # ---- Async server run plan (Phase 2) ----------------------------------

    def _active_run_plan(self, prompt: str) -> dict | None:
        """The stored run plan IF it was fetched for ``prompt`` (case/space
        insensitive), else None. The single gate every apply site goes through,
        so a stale plan for a different prompt never leaks into a run/review."""
        rp = getattr(self, "_auto_run_plan", None)
        if not isinstance(rp, dict):
            return None
        if (rp.get("prompt") or "").strip().lower() != (prompt or "").strip().lower():
            return None
        plan = rp.get("plan")
        return plan if isinstance(plan, dict) else None

    def _fetch_auto_run_plan(self, prompt: str) -> None:
        """Fire-and-forget: fetch the server run plan for the committed prompt.

        Off the GUI thread via a hidden task; never blocks, fails open (the
        blob/generic path stands on any error or timeout). A new commit clears
        the previous plan so a stale one can never apply between commit and
        reply."""
        prompt = (prompt or "").strip()
        # Every commit invalidates the previous plan (it was for the old prompt).
        self._auto_run_plan = None
        if not prompt or not self.dock_widget:
            return
        # Never fetch mid-run/review: the prompt is locked then and a late
        # apply must not fight an in-flight run.
        if self._auto_worker is not None or self._auto_review is not None:
            return
        from ...core.activation_manager import get_auth_header, is_plugin_activated
        if not is_plugin_activated():
            return
        auth = get_auth_header()
        if not auth:
            return
        # Only one in flight; a newer commit supersedes the older request.
        self._cancel_task("_auto_run_plan_task")
        zone_area_m2, native_mupp = self._auto_run_plan_inputs()
        try:
            from qgis.core import QgsApplication

            from ...api.terralab_client import TerraLabClient
            from ...workers.generic_request_task import GenericRequestTask
            client = TerraLabClient()
            task = GenericRequestTask(
                tr("Planning AI Segmentation run"),
                lambda: client.get_seg_run_plan(
                    prompt, zone_area_m2, native_mupp, auth=auth),
                hidden=True,
            )
            task.succeeded.connect(
                lambda plan, p=prompt: self._on_auto_run_plan_ready(p, plan))
            task.failed.connect(lambda *_a: self._on_auto_run_plan_failed())
            self._auto_run_plan_task = task
            QgsApplication.taskManager().addTask(task)
        except Exception:  # noqa: BLE001 -- planning is best-effort
            self._auto_run_plan_task = None  # nosec B110

    def _auto_run_plan_inputs(self) -> tuple[float | None, float | None]:
        """Best-effort (zone_area_m2, native_mupp) for the run-plan request.

        Either may be None (both optional server-side). One cheap geodesic
        measurement each, on the debounced commit; never raises."""
        zone_area_m2: float | None = None
        native_mupp: float | None = None
        try:
            layer = self._get_active_raster_layer()
        except (RuntimeError, AttributeError):
            layer = None
        if layer is None:
            return zone_area_m2, native_mupp
        zone_in_layer = None
        if self._auto_zone is not None:
            try:
                zone_in_layer = self._reproject_zone_to_layer_crs(self._auto_zone, layer)
            except (RuntimeError, AttributeError):
                zone_in_layer = None
        if zone_in_layer is not None:
            try:
                from qgis.core import QgsDistanceArea, QgsGeometry
                da = QgsDistanceArea()
                da.setSourceCrs(layer.crs(), QgsProject.instance().transformContext())
                da.setEllipsoid(QgsProject.instance().ellipsoid() or "WGS84")
                area = da.measureArea(QgsGeometry.fromRect(zone_in_layer))
                if area and area > 0:
                    zone_area_m2 = float(area)
            except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
                zone_area_m2 = None
        try:
            layer_w = layer.width()
            layer_h = layer.height()
            ext = layer.extent()
            if self._is_online_provider(layer) or layer_w <= 0 or layer_h <= 0:
                native_units = self._online_native_mupp(layer)
            elif ext.width() > 0 and ext.height() > 0:
                native_units = max(ext.width() / layer_w, ext.height() / layer_h)
            else:
                native_units = 0.0
            if native_units and native_units > 0:
                ref = zone_in_layer if zone_in_layer is not None else ext
                meters = self._mupp_to_meters(layer, ref, native_units)
                if meters and meters > 0:
                    native_mupp = float(meters)
        except (RuntimeError, AttributeError, ValueError):
            native_mupp = None
        return zone_area_m2, native_mupp

    def _on_auto_run_plan_ready(self, prompt: str, plan: object) -> None:
        """Main thread: store the plan under its prompt and refine the detail
        seed from it, but only while that prompt is still the committed one."""
        self._auto_run_plan_task = None
        if not isinstance(plan, dict) or plan.get("error"):
            return
        prompt = (prompt or "").strip()
        if not prompt:
            return
        if prompt.lower() != self._resolved_auto_object_class().strip().lower():
            return  # the user moved on to a different object since the fetch
        self._auto_run_plan = {"prompt": prompt, "plan": plan}
        self._reseed_auto_detail_from_plan(prompt, plan)

    def _on_auto_run_plan_failed(self) -> None:
        """Run-plan fetch failed/timed out: keep the blob path silently."""
        self._auto_run_plan_task = None

    # ---- Async server token resolution (offline-lexicon miss) --------------

    def _fetch_auto_token(self, raw: str) -> None:
        """Fire-and-forget: resolve a prompt the offline lexicon could not to
        its English token via the server, so the detail seed can key on it
        without waiting for Detect.

        Off the GUI thread via a hidden task; never blocks, fails open (the raw
        prompt keeps the generic seed on any error). A no-op for an already
        English object word (nothing to translate) and mid-run/review (the
        prompt is locked then). A late answer re-seeds only while the prompt is
        still the one shown (see _on_auto_token_ready)."""
        raw = (raw or "").strip()
        if not raw or not self.dock_widget:
            return
        if self._auto_worker is not None or self._auto_review is not None:
            return
        try:
            from ..dock.prompt_guard import is_known_object
            if is_known_object(raw):
                return  # already an English object word: nothing to resolve
        except Exception:  # noqa: BLE001 -- best-effort gate  # nosec B110
            pass
        # Only one in flight; a newer commit supersedes the older request.
        self._cancel_task("_auto_token_task")
        try:
            from qgis.core import QgsApplication

            from ...api.prompt_translation import resolve_english_prompt
            from ...workers.generic_request_task import GenericRequestTask
            task = GenericRequestTask(
                tr("Resolving object name"),
                lambda: {"token": resolve_english_prompt(raw)},
                hidden=True,
            )
            task.succeeded.connect(
                lambda res, r=raw: self._on_auto_token_ready(r, res))
            task.failed.connect(lambda *_a: self._on_auto_token_failed())
            self._auto_token_task = task
            QgsApplication.taskManager().addTask(task)
        except Exception:  # noqa: BLE001 -- translation is best-effort
            self._auto_token_task = None  # nosec B110

    def _on_auto_token_ready(self, raw: str, result: object) -> None:
        """Main thread: cache the server-resolved token and re-seed the detail,
        but only while ``raw`` is still the committed prompt. The blob re-seed
        enforces the manual slider lock, so a user who took over the slider is
        never overridden."""
        self._auto_token_task = None
        token = result.get("token") if isinstance(result, dict) else None
        if not isinstance(token, str) or not token:
            return
        raw = (raw or "").strip()
        if not raw or token.strip().lower() == raw.lower():
            return  # server confirmed it was already English: nothing to swap
        if raw.lower() != self._current_auto_object_class().strip().lower():
            return  # the user moved on to a different object since the fetch
        cache = getattr(self, "_auto_token_cache", None)
        if cache is None:
            cache = {}
            self._auto_token_cache = cache
        cache[raw.lower()] = token
        # Re-seed with the resolved token and refresh the run plan under it.
        self._reseed_auto_detail_from_blob(token)
        self._fetch_auto_run_plan(token)

    def _on_auto_token_failed(self) -> None:
        """Token resolution failed/timed out: keep the raw-prompt path."""
        self._auto_token_task = None

    def _reseed_auto_detail_from_plan(self, prompt: str, plan: dict) -> None:
        """Re-seed the detail slider from the plan's target resolution, reusing
        the same coarse -> fine walk as the blob seed. No-op while a run/review
        is in flight or when the user already took over the slider for this
        prompt."""
        if not self.dock_widget or self._auto_zone is None:
            return
        if self._auto_worker is not None or self._auto_review is not None:
            return
        prompt = (prompt or "").strip()
        if not prompt or prompt.lower() != self._resolved_auto_object_class().strip().lower():
            return
        if self._auto_detail_user_locked and getattr(self, "_auto_detail_lock_prompt", "") == prompt.lower():
            return
        target_mupp = plan.get("target_mupp")
        if not isinstance(target_mupp, (int, float)) or isinstance(target_mupp, bool) or target_mupp <= 0:
            return
        obj_m = plan.get("object_size_m")
        is_valid_obj_m = isinstance(obj_m, (int, float)) and not isinstance(obj_m, bool) and obj_m > 0
        obj_m = float(obj_m) if is_valid_obj_m else 10.0
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            zone_in_layer = self._reproject_zone_to_layer_crs(self._auto_zone, layer)
            detail = self._auto_detail_for_target(
                layer, zone_in_layer, obj_m, float(target_mupp))
            self.dock_widget.set_auto_detail_value(detail)
            self._update_credit_estimate()
        except (RuntimeError, AttributeError):
            pass

    def _push_detail_feedback(self, layer, zone_in_layer, ground_mupp: float) -> None:
        """Classify the CURRENT detail level against the named object and this
        zone, and hand the verdict to the dock's one-line slider guidance.

        The verdict compares the level the user chose with the level the seed
        logic recommends for the same prompt + zone (so the guidance moves with
        the object AND the zone), plus two physical bounds: the object no
        longer resolvable at this resolution (too coarse to spot), and a
        resolution far finer than the object's target (past diminishing
        returns: extra credits, and large objects fragment across tiles).
        Runs at the debounced credit-estimate chokepoint, so it tracks every
        slider drag, prompt commit, zone redraw and layer switch.
        """
        if self.dock_widget is None:
            return
        if ground_mupp <= 0:
            self.dock_widget.set_auto_detail_feedback(None, "")
            return
        obj = self._current_auto_object_class()
        # Key the sizing verdict on the resolved English token (so a localized
        # prompt is judged like its English equivalent), but display the user's
        # own word in the hint.
        obj_token = self._resolve_object_token(obj)
        has_reference = False
        try:
            has_reference = self._auto_exemplar_store.count() > 0
        except (RuntimeError, AttributeError):
            pass
        if not obj and not has_reference:
            # No object defined: the detail card is gated with its own hint.
            self.dock_widget.set_auto_detail_feedback(None, "")
            return
        try:
            from ...core.detection_policy import (
                detail_over_ratio,
                detail_over_ratio_free,
                object_min_px,
            )

            obj_m, target_mupp = self._object_detail_profile(obj_token)
            value = self._get_auto_detail_level()
            recommended = self._recommended_detail_now(layer, zone_in_layer, obj_token)
            target_met = ground_mupp <= target_mupp
            # Free runs spend scarce lifetime trial credits, so their
            # past-diminishing-returns nudge fires earlier than a subscriber's.
            is_subscriber = bool(getattr(
                self.dock_widget, "_auto_is_subscriber", False))
            over_ratio = (detail_over_ratio() if is_subscriber
                          else detail_over_ratio_free())
            if obj_m / ground_mupp < object_min_px():
                state = "coarse"
            elif ground_mupp < target_mupp * over_ratio:
                state = "over"
            elif value > recommended:
                # Past a budget-capped recommendation extra detail genuinely
                # helps; past a target-met one it is mostly extra cost.
                state = "above" if target_met else "helps"
            elif value < recommended:
                state = "below"
            else:
                state = "recommended"
            self.dock_widget.set_auto_detail_feedback(state, obj)
        except (RuntimeError, AttributeError, ValueError, ZeroDivisionError):
            pass

    def _recommended_detail_now(self, layer, zone_in_layer, object_class: str) -> int:
        """The level the seed logic recommends RIGHT NOW for this prompt + zone.

        Recomputed live (never cached) so a layer or zone change can never
        leave the guidance comparing against a stale pick. Prefers the active
        server run plan's target when one matches the prompt, exactly like the
        seed appliers, so the guidance never disagrees with the seed."""
        plan = self._active_run_plan(object_class) if object_class else None
        if plan:
            target_mupp = plan.get("target_mupp")
            if isinstance(target_mupp, (int, float)) and not isinstance(target_mupp, bool) and target_mupp > 0:
                obj_m = plan.get("object_size_m")
                is_valid_obj_m = isinstance(obj_m, (int, float)) and not isinstance(obj_m, bool) and obj_m > 0
                obj_m = float(obj_m) if is_valid_obj_m else 10.0
                return self._auto_detail_for_target(
                    layer, zone_in_layer, obj_m, float(target_mupp))
        if object_class:
            return self._auto_detail_for_object(layer, zone_in_layer, object_class)
        return self._default_detail_for_zone(layer, zone_in_layer)

    def _mupp_to_meters(self, layer, zone_in_layer, mupp: float) -> float:
        """Convert a layer-CRS resolution (CRS units per pixel) to meters per
        pixel, measured at the zone center.

        An ellipsoidal measurement (WGS84) handles every CRS uniformly:
        projected layers in metres pass through, projected feet convert, and
        geographic layers in degrees get a real ground distance instead of a
        meaningless degree value. Returns 0.0 when it cannot be computed.
        """
        try:
            from qgis.core import (
                QgsDistanceArea,
                QgsPointXY,
                QgsProject,
            )

            da = QgsDistanceArea()
            da.setSourceCrs(layer.crs(), QgsProject.instance().transformContext())
            da.setEllipsoid("WGS84")
            cx = (zone_in_layer.xMinimum() + zone_in_layer.xMaximum()) / 2.0
            cy = (zone_in_layer.yMinimum() + zone_in_layer.yMaximum()) / 2.0
            # Measure BOTH axes and take the coarser ground distance. For a
            # geographic CRS (EPSG:4326) a pixel is square in degrees but a
            # degree of longitude shrinks with latitude, so a horizontal-only
            # measurement underestimates the effective GSD at high latitude and
            # skews the detail seed. Seeding on the coarser axis stays conservative
            # (never over-tiles). Projected metric CRSes are isotropic, so both
            # axes agree and this is a no-op there.
            dist_x = da.measureLine(QgsPointXY(cx, cy), QgsPointXY(cx + mupp, cy))
            dist_y = da.measureLine(QgsPointXY(cx, cy), QgsPointXY(cx, cy + mupp))
            dist = max(dist_x, dist_y)
            return da.convertLengthMeasurement(dist, DistanceMeters)
        except (RuntimeError, AttributeError, ValueError):
            return 0.0

    def _compute_auto_grid(self, layer) -> dict | None:
        """Compute the pixel grid for one automatic-detection run.

        Returns a dict with keys:
            pixel_w  (int)  -- total image width in pixels
            pixel_h  (int)  -- total image height in pixels
            zone_x   (int)  -- always 0 (bbox carries the extent for both paths)
            zone_y   (int)  -- always 0
            bbox     (tuple) -- (minx, miny, maxx, maxy) in layer CRS
            online   (bool) -- True when the map renderer path must be used
                               (always True once a zone is drawn, regardless of
                               layer type; False only for the no-zone local path)

        Returns None when a grid cannot be computed (e.g. no zone for an online layer).

        Slider-driven path (zone set, any layer type):
            Resolution is driven by the detail slider: the zone's longer side is
            rendered as exactly n tiles (snap formula: TILE_SIZE + (n-1) * stride
            pixels), so the user directly chooses how finely the zone is divided
            and therefore the credit cost. Local rasters are clamped to their
            native resolution so the render never upsamples the source pixels.
            online=True. zone_x and zone_y are 0 (the bbox IS the zone extent).

        Native path (no zone, local raster only):
            Derives pixel dimensions from the layer's own pixel grid. online=False.
            Online providers cannot run without a zone and return None here.
        """
        if self._tile_manager is None:
            self._setup_auto_mode()

        try:
            layer_w = layer.width()
            layer_h = layer.height()
            ext = layer.extent()
        except (RuntimeError, AttributeError):
            return None

        use_online = layer_w <= 0 or layer_h <= 0 or self._is_online_provider(layer)

        if self._auto_zone is not None:
            # Slider-driven path for EVERY layer once a zone is drawn: the user
            # picks the grid subdivision n (zone's longer side = n tiles), and
            # therefore the credit cost. The actual sizing lives in
            # _grid_for_detail so the slider cap and the m/px hint share it.
            zone_in_layer = self._reproject_zone_to_layer_crs(self._auto_zone, layer)
            detail_n = self._get_auto_detail_level()
            sized = self._grid_for_detail(layer, zone_in_layer, detail_n)
            if sized is None:
                return None
            pixel_w, pixel_h, mupp, _tiles = sized

            # Extend the geo bbox so it matches the snapped pixel grid at the
            # SAME mupp on both axes. snap_dimensions() rounds each axis up
            # independently, so the snapped pixel aspect no longer equals the
            # zone aspect; rendering the raw zone into that grid stretches the
            # image (up to ~1.8x) and the model sees deformed objects. Anchor
            # at the top-left corner (image origin) and grow right/down. The
            # forward render and the worker's inverse mapping both read this
            # bbox, so detections stay correctly georeferenced; the only
            # visible effect is the preview grid spilling slightly past the
            # drawn zone (blank margin at a raster edge renders harmlessly).
            minx = zone_in_layer.xMinimum()
            maxy = zone_in_layer.yMaximum()
            maxx = minx + pixel_w * mupp
            miny = maxy - pixel_h * mupp

            return {
                "pixel_w": pixel_w,
                "pixel_h": pixel_h,
                "zone_x": 0,
                "zone_y": 0,
                "bbox": (minx, miny, maxx, maxy),
                "online": True,
            }

        # No zone (MCP / full-raster path): keep the native pixel grid. Online
        # providers cannot run without a zone.
        if use_online:
            return None
        if ext.width() <= 0:
            return None
        pixel_w, pixel_h = layer_w, layer_h
        zone_x, zone_y = 0, 0
        bbox = (ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum())
        return {
            "pixel_w": pixel_w, "pixel_h": pixel_h,
            "zone_x": zone_x, "zone_y": zone_y,
            "bbox": bbox, "online": False,
        }
