from __future__ import annotations

import os
import sys
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)
from qgis.gui import QgisInterface, QgsRubberBand
from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtGui import QAction, QIcon
from qgis.PyQt.QtWidgets import QMenu

from ..core.i18n import tr
from ..core.qt_compat import PolygonGeometry
from ..core.prompt_manager import FrozenCropSession, PromptManager
from ..core.review_defaults import (
    AUTO_DEFAULT_CONFIDENCE as _AUTO_DEFAULT_CONFIDENCE,
)
from ..core.review_defaults import (
    REFINE_EXPAND_DEFAULT,
    REFINE_FILL_HOLES_DEFAULT,
    REFINE_MAX_SIZE_M2_DEFAULT,
    REFINE_MIN_SIZE_M2_DEFAULT,
    REFINE_ORTHO_DEFAULT,
    REFINE_SIMPLIFY_DEFAULT,
    REFINE_SMOOTH_DEFAULT,
)
from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool
from .error_report_dialog import start_log_collector, stop_log_collector
from .canvas_palette import PENDING_FILL, PENDING_STROKE
from .plugin.auto_flow import AutoFlowMixin
from .plugin.auto_run import AutoRunMixin
from .plugin.auto_results import AutoResultsMixin
from .plugin.auto_review import AutoReviewMixin
from .plugin.manual_handoff import ManualHandoffMixin
from .plugin.handoff_shape import HandoffShapeMixin
from .plugin.exemplars import ExemplarsMixin
from .plugin.auto_lifecycle import AutoLifecycleMixin
from .plugin.auto_zone import AutoZoneMixin
from .plugin.demo_scene import DemoSceneMixin
from .plugin.env_setup import EnvSetupMixin
from .plugin.manual_workflow import ManualWorkflowMixin
from .plugin.manual_crops import ManualCropsMixin
from .plugin.manual_predict import ManualPredictMixin
from .plugin.shared import park_orphaned_worker


class AISegmentationPlugin(
    AutoFlowMixin,
    AutoRunMixin,
    AutoResultsMixin,
    AutoReviewMixin,
    ManualHandoffMixin,
    HandoffShapeMixin,
    ExemplarsMixin,
    AutoLifecycleMixin,
    AutoZoneMixin,
    DemoSceneMixin,
    EnvSetupMixin,
    ManualWorkflowMixin,
    ManualCropsMixin,
    ManualPredictMixin,
):
    """The plugin controller. Behaviour is split across the mixins above
    (one file per concern in src/ui/plugin/); this module keeps construction,
    QGIS lifecycle (initGui/unload), dock management and shared helpers."""

    def __init__(self, iface: QgisInterface):
        self.iface = iface
        self.plugin_dir = Path(__file__).parent.parent.parent

        self.dock_widget: AISegmentationDockWidget | None = None
        self._dock_created = False
        self.map_tool: AISegmentationMapTool | None = None
        self.action: QAction | None = None
        self.terralab_menu: QMenu | None = None
        self.terralab_toolbar = None

        self.predictor = None
        self.prompts = PromptManager()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.current_low_res_mask = None  # For iterative refinement with negative points
        self.saved_polygons = []
        # Refine-in-Manual handoff: while True, a Manual session is refining the
        # Automatic review, so the mode switch must NOT run its destructive reset
        # and the held _auto_review survives the Manual<->Auto round trip.
        self._refine_handoff_active = False
        # True while a saved/seeded object is OPEN for editing (pending-blue with
        # a bolder outline); drives the active-object emphasis and Delete-key gate.
        self._is_refining_saved_object = False
        # The original saved entry re-opened for editing, so a Delete-undo can
        # restore it exactly.
        self._active_refine_origin_entry = None
        # Undo stack for Delete-key removals: each unit is a list of entry
        # dicts (one Suppr press = one unit, possibly several selected objects).
        self._deleted_objects_stack: list = []
        # Selection-first review state (refine handoff): the selected entry
        # dicts, their white selection outline, and the hover highlight band.
        self._handoff_selected_entries: list = []
        self._handoff_selection_band = None
        self._handoff_hover_band = None
        self._handoff_hover_entry = None
        # Bbox spatial index over saved_polygons, keyed by a stable per-entry
        # token (_htok) so single-object changes maintain it incrementally;
        # _handoff_tok2entry resolves a token back to its entry and
        # _handoff_hit_tok_seq mints tokens (never reused in-session).
        self._handoff_hit_index = None
        self._handoff_tok2entry: dict = {}
        self._handoff_hit_tok_seq: int = 0
        # Speculative selection prewarm (refine handoff): debounce timer + the
        # crop spec (center/scale) of the last handoff-initiated encode, so the
        # open can skip a duplicate encode the selection already started.
        self._handoff_prewarm_timer = None
        self._handoff_crop_spec: tuple | None = None
        # Synthetic det_id sequence for hand-drawn/legacy entries (see
        # _next_handoff_det_id).
        self._handoff_det_id_seq = None
        # Geometries the user hand-edited/added/carved during a refine handoff.
        # They are PROTECTED: a confidence change re-filters only the untouched
        # auto detections and never drops these, so the slider stays usable after
        # a manual refine without wiping hand work (run CRS QgsGeometry list).
        self._auto_protected_geoms: list = []
        # The local SAM predictor loads lazily/async (None until ready), so a
        # Refine-in-Manual click can arrive before the model is up. When that
        # happens we hold the import and complete it from _on_predictor_loaded.
        self._handoff_source_layer = None
        self._pending_refine_import = False
        # D1: a Refine click with no local AI starts the install in the background
        # while the user stays on the Automatic review. True from that click until
        # the predictor loads (then the handoff opens automatically) or the review
        # is torn down / the install fails.
        self._refine_install_pending = False
        # Refine-in-Manual seeds render as TWO in-memory layers, not N rubber
        # bands (500-2000 canvas items froze QGIS): _handoff_pending_layer holds
        # the not-yet-validated (blue) seeds, _handoff_kept_layer the ones
        # validated (green) this session. Only the ACTIVE object keeps a band
        # (pending-blue, bolder outline). saved_rubber_bands stays index-locked with
        # saved_polygons but holds None for every layer-rendered entry (plan 11
        # §1.1). Both layers live only for the handoff and are removed on teardown.
        self._handoff_pending_layer = None
        self._handoff_kept_layer = None
        self._mask_state_history: list = []  # Stack of mask states for per-point undo (capped at 30)
        self._frozen_sessions: list[FrozenCropSession] = []  # Frozen crop polygons
        self._active_crop_points_positive: list[tuple[float, float]] = []
        self._active_crop_points_negative: list[tuple[float, float]] = []
        # Polygon of the last unfrozen session, displayed until the next
        # prediction replaces it (the session has points but no numpy mask).
        self._unfrozen_display_polygon: QgsGeometry | None = None

        self._initialized = False
        self._setup_done = False
        self._current_layer = None
        self._current_layer_name = ""

        # MCP/headless mode: when True, skip modal dialogs and cursors
        self._headless = False
        self._headless_error = None

        # Refinement settings (#12, #23: defaults tuned for ease-of-use).
        # Shared with the dock and the session-reset/restore fallbacks via
        # core/review_defaults.py (no local copies to keep in sync).
        self._refine_simplify = REFINE_SIMPLIFY_DEFAULT
        self._refine_smooth = REFINE_SMOOTH_DEFAULT
        self._refine_expand = REFINE_EXPAND_DEFAULT
        self._refine_fill_holes = REFINE_FILL_HOLES_DEFAULT
        self._refine_ortho = REFINE_ORTHO_DEFAULT
        self._refine_min_area = 200  # overridden by _compute_auto_min_area() × 2
        # User Min/Max size window in ground m2 (0 = off); a Refine-in-Manual
        # handoff seeds these from the Automatic review's size filters.
        self._refine_min_size_m2 = REFINE_MIN_SIZE_M2_DEFAULT
        self._refine_max_size_m2 = REFINE_MAX_SIZE_M2_DEFAULT

        self._is_non_georeferenced_mode = False  # Track if current layer is non-georeferenced
        self._is_online_layer = False  # Track if current layer is online (WMS, XYZ, etc.)
        self._disjoint_warning_shown = False

        # On-demand encoding state
        self._current_crop_info = None  # dict with 'bounds', 'img_shape'
        self._current_raster_path = None
        self._encoding_in_progress = False  # Guard against concurrent clicks (main/UI thread only)
        self._shortcut_filter = None  # Event filter for keyboard shortcuts
        self._current_crop_canvas_mupp = None  # canvas mupp at encode time (zoom detection)
        self._current_crop_actual_mupp = None  # actual mupp used for the crop (may differ if zoomed out)
        self._current_crop_scale_factor = None  # scale_factor used for file-based crop
        self.deps_install_worker = None
        self.download_worker = None
        self._verify_worker = None
        self._predictor_worker = None
        self._startup_check_worker = None
        self._device_info_worker = None
        # Cached "local venv is installed" flag (from the startup check). Drives
        # the Refine-in-Manual env gate so an uninstalled env shows the install
        # dialog instead of trapping the user in "Preparing Manual mode".
        self._env_ready = False
        # Network requests run as hidden QgsTasks (cooperative cancel, never
        # QThread.terminate which crashes QGIS when the socket is wedged).
        self._key_revalidate_task = None  # GenericRequestTask | None
        self._config_prefetch_task = None  # GenericRequestTask | None
        # Warms the segment-library catalogue cache off the GUI thread so the
        # library opens instantly (the dialog reads cache-only, never network).
        self._catalog_prefetch_task = None  # GenericRequestTask | None
        # Throttle startup revalidation so it does not refire "constantly".
        self._last_key_validation_unix: float = 0.0
        # Anti-spam guard for the transient "no connection" message-bar notice.
        self._last_conn_notice_monotonic: float = 0.0
        self._pairing_worker = None  # QgsTask polling the one-click sign-in
        self._pairing_cancel_task = None  # fire-and-forget server-side cancel

        self.mask_rubber_band: QgsRubberBand | None = None
        # Index-locked with saved_polygons. Holds None for entries drawn by the
        # handoff memory layers instead of a per-object band; the
        # length must always equal len(saved_polygons) or
        # _ensure_polygon_rubberband_sync will truncate the polygons as "repair".
        self.saved_rubber_bands: list[QgsRubberBand | None] = []

        self._previous_map_tool = None  # Store the tool active before segmentation
        self._stopping_segmentation = False  # Flag to track if we're stopping programmatically
        self._exporting_in_progress = False  # Guard against double-click on export

        # CRS transforms (canvas CRS <-> raster CRS), created when features load.
        # None when both CRS are the same (no transform needed).
        self._canvas_to_raster_xform: QgsCoordinateTransform | None = None
        self._raster_to_canvas_xform: QgsCoordinateTransform | None = None

        # Auto mode state (Pro tier) - populated by _setup_auto_mode()
        self._auto_zone: QgsRectangle | None = None
        # The drawn polygon zone (canvas CRS). Its bounding box IS _auto_zone, so
        # the whole bbox-based grid/render pipeline is unchanged; the polygon is
        # an extra constraint that culls tiles falling outside the drawn shape.
        self._auto_zone_polygon = None  # QgsGeometry | None
        # The drawn polygon reprojected into the run CRS, set per run: every
        # detection is clipped to it so nothing outside the shape is shown or
        # exported (None on the rectangle/MCP path = no clip).
        self._auto_clip_polygon = None  # QgsGeometry | None
        self._auto_clip_engine = None   # prepared GEOS engine for the clip polygon
        self._zone_selection_tool = None  # PolygonZoneMapTool | None
        # True once the user moves the detail slider themselves: the debounced
        # object-aware re-seed then stops overriding their manual pick, for the
        # prompt recorded below. Reset whenever a new zone is drawn (a fresh
        # zone = a fresh default) or the prompt changes (a new object = a new
        # sizing problem, the auto seed owns the slider again).
        self._auto_detail_user_locked = False
        self._auto_detail_lock_prompt = ""
        self._zone_rubber_band: QgsRubberBand | None = None
        self._zone_delete_badge = None  # ZoneDeleteBadge | None
        self._zone_badge_filter = None  # ZoneBadgeClickFilter | None
        self._zone_escape_filter = None  # ZoneEscapeFilter | None
        self._zone_grid_rubber_band = None  # QgsRubberBand | None (tile grid preview)
        self._tile_manager = None  # TileManager | None

        # Visual exemplars ("draw one example, find all"). The store
        # holds the example boxes (canvas CRS) + labels; persistent rubber bands
        # keep them visible on the map; the draw tool is armed on demand.
        from ..core.exemplar_store import ExemplarStore
        self._auto_exemplar_store = ExemplarStore()
        self._exemplar_maptool = None  # PolygonZoneMapTool | None (example draw)
        self._exemplar_bands: dict = {}  # exemplar id -> QgsRubberBand
        self._maptool_before_exemplar = None  # restore after a one-shot draw
        self._pending_exemplar_label = 1  # label (1/0) for the armed example draw
        # Tool active before the zone drawing tool was armed (QGIS's pan tool by
        # default). Restored when the zone is drawn or the flow exits, so the
        # user gets the hand back instead of a bare cursor (mirrors Manual's
        # _previous_map_tool / _restore_previous_map_tool).
        self._maptool_before_zone = None

        # Auto detection worker state (plan #78)
        self._auto_worker = None  # AutoDetectionWorker | None
        # Main-thread per-tile render bridge for the active run (held so it is
        # not garbage-collected mid-run; nulled when the run winds down).
        self._auto_tile_bridge = None  # TileRenderBridge | None
        # Tile fragments are converted to geometry once on arrival and folded into
        # a running IncrementalMerger, so objects split across tiles are stitched
        # live (the preview shows whole objects, not cut pieces) and no raw mask
        # is held for the whole run.
        self._auto_merger = None  # IncrementalMerger | None
        self._auto_crs_authid: str | None = None  # captured from the first detection
        self._auto_gsd: float = 0.0  # ground sample distance (map units/px) of the run
        self._auto_gsd_m: float = 0.0  # the same GSD in meters/px (m2 size floors)
        # Ground units per RETURNED-mask pixel, observed from the run's server
        # responses (coarser than _auto_gsd when the model answers at a reduced
        # grid). The review's px->ground refine scales by it; 0.0 = none seen.
        self._auto_mask_gsd: float = 0.0
        # Merge policy: True = keep objects SEPARATE (count, never seam-merge),
        # False = merge tile-split objects (map continuous features). Smart
        # default per object type at run start; user-overridable in the review.
        self._auto_merge_separate: bool = True
        # How the merge policy was decided: "prompt" (object token), "signal"
        # (exemplar-only auto count-vs-map from the run's own masks) or
        # "override" (the user re-grouped it in the review). Telemetry only.
        self._auto_merge_mode_source: str = "prompt"
        # Exemplar-only count-vs-map auto decision: the retained raw per-tile
        # fragments (None = not an exemplar-only run, or retention overflowed).
        # The map-likeness signal is the area-weighted mean tile coverage of the
        # non-failure fragments (sum(cov^2)/sum(cov), cov = fragment area / tile
        # ground area, failure blobs above the hard cap excluded): high for
        # continuous cover, near zero for small countable objects. Accumulated as
        # two running sums plus a fragment count (0 = signal cannot run).
        self._auto_is_exemplar_only: bool = False
        self._auto_raw_fragments: list | None = None
        self._auto_raw_n_total: int = 0
        self._auto_raw_cov_sum: float = 0.0
        self._auto_raw_cov_sq_sum: float = 0.0
        self._auto_tile_ground_area: float = 0.0
        self._auto_merge_override_used: bool = False
        self._auto_selection_layer = None  # QgsVectorLayer | None (in-progress results)
        # Review display colour mode: 'normal' / 'outline' / 'confidence' /
        # 'random' (visual only; never touches geometry, filters or export).
        # Random by default, matching the dock combo; re-seeded to Random for
        # every NEW review (_seed_review_display_mode).
        self._auto_display_mode = "random"
        self._auto_run_id: str | None = None
        self._auto_run_ctx: dict | None = None     # inputs of the active run
        self._last_usage: dict = {}  # last fetched usage (credits/is_free_tier) for telemetry
        self._usage_fetch_task = None  # GenericRequestTask | None (plan #79)
        # One-shot guard so a lapsed-subscription (failed payment) notice is
        # shown at most once per session, not on every credits refresh.
        self._billing_warning_shown = False
        # Best-effort backend cold-start ping when entering the Automatic flow.
        self._warmup_task = None  # GenericRequestTask | None
        self._last_warmup_monotonic: float = 0.0
        # Per-prompt run plan fetched async on prompt commit (target resolution,
        # recall floors, confidence, review shape). {"prompt": str, "plan": dict}
        # or None; applied only while its prompt still matches, else the blob /
        # generic path stands. Fire-and-forget, fails open.
        self._auto_run_plan: dict | None = None
        self._auto_run_plan_task = None  # GenericRequestTask | None
        # Localized prompt -> English cloud-model token, resolved once per prompt
        # on commit (offline lexicon, with an async server fallback) so the
        # detail seed keys on the SAME token the run will send. Display stays the
        # user's own words; this only steers the policy lookups.
        self._auto_token_cache: dict[str, str] = {}
        self._auto_token_task = None  # GenericRequestTask | None
        # MCP headless result bookkeeping (plan #79): set by signal handlers.
        self._last_auto_result: dict | None = None
        # Timing/observability for the auto run: render duration (the upfront
        # basemap fetch) and the detection-phase start, logged as a run summary at
        # finalize so a slow run is debuggable from the message log alone.
        self._auto_render_ms: int = 0
        self._auto_detect_t0: float = 0.0
        # Per-run telemetry bookkeeping: terminal reason + degraded-tile counters
        # aggregated from worker warnings, read once at finalize.
        self._auto_tel_stop_reason: str | None = None
        self._auto_skipped_tiles: int = 0
        self._auto_timeout_tiles: int = 0
        # Post-run review state (plan #78 round 5): geoms waiting for explicit Export.
        self._auto_review: dict | None = None
        # True while _run_auto_detect_headless drives a synchronous MCP call.
        self._auto_headless_run: bool = False
        # Live confidence re-filter: the run keeps every detection above a low
        # recall floor as (per-tile geom, score); the review confidence slider
        # re-filters this list with no re-detection. _auto_confidence is the
        # active cutoff (the live merge during the run uses it too).
        self._auto_confidence: float = _AUTO_DEFAULT_CONFIDENCE
        # Count of raw fragments fed to the merger this run, for the run-summary
        # log only. The fragments themselves are NOT kept: the merger owns the
        # whole-object result the review reads (_auto_objects), so accumulating
        # every raw (geom, score) tuple was pure dead memory.
        self._auto_raw_count: int = 0
        # How many tiles this run stayed at the per-inference mask ceiling
        # AFTER the re-split ladder ran (residual truncation, logged
        # internally). Read from the worker before it is nulled at run end.
        self._auto_dense_tiles: int = 0
        # Canonical result set: WHOLE merged objects, confidence-agnostic. Each
        # entry is (base_geom, score, area_m2) where score is the MAX of the
        # object's constituent fragment scores and area_m2 is the geodesic area.
        # Built once per run (fragments are unioned regardless of score so seam
        # halves always stitch); the confidence + min/max-size filters then act
        # on THESE whole objects, never on raw fragments. This is what makes the
        # confidence slider drop weak OBJECTS instead of cutting buildings in half.
        self._auto_objects: list = []
        # Once-per-generation guard so a swallowed review-rebuild geometry error
        # is logged once (not on every confidence-drag tick); reset when a new
        # reslice generation starts.
        self._review_push_err_logged: bool = False
        # Simplified WHOLE-object (geom, score) pairs sorted by score desc, built
        # when a review starts so the confidence slider's live drag is a cheap
        # prefix slice instead of re-simplifying every object on each tick. Built
        # COOPERATIVELY in the background; until it is ready the slider drag falls
        # back to filtering _auto_objects directly (still whole objects, never
        # fragments). _build_state holds the in-flight build; _build_gen
        # invalidates it on a new run / teardown.
        self._auto_preview_geoms: list = []
        self._auto_preview_build_state: dict | None = None
        self._auto_preview_build_gen: int = 0
        # Live tile processing is decoupled from the worker signal so the GUI
        # thread is never blocked converting a whole tile's masks at once (the
        # freeze the user hit). Arriving detections queue here; a cooperative,
        # time-budgeted pump (_pump_auto_tiles) drains them a slice at a time,
        # and the live preview repaint is coalesced via a single-shot timer.
        from collections import deque as _deque
        self._auto_tile_queue = _deque()
        self._auto_pump_scheduled: bool = False
        self._auto_repaint_timer = None  # QTimer | None (coalesced live repaint)
        # Live preset-refine cache: fid -> (visible, refined geom). A merger
        # keeper is immutable (merges retire fids), so each object is refined
        # once; reset per run via _reset_auto_live_pipeline.
        self._auto_live_refine_cache: dict = {}
        self._auto_live_refine_px: float = -1.0
        self._auto_live_measurer = None  # QgsDistanceArea | None
        self._auto_live_params = None  # per-run memo of the review preset
        # Live preview provider mapping: merger keeper fid -> (provider_fid,
        # stamp, is_full, score). Lets the live repaint update the selection
        # layer incrementally (add/change/delete only the delta) instead of
        # truncating + re-adding every feature. Reset per run via
        # _stop_auto_live_pump; the layer is recreated fresh each run.
        self._auto_live_fid_map: dict = {}
        # End-of-run refine is also cooperative: refining hundreds of objects in
        # one synchronous pass froze QGIS at the very end of a run. _auto_finalize
        # _state holds the in-flight batch; _gen invalidates a stale step if a new
        # run starts or the flow is torn down before the refine finishes.
        self._auto_finalize_state: dict | None = None
        self._auto_finalize_gen: int = 0
        # Reslice refine cache: the refined (repaired, MultiPolygon-coerced)
        # geometry per canonical object index, valid for ONE shape-params key at
        # a time (see _review_shape_key). A filter-only reslice (Confidence /
        # Min / Max size) then reuses every refined geometry instead of re-
        # running the GEOS refine on the whole visible set; a shape-params
        # change naturally resets the key and recomputes. Reset whenever
        # _auto_objects is rebuilt (_reset_review_refine_cache).
        self._auto_reslice_cache: dict = {"key": None, "geoms": {}}
        # Review provider mapping: det_id -> (provider_fid, stamp, is_full,
        # score), the review twin of _auto_live_fid_map. Lets a reslice or a
        # confidence-drag tick update the selection layer incrementally
        # (add/change/delete only the delta) instead of truncating + re-adding
        # every feature. Reset whenever the selection layer is (re)created.
        self._review_fid_map: dict = {}

    @staticmethod
    def _safe_remove_rubber_band(rb):
        """Remove a rubber band from the canvas scene, handling C++ deletion."""
        if rb is None:
            return
        try:
            # QgsRubberBand doesn't expose parentWidget; use scene directly
            scene = rb.scene()
            if scene is not None:
                scene.removeItem(rb)
        except (RuntimeError, AttributeError):
            pass  # C++ object already deleted (QGIS shutdown)

    def _is_layer_valid(self, layer=None) -> bool:
        """Check if a layer's C++ object is still alive."""
        if layer is None:
            layer = self._current_layer
        if layer is None:
            return False
        try:
            layer.id()
            return True
        except RuntimeError:
            return False

    def _is_layer_georeferenced(self, layer) -> bool:
        """Check if a raster layer is properly georeferenced."""
        if layer is None or not isinstance(layer, QgsRasterLayer):
            return False

        try:
            source = layer.source().lower()
        except RuntimeError:
            return False

        # PNG, JPG, BMP etc. without world files are not georeferenced
        non_georef_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        has_non_georef_ext = any(source.endswith(ext) for ext in non_georef_extensions)

        # If it's a known non-georeferenced format, check if it has a valid CRS
        if has_non_georef_ext:
            try:
                # Check if the layer has a valid CRS (not just default)
                if not layer.crs().isValid():
                    return False
                # Check if extent looks like pixel coordinates (0,0 to width,height)
                extent = layer.extent()
                if extent.xMinimum() == 0 and extent.yMinimum() == 0:
                    # Likely not georeferenced - just pixel dimensions
                    return False
            except RuntimeError:
                return False

        return True

    @staticmethod
    def _is_online_provider(layer) -> bool:
        """Check if a raster layer uses an online data provider."""
        if layer is None:
            return False
        try:
            provider = layer.dataProvider()
            if provider is None:
                return False
            from ..core.feature_encoder import ONLINE_PROVIDERS
            return provider.name() in ONLINE_PROVIDERS
        except (RuntimeError, AttributeError):
            return False

    def _ensure_polygon_rubberband_sync(self):
        """Check polygon/rubber band list consistency. Repair on mismatch."""
        n_polygons = len(self.saved_polygons)
        n_bands = len(self.saved_rubber_bands)
        if n_polygons != n_bands:
            QgsMessageLog.logMessage(
                f"BUG: polygon/rubber band mismatch: {n_polygons} vs {n_bands}. "
                "Truncating to min. Please report.",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            min_len = min(n_polygons, n_bands)
            while len(self.saved_rubber_bands) > min_len:
                rb = self.saved_rubber_bands.pop()
                self._safe_remove_rubber_band(rb)
            self.saved_polygons = self.saved_polygons[:min_len]

    @staticmethod
    def _compute_simplification_tolerance(transform_info, simplify_value):
        """Compute simplification tolerance from transform_info and slider value.

        Returns 0 if inputs are invalid or simplify_value is 0.
        """
        if simplify_value <= 0 or transform_info is None:
            return 0
        bbox = transform_info.get("bbox", [0, 1, 0, 1])
        img_shape = transform_info.get("img_shape", (1024, 1024))
        width_pixels = max(img_shape[1], 1)
        bbox_width = bbox[1] - bbox[0]
        if bbox_width == 0:
            return 0
        pixel_size = bbox_width / width_pixels
        return pixel_size * simplify_value * 0.5

    def initGui(self):
        from ..mcp_api import SegmentationMCPAPI
        self.mcp_api = SegmentationMCPAPI(self)

        start_log_collector()

        # Move any plain-QSettings activation key into QgsAuthManager.
        # Cheap, idempotent, and silent (never prompts for a master password).
        try:
            from ..core.activation_manager import migrate_legacy_key
            migrate_legacy_key()
        except Exception:  # nosec B110
            pass

        icon_path = str(self.plugin_dir / "resources" / "icons" / "icon.png")
        if not os.path.exists(icon_path):
            icon = QIcon()
        else:
            icon = QIcon(icon_path)

        self.action = QAction(
            icon,
            "AI Segmentation",
            self.iface.mainWindow()
        )
        self.action.setToolTip(
            "AI Segmentation by TerraLab\n{}".format(
                tr("Segment elements on raster images using AI"))
        )
        self.action.triggered.connect(self.toggle_dock_widget)

        from .terralab_toolbar import add_action_to_toolbar, get_or_create_terralab_toolbar
        self.terralab_toolbar = get_or_create_terralab_toolbar(self.iface)
        add_action_to_toolbar(self.terralab_toolbar, self.action, "ai-segmentation")

        from .terralab_menu import (
            add_plugin_to_menu,
            add_to_plugins_menu,
            get_or_create_terralab_menu,
        )
        self.terralab_menu = get_or_create_terralab_menu(self.iface.mainWindow())
        add_plugin_to_menu(self.terralab_menu, self.action, "ai-segmentation")
        add_to_plugins_menu(self.iface, self.action)

        # No "Settings" entry in the TerraLab menu: each plugin exposes its own
        # settings inside its dock (footer gear), a shared menu entry was
        # ambiguous with two plugins installed. Remove any leftover action from
        # an older plugin version still loaded in this session.
        for a in list(self.terralab_menu.actions()):
            if a.objectName() == "_terralab_settings_action":
                self.terralab_menu.removeAction(a)
                break

        # Cross-plugin discovery: show AI Edit entry even when it's not installed (#30).
        from .cross_plugin_discovery import make_ai_edit_action
        ai_edit_icon_path = str(self.plugin_dir / "resources" / "icons" / "ai_edit_icon.png")
        ai_edit_icon = QIcon(ai_edit_icon_path) if os.path.exists(ai_edit_icon_path) else None
        self.ai_edit_action = make_ai_edit_action(
            self.iface.mainWindow(),
            self.iface,
            tr("AI Edit"),
            tr("Generate imagery with AI on map zones (opens AI Edit plugin)"),
            icon=ai_edit_icon,
        )
        add_action_to_toolbar(self.terralab_toolbar, self.ai_edit_action, "ai-edit", is_cross_promo=True)
        add_plugin_to_menu(self.terralab_menu, self.ai_edit_action, "ai-edit")
        add_to_plugins_menu(self.iface, self.ai_edit_action)

        # Defer dock widget creation to first toggle for fast plugin load
        self.dock_widget = None
        self._dock_created = False

        self.map_tool = AISegmentationMapTool(self.iface.mapCanvas())
        self.map_tool.positive_click.connect(self._on_positive_click)
        self.map_tool.negative_click.connect(self._on_negative_click)
        # Refine-handoff selection model: double-click opens a detection for
        # editing, cursor motion drives the hover highlight (both no-op outside
        # the handoff).
        self.map_tool.double_click.connect(self._on_canvas_double_click)
        self.map_tool.cursor_moved.connect(self._on_handoff_cursor_moved)
        self.map_tool.tool_deactivated.connect(self._on_tool_deactivated)
        self.map_tool.undo_requested.connect(self._on_undo)
        self.map_tool.save_polygon_requested.connect(self._on_save_polygon)
        self.map_tool.export_layer_requested.connect(self._on_export_layer)
        self.map_tool.stop_segmentation_requested.connect(self._on_stop_segmentation)

        # Layer-removal lifecycle (T15/T16/T17): end any flow whose source
        # raster is about to leave the project. Disconnected in unload().
        QgsProject.instance().layersWillBeRemoved.connect(
            self._on_layers_will_be_removed)

        # A mid-review project save persists our Private working layers into
        # the .qgz; they reload as empty invisible memory layers. Sweep them
        # now (plugin reloaded mid-session) and on every project open.
        try:
            from ..core.output_store import sweep_stale_temp_layers
            sweep_stale_temp_layers()
            QgsProject.instance().readProject.connect(
                self._on_project_read_sweep_temp)
        except Exception:  # nosec B110
            pass

        self.mask_rubber_band = QgsRubberBand(
            self.iface.mapCanvas(),
            PolygonGeometry
        )
        self.mask_rubber_band.setColor(PENDING_FILL)
        self.mask_rubber_band.setStrokeColor(PENDING_STROKE)
        self.mask_rubber_band.setWidth(2)

        # Log plugin version and environment for diagnostics (no personal paths)
        try:
            plugin_version = self._read_plugin_version()
            qgis_version = Qgis.version() if hasattr(Qgis, "version") else "unknown"
            QgsMessageLog.logMessage(
                f"AI Segmentation v{plugin_version} | QGIS {qgis_version} | "
                f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} | {sys.platform}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
        except Exception:
            QgsMessageLog.logMessage(
                "AI Segmentation plugin loaded",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )

        # Create the dock now and register it with QGIS. With its stable
        # objectName, QGIS's own window-state restore then reopens it
        # automatically whenever the user left it open in the previous
        # session (exactly how AI Edit persists the panel). Construction is
        # pure Qt and cheap; the heavy environment checks only run when the
        # dock first becomes visible, so a launch with the panel closed
        # costs nothing.
        self._ensure_dock_widget()

        # Auto-open the panel on first install and after every upgrade (new
        # version), but never on a routine launch. Same-version launches let
        # QGIS restore the dock to the state the user left it in
        # (open/closed + position), via its objectName. Mirrors AI Edit.
        settings = QSettings()
        settings.remove("AISegmentation/dock_shown_once")  # superseded key
        current_version = self._read_plugin_version()
        last_shown_version = settings.value(
            "AISegmentation/dock_shown_version", "", type=str)
        if last_shown_version != current_version:
            settings.setValue(
                "AISegmentation/dock_shown_version", current_version)
            if self.dock_widget:
                self.dock_widget.show()
                self.dock_widget.raise_()
                self._ensure_dock_height()

    @staticmethod
    def _read_plugin_version() -> str:
        """Read the plugin version from metadata.txt (plugin root)."""
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        metadata_path = os.path.join(plugin_dir, "metadata.txt")
        try:
            with open(metadata_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("version="):
                        return line.split("=", 1)[1].strip()
        except OSError:
            pass
        return "unknown"

    def _ensure_dock_height(self):
        """Open the panel tall enough to actually work in. QGIS can dock it
        as a short box; grow it to most of the window height. Never shrinks
        a dock the user already made taller. Deferred one tick so the resize
        runs after QGIS finishes laying the dock out (mirrors AI Edit)."""
        def _apply():
            try:
                dock = self.dock_widget
                mw = self.iface.mainWindow()
                if dock is None or mw is None or not dock.isVisible():
                    return
                target = int(mw.height() * 0.85)
                if dock.height() >= target:
                    return
                mw.resizeDocks([dock], [target], Qt.Orientation.Vertical)
            except Exception:  # nosec B110
                pass
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, _apply)

    def _on_project_read_sweep_temp(self, *_args):
        """Remove stale Private working layers a saved project brought back.

        Deferred one event-loop turn: removing layers WHILE QGIS is still
        restoring the project races the snapping-config restore and leaves a
        dangling layer pointer in QgsSnappingConfig, which then crashes the
        NEXT project save (often the save-on-exit; upstream qgis/QGIS#42651).
        """
        try:
            from qgis.PyQt.QtCore import QTimer

            def _sweep():
                try:
                    from ..core.output_store import sweep_stale_temp_layers
                    sweep_stale_temp_layers()
                except Exception:  # nosec B110
                    pass

            QTimer.singleShot(0, _sweep)
        except Exception:  # nosec B110
            pass

    def unload(self):
        # Ship any queued telemetry before teardown (main thread here).
        try:
            from ..core.telemetry import flush as _telemetry_flush
            _telemetry_flush()
        except Exception:
            pass  # nosec B110
        # Data-loss guard: if a Refine-in-Manual handoff is live with hand edits,
        # fold them into the held review BEFORE clearing the flag, so the autosave
        # later in unload writes the MERGED set (not just the original detections).
        try:
            if getattr(self, "_refine_handoff_active", False) and self.saved_polygons:
                self._collect_manual_refine_into_review()
        except Exception:  # unload must never raise
            pass  # nosec B110
        # Clear any in-flight Refine-in-Manual handoff so the mode-switch guard
        # and review chokepoints below run their normal teardown, not the
        # handoff branch, on a half-torn-down state.
        self._refine_handoff_active = False
        self._auto_protected_geoms = []
        self._pending_refine_import = False
        self._handoff_source_layer = None
        # Detach the layer-removal lifecycle hook wired in initGui.
        try:
            QgsProject.instance().layersWillBeRemoved.disconnect(
                self._on_layers_will_be_removed)
        except (TypeError, RuntimeError):
            pass
        try:
            QgsProject.instance().readProject.disconnect(
                self._on_project_read_sweep_temp)
        except (TypeError, RuntimeError):
            pass
        # 0. Remove keyboard shortcut filter
        try:
            if self._shortcut_filter is not None:
                try:
                    self.iface.mainWindow().removeEventFilter(self._shortcut_filter)
                    canvas = self.iface.mapCanvas()
                    canvas.viewport().removeEventFilter(self._shortcut_filter)
                    canvas.removeEventFilter(self._shortcut_filter)
                except RuntimeError:
                    pass
                self._shortcut_filter = None
        except (RuntimeError, AttributeError):
            pass

        # 1. Disconnect ALL signals FIRST to prevent callbacks on partially-cleaned state
        if self.dock_widget:
            try:
                self.dock_widget.cleanup_signals()
            except (TypeError, RuntimeError, AttributeError):
                pass
            try:
                self.dock_widget.layer_combo.layerChanged.disconnect(self._on_layer_combo_changed)
            except (TypeError, RuntimeError, AttributeError):
                pass
            # Disconnect all dock widget signals connected in initGui()
            _dock_signals = [
                (self.dock_widget.install_requested, self._on_install_requested),
                (self.dock_widget.cancel_install_requested, self._on_cancel_install),
                (self.dock_widget.start_segmentation_requested, self._on_start_segmentation),
                (self.dock_widget.save_polygon_requested, self._on_save_polygon),
                (self.dock_widget.export_layer_requested, self._on_export_layer),
                (self.dock_widget.undo_requested, self._on_undo),
                (self.dock_widget.stop_segmentation_requested, self._on_stop_segmentation),
                (self.dock_widget.refine_settings_changed, self._on_refine_settings_changed),
                (self.dock_widget.size_filter_changed, self._on_size_filter_changed),
                (self.dock_widget.settings_clicked, self._on_settings_clicked),
                (self.dock_widget.pairing_requested, self._on_pairing_requested),
                (self.dock_widget.pairing_cancel_requested, self._on_cancel_pairing),
                (self.dock_widget.visibilityChanged, self._on_dock_visibility_changed),
                (self.dock_widget.mode_changed, self._on_mode_changed),
                (self.dock_widget.auto_detect_requested, self._on_auto_detect_requested),
                (self.dock_widget.history_rerun_requested, self._on_history_rerun_requested),
                (self.dock_widget.history_reuse_prompt_requested,
                 self._on_history_reuse_prompt_requested),
                (self.dock_widget.zone_draw_requested, self._on_zone_draw_requested),
                (self.dock_widget.auto_step_changed, self._on_auto_step_changed),
                (self.dock_widget.auto_detail_changed, self._on_auto_detail_changed),
                (self.dock_widget.auto_prompt_committed, self._reseed_auto_detail_for_object),
                (self.dock_widget.auto_layer_combo.layerChanged, self._on_auto_layer_combo_changed),
                (self.dock_widget.auto_cancel_btn.clicked, self._on_auto_cancel_clicked),
                (self.dock_widget.auto_refine_changed, self._on_auto_refine_changed_debounced),
                (self.dock_widget.auto_export_requested, self._on_auto_export_clicked),
                (self.dock_widget.auto_retry_requested, self._on_auto_retry_clicked),
                (self.dock_widget.auto_review_exit_requested, self._on_auto_review_exit_clicked),
                (self.dock_widget.auto_display_mode_changed, self._on_auto_display_mode_changed),
                (self.dock_widget.auto_merge_override_requested,
                 self._on_auto_merge_override_requested),
                (self.dock_widget.auto_library_requested, self._on_auto_library_clicked),
                (self.dock_widget.auto_demo_requested, self._on_auto_demo_requested),
                (self.dock_widget.auto_refine_in_manual_requested, self._on_refine_in_manual_clicked),
                (self.dock_widget.back_to_review_requested, self._on_back_to_review_clicked),
                (self.dock_widget.handoff_edit_requested, self._on_handoff_edit_clicked),
                (self.dock_widget.handoff_delete_requested, self._on_handoff_delete_clicked),
                (self.dock_widget.auto_exit_requested, self._on_auto_exit_clicked),
                (self.dock_widget.auto_add_exemplar_requested, self._on_add_exemplar_requested),
                (self.dock_widget.auto_exemplar_retry_requested, self._on_auto_exemplar_retry_clicked),
                (self.dock_widget.auto_exemplar_remove_requested, self._on_exemplar_remove_requested),
                (self.dock_widget.auto_zero_assist_clicked, self._on_auto_zero_assist_clicked),
                (self.dock_widget.auto_escape_pressed, self._on_auto_escape_shortcut),
                (self.dock_widget.auto_enter_pressed, self._route_enter),
                (self.dock_widget.auto_review_confidence_changed, self._on_auto_review_confidence_changed),
                (self.dock_widget.auto_review_confidence_preview, self._on_auto_review_confidence_preview),
                (self.dock_widget.auto_show_tiles_changed, self._on_auto_show_tiles_toggled),
                (self.dock_widget._auto_review_debounce_timer.timeout,
                 self._on_auto_review_refine_debounced),
            ]
            for sig, slot in _dock_signals:
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError, AttributeError):
                    pass
            # Stop timers before disconnection
            try:
                self.dock_widget._progress_timer.stop()
                self.dock_widget._refine_debounce_timer.stop()
                self.dock_widget._auto_review_debounce_timer.stop()
                self.dock_widget._auto_prompt_debounce_timer.stop()
            except (AttributeError, RuntimeError):
                pass
        try:
            if self.map_tool:
                self.map_tool.positive_click.disconnect(self._on_positive_click)
                self.map_tool.negative_click.disconnect(self._on_negative_click)
                self.map_tool.double_click.disconnect(self._on_canvas_double_click)
                self.map_tool.cursor_moved.disconnect(self._on_handoff_cursor_moved)
                self.map_tool.tool_deactivated.disconnect(self._on_tool_deactivated)
                self.map_tool.undo_requested.disconnect(self._on_undo)
                self.map_tool.save_polygon_requested.disconnect(self._on_save_polygon)
                self.map_tool.export_layer_requested.disconnect(self._on_export_layer)
                self.map_tool.stop_segmentation_requested.disconnect(self._on_stop_segmentation)
        except (TypeError, RuntimeError, AttributeError):
            pass

        # 2. Cleanup predictor subprocess (with timeout to avoid blocking unload)
        if self.predictor:
            import threading
            pred = self.predictor
            self.predictor = None
            t = threading.Thread(target=lambda: pred.cleanup(), daemon=True)
            t.start()
            t.join(timeout=8)
            if t.is_alive():
                QgsMessageLog.logMessage(
                    "Predictor cleanup did not finish within 8s",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning
                )

        # 2b. Cancel the QgsTask-based requests (cancel, never terminate). These
        # run network I/O; QThread.terminate() on a wedged socket crashes QGIS.
        self._cancel_pairing_worker()
        self._pairing_worker = None
        self._pairing_cancel_task = None
        self._cancel_task("_key_revalidate_task")
        self._cancel_task("_config_prefetch_task")
        self._cancel_task("_catalog_prefetch_task")
        self._cancel_task("_usage_fetch_task")
        self._cancel_task("_warmup_task")
        self._cancel_task("_auto_run_plan_task")
        self._cancel_task("_auto_token_task")

        # 3. Disconnect worker signals before termination to prevent callbacks on deleted UI
        _qthread_workers = [
            self.deps_install_worker, self.download_worker, self._verify_worker,
            getattr(self, "_predictor_worker", None),
            getattr(self, "_startup_check_worker", None),
            getattr(self, "_device_info_worker", None),
        ]
        for worker in _qthread_workers:
            if worker:
                try:
                    if hasattr(worker, "progress"):
                        worker.progress.disconnect()
                except (TypeError, RuntimeError):
                    pass
                try:
                    # `done` is the app-level completion signal (see
                    # background_workers); the built-in QThread.finished stays
                    # connected so a parked worker can release itself.
                    if hasattr(worker, "done"):
                        worker.done.disconnect()
                except (TypeError, RuntimeError):
                    pass

        # 4. Stop workers. Cooperatively cancel first: the install-family workers
        # run subprocess-based venv/pip installs, and a hard terminate() kills the
        # Python thread while its child process keeps running, risking a
        # half-written venv and an orphaned installer. Only terminate() as a last
        # resort, after a bounded wait for the cancel to unwind.
        for worker in _qthread_workers:
            if worker and worker.isRunning() and hasattr(worker, "cancel"):
                try:
                    worker.cancel()
                except (RuntimeError, AttributeError):
                    pass
        for worker in _qthread_workers:
            if worker and worker.isRunning():
                try:
                    # A cancellable worker gets a bounded chance to finish cleanly
                    # before the hard stop.
                    if hasattr(worker, "cancel") and worker.wait(3000):
                        continue
                    worker.terminate()
                    if not worker.wait(5000):
                        # Still blocked in a long network/subprocess call.
                        # Never null a running QThread (that GC-deletes the C++
                        # object mid-run and hard-aborts QGIS): park the last
                        # reference until its finished signal fires, mirroring
                        # the auto worker path. (See park_orphaned_worker.)
                        park_orphaned_worker(worker)
                except RuntimeError:
                    pass
        self.deps_install_worker = None
        self.download_worker = None
        self._verify_worker = None
        self._predictor_worker = None
        self._startup_check_worker = None
        self._device_info_worker = None

        # 5. Disconnect action signal and remove menu/toolbar
        try:
            self.action.triggered.disconnect(self.toggle_dock_widget)
        except (TypeError, RuntimeError, AttributeError):
            pass

        from .terralab_menu import remove_from_plugins_menu, remove_plugin_from_menu
        try:
            remove_from_plugins_menu(self.iface, self.action)
        except (RuntimeError, AttributeError):
            pass
        ai_edit_action = getattr(self, "ai_edit_action", None)
        if ai_edit_action is not None:
            try:
                remove_from_plugins_menu(self.iface, ai_edit_action)
            except (RuntimeError, AttributeError):
                pass
        if self.terralab_menu:
            try:
                remove_plugin_from_menu(
                    self.terralab_menu, self.action, self.iface.mainWindow())
            except (RuntimeError, AttributeError):
                pass
            ai_edit_action = getattr(self, "ai_edit_action", None)
            if ai_edit_action is not None:
                try:
                    remove_plugin_from_menu(
                        self.terralab_menu, ai_edit_action, self.iface.mainWindow())
                except (RuntimeError, AttributeError):
                    pass
            self.terralab_menu = None

        from .terralab_toolbar import remove_action_from_toolbar
        if self.terralab_toolbar:
            try:
                remove_action_from_toolbar(
                    self.terralab_toolbar, self.action, self.iface.mainWindow())
            except (RuntimeError, AttributeError):
                pass
            ai_edit_action = getattr(self, "ai_edit_action", None)
            if ai_edit_action is not None:
                try:
                    remove_action_from_toolbar(
                        self.terralab_toolbar, ai_edit_action, self.iface.mainWindow())
                except (RuntimeError, AttributeError):
                    pass
            self.terralab_toolbar = None
        self.ai_edit_action = None

        # 6. Remove dock widget
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None

        # 7. Clear markers and unset map tool
        if self.map_tool:
            try:
                self.map_tool.clear_markers()
            except (RuntimeError, AttributeError):
                pass
            try:
                if self.iface.mapCanvas().mapTool() == self.map_tool:
                    self.iface.mapCanvas().unsetMapTool(self.map_tool)
            except RuntimeError:
                pass
            self.map_tool = None

        # 8. Remove rubber bands safely
        self._safe_remove_rubber_band(self.mask_rubber_band)
        self.mask_rubber_band = None

        for rb in self.saved_rubber_bands:
            self._safe_remove_rubber_band(rb)
        self.saved_rubber_bands = []
        self._remove_handoff_layers()  # handoff seed layers

        # 9. Stop any running auto detection worker, then tear down Pro auto mode.
        # _stop_auto_detection keeps the worker reference (the thread is winding
        # down its last network call in the background), so join it here before
        # the plugin is destroyed, and drop its last signal so a late emission
        # cannot call back into a half-torn-down plugin (shutdown-crash guard).
        self._stop_auto_detection()
        auto_worker = self._auto_worker
        if auto_worker is not None:
            try:
                auto_worker.cancelled.disconnect(self._on_auto_cancelled)
            except (TypeError, RuntimeError):
                pass
            try:
                still_running = auto_worker.isRunning() and not auto_worker.wait(5000)
            except RuntimeError:
                still_running = False
            if still_running:
                # The thread is blocked in a long network call (up to 110 s
                # direct-submit timeout). Never delete a running QThread:
                # park the last reference until finished fires (the park
                # helper also handles a thread that finished in the gap).
                park_orphaned_worker(auto_worker)
            self._auto_worker = None
        self._drop_auto_tile_bridge()
        self._teardown_auto_mode()
        # The usage/warmup/key-revalidate/config requests are QgsTasks now and
        # are cancelled cooperatively above (step 2b); nothing to wait on here.

        # 10. Disconnect log collector signal
        stop_log_collector()

    def _ensure_dock_widget(self):
        """Create the dock widget and register it with QGIS (idempotent)."""
        if self._dock_created:
            return
        self._dock_created = True

        # Fresh telemetry session id so this dock's events group together.
        try:
            from ..core.telemetry import new_session
            new_session()
        except Exception:
            pass  # nosec B110

        self.dock_widget = AISegmentationDockWidget(self.iface.mainWindow())

        self.dock_widget.install_requested.connect(self._on_install_requested)
        self.dock_widget.cancel_install_requested.connect(self._on_cancel_install)
        self.dock_widget.start_segmentation_requested.connect(self._on_start_segmentation)
        self.dock_widget.save_polygon_requested.connect(self._on_save_polygon)
        self.dock_widget.export_layer_requested.connect(self._on_export_layer)
        self.dock_widget.undo_requested.connect(self._on_undo)
        self.dock_widget.stop_segmentation_requested.connect(self._on_stop_segmentation)
        self.dock_widget.refine_settings_changed.connect(self._on_refine_settings_changed)
        self.dock_widget.size_filter_changed.connect(self._on_size_filter_changed)
        self.dock_widget.settings_clicked.connect(self._on_settings_clicked)
        self.dock_widget.pairing_requested.connect(self._on_pairing_requested)
        self.dock_widget.pairing_cancel_requested.connect(self._on_cancel_pairing)
        self.dock_widget.layer_combo.layerChanged.connect(self._on_layer_combo_changed)
        self.dock_widget.mode_changed.connect(self._on_mode_changed)
        self.dock_widget.auto_detect_requested.connect(self._on_auto_detect_requested)
        self.dock_widget.auto_library_requested.connect(self._on_auto_library_clicked)
        self.dock_widget.auto_demo_requested.connect(self._on_auto_demo_requested)
        self.dock_widget.history_rerun_requested.connect(self._on_history_rerun_requested)
        self.dock_widget.history_reuse_prompt_requested.connect(
            self._on_history_reuse_prompt_requested)
        self.dock_widget.zone_draw_requested.connect(self._on_zone_draw_requested)
        self.dock_widget.auto_step_changed.connect(self._on_auto_step_changed)
        self.dock_widget.auto_detail_changed.connect(self._on_auto_detail_changed)
        self.dock_widget.auto_prompt_committed.connect(self._reseed_auto_detail_for_object)
        self.dock_widget.auto_layer_combo.layerChanged.connect(
            self._on_auto_layer_combo_changed)
        # Cancel button (dock stub is a pass; we wire the real handler here).
        self.dock_widget.auto_cancel_btn.clicked.connect(self._on_auto_cancel_clicked)
        # Auto review panel signals (plan #78 round 5).
        self.dock_widget.auto_refine_changed.connect(self._on_auto_refine_changed_debounced)
        self.dock_widget.auto_export_requested.connect(self._on_auto_export_clicked)
        self.dock_widget.auto_retry_requested.connect(self._on_auto_retry_clicked)
        self.dock_widget.auto_review_exit_requested.connect(self._on_auto_review_exit_clicked)
        self.dock_widget.auto_display_mode_changed.connect(self._on_auto_display_mode_changed)
        self.dock_widget.auto_merge_override_requested.connect(
            self._on_auto_merge_override_requested)
        self.dock_widget.auto_refine_in_manual_requested.connect(
            self._on_refine_in_manual_clicked)
        self.dock_widget.back_to_review_requested.connect(self._on_back_to_review_clicked)
        # Handoff state-card actions (Edit shape / Remove).
        self.dock_widget.handoff_edit_requested.connect(self._on_handoff_edit_clicked)
        self.dock_widget.handoff_delete_requested.connect(self._on_handoff_delete_clicked)
        self.dock_widget.auto_exit_requested.connect(self._on_auto_exit_clicked)
        # Visual exemplar controls (+ Example / + Exclude / chip remove).
        self.dock_widget.auto_add_exemplar_requested.connect(self._on_add_exemplar_requested)
        self.dock_widget.auto_exemplar_retry_requested.connect(
            self._on_auto_exemplar_retry_clicked)
        self.dock_widget.auto_exemplar_remove_requested.connect(self._on_exemplar_remove_requested)
        self.dock_widget.auto_zero_assist_clicked.connect(self._on_auto_zero_assist_clicked)
        self.dock_widget.auto_escape_pressed.connect(self._on_auto_escape_shortcut)
        self.dock_widget.auto_enter_pressed.connect(self._route_enter)
        self.dock_widget.auto_review_confidence_changed.connect(
            self._on_auto_review_confidence_changed)
        self.dock_widget.auto_review_confidence_preview.connect(
            self._on_auto_review_confidence_preview)
        self.dock_widget.auto_show_tiles_changed.connect(self._on_auto_show_tiles_toggled)
        self.dock_widget._auto_review_debounce_timer.timeout.connect(
            self._on_auto_review_refine_debounced)

        # Environment checks (venv scan, checkpoint, key revalidation) run
        # only once the dock is actually seen: toolbar click, the
        # install/upgrade auto-open, or QGIS restoring the dock at startup.
        # Connect BEFORE addDockWidget: when QGIS re-docks an already-open
        # panel, visibilityChanged fires during addDockWidget itself.
        self._first_time_setup_done = False
        # Separate guard for the local env (deps + checkpoint) check: Automatic
        # mode skips it, so it must NOT be consumed by the shared
        # _first_time_setup_done flag, or switching to Interactive later leaves
        # the install prompt inert (no dependency status ever set).
        self._interactive_setup_done = False
        self.dock_widget.visibilityChanged.connect(self._on_dock_visibility_changed)
        self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_widget)
        self._initialized = True
        self._setup_done = True
        if self.dock_widget.isVisible():
            self._on_dock_visibility_changed(True)

    def _on_dock_visibility_changed(self, visible: bool):
        if not visible or self._first_time_setup_done:
            return
        self._first_time_setup_done = True
        # One tick so the dock paints before the workers spin up.
        from qgis.PyQt.QtCore import QTimer

        from .ai_segmentation_dockwidget import Mode
        # Prefetch the server config (Automatic kill-switch + tutorial URL) off
        # the GUI thread so the synchronous getters never block on the network.
        QTimer.singleShot(0, self._prefetch_server_config)
        # Warm the segment-library catalogue cache now, well before the user
        # clicks Library, so the gallery never blocks the GUI on the network.
        QTimer.singleShot(0, self._prefetch_segment_catalog)
        if self.dock_widget and self.dock_widget._mode == Mode.AUTOMATIC:
            # Automatic mode needs no local install - only key revalidation.
            QTimer.singleShot(0, self._refresh_activation_async)
            return
        self._interactive_setup_done = True
        QTimer.singleShot(0, self._do_first_time_setup)

    def _prefetch_server_config(self) -> None:
        """Fetch the product config once, off the GUI thread, into the cache.

        get_server_config() is cache-only (it must never block the GUI), so this
        hidden task is what actually populates it. Fails open: an empty cache
        keeps Automatic mode available and the tutorial-URL fallback in place.
        """
        if self._config_prefetch_task is not None and self._config_prefetch_task.is_active():
            return
        from qgis.core import QgsApplication

        from ..api.terralab_client import TerraLabClient
        from ..core.activation_manager import PRODUCT_ID
        from ..workers.generic_request_task import GenericRequestTask
        client = TerraLabClient()
        self._config_prefetch_task = GenericRequestTask(
            tr("Loading AI Segmentation settings"),
            lambda: client.get_config(PRODUCT_ID),
            hidden=True,
        )
        self._config_prefetch_task.succeeded.connect(self._on_config_prefetched)
        self._config_prefetch_task.failed.connect(self._on_config_prefetch_failed)
        QgsApplication.taskManager().addTask(self._config_prefetch_task)

    def _on_config_prefetched(self, config: object) -> None:
        self._config_prefetch_task = None
        if isinstance(config, dict):
            from ..core.activation_manager import set_cached_config
            set_cached_config(config)

    def _on_config_prefetch_failed(self, message: str, code: str) -> None:
        self._config_prefetch_task = None
        self._notify_connection_issue(code, message)

    def _prefetch_segment_catalog(self) -> None:
        """Force-refresh the segment-library catalogue into its QSettings cache,
        off the GUI thread. The library dialog reads cache-only
        (``cached_or_offline_catalog``), so this is what keeps it fresh without
        ever stalling the UI on ``/api/ai-segmentation/presets``. Fails open: a
        cold cache just shows the bundled offline catalogue."""
        if self._catalog_prefetch_task is not None and self._catalog_prefetch_task.is_active():
            return
        from qgis.core import QgsApplication

        from ..core.presets.segmentation_presets_client import fetch_catalog
        from ..workers.generic_request_task import GenericRequestTask
        self._catalog_prefetch_task = GenericRequestTask(
            tr("Loading segment library"),
            lambda: fetch_catalog(force=True),
            hidden=True,
        )
        self._catalog_prefetch_task.succeeded.connect(self._on_catalog_prefetched)
        self._catalog_prefetch_task.failed.connect(
            lambda *_a: setattr(self, "_catalog_prefetch_task", None))
        QgsApplication.taskManager().addTask(self._catalog_prefetch_task)

    def _on_catalog_prefetched(self, _result: object) -> None:
        # The side effect (a warm QSettings cache) is all we need; just release
        # the task ref.
        self._catalog_prefetch_task = None

    def _ensure_interactive_setup(self) -> None:
        """Run the local dependency/checkpoint check the first time Interactive
        mode is shown.

        If the dock first opened in Automatic mode the env check was skipped
        (cloud-only), leaving the Install prompt inert with no Install button.
        Switching to Interactive must trigger the check once so the user can
        actually install or start.
        """
        if self._interactive_setup_done:
            return
        self._interactive_setup_done = True
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, self._do_first_time_setup)
