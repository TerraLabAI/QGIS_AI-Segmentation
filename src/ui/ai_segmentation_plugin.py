from __future__ import annotations

import os
import sys
import time
from functools import lru_cache
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRectangle,
)
from qgis.gui import QgisInterface, QgsRubberBand
from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QMenu

from ..core.i18n import tr
from ..core.log_scrub import start_log_collector, stop_log_collector
from ..core.prompt_manager import FrozenCropSession, PromptManager
from ..core.qt_compat import PolygonGeometry, QAction
from ..core.review_defaults import (
    AUTO_DEFAULT_CONFIDENCE as _AUTO_DEFAULT_CONFIDENCE,
)
from ..core.review_defaults import (
    REFINE_CLEAN_DEFAULT,
    REFINE_EXPAND_DEFAULT,
    REFINE_FILL_HOLES_DEFAULT,
    REFINE_FILL_HOLES_MAX_M2_DEFAULT,
    REFINE_MAX_SIZE_M2_DEFAULT,
    REFINE_MIN_SIZE_M2_DEFAULT,
    REFINE_ORTHO_DEFAULT,
    REFINE_POINTS_PCT_DEFAULT,
    REFINE_SIMPLIFY_DEFAULT,
    REFINE_SMOOTH_DEFAULT,
)
from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool
from .canvas_palette import PENDING_FILL, PENDING_STROKE
from .plugin.auto_correct import AutoCorrectMixin
from .plugin.auto_detail_window import AutoDetailWindowMixin
from .plugin.auto_exemplar_grouping import AutoExemplarGroupingMixin
from .plugin.auto_finalize_steps import AutoFinalizeStepsMixin
from .plugin.auto_flow import AutoFlowMixin
from .plugin.auto_imagery_guard import AutoImageryGuardMixin
from .plugin.auto_lifecycle import AutoLifecycleMixin
from .plugin.auto_object_build import AutoObjectBuildMixin
from .plugin.auto_results import AutoResultsMixin
from .plugin.auto_review import AutoReviewMixin
from .plugin.auto_review_display import AutoReviewDisplayMixin
from .plugin.auto_review_geometry import AutoReviewGeometryMixin
from .plugin.auto_review_open import AutoReviewOpenMixin
from .plugin.auto_review_params import AutoReviewParamsMixin
from .plugin.auto_run import AutoRunMixin
from .plugin.auto_run_terminal import AutoRunTerminalMixin
from .plugin.auto_shape_edit import AutoShapeEditMixin
from .plugin.auto_shape_overrides import AutoShapeOverridesMixin
from .plugin.auto_zone import AutoZoneMixin
from .plugin.bridge_isolation import BridgeIsolationMixin
from .plugin.correct_ai_route import CorrectAiRouteMixin
from .plugin.correct_focus import CorrectFocusMixin
from .plugin.credits_watch import AutoCreditsWatchMixin
from .plugin.demo_scene import DemoSceneMixin
from .plugin.env_setup import EnvSetupMixin
from .plugin.exemplars import ExemplarsMixin
from .plugin.handoff_seed_layers import HandoffSeedLayersMixin
from .plugin.handoff_shape import HandoffShapeMixin
from .plugin.local_ai_install_lock import LocalAiInstallLockMixin
from .plugin.local_ai_warm import LocalAiWarmMixin
from .plugin.manual_add import ManualAddMixin
from .plugin.manual_cloud_predictor import ManualCloudPredictorMixin
from .plugin.manual_crop_window import ManualCropWindowMixin
from .plugin.manual_crops import ManualCropsMixin
from .plugin.manual_handoff import ManualHandoffMixin
from .plugin.manual_hover_preview import ManualHoverPreviewMixin
from .plugin.manual_object_billing import ManualObjectBillingMixin
from .plugin.manual_predict import ManualPredictMixin
from .plugin.manual_shape_cache import ManualShapeCacheMixin
from .plugin.manual_workflow import ManualWorkflowMixin
from .plugin.qgis_edit_bridge import QgisEditBridgeMixin
from .plugin.qgis_edit_tool_messages import QgisEditToolMessagesMixin
from .plugin.shared import park_orphaned_worker


class AISegmentationPlugin(
    AutoFlowMixin,
    AutoCreditsWatchMixin,
    AutoDetailWindowMixin,
    AutoCorrectMixin,
    LocalAiWarmMixin,
    LocalAiInstallLockMixin,
    CorrectAiRouteMixin,
    ManualCloudPredictorMixin,
    AutoShapeEditMixin,
    AutoShapeOverridesMixin,
    CorrectFocusMixin,
    QgisEditBridgeMixin,
    BridgeIsolationMixin,
    QgisEditToolMessagesMixin,
    AutoRunMixin,
    AutoImageryGuardMixin,
    AutoResultsMixin,
    AutoReviewDisplayMixin,
    HandoffSeedLayersMixin,
    AutoRunTerminalMixin,
    AutoExemplarGroupingMixin,
    AutoObjectBuildMixin,
    AutoReviewParamsMixin,
    AutoReviewGeometryMixin,
    AutoFinalizeStepsMixin,
    AutoReviewOpenMixin,
    AutoReviewMixin,
    ManualHandoffMixin,
    ManualHoverPreviewMixin,
    ManualAddMixin,
    HandoffShapeMixin,
    ExemplarsMixin,
    AutoLifecycleMixin,
    AutoZoneMixin,
    DemoSceneMixin,
    EnvSetupMixin,
    ManualObjectBillingMixin,
    ManualWorkflowMixin,
    ManualCropsMixin,
    ManualCropWindowMixin,
    ManualShapeCacheMixin,
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
        # Where an on-device predictor waits while the review's AI fix is served
        # over the network. Manual gets it back without a second model load.
        self._local_predictor_held = None
        # What the Semi-Auto session on TerraLab's servers has already paid for.
        # None means no session, or a session answered on this computer, and
        # either way nothing about the account is read or spent.
        self._manual_credit_ledger = None
        # Charges still in flight. Held only so the task manager's task is not
        # collected under it; each one drops itself when it finishes.
        self._manual_charge_tasks = []
        self.prompts = PromptManager()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        # The Semi-Auto session's two shape memos (ManualShapeCacheMixin): the
        # mask stage, and the finished outline the preview draws and Save and
        # Export keep. Both keyed on the mask OBJECT and the crop OBJECT, which
        # every writer rebinds rather than mutates, so a new prediction misses
        # them with no explicit invalidation.
        self._mask_preview_memo = None
        self._manual_outline_memo = None
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
        # Debounce for the speculative crop warm-up (hover on the Correct step,
        # and selection inside a fix session). Which windows are encoded or in
        # flight is tracked by _ensure_manual_encode_state, not here.
        self._handoff_prewarm_timer = None
        self._correct_hover_warm_timer = None
        # Synthetic det_id sequence for hand-drawn/legacy entries (see
        # _next_handoff_det_id).
        self._handoff_det_id_seq = None
        # det_ids the live fix session imported, so the fold's deletion diff
        # can never speak for a detection the review filters were hiding.
        self._handoff_imported_det_ids: set[int] = set()
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
        # Share of its own points an outline may keep (100 = dial off). Same
        # control and same unit as the Automatic review's Points dial.
        self._refine_points_pct = REFINE_POINTS_PCT_DEFAULT
        self._refine_smooth = REFINE_SMOOTH_DEFAULT
        # Clean edges (morphological opening, px; 0 = off). Mirrors the
        # Automatic review's control; a Refine-in-Manual handoff seeds it from
        # the review's value.
        self._refine_clean = REFINE_CLEAN_DEFAULT
        self._refine_expand = REFINE_EXPAND_DEFAULT
        self._refine_fill_holes = REFINE_FILL_HOLES_DEFAULT
        # Fill holes SMALLER than this ground area (m2); 0 = fill every hole,
        # the behaviour before the threshold existed.
        self._refine_fill_holes_max_m2 = REFINE_FILL_HOLES_MAX_M2_DEFAULT
        self._refine_ortho = REFINE_ORTHO_DEFAULT
        self._refine_min_area = 200  # overridden by _compute_auto_min_area() × 2
        # User Min/Max size window in ground m2 (0 = off); a Refine-in-Manual
        # handoff seeds these from the Automatic review's size filters.
        self._refine_min_size_m2 = REFINE_MIN_SIZE_M2_DEFAULT
        self._refine_max_size_m2 = REFINE_MAX_SIZE_M2_DEFAULT

        self._is_non_georeferenced_mode = False  # Track if current layer is non-georeferenced
        self._is_online_layer = False  # Track if current layer is online (WMS, XYZ, etc.)
        self._disjoint_warning_shown = False
        # Said once per object, like the disjoint one above: a user correcting a
        # hard shape would otherwise read it on every click of that shape.
        self._unsure_warning_shown = False
        # One automatic venv repair per QGIS session: a second one would never
        # cure what the first could not, and the retry loop is a dead end.
        self._rasterio_repair_attempted = False
        # Crop-error dialogs already shown this session, keyed by
        # (raster path, error code): an unreadable file stays unreadable, so
        # re-clicking must not re-raise the same modal on every attempt.
        self._crop_errors_reported = set()

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
        # Which half the last install ran: everything, or the light packages
        # Automatic loads. Decided once in _on_install_requested, then read by
        # the workers and by the review's env gate, which must not take a
        # light install in flight for a local model on its way.
        self._install_includes_local_model = True
        # Cached "local venv is installed" flag (from the startup check). Drives
        # the Refine-in-Manual env gate so an uninstalled env shows the install
        # dialog instead of trapping the user in "Preparing Manual mode".
        self._env_ready = False
        # Network requests run as hidden QgsTasks (cooperative cancel, never
        # QThread.terminate which crashes QGIS when the socket is wedged).
        self._key_revalidate_task = None  # GenericRequestTask | None
        self._config_prefetch_task = None  # GenericRequestTask | None
        # Repeats that fetch for the rest of the session, so a server dial
        # reaches a user who leaves QGIS open rather than waiting for their
        # next start (see _arm_config_refresh).
        self._config_refresh_timer = None  # QTimer | None
        # Keeps the Automatic balance in step with the account while QGIS stays
        # open, so credits granted (or a subscription just paid for) do not wait
        # for a mode switch to show up (see AutoCreditsWatchMixin).
        self._credits_watch_timer = None  # QTimer | None
        self._credits_activation_relay = None  # MainWindowActivationRelay | None
        self._credits_last_read_unix: float = 0.0
        self._plan_upgrade_announced = False
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
        # Background render of the Recent card's picture, held while it runs
        # (see _render_history_thumbnail).
        self._history_thumb_job = None  # QgsMapRendererSequentialJob | None
        # Review correction loop state (journal, sets, queue, batch flags).
        self._init_auto_correct_state()
        # QGIS digitizing bridge state (native editing on the review layer).
        self._init_qgis_bridge_state()
        # The subset and the context layer that hold a Manual session to one
        # polygon (see plugin/bridge_isolation.py).
        self._init_bridge_isolation_state()
        # True once the user moves the detail slider themselves: the debounced
        # object-aware re-seed then stops overriding their manual pick, for the
        # prompt recorded below. Reset whenever a new zone is drawn (a fresh
        # zone = a fresh default) or the prompt changes (a new object = a new
        # sizing problem, the auto seed owns the slider again).
        self._auto_detail_user_locked = False
        self._auto_detail_lock_prompt = ""
        # The level the plugin itself last recommended for the current zone and
        # prompt, as the slider ended up holding it. None means no seed stands,
        # so the started event cannot say where the run's level came from.
        self._auto_detail_seeded: int | None = None
        self._zone_rubber_band: QgsRubberBand | None = None
        self._zone_delete_badge = None  # ZoneDeleteBadge | None
        self._zone_badge_filter = None  # ZoneBadgeClickFilter | None
        self._zone_escape_filter = None  # ZoneEscapeFilter | None
        self._zone_grid_rubber_band = None  # QgsRubberBand | None (tile grid preview)
        # Clipped grid geometries already built, keyed by (zone WKB, cols, rows).
        # The Detail slider is quantised, so neighbouring positions land on the
        # same grid and used to re-clip up to 800 cells against the zone for a
        # picture the user is already looking at.
        self._zone_grid_geom_cache = {}
        # Ground the running detection is re-reading finer, whose objects are
        # withheld until it lands (see auto_results._on_auto_rescan_state).
        self._auto_rescan_band = None       # QgsRubberBand | None
        self._auto_rescan_rects: dict = {}  # base tile idx -> bbox in run CRS
        # True from the Detect click until the flow is back on the pre-Detect
        # setup screen. The grid is a setup aid, and this latch is what keeps it
        # off the canvas for the whole run and the review that follows, because
        # every other signal for "a run owns the canvas" is transient (see
        # AutoZoneMixin._tile_grid_allowed).
        self._auto_grid_suppressed = False
        self._tile_manager = None  # TileManager | None

        # Visual exemplars ("draw one example, find all"). The store
        # holds the example boxes (canvas CRS) + labels; persistent rubber bands
        # keep them visible on the map; the draw tool is armed on demand.
        from ..core.exemplar_store import ExemplarStore
        self._auto_exemplar_store = ExemplarStore()
        self._exemplar_maptool = None  # BoxDrawMapTool | None (example draw)
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
        # False = merge tile-split objects (map continuous features). Decided
        # per object type at run start, never surfaced as a user control.
        self._auto_merge_separate: bool = True
        # How the merge policy was decided: "prompt" (object token) or "signal"
        # (exemplar-only auto count-vs-map from the run's own masks).
        # Telemetry only.
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
        # Whether THIS run keeps its per-tile fragments. Only an exemplar-only
        # run needs them: it re-merges them SEPARATE once the map-likeness
        # signal has read every fragment (see _resolve_exemplar_finalize_ided).
        self._auto_retain_raw: bool = False
        # Whether the worker was asked for raw (un-gated, un-pre-stitched)
        # fragments this run. Exemplar-only runs only; resolved in
        # _start_auto_detection.
        self._auto_collect_raw: bool = False
        self._auto_selection_layer = None  # QgsVectorLayer | None (in-progress results)
        # Review display colour mode: 'normal' / 'outline' / 'confidence' /
        # 'random' (visual only; never touches geometry, filters or export).
        # Random by default, matching the dock combo; re-seeded to Random for
        # every NEW review (_seed_review_display_mode).
        self._auto_display_mode = "random"
        self._auto_run_id: str | None = None
        # run_id of the last run whose auto-default export was archived at
        # review-open, so that background upload fires at most once per run
        # (Finish still uploads the reviewed set later; same run_id, latest wins).
        self._auto_default_export_run_id: str | None = None
        self._auto_run_ctx: dict | None = None     # inputs of the active run
        # The per-run lambda wired to the worker's cancelled signal, kept so
        # unload can disconnect the exact slot it connected.
        self._auto_cancelled_slot = None
        # The line a quota-refused run leaves for the setup step it lands on,
        # so the generic "no matches" note never covers the real reason.
        self._auto_quota_stop_banner: str | None = None
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
        # Attribute filters carried by the run plan's optional prompt_rewrite
        # block ([{"attribute", "value"}, ...]). Informational this pass (stored
        # for a later honest-count UI, never used for filtering yet); cleared on
        # every prompt change and run reset alongside _auto_run_plan.
        self._auto_attribute_filters: list[dict[str, str]] = []
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
        # Shape-cleanup overrides a programmatic caller asked for, merged onto
        # the run's review preset and cleared when that run ends. None outside
        # such a call, which is what makes the panel's own behaviour untouched.
        self._auto_review_preset_overrides: dict | None = None
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
        # Arriving tile geometry is folded, gated and shaped on the live
        # stitcher thread, which owns the run's merger from the moment the
        # worker starts until the finalize takes it back. The GUI thread only
        # writes the objects it hands over, on a coalesced repaint tick.
        self._auto_stitcher = None  # LiveStitchThread | None
        self._auto_repaint_timer = None  # QTimer | None (coalesced live write)
        # The live preview paces itself on the canvas, not on a clock: a refresh
        # asked for while the previous one is still drawing KILLS it and starts
        # over, so on a big zone the map never finished a frame. These hold the
        # "data landed while it was drawing" flag and the one-shot connection to
        # the canvas's finished signal (see _repaint_live_layer).
        self._auto_live_repaint_pending = False
        self._auto_live_pacer_canvas = None
        # What one frame of that preview costs, measured, plus the cool-down it
        # buys the run (see _note_live_repaint_sent). A frame grows with the
        # objects found, and it is drawn on the machine that is folding tiles,
        # so past a few thousand objects the preview has to sit out its turn.
        self._auto_live_frame_s = 0.0
        self._auto_live_frame_started = 0.0
        self._auto_live_repaint_not_before = 0.0
        self._auto_live_cooldown_timer = None  # QTimer | None
        # Live preview provider mapping: merger keeper fid -> (provider_fid,
        # score). Lets the live tick add / change / delete only the objects the
        # stitcher reported, instead of rebuilding the layer. Reset per run via
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
        # Run refiners: shape-params key -> core.live_refine.LiveRefiner, the
        # run's server dials resolved once so the per-object refine only does
        # the object's own work. Reset with the reslice cache.
        self._review_live_refiners: dict = {}

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
        """Check if a raster layer is properly georeferenced. Kept as a method
        (MCP session setup and the mixins call it); logic lives in shared."""
        from .plugin.shared import is_layer_georeferenced
        return is_layer_georeferenced(layer)

    @staticmethod
    def _needs_canvas_render(layer) -> bool:
        """Whether this raster is rendered through QGIS instead of read as a file.

        True for the online services and for the local providers that hold no
        file (PostGIS raster, virtual raster). Every caller, Manual and
        Automatic alike, branches on exactly this: read the pixels from disk,
        or ask QGIS to draw them.
        """
        if layer is None:
            return False
        try:
            provider = layer.dataProvider()
            if provider is None:
                return False
            from ..core.feature_encoder import CANVAS_RENDERED_PROVIDERS
            return provider.name() in CANVAS_RENDERED_PROVIDERS
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
        # The crop covers a square GROUND region (core.layer_conventions.
        # ground_unit_aspect), so on a geographic raster the x and y axes
        # resolve at a different CRS-unit-per-pixel size. This value feeds a
        # direction-agnostic simplify, so use the finer of the two axes: the
        # coarser one would erase real geometry on its axis, the finer one
        # only leaves a few extra vertices on the other. On a projected
        # raster the two axes already agree, so this is exactly pixel_size.
        height_pixels = max(img_shape[0], 1)
        bbox_height = bbox[3] - bbox[2]
        if bbox_height != 0:
            pixel_size = min(pixel_size, bbox_height / height_pixels)
        # One crop pixel per unit, the same scale the Automatic review's
        # Simplify uses (shape_polygon_geometry keeps the twin of this line).
        return pixel_size * simplify_value

    def initGui(self):
        """Register everything this plugin puts into QGIS.

        A failure part way through used to leave the half that had already
        landed: a toolbar action pointing at a dead controller, a Processing
        provider whose algorithms answer to nothing, the log collector still
        attached. QGIS reports the error and carries on, so the cleanup has to
        happen here: undo what took, then let the error travel as before.
        """
        try:
            self._build_gui()
        except Exception:
            try:
                self.unload()
            except Exception:  # noqa: BLE001 -- the first error is the one to report
                pass  # nosec B110
            raise

    def _build_gui(self):
        from ..mcp_api import SegmentationMCPAPI
        self.mcp_api = SegmentationMCPAPI(self)

        # Publish a `terralab` module an outside agent can import, so a QGIS MCP
        # server that only offers code execution can still find this plugin.
        # Never fatal: the panel loads with or without it.
        try:
            from ..agent_bridge import register_product
            register_product("segmentation", self.mcp_api)
        except Exception as err:  # noqa: BLE001 -- discovery is a bonus, not the product
            QgsMessageLog.logMessage(
                f"Agent bridge not published: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )

        # Publish the algorithms to the Processing Toolbox, the modeler, batch
        # mode and every tool that drives Processing. Never fatal: a provider
        # that fails to register must not stop the panel from loading.
        self.processing_provider = None
        try:
            self._register_processing_provider()
        except Exception as err:  # noqa: BLE001 -- discovery is a bonus, not the product
            QgsMessageLog.logMessage(
                f"Processing provider not registered: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )

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
        # The preview that follows the cursor in Semi-Auto on the cloud engine.
        # A second slot rather than a branch inside the handoff one: the two
        # answer the same motion for unrelated reasons.
        self.map_tool.cursor_moved.connect(self._on_hover_cursor_moved)
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

        # A pointer left by a session that died with a review open: log where
        # the table sits and drop it. Nothing is shown and nothing is loaded.
        try:
            from ..core.run_autosave import log_and_clear_stale_pending
            log_and_clear_stale_pending(self._auto_run_id)
        except Exception:  # nosec B110
            pass

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

    def _register_processing_provider(self):
        """Add the TerraLab provider to the Processing registry.

        Imported here rather than at module level so plugin load stays light,
        and so a QGIS build without the Processing plugin enabled fails on this
        one call instead of on the import of the whole controller.
        """
        from qgis.core import QgsApplication

        from ..processing.segmentation_provider import (
            TERRALAB_PROVIDER_ID,
            TerraLabProcessingProvider,
        )
        registry = QgsApplication.processingRegistry()
        provider = TerraLabProcessingProvider()
        # addProvider DELETES the provider it was handed when the id is already
        # taken, so keeping the reference leaves a wrapper around a dead C++
        # object and the later removeProvider raises on it.
        if not registry.addProvider(provider):
            # Reloading the plugin can leave the previous provider behind with
            # its Python half collected: it answers to no id and lists no
            # algorithm, and it holds the name against us. Whoever reloaded
            # would have no algorithms until they restart QGIS, so take the id
            # back rather than stopping here. A fresh instance is needed
            # because the one above is already deleted.
            stale = registry.providerById(TERRALAB_PROVIDER_ID)
            if stale is not None:
                registry.removeProvider(stale)
            provider = TerraLabProcessingProvider()
            if not registry.addProvider(provider):
                self.processing_provider = None
                QgsMessageLog.logMessage(
                    "Processing provider not registered: the id "
                    f"'{TERRALAB_PROVIDER_ID}' is already taken.",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning
                )
                return
        self.processing_provider = provider

    def _unregister_processing_provider(self):
        """Remove the provider, so a reload does not leave two of them registered."""
        provider = getattr(self, "processing_provider", None)
        if provider is None:
            return
        from qgis.core import QgsApplication
        QgsApplication.processingRegistry().removeProvider(provider)
        self.processing_provider = None

    @staticmethod
    @lru_cache(maxsize=1)
    def _read_plugin_version() -> str:
        """Read the plugin version from metadata.txt (plugin root).

        Cached: the file cannot change while the plugin is loaded, and every
        export used to open and scan it again on the click the user is waiting
        on.
        """
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
        # A headless run blocks its caller inside a nested event loop, and this
        # call can be delivered from that very loop. Tearing the plugin down
        # under a live run frees objects its stack still holds, which takes
        # QGIS with it. Wait instead: the run finishes the unload on its way
        # out (see _run_auto_detect_headless).
        if getattr(self, "_auto_headless_run", False):
            self._unload_deferred = True
            QgsMessageLog.logMessage(
                "Unload deferred: a detection started from the API is still "
                "running", "AI Segmentation", level=Qgis.MessageLevel.Info)
            return
        self._unload_deferred = False
        # A successful export parks the drop of its crash-net copy on a timer
        # owned by the dock, and the dock is about to go. Run it here instead,
        # or the duplicate table stays on disk for good.
        parked_drop = getattr(self, "_pending_autosave_drop", None)
        if parked_drop is not None:
            try:
                parked_drop()
            except Exception:  # noqa: BLE001 -- unload must never raise
                pass  # nosec B110
        # First, because the registry outlives the plugin object: a provider
        # left behind keeps algorithms in the Toolbox that call a dead facade.
        try:
            self._unregister_processing_provider()
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # Stop advertising this plugin to outside agents once it is disabled.
        try:
            from ..agent_bridge import unregister_product
            unregister_product("segmentation")
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # Ship any queued telemetry before teardown (main thread here), then
        # stop the periodic flush: its timer has no parent, so nothing else
        # takes it down with the plugin.
        try:
            from ..core.telemetry import flush as _telemetry_flush
            _telemetry_flush()
        except Exception:
            pass  # nosec B110
        try:
            from ..core.telemetry import stop_flush_timer as _telemetry_stop_timer
            _telemetry_stop_timer()
        except Exception:
            pass  # nosec B110
        # Up here, not at the end: the flag behind it is a module global, so a
        # step below that raises used to leave the message log connected with an
        # unbounded buffer for the rest of the QGIS session, and the next reload
        # read a fresh False and connected a second time.
        try:
            stop_log_collector()
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # Same reason: the task list is a module global that a reload replaces,
        # so a task left queued resolves against a module dict nobody reads.
        try:
            from .plugin.run_export_upload import cancel_inflight_uploads
            cancel_inflight_uploads()
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # The crop read owns the encode lock through a dict the call below
        # clears, and step 3 still has to drop its `done`: take the worker now.
        _crop_read_worker = None
        try:
            _read = getattr(self, "_crop_read", None)
            if isinstance(_read, dict):
                _crop_read_worker = _read.get("worker")
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # The Semi-Auto encode holds an APPLICATION-GLOBAL busy cursor and arms
        # the lock watchdog. Nothing else in unload pops either, so a reload
        # during an encode left QGIS showing the hourglass until restart. Runs
        # here, while the dock is alive, because it clears the panel note too.
        try:
            self._invalidate_manual_encode()
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # The private copy of a rendered layer that the warm-ups read through.
        # Held by a module, so nothing else here would ever let it go.
        try:
            from ..core.online_layer_twin import release_online_layer_twin
            release_online_layer_twin()
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # Which basemaps refused to serve their tiles by number. Also module
        # state, and a network that was down at the wrong moment must not
        # outlive the session that hit it.
        try:
            from ..core.xyz_tile_fetch import forget_direct_tile_fetch_failures
            forget_direct_tile_fetch_failures()
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # Same for the raster held open for windowed reads: a module holds it,
        # so an unload that left it there would keep the file locked with no
        # plugin left to free it.
        try:
            from ..core.raster_dataset_cache import release_raster_datasets
            release_raster_datasets()
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # Roll back and restore first if a QGIS digitizing bridge is open, so
        # the user's project never keeps our snapping / topology / avoid-overlap
        # forced on after the plugin is gone (idempotent, never raises).
        self._abort_qgis_edit_bridge_if_active()
        # The abort above already clears it when a session was open. Repeated
        # here unconditionally because a subset left on the review layer would
        # show one polygon for the rest of the QGIS session, plugin or no
        # plugin, and this is the last chance to take it off.
        try:
            self._clear_bridge_isolation()
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # The waiting cursor is application-global, so it has to come off before
        # the flags below make the session look already gone.
        try:
            self._end_correct_wait()
        except Exception:  # noqa: BLE001 -- unload must never raise
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
        self._pending_refine_import = False
        self._handoff_source_layer = None
        # The AI-assisted Add flags ride the handoff: drop them too, or a
        # reloaded plugin instance could inherit a stale True.
        self._refine_add_mode_active = False
        self._ai_add_install_pending = False
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
        # A manual session follows the canvas CRS while it runs, and only the
        # session teardown drops that. Unload does not run it, so without this
        # a disabled or reloaded plugin leaves a dead controller wired to a
        # live canvas, and the next project CRS change calls into it. The stop
        # is idempotent, so a session that already ended costs nothing.
        try:
            self._stop_canvas_crs_watch()
        except (RuntimeError, AttributeError):
            pass
        # A commit right before unload may still be holding the canvas picture
        # for its redraw: give the map its normal update rate back, or the
        # user's canvas keeps the parked one after the plugin is gone.
        try:
            from .plugin.canvas_redraw_handover import release_map_picture_hold
            release_map_picture_hold(self.iface.mapCanvas())
        except (RuntimeError, AttributeError, ImportError):
            pass  # nosec B110
        # 0. Remove keyboard shortcut filter
        try:
            if self._shortcut_filter is not None:
                # One guard per target. Under a single try a raise on the first
                # left the other two installed, and the reference is dropped on
                # the next line whatever happened, so the filter went on
                # answering key presses with nothing left owning it.
                for target in (
                    lambda: self.iface.mainWindow(),
                    lambda: self.iface.mapCanvas().viewport(),
                    lambda: self.iface.mapCanvas(),
                ):
                    try:
                        target().removeEventFilter(self._shortcut_filter)
                    except (RuntimeError, AttributeError):
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
            # Disconnect every dock signal wired when the dock was created.
            # This list must stay the mirror image of the connects in
            # _ensure_dock_widget (and of the two _connect_auto_* helpers it
            # calls); tests/test_unload_teardown_guards.py fails the build if
            # a connect has no disconnect here. Each entry is tried on its own
            # so one stale connection cannot abort the rest of unload().
            # Building the list touches child widgets: if one has already
            # lost its C++ side there is nothing left to disconnect from it,
            # and unload still has to finish the rest of its steps.
            try:
                _dock_signals = [
                    (self.dock_widget.layer_combo.layerChanged, self._on_layer_combo_changed),
                    (self.dock_widget.manual_engine_changed, self._on_manual_engine_changed),
                    (self.dock_widget.install_requested, self._on_install_requested),
                    (self.dock_widget.cancel_install_requested, self._on_cancel_install),
                    (self.dock_widget.start_segmentation_requested, self._on_start_segmentation),
                    (self.dock_widget.save_polygon_requested, self._on_save_polygon),
                    (self.dock_widget.export_layer_requested, self._on_export_layer),
                    (self.dock_widget.undo_requested, self._on_undo),
                    (self.dock_widget.stop_segmentation_requested, self._on_stop_segmentation),
                    (self.dock_widget.refine_settings_changed, self._on_refine_settings_changed),
                    (self.dock_widget.size_filter_changed, self._on_size_filter_changed),
                    (self.dock_widget.fill_holes_size_changed,
                     self._on_fill_holes_size_changed),
                    (self.dock_widget.clean_edges_changed, self._on_clean_edges_changed),
                    (self.dock_widget.outline_budget_changed,
                     self._on_outline_budget_changed),
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
                    (self.dock_widget.auto_advanced_toggled, self._on_auto_advanced_toggled),
                    (self.dock_widget.auto_prompt_committed, self._reseed_auto_detail_for_object),
                    (self.dock_widget.auto_layer_combo.layerChanged, self._on_auto_layer_combo_changed),
                    (self.dock_widget.auto_cancel_btn.clicked, self._on_auto_cancel_clicked),
                    (self.dock_widget.auto_refine_changed, self._on_auto_refine_changed_debounced),
                    (self.dock_widget.auto_export_requested, self._on_auto_export_clicked),
                    (self.dock_widget.auto_retry_requested, self._on_auto_retry_guarded),
                    (self.dock_widget.auto_review_exit_requested, self._on_auto_review_exit_clicked),
                    (self.dock_widget.auto_display_mode_changed, self._on_auto_display_mode_changed),
                    (self.dock_widget.auto_library_requested, self._on_auto_library_clicked),
                    (self.dock_widget.auto_demo_requested, self._on_auto_demo_requested),
                    (self.dock_widget.auto_reshape_ai_requested, self._on_reshape_ai_requested),
                    (self.dock_widget.auto_reshape_done_requested, self._on_reshape_done),
                    (self.dock_widget.auto_correct_method_changed, self._on_correct_method_changed),
                    (self.dock_widget.auto_ai_add_requested, self._on_ai_add_requested),
                    (self.dock_widget.auto_ai_add_keep_requested, self._route_save_add_mode),
                    (self.dock_widget.auto_review_install_cancel_requested,
                     self._on_review_install_cancel_requested),
                    (self.dock_widget.auto_exit_requested, self._on_auto_exit_clicked),
                    (self.dock_widget.auto_add_exemplar_requested, self._on_add_exemplar_requested),
                    (self.dock_widget.auto_exemplar_remove_requested, self._on_exemplar_remove_requested),
                    (self.dock_widget.auto_zero_assist_clicked, self._on_auto_zero_assist_clicked),
                    (self.dock_widget.auto_escape_pressed, self._on_auto_escape_shortcut),
                    (self.dock_widget.auto_enter_pressed, self._on_auto_enter_pressed),
                    (self.dock_widget.auto_correct_undo_shortcut.activated,
                     self._on_auto_undo_pressed),
                    (self.dock_widget.auto_review_confidence_changed, self._on_auto_review_confidence_changed),
                    (self.dock_widget.auto_review_confidence_preview, self._on_auto_review_confidence_preview),
                    (self.dock_widget.auto_show_tiles_changed, self._on_auto_show_tiles_toggled),
                    (self.dock_widget.auto_edit_in_qgis_requested, self.enter_qgis_edit_bridge),
                    (self.dock_widget.auto_add_polygon_requested, self._on_add_polygon_requested),
                    (self.dock_widget.auto_qgis_bridge_done_requested,
                     self.finish_qgis_edit_bridge),
                    (self.dock_widget.auto_qgis_bridge_tool_requested,
                     self.activate_qgis_bridge_tool),
                    (self.dock_widget.auto_qgis_bridge_undo_requested,
                     self.undo_qgis_bridge_edit),
                    (self.dock_widget.auto_qgis_bridge_gesture_requested,
                     self._on_bridge_gesture_requested),
                    (self.dock_widget.auto_qgis_bridge_points_changed,
                     self._on_bridge_points_changed),
                    (self.dock_widget.auto_qgis_bridge_delete_requested,
                     self.delete_bridge_target_polygon),
                    (self.dock_widget._auto_review_debounce_timer.timeout,
                     self._on_auto_review_refine_debounced),
                    # Mirrors _connect_auto_correct_signals.
                    (self.dock_widget.auto_correction_undo_requested,
                     self._on_auto_correction_undo_requested),
                    (self.dock_widget.auto_correction_clear_requested,
                     self._on_auto_correction_clear_requested),
                    (self.dock_widget.auto_review_step_requested,
                     self._on_auto_review_step_requested),
                    (self.dock_widget.auto_correct_status_action_requested,
                     self._on_correct_status_action_requested),
                    # Mirrors _connect_auto_shape_edit_signals.
                    (self.dock_widget.auto_shape_edit_requested,
                     self._on_auto_shape_edit_requested),
                    (self.dock_widget.auto_remove_requested, self._on_remove_requested),
                    (self.dock_widget.auto_shape_only_changed, self._on_shape_only_changed),
                    (self.dock_widget.auto_shape_only_reset_requested,
                     self._on_shape_only_reset),
                ]
            except (TypeError, RuntimeError, AttributeError):
                _dock_signals = []
            for sig, slot in _dock_signals:
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError, AttributeError):
                    pass
            # Stop timers before disconnection. One try per timer: the ease
            # timer is built on first use and is None until then, and a single
            # shared try let that AttributeError skip every timer after it.
            for _timer_name in (
                "_progress_timer",
                "_refine_debounce_timer",
                "_auto_review_debounce_timer",
                "_auto_prompt_debounce_timer",
                "_visibility_debounce_timer",
                "_auto_progress_ease_timer",
            ):
                try:
                    _timer = getattr(self.dock_widget, _timer_name, None)
                    if _timer is not None:
                        _timer.stop()
                except (AttributeError, RuntimeError):
                    pass
        if self.map_tool:
            # One try per signal: a single stale connection used to abort the
            # whole block and leave the rest of the tool wired to a dead plugin.
            for sig_name, slot in (
                ("positive_click", self._on_positive_click),
                ("negative_click", self._on_negative_click),
                ("double_click", self._on_canvas_double_click),
                ("cursor_moved", self._on_handoff_cursor_moved),
                ("cursor_moved", self._on_hover_cursor_moved),
                ("tool_deactivated", self._on_tool_deactivated),
                ("undo_requested", self._on_undo),
                ("save_polygon_requested", self._on_save_polygon),
                ("export_layer_requested", self._on_export_layer),
                ("stop_segmentation_requested", self._on_stop_segmentation),
            ):
                try:
                    getattr(self.map_tool, sig_name).disconnect(slot)
                except (TypeError, RuntimeError, AttributeError):
                    pass

        # The hover preview owns a timer on the canvas, a filter on its
        # viewport and a canvas item. All three outlive this controller unless
        # they are given back here, and each one would keep a torn-down plugin
        # alive behind a callback.
        self._teardown_hover_preview()

        # 2. Cleanup predictor subprocess (with timeout to avoid blocking unload)
        # A remote route holding the slot has an on-device predictor parked
        # behind it, and that one owns the subprocess. Give the slot back first
        # so the cleanup below reaches the thing that has something to close.
        try:
            self._drop_cloud_correct_predictor()
        except Exception:  # noqa: BLE001 -- teardown must never raise  # nosec B110
            pass
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
        # Stopped before the tasks: a tick landing during teardown would queue
        # a fetch on a controller that is already coming apart.
        if self._config_refresh_timer is not None:
            try:
                self._config_refresh_timer.stop()
            except RuntimeError:
                pass  # the dock took its children with it
            self._config_refresh_timer = None
        self._disarm_credits_watch()
        self._cancel_task("_key_revalidate_task")
        self._cancel_task("_config_prefetch_task")
        self._cancel_task("_catalog_prefetch_task")
        self._cancel_task("_usage_fetch_task")
        self._cancel_task("_warmup_task")
        self._cancel_task("_auto_run_plan_task")
        self._cancel_task("_auto_token_task")
        self._cancel_manual_charge_tasks()

        # 3. Disconnect worker signals before termination to prevent callbacks on deleted UI
        # Every QThread this controller owns belongs here. Three used to be
        # missing, and each one kept its `done` wired to a dock that step 6
        # deletes; the removal worker's completion also holds this controller
        # through its lambda, so the whole plugin survived unload behind a
        # multi-gigabyte delete.
        _qthread_workers = [
            self.deps_install_worker, self.download_worker, self._verify_worker,
            getattr(self, "_predictor_worker", None),
            getattr(self, "_startup_check_worker", None),
            getattr(self, "_device_info_worker", None),
            getattr(self, "_manual_encode_worker", None),
            _crop_read_worker,
            getattr(self, "_remove_data_worker", None),
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

        # 4. Stop workers. Cooperatively cancel where supported, then give EVERY
        # worker a bounded wait and park the survivors. Never terminate(): these
        # threads run subprocess installs, network I/O, or in-process native
        # imports, and a hard stop there can orphan a child installer, leave a
        # half-written venv, or abort all of QGIS outright. A thread that
        # outlives the wait keeps its last reference parked until its finished
        # signal fires, mirroring the auto worker path (see park_orphaned_worker).
        for worker in _qthread_workers:
            # isRunning() INSIDE the try, like the wait loop below. A worker
            # whose C++ half is already gone raises there, and this is step 4
            # of a teardown whose remaining steps remove the dock, take the map
            # tools off the canvas, drop the rubber bands and save a pending
            # review. Letting one dead wrapper abort unload leaves all of those
            # behind, alive, holding this controller.
            try:
                if worker and worker.isRunning() and hasattr(worker, "cancel"):
                    worker.cancel()
            except (RuntimeError, AttributeError):
                pass
        # ONE budget for the whole set, not 3 s per worker: six that ignore
        # cancel used to mean eighteen seconds of frozen QGIS on quit, on top
        # of the auto worker's own wait below. They were all cancelled just
        # above, so the ones that can stop stop together; this only bounds how
        # long we wait before parking the rest.
        deadline = time.monotonic() + 3.0
        _parked_workers = []
        for worker in _qthread_workers:
            # isRunning() inside the try here too: a dead C++ wrapper raised
            # RuntimeError on the line that reads it and took the rest of
            # unload with it.
            try:
                if not (worker and worker.isRunning()):
                    continue
                left_ms = int(max(0.0, deadline - time.monotonic()) * 1000)
                if left_ms > 0 and worker.wait(left_ms):
                    continue
                park_orphaned_worker(worker)
                _parked_workers.append(worker)
            except RuntimeError:
                pass
        self.deps_install_worker = None
        self.download_worker = None
        self._verify_worker = None
        self._predictor_worker = None
        self._startup_check_worker = None
        self._device_info_worker = None
        self._manual_encode_worker = None
        self._remove_data_worker = None

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

        # 5b. Save a still-pending review NOW, while the project and the dock
        # are both alive. _teardown_auto_mode runs the same rescue, but it runs
        # in step 9, after the dock is deleted and while QGIS may already be
        # taking the project apart, and a detection the user paid for must not
        # ride on that. It is a no-op once this one has written the layer.
        try:
            self._autosave_pending_auto_review(exit_path="unload")
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110
        # The same net for the other mode. Semi-Auto's Save only appends to a
        # list in memory until the user presses Export, and a docked panel gets
        # no close event when QGIS quits, so polygons the user was told were
        # saved went with the session. Here, beside its sibling, and for the
        # same reason: the project and the dock are both still alive.
        try:
            self._autosave_manual_saved_polygons()
        except Exception:  # noqa: BLE001 -- unload must never raise
            pass  # nosec B110

        # 6. Remove dock widget
        # Guarded like every other step. Qt destroys main-window children on
        # quit, so a floating dock can already be gone by the time this runs,
        # and these two bare calls were the last unguarded statements in the
        # method. A raise here skipped the map tool, the rubber bands, the auto
        # worker's join, the canvas event filters and the log collector.
        if self.dock_widget:
            try:
                self.iface.removeDockWidget(self.dock_widget)
                self.dock_widget.deleteLater()
            except (RuntimeError, AttributeError):
                pass
            self.dock_widget = None

        # 7. Clear markers and unset map tool
        # A QgsMapTool is a CHILD of the canvas, so unsetting it only stops it
        # being active: it stays alive with every connection to this dead
        # controller, and a reload adds one more generation. Collected here and
        # deleted after the auto teardown, which nulls the zone tool without
        # deleting it.
        _dead_map_tools = [self.map_tool]
        if self.map_tool:
            try:
                self.map_tool.clear_markers()
            except (RuntimeError, AttributeError):
                pass
            try:
                if self.iface.mapCanvas().mapTool() == self.map_tool:
                    self.iface.mapCanvas().unsetMapTool(self.map_tool)
                    # Unsetting leaves the canvas with no tool at all, so a
                    # plugin removed mid-session left the user clicking on a
                    # dead map. Hand back the tool the session took over from.
                    self._restore_previous_map_tool()
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

        # A history thumbnail may still be rendering in the background: stop it
        # here (blocking, bounded by one small render) so no render job and no
        # callback outlives the plugin that owns them.
        self._cancel_history_thumbnail()

        # 9. Stop any running auto detection worker, then tear down Pro auto mode.
        # _stop_auto_detection keeps the worker reference (the thread is winding
        # down its last network call in the background), so join it here before
        # the plugin is destroyed, and drop its last signal so a late emission
        # cannot call back into a half-torn-down plugin (shutdown-crash guard).
        self._stop_auto_detection()
        auto_worker = self._auto_worker
        if auto_worker is not None:
            # The slot is the per-run bound lambda wired in _launch_auto_worker,
            # so the disconnect has to name it, not the method it calls.
            for slot in (getattr(self, "_auto_cancelled_slot", None),
                         self._on_auto_cancelled):
                if slot is None:
                    continue
                try:
                    auto_worker.cancelled.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            self._auto_cancelled_slot = None
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
        _dead_map_tools += [
            getattr(self, "_zone_selection_tool", None),
            getattr(self, "_exemplar_maptool", None),
            getattr(self, "_shape_maptool", None),
        ]
        self._teardown_auto_mode()
        # The usage/warmup/key-revalidate/config requests are QgsTasks now and
        # are cancelled cooperatively above (step 2b); nothing to wait on here.

        # 10. Delete every map tool this controller made. The canvas watches
        # its tool's destroyed signal, so dropping the active one is safe, but
        # unset it first anyway: nothing below this point can repair a canvas
        # left pointing at a tool that is going away.
        for tool in _dead_map_tools:
            if tool is None:
                continue
            try:
                if self.iface.mapCanvas().mapTool() is tool:
                    self.iface.mapCanvas().unsetMapTool(tool)
            except (RuntimeError, AttributeError):
                pass
            try:
                tool.deleteLater()
            except (RuntimeError, AttributeError):
                pass
        self._zone_selection_tool = None
        self._exemplar_maptool = None
        self._shape_maptool = None

        # 11. One last bounded wait on the threads parked above. Nothing else
        # ever joins them: a parked thread that is still running when Python
        # shuts down is destroyed by ~QThread, which aborts the whole process.
        # A wait here cannot cover a long install, so it is a narrower window,
        # not a closed one, and it stays bounded so quitting never hangs.
        if _parked_workers:
            park_deadline = time.monotonic() + 5.0
            for worker in _parked_workers:
                try:
                    left_ms = int(
                        max(0.0, park_deadline - time.monotonic()) * 1000)
                    if left_ms <= 0:
                        break
                    worker.wait(left_ms)
                except (RuntimeError, AttributeError):
                    pass

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

        self.dock_widget.manual_engine_changed.connect(
            self._on_manual_engine_changed)
        self.dock_widget.install_requested.connect(self._on_install_requested)
        self.dock_widget.cancel_install_requested.connect(self._on_cancel_install)
        self.dock_widget.start_segmentation_requested.connect(self._on_start_segmentation)
        self.dock_widget.save_polygon_requested.connect(self._on_save_polygon)
        self.dock_widget.export_layer_requested.connect(self._on_export_layer)
        self.dock_widget.undo_requested.connect(self._on_undo)
        self.dock_widget.stop_segmentation_requested.connect(self._on_stop_segmentation)
        self.dock_widget.refine_settings_changed.connect(self._on_refine_settings_changed)
        self.dock_widget.size_filter_changed.connect(self._on_size_filter_changed)
        self.dock_widget.fill_holes_size_changed.connect(
            self._on_fill_holes_size_changed)
        self.dock_widget.clean_edges_changed.connect(
            self._on_clean_edges_changed)
        self.dock_widget.outline_budget_changed.connect(
            self._on_outline_budget_changed)
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
        self.dock_widget.auto_advanced_toggled.connect(self._on_auto_advanced_toggled)
        self.dock_widget.auto_prompt_committed.connect(self._reseed_auto_detail_for_object)
        self.dock_widget.auto_layer_combo.layerChanged.connect(
            self._on_auto_layer_combo_changed)
        # Cancel button (dock stub is a pass; we wire the real handler here).
        self.dock_widget.auto_cancel_btn.clicked.connect(self._on_auto_cancel_clicked)
        # Auto review panel signals (plan #78 round 5).
        self.dock_widget.auto_refine_changed.connect(self._on_auto_refine_changed_debounced)
        self.dock_widget.auto_export_requested.connect(self._on_auto_export_clicked)
        self.dock_widget.auto_retry_requested.connect(self._on_auto_retry_guarded)
        self.dock_widget.auto_review_exit_requested.connect(self._on_auto_review_exit_clicked)
        self.dock_widget.auto_display_mode_changed.connect(self._on_auto_display_mode_changed)
        # Correct step: in-place AI reshape (no mode switch, no separate screen).
        self.dock_widget.auto_reshape_ai_requested.connect(
            self._on_reshape_ai_requested)
        self.dock_widget.auto_reshape_done_requested.connect(self._on_reshape_done)
        # Round 3: the AI | Manual method switch and the AI-assisted Add lane.
        self.dock_widget.auto_correct_method_changed.connect(
            self._on_correct_method_changed)
        self.dock_widget.auto_ai_add_requested.connect(self._on_ai_add_requested)
        self.dock_widget.auto_ai_add_keep_requested.connect(
            self._route_save_add_mode)
        # The setup banner's Cancel: the way out while an install holds the
        # review (see plugin/local_ai_install_lock.py).
        self.dock_widget.auto_review_install_cancel_requested.connect(
            self._on_review_install_cancel_requested)
        self.dock_widget.auto_exit_requested.connect(self._on_auto_exit_clicked)
        # Visual exemplar controls (+ Example / + Exclude / chip remove).
        self.dock_widget.auto_add_exemplar_requested.connect(self._on_add_exemplar_requested)
        self.dock_widget.auto_exemplar_remove_requested.connect(self._on_exemplar_remove_requested)
        self.dock_widget.auto_zero_assist_clicked.connect(self._on_auto_zero_assist_clicked)
        self.dock_widget.auto_escape_pressed.connect(self._on_auto_escape_shortcut)
        self.dock_widget.auto_enter_pressed.connect(self._on_auto_enter_pressed)
        # Undo has no dock signal of its own: the window-level Undo QShortcut
        # is the channel, and the dock's handler on it only covers the Correct
        # step. Listen to the same shortcut so a press during a zone draw is
        # routed to the map tool instead of being eaten.
        self.dock_widget.auto_correct_undo_shortcut.activated.connect(
            self._on_auto_undo_pressed)
        self.dock_widget.auto_review_confidence_changed.connect(
            self._on_auto_review_confidence_changed)
        self.dock_widget.auto_review_confidence_preview.connect(
            self._on_auto_review_confidence_preview)
        self.dock_widget.auto_show_tiles_changed.connect(self._on_auto_show_tiles_toggled)
        self.dock_widget._auto_review_debounce_timer.timeout.connect(
            self._on_auto_review_refine_debounced)
        # Review correction loop (linear ladder, gestures, batch).
        self._connect_auto_correct_signals()
        # QGIS digitizing bridge seam: the Correct step's "Edit precisely in
        # QGIS" arms native editing; the banner's "Done editing" commits and
        # folds back. Both engine entries are idempotent, so a double connect is
        # a harmless no-op. finish_qgis_edit_bridge's commit arg defaults to
        # True, so the no-arg Done signal commits.
        try:
            self.dock_widget.auto_edit_in_qgis_requested.connect(
                self.enter_qgis_edit_bridge)
            self.dock_widget.auto_add_polygon_requested.connect(
                self._on_add_polygon_requested)
            self.dock_widget.auto_qgis_bridge_done_requested.connect(
                self.finish_qgis_edit_bridge)
            self.dock_widget.auto_qgis_bridge_tool_requested.connect(
                self.activate_qgis_bridge_tool)
            self.dock_widget.auto_qgis_bridge_undo_requested.connect(
                self.undo_qgis_bridge_edit)
            self.dock_widget.auto_qgis_bridge_gesture_requested.connect(
                self._on_bridge_gesture_requested)
            self.dock_widget.auto_qgis_bridge_points_changed.connect(
                self._on_bridge_points_changed)
            self.dock_widget.auto_qgis_bridge_delete_requested.connect(
                self.delete_bridge_target_polygon)
        except (AttributeError, RuntimeError):
            pass

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

    def _on_auto_enter_pressed(self) -> bool:
        """Single entry point for the Automatic flow's Enter key.

        The zone draw gets first refusal. Its map tool handles Enter itself,
        but the dock's window-level Enter shortcut matches the key before the
        canvas ever sees it, so the draw would never close from the keyboard.
        Anything else falls through to the shared dispatcher.
        """
        tool = self._zone_selection_tool
        if tool is not None:
            try:
                if self.iface.mapCanvas().mapTool() is tool and tool.has_points():
                    if tool.finish():
                        return True
            except (RuntimeError, AttributeError):
                pass
        return self._route_enter()

    def _on_auto_undo_pressed(self) -> bool:
        """Platform Undo (Ctrl+Z, Cmd+Z on macOS) during a zone draw.

        Same routing as Enter: the zone draw gets first refusal. The map tool
        undoes its own last point, but the dock's window-level Undo shortcut
        matches the key before the canvas sees the press, so mid-draw Ctrl+Z
        was swallowed and only Backspace worked. The dock's own handler keeps
        the Correct-step undo and its gate is down while the zone is being
        drawn, so exactly one of the two acts on any press.
        """
        tool = self._zone_selection_tool
        if tool is None:
            return False
        try:
            if self.iface.mapCanvas().mapTool() is tool:
                return tool.undo_point()
        except (RuntimeError, AttributeError):
            pass
        return False

    def _on_dock_visibility_changed(self, visible: bool):
        if not visible:
            # The panel closed or was tabbed away, so nothing on the map should
            # still be describing what the cursor was over.
            self._stop_hover_preview("dock hidden")
            # A panel that is gone cannot show what the clicks are doing, so
            # the session goes with it rather than staying armed underneath.
            self._stop_manual_session_for_hidden_dock()
            return
        # The panel is back: a session parked when it went away is armed again.
        self._resume_parked_manual_session()
        if self._first_time_setup_done:
            return
        self._first_time_setup_done = True
        # Armed for the whole session, whatever mode the dock opens in: the
        # watch decides mode by mode whether a read is worth making.
        self._arm_credits_watch()
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
        """Bring the product configuration up to date, entirely off the GUI thread.

        Every read of the configuration is a memory read, so this hidden task is
        the one thing that fills it, and it owns all three slow steps: the copy
        an earlier session left on disk, the fetch, and mirroring the result back
        to disk. None of them may run on the GUI thread; get_config() is called
        while the dock is being built.

        Fails open at every step: no configuration keeps Automatic mode
        available and the tutorial-URL fallback in place.
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
            lambda: self._refresh_server_config(client, PRODUCT_ID),
            hidden=True,
        )
        self._config_prefetch_task.succeeded.connect(self._on_config_prefetched)
        self._config_prefetch_task.failed.connect(self._on_config_prefetch_failed)
        QgsApplication.taskManager().addTask(self._config_prefetch_task)

    @staticmethod
    def _refresh_server_config(client, product_id: str) -> dict:
        """Publish the disk copy, fetch, publish and mirror the result.

        Runs on the task's thread. Pure Python and no Qt, so nothing here needs
        the GUI thread and nothing here may block it. The disk copy goes in
        first, so a failed or slow fetch still leaves the last known
        configuration in force. Priming is best-effort on its own: a damaged
        file must not cost us the fetch that follows.
        """
        from ..core.config_cache import prime_from_disk
        try:
            prime_from_disk()
        except Exception:  # noqa: BLE001 -- the disk copy is a bonus  # nosec B110
            pass
        config = client.get_config(product_id)
        if isinstance(config, dict) and "error" not in config:
            from ..core.activation_manager import set_cached_config
            set_cached_config(config)
        return config

    def _on_config_prefetched(self, _config: object) -> None:
        """The configuration is already published; only the dock needs telling."""
        self._config_prefetch_task = None
        self._reapply_server_switches()
        self._arm_config_refresh()

    def _on_config_prefetch_failed(self, message: str, code: str) -> None:
        self._config_prefetch_task = None
        # The fetch failed, but the task may still have published the copy an
        # earlier session left on disk, so the dock is nudged either way.
        self._reapply_server_switches()
        self._notify_connection_issue(code, message)
        # Armed on failure too: a user who opened QGIS offline is exactly the
        # one who should pick the configuration up once the link comes back,
        # instead of running the whole session on no configuration at all.
        self._arm_config_refresh()

    def _arm_config_refresh(self) -> None:
        """Re-fetch the configuration on a slow repeat for the rest of the session.

        Without this the configuration is read once, when the dock first
        opens, and every server dial in the plugin then reaches a user only
        when they next start QGIS. A session left open for days would ride out
        an incident on the values it woke up with, which is the thing the dials
        exist to avoid.

        Safe to repeat because the grid geometry is client-side by design (see
        the OVERLAP_FRACTION note in the placement rules): the credit estimate
        and the run compute the same tile count from shipped constants, so a
        configuration arriving between the two cannot bill a grid the preview
        never showed. The one served value the estimate reads is the tile cap,
        and a cap can only refuse a run, never enlarge its bill.

        Idempotent: the timer is created once and left running.
        """
        if self._config_refresh_timer is not None or self.dock_widget is None:
            return
        from qgis.PyQt.QtCore import QTimer

        # Parented to the dock, never to the controller: the controller is not
        # a QObject, and an unparented timer outlives the plugin.
        timer = QTimer(self.dock_widget)
        timer.setInterval(self._config_refresh_interval_ms())
        timer.timeout.connect(self._refresh_config_if_idle)
        timer.start()
        self._config_refresh_timer = timer

    @staticmethod
    def _config_refresh_interval_ms() -> int:
        """How often the configuration is re-read, in ms.

        A dial on itself, so the interval can be shortened during an incident
        and lengthened if it ever costs more than it is worth. Bounded at five
        minutes so no served value can turn this into a polling loop.
        """
        from ..core.server_dials import dial_in_range

        minutes = dial_in_range("config_refresh_minutes", 30, 5, 720)
        return int(minutes) * 60 * 1000

    def _refresh_config_if_idle(self) -> None:
        """One repeat fetch, skipped while a run or a review is on screen.

        A run resolves its own dials once at construction, so it is already
        immune, but the review reads them live and a shape changing under a
        user mid-review would read as a bug. Nothing is lost by waiting: the
        timer comes back.
        """
        if getattr(self, "_auto_worker", None) is not None:
            return
        if getattr(self, "_auto_review", None):
            return
        self._prefetch_server_config()

    def _reapply_server_switches(self) -> None:
        """Have the dock re-read the configuration in force. GUI thread only."""
        if self.dock_widget is None:
            return
        try:
            self.dock_widget.apply_server_feature_switches()
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- the dock can be gone by now

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
