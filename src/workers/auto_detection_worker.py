









































from __future__ import annotations

import logging
import threading
import uuid
from collections import deque

from qgis.core import Qgis
from qgis.PyQt.QtCore import QThread, pyqtSignal

from ..core import transport_dials as _td
from ..core.error_policy import (
    BACKEND_UNAVAILABLE_CODES,
    EXHAUSTED_CODES,
    OFFLINE_STOP_CODE,
    RUN_FATAL_CODES,
    TRANSIENT_CODES,
)
from ..core.server_dials import dial_bool as _dial_bool
from ..core.server_dials import dial_in_range as _dial_in_range
from ..core.server_dials import feature_enabled as _feature_on
from .adaptive_concurrency import (
    DEFAULT_COOLDOWN_CYCLES,
    DEFAULT_FAILURE_THRESHOLD,
    AdaptiveConcurrency,
    OfflineFastFail,
)
from .auto_worker.convert_pool import (
    _CONVERT_BACKLOG_PER_WORKER,
    _CONVERT_DRAIN_BUDGET_S,
    _CONVERT_RESCUE_PROBES,
    _CONVERT_WORKERS,
    _CONVERT_WORKERS_CEILING,
    _GEOS_THREAD_LOCAL_MIN_VERSION,
    AutoConvertPoolMixin,
    _convert_failure_reason,
    _resolve_convert_workers,
)
from .auto_worker.density_probe_run import AutoDensityProbeMixin
from .auto_worker.gate_scan import (
    _GATE_RENDER_CACHE_MAX,
    _GATE_SCAN_RENDER_TRIES,
    AutoGateScanMixin,
)
from .auto_worker.mask_geometry import (
    _COMPACT_MIN_FILL,
    _HARD_COVER_SHAPE_ESCAPE,
    _HARD_TILE_COVERAGE,
    _MASK_CAP_TRIGGER_FRAC,
    _MAX_MASKS_PER_TILE,
    _MAX_TILE_COVERAGE,
    _MIN_KEEP_PX,
    _TILE_SPAN_FRACTION,
    AutoMaskGeometryMixin,
)
from .auto_worker.rescan_policy import (
    AutoRescanPolicyMixin,
)
from .auto_worker.retry_policy import (
    _ABORT_REQUEUE_BASE_S,
    _AIMD_MIN,
    _AIMD_START,
    _AIMD_UPLOAD_GROW_S,
    _ANSWER_QUIET_P90_FACTOR,
    _ANSWER_QUIET_S,
    _BACKEND_UNAVAILABLE_DELAY_S,
    _BACKEND_UNAVAILABLE_RETRIES,
    _BUSY_JITTER,
    _DEAD_LINK_QUIET_S,
    _DEFAULT_MAX_WAIT_S,
    _DEFAULT_POLL_INTERVAL_S,
    _HANDOFF_MIN_DELAY_S,
    _HANDOFF_OPEN_WINDOW_MAX_S,
    _MAX_RATE_LIMIT_RETRIES,
    _MIDRUN_OFFLINE_STREAK,
    _MIN_POLL_BACKOFF_S,
    _OUTAGE_MAX_S,
    _OUTAGE_PROBE_S,
    _OUTAGE_RESUME_FRACTION,
    _OUTAGE_RESUME_MIN,
    _QUEUE_RETRY_BUDGET_S,
    _REFUSAL_WINDOW,
    _SWEEP_ANSWER_P90_FACTOR,
    _UPLOAD_SLOW_S,
    _UPLOAD_STALL_S,
    HANDOFF_CODE,
    HANDOFF_OVERLOAD_CODE,
    RATE_LIMIT_SETBACK_CODES,
    AutoRetryPolicyMixin,
)
from .auto_worker.run_lifecycle import (
    _BILLED_DRAIN_STOP_REASONS,
    _EMPTY_TILES_BEFORE_NOTICE,
    _MAX_CONSECUTIVE_TILE_FATALS,
    _STOP_DRAIN_BUDGET_S,
    AutoRunLifecycleMixin,
)
from .auto_worker.run_loops import AutoRunLoopsMixin
from .auto_worker.self_exemplar_pass import AutoSelfExemplarMixin
from .auto_worker.tile_render import (
    _PREFETCH_DEPTH,
    _PREFETCH_HOLDOFF_S,
    _RENDER_RETRY_DELAY_S,
    _RENDER_RETRY_MAX,
    _RENDER_SLOW_S,
    AutoTileRenderMixin,
)
from .auto_worker.tile_submit import (
    AutoTileSubmitMixin,
    _as_float,
    _as_int,
    _handoff_wait_s,
)
from .tile_convert_pool import TileConvertPool, TileConvertProcessPool
from .tile_render_bridge import TileRenderBridge

logger = logging.getLogger(__name__)

__all__ = [
    "logger",
    "_BACKEND_UNAVAILABLE_RETRIES",
    "_BACKEND_UNAVAILABLE_DELAY_S",
    "_MAX_RATE_LIMIT_RETRIES",
    "_QUEUE_RETRY_BUDGET_S",
    "_HANDOFF_MIN_DELAY_S",
    "_HANDOFF_OPEN_WINDOW_MAX_S",
    "HANDOFF_CODE",
    "HANDOFF_OVERLOAD_CODE",
    "RATE_LIMIT_SETBACK_CODES",
    "_REFUSAL_WINDOW",
    "_BUSY_JITTER",
    "_PREFETCH_DEPTH",
    "_CONVERT_WORKERS",
    "_CONVERT_WORKERS_CEILING",
    "_GEOS_THREAD_LOCAL_MIN_VERSION",
    "_CONVERT_BACKLOG_PER_WORKER",
    "_CONVERT_DRAIN_BUDGET_S",
    "_STOP_DRAIN_BUDGET_S",
    "_BILLED_DRAIN_STOP_REASONS",
    "_MAX_MASKS_PER_TILE",
    "_MASK_CAP_TRIGGER_FRAC",
    "_MAX_TILE_COVERAGE",
    "_HARD_TILE_COVERAGE",
    "_HARD_COVER_SHAPE_ESCAPE",
    "_COMPACT_MIN_FILL",
    "_TILE_SPAN_FRACTION",
    "_MIN_KEEP_PX",
    "_UPLOAD_SLOW_S",
    "_UPLOAD_STALL_S",
    "_DEFAULT_POLL_INTERVAL_S",
    "_DEFAULT_MAX_WAIT_S",
    "_MIN_POLL_BACKOFF_S",
    "_AIMD_START",
    "_AIMD_UPLOAD_GROW_S",
    "_AIMD_MIN",
    "_MAX_CONSECUTIVE_TILE_FATALS",
    "_EMPTY_TILES_BEFORE_NOTICE",
    "_RENDER_RETRY_MAX",
    "_RENDER_RETRY_DELAY_S",
    "_PREFETCH_HOLDOFF_S",
    "_RENDER_SLOW_S",
    "_MIDRUN_OFFLINE_STREAK",
    "_GATE_RENDER_CACHE_MAX",
    "_GATE_SCAN_RENDER_TRIES",
    "_as_int",
    "_handoff_wait_s",
    "_as_float",
    "_CONVERT_RESCUE_PROBES",
    "_convert_failure_reason",
    "_resolve_convert_workers",
    "AutoDetectionWorker",
    "TileRenderBridge",
    "TRANSIENT_CODES",
    "EXHAUSTED_CODES",
    "RUN_FATAL_CODES",
    "BACKEND_UNAVAILABLE_CODES",
    "OFFLINE_STOP_CODE",
]


class AutoDetectionWorker(
    AutoRunLifecycleMixin,
    AutoRunLoopsMixin,
    AutoTileRenderMixin,
    AutoTileSubmitMixin,
    AutoGateScanMixin,
    AutoRetryPolicyMixin,
    AutoConvertPoolMixin,
    AutoMaskGeometryMixin,
    AutoRescanPolicyMixin,
    AutoDensityProbeMixin,
    AutoSelfExemplarMixin,
    QThread,
):















































    tile_completed = pyqtSignal(int, list)
    all_tiles_finished = pyqtSignal(list)
    progress = pyqtSignal(int, int)





    rescan_state = pyqtSignal(int, object, bool)
    warning = pyqtSignal(str)


    nothing_found_yet = pyqtSignal(int)
    error = pyqtSignal(str)
    credits_exhausted = pyqtSignal(int)
    cancelled = pyqtSignal()




    queue_state = pyqtSignal(int, int, int)




    run_phase = pyqtSignal(str)



    density_replan = pyqtSignal(object)

    def __init__(
        self,
        tiles: list[tuple[int, int, int, int]],
        geo_transform: dict,
        crs_authid: str,
        prompt: str,
        auth: dict,
        run_id: str | None = None,
        max_concurrent: int = 4,
        score_threshold: float = 0.0,
        detection_threshold: float = 0.30,
        exemplar_stamps: list | None = None,
        progress_offset: int = 0,
        progress_total: int | None = None,
        clip_polygon_wkb: bytes | None = None,
        zone_keep_margin_m: float = 0.0,
        gsd: float = 0.0,
        merge_separate: bool = True,
        seam_min_dim: float = 0.0,
        merge_scalars: dict | None = None,
        subdivide_budget: int = 0,
        collect_raw: bool = False,
        return_semantic: bool = False,
        gate_config: dict | None = None,
        mask_scale: int = 1,
        client_meta: dict | None = None,
        tile_renderer=None,
        source_is_online: bool = False,
        transform_context=None,
        density_probe: dict | None = None,
        self_exemplar: dict | None = None,
        land_cover: bool = False,
        parent=None,
    ):












































































        super().__init__(parent)



        self._tile_renderer = tile_renderer






        self._skip_unavailable_tiles = bool(source_is_online) and _feature_on(
            "unavailable_tile_skip")












        self._map_hypothesis_nms = _dial_bool("features.map_hypothesis_nms", False)





        self._tile_renderer_cancel = None



        self._render_request = None
        self._render_collect = None
        self._render_ready = None
        self._render_duration = None

        self._prefetched: dict[int, int] = {}


        self._prefetch_holdoff_until = 0.0



        self._render_ramp_pending = False
        self._stream_pending = None
        bridge = getattr(tile_renderer, "__self__", None)
        if bridge is not None and hasattr(bridge, "cancel"):
            self._tile_renderer_cancel = bridge.cancel
        if bridge is not None and hasattr(bridge, "request_render"):
            self._render_request = bridge.request_render
            self._render_collect = bridge.collect_render
            self._render_ready = getattr(bridge, "render_ready", None)
            self._render_duration = getattr(bridge, "last_render_duration", None)


        self._encode_ahead = None
        self._render_bridge = bridge if (
            bridge is not None and hasattr(bridge, "set_landed_hook")
            and hasattr(bridge, "collect_render_timed")) else None
        self._tiles = tiles



        self._plan_len = len(tiles)





        self._answer_cache: dict[int, str] = {}
        self._answer_cache_bytes = 0
        self._answer_cache_dropped = False
        self._answer_cache_max_bytes = int(_dial_in_range(
            "detection_policy.resume.cache_max_mb", 0, 0, 4096)) * 1048576
        self._replay_answers: dict[int, str] = {}
        self._resuming = False
        self.tiles_replayed_from_cache = 0



        self._self_ex_settings = self_exemplar if isinstance(self_exemplar, dict) else None

        self._self_ex_held: dict[int, tuple] = {}
        self._self_ex_tried: set[int] = set()
        self._self_ex_sent_at: dict[int, float] = {}
        self.self_ex_sent = 0
        self.self_ex_used = 0
        self.self_ex_no_confident = 0
        self.self_ex_fallback: dict[str, int] = {}
        self.self_ex_upload_bytes = 0
        self.self_ex_answer_s = 0.0
        self.tiles_resent = 0
        self.stopped_offline = False


        self._density_setup(density_probe)
        self._geo_transform = geo_transform
        self._crs_authid = crs_authid



        self._transform_context = transform_context






        self._quota_refusal: dict | None = None  # type: ignore[assignment]


        self._distance_area = None






        self._ground_kx = 1.0
        self._ground_ky = 1.0
        self._prompt = prompt
        self._auth = dict(auth)
        from ..core.activation_manager import auth_revision

        self._auth_revision = auth_revision()
        self._run_id = run_id or str(uuid.uuid4())
        self._max_concurrent = max(1, max_concurrent)
        self._score_threshold = score_threshold
        self._detection_threshold = detection_threshold



        self._exemplar_stamps_in = exemplar_stamps or []
        self._stamps: list = []



        self._stamp_full_boxes: list = []



        self._stamp_regions: list = []
        self._tile_exemplars: dict = {}
        self._tile_stamp_norm: dict = {}




        self._top_stamp_ty: int | None = None
        self._stamp_bottom_top_row = False
        self._progress_offset = max(0, progress_offset)
        self._progress_total = progress_total





        self._clip_polygon_wkb = clip_polygon_wkb



        self._zone_keep_margin_m = max(0.0, float(zone_keep_margin_m or 0.0))
        self._gsd = gsd
        self._merge_separate = merge_separate





        self._collect_raw = bool(collect_raw)





        self._return_semantic = bool(return_semantic)


        self._land_cover = bool(land_cover)
        if self._land_cover:


            self._exemplar_stamps_in = []





        self._mask_scale = int(mask_scale) if isinstance(mask_scale, int) else 1





        self._client_meta = client_meta if isinstance(client_meta, dict) else None
        self._tile_clean_image: dict[int, str] = {}

        self._tile_clean_jobs: dict = {}
        self._clean_image_pool = None
        self._clean_image_closed = False
        self._clean_image_lock = threading.Lock()







        self._run_fields_tile_index: int | None = None

        self._wgs84_transform = None
        self._wgs84_transform_failed = False



        self._run_nam = None
        self._dead_reply_tiles: set[int] = set()






        self._gate_config = gate_config if isinstance(gate_config, dict) else None
        self._gate_skip: set[int] = set()
        self._gate_prepaid: set[int] = set()





        self._prefilter_skip: set[int] = set()




        self._gate_tile_bytes: dict[int, tuple] = {}



        self._seam_min_dim = seam_min_dim



        self._merge_scalars = merge_scalars if isinstance(merge_scalars, dict) else {}



        self._clip_geom = None
        self._clip_engine = None
        self._clip_local = threading.local()



        self._stat_lock = threading.Lock()






        self._convert_pool: TileConvertPool | None = None  # type: ignore[assignment]


        self._convert_prespawned: TileConvertProcessPool | None = None


        self._convert_pool_is_processes = False






        self._subdivide_budget = max(0, int(subdivide_budget))






        from ..core import detection_policy as _dp
        from ..core.served_config import require_served_int, require_served_number
        self._prefilter = _dp.gate_prefilter_config()
        self._max_masks = _dp.max_masks_per_tile(_MAX_MASKS_PER_TILE)
        self._mask_cap_trigger = int(
            _dp.mask_cap_trigger_frac(_MASK_CAP_TRIGGER_FRAC) * self._max_masks)
        self._subdiv_max_depth = require_served_int(
            "detection_policy.seed.saturation.subdiv_max_depth", 0, 8)
        self._resplit_time_ratio = require_served_number(
            "detection_policy.seed.saturation.resplit_time_ratio", 0.0, 100.0)

        self._run_started_at = 0.0
        self._paid_tiles_total = 0
        self._paid_tiles_done = 0
        self._resplit_deadline: float = 0.0
        self._resplit_dropped = 0
        self._max_tile_coverage = _dp.max_tile_coverage(_MAX_TILE_COVERAGE)
        self._hard_tile_coverage = _dp.hard_tile_coverage(_HARD_TILE_COVERAGE)
        self._hard_cover_shape_escape = _dp.hard_cover_shape_escape(
            _HARD_COVER_SHAPE_ESCAPE)
        self._subdiv_overlap = require_served_number(
            "detection_policy.seed.saturation.subdivide_overlap_fraction", 0.0, 0.49)
        self._subdiv_min_parent_px = require_served_int(
            "detection_policy.seed.saturation.subdivide_min_parent_px", 1, 100_000)
        self._compact_min_fill = _dp.compact_min_fill(_COMPACT_MIN_FILL)
        self._tile_span_fraction = _dp.tile_span_fraction(_TILE_SPAN_FRACTION)
        self._min_keep_px = _dp.min_keep_px(_MIN_KEEP_PX)



        self._min_keep_floor_m2 = require_served_number(
            "detection_policy.seed.saturation.min_keep_floor_m2", 0.0, 1_000_000.0)



        self._map_cover_score_floor = _dp.map_cover_score_floor(0.0)





        self._pinhole_m = _dp.pinhole_fill_m(0.0)
        self._tile_simplify_mult = _dp.tile_simplify_mult(0.0)



        self._semantic_coverage_floor = _dp.semantic_rescue_coverage_floor()




        self._max_rate_limit_retries = _dp.max_rate_limit_retries(
            _MAX_RATE_LIMIT_RETRIES)
        self._queue_retry_budget_s = _dp.queue_retry_budget_s(
            _QUEUE_RETRY_BUDGET_S)
        self._midrun_offline_streak = _dp.midrun_offline_streak(
            _MIDRUN_OFFLINE_STREAK)
        self._backend_unavailable_retries = _dp.backend_unavailable_retries(
            _BACKEND_UNAVAILABLE_RETRIES)
        self._backend_unavailable_delay_s = _dp.backend_unavailable_delay_s(
            _BACKEND_UNAVAILABLE_DELAY_S)


        self._busy_jitter = _dp.busy_jitter(_BUSY_JITTER)



        self._prefetch_depth = _dp.prefetch_depth(
            max(_PREFETCH_DEPTH, self._max_concurrent))


        self._window_hint_max = _td.window_hint_ceiling()








        self._render_window = AdaptiveConcurrency(
            start=self._prefetch_depth, minimum=min(2, self._prefetch_depth),
            maximum=self._prefetch_depth)



        self._render_slow_s = _dp.render_slow_s(_RENDER_SLOW_S)




        from ..core import xyz_tile_fetch as _xyz
        _xyz.set_parallel_tile_requests(
            _dp.tile_fetch_parallel(_xyz.parallel_tile_requests()))
        self._convert_workers = _resolve_convert_workers(
            _dp.convert_workers(_CONVERT_WORKERS), Qgis.QGIS_VERSION_INT,
            _dp.convert_workers_ceiling(_CONVERT_WORKERS_CEILING))
        self._convert_backlog_per_worker = _dp.convert_backlog_per_worker(
            _CONVERT_BACKLOG_PER_WORKER)
        self._convert_drain_budget_s = _dp.convert_drain_budget_s(
            _CONVERT_DRAIN_BUDGET_S)
        self._stop_drain_budget_s = _dp.stop_drain_budget_s(_STOP_DRAIN_BUDGET_S)
        self._poll_interval_s = _dp.poll_interval_s(_DEFAULT_POLL_INTERVAL_S)
        self._poll_max_wait_s = _dp.poll_max_wait_s(_DEFAULT_MAX_WAIT_S)





        self._stream_reply_budget_s = max(
            self._poll_max_wait_s,
            _dp.submit_timeout_ms(int(self._poll_max_wait_s * 1000)) / 1000.0,
        )
        self._min_poll_backoff_s = _dp.min_poll_backoff_s(_MIN_POLL_BACKOFF_S)
        self._gate_render_cache_max = _dp.gate_render_cache_max(_GATE_RENDER_CACHE_MAX)
        self._prefetch_holdoff_s = _dp.prefetch_holdoff_s(_PREFETCH_HOLDOFF_S)
        self._max_tile_fatals = _dp.max_consecutive_tile_fatals(
            _MAX_CONSECUTIVE_TILE_FATALS)
        self._empty_tiles_before_notice = _dp.empty_tiles_before_notice(
            _EMPTY_TILES_BEFORE_NOTICE)
        self._empty_tile_streak = 0
        self._empty_notice_sent = False
        self._render_retry_max = _dp.render_retry_max(_RENDER_RETRY_MAX)
        self._render_retry_delay_s = _dp.render_retry_delay_s(
            _RENDER_RETRY_DELAY_S)
        self._gate_scan_render_tries = _dp.gate_scan_render_tries(
            _GATE_SCAN_RENDER_TRIES)
        self._tile_depth: dict[int, int] = {}
        self._tile_outsize: dict[int, tuple[int, int]] = {}
        self._pending_subtiles: list = []







        self._withheld: dict[int, list] = {}
        self._parent_of: dict[int, int] = {}
        self._parents_with_child_results: set[int] = set()


        self._rescanning: dict[int, int] = {}
        self.tiles_subdivided = 0



        self.tiles_capped_final = 0




        self._aimd = AdaptiveConcurrency(
            start=_dp.aimd_start(_AIMD_START),
            minimum=min(self._max_concurrent, _dp.aimd_min(_AIMD_MIN)),
            maximum=self._max_concurrent,
            cooldown_cycles=_td.aimd_cooldown_cycles(DEFAULT_COOLDOWN_CYCLES),
        )






        self._window_hint: int | None = None  # type: ignore[assignment]
        self._window_hint_logged = False








        self.last_tile_balance: dict | None = None  # type: ignore[assignment]




        self._fastfail = OfflineFastFail(
            threshold=_td.aimd_failure_threshold(DEFAULT_FAILURE_THRESHOLD))





        self._render_attempts: dict[int, int] = {}
        self._render_deferred: deque = deque()


        self._refusal_window: deque = deque(maxlen=_dial_in_range(
            "tuning.network.refusal_window", _REFUSAL_WINDOW, 4, 64))

        self._stop_requested = False







        self._stop_reason: str | None = None  # type: ignore[assignment]




        self._terminal_sent = False




        self.tiles_succeeded = 0



        self.raw_detections_total = 0




        self.tiles_skipped_blank = 0






        self.tiles_prefiltered = 0








        self.tiles_gate_skipped = 0








        self.masks_dropped_whole_tile = 0



        self.masks_whole_tile_armed = 0
        self.masks_dropped_hard_cover = 0
        self.masks_dropped_tile_span = 0
        self.masks_dropped_not_compact = 0





        self.masks_whole_tile_kept_map = 0



        self.masks_dropped_map_lowscore = 0
        self.map_cover_scores: list[float] = []




        self.polygonized_gdal = 0
        self.polygonized_tracer = 0
        self.polygonized_fallback = 0
        self.polygonized_fallback_fast = 0




        self.tiles_render_failed = 0






        self.tiles_unavailable = 0





        self.renders_slow = 0
        self.render_window_floor = self._prefetch_depth






        self.phase_render_s = 0.0
        self.phase_encode_s = 0.0
        self.phase_predict_s = 0.0
        self.phase_convert_s = 0.0




        self.phase_upload_s = 0.0
        self.uploads_slow = 0
        self._uploaded_at: dict[int, float] = {}


        self._upload_progress_at: dict[int, float] = {}
        self._upload_last_sent: dict[int, int] = {}
        self._upload_stall_s = float(_dial_in_range(
            "detection_policy.network.upload_stall_s", _UPLOAD_STALL_S, 0.0, 300.0))


        self.uploads_requeued = 0
        self.upload_stalls = 0



        self._reply_byte_at: dict[int, float] = {}
        self._answer_times: list[float] = []
        self._dead_link_quiet_s = float(_dial_in_range(
            "detection_policy.network.dead_link_quiet_s", _DEAD_LINK_QUIET_S,
            0.0, 60.0))
        self._answer_quiet_s = float(_dial_in_range(
            "detection_policy.network.answer_quiet_s", _ANSWER_QUIET_S,
            0.0, 600.0))
        self.dead_link_sweeps = 0
        self.answer_quiet_reposts = 0

        self._outage_since: float | None = None
        self._outage_cap_before = 0
        self._outage_s_total = 0.0
        self.outages = 0



        self.outage_probe_cuts = 0
        self._link_setup_max_s = 0.0
        self._outage_probe_s = float(_dial_in_range(
            "detection_policy.network.outage_probe_s", _OUTAGE_PROBE_S,
            0.5, 60.0))
        self._outage_max_s = float(_dial_in_range(
            "detection_policy.network.outage_max_s", _OUTAGE_MAX_S,
            10.0, 3600.0))
        self._outage_resume_min = int(_dial_in_range(
            "detection_policy.network.outage_resume_min", _OUTAGE_RESUME_MIN,
            1, 64))
        self._outage_resume_fraction = float(_dial_in_range(
            "detection_policy.network.outage_resume_fraction",
            _OUTAGE_RESUME_FRACTION, 0.05, 1.0))
        self._answer_quiet_p90_factor = float(_dial_in_range(
            "detection_policy.network.answer_quiet_p90_factor",
            _ANSWER_QUIET_P90_FACTOR, 0.0, 20.0))
        self._sweep_p90_factor = float(_dial_in_range(
            "detection_policy.network.sweep_p90_factor", _SWEEP_ANSWER_P90_FACTOR,
            0.0, 20.0))
        self._abort_requeue_base_s = float(_dial_in_range(
            "detection_policy.network.abort_requeue_base_s", _ABORT_REQUEUE_BASE_S,
            0.1, 60.0))
        self.uploaded_aborts_reposted = 0
        self._upload_slow_s = float(_dial_in_range(
            "detection_policy.network.upload_slow_s", _UPLOAD_SLOW_S, 1.0, 120.0))

        self._aimd_upload_grow_s = float(_dial_in_range(
            "detection_policy.network.aimd_upload_grow_s", _AIMD_UPLOAD_GROW_S,
            0.0, 30.0))





        self.loop_phase_s: dict[str, float] = {}
        self._last_render_wait_s = 0.0
        self._submit_at: dict[int, float] = {}





        self.upload_bytes = 0
        self.uploads_sent = 0








        self.submit_network_retries = 0
        self.tiles_skipped_network = 0







        self.tiles_timed_out = 0



        self.tiles_failed_server = 0
        self._completed_idx: set[int] = set()





        self._hit_mask_cap = False



        self.tiles_mask_capped = 0






        self.observed_mask_gsd = 0.0





        self._last_queue_emit: tuple[int, int, int] | None = None  # type: ignore[assignment]







        self.http_429 = 0
        self.http_503 = 0
        self.inflight_now = 0
        self._convert_pool_kind = ""
        self._convert_pool_workers = 0
        self._convert_fallback_reason = ""






        self.tiles_convert_failed = 0
        self._convert_fail_reason = ""



        self._convert_children_broken = False
        self._convert_rescued_tiles = 0
        self._convert_rescue_probes = _dial_in_range(
            "tuning.convert.rescue_probes", _CONVERT_RESCUE_PROBES, 1, 100)





    def request_stop(self) -> None:







        if self._stop_reason is None or self._stop_reason == "replan":
            self._stop_reason = "user"
        self._stop_requested = True
        if self._tile_renderer_cancel is not None:
            try:
                self._tile_renderer_cancel()
            except (RuntimeError, AttributeError):
                pass





    def run(self) -> None:




        from ..core.power_inhibit import begin_keep_awake, end_keep_awake

        activity = begin_keep_awake("AI Segmentation cloud detection")



        from ..core import network_busy
        busy_token = network_busy.begin("auto_run")


        self.last_tile_balance = None
        try:
            self._run_detection()
        except Exception as exc:  # noqa: BLE001



            logger.error("AutoDetectionWorker crashed", exc_info=True)




            try:
                from ..core.telemetry_errors import report_exception
                report_exception(exc, stage="segment", module="auto_detection_worker")
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:





                self.error.emit(f"Detection stopped unexpectedly (internal error): {exc}")
            except Exception as emit_exc:  # noqa: BLE001


                try:
                    from ..core.telemetry_errors import track_plugin_error
                    track_plugin_error(stage="segment",
                                       error_code="auto_error_emit_failed",
                                       message=type(emit_exc).__name__)
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
        finally:
            network_busy.end(busy_token)
            end_keep_awake(activity)




            try:
                from qgis.core import QgsMessageLog
                QgsMessageLog.logMessage(
                    f"Auto detection: imagery summary - {self.renders_slow} "
                    f"slow render(s), fetch window narrowed to "
                    f"{self.render_window_floor} of {self._prefetch_depth}, "
                    f"{self.tiles_render_failed} tile(s) with no imagery",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )















                posts = max(1, self.uploads_sent)
                answered = max(1, self.tiles_succeeded)
                loop_wall_s = sum(self.loop_phase_s.values())
                QgsMessageLog.logMessage(
                    "Auto detection: stage summary - render wait "
                    f"{self.phase_render_s:.1f}s, encode {self.phase_encode_s:.1f}s, "
                    f"predict {self.phase_predict_s:.1f}s summed over "
                    f"{loop_wall_s:.1f}s of loop wall, convert "
                    f"{self.phase_convert_s:.1f}s over {self.tiles_succeeded} tile(s), "
                    f"upload {self.upload_bytes / 1048576:.1f} MB in "
                    f"{self.uploads_sent} post(s) "
                    f"({self.upload_bytes / posts / 1024:.0f} kB per tile), "
                    f"round-trip {self.phase_predict_s / answered * 1000:.0f} ms "
                    f"per tile after upload, upload {self.phase_upload_s:.1f}s "
                    f"({self.uploads_slow} slow), in-flight cap {self._aimd.maximum}, "
                    f"{self._aimd.setbacks} window setback(s)",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
                if self._self_ex_settings:
                    QgsMessageLog.logMessage(
                        "Auto detection: tree second pass - "
                        f"{self.self_ex_sent} sent, {self.self_ex_used} used, "
                        f"{self.self_ex_no_confident} without a confident mask, "
                        f"fallbacks {self.self_ex_fallback or 'none'}, "
                        f"{self.self_ex_upload_bytes / 1048576:.1f} MB, "
                        f"{self.self_ex_answer_s:.1f}s summed",
                        "AI Segmentation", level=Qgis.MessageLevel.Info,
                    )




                if self.loop_phase_s:
                    parts = ", ".join(
                        f"{name} {value:.1f}s" for name, value
                        in sorted(self.loop_phase_s.items(),
                                  key=lambda kv: -kv[1]))
                    QgsMessageLog.logMessage(
                        "Auto detection: loop summary - total "
                        f"{sum(self.loop_phase_s.values()):.1f}s: {parts}",
                        "AI Segmentation", level=Qgis.MessageLevel.Info,
                    )
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            self._report_convert_failures()




            try:
                self._close_convert_pool(0.0, emit=False)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            self._close_encode_ahead()
            self._close_clean_image_pool()
            self._drop_prespawned_children()
            client = getattr(self, "_client", None)
            if client is not None:
                try:



                    self._run_nam = None




                    client.release_thread_nam()
                except Exception:  # noqa: BLE001
                    pass  # nosec B110
