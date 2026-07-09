









































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
    _RESPLIT_TIME_RATIO,
    _SUBDIV_MAX_DEPTH,
    AutoRescanPolicyMixin,
)
from .auto_worker.retry_policy import (
    _AIMD_MIN,
    _AIMD_START,
    _BACKEND_UNAVAILABLE_DELAY_S,
    _BACKEND_UNAVAILABLE_RETRIES,
    _BUSY_JITTER,
    _DEFAULT_MAX_WAIT_S,
    _DEFAULT_POLL_INTERVAL_S,
    _HANDOFF_MIN_DELAY_S,
    _HANDOFF_OPEN_WINDOW_MAX_S,
    _MAX_RATE_LIMIT_RETRIES,
    _MIDRUN_OFFLINE_STREAK,
    _MIN_POLL_BACKOFF_S,
    _QUEUE_RETRY_BUDGET_S,
    _REFUSAL_WINDOW,
    _UPLOAD_SLOW_S,
    _WINDOW_HINT_MAX,
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
    "_SUBDIV_MAX_DEPTH",
    "_RESPLIT_TIME_RATIO",
    "_MAX_TILE_COVERAGE",
    "_HARD_TILE_COVERAGE",
    "_HARD_COVER_SHAPE_ESCAPE",
    "_COMPACT_MIN_FILL",
    "_TILE_SPAN_FRACTION",
    "_MIN_KEEP_PX",
    "_UPLOAD_SLOW_S",
    "_DEFAULT_POLL_INTERVAL_S",
    "_DEFAULT_MAX_WAIT_S",
    "_MIN_POLL_BACKOFF_S",
    "_AIMD_START",
    "_AIMD_MIN",
    "_WINDOW_HINT_MAX",
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
        self._geo_transform = geo_transform
        self._crs_authid = crs_authid



        self._transform_context = transform_context


        self._quota_refusal: dict | None = None


        self._distance_area = None






        self._ground_kx = 1.0
        self._ground_ky = 1.0
        self._prompt = prompt
        self._auth = auth
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
        self._gsd = gsd
        self._merge_separate = merge_separate





        self._collect_raw = bool(collect_raw)





        self._return_semantic = bool(return_semantic)





        self._mask_scale = int(mask_scale) if isinstance(mask_scale, int) else 1





        self._client_meta = client_meta if isinstance(client_meta, dict) else None
        self._tile_clean_image: dict[int, str] = {}







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
        self._gate_stats: dict = {}



        self._seam_min_dim = seam_min_dim



        self._merge_scalars = merge_scalars if isinstance(merge_scalars, dict) else {}



        self._clip_geom = None
        self._clip_engine = None
        self._clip_local = threading.local()



        self._stat_lock = threading.Lock()


        self._convert_pool: TileConvertPool | None = None


        self._convert_prespawned: TileConvertProcessPool | None = None


        self._convert_pool_is_processes = False






        self._subdivide_budget = max(0, int(subdivide_budget))





        from ..core import detection_policy as _dp
        from ..core.tile_manager import (
            SUBDIVIDE_MIN_PARENT_PX,
            SUBDIVIDE_OVERLAP_FRACTION,
        )
        self._prefilter = _dp.gate_prefilter_config()
        self._max_masks = _dp.max_masks_per_tile(_MAX_MASKS_PER_TILE)
        self._mask_cap_trigger = int(
            _dp.mask_cap_trigger_frac(_MASK_CAP_TRIGGER_FRAC) * self._max_masks)
        self._subdiv_max_depth = _dp.subdiv_max_depth(_SUBDIV_MAX_DEPTH)
        self._resplit_time_ratio = _dp.resplit_time_ratio(_RESPLIT_TIME_RATIO)

        self._run_started_at = 0.0
        self._paid_tiles_total = 0
        self._paid_tiles_done = 0
        self._resplit_deadline = 0.0
        self._resplit_dropped = 0
        self._max_tile_coverage = _dp.max_tile_coverage(_MAX_TILE_COVERAGE)
        self._hard_tile_coverage = _dp.hard_tile_coverage(_HARD_TILE_COVERAGE)
        self._hard_cover_shape_escape = _dp.hard_cover_shape_escape(
            _HARD_COVER_SHAPE_ESCAPE)
        self._subdiv_overlap = _dp.subdivide_overlap_fraction(
            SUBDIVIDE_OVERLAP_FRACTION)
        self._subdiv_min_parent_px = _dp.subdivide_min_parent_px(
            SUBDIVIDE_MIN_PARENT_PX)
        self._compact_min_fill = _dp.compact_min_fill(_COMPACT_MIN_FILL)
        self._tile_span_fraction = _dp.tile_span_fraction(_TILE_SPAN_FRACTION)
        self._min_keep_px = _dp.min_keep_px(_MIN_KEEP_PX)




        self._min_keep_floor_m2 = _dp.min_keep_floor_m2(0.0)



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


        self._window_hint_max = _dp.window_hint_max(_WINDOW_HINT_MAX)








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
            start=_dp.aimd_start(_AIMD_START), minimum=_dp.aimd_min(_AIMD_MIN),
            maximum=self._max_concurrent,
            cooldown_cycles=_td.aimd_cooldown_cycles(DEFAULT_COOLDOWN_CYCLES),
        )



        self._window_hint: int | None = None
        self._window_hint_logged = False




        self.last_tile_balance: dict | None = None




        self._fastfail = OfflineFastFail(
            threshold=_td.aimd_failure_threshold(DEFAULT_FAILURE_THRESHOLD))





        self._render_attempts: dict[int, int] = {}
        self._render_deferred: deque = deque()


        self._refusal_window: deque = deque(maxlen=_dial_in_range(
            "tuning.network.refusal_window", _REFUSAL_WINDOW, 4, 64))

        self._stop_requested = False




        self._stop_reason: str | None = None




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
        self._upload_slow_s = float(_dial_in_range(
            "detection_policy.network.upload_slow_s", _UPLOAD_SLOW_S, 1.0, 120.0))





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


        self._last_queue_emit: tuple[int, int, int] | None = None







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





        if self._stop_reason is None:
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
            self._drop_prespawned_children()
            client = getattr(self, "_client", None)
            if client is not None:
                try:



                    self._run_nam = None




                    client.release_thread_nam()
                except Exception:  # noqa: BLE001
                    pass  # nosec B110
