"""AutoDetectionWorker: cloud detection of all tiles in one run.

Submits tiles concurrently (bounded by max_concurrent), polls until
each tile completes or fails, then emits decoded results.

Thread safety notes:
- Per-tile JIT render (default): the caller passes a `tile_renderer` callback
  bound to a main-thread bridge. The worker calls it just before submitting each
  tile to render ONLY that tile's ground sub-extent (its bbox_native) to a
  QImage on the MAIN thread (QgsMapRendererParallelJob is GUI-thread only), then
  PNG/JPEG-encodes it here off the GUI thread. So only ~max_concurrent tiles
  render ahead and the first tile submits in ~1s instead of after the whole zone
  renders. The per-tile render is byte-identical to slicing one big zone render
  (same destination CRS, same map units per pixel, same tile origin), so the
  geo-referencing is unchanged. This per-tile bridge is the only path: the old
  whole-zone-slice fallback (a pre-rendered zone QImage) was removed.
- Qt SIGNALS are only ever emitted from this QThread. Nothing else may emit.
- The streaming path converts each finished tile's masks into geometry on a
  small converter pool (`tile_convert_pool`), NOT on the loop that drives the
  sockets: that conversion is heavier than the inference it waits on, and doing
  it inline turned the sliding window into a barrier that left the service idle
  for most of a large run. The pool runs only the pure-CPU half
  (`_convert_completed`); everything that touches run state stays on this
  thread, in `_plan_completed` before it and `_settle_converted` after it.
  Converter threads never emit a signal and never touch the tile queues.
  A prepared GEOS clip engine cannot be shared across them, so each builds its
  own (`_clip_for_thread`), and the two run-wide accumulators they feed are
  merged under `_stat_lock`. The QgsGeometry calls themselves are safe on
  several threads at once only from QGIS 3.38 on, where QGIS keeps its GEOS
  context in thread-local storage (`QgsGeosContext::get()`) and calls the
  reentrant GEOS API throughout, so no state is shared between them (a
  PREPARED engine is still the exception, because it caches inside the
  geometry instance it was built from). Below 3.38 that guarantee does not
  hold, so `__init__` forces the pool down to a single worker regardless of
  the served count (see `_CONVERT_WORKERS_CEILING`).

HTTP stack: everything goes through the QGIS network layer (TerraLabClient's
QgsBlockingNetworkRequest paths, plus QgsNetworkAccessManager for the
streaming path), so proxy/TLS settings configured in QGIS are inherited.
No raw requests/urllib transport in this module.
"""
from __future__ import annotations

import itertools
import logging
import math
import random
import threading
import time
import uuid
from collections import deque

from qgis.core import Qgis
from qgis.PyQt.QtCore import (
    QThread,
    pyqtSignal,
)

from ..core.error_policy import (
    BACKEND_UNAVAILABLE_CODES,
    EXHAUSTED_CODES,
    OFFLINE_STOP_CODE,
    RUN_FATAL_CODES,
    TRANSIENT_CODES,
)
from ..core.server_dials import dial_bool as _dial_bool
from ..core.server_dials import feature_enabled as _feature_on
from .adaptive_concurrency import AdaptiveConcurrency, OfflineFastFail
from .tile_convert_pool import TileConvertPool
from .tile_render_bridge import TileRenderBridge  # noqa: F401

logger = logging.getLogger(__name__)

# The four code sets above all live in core/error_policy.py, where each one is a
# shipped base the server may ADD to without a plugin release. Read them there,
# including why each set exists and what it costs to get one wrong. What they
# mean HERE:
#
# TRANSIENT_CODES     retry this tile, bounded by the same
#                      _MAX_RATE_LIMIT_RETRIES ceiling as rate-limit retries
#                      (see _run_streaming / _run_batched).
# EXHAUSTED_CODES     end the run cleanly, it is not a tile failure.
# BACKEND_UNAVAILABLE_CODES
#                      retry on the small attempt count below, not on the
#                      connectivity ladder, and never feed the offline
#                      fast-fail: the link is fine, the service is warming.
#                      Billing-safe, the rejection is raised before any charge.
# RUN_FATAL_CODES     stop the whole run; anything else rejects one tile.
# OFFLINE_STOP_CODE   client-side sentinel for the offline fast-fail.

# A handful of attempts spaced a short beat apart cover the cold window without
# stalling the run; the delay is jittered so the first tiles of a run (all
# rejected at once) do not retry in one synchronized wave. Both are client
# fallbacks, server-overridable via the top-level `network` policy.
_BACKEND_UNAVAILABLE_RETRIES = 3
_BACKEND_UNAVAILABLE_DELAY_S = 1.75

# Rate limiting is expected on large runs (many tiles, shared per-key server
# limit). Retried much longer than transient network errors, honoring the
# server's retry_after; if still limited, the tile is skipped, never the run.
_MAX_RATE_LIMIT_RETRIES = 8
# Launch-spike waiting room: a RATE_LIMITED that is really "the service queue is
# full, you're in line" (the server sends queue_position/queue_depth) is retried
# on TIME, not attempts - a count cap of 8 x retry_after silently skipped tiles
# under load, which surfaced as holes in the result. Per-tile patience window
# from the tile's FIRST busy answer; within it the tile keeps its place in line.
_QUEUE_RETRY_BUDGET_S = 300.0
# Retry delays are jittered (AWS full-jitter rationale): N clients told
# "retry in 5s" must not all come back at t+5.000 in one synchronized wave.
_BUSY_JITTER = (0.85, 1.30)
# How many upcoming tiles the streaming path asks the main thread to render
# AHEAD of need (async jobs, overlapped with the in-flight inference). This is
# a FLOOR, not the value: the run uses whichever is larger, this or the number
# of requests it keeps in flight (see __init__). A prefetch narrower than the
# network window serializes the run on renders, because filling the window
# blocks on a render slot, and an online basemap makes a render wait for every
# source image it needs.
_PREFETCH_DEPTH = 2
# Converter threads for the streaming path's mask -> geometry stage.
# 0 = size from the machine (tile_convert_pool.default_workers).
_CONVERT_WORKERS = 0
# Hard ceiling on the served dial, independent of QGIS version: it has a
# floor (max(0, ...) below) but no upper bound of its own, and every worker
# is a thread that touches GEOS.
_CONVERT_WORKERS_CEILING = 8
# QGIS version (Qgis.QGIS_VERSION_INT, MAJOR*10000 + MINOR*100 + PATCH) from
# which GEOS keeps its context in thread-local storage (see the module
# docstring). Below this the pool is forced to a single worker.
_GEOS_THREAD_LOCAL_MIN_VERSION = 33800
# How many finished-but-unconverted tiles may queue before the run loop stops
# firing new tiles and spends the cycle draining instead. Converters slower than
# the network would otherwise only grow a backlog of undelivered masks in
# memory, which buys nothing: this bounds it, per converter thread.
_CONVERT_BACKLOG_PER_WORKER = 4
# Ceiling on the end-of-run wait for the last conversions. They carry billed
# geometry, so a normal end waits for them; this only stops a wedged converter
# from holding the terminal open for good.
_CONVERT_DRAIN_BUDGET_S = 90.0
# On a USER cancel we stop firing new tiles at once, then wait this long for
# the handful ALREADY in flight to land so their billed masks are kept, not
# thrown away. Bounded so one hung reply can never hold the stop open: past it
# the stragglers are aborted, and an abort does not undo the charge. The
# in-flight set is <= max_concurrent and each direct tile is ~1s, so a real
# cancel drains in well under this ceiling; kept short so Cancel feels prompt
# (a tile still computing past it is aborted rather than making the user wait).
_STOP_DRAIN_BUDGET_S = 2.5
# Stop reasons whose still-open requests are read before the sockets are
# released. The service bills a tile the moment it accepts the request, so a
# reply aborted unread is a detection already paid for. A fatal or offline stop
# stays out: it has nothing to salvage.
#
# "stalled" is out too, and not because those requests are unbilled. The stall
# watchdog detaches tile_completed before the worker reaches its drain, so
# every tile read here would be emitted into nothing while the wind-down grew
# by the drain budget. Adding it pays only together with a controller that
# waits for the wind-down before it finalizes.
_BILLED_DRAIN_STOP_REASONS = ("user", "exhausted")
# The service caps the number of instances it returns per inference, which
# silently truncates dense scenes at its default. Request the full cap and let
# tile sizing (not this number) keep the expected object count well under it.
_MAX_MASKS_PER_TILE = 200
# Saturation trigger: a truncated tile rarely lands EXACTLY on the ceiling,
# because the model fills all its slots and then its own score filtering
# drops a few, so the trigger sits below the cap with margin. Anything at
# or above it is treated as truncated for both the re-split ladder and the
# review dense hint. Client fallback; the run value is server-overridable
# (seed.saturation.cap_trigger_frac), resolved per run in __init__.
_MASK_CAP_TRIGGER_FRAC = 0.80
# Saturated-tile re-split recursion ceiling. Depth 1 quarters the object count
# per inference; depth 2 covers extreme dense scenes. Past that the quadrants
# are too small/interpolated to add signal.
_SUBDIV_MAX_DEPTH = 2
# The re-split tail is free, but it is not free of TIME: it runs after the paid
# grid, on the same machine, and a dense zone can queue more quadrants than the
# grid had tiles. It gets this share of what the paid grid itself took, and then
# it stops, whatever budget is left. Server-overridable
# (seed.saturation.resplit_time_ratio); 0 or less disables the clock.
_RESPLIT_TIME_RATIO = 1.0
# Fraction of a tile above which a single mask is treated as a whole-tile "everything"
# failure (a near-whole-tile blob on edge-to-edge uniform texture - dense forest,
# water - not an individual object) and dropped. Applied ONLY in SEPARATE/count
# mode: there a whole-tile mask is unambiguously the failure mode and, left in,
# the seam-merger chains adjacent ones into multi-tile mega-blocks. In MAP/merge
# mode it is skipped so a genuine whole-tile lake/field mask is kept. 0.55 spares
# real objects that fill up to half a tile.
_MAX_TILE_COVERAGE = 0.55
# Above this the mask is a fill-everything failure regardless of shape: even a
# tightly-framed real building leaves streets/margins, so >80% of a tile is
# texture, not an object. Between 0.55 and 0.80 a compactness check decides.
_HARD_TILE_COVERAGE = 0.80
# Share of its oriented bounding box a large mask must fill for that
# compactness check to keep it as a real solid object. Client fallback; the run
# value is server-overridable (seed.saturation.compact_min_fill).
_COMPACT_MIN_FILL = 0.85
# Share of the tile a mask's bounding box must span, in BOTH directions, for the
# tile itself to count as what drew the outline. Such a mask never reaches the
# compactness check: it fills its oriented box perfectly, because it IS a
# rectangle, so that check would keep exactly the shape it exists to drop.
# Client fallback; the run value is server-overridable
# (seed.saturation.tile_span_fraction).
_TILE_SPAN_FRACTION = 0.95
# Anti-sliver floor: a detection smaller than this many pixels on a side is
# sub-pixel noise, not an object, and is dropped. Client fallback; the run
# value is server-overridable (seed.saturation.min_keep_px).
_MIN_KEEP_PX = 1.5

_DEFAULT_POLL_INTERVAL_S = 2.0
_DEFAULT_MAX_WAIT_S = 120.0
# Floor for the coalesced per-cycle poll back-off. The server's retry_after is
# honoured, but never below this, so a tiny/zero hint can't turn the poll loop
# into a tight status-GET storm that trips the server's read rate bucket.
_MIN_POLL_BACKOFF_S = 0.5

# Adaptive concurrency (AIMD, see adaptive_concurrency.AdaptiveConcurrency). A run
# opens NARROW and grows one step per clean cycle up to max_concurrent, halving on
# a timeout / latency setback. Opening at the full width punished slow-link users:
# N concurrent tile uploads split the uplink into N starving trickles that all
# time out together and re-upload the same bytes. Starting at _AIMD_START and
# climbing keeps healthy, already-warm links at the full width within a few cycles
# while a bad link collapses toward 1-2. max_concurrent still matches the deployed
# max concurrency (extra in-flight tiles would only queue at the instance); bump
# max_concurrent (at the launch call site) if that is raised for launch.
_AIMD_START = 3
_AIMD_MIN = 1

# A code outside RUN_FATAL_CODES is a PER-TILE rejection: that one tile is
# skipped and the run continues, because one bad tile (a 4xx for its image, a
# code this client version does not know) must never kill a paid multi-tile
# run. This streak of consecutive rejections with no success in between still
# aborts the run, so a run-level code the plugin has not been told about costs
# at most a handful of requests.
_MAX_CONSECUTIVE_TILE_FATALS = 5

# A tile whose JIT render comes back blank or empty is NOT dropped on the
# first try: on an online basemap that usually means the imagery for that
# area was not downloaded yet (never viewed at detection zoom, provider
# hiccup, burst throttling), and a short-delay re-render succeeds once the
# provider has fetched it. Left unretried, a run over a never-viewed area can
# silently skip most of its tiles. Bounded so a REAL nodata hole (mosaic gap)
# only costs a few cheap local renders before the existing skip path. The
# delay DOUBLES per attempt (1.5s, 3s, 6s): a provider hiccup lasting a few
# seconds hits several concurrently-rendered tiles at once, and fixed-delay
# retries all landed back inside the same hiccup window.
_RENDER_RETRY_MAX = 3
_RENDER_RETRY_DELAY_S = 1.5
# After ANY blank/failed render, hold the render prefetch for a beat: a blank
# means the imagery provider is struggling, and stacking more concurrent
# fetches into that window makes the burst worse. The tile in hand still
# renders synchronously.
_PREFETCH_HOLDOFF_S = 4.0
# Seconds a tile may spend waiting on its own imagery before the run reads the
# link as too narrow for the current number of concurrent fetches. N fetches
# sharing one uplink each get a fraction of it, so they approach the render
# deadline together, expire together, and every one of those tiles is retried
# and then dropped as a coverage hole. Narrowing the window instead makes the
# same tiles arrive later but whole. Above the delay a healthy link ever shows,
# so nothing narrows on a good connection.
_RENDER_SLOW_S = 8.0

# Mid-run offline abort threshold. After the first successful tile the
# offline fast-fail no longer trips at its small pre-success threshold; a
# brief wifi blip must be absorbed by the normal retry backoff. But a link
# that stays dead used to grind every remaining tile's full retry budget
# (minutes of "Detecting...") before the run gave up. This streak of
# CONSECUTIVE hard-connectivity failures (roughly 1-2 minutes of continuous
# outage across the in-flight window) ends the run instead; the billed
# partials are salvaged into the review either way.
_MIDRUN_OFFLINE_STREAK = 30

# Empty-tile scan gate (see _run_gate_scan). Full-res renders produced during
# the scan phase are kept for the detect phase so a kept tile renders once;
# the cache is bounded (entries pop as consumed) to keep run memory flat at
# any tile count. Scan renders retry only briefly: a member that fails to
# render simply stays unscanned and falls open to the normal detect path,
# where the full render-retry ladder applies.
_GATE_RENDER_CACHE_MAX = 64
_GATE_SCAN_RENDER_TRIES = 2


def _as_int(value, default: int = -1) -> int:
    """Lenient int coercion for optional server-sent queue fields."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float) -> float:
    """Lenient float coercion for optional server-sent timing fields. A JSON
    null arrives as None, which dict.get(key, default) hands back instead of
    the default, so plain float() would raise and kill a paid run."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_convert_workers(
    served: int, qgis_version_int: int, ceiling: int = _CONVERT_WORKERS_CEILING
) -> int:
    """Converter-pool worker count: the served dial, hard-capped regardless
    of version, then forced to a single worker below the QGIS version whose
    GEOS context is thread-local (see the module docstring). 0 (no server
    override) is passed through so the pool sizes itself from the machine."""
    workers = min(max(1, ceiling), max(0, served))
    if qgis_version_int < _GEOS_THREAD_LOCAL_MIN_VERSION:
        return 1
    return workers


class AutoDetectionWorker(QThread):
    """Submit all pre-rendered tiles to the cloud API and collect results.

    Signals (all emitted on the worker thread; connect with
    Qt.ConnectionType.QueuedConnection when the slot touches QGIS objects):

        tile_completed(tile_idx, detections)
            Emitted after each tile is decoded AND converted to geometry on
            this worker thread (mask -> refine -> polygonize -> clip-to-zone ->
            repair). detections: list of (geom_wkb: bytes, score: float). WKB is
            an unambiguous, allocation-clean way to move geometry across the
            queued signal (QgsGeometry/QgsSpatialIndex are value/thread-safe per
            the QGIS API, but WKB sidesteps any COW/refcount subtlety at the
            boundary). The GUI rehydrates with QgsGeometry().fromWkb(wkb).

        all_tiles_finished(results)
            Emitted once when all tiles have been processed (success, skip,
            or error). results is always an empty list: it is a completion
            signal only. Per-tile detections are delivered via tile_completed;
            no consumer reads this payload.

        progress(completed, total)
            Emitted after each tile finishes (success OR skip).

        warning(message)
            Non-fatal issue (e.g. a single tile timed out).

        error(message)
            Fatal error that stopped the run early.

        credits_exhausted(credits_remaining)
            Credits or free quota ran out.  credits_remaining may be 0.

        cancelled()
            The run was cancelled via request_stop().

        queue_state(position, depth, eta_seconds)
            Server-load feedback for the "you're in line" UI. position >= 1 is
            a real place in the server's fair queue, -1 means busy-but-unknown
            (old server / platform rejection / cold start), and (0, 0, 0)
            clears the state once tiles flow again.

        run_phase(name)
            Which half of the run the wait belongs to before the first tile
            answers: "imagery" while the basemap is still being fetched,
            "detecting" from the first submit on.
    """

    tile_completed = pyqtSignal(int, list)
    all_tiles_finished = pyqtSignal(list)
    progress = pyqtSignal(int, int)
    # Ground being re-read at a finer scale: (tile index, bbox in the run CRS as
    # (minx, miny, maxx, maxy), active). A saturated tile's own objects are
    # withheld until its quadrants land, so that ground goes bare while the run
    # works hardest on it; the GUI marks it instead of showing a hole. Index -1
    # with active False means "clear everything", sent at the terminal.
    rescan_state = pyqtSignal(int, object, bool)
    warning = pyqtSignal(str)
    error = pyqtSignal(str)
    credits_exhausted = pyqtSignal(int)
    cancelled = pyqtSignal()
    # Server-load feedback: (queue_position, queue_depth, eta_seconds).
    # position >= 1 -> a queue-aware server told us our place in line;
    # position == -1 -> server busy but no position known (old server / platform
    # 429 / cold start); (0, 0, 0) -> flowing again, clear any busy UI.
    queue_state = pyqtSignal(int, int, int)
    # Which work the user is actually waiting on before the first tile answers.
    # A run opens by fetching basemap imagery, which on a slow link is most of
    # the wait; telling the user "waking up the AI" through it names the wrong
    # cause and reads as a hang. "imagery" then "detecting", once each.
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
        parent=None,
    ):
        """Initialise the worker.

        Args:
            client_meta:     Additive, optional per-run provenance + benchmark
                             fields attached to every /predict submission (the
                             run's plugin_version, policy_rev, prompt_mode, plus
                             the drawn zone_geojson sent once on tile 0). None =
                             the payload stays byte-identical to before. Old
                             servers ignore unknown fields. When set, the
                             pre-stamp tile image is also captured (clean_image)
                             for every tile a reference stamp was composited in.
            gate_config:     Empty-tile scan gate settings resolved by the
                             plugin from server policy ({group, max_group,
                             min_score, min_pixels, max_scan_mupp}), or None =
                             gate OFF = today's behaviour, bit-identical. The
                             effective block side adapts between group and
                             max_group under the resolution cap. See
                             _run_gate_scan.
            return_semantic: When True, ask the service for a coverage map
                             alongside the per-instance masks (an additive,
                             optional request field). Set only for a map-like
                             text prompt when the server dial is on; it drives
                             the zero-instance coverage rescue in
                             _handle_completed. Default False keeps the request
                             and the result bit-identical to today.
            mask_scale:      Per-run mask-grid scale resolved once by the plugin
                             from server policy (1 = full grid, the default; 2 =
                             the validated coarser grid). Attached to every
                             /predict submission of this run via
                             cloud_detection.mask_scale_field, so a re-run over
                             the same grid stays consistent. Default 1 keeps the
                             request bit-identical to today.
            tile_renderer:   Callable (tx, ty, tw, th) -> QImage|None that renders
                             ONE tile on the main thread, called just before each
                             tile is encoded: it is the per-tile pixel SOURCE, so
                             rendering overlaps detection and the first tile
                             submits without waiting for the whole zone. The
                             returned QImage is the tile at origin (0,0), exactly
                             tw x th. Always supplied on every live path (the old
                             whole-zone-slice fallback was removed).
            tiles:           List of (x, y, w, h) tile pixel rectangles.
            geo_transform:   {"bbox": (minx, miny, maxx, maxy), "img_shape": (H, W),
                             "crs": authid}  -- bbox is standard (minx, miny, maxx, maxy).
            crs_authid:      CRS authority ID of the source raster.
            prompt:          Text prompt for detection (e.g. "tree", "building").
            auth:            Auth headers dict from get_auth_header().
            run_id:          UUID4 string; generated client-side if None.
            max_concurrent:  Maximum in-flight submissions at once.
            score_threshold: Discard masks with score below this value.
            detection_threshold: Detection-confidence cutoff sent to the server
                             (0..1). Lower = more objects/recall, higher =
                             fewer/cleaner. The cloud endpoint honours it.
            progress_offset: Added to every progress emission. A resumed run
                             passes the count of tiles already done so the
                             dock's bar continues instead of restarting at 0.
            progress_total:  Displayed total for progress emissions; defaults
                             to len(tiles). A resumed run passes the original
                             run's tile count.
            clip_polygon_wkb: WKB bytes of the drawn zone polygon (run CRS), or
                             None for the rectangle/MCP path. The worker rebuilds
                             its OWN QgsGeometry + prepared engine from this on the
                             worker thread and clips every detection to it (a
                             prepared engine is bound to its geometry instance, so
                             it cannot be passed across the thread; it is rebuilt
                             worker-side).
            gsd:             Ground sample distance (map units/px) of the run.
                             Drives the per-mask vectorization scale (simplify
                             tolerance + pinhole ceiling), together with each
                             returned mask's own grid cell when coarser.
            parent:          Optional Qt parent.
        """
        super().__init__(parent)

        # Per-tile JIT render source (main-thread bridge callback): _encode_tile
        # renders each tile on demand. Always set on every live path.
        self._tile_renderer = tile_renderer
        # Whether a rendered tile may be read as the source's "no image here"
        # placeholder card. Online sources only: a local raster has no such
        # card, and its neutral-grey tiles are real ground. Resolved once here,
        # like the other policy dials, so one run keeps one behaviour; off is
        # the pre-existing behaviour (the card is sent and billed), which is
        # what makes the switch safe to pull.
        self._skip_unavailable_tiles = bool(source_is_online) and _feature_on(
            "unavailable_tile_skip")
        # Run the per-tile hypothesis NMS in MAP mode too. OFF is what shipped:
        # MAP takes coverage as the union of every hypothesis, so a tile that
        # answers with BOTH a precise outline and a mask over the same ground
        # contributes both, and the merger unions them across the whole run into
        # one shape the review cannot take apart. Resolved once per run at
        # construction, like every other dial the worker reads.
        #
        # NOT _feature_on: that helper is fail-OPEN, so an absent key would turn
        # this ON for the whole fleet the moment the reader shipped, which is the
        # opposite of a switch whose off state is the behaviour that exists.
        # Read as a plain bool dial defaulting to False, so it takes an explicit
        # served true to arm.
        self._map_hypothesis_nms = _dial_bool("features.map_hypothesis_nms", False)
        # If the renderer is a TileRenderBridge bound method, keep its cancel()
        # so request_stop can unblock an in-progress render at once (the unload
        # deadlock guard: the main thread blocks in worker.wait() and can no
        # longer service a queued render, so the parked render_tile must be woken
        # by cancel(), not left to time out).
        self._tile_renderer_cancel = None
        # Async render API (prefetch): request without blocking, collect when
        # the tile is actually encoded. Only a real TileRenderBridge has it;
        # plain-callable renderers (tests, mocks) keep the synchronous path.
        self._render_request = None
        self._render_collect = None
        # tile_idx -> collect token of a render requested ahead of time.
        self._prefetched: dict[int, int] = {}
        # Monotonic instant before which the prefetch stays quiet (set by any
        # blank/failed render: evidence the imagery provider is struggling).
        self._prefetch_holdoff_until = 0.0
        bridge = getattr(tile_renderer, "__self__", None)
        if bridge is not None and hasattr(bridge, "cancel"):
            self._tile_renderer_cancel = bridge.cancel
        if bridge is not None and hasattr(bridge, "request_render"):
            self._render_request = bridge.request_render
            self._render_collect = bridge.collect_render
        self._tiles = tiles  # list of (x, y, w, h)
        self._geo_transform = geo_transform
        self._crs_authid = crs_authid
        # Lazily built QgsDistanceArea for the per-tile ground-resolution
        # report (see _tile_pixel_size_m); one instance serves the whole run.
        self._distance_area = None
        self._prompt = prompt
        self._auth = auth
        self._run_id = run_id or str(uuid.uuid4())
        self._max_concurrent = max(1, max_concurrent)
        self._score_threshold = score_threshold
        self._detection_threshold = detection_threshold
        # Pre-rendered, pre-masked example crops (crop QImage, label, obj_box)
        # from the plugin: crisp, well-sized, and stamped into every tile so one
        # drawn example works on all tiles. None/empty for text-only runs.
        self._exemplar_stamps_in = exemplar_stamps or []
        self._stamps: list = []                 # [(crop QImage, label, obj_box)]
        # Parallel to _stamps: each exemplar's drawn box in FULL-image pixel
        # coords (unclamped xyxy, or None). The one tile whose rect fully
        # contains a box sends it in-situ instead of pasting that crop.
        self._stamp_full_boxes: list = []
        # Parallel to _stamps: True for REGION markers (review correction
        # boxes). A region box is clipped to EVERY tile it overlaps (a partial
        # view of an area keeps its meaning) and is never pasted.
        self._stamp_regions: list = []
        self._tile_exemplars: dict = {}         # tile_idx -> [{box, label}] (tile coords)
        self._tile_stamp_norm: dict = {}        # tile_idx -> [nx0,ny0,nx1,ny1] normalized
        # Top-row band edge (first-row coverage blind-spot fix). Resolved in
        # _prepare_stamps: the top grid row bands its BOTTOM edge (instead of the
        # top, which has no row above to re-see it) when the row below overlaps
        # enough that the two bands stay clear of each other.
        self._top_stamp_ty: int | None = None
        self._stamp_bottom_top_row = False
        self._progress_offset = max(0, progress_offset)
        self._progress_total = progress_total
        # Zone clip polygon as WKB (run CRS) + the run's ground sample distance.
        # The geometry pipeline (mask -> polygon -> clip -> repair) now runs on
        # THIS worker thread, so the clip geom + prepared engine are rebuilt in
        # run() from these copied/immutable inputs; nothing GUI-thread-owned is
        # touched after start(). _clip_geom/_clip_engine are set in _run_detection.
        self._clip_polygon_wkb = clip_polygon_wkb
        self._gsd = gsd
        self._merge_separate = merge_separate
        # Raw-collect mode (exemplar-only runs): keep the per-tile hypothesis NMS
        # (it dedups the model's overlapping hypothesis stacks, not instances),
        # but leave the coverage/compactness gates and the MAP per-tile premerge
        # OFF, so the GUI receives the run's own fragments unaltered. The client
        # then decides count-vs-map from these fragments after the run.
        self._collect_raw = bool(collect_raw)
        # Coverage-map ("semantic") zero-instance rescue: request the coverage
        # map (set only for a map-like text prompt when the server dial is on)
        # and, when a tile's per-instance pass returns nothing, keep its
        # coverage mask above the floor as a single detection. Off = today's
        # behaviour, bit-identical.
        self._return_semantic = bool(return_semantic)
        # Per-run mask-grid scale (1 = full, 2 = the validated coarser grid),
        # resolved once by the plugin from server policy and applied to every
        # detection submission of this run (cloud_detection.mask_scale_field
        # decides the wire value: 2 or absent). One run = one scale, so a
        # re-detect over the same grid compares polygons on the same grid.
        self._mask_scale = int(mask_scale) if isinstance(mask_scale, int) else 1
        # Additive, optional per-run provenance + benchmark data (see the
        # client_meta docstring). None keeps every submission byte-identical to
        # today. _tile_clean_image holds the base64 pre-stamp image for each
        # tile a reference stamp was composited into, filled in _encode_tile and
        # read once per submission; only populated when client_meta is set.
        self._client_meta = client_meta if isinstance(client_meta, dict) else None
        self._tile_clean_image: dict[int, str] = {}
        # The run's private network manager, held for as long as replies must
        # stay readable (see _run_detection), and the tiles already re-posted
        # once because their reply was destroyed under them (see _read_reply).
        self._run_nam = None
        self._dead_reply_tiles: set[int] = set()
        # Empty-tile scan gate (policy-resolved by the plugin, None = OFF =
        # today's behaviour, bit-identical). When set, _run_gate_scan sends one
        # packed low-res scan per tile group before detection, marks the empty
        # tiles to skip (_gate_skip) and records which tiles a completed scan
        # already paid for (_gate_prepaid: their detect requests carry
        # charge_tiles=0 so the run's total charge stays the full grid).
        self._gate_config = gate_config if isinstance(gate_config, dict) else None
        self._gate_skip: set[int] = set()
        self._gate_prepaid: set[int] = set()
        # Degenerate-tile prefilter (all no-data / per-band uniform renders
        # settle as empty with no request at all). Unlike the scan gate it is
        # fail-OPEN (safe by construction, see tile_is_degenerate_array), so it
        # runs on every run unless the server kill switch turns it off.
        # Resolved once at construction like the other policy dials below.
        self._prefilter_skip: set[int] = set()
        # Full-res renders produced during the scan phase, reused by the detect
        # phase so a kept tile renders once. Bounded (entries are popped as
        # consumed and never exceed _GATE_RENDER_CACHE_MAX) to keep run memory
        # flat at any tile count.
        self._gate_tile_bytes: dict[int, tuple] = {}
        self._gate_stats: dict = {}
        # Inter-tile overlap span (ground units), the run merger's size-aware
        # anti-over-merge gate. Used by the MAP-mode per-tile pre-merge below
        # so its local merger applies the exact same policy as the GUI's.
        self._seam_min_dim = seam_min_dim
        # Merge/dedup scalars resolved by the plugin (server policy or generic
        # fallback). Passed to the per-tile merge helpers below; absent keys fall
        # through to those callees' own signature defaults.
        self._merge_scalars = merge_scalars if isinstance(merge_scalars, dict) else {}
        # The worker thread's own clip pair (read directly by
        # _quad_intersects_zone). Converter threads never touch these; they
        # build their own through _clip_local (see _clip_for_thread).
        self._clip_geom = None
        self._clip_engine = None
        self._clip_local = threading.local()
        # Guards the two run-wide accumulators several converter threads write
        # (raw_detections_total, observed_mask_gsd). Held for a single merge per
        # tile, never around the conversion itself.
        self._stat_lock = threading.Lock()
        # Converter pool for the streaming path, created at run start and closed
        # in run()'s finally. None on the batched path and before the run opens.
        self._convert_pool: TileConvertPool | None = None
        # Saturated-tile re-split: when a tile returns the model's per-inference
        # ceiling, the objects beyond it were silently truncated. With budget
        # left, that tile is re-queued as 4 overlapping quadrants rendered at
        # 2x the run scale (fewer objects per inference AND larger apparent
        # size), recursively up to _SUBDIV_MAX_DEPTH. The budget is extra tiles
        # (= extra credits) this run may spend on re-splits; 0 disables.
        self._subdivide_budget = max(0, int(subdivide_budget))
        # Saturation tuning, server-overridable (seed.saturation in the
        # detection policy) with the module constants as the client fallbacks.
        # Resolved ONCE here on the construction (main) thread: the policy read
        # is a memory-cache lookup, and the values must stay constant for the
        # whole run.
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
        # Armed when the paid grid is complete (see _run_detection).
        self._run_started_at = 0.0
        self._paid_tiles_total = 0
        self._paid_tiles_done = 0
        self._resplit_deadline = 0.0
        self._resplit_dropped = 0
        self._max_tile_coverage = _dp.max_tile_coverage(_MAX_TILE_COVERAGE)
        self._hard_tile_coverage = _dp.hard_tile_coverage(_HARD_TILE_COVERAGE)
        self._subdiv_overlap = _dp.subdivide_overlap_fraction(
            SUBDIVIDE_OVERLAP_FRACTION)
        self._subdiv_min_parent_px = _dp.subdivide_min_parent_px(
            SUBDIVIDE_MIN_PARENT_PX)
        self._compact_min_fill = _dp.compact_min_fill(_COMPACT_MIN_FILL)
        self._tile_span_fraction = _dp.tile_span_fraction(_TILE_SPAN_FRACTION)
        self._min_keep_px = _dp.min_keep_px(_MIN_KEEP_PX)
        # Ground floor under the sliver drop, m2. The pixel floor scales with
        # gsd^2 and stops dropping anything on very fine imagery; this keeps a
        # resolution-independent minimum. 0.0 = pixel floor alone (shipped
        # behaviour); served value only.
        self._min_keep_floor_m2 = _dp.min_keep_floor_m2(0.0)
        # Confidence a whole-tile MAP mask must reach to be kept. 0.0 = off,
        # which is what shipped. Server-overridable, so the value can be picked
        # from the scores the run logs and retuned without a release.
        self._map_cover_score_floor = _dp.map_cover_score_floor(0.0)
        # Per-mask vectorization dials (review block), resolved ONCE here for
        # the same reason: a mid-run policy refresh must not change the shape
        # of the polygons a single run produces. 0.0 means "no server value",
        # and the vectorizer then applies its own constant, so the client
        # fallback stays in one place (core.cloud_detection).
        self._pinhole_m = _dp.pinhole_fill_m(0.0)
        self._tile_simplify_mult = _dp.tile_simplify_mult(0.0)
        # Coverage floor for the zero-instance rescue, server-overridable
        # (review.semantic_rescue.coverage_floor) with one generic client
        # fallback. Resolved once here so it stays constant for the whole run.
        self._semantic_coverage_floor = _dp.semantic_rescue_coverage_floor()
        # Network/queue budgets, server-overridable (top-level `network` in
        # the policy): an operations dial to loosen retry budgets fleet-wide
        # during a backend incident, no plugin release. Same once-per-run
        # resolution as the saturation block above.
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
        # Pacing dials on the same `network` block: what you retune when the
        # backend or an imagery provider degrades. Same once-per-run read.
        self._busy_jitter = _dp.busy_jitter(_BUSY_JITTER)
        # Default to the width of the network window: preparing fewer images
        # than the run has requests in flight caps its throughput at the render
        # time, whatever the service does. An explicit server value still wins.
        self._prefetch_depth = _dp.prefetch_depth(
            max(_PREFETCH_DEPTH, self._max_concurrent))
        # Ceiling, not the live width: _render_window narrows it on a link that
        # cannot feed that many basemap fetches at once (see the attribute).
        self._render_window = AdaptiveConcurrency(
            start=self._prefetch_depth, minimum=1, maximum=self._prefetch_depth)
        # A render this slow means the imagery fetches are competing for one
        # narrow uplink rather than each finishing. Server-tunable on the same
        # network block.
        self._render_slow_s = _dp.render_slow_s(_RENDER_SLOW_S)
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
        # Total-time deadline for ONE streaming reply. The timeout the request
        # itself carries measures INACTIVITY, so a service that trickles bytes
        # never trips it and the reply keeps its window slot for the session.
        # Never shorter than that timeout, or a slow but healthy answer would
        # be dropped after the user paid for it.
        self._stream_reply_budget_s = max(
            self._poll_max_wait_s,
            _dp.submit_timeout_ms(int(self._poll_max_wait_s * 1000)) / 1000.0,
        )
        self._min_poll_backoff_s = _dp.min_poll_backoff_s(_MIN_POLL_BACKOFF_S)
        self._gate_render_cache_max = _dp.gate_render_cache_max(_GATE_RENDER_CACHE_MAX)
        self._prefetch_holdoff_s = _dp.prefetch_holdoff_s(_PREFETCH_HOLDOFF_S)
        self._max_tile_fatals = _dp.max_consecutive_tile_fatals(
            _MAX_CONSECUTIVE_TILE_FATALS)
        self._render_retry_max = _dp.render_retry_max(_RENDER_RETRY_MAX)
        self._render_retry_delay_s = _dp.render_retry_delay_s(
            _RENDER_RETRY_DELAY_S)
        self._gate_scan_render_tries = _dp.gate_scan_render_tries(
            _GATE_SCAN_RENDER_TRIES)
        self._tile_depth: dict[int, int] = {}      # tile_idx -> re-split depth
        self._tile_outsize: dict[int, tuple[int, int]] = {}  # idx -> render px
        self._pending_subtiles: list = []          # [(spec, depth, parent)] queued
        # A re-split parent's own detections are WITHHELD, not emitted: at its
        # (truncated) scale several touching objects often come back as ONE
        # coarse mask, and unioning that blob with the quadrants' clean
        # separated objects bridged neighbours into chains. The quadrants
        # REPLACE the parent; its withheld detections are flushed at the
        # terminal only if NO quadrant delivered (so a paid parent is never
        # lost to a failed ladder).
        self._withheld: dict[int, list] = {}       # parent idx -> detections
        self._parent_of: dict[int, int] = {}       # child idx -> parent idx
        self._parents_with_child_results: set[int] = set()
        # base tile idx -> quadrant inferences still owed for that ground. Drives
        # the canvas mark over ground whose objects are withheld (rescan_state).
        self._rescanning: dict[int, int] = {}
        self.tiles_subdivided = 0                  # parents re-split (plain int)
        # Tiles still at the ceiling AFTER the re-split ladder (depth/budget
        # exhausted, or re-split disabled): the residual truncation the review
        # dense hint reports. Plain ints, GIL-safe.
        self.tiles_capped_final = 0

        # AIMD in-flight width controller: opens narrow (_AIMD_START), grows per
        # clean cycle up to max_concurrent, halves on a timeout/latency setback.
        # Drives effective_cap (_run_batched) and the window (_run_streaming).
        self._aimd = AdaptiveConcurrency(
            start=_dp.aimd_start(_AIMD_START), minimum=_dp.aimd_min(_AIMD_MIN),
            maximum=self._max_concurrent,
        )
        # Consecutive hard-connectivity failure counter: aborts a doomed offline
        # run in a few seconds (see _run_batched / _run_streaming) instead of
        # grinding every tile's full retry budget. Only consulted while zero
        # tiles have succeeded.
        self._fastfail = OfflineFastFail()

        # Render-retry ladder for blank/failed JIT renders (see
        # _RENDER_RETRY_MAX): tile_idx -> attempts so far, plus the deferred
        # deque of (not_before_monotonic, tile_idx, spec) waiting out the
        # re-render delay. Both are worker-thread-only state.
        self._render_attempts: dict[int, int] = {}
        self._render_deferred: deque = deque()

        self._stop_requested = False
        # Why the run stopped early: "user" | "error" | "exhausted". Only a
        # user stop emits cancelled at the end of run(); error and exhausted
        # already emitted their own signal, and a trailing cancelled would
        # let its handler wipe the banner those handlers just showed.
        self._stop_reason: str | None = None
        # True once a terminal signal has been SENT for this run (cancelled,
        # error, credits_exhausted or all_tiles_finished). The main thread
        # finalizes on the first one, so a second one rebuilds run state on a
        # run that is already over. One run, one terminal.
        self._terminal_sent = False
        # Tiles that reached server status "completed" (zero-mask included,
        # they consume a credit). Tiles the client saw fail or time out are
        # excluded, so this is the count DELIVERED, which is a floor on what
        # was charged, not the exact charge.
        self.tiles_succeeded = 0
        # Raw decoded detections across the run, BEFORE the per-tile NMS /
        # MAP pre-merge, so the run-summary "raw detection(s)" keeps meaning
        # "what the model returned" (the GUI now receives a reduced stream).
        self.raw_detections_total = 0
        # Tiles dropped BEFORE submit because the render came back essentially
        # blank/nodata (uniform fill, mosaic gap, out-of-footprint): never
        # submitted, never billed. Surfaced once at run end as
        # "Skipped N empty tiles (not charged)". Plain ints, GIL-safe.
        self.tiles_skipped_blank = 0
        # Tiles the degenerate prefilter settled as EMPTY results (all no-data
        # or per-band uniform at full resolution, so provably objectless): they
        # complete like a zero-mask detection, with no request and no retry
        # ladder. Distinct from tiles_skipped_blank, which is the broader
        # near-uniform drop that still gets the render-retry ladder and is
        # surfaced as a load problem. Plain int, GIL-safe.
        self.tiles_prefiltered = 0
        # Tiles the empty-tile scan gate decided held nothing, from a PACKED
        # LOW-RES scan, and that therefore never got a detection pass. Unlike
        # every other counter here these tiles were CHARGED (the scan they rode
        # in is billed), and unlike the degenerate prefilter the decision is a
        # judgement on downsampled pixels, not a proof. So a gate that reads a
        # tile wrong costs the user a tile-shaped hole in a run they paid for,
        # and until this counter existed the only trace was a logger.debug line
        # no user or log reader ever sees. Plain int, GIL-safe.
        self.tiles_gate_skipped = 0
        # Masks the whole-tile blob guard dropped, run-wide, and the split by
        # which test fired. The guard runs in SEPARATE/count mode only, where a
        # near-whole-tile mask is normally an "everything" failure on uniform
        # texture. A large real parcel is the false positive it cannot tell
        # apart, and dropping one leaves a tile-shaped hole with nothing in it.
        # The thresholds behind these are server dials (max_tile_coverage,
        # hard_tile_coverage, tile_span_fraction, compact_min_fill), so what a
        # retune needs is the count it moved. Written under _stat_lock.
        self.masks_dropped_whole_tile = 0
        self.masks_dropped_hard_cover = 0
        self.masks_dropped_tile_span = 0
        self.masks_dropped_not_compact = 0
        # MAP-mode counterpart, counted and never acted on: masks past
        # max_tile_coverage that MAP keeps by design. A run whose object outlines
        # are correct can still come back as one shape covering everything,
        # because the merger unions these with the good outlines and the review
        # has no filter that separates them again. Written under _stat_lock.
        self.masks_whole_tile_kept_map = 0
        # Whole-tile MAP masks the score floor cut, and the scores of every one
        # the tile produced. The count says how hard the floor bit; the scores
        # are what the next floor is chosen from. Written under _stat_lock.
        self.masks_dropped_map_lowscore = 0
        self.map_cover_scores: list[float] = []
        # Tiles whose render returned nothing (provider error, WMS/WMTS timeout,
        # coverage hole): also never submitted or billed. Surfaced at run end as
        # "N tiles could not be loaded" so a slow-server run's blank regions are
        # not a silent coverage gap.
        self.tiles_render_failed = 0
        # Tiles whose render came back as the online source's "no image here"
        # placeholder card instead of imagery (zoomed past what the source
        # serves, a gap in its coverage, a provider that answers with a card
        # rather than an error). Never submitted, never billed, and reported
        # separately from a render hole because the two have different fixes.
        # Plain int, GIL-safe.
        self.tiles_unavailable = 0
        # Imagery-side health, read at the terminal so a run that felt slow can
        # be told apart from one that was: how many tiles waited past
        # _render_slow_s for their basemap, and how narrow the adaptive render
        # window had to go. A floor of 1 on a wide ceiling means the link could
        # feed exactly one fetch at a time. Plain ints, GIL-safe.
        self.renders_slow = 0
        self.render_window_floor = self._prefetch_depth
        # Backend-distress counters, read at the terminal via run_health_summary()
        # so a sick backend can be told apart from a healthy run in telemetry
        # (a stalled service that times out every submit used to read as a plain
        # user cancel). Plain ints, GIL-safe (same pattern as tiles_succeeded).
        # submit_network_retries counts each transient network-error requeue (NOT
        # busy/queue waits, which surface separately as the queue_state warming
        # time); tiles_skipped_network counts tiles dropped after their
        # submit-retry budget was exhausted (never reached the service).
        self.submit_network_retries = 0
        self.tiles_skipped_network = 0
        # Tiles the service ACCEPTED and never answered inside their own budget
        # (see _expire_stalled_replies and the batched poll deadline). Their own
        # outcome, not a completion and not a skip: the service took the request,
        # so the tile is billed, and the user got nothing back for it. Never
        # re-posted, because a second request is a second charge. Plain int,
        # GIL-safe.
        self.tiles_timed_out = 0
        self._completed_idx: set[int] = set()
        # Set True if any tile came back at the per-inference ceiling
        # (self._max_masks masks): the model emits a bounded number of object
        # queries per forward pass, so a tile at that ceiling was likely
        # truncated. Read on the main thread at run end to nudge "raise Detail"
        # (finer tiling puts fewer objects per tile). Plain bool, GIL-safe.
        self._hit_mask_cap = False
        # How many tiles hit that ceiling: the review hint quantifies the
        # truncation ("N tiles maxed out") so a dense-orchard user knows how
        # much was cut, not just that something was. Plain int, GIL-safe.
        self.tiles_mask_capped = 0
        # Ground units per pixel of the RETURNED masks, observed from the
        # server responses (the cloud model can answer at a coarser grid than
        # the sent tile, e.g. an internal half-res mode). Max across tiles =
        # the coarsest (full-tile) value; boundary slivers only report finer.
        # This is the TRUE staircase step of the run's polygons, so the review
        # px->ground refine scales by it. Plain float, GIL-safe; 0.0 = unknown.
        self.observed_mask_gsd = 0.0
        # Last queue_state payload emitted (None = flowing). Dedupes the signal
        # so the UI only repaints when the position/state actually moves.
        self._last_queue_emit: tuple[int, int, int] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Thread-safe cancellation request. The worker checks this flag
        between tiles and aborts the poll loop of in-flight tiles. Also wakes the
        per-tile render bridge so a render_tile blocked waiting on the main thread
        (which may itself be blocked in worker.wait() during unload) returns at
        once instead of deadlocking until the condition times out."""
        if self._stop_reason is None:
            self._stop_reason = "user"
        self._stop_requested = True
        if self._tile_renderer_cancel is not None:
            try:
                self._tile_renderer_cancel()
            except (RuntimeError, AttributeError):
                pass

    def _should_abort(self) -> bool:
        """Stop predicate handed to the client's concurrent network calls so their
        nested event loop quits the instant a stop is requested (submit/poll can
        otherwise block this thread for the whole submit timeout, which leaves the
        thread un-joinable at unload and crashes QGIS at teardown)."""
        return self._stop_requested

    # UNREACHABLE (2026-07-25): no caller anywhere. The resume flow this was
    # written for was removed as confusing, and every Detect is a fresh run.
    # Kept on purpose so the worker surface stays stable. Do not delete it, and
    # do not treat it as live.
    def remaining_tiles(self) -> list[tuple[int, int, int, int]]:
        """Input tile rects not billed as completed, in original order.

        Written for the removed resume flow, on the assumption that a tile the
        client never saw complete was never charged. That does not hold, so a
        resubmit has to be treated as a fresh charge. Returns (x, y, w, h) specs.
        """
        return [t for i, t in enumerate(self._tiles) if i not in self._completed_idx]

    def run_health_summary(self) -> dict:
        """Cheap run-level backend-distress counters, read by the plugin at the
        terminal to tell a stalled service apart from a healthy run. A snapshot
        of plain-int fields (GIL-safe); safe to read after the run loop ends.

        Keys:
          submit_retries        total transient submit network-error requeues
          tiles_skipped_network tiles dropped after their submit-retry budget
                                was exhausted (never reached the service)
          tiles_timed_out       tiles the service accepted and never answered
                                inside their budget (billed, nothing delivered)
          renders_slow          tiles that waited past the slow-render mark for
                                their basemap imagery
          render_window_floor   narrowest the adaptive render window went; equal
                                to the ceiling means the link never limited it
        """
        return {
            "submit_retries": int(self.submit_network_retries),
            "tiles_skipped_network": int(self.tiles_skipped_network),
            "tiles_timed_out": int(self.tiles_timed_out),
            "renders_slow": int(self.renders_slow),
            "render_window_floor": int(self.render_window_floor),
        }

    def _emit_run_phase(self, name: str) -> None:
        """Announce which work the pre-first-tile wait belongs to, once per
        phase. Deduped here rather than at every call site, so the submit loop
        can say "detecting" on every tile without spamming the UI."""
        if getattr(self, "_run_phase_sent", None) == name:
            return
        self._run_phase_sent = name
        try:
            self.run_phase.emit(name)
        except RuntimeError:
            pass  # the receiver went away mid-teardown

    def _emit_progress(self, completed: int, total: int) -> None:
        """Emit progress shifted by the resume offset (a resumed run keeps
        counting from where the original run stopped)."""
        shown_total = self._progress_total or total
        self.progress.emit(self._progress_offset + completed, shown_total)

    def _note_busy(self, position: int, depth: int, eta_s: int) -> None:
        """Surface a server-busy/queued answer to the UI (deduped so the label
        only repaints when the numbers actually move)."""
        payload = (position, depth, eta_s)
        if payload != self._last_queue_emit:
            self._last_queue_emit = payload
            self.queue_state.emit(position, depth, eta_s)

    def _note_flowing(self) -> None:
        """A tile completed: clear any on-screen busy/queue state."""
        if self._last_queue_emit is not None:
            self._last_queue_emit = None
            self.queue_state.emit(0, 0, 0)

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """QThread entry point. Holds a keep-awake activity for the whole run
        so the OS does not throttle this thread's network event delivery (macOS
        App Nap) or suspend the system mid-run and drop already-billed tiles
        (Windows sleep, Linux idle). Best-effort; always released."""
        from ..core.power_inhibit import begin_keep_awake, end_keep_awake

        activity = begin_keep_awake("AI Segmentation cloud detection")
        try:
            self._run_detection()
        except Exception as exc:  # noqa: BLE001 - last-resort net so a crash in the
            # worker-thread geometry pipeline can never leave the UI stuck in
            # "Detecting..." forever. Route to error(), which resets the run
            # state and salvages any billed partial results.
            logger.error("AutoDetectionWorker crashed", exc_info=True)
            # Capture the crash with a traceback fingerprint so a paid-path
            # regression is visible and groupable, not just a console line.
            # Off the GUI thread: track() queues, the next main-thread flush
            # ships it. Never re-raise here.
            try:
                from ..core.telemetry_errors import report_exception
                report_exception(exc, stage="segment", module="auto_detection_worker")
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                # "internal error" is load-bearing: it is what tells the
                # plugin's classifier this is OUR fault, so the user is never
                # told to check a connection that was fine. Qt exception text
                # can carry the word "network" (QNetworkReply), which the
                # classifier's connectivity scan would otherwise match.
                self.error.emit(f"Detection stopped unexpectedly (internal error): {exc}")
            except Exception:  # noqa: BLE001 - signal delivery must never re-raise here  # nosec B110
                pass
        finally:
            end_keep_awake(activity)
            # Imagery-side summary, the counterpart of the plugin's run summary:
            # it says whether a run that felt slow was slow on the imagery and
            # how far the link forced the fetch window down. Counts only, so it
            # is safe in a production log.
            try:
                from qgis.core import QgsMessageLog
                QgsMessageLog.logMessage(
                    f"Auto detection: imagery summary - {self.renders_slow} "
                    f"slow render(s), fetch window narrowed to "
                    f"{self.render_window_floor} of {self._prefetch_depth}, "
                    f"{self.tiles_render_failed} tile(s) with no imagery",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
            except Exception:  # noqa: BLE001 - a log line must never end a run
                pass  # nosec B110
            # Safety net: a crash above leaves the pool open, and its threads are
            # non-daemon, so the process would wait on them at exit. Quiet and
            # non-waiting: the terminal has already gone out on that path, and
            # the normal path drained the pool before it.
            try:
                self._close_convert_pool(0.0, emit=False)
            except Exception:  # noqa: BLE001 - teardown must never re-raise
                pass  # nosec B110
            client = getattr(self, "_client", None)
            if client is not None:
                try:
                    # Drop the run's own reference FIRST, or the release below
                    # leaves the manager alive and main-thread GC destroys it
                    # from the wrong thread later (see _run_detection).
                    self._run_nam = None
                    # Destroy this thread's private network manager while the
                    # thread is still alive: left to main-thread GC after the
                    # thread exits, Qt would tear down a worker-affine QObject
                    # from the wrong thread (timer warnings, socket crashes).
                    client.release_thread_nam()
                except Exception:  # noqa: BLE001 - teardown must never re-raise
                    pass  # nosec B110

    def _run_detection(self) -> None:
        from ..api.terralab_client import TerraLabClient

        self._client = TerraLabClient()
        # Hold the run's network manager for the whole run. Every reply is a
        # child of it, so if it were destroyed mid-run every tile still in
        # flight would die with it and the next read would raise RuntimeError on
        # a dead wrapper, ending a paid run at whatever tile it had reached.
        # See TerraLabClient.acquire_predict_nam.
        self._run_nam = self._client.acquire_predict_nam()
        total = len(self._tiles)

        if total == 0:
            self._terminal_sent = True
            self.all_tiles_finished.emit([])
            return

        # Clock for the free re-split tail (see _resplit_time_left). The grid
        # the user paid for is what is on the clock now; anything queued past it
        # is extra work nobody asked for, so it gets a share of that time and no
        # more.
        self._run_started_at = time.monotonic()
        self._paid_tiles_total = total
        self._paid_tiles_done = 0
        self._resplit_deadline = 0.0
        self._resplit_dropped = 0

        # Everything from here to the first submit is imagery work: stamps,
        # the scan gate and the first tile renders. Say so, or the wait is
        # labelled as the service warming up and a slow basemap reads as a
        # hang on a component that is not even busy yet.
        self._emit_run_phase("imagery")

        # Crop the reference-example stamps from the zone image once (off the GUI
        # thread); each tile's encode then composites them in.
        self._prepare_stamps()

        # Rebuild the zone clip geometry + a prepared GEOS engine ON THIS THREAD
        # from the copied WKB (a prepared engine is bound to its geometry
        # instance, so it must be rebuilt worker-side, never passed). Every
        # detection is clipped to this in _detections_to_geoms, exactly as the
        # GUI used to do via _auto_clip_polygon / _auto_clip_engine.
        self._build_clip_engine()

        logger.debug(
            "AutoDetectionWorker: run_id=%s tiles=%d prompt=%r exemplars=%d",
            self._run_id, total, self._prompt, len(self._stamps),
        )

        # The DIRECT inference endpoint is synchronous (each POST returns that
        # tile's masks, never a "pending" to poll), so we drive a CONTINUOUS
        # sliding window: keep max_concurrent posts in flight and refill the
        # instant any one returns. This removes the barrier (submit N -> wait for
        # ALL N -> submit next N) that made tiles arrive in bursts with a service-idle
        # gap between each batch; tiles now stream in one-by-one and the service stays
        # saturated. The async backend path is pollable, so it keeps the
        # batched submit+poll model below (also the rollback path).
        if getattr(self._client, "detection_direct", False):
            # Empty-tile scan gate (policy-gated, streaming path only): settle
            # the run's empty tiles from cheap packed scans first, so the full
            # per-tile detection below only runs where something showed up.
            # No-ops unless the plugin resolved a gate config for this run.
            self._run_gate_scan()
            self._run_streaming(total)
        else:
            self._run_batched(total)

    def _settle_converted_batch(self, items) -> None:
        """Fold and emit a drained batch of finished conversions. WORKER THREAD.

        Every Qt signal of the streaming path passes through here, so converter
        threads never touch one.
        """
        for ok, job, payload in items:
            self._settle_converted(ok, job, payload)

    def _close_convert_pool(self, budget_s: float, emit: bool = True) -> None:
        """Shut the converter pool down and emit everything it still owes.

        Every conversion it holds is a tile the user was already billed for, so
        this waits for them rather than dropping them. ``budget_s`` caps that
        wait: a run that ends normally can afford the long one, a stop gets the
        same short budget the in-flight reply drain uses, because unload joins
        this thread on the main thread and must never be held for long. Whatever
        has not finished by then is cancelled; the threads are left to end on
        their own rather than joined.

        A stop that lands DURING the long budget cuts it down to the short one
        from that instant. The GUI stops waiting for the worker's terminal after
        a few seconds, so a drain that keeps its full budget through a Cancel
        forces the user out and leaves this thread running. Nothing already
        dequeued is lost: each pass emits what it drained before the check.

        ``emit`` False closes QUIETLY. Used by run()'s crash net: the terminal
        has already gone out by then, the main thread has finalized on it, and
        a late tile_completed would rebuild review state on a dead run.
        """
        pool, self._convert_pool = self._convert_pool, None
        if pool is None:
            return
        started = time.monotonic()
        owed = int(pool.pending or 0)
        deadline = started + max(0.0, budget_s)
        stop_seen = self._stop_requested
        while pool.pending and time.monotonic() < deadline:
            items = pool.drain(timeout=0.25)
            if emit:
                self._settle_converted_batch(items)
            if self._stop_requested and not stop_seen:
                stop_seen = True
                deadline = min(
                    deadline, time.monotonic() + self._stop_drain_budget_s)
        leftover = pool.close(wait=False)
        if emit:
            self._settle_converted_batch(leftover)
        # This wait owns the stretch where the tile bar reads 100% and the dock
        # says "Almost done": progress counts REPLIES, and a reply's masks are
        # still unconverted when it lands. Nothing used to say how long it took
        # or how much it was holding, so a run that spent a minute here and a
        # run that spent none looked identical from the log.
        spent = time.monotonic() - started
        if owed and spent >= 1.0:
            logger.info(
                "AutoDetectionWorker: converted the last %d tile(s) in %.1fs "
                "after the final reply", owed, spent,
            )

    def _flush_withheld(self) -> None:
        """Emit the withheld detections of re-split parents whose quadrants ALL
        failed to deliver (stop mid-ladder, offline, exhausted): the paid parent
        read is better than a hole. Parents with ANY quadrant result stay
        withheld, so a coarse multi-object blob can never union-bridge the
        quadrants' separated objects."""
        for parent_idx, dets in self._withheld.items():
            if parent_idx in self._parents_with_child_results or not dets:
                continue
            try:
                self.tile_completed.emit(parent_idx, dets)
            except RuntimeError:
                return  # receiver gone (teardown); nothing more to flush
        self._withheld.clear()

    def _emit_terminal(self) -> None:
        """Emit the run's terminal signal. A user cancel emits cancelled(); an
        error/exhausted already emitted its own terminal; otherwise the run
        finished normally.

        Sends nothing at all once _emit_stop has spoken for this run, whatever
        its emit did: the main thread finalizes on the first terminal, so a
        second one lands on a run that is already over."""
        self._flush_withheld()
        # Nothing can be submitted after the terminal, so the per-tile pre-stamp
        # images of tiles that never settled (a stop mid-flight) go now.
        self._tile_clean_image.clear()
        # Same for the canvas marks: a quadrant dropped by the time budget or by
        # a stop owes an answer that will never come, and the flush above has
        # just put the coarse read back on that ground anyway.
        if self._rescanning:
            self._rescanning.clear()
            try:
                self.rescan_state.emit(-1, None, False)
            except RuntimeError:
                pass
        if self._terminal_sent:
            return
        self._terminal_sent = True
        if self._stop_requested:
            if self._stop_reason == "user":
                self.cancelled.emit()
        else:
            self.all_tiles_finished.emit([])

    def _settle_concurrency(self, setback: bool, progress: bool) -> None:
        """Fold one cycle's outcome into the AIMD width: a timeout/latency
        setback halves it, an otherwise-clean cycle that made progress grows it
        by one. A cycle that neither progressed nor set back leaves it unchanged
        (e.g. every tile was rate-limited: server capacity, not a link problem)."""
        if setback:
            self._aimd.on_setback()
        elif progress:
            self._aimd.on_clean_cycle()

    def _submit_error_message(self, code: str) -> str:
        """User-facing message for a fatal terminal error. The offline fast-fail
        sentinel gets a clear connectivity line (also classified NETWORK by the
        plugin) instead of a raw internal code."""
        if code == OFFLINE_STOP_CODE:
            return "No internet connection. Check your connection and try again."
        return f"Tile submit failed: {code}"

    def _retry_decision(
        self,
        tile_idx: int,
        outcome: tuple,
        busy_since: dict[int, float],
        submit_attempts: dict[int, int],
    ) -> tuple[bool, float, bool]:
        """Shared submit-retry policy for both run loops. Queue-busy retries
        burn a TIME budget (waiting in line is not failing) with jittered
        server-suggested delays; transient network errors keep the attempt
        ceiling, back off exponentially and feed the offline fast-fail.
        Returns (give_up, delay_s, setback); setback is True only for a
        transient network error (an AIMD setback)."""
        delay, is_busy = outcome[1], outcome[2]
        retry_code = outcome[3] if len(outcome) > 3 else ""
        now = time.monotonic()
        if is_busy:
            first = busy_since.setdefault(tile_idx, now)
            give_up = (now - first) > self._queue_retry_budget_s
            # Clamp both ends. The delay is the server's own retry_after off the
            # wire, and an unbounded one parks a paid tile until the run-wide
            # stall watchdog kills the whole run instead of just this tile. The
            # transient-network branch below already caps at 30 s.
            delay = min(60.0, max(1.0, delay)) * random.uniform(*self._busy_jitter)  # nosec B311 - jitter, not crypto
            # A busy/queue answer means the server was reached, so it is not
            # an offline run and not a link setback.
            self._fastfail.reset()
            return give_up, delay, False
        if retry_code in BACKEND_UNAVAILABLE_CODES:
            # Cold-instance backend-unavailable (HTTP 503, pre-charge): bounded,
            # short-spaced retries, because the instance stabilizes within
            # seconds. Not a link setback (the service is warming, not the link),
            # not fed to the offline fast-fail, and not counted as network
            # distress. Billing-safe: the same submission is safe to resend
            # because the rejection is raised before any charge (fail-closed
            # server side), so a retry cannot double-charge.
            n = submit_attempts.get(tile_idx, 0) + 1
            submit_attempts[tile_idx] = n
            give_up = n > self._backend_unavailable_retries
            self._fastfail.reset()
            jitter = random.uniform(*self._busy_jitter)  # nosec B311 - jitter, not crypto
            delay = self._backend_unavailable_delay_s * jitter
            return give_up, delay, False
        n = submit_attempts.get(tile_idx, 0) + 1
        submit_attempts[tile_idx] = n
        # Transient network-error requeue: count it for the backend-distress
        # telemetry (see run_health_summary).
        self.submit_network_retries += 1
        give_up = n > self._max_rate_limit_retries
        # Exponential-ish with jitter so transient blips don't retry in
        # synchronized waves.
        delay = min(30.0, delay * (2 ** min(n - 1, 4)))
        delay *= random.uniform(0.5, 1.0)  # nosec B311 - jitter, not crypto
        # A hard-connectivity code advances the offline fast-fail counter
        # (pre-first-success it trips at the small default threshold; mid-run
        # only at the much larger _MIDRUN_OFFLINE_STREAK).
        self._fastfail.record(retry_code)
        return give_up, delay, True

    def _skip_network_tile(self, tile_idx: int) -> None:
        """Terminal give-up for one tile that never reached the service (busy
        budget or retry ceiling), counted for the backend-distress telemetry
        (see run_health_summary). Callers still advance their progress count."""
        self.tiles_skipped_network += 1
        self._release_tile_clean_image(tile_idx)
        self.warning.emit(f"Tile {tile_idx}: submit retries exhausted; skipping")

    def _offline_stop(self, stop_payload: tuple | None) -> tuple | None:
        """Offline fast-fail, shared by both run loops: a run that only ever
        sees hard-connectivity errors (DNS / connection refused / proxy) is
        offline. Before the first success it aborts within a few failures;
        after it, only a long unbroken streak (a link that stays dead, not a
        blip) ends the run, and the billed partials are salvaged into the
        review. An existing terminal payload always wins."""
        if stop_payload is not None or not self._fastfail.tripped:
            return stop_payload
        if (
            self.tiles_succeeded == 0 or self._fastfail.streak >= self._midrun_offline_streak
        ):
            return ("fatal", OFFLINE_STOP_CODE)
        return stop_payload

    def _mark_stop(self, stop_payload: tuple) -> None:
        """Record a worker-decided stop and halt both run loops, with NO signal.

        The signal goes out in _emit_stop, once the loop has drained what it
        still owes. Splitting the two is what lets a run out of credits keep
        the tiles it already paid for: the main thread finalizes the run on the
        terminal signal, so anything emitted after it is dropped, and both the
        in-flight replies and the converter pool still hold billed geometry
        when the wall is hit."""
        self._stop_reason = (
            "exhausted" if stop_payload[0] == "exhausted" else "error")
        self._stop_requested = True

    def _emit_stop(self, stop_payload: tuple) -> None:
        """Terminal signal for a worker-decided stop (out of credits, fatal,
        offline). Call it LAST, after the drains.

        The stop is RECORDED FIRST, before anything is emitted. _emit_terminal
        reads that state to decide whether the run ended normally, so a signal
        that raises must not be able to leave it unset: the run would then read
        as a clean finish and the GUI would get a second terminal for it.
        Recording first also makes this safe on the one caller that has no
        _mark_stop of its own (the scan gate).

        Flushes withheld re-split parents next: the main thread finalizes
        (nulls the merger) on the terminal, so a parent's late tile_completed
        would race the finalize and its billed detections could be dropped
        (_flush_withheld clears _withheld, making the later call in
        _emit_terminal a no-op). The flush also needs every quadrant conversion
        settled, or a parent whose quadrants have not been folded in yet reads
        as orphaned and emits its coarse blob beside their fine outlines."""
        self._mark_stop(stop_payload)
        self._flush_withheld()
        self._terminal_sent = True
        try:
            if stop_payload[0] == "exhausted":
                self.credits_exhausted.emit(stop_payload[1])
            else:
                self.error.emit(self._submit_error_message(stop_payload[1]))
        except RuntimeError:
            pass  # receiver gone mid-teardown; the stop is already recorded

    @staticmethod
    def _free_read_replies(replies) -> None:
        """Destroy these already-read replies now, one by one.

        Both reply-driven loops run on this worker thread, which never enters a
        QEventLoop: they pump with processEvents, and processEvents does not
        deliver DeferredDelete. So a finished reply and the upload buffer
        holding its tile image stayed alive until the thread ended, several
        hundred MB on a large run. This delivers those pending deletions.

        Addressed PER REPLY on purpose. The thread-wide form
        (sendPostedEvents(None, ...)) destroys every object on this thread that
        is waiting on deleteLater, whoever queued it, so one stray queued
        deletion elsewhere in the run could take a live object with it. Pass
        only replies already read and dropped from the in-flight map.
        """
        from qgis.PyQt.QtCore import QCoreApplication, QEvent

        # Qt6 (QGIS 4) scopes the event types under QEvent.Type; Qt5 exposes
        # them flat on QEvent. Resolve compatibly, like the QEventLoop flags.
        deferred_delete = getattr(QEvent, "Type", QEvent).DeferredDelete
        # PyQt6 exposes it as a Python enum member; sendPostedEvents wants the
        # plain int, which .value gives on Qt6 and int() gives on Qt5.
        deferred_delete = getattr(deferred_delete, "value", deferred_delete)
        for reply in replies:
            try:
                reply.deleteLater()
                QCoreApplication.sendPostedEvents(reply, int(deferred_delete))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                continue  # freeing memory early must never break a run

    @staticmethod
    def _reply_is_finished(reply) -> bool:
        """isFinished() that reports a DESTROYED reply as finished.

        A reply whose C++ half is gone can never answer, so it has to be drained
        out of the in-flight map this cycle. Reported as pending it would keep
        its slot for good and the run would stall on a tile that cannot land.
        """
        try:
            return bool(reply.isFinished())
        except RuntimeError:
            return True

    def _read_reply(self, tile_idx: int, reply) -> dict:
        """parse_reply() that turns a destroyed reply into a transient per-tile
        error instead of an exception.

        A destroyed reply used to propagate RuntimeError out of the run loop to
        run()'s last-resort net, which ended the whole paid run at whatever tile
        it had reached. One unreadable answer is a per-tile condition: the tile
        goes back as retryable (TIMEOUT is retryable and, unlike the
        connectivity codes, is never fed to the offline fast-fail, so an
        internal fault cannot be mistaken for the user being offline), and a
        second death on the same tile skips it and keeps the run going. This
        only ASKS for the retry: the run loop decides, against the same budget
        every other transient failure draws on, and a stopping run never
        re-posts at all. Cost is bounded either way, since the destroyed answer
        may already have been billed.
        """
        try:
            return self._client.parse_reply(reply)
        except RuntimeError as err:
            first_time = tile_idx not in self._dead_reply_tiles
            self._dead_reply_tiles.add(tile_idx)
            # A stopping run never re-posts, and the retry budget may already be
            # spent, so promising a re-post here reads as a billing event to a
            # user who just pressed Cancel. Say what this call knows.
            if self._stop_requested:
                outcome = "run stopping, dropping it"
            elif first_time:
                outcome = "retrying it"
            else:
                outcome = "skipping"
            self.warning.emit(
                f"Tile {tile_idx}: reply was destroyed before it could be read "
                f"({err}); {outcome}"
            )
            if first_time:
                return {"error": "Reply destroyed before it was read",
                        "code": "TIMEOUT"}
            return {"error": "Reply destroyed before it was read",
                    "code": "REPLY_DESTROYED"}

    def _expire_stalled_replies(self, in_flight: dict) -> int:
        """Abort the in-flight replies past their per-tile deadline and drop
        them from ``in_flight``. Returns how many, so the caller advances its
        progress count.

        The streaming loop only ever frees a slot when a reply FINISHES, and the
        request's own timeout measures inactivity, so one reply the service
        never completes would hold its slot for the rest of the session and park
        this thread with its network manager. Budget in _stream_reply_budget_s.
        A reply that has finished is never expired: its answer may be billed,
        and the caller is about to read it.

        Each expiry is counted as a TIMED-OUT tile, never as a completed one.
        The service accepted the request, so the tile is billed and the user got
        nothing for it; counting it done would hide that from the run summary
        and from the credit reporting. It is not re-posted either, because a
        second request on the same tile is a second charge.
        """
        now = time.monotonic()
        expired = [
            reply for reply, entry in in_flight.items()
            if now > entry[4] and not self._reply_is_finished(reply)
        ]
        for reply in expired:
            tile_idx = in_flight.pop(reply)[0]
            try:
                reply.abort()
            except (RuntimeError, AttributeError):
                pass
            self._release_tile_clean_image(tile_idx)
            self.tiles_timed_out += 1
            self.warning.emit(
                f"Tile {tile_idx} timed out after "
                f"{int(self._stream_reply_budget_s)}s")
        self._free_read_replies(expired)
        return len(expired)

    def _drain_polled_on_stop(
        self, in_flight: dict, completed: int, total: int
    ) -> int:
        """Poll the requests still open when the batched loop stopped, and emit
        the ones that answer. Returns the new progress count.

        The server bills a tile when it accepts the request, so a request the
        client drops unpolled is a detection the user paid for and never sees.
        The loop stops SUBMITTING the instant the stop lands, so this only waits
        out the small accepted set. Bounded by the stop drain budget, because
        unload joins this thread from the main thread and the GUI stops waiting
        for the terminal after a few seconds.

        The caller decides which stops reach here (_BILLED_DRAIN_STOP_REASONS).
        Nothing is re-posted: a second request on the same tile is a second
        charge. Same rule as the streaming loop's reply drain, on the path that
        polls instead of holding replies.
        """
        drain_deadline = time.monotonic() + self._stop_drain_budget_s

        # The run loop hands the client _should_abort, which is the stop flag
        # itself, and that flag is already set here: every poll would quit
        # before it left. This drain rides its own clock instead, so the polls
        # go out and still cannot outlive the budget.
        def past_budget() -> bool:
            return time.monotonic() >= drain_deadline

        while in_flight and not past_budget():
            poll_ids = list(in_flight.keys())
            try:
                responses = self._client.get_detection_status_many(
                    poll_ids, self._auth, should_abort=past_budget)
            except Exception:  # noqa: BLE001 - the terminal still has to go out
                logger.debug(
                    "AutoDetectionWorker: stop drain poll failed", exc_info=True)
                break
            answered = False
            for request_id, resp in zip(poll_ids, responses):
                status = resp.get("status")
                if status not in ("completed", "failed", "cancelled"):
                    # Still working, or the poll itself failed: ask again while
                    # the budget lasts.
                    continue
                entry = in_flight.pop(request_id, None)
                if entry is None:
                    continue
                answered = True
                if status != "completed":
                    # Answered with nothing to keep. Dropped rather than
                    # re-polled, so the budget goes to the tiles that can still
                    # deliver.
                    continue
                tile_idx, tile_spec, _, _, _, tile_transform = entry
                _, _, tile_w, tile_h = tile_spec
                if self._emit_completed(
                    resp, tile_idx, tile_w, tile_h, tile_transform
                ):
                    self.tiles_succeeded += 1
                    self._completed_idx.add(tile_idx)
                completed += 1
                self._emit_progress(completed, total)
            if in_flight and not answered:
                # Nothing landed this pass: give the service a beat instead of
                # re-polling flat out. Not _interruptible_sleep, which returns
                # at once on the stop flag this drain runs under.
                time.sleep(
                    max(0.0, min(0.25, drain_deadline - time.monotonic())))
        return completed

    def _run_batched(self, total: int) -> None:
        # Process tiles in bounded concurrent batches.
        # We maintain a queue of tiles and a dict of in-flight requests.
        # "Concurrent" here means overlapping submit+poll cycles, all on
        # this single thread via a cooperative polling loop.
        # (tile_idx, (x, y, w, h)) -- encoded lazily when its slot is filled.
        pending: deque = deque(enumerate(self._tiles))
        # Tiles to RE-submit (already encoded) after a rate-limit/transient error,
        # so a retry never re-encodes. submit_attempts bounds the retries per tile.
        # (tile_idx, tile_spec, png_bytes, not_before), where not_before is the
        # monotonic instant the retry may be re-posted: same stamp the streaming
        # loop carries. Without it a partly rate-limited batch re-posted the
        # limited tiles with no delay on every cycle.
        resubmit: deque = deque()
        submit_attempts: dict[int, int] = {}
        # tile_idx -> monotonic time of its FIRST queue-busy answer. Busy retries
        # are bounded by _QUEUE_RETRY_BUDGET_S from that instant (time budget),
        # not by an attempt count - waiting in line is not failing.
        busy_since: dict[int, float] = {}
        # in_flight: { request_id: (tile_idx, tile_spec, poll_interval, max_wait,
        #               deadline, tile_transform) }
        in_flight: dict = {}
        completed = 0
        # Consecutive per-tile rejections with no accepted submit in between
        # (see RUN_FATAL_CODES / _MAX_CONSECUTIVE_TILE_FATALS).
        fatal_streak = 0
        # The terminal stop this run decided. Held, not emitted: see _mark_stop.
        terminal_stop: tuple | None = None

        # resubmit MUST be in the guard: a cycle where a whole batch comes back
        # rate-limited (launch spike) drains pending into resubmit with nothing
        # in flight; without it the loop exited and silently dropped the tiles.
        # _render_deferred likewise: tiles waiting out a blank-render retry are
        # still owed, so the run must not end from under them.
        while (
            pending or in_flight or resubmit or self._render_deferred
        ) and not self._stop_requested:
            # Blank/failed renders whose retry delay matured re-enter the queue.
            self._pump_render_deferred(pending)
            # Per-cycle AIMD signals: a completed tile is progress, a timeout /
            # transient-network retry is a setback. Folded into the width once at
            # the end of the cycle (see _settle_concurrency).
            cycle_setback = False
            cycle_progress = False
            # Fill up in-flight slots, up to the adaptive AIMD cap: it opens
            # narrow, grows per clean cycle, and halves on a setback.
            effective_cap = self._aimd.cap
            # Gather a batch up to the free slots (re-submits first, no re-encode),
            # then submit them ALL CONCURRENTLY in one batched round-trip. Serial
            # uploads were the wall once polling went concurrent: each submit ships
            # a ~200-400KB JPEG and blocked ~1.4s, so N tiles took N x 1.4s.
            batch = []
            while (resubmit or pending) and (len(in_flight) + len(batch)) < effective_cap:
                # Jittered delays make later entries mature before the head, so
                # scan for the FIRST ready retry rather than gating on index 0
                # (same rule as the streaming loop).
                ready_i = None
                now = time.monotonic()
                for i, entry in enumerate(resubmit):
                    if entry[3] <= now:
                        ready_i = i
                        break
                if ready_i is not None:
                    tile_idx, tile_spec, png_bytes, _ = resubmit[ready_i]
                    del resubmit[ready_i]
                    batch.append((tile_idx, tile_spec, png_bytes))  # already encoded
                    continue
                if not pending:
                    break  # only retries left, all still waiting out their delay
                tile_idx, spec = pending.popleft()
                # Encode off the GUI thread, just before submit. A blank/failed
                # render is deferred for a re-render (the worker keeps it);
                # only a permanent skip counts toward progress here.
                status, payload = self._encode_or_defer(tile_idx, spec)
                if status == "defer":
                    continue
                if status == "empty":
                    # Degenerate render (provably objectless): settle as a
                    # completed empty tile, no request spent so no charge.
                    self._settle_empty_tile(tile_idx, charged=False)
                    completed += 1
                    self._emit_progress(completed, total)
                    continue
                if status == "skip":
                    # Empty / unencodable tile (e.g. clamped to nothing): skip it
                    # but still count it so progress reaches 100%.
                    completed += 1
                    self._emit_progress(completed, total)
                    continue
                tile_spec, png_bytes = payload
                batch.append((tile_idx, tile_spec, png_bytes))

            submit_backoff: float | None = None
            batch_stop: tuple | None = None  # first ("exhausted"|"fatal", payload)
            if batch and not self._stop_requested:
                for (tile_idx, tile_spec, png_bytes), outcome in zip(
                    batch, self._submit_batch(batch)
                ):
                    kind = outcome[0]
                    if kind == "ok":
                        _, request_id, poll_interval, max_wait, tile_transform = outcome
                        deadline = time.monotonic() + max_wait
                        in_flight[request_id] = (
                            tile_idx, tile_spec, poll_interval, max_wait,
                            deadline, tile_transform,
                        )
                        # A tile is on the wire: the wait is no longer imagery.
                        self._emit_run_phase("detecting")
                        fatal_streak = 0  # server accepted a tile: not systematic
                    elif kind == "completed_inline":
                        # Sync fast path: masks came back in the submit response.
                        # Decode + emit immediately, never poll. Same handler as
                        # the polled-completed branch (identical response shape).
                        _, response, tile_transform = outcome
                        _, _, tile_w, tile_h = tile_spec
                        if self._emit_completed(
                            response, tile_idx, tile_w, tile_h, tile_transform
                        ):
                            self.tiles_succeeded += 1
                            self._completed_idx.add(tile_idx)
                            cycle_progress = True
                        # The network round-trip got through regardless of local
                        # decode (a bad tile only fails its own conversion), so
                        # this is never an offline run: reset the fatal streak
                        # and the offline fast-fail either way.
                        fatal_streak = 0
                        self._fastfail.reset()
                        completed += 1
                        self._emit_progress(completed, total)
                    elif kind == "skip":
                        completed += 1
                        self._emit_progress(completed, total)
                    elif kind == "tile_fatal":
                        # This tile was rejected with a non-retryable code that
                        # is not run-level: skip it, keep the run alive. A
                        # streak of these with zero accepted submits in between
                        # is systematic (e.g. an unknown new run-level code):
                        # stop the run with that code instead of skipping all.
                        bad_code = outcome[1] or "UNKNOWN"
                        fatal_streak += 1
                        self.warning.emit(
                            f"Tile {tile_idx}: rejected ({bad_code}); skipping")
                        completed += 1
                        self._emit_progress(completed, total)
                        if batch_stop is None and fatal_streak >= self._max_tile_fatals:
                            batch_stop = ("fatal", bad_code)
                    elif kind == "retry":
                        # Requeue (no inline sleep) and let one coalesced back-off
                        # pace the next cycle. Policy shared with the streaming
                        # loop: see _retry_decision.
                        give_up, delay, setback = self._retry_decision(
                            tile_idx, outcome, busy_since, submit_attempts)
                        cycle_setback = cycle_setback or setback
                        if give_up:
                            self._skip_network_tile(tile_idx)
                            completed += 1
                            self._emit_progress(completed, total)
                        else:
                            resubmit.append(
                                (tile_idx, tile_spec, png_bytes,
                                 time.monotonic() + delay))
                            submit_backoff = (
                                delay if submit_backoff is None
                                else max(submit_backoff, delay)
                            )
                    elif batch_stop is None:  # "exhausted" or "fatal"
                        batch_stop = outcome

            # Saturated tiles that completed inline may have queued quadrants:
            # fold them into the submit deque NOW, before the empty-queue exit
            # checks below can end the run with re-splits still owed.
            total += self._drain_subtiles(pending)

            # Offline fast-fail + terminal stop, shared with the streaming loop
            # (see _offline_stop). The whole batch was processed first, so
            # already-charged "ok" tiles are kept. Recorded only; the signal
            # goes out after the loop, on the one stop protocol both loops use.
            batch_stop = self._offline_stop(batch_stop)
            if batch_stop is not None:
                terminal_stop = batch_stop
                self._mark_stop(batch_stop)

            if self._stop_requested:
                break

            # Nothing in flight to poll and we only have re-submits waiting: back
            # off once (rate-limit retry_after) before looping to re-send them.
            if submit_backoff is not None and not in_flight:
                self._settle_concurrency(cycle_setback, cycle_progress)
                self._interruptible_sleep(
                    min(max(submit_backoff, self._min_poll_backoff_s), 60.0)
                )
                continue

            if not in_flight:
                self._settle_concurrency(cycle_setback, cycle_progress)
                # Nothing can be fired this instant: every render retry and
                # every re-submit is still waiting out its delay. Pace the loop
                # instead of spinning hot until one matures.
                if (self._render_deferred or resubmit) and not pending:
                    self._interruptible_sleep(0.25)
                continue

            # Poll one cycle: check each in-flight request once. Sleeps are
            # COALESCED to a single back-off at the end of the cycle (not one
            # per pending tile): with 8 tiles in flight, per-tile sleeps stacked
            # up to 8 x 5s = 40s of dead time per cycle and stalled completion
            # discovery. Here each tile only fires its (cheap) status GET, and
            # the loop backs off once, for the smallest back-off any pending
            # tile asked for.
            finished_ids = []
            next_backoff: float | None = None
            # Poll EVERY in-flight tile in ONE concurrent batch (~1 round-trip)
            # instead of one blocking GET per tile. This is the fix for the worker
            # under-driving the cloud backend: the old serial polls capped a run at
            # ~0.5 tiles/s no matter how high max_concurrent was, because all the
            # QgsBlockingNetworkRequest calls queued on this single thread. The
            # batch flows through the same QGIS network stack, just concurrently.
            poll_ids = list(in_flight.keys())
            responses = self._client.get_detection_status_many(
                poll_ids, self._auth, should_abort=self._should_abort)

            if self._stop_requested:
                break

            for request_id, resp in zip(poll_ids, responses):
                tile_idx, tile_spec, poll_interval, max_wait, deadline, tile_transform = (
                    in_flight[request_id]
                )

                status = resp.get("status")

                if status == "completed":
                    _, _, tile_w, tile_h = tile_spec
                    if self._emit_completed(
                        resp, tile_idx, tile_w, tile_h, tile_transform
                    ):
                        self.tiles_succeeded += 1
                        self._completed_idx.add(tile_idx)
                        cycle_progress = True
                    completed += 1
                    self._emit_progress(completed, total)
                    finished_ids.append(request_id)

                elif status == "failed":
                    err = resp.get("error", "unknown failure")
                    self.warning.emit(
                        f"Tile {tile_idx} failed: {err}"
                    )
                    completed += 1
                    self._emit_progress(completed, total)
                    finished_ids.append(request_id)

                elif status == "pending":
                    retry_after = float(resp.get("retry_after", poll_interval))
                    if time.monotonic() > deadline:
                        # Accepted, billed, never answered: its own outcome, not
                        # a completion (see tiles_timed_out). Progress still
                        # advances, since the run stops waiting on it.
                        self.tiles_timed_out += 1
                        self.warning.emit(
                            f"Tile {tile_idx} timed out after {int(max_wait)}s"
                        )
                        completed += 1
                        cycle_setback = True  # latency setback: narrow the window
                        self._emit_progress(completed, total)
                        finished_ids.append(request_id)
                    else:
                        # Record the smallest requested back-off; sleep once
                        # after the cycle (see below) instead of here per tile.
                        next_backoff = (
                            retry_after if next_backoff is None
                            else min(next_backoff, retry_after)
                        )

                else:
                    # Network/server error from get_detection_status.
                    code = resp.get("code", "")
                    if code in TRANSIENT_CODES:
                        # Transient poll error. Enforce the same deadline as the
                        # pending branch so a tile whose status keeps failing
                        # transiently times out instead of looping forever.
                        if time.monotonic() > deadline:
                            self.tiles_timed_out += 1
                            self.warning.emit(
                                f"Tile {tile_idx} timed out after {int(max_wait)}s"
                            )
                            completed += 1
                            cycle_setback = True  # latency setback: narrow the window
                            self._emit_progress(completed, total)
                            finished_ids.append(request_id)
                        else:
                            next_backoff = (
                                poll_interval if next_backoff is None
                                else min(next_backoff, poll_interval)
                            )
                    else:
                        self.warning.emit(
                            f"Tile {tile_idx}: unexpected poll response code={code}"
                        )
                        completed += 1
                        self._emit_progress(completed, total)
                        finished_ids.append(request_id)

            for rid in finished_ids:
                in_flight.pop(rid, None)

            # Quadrants queued by tiles that completed in THIS poll cycle.
            total += self._drain_subtiles(pending)

            # Fold this full cycle (submit + poll) into the adaptive width.
            self._settle_concurrency(cycle_setback, cycle_progress)

            # Back off ONCE per cycle, and only when nothing finished: if a tile
            # completed we loop straight back to refill its freed slot (keep the
            # server pipeline full) and re-poll without waiting. Capped at 5s and
            # sliced internally so a cancel still registers within ~0.25s.
            if in_flight and not finished_ids and next_backoff is not None and not self._stop_requested:
                self._interruptible_sleep(
                    min(max(next_backoff, self._min_poll_backoff_s), 5.0)
                )

        # Before the terminal, read the answers of the tiles already accepted.
        # Every break above leaves them open, and they are billed: the loop
        # exits on the stop flag with requests in flight, and the poll right
        # before the last break threw its own responses away. Which stops drain
        # is _BILLED_DRAIN_STOP_REASONS, the same set as the streaming loop.
        if self._stop_reason in _BILLED_DRAIN_STOP_REASONS and in_flight:
            completed = self._drain_polled_on_stop(in_flight, completed, total)

        # The terminal LAST, once nothing more will be emitted. This path runs
        # no converter pool, so there is nothing to drain first.
        if terminal_stop is not None:
            self._emit_stop(terminal_stop)
        self._emit_terminal()

    def _bbox_ground_width_m(self, bbox_native) -> float | None:
        """Geodesic ground width (meters) of one tile bbox at its
        mid-latitude, or None. Fail-open: any failure reports unknown."""
        try:
            da = self._distance_area
            if da is None:
                from qgis.core import (
                    QgsCoordinateReferenceSystem,
                    QgsDistanceArea,
                    QgsProject,
                )
                da = QgsDistanceArea()
                da.setSourceCrs(
                    QgsCoordinateReferenceSystem(self._crs_authid),
                    QgsProject.instance().transformContext(),
                )
                da.setEllipsoid("WGS84")
                self._distance_area = da
            from qgis.core import QgsPointXY

            from ..core.qt_compat import DistanceMeters

            xmin, ymin, xmax, ymax = bbox_native
            ymid = (ymin + ymax) / 2.0
            width = da.measureLine(
                QgsPointXY(xmin, ymid), QgsPointXY(xmax, ymid))
            width_m = da.convertLengthMeasurement(width, DistanceMeters)
            # Accept only a finite positive width. A rejection test written as
            # "width_m <= 0" lets NaN through, because every comparison against
            # NaN is False, and measureLine returns NaN when the tile falls
            # outside the source CRS transform's validity domain. That NaN then
            # travelled the whole way to the audit row and made the JSON encoder
            # reject it server side, so the run was billed but never recorded:
            # it disappeared from the user's library and became unreplayable.
            if not math.isfinite(width_m) or width_m <= 0:
                return None
            return width_m
        except Exception:  # nosec B110 -- best-effort measure
            return None

    def _tile_pixel_size_m(self, bbox_native, png_bytes) -> float | None:
        """True ground meters per SENT pixel for one encoded tile, or None.

        Geodesic width of the tile at its mid-latitude divided by the encoded
        image's pixel width, so a re-split quadrant (same ground extent, finer
        pixel grid) reports its real, finer resolution. Analytics-grade and
        fail-open: any failure reports unknown, never blocks the submission.
        """
        try:
            from ..core.cloud_detection import encoded_image_size

            size = encoded_image_size(png_bytes)
            if not size or size[0] <= 0:
                return None
            width_m = self._bbox_ground_width_m(bbox_native)
            if width_m is None:
                return None
            ratio = width_m / size[0]
            return round(ratio, 4) if math.isfinite(ratio) else None
        except Exception:  # nosec B110 -- best-effort analytics field
            return None

    def _apply_client_meta(self, submission: dict) -> None:
        """Merge the additive, optional per-run provenance + benchmark fields
        into one /predict submission. No-op when no client_meta was supplied,
        so the payload stays byte-identical to before; old servers ignore any
        unknown field.

        - plugin_version / policy_rev / prompt_mode ride every submission (which
          client and policy produced the run).
        - zone_geojson is the same polygon for the whole run, so it rides tile 0
          only.
        - clean_image is the pre-stamp tile image, present only for a tile a
          reference stamp was composited into (captured in _encode_tile).
        """
        meta = self._client_meta
        if not meta:
            return
        tile_idx = submission.get("tile_index")
        for key in ("plugin_version", "policy_rev", "prompt_mode"):
            val = meta.get(key)
            if val is not None:
                submission[key] = val
        if tile_idx == 0:
            zone = meta.get("zone_geojson")
            if zone is not None:
                submission["zone_geojson"] = zone
        clean = self._tile_clean_image.get(tile_idx)
        if clean is not None:
            submission["clean_image"] = clean

    def _release_tile_clean_image(self, tile_idx: int) -> None:
        """Drop one tile's pre-stamp image once that tile can no longer be
        submitted again. Each entry is a base64 PNG of a whole tile, so keeping
        them for the run costs hundreds of MB on a large grid. Called only from
        settle points (a response came back, or the tile was given up on):
        a tile waiting in the resubmit queue still needs its entry."""
        self._tile_clean_image.pop(tile_idx, None)

    def _build_submission(self, tile_idx: int, tile_spec, png_bytes) -> tuple[dict, dict]:
        """Build the (submission, tile_transform) pair for one encoded tile,
        shared by the batched and streaming paths."""
        from ..core.cloud_detection import mask_scale_field, tile_png_to_base64

        tile_x, tile_y, tile_w, tile_h = tile_spec
        tile_transform = self._make_tile_transform(tile_x, tile_y, tile_w, tile_h)
        bbox_native = tile_transform["bbox_native"]
        submission = {
            "run_id": self._run_id,
            "prompt": self._prompt,
            "image_b64": tile_png_to_base64(png_bytes),
            "tile_index": tile_idx,
            "crs_authid": self._crs_authid,
            "tile_bbox_wgs84": None,
            "tile_bbox_native": {
                "xmin": bbox_native[0], "ymin": bbox_native[1],
                "xmax": bbox_native[2], "ymax": bbox_native[3],
            },
            "pixel_size_m": self._tile_pixel_size_m(bbox_native, png_bytes),
            "max_masks": self._max_masks,
            "threshold": self._detection_threshold,
            "mask_threshold": None,
            "exemplars": self._tile_exemplars.get(tile_idx) or None,
            # Set for re-split quadrants only: lets the server bill the parent
            # once and treat its finer re-scan as part of the same paid work.
            # Older servers ignore the field (the quadrant is billed normally).
            "parent_tile_index": self._billed_ancestor_of(tile_idx),
        }
        # Additive, optional: ask for the coverage map only when the run opted
        # in (map-like text prompt + server dial on). Absent = today's request.
        if self._return_semantic:
            submission["return_semantic"] = True
        # Additive, optional: request the coarser mask grid for the whole run
        # (2 or absent). One value for the run, so re-splits and any re-detect
        # over the same grid stay on the same grid. Absent = the full grid.
        run_mask_scale = mask_scale_field(self._mask_scale)
        if run_mask_scale is not None:
            submission["mask_scale"] = run_mask_scale
        # Decoupled scan-gate billing: a tile whose completed scan already
        # carried its charge submits prepaid. Servers without the decoupled
        # flag ignore the field (the gate policy only turns on once the fleet
        # server understands it, so billing never drifts).
        if tile_idx in self._gate_prepaid:
            submission["charge_tiles"] = 0
        # Additive, optional per-run provenance + benchmark fields (None-safe:
        # absent client_meta leaves the payload byte-identical to today).
        self._apply_client_meta(submission)
        return submission, tile_transform

    def _run_streaming(self, total: int) -> None:
        """Continuous sliding-window detection for the synchronous direct
        endpoint. Keeps up to max_concurrent /predict posts in flight at all
        times; the instant any reply finishes it is converted + emitted and a new
        tile is fired, so the service never idles between batches and tiles stream in
        one-by-one. No poll phase (the direct endpoint answers each post with the
        finished masks), so each reply carries its own deadline and is given up
        on past it (_expire_stalled_replies). A cancel or an out-of-credits stop
        stops firing new tiles at once, then drains the already-in-flight set
        (bounded by the wind-down clock) so their billed masks are kept before
        sockets are released. Falls back to requeue+retry on transient and
        rate-limit codes and stops cleanly on exhausted/fatal, mirroring the
        batched path."""
        from qgis.core import QgsNetworkAccessManager
        from qgis.PyQt.QtCore import QCoreApplication, QEventLoop

        nam = QgsNetworkAccessManager.instance()
        pending: deque = deque(enumerate(self._tiles))
        # (tile_idx, tile_spec, png_bytes, not_before) - already encoded; not_before
        # is the monotonic instant the retry may be re-posted (0.0 = immediately).
        # Without it a busy server was re-hammered with zero delay, exactly the
        # synchronized-retry storm the queue is meant to absorb.
        resubmit: deque = deque()
        # reply -> (tile_idx, tile_spec, tile_transform, png_bytes, deadline).
        # deadline is the monotonic instant this reply is given up on: the
        # endpoint answers each post itself, so nothing else ever times one out
        # and a service that trickles bytes would hold the slot for the session.
        in_flight: dict = {}
        submit_attempts: dict[int, int] = {}
        busy_since: dict[int, float] = {}  # see _run_batched: time-budget busy retries
        completed = 0
        # The terminal stop this run decided (out of credits, fatal, offline).
        # Held, not emitted: it goes out after the drains below (see _mark_stop).
        terminal_stop: tuple | None = None
        # Consecutive per-tile rejections with no success in between
        # (see RUN_FATAL_CODES / _MAX_CONSECUTIVE_TILE_FATALS).
        fatal_streak = 0
        # Qt6 (QGIS 4) scopes these enums under ProcessEventsFlag; Qt5 (our 3.x
        # floor) exposes them flat on QEventLoop. Resolve compatibly.
        _ef = getattr(QEventLoop, "ProcessEventsFlag", QEventLoop)
        _wait = _ef.WaitForMoreEvents | _ef.AllEvents

        def fire_next() -> bool:
            """Encode+post the next pending/resubmit tile. Returns True if one was
            fired, False if nothing left to fire."""
            nonlocal completed
            # Blank/failed renders whose retry delay matured re-enter the queue.
            self._pump_render_deferred(pending)
            while resubmit or pending:
                # Jittered delays make later entries mature before the head, so
                # scan for the FIRST ready entry instead of gating on index 0
                # (head-of-line blocking wasted throughput and burned a blocked
                # tile's busy budget while it sat ready behind the head).
                ready_i = None
                now = time.monotonic()
                for i, entry in enumerate(resubmit):
                    if entry[3] <= now:
                        ready_i = i
                        break
                if ready_i is not None:
                    tile_idx, tile_spec, png_bytes, _ = resubmit[ready_i]
                    del resubmit[ready_i]
                elif pending:
                    tile_idx, spec = pending.popleft()
                    if tile_idx in self._gate_skip or tile_idx in self._prefilter_skip:
                        # Scan-settled or prefilter-degenerate tile: emit the
                        # fast empty result (the former already billed via its
                        # scan's charge_tiles, the latter never requested) and
                        # move on without rendering or posting anything. A
                        # render requested for it before the skip was known is
                        # released, never left holding a prefetch slot.
                        self._discard_prefetch(tile_idx)
                        self._settle_empty_tile(
                            tile_idx, charged=tile_idx in self._gate_skip)
                        completed += 1
                        self._emit_progress(completed, total)
                        continue
                    status, payload = self._encode_or_defer(tile_idx, spec)
                    if status == "defer":
                        continue
                    if status == "empty":
                        # Degenerate render caught at encode time: settle as a
                        # completed empty tile, no request spent so no charge
                        # (see _settle_empty_tile).
                        self._settle_empty_tile(tile_idx, charged=False)
                        completed += 1
                        self._emit_progress(completed, total)
                        continue
                    if status == "skip":
                        completed += 1
                        self._emit_progress(completed, total)
                        continue
                    tile_spec, png_bytes = payload
                else:
                    # Only back-off-delayed resubmits remain; nothing to fire yet.
                    return False
                submission, tile_transform = self._build_submission(
                    tile_idx, tile_spec, png_bytes
                )
                reply = self._client.post_detection_async(nam, submission, self._auth)
                in_flight[reply] = (
                    tile_idx, tile_spec, tile_transform, png_bytes,
                    time.monotonic() + self._stream_reply_budget_s,
                )
                # Imagery is no longer what the user is waiting on: a tile is on
                # the wire. Deduped, so this costs nothing on later tiles.
                self._emit_run_phase("detecting")
                # This tile's inference just started: use the wait for its
                # result to render the NEXT tiles concurrently on the main thread.
                self._request_render_prefetch(pending)
                return True
            return False

        # Geometry conversion runs off this loop (see the module docstring): a
        # dense tile takes longer to convert than to infer, and converting
        # inline stalled every socket in the window while it ran.
        self._convert_pool = TileConvertPool(self._convert_completed,
                                             workers=self._convert_workers)

        # Ask for the first renders BEFORE priming the window. Every later
        # prefetch is fired once a tile is already on the wire, so it hides
        # behind that tile's inference; the first one has nothing to hide
        # behind, and without this the very first tile rendered synchronously
        # on the main thread while nothing at all was in flight. It only reads
        # the head of the queue and records render tokens, so the tile order is
        # the same either way.
        self._request_render_prefetch(pending)

        # Prime the window at the current adaptive (AIMD) width, not the full
        # max_concurrent: it opens narrow and grows per clean cycle. Guarded
        # like every other refill: a cancel during the first tile render must
        # not still put tiles on the wire.
        while (not self._stop_requested and len(in_flight) < self._aimd.cap and fire_next()):
            pass

        while (
            in_flight or resubmit or pending or self._render_deferred or self._convert_pool.pending
        ) and not self._stop_requested:
            if not in_flight:
                if not (pending or resubmit or self._render_deferred):
                    # Only conversions are still owed: wait ON them rather than
                    # spinning, so the run's tail is as short as the last tile's
                    # geometry and not a sleep ladder.
                    self._settle_converted_batch(
                        self._convert_pool.drain(timeout=0.25))
                    continue
                # Everything in flight drained while retries wait out their
                # back-off (a fully busy server can reach this): pace with an
                # interruptible sleep and try to refill, instead of exiting the
                # run with tiles still owed.
                self._interruptible_sleep(0.25)
                # The sleep returns EARLY on a stop, and fire_next() drains the
                # retry queue before the pending one, so without this guard a
                # cancel arriving here posts and bills up to cap more tiles.
                while (not self._stop_requested and len(in_flight) < self._aimd.cap and fire_next()):
                    pass
                self._settle_converted_batch(self._convert_pool.drain())
                continue
            # Block until a network event arrives (or 250ms), so this loop never
            # busy-spins; a cancel registers within one slice.
            QCoreApplication.processEvents(_wait, 250)
            if self._stop_requested:
                break

            done = [r for r in in_flight if self._reply_is_finished(r)]
            # Give up on replies past their per-tile budget. Checked before the
            # empty-cycle exit, because a stalled reply is exactly the case
            # where nothing ever finishes.
            expired = self._expire_stalled_replies(in_flight)
            if expired:
                # Progress only: the bar counts tiles the run is done waiting
                # on. The tile itself is booked as timed out (tiles_timed_out),
                # never as succeeded, so a billed tile that answered nothing
                # stays visible in the run summary.
                completed += expired
                self._emit_progress(completed, total)
                self._aimd.on_setback()  # latency setback: narrow the window
            if not done:
                # Nothing finished on the wire this slice, so the converters own
                # the cycle: hand over what they have instead of holding it
                # until some reply completes. A response body arrives in
                # chunks, so most wakes land here, and the geometry of a tile
                # that converted early used to wait out a whole other tile's
                # inference before it could reach the canvas.
                self._settle_converted_batch(self._convert_pool.drain())
                continue

            # Per-cycle AIMD signals for this drain (see _run_batched).
            cycle_setback = False
            cycle_progress = False
            stop_payload = None
            for reply in done:
                tile_idx, tile_spec, tile_transform, png_bytes, _ = in_flight.pop(reply)
                response = self._read_reply(tile_idx, reply)
                outcome = self._classify_submit_response(tile_idx, response, tile_transform)
                kind = outcome[0]
                if kind == "completed_inline":
                    _, resp, ttf = outcome
                    _, _, tile_w, tile_h = tile_spec
                    # Hand the masks to the converter pool and move on: the
                    # window must refill while this tile turns into geometry,
                    # not after. The tile is billed and answered either way, so
                    # it counts as succeeded here; a conversion that later
                    # throws forfeits only its geometry (see _settle_converted).
                    self._convert_pool.submit(
                        self._plan_completed(resp, tile_idx, tile_w, tile_h, ttf))
                    self.tiles_succeeded += 1
                    self._completed_idx.add(tile_idx)
                    cycle_progress = True
                    # Network round-trip succeeded regardless of local decode:
                    # not an offline run, so reset the fatal streak + fast-fail.
                    fatal_streak = 0
                    self._fastfail.reset()
                    completed += 1
                    self._emit_progress(completed, total)
                elif kind == "retry":
                    # Policy shared with the batched loop (_retry_decision);
                    # the not_before stamp paces the re-post (see resubmit).
                    give_up, delay, setback = self._retry_decision(
                        tile_idx, outcome, busy_since, submit_attempts)
                    cycle_setback = cycle_setback or setback
                    if give_up:
                        self._skip_network_tile(tile_idx)
                        completed += 1
                        self._emit_progress(completed, total)
                    else:
                        resubmit.append(
                            (tile_idx, tile_spec, png_bytes,
                             time.monotonic() + delay))
                elif kind == "ok":
                    # A "pending" reply on the direct path is unexpected (the
                    # endpoint is synchronous). Treat as a skip so the run never
                    # hangs waiting to poll a path that does not exist here.
                    self.warning.emit(
                        f"Tile {tile_idx}: unexpected pending on direct path; skipping"
                    )
                    self._release_tile_clean_image(tile_idx)
                    completed += 1
                    self._emit_progress(completed, total)
                elif kind == "skip":
                    self._release_tile_clean_image(tile_idx)
                    completed += 1
                    self._emit_progress(completed, total)
                elif kind == "tile_fatal":
                    # Per-tile rejection: skip this tile, keep the run alive.
                    # A streak with no success in between is systematic (see
                    # _run_batched) and stops the run with that code.
                    bad_code = outcome[1] or "UNKNOWN"
                    fatal_streak += 1
                    self.warning.emit(
                        f"Tile {tile_idx}: rejected ({bad_code}); skipping")
                    self._release_tile_clean_image(tile_idx)
                    completed += 1
                    self._emit_progress(completed, total)
                    if stop_payload is None and fatal_streak >= self._max_tile_fatals:
                        stop_payload = ("fatal", bad_code)
                elif stop_payload is None:  # "exhausted" or "fatal"
                    stop_payload = outcome

            # Every reply of this cycle has been read and is out of in_flight,
            # so each one can be destroyed now: that frees its upload buffer
            # instead of holding it for the whole run.
            self._free_read_replies(done)

            # Offline fast-fail + terminal stop, shared with the batched loop
            # (see _offline_stop). Recorded only: the signal goes out after the
            # drains below, so the tiles this run already paid for are not
            # emitted into a finalized review.
            stop_payload = self._offline_stop(stop_payload)
            if stop_payload is not None:
                terminal_stop = stop_payload
                self._mark_stop(stop_payload)
                break

            # Fold this drain cycle into the adaptive width, then refill to it so
            # the window stays full (continuous, no barrier). Quadrants queued by
            # saturated tiles in THIS cycle join the deque first, so the refill
            # can fire them and the loop guard sees them before exiting.
            total += self._drain_subtiles(pending)
            self._settle_concurrency(cycle_setback, cycle_progress)
            # Backpressure. When the converters are slower than the network,
            # firing more tiles only grows a backlog of undelivered masks in
            # memory; the run cannot finish sooner than they can convert. Past
            # the cap, spend the cycle draining instead of firing.
            backlog_cap = max(1, self._convert_pool.workers * self._convert_backlog_per_worker)
            if self._convert_pool.pending >= backlog_cap:
                self._settle_converted_batch(
                    self._convert_pool.drain(timeout=0.25))
            # Do not start new tiles once a stop is pending: a cancel that landed
            # during this cycle's reply-processing would otherwise fire fresh work
            # (a blocking main-thread render + a POST) that then has to be drained
            # or aborted, delaying the wind-down. A tile not fired is not billed.
            while not self._stop_requested and len(in_flight) < self._aimd.cap and fire_next():
                pass
            # Keep the render pipeline fed even on cycles where no slot freed
            # (e.g. only deferred retries matured): the prefetch is what hides
            # the per-tile render behind the in-flight inference.
            if not self._stop_requested:
                self._request_render_prefetch(pending)
            # Emit whatever the converters finished while the window refilled.
            # LAST on purpose: the sockets are already busy again, so this only
            # spends time the run would have spent waiting anyway.
            self._settle_converted_batch(self._convert_pool.drain())

        # ONE clock for the whole wind-down, shared by the reply drain and the
        # converter drain below. Each used to take the full stop budget, so a
        # drain that finished early let its share lapse while the converters
        # were cut off holding billed tiles. Sized so the terminal still goes
        # out inside the window the GUI waits for it, and unload's join with it.
        wind_down_end = time.monotonic() + 2 * self._stop_drain_budget_s

        # Before releasing sockets, drain the tiles ALREADY in flight. The
        # server bills a tile when it processes the request, so a reply we abort
        # unread is a detection paid for and thrown away. New tiles stopped
        # firing the instant the stop landed (the real cost + time driver), so
        # this only waits out the small in-flight set already sent, bounded by
        # the wind-down clock so a hung reply can never hold the stop open.
        # Which stops drain is _BILLED_DRAIN_STOP_REASONS: out of credits, where
        # the wall stops the NEXT tile and the ones already accepted were
        # charged before it, and the run-wide stall, where the watchdog winds
        # the worker down with requests still open on the service.
        # Drain whatever is in flight, including on a run where no tile has
        # landed yet. A cold service is the case where this matters most: the
        # user waited, saw nothing, and pressed Cancel while the first tiles
        # were mid-inference. Aborting those replies unread destroys masks that
        # are already charged, and nothing hands the credits back. The budget
        # below is what keeps the stop prompt, not a precondition on success.
        if self._stop_reason in _BILLED_DRAIN_STOP_REASONS and in_flight:
            drain_deadline = min(
                wind_down_end, time.monotonic() + self._stop_drain_budget_s)
            while in_flight and time.monotonic() < drain_deadline:
                QCoreApplication.processEvents(_wait, 100)
                drained = [r for r in in_flight if self._reply_is_finished(r)]
                for reply in drained:
                    tile_idx, tile_spec, tile_transform, png_bytes, _ = (
                        in_flight.pop(reply))
                    response = self._read_reply(tile_idx, reply)
                    outcome = self._classify_submit_response(
                        tile_idx, response, tile_transform)
                    if outcome[0] == "completed_inline":
                        _, resp, ttf = outcome
                        _, _, tile_w, tile_h = tile_spec
                        if self._emit_completed(resp, tile_idx, tile_w, tile_h, ttf):
                            self.tiles_succeeded += 1
                            self._completed_idx.add(tile_idx)
                        completed += 1
                        self._emit_progress(completed, total)
                    # A non-completed reply on a stop is not retried: it drops
                    # to the abort path below.
                self._free_read_replies(drained)

        # On stop, abort any still-in-flight replies so sockets are released.
        stragglers = list(in_flight.keys())
        for reply in stragglers:
            try:
                if not self._reply_is_finished(reply):
                    reply.abort()
            except (RuntimeError, AttributeError):
                pass
        in_flight.clear()
        self._free_read_replies(stragglers)

        # Every conversion must be emitted BEFORE the terminal: the main thread
        # finalizes the run on that signal, so geometry arriving after it would
        # be dropped after the user was billed for it. A stop settles what it
        # can too, for the same reason the reply drain above does, on whatever
        # the wind-down clock has left, so unload's join is never held.
        self._close_convert_pool(
            max(0.0, wind_down_end - time.monotonic()) if self._stop_requested
            else self._convert_drain_budget_s)

        # The terminal LAST, after every billed tile has been emitted. A run
        # that stopped itself says so here; _emit_terminal covers a user cancel
        # and a clean finish.
        if terminal_stop is not None:
            self._emit_stop(terminal_stop)
        self._emit_terminal()

    # ------------------------------------------------------------------
    # Empty-tile scan gate (packed multi-tile pre-scan)
    # ------------------------------------------------------------------

    def gate_summary(self) -> dict:
        """Run-level scan-gate counters for telemetry. Empty dict = the gate
        never armed for this run (no config / non-streaming path)."""
        return dict(self._gate_stats)

    def _gate_ground_mupp(self) -> float | None:
        """Ground meters per grid pixel of this run, or None. Same geodesic
        measure as _tile_pixel_size_m, taken on the first tile's rect."""
        try:
            tx, ty, tw, th = self._tiles[0]
            if tw <= 0:
                return None
            transform = self._make_tile_transform(tx, ty, tw, th)
            width_m = self._bbox_ground_width_m(transform["bbox_native"])
            if width_m is None or width_m <= 0:
                return None
            return width_m / tw
        except Exception:  # noqa: BLE001 - doubt disables the gate, not the run
            return None

    def _run_gate_scan(self) -> None:
        """Packed scan phase before detection (policy-gated, fail-open).

        Groups the run grid into blocks of neighboring tiles, renders each
        member once, downsamples the group into ONE tile-sized scan image and
        posts it as a normal detection request that carries the whole group's
        charge (charge_tiles = member count, so the run still bills the full
        grid; the kept tiles' detect requests are then prepaid). Quadrants
        with no instance evidence mark their tile as skip; kept tiles reuse
        the full-res render. EVERY doubt (render failure, network give-up,
        decode problem, resolution cap, credit exhaustion mid-scan) falls
        open: affected tiles just take the normal full-detection path, so the
        gate can cost requests but never drop objects beyond its validated
        operating point.
        """
        cfg = self._gate_config
        if not cfg:
            return
        stats = {"scans": 0, "blocks": 0, "skipped": 0, "prepaid": 0,
                 "unscanned": 0, "prefiltered": 0, "fallback": None,
                 "scan_ms": 0}
        self._gate_stats = stats
        if self._stamps or self._collect_raw or not (self._prompt or "").strip():
            stats["fallback"] = "not_text_run"
            self._track_gate_scan(stats, 0)
            return
        from ..core import scan_gate

        try:
            base_group = int(cfg.get("group", 2))
            max_group = int(cfg.get("max_group", base_group))
            min_score = float(cfg.get("min_score", 0.0))
            min_px = max(1, int(cfg.get("min_pixels", 8)))
        except (TypeError, ValueError):
            stats["fallback"] = "bad_config"
            self._track_gate_scan(stats, 0)
            return
        if base_group < 2 or not 0.0 < min_score <= 1.0:
            stats["fallback"] = "bad_config"
            self._track_gate_scan(stats, 0)
            return
        cap = cfg.get("max_scan_mupp")
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
            cap = None
        # Adaptive packing with a resolution guard: the packed scan sees the
        # ground at block-side-times the run resolution, so the side deepens
        # only while the class's validated cap holds, and past the cap even
        # the smallest packing can no longer prove a tile empty, so the gate
        # stands down entirely (scan_gate.scan_group).
        mupp = self._gate_ground_mupp() if cap is not None else None
        group = scan_gate.scan_group(
            base_group, max_group, None if cap is None else float(cap), mupp)
        if group == 0:
            stats["fallback"] = "resolution"
            self._track_gate_scan(stats, base_group)
            return

        # Singleton blocks are never scanned: one scan there costs exactly one
        # detect, so the detect itself is always the better request.
        blocks = [b for b in scan_gate.group_tiles(self._tiles, group)
                  if len(b) >= 2]
        if not blocks:
            stats["fallback"] = "no_blocks"
            self._track_gate_scan(stats, group)
            return
        stats["blocks"] = len(blocks)

        from qgis.core import QgsNetworkAccessManager
        from qgis.PyQt.QtCore import QCoreApplication, QEventLoop

        nam = QgsNetworkAccessManager.instance()
        _ef = getattr(QEventLoop, "ProcessEventsFlag", QEventLoop)
        _wait = _ef.WaitForMoreEvents | _ef.AllEvents

        t0 = time.monotonic()
        pending: deque = deque(enumerate(blocks))
        resubmit: deque = deque()  # (block_i, block, submission, not_before)
        in_flight: dict = {}       # reply -> (block_i, block, submission)
        submit_attempts: dict[int, int] = {}
        busy_since: dict[int, float] = {}
        exhausted_payload = None

        def fire() -> bool:
            while resubmit or pending:
                now = time.monotonic()
                ready_i = None
                for i, entry in enumerate(resubmit):
                    if entry[3] <= now:
                        ready_i = i
                        break
                if ready_i is not None:
                    block_i, block, submission, _ = resubmit[ready_i]
                    del resubmit[ready_i]
                elif pending:
                    block_i, block = pending.popleft()
                    submission, block = self._build_scan_submission(
                        block_i, block, group)
                    if submission is None:
                        continue  # unscannable block: its tiles stay kept
                else:
                    return False
                reply = self._client.post_detection_async(
                    nam, submission, self._auth)
                in_flight[reply] = (block_i, block, submission)
                return True
            return False

        while (in_flight or resubmit or pending) and not self._stop_requested:
            while len(in_flight) < self._max_concurrent and fire():
                pass
            if not in_flight:
                if resubmit or pending:
                    self._interruptible_sleep(0.25)
                    continue
                break
            QCoreApplication.processEvents(_wait, 250)
            if self._stop_requested:
                break
            # Only the replies actually popped below may be freed: this loop can
            # break early (exhausted), and the wind-down owns whatever is left.
            read_replies: list = []
            for reply in [r for r in in_flight if self._reply_is_finished(r)]:
                block_i, block, submission = in_flight.pop(reply)
                read_replies.append(reply)
                response = self._read_reply(-(block_i + 1), reply)
                outcome = self._classify_submit_response(
                    -(block_i + 1), response, {})
                kind = outcome[0]
                if kind == "completed_inline":
                    self._fastfail.reset()
                    skip, _keep = scan_gate.classify_block(
                        block, outcome[1], group, min_score, min_px)
                    self._gate_skip |= skip
                    self._gate_prepaid |= {idx for idx, _qr, _qc in block}
                    stats["scans"] += 1
                elif kind == "retry":
                    give_up, delay, _setback = self._retry_decision(
                        -(block_i + 1), outcome, busy_since, submit_attempts)
                    if give_up:
                        stats["unscanned"] += 1  # fail open: tiles stay kept
                    else:
                        resubmit.append((block_i, block, submission,
                                         time.monotonic() + delay))
                elif kind == "exhausted":
                    # Credits ran out on a scan: end the run exactly like a
                    # detect-phase exhaustion (the streaming loop below exits
                    # immediately and the terminal handling salvages nothing,
                    # since no detection has run yet).
                    exhausted_payload = outcome
                    break
                else:
                    # skip / tile_fatal / fatal / unexpected pending: fail
                    # open for this block. A systemic failure (bad key,
                    # subscription) surfaces identically in the detect phase,
                    # which owns the full terminal handling for it.
                    stats["unscanned"] += 1
            # Free this cycle's replies and their packed-scan upload buffers
            # now: nothing here enters a QEventLoop, so deleteLater alone would
            # hold every scan image until the thread ends.
            self._free_read_replies(read_replies)
            if exhausted_payload is not None:
                break
            # A link that only ever fails hard is offline: stop burning the
            # scan retry budget; the detect phase trips its own offline stop
            # quickly and salvages/aborts with the normal messaging.
            if self._fastfail.tripped and self.tiles_succeeded == 0:
                stats["fallback"] = "offline"
                break

        # Wind-down: abort whatever is still in flight (stop requested,
        # exhausted, or offline bail). A completed-but-unread scan's skip
        # decisions are simply not applied; its tiles detect normally.
        unread = list(in_flight)
        for reply in unread:
            try:
                if not self._reply_is_finished(reply):
                    reply.abort()
            except (RuntimeError, AttributeError):
                pass
        in_flight.clear()
        self._free_read_replies(unread)

        if exhausted_payload is not None:
            self._emit_stop(exhausted_payload)

        # Only kept tiles reuse their scan-phase render; skipped tiles never
        # encode again, so their cached bytes are dropped now.
        for idx in list(self._gate_tile_bytes):
            if idx in self._gate_skip:
                self._gate_tile_bytes.pop(idx, None)
        stats["skipped"] = len(self._gate_skip)
        stats["prepaid"] = len(self._gate_prepaid)
        stats["prefiltered"] = len(self._prefilter_skip)
        stats["scan_ms"] = int((time.monotonic() - t0) * 1000)
        self.tiles_gate_skipped = len(self._gate_skip)
        logger.debug(
            "AutoDetectionWorker: gate scan %d blocks -> %d scans, "
            "%d skipped, %d prepaid, %d prefiltered (%d ms)",
            stats["blocks"], stats["scans"], stats["skipped"],
            stats["prepaid"], stats["prefiltered"], stats["scan_ms"],
        )
        # Same counts to the QGIS log, not just the Python logger. A gate skip
        # is the one drop that leaves a tile-shaped hole in a run the user paid
        # for, so it has to be readable from the message log alone: that is the
        # only place a field report ever comes with.
        if self._gate_skip:
            try:
                from qgis.core import QgsMessageLog

                QgsMessageLog.logMessage(
                    f"Auto detection: scan gate skipped "
                    f"{len(self._gate_skip)} of {len(self._tiles)} tile(s) "
                    f"as empty at 1/{group} resolution "
                    f"({stats['scans']} scan(s), {stats['scan_ms']} ms). "
                    f"These were charged and got no detection pass.",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
            except Exception:  # noqa: BLE001 - a log line must never end a run
                pass  # nosec B110
        self._track_gate_scan(stats, group)

    def _track_gate_scan(self, stats: dict, group: int) -> None:
        """One auto_gate_scan event per armed run (scan ran OR stood down).
        Off the GUI thread: track() only queues, the next main-thread flush
        ships it (same contract as run()'s crash reporting)."""
        try:
            from ..core.telemetry_run_events import track_auto_gate_scan
            track_auto_gate_scan(
                run_id=self._run_id,
                tiles=len(self._tiles),
                group=group,
                scans=int(stats.get("scans", 0)),
                blocks=int(stats.get("blocks", 0)),
                tiles_skipped=int(stats.get("skipped", 0)),
                tiles_prepaid=int(stats.get("prepaid", 0)),
                tiles_unscanned=int(stats.get("unscanned", 0)),
                tiles_prefiltered=int(stats.get("prefiltered", 0)),
                fallback=str(stats.get("fallback") or ""),
                scan_ms=int(stats.get("scan_ms", 0)),
            )
        except Exception:  # noqa: BLE001 - telemetry must never hurt the run
            pass  # nosec B110

    def _build_scan_submission(self, block_i: int, block: list, group: int):
        """Render + pack one gate block into a single scan submission.

        Returns (submission, scanned_block) where scanned_block holds only the
        members that actually rendered (the rest stay unscanned = kept), or
        (None, None) when fewer than 2 members rendered (scanning one tile
        costs exactly one detect, so the detect is always the better call).
        The scan bills its rendered members (charge_tiles), keeping the run's
        total at the full grid regardless of how many tiles are skipped.
        Degenerate members (prefilter) are the one exception: they settle as
        empty with no request at all, so they join neither the scan nor its
        charge, exactly like the pre-existing blank-tile drop.
        """
        from qgis.PyQt.QtCore import QRect
        from qgis.PyQt.QtGui import QImage, QPainter

        from ..core.cloud_detection import (
            encode_tile_png,
            tile_is_blank,
            tile_is_degenerate,
            tile_png_to_base64,
        )
        from ..core.qt_compat import resolve_qt_enum
        from ..core.tile_manager import TILE_SIZE

        cell_px = max(1, TILE_SIZE // group)
        canvas_px = cell_px * group
        # Scoped-then-flat enum resolution (Qt5 flat / Qt6 scoped), the same
        # string-based helper the rest of the plugin uses for Qt enum compat.
        fmt = resolve_qt_enum(QImage, "Format", "Format_RGB32")
        canvas = QImage(canvas_px, canvas_px, fmt)
        canvas.fill(0xFF808080)  # mid-gray: neutral fill for absent cells
        painter = QPainter(canvas)
        try:
            hint = resolve_qt_enum(QPainter, "RenderHint", "SmoothPixmapTransform")
            painter.setRenderHint(hint, True)
            scanned: list = []
            bbox_union: list | None = None
            for idx, qr, qc in block:
                tx, ty, tw, th = self._tiles[idx]
                tile_img = None
                for _ in range(self._gate_scan_render_tries):
                    if self._stop_requested:
                        return None, None
                    img = self._tile_renderer(tx, ty, tw, th)
                    if img is None or img.isNull():
                        continue
                    # Degenerate prefilter: a provably-objectless member never
                    # joins the scan (nor its charge_tiles); it settles as an
                    # empty result in the detect loop. Dropping it here also
                    # densifies the block with tiles worth scanning.
                    if self._prefilter is not None and tile_is_degenerate(
                        img,
                        self._prefilter["nodata_frac"],
                        self._prefilter["band_eps"],
                        self._prefilter["nodata_rgb_eps"],
                        self._prefilter["min_valid_px"],
                    ):
                        self._prefilter_skip.add(idx)
                        self.tiles_prefiltered += 1
                        break
                    if not tile_is_blank(img):
                        tile_img = img
                        break
                if idx in self._prefilter_skip:
                    continue  # settled empty later, never scanned or charged
                if tile_img is None:
                    continue  # unscanned member: normal detect path later
                # Reuse: encode the full-res render once for the detect phase
                # (bounded; a miss just re-renders there).
                if len(self._gate_tile_bytes) < self._gate_render_cache_max:
                    encoded = encode_tile_png(tile_img, 0, 0, tw, th)
                    if encoded is not None:
                        (_sx, _sy, cw, ch), data = encoded
                        self._gate_tile_bytes[idx] = ((tx, ty, cw, ch), data)
                painter.drawImage(
                    QRect(qc * cell_px, qr * cell_px, cell_px, cell_px),
                    tile_img)
                scanned.append((idx, qr, qc))
                transform = self._make_tile_transform(tx, ty, tw, th)
                bn = transform["bbox_native"]
                if bbox_union is None:
                    bbox_union = list(bn)
                else:
                    bbox_union = [
                        min(bbox_union[0], bn[0]), min(bbox_union[1], bn[1]),
                        max(bbox_union[2], bn[2]), max(bbox_union[3], bn[3]),
                    ]
        finally:
            painter.end()
        if len(scanned) < 2:
            for idx, _qr, _qc in scanned:
                self._gate_tile_bytes.pop(idx, None)
            return None, None
        packed = encode_tile_png(canvas, 0, 0, canvas_px, canvas_px)
        if packed is None:
            return None, None
        _crop, data = packed
        submission = {
            "run_id": self._run_id,
            "prompt": self._prompt,
            "image_b64": tile_png_to_base64(data),
            # Negative index namespace: scan requests never collide with a
            # real tile's audit/idempotency identity in the same run.
            "tile_index": -(block_i + 1),
            "crs_authid": self._crs_authid,
            "tile_bbox_wgs84": None,
            "tile_bbox_native": None if bbox_union is None else {
                "xmin": bbox_union[0], "ymin": bbox_union[1],
                "xmax": bbox_union[2], "ymax": bbox_union[3],
            },
            "pixel_size_m": None,
            "max_masks": self._max_masks,
            # The run's recall floor: the scan must see everything plausible.
            "threshold": self._detection_threshold,
            "mask_threshold": None,
            "exemplars": None,
            "parent_tile_index": None,
            # Decoupled billing: this scan carries its rendered members'
            # whole charge; their later detect requests are prepaid (0).
            # Servers without the decoupled-charge flag ignore the field.
            "charge_tiles": len(scanned),
        }
        # Provenance fields ride the scan too (same run). The negative scan
        # tile_index carries no zone_geojson / clean_image by construction.
        self._apply_client_meta(submission)
        return submission, scanned

    def _settle_empty_tile(self, tile_idx: int, charged: bool = True) -> None:
        """Settle one tile as a fast empty result: it flows through the same
        signals as a zero-mask detection (tile_completed + progress), so run
        accounting, review and export see a completed empty tile, never a
        hole. Two producers: a scan-skipped tile (billed via its scan's
        charge_tiles) and a prefilter-degenerate tile (no request at all).

        ``charged`` False is the prefilter case. It stays out of
        tiles_succeeded, which counts the tiles that spent a request or a scan
        charge and is what the user is shown as "N of M tiles"; the prefilter
        has its own count in tiles_prefiltered.
        """
        try:
            self.tile_completed.emit(tile_idx, [])
        except RuntimeError:
            return
        if charged:
            self.tiles_succeeded += 1
        self._completed_idx.add(tile_idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep up to `seconds`, waking early if a stop has been requested.

        Keeps the poll-loop back-offs responsive to cancellation: a stop
        registers within one slice (~0.25s) instead of after the full back-off.
        """
        waited = 0.0
        step = 0.25
        while waited < seconds and not self._stop_requested:
            time.sleep(step)
            waited += step

    def _prepare_stamps(self) -> None:
        """Prepare the reference-example crops, sized for stamping into tiles.

        Preferred (and the default for every exemplar run): the plugin pre-renders
        crisp natural-context crops from the LAYER (exemplar_stamps) as
        ``(crop, label, obj_box)`` and passes them in; we just resize them (and
        scale obj_box with the resize). obj_box frames the drawn object inside the
        crop so the tile compositor sends the cloud model a box tight to the object while the
        crop still provides the surrounding context the model uses. The crop is
        the NATURAL imagery (real surrounding pixels, no grey mask, no blur): the
        tight box, not a painted background, is what keeps neighbour leak down.
        Each item may carry a 4th element, the exemplar's FULL-image pixel box
        (kept in _stamp_full_boxes, parallel to _stamps): the home-tile split
        in _split_stamps_for_tile reads it. No-op (empty stamps) for text-only
        runs."""
        self._stamps = []
        self._stamp_full_boxes = []
        self._stamp_regions = []
        from qgis.PyQt.QtCore import Qt as _Qt

        from ..core.cloud_detection import stamp_size_cap

        if not self._exemplar_stamps_in:
            return

        # Collect the valid pre-rendered crops FIRST: the paste size depends on
        # how many there are, since they share one horizontal row that must stay
        # inside the tile overlap (stamp_size_cap enforces that invariant, shared
        # with the compositor). Detections whose centroid lands on a pasted stamp
        # are dropped (the stamp is the example, not the ground), so the ground it
        # hides must always be seen CLEAN by a neighbouring tile; that only holds
        # while the whole band fits the overlap.
        valid: list = []
        for item in self._exemplar_stamps_in:
            if len(item) >= 3:
                crop, label, obj_box = item[0], item[1], item[2]
            else:
                crop, label = item[0], item[1]
                obj_box = None
            full_box = item[3] if len(item) > 3 else None
            region = bool(item[4]) if len(item) > 4 else False
            if crop is not None and crop.isNull():
                crop = None
            if region:
                crop = None  # region markers are never pasted
            if crop is None and full_box is None:
                # Neither pixels to paste nor a real object to point at.
                continue
            valid.append((crop, int(label), obj_box, full_box, region))
        if not valid:
            return

        # Size cap for the RUN-TOTAL exemplar count, even though a tile may
        # paste only a subset (home-tile exemplars are sent in-situ instead):
        # any subset then always fits the band, and a crop keeps ONE apparent
        # scale on every tile it is pasted on. Small crops keep their native
        # crispness; only larger ones are downsized here (smooth filter).
        # A None crop is an IN-SITU-ONLY exemplar (the plugin decided pasting
        # a shrunk copy hurts more than it helps, see should_paste_stamp): it
        # rides along for _split_stamps_for_tile's home-tile boxes and is
        # never pasted.
        cap = stamp_size_cap(len(valid))
        for crop, label, obj_box, full_box, region in valid:
            if crop is not None and max(crop.width(), crop.height()) > cap:
                prev_w = crop.width()
                crop = crop.scaled(
                    cap, cap,
                    _Qt.AspectRatioMode.KeepAspectRatio,
                    _Qt.TransformationMode.SmoothTransformation)
                # KeepAspectRatio scales both axes by the same factor.
                if obj_box is not None and prev_w > 0:
                    s = crop.width() / prev_w
                    obj_box = [float(v) * s for v in obj_box]
            self._stamps.append((crop, int(label), obj_box))
            self._stamp_full_boxes.append(full_box)
            self._stamp_regions.append(region)
        self._resolve_top_row_band_edge()

    def _split_stamps_for_tile(self, tx: int, ty: int, tw: int, th: int,
                               bottom: bool) -> tuple[list, list]:
        """Partition the run's stamps for ONE tile: (paste_stamps, in_situ_boxes).

        An exemplar whose full-image box is FULLY inside this tile's rect is its
        own best reference there: its example box is sent pointing at the REAL
        object (in_situ_boxes, tile-local px, same dict shape as the pasted
        boxes) and its crop is NOT pasted, so fewer real pixels are occluded.
        Every other exemplar keeps the pasted-stamp path on this tile. Two
        fallbacks keep it safe: a box straddling a tile edge stays on the stamp
        path everywhere (a partial view is not a faithful reference; see
        in_situ_exemplar_box), and an in-situ candidate lying under the tile's
        paste band reverts to its stamp whenever anything else pastes there
        (the band would overwrite its pixels)."""
        from ..core.cloud_detection import in_situ_exemplar_box, region_exemplar_box
        from ..core.tile_manager import OVERLAP_FRACTION, TILE_SIZE

        paste: list = []
        in_situ: list = []  # [(stamp, tile-local box)]
        region_boxes: list = []  # [(stamp, tile-local CLIPPED box)]
        for i, stamp in enumerate(self._stamps):
            full_box = (self._stamp_full_boxes[i]
                        if i < len(self._stamp_full_boxes) else None)
            if i < len(self._stamp_regions) and self._stamp_regions[i]:
                # Region marker (correction box): clipped to every overlapping
                # tile; a partial view of an AREA keeps its meaning, unlike an
                # object exemplar. Never pasted.
                local = region_exemplar_box(full_box, tx, ty, tw, th)
                if local is not None:
                    region_boxes.append((stamp, local))
                continue
            local = in_situ_exemplar_box(full_box, tx, ty, tw, th)
            if local is None:
                # An in-situ-only exemplar (None crop) has nothing to paste:
                # on tiles that do not fully contain it, it contributes
                # nothing (by design; a shrunk pasted copy tested worse).
                if stamp[0] is not None:
                    paste.append(stamp)
            else:
                in_situ.append((stamp, local))
        if paste and in_situ:
            # Stamps are sized for the RUN-total count (_prepare_stamps), so the
            # band still fits when a candidate moves back into the paste row.
            band_h = min(th, int(TILE_SIZE * OVERLAP_FRACTION))
            kept = []
            for stamp, local in in_situ:
                under_band = (
                    local[3] > th - band_h if bottom else local[1] < band_h)
                if under_band:
                    # The band would overwrite this real object's pixels; an
                    # in-situ-only exemplar (no crop) is dropped for THIS tile,
                    # a pastable one reverts to its stamp.
                    if stamp[0] is not None:
                        paste.append(stamp)
                else:
                    kept.append((stamp, local))
            in_situ = kept
        if paste and region_boxes:
            # A pasted band occludes real ground: shave the band strip off any
            # clipped region box instead of dropping it (a region spanning the
            # tile would otherwise die the moment anything pastes).
            band_h = min(th, int(TILE_SIZE * OVERLAP_FRACTION))
            shaved = []
            for stamp, local in region_boxes:
                x0, y0, x1, y1 = local
                if bottom:
                    y1 = min(y1, float(th - band_h))
                else:
                    y0 = max(y0, float(band_h))
                if (x1 - x0) >= 8.0 and (y1 - y0) >= 8.0:
                    shaved.append((stamp, [x0, y0, x1, y1]))
            region_boxes = shaved
        boxes = [
            {"box": [float(v) for v in local], "label": int(stamp[1])}
            for stamp, local in in_situ + region_boxes
        ]
        return paste, boxes

    def _resolve_top_row_band_edge(self) -> None:
        """Decide whether the TOP grid row bands its BOTTOM edge (fix the
        first-row coverage blind spot).

        A tile's band hides ground only the NEIGHBOURING row re-sees clean, so
        the top row (no row above) loses its top-band ground. When a row below
        overlaps enough that the top row's bottom band clears that row's own top
        band, stamp the top row along the bottom so its hidden ground falls in
        the overlap and is re-seen clean below. Too little overlap (the two bands
        would hide the same strip) keeps the top band, so the top-edge residual
        is never traded for an interior hole. Single-row grids keep the top band
        (residual, no row below). No-op for text-only runs (no stamps)."""
        self._top_stamp_ty = None
        self._stamp_bottom_top_row = False
        if not self._stamps or not self._tiles:
            return
        from ..core.cloud_detection import top_row_bottom_stamp_ok

        tys = sorted({int(t[1]) for t in self._tiles})
        if len(tys) < 2:
            return
        top_ty = tys[0]
        self._top_stamp_ty = top_ty
        top_th = max(int(t[3]) for t in self._tiles if int(t[1]) == top_ty)
        next_ty = tys[1]
        pasted = [crop for crop, _l, _b in self._stamps if crop is not None]
        if not pasted:
            # Every exemplar is in-situ-only: no band, nothing to place.
            self._top_stamp_ty = None
            return
        band_content_h = max(int(crop.height()) for crop in pasted)
        self._stamp_bottom_top_row = top_row_bottom_stamp_ok(
            top_ty, top_th, next_ty, band_content_h)

    def _pump_render_deferred(self, pending: deque) -> None:
        """Move render-retry tiles whose delay has matured back into the submit
        deque (front, so they keep priority over untouched tiles). Called by
        both run loops at the top of each fill/fire cycle."""
        if not self._render_deferred:
            return
        now = time.monotonic()
        matured = [e for e in self._render_deferred if e[0] <= now]
        if not matured:
            return
        for entry in matured:
            self._render_deferred.remove(entry)
            pending.appendleft((entry[1], entry[2]))

    def _request_render_prefetch(self, pending: deque) -> None:
        """Ask the main thread to render the next pending tiles NOW, while the
        current window's inference is in flight.

        The per-tile render is the serialized bottleneck of large runs, so it
        is pipelined ahead of the submits: prefetching hides the render behind
        the wait for the current tile's result. Depth-bounded so at most a
        couple of jobs run ahead; no-op when the renderer has no async API
        (tests, mocks) or the run is stopping. Only fresh pending tiles are
        prefetched: resubmits are already encoded, and deferred blank-retries
        only re-enter pending once their retry delay matured (so the ladder's
        delay is preserved)."""
        if self._render_request is None or self._stop_requested:
            return
        if time.monotonic() < getattr(self, "_prefetch_holdoff_until", 0.0):
            return
        # The LIVE width, which the adaptive window narrows on a link that
        # cannot feed the full depth (see _note_render_health).
        width = self._render_window.cap
        # islice, not list(pending)[:n]: this runs once per tile and pending
        # can hold hundreds of entries; only the head is ever needed.
        for tile_idx, spec in list(itertools.islice(pending, width)):
            if len(self._prefetched) >= width:
                return
            if tile_idx in self._prefetched:
                continue
            # Tiles that will never be encoded from a fresh render: the scan
            # gate settled them empty, the prefilter proved them degenerate, or
            # their full-res pixels are already cached from the scan phase.
            # Rendering them would waste a job and, worse, park a token in
            # _prefetched that no encode ever collects.
            if (
                tile_idx in self._gate_skip or tile_idx in self._prefilter_skip or tile_idx in self._gate_tile_bytes
            ):
                continue
            tx, ty, tw, th = spec
            out_w, out_h = self._tile_outsize.get(tile_idx, (0, 0))
            seq = self._render_request(tx, ty, tw, th, out_w, out_h)
            if seq is None:
                return
            self._prefetched[tile_idx] = seq

    def _note_render_health(self, waited_s: float, got_pixels: bool) -> None:
        """Feed one tile's imagery wait to the adaptive render window.

        ``waited_s`` is how long the worker sat on this tile's render, which on
        a prefetched tile is near zero (the pipeline was ahead) and rises the
        moment the renders stop keeping up. A wait past ``_render_slow_s``, or a
        render that came back with nothing, halves the number of basemap
        fetches the run keeps in flight; a prompt one grows it back one step.
        Nothing narrows on a link that answers quickly, so a good connection
        runs at the full depth exactly as before.
        """
        window = getattr(self, "_render_window", None)
        if window is None:
            return
        if not got_pixels or waited_s >= self._render_slow_s:
            window.on_setback()
            self.renders_slow += 1
        else:
            window.on_clean_cycle()
        self.render_window_floor = min(self.render_window_floor, window.cap)

    def _discard_prefetch(self, tile_idx: int) -> None:
        """Release a render requested ahead of time for a tile that ends up not
        being encoded (scan-settled, prefilter-settled, or served from the scan
        phase's cached bytes).

        Only _encode_tile collects a prefetch token, so a tile that bypasses it
        would leave its entry in _prefetched for good. Two of those fill the
        depth-bounded slot set permanently and every later tile falls back to a
        synchronous render, which is exactly the serialization the prefetch
        exists to hide. Collecting also hands the bridge's QImage back so it is
        freed instead of sitting in the bridge's result map."""
        seq = self._prefetched.pop(tile_idx, None)
        if seq is None or self._render_collect is None:
            return
        try:
            self._render_collect(seq)
        except Exception:  # nosec B110 -- releasing a render must never fail a run
            pass

    def _encode_or_defer(self, tile_idx: int, spec) -> tuple:
        """Encode one tile with the blank/failed-render retry ladder.

        Returns one of:
          ("ok", (tile_spec, image_bytes))  - ready to submit
          ("defer", None)                   - re-render queued (worker keeps it)
          ("empty", None)                   - degenerate render, settle as a
                                              completed EMPTY tile (no retry:
                                              the verdict is provable from the
                                              full-res pixels)
          ("skip", None)                    - permanently dropped (caller counts
                                              it done, exactly the old None path)

        A blank, unavailable or empty render is retried up to _RENDER_RETRY_MAX
        times with a PER-ATTEMPT DOUBLING delay from _RENDER_RETRY_DELAY_S (an
        online basemap usually just had not fetched that area yet, and a
        multi-second provider hiccup must not swallow the whole ladder); only
        when the ladder is exhausted is the tile counted in
        tiles_skipped_blank / tiles_render_failed / tiles_unavailable and
        skipped for good. A failure also holds the render PREFETCH off for
        _PREFETCH_HOLDOFF_S so concurrent fetches stop piling into a struggling
        provider.
        """
        # A kept tile whose full-res render was already produced (and encoded)
        # during the gate scan phase submits those bytes directly: render once,
        # reuse for detect. Popped so the cache never outlives its one use.
        cached = self._gate_tile_bytes.pop(tile_idx, None)
        if cached is not None:
            # This path skips _encode_tile, the only collector, so any render
            # already requested for the tile has to be released here.
            self._discard_prefetch(tile_idx)
            return ("ok", cached)
        tx, ty, tw, th = spec
        status, payload = self._encode_tile(tile_idx, tx, ty, tw, th)
        if status == "ok":
            self._render_attempts.pop(tile_idx, None)
            return ("ok", payload)
        if status == "empty":
            # Provably-objectless render: no retry (the ladder exists for
            # renders that might still LOAD, not for real uniform ground) and
            # no prefetch holdoff (the render itself was healthy).
            self.tiles_prefiltered += 1
            self._render_attempts.pop(tile_idx, None)
            return ("empty", None)
        if status in ("blank", "render", "unavailable"):
            self._prefetch_holdoff_until = time.monotonic() + self._prefetch_holdoff_s
        if status in ("blank", "render", "unavailable") and not self._stop_requested:
            attempts = self._render_attempts.get(tile_idx, 0)
            if attempts < self._render_retry_max:
                self._render_attempts[tile_idx] = attempts + 1
                delay = self._render_retry_delay_s * (2 ** attempts)
                self._render_deferred.append(
                    (time.monotonic() + delay, tile_idx, spec))
                return ("defer", None)
        # Ladder exhausted (or a non-retryable failure): count + skip.
        if status == "blank":
            self.tiles_skipped_blank += 1
        elif status == "render":
            self.tiles_render_failed += 1
        elif status == "unavailable":
            self.tiles_unavailable += 1
        self._render_attempts.pop(tile_idx, None)
        return ("skip", None)

    def _encode_tile(self, tile_idx: int, tx: int, ty: int, tw: int, th: int):
        """Produce + encode one tile, on this thread. The tile pixels come from
        the per-tile JIT render (tile_renderer, main-thread bridge). When
        reference examples exist, their crops are STAMPED into the tile and the
        per-tile example boxes + stamp region are recorded for submit/filter.

        Returns a (status, payload) pair:
          ("ok", ((tx, ty, cw, ch), image_bytes)) on success;
          ("render", None) when the render produced nothing (provider hole);
          ("empty", None) when the render is provably objectless (degenerate
          prefilter: all no-data or per-band uniform at full res), so the tile
          settles as a completed empty result with no request;
          ("blank", None) when the render is a uniform/nodata fill;
          ("unavailable", None) when an ONLINE source answered with its "no
          image here" placeholder card instead of imagery;
          ("skip", None) for everything else (cancelled, no bridge, encode
          failure). The caller (_encode_or_defer) owns the retry ladder and the
          skip counters, so this method never counts.

        The returned spec always carries the REAL (tx, ty) so
        _make_tile_transform maps it to the right ground bbox_native, even
        though a JIT tile image is sliced from its own origin (0, 0). Runs
        off the GUI thread (only the render itself hops to the main thread via the
        bridge); QImage/QBuffer are reentrant so the encode is safe here.
        """
        from ..core.cloud_detection import (
            composite_tile_with_stamps,
            encode_tile_png,
            tile_is_blank,
            tile_is_degenerate,
            tile_is_unavailable,
            tile_png_to_base64,
        )

        try:
            # Per-tile JIT render: get just THIS tile's pixels from the main-thread
            # bridge. The returned QImage is the tile at origin (0,0), so slice it
            # from (0, 0) but re-stamp the REAL (tx, ty) into the returned spec so
            # _make_tile_transform still maps it to the right ground bbox_native.
            # The bridge is always supplied (the whole-zone-slice path was removed);
            # a missing bridge is a bug, so fail the tile rather than guessing.
            if self._tile_renderer is None:
                return ("skip", None)
            if self._stop_requested:
                return ("skip", None)
            # A re-split quadrant renders its rect at 2x per depth (out size
            # from _tile_outsize): same ground, finer pixels. The tile SPEC
            # keeps the rect's grid coords so bbox_native stays exact; masks
            # map by their own returned pixel grid, so the geo-referencing is
            # unaffected by the upscale.
            out_w, out_h = self._tile_outsize.get(tile_idx, (0, 0))
            # A prefetched render (requested while earlier tiles were in
            # flight) is collected here; it is usually already done. Tiles
            # never prefetched take the synchronous request+wait path.
            prefetch_seq = self._prefetched.pop(tile_idx, None)
            render_t0 = time.monotonic()
            if prefetch_seq is not None and self._render_collect is not None:
                tile_img = self._render_collect(prefetch_seq)
            else:
                tile_img = self._tile_renderer(tx, ty, tw, th, out_w, out_h)
            got_pixels = tile_img is not None and not tile_img.isNull()
            if not got_pixels and self._stop_requested:
                # A stop wakes every pending render with None (request_stop
                # cancels the bridge). Nothing tried to load this tile, so it
                # is neither a slow link nor a coverage hole: skip it without
                # touching the render health or the failure counts, which is
                # what the run summary reports to the user.
                return ("skip", None)
            self._note_render_health(time.monotonic() - render_t0, got_pixels)
            if not got_pixels:
                # The render produced nothing: a provider/WMS hole or timeout,
                # not real ground. Retryable (the caller's ladder decides when
                # it becomes a counted coverage hole); never submitted, so
                # never billed.
                return ("render", None)
            # Degenerate prefilter: a FULL-RES render that is all no-data or
            # uniform in every band provably contains no object, so the tile
            # settles as a completed EMPTY result immediately (no request, no
            # retry ladder). Checked on the raw render, before any example
            # stamp is composited in. Re-split quadrants are exempt: a
            # saturated parent is dense by definition, and their withheld-
            # parent bookkeeping expects a real response per quadrant.
            if (
                self._prefilter is not None and not (out_w and out_h) and tile_is_degenerate(
                    tile_img,
                    self._prefilter["nodata_frac"],
                    self._prefilter["band_eps"],
                    self._prefilter["nodata_rgb_eps"],
                    self._prefilter["min_valid_px"],
                )
            ):
                return ("empty", None)
            # Blank/nodata tile: on an online basemap this is usually imagery
            # that was not downloaded yet, so it is retryable too. Once the
            # ladder is exhausted the caller skips it before submit, so an
            # empty region inside the zone (mosaic gap, out-of-footprint
            # corner) never spends a credit to return nothing. Checked on the
            # raw render, before any example stamp is composited in.
            if tile_is_blank(tile_img):
                return ("blank", None)
            # "No image here" placeholder card from an online source: real
            # pixels, so nothing above catches it, and sending it buys a
            # picture of a grey card. Retryable like a blank (a card can also
            # mean the source has not caught up yet), then dropped unbilled.
            if self._skip_unavailable_tiles and tile_is_unavailable(tile_img):
                return ("unavailable", None)
            src_x, src_y = 0, 0

            if out_w and out_h:
                # Upscaled quadrant: encode the WHOLE rendered image (its pixel
                # size is out_w x out_h, larger than the rect), but keep the
                # GRID-space rect in the returned spec so _make_tile_transform
                # maps it to the right ground bbox. Re-split tiles never carry
                # stamps (guarded in _maybe_subdivide).
                encoded = encode_tile_png(tile_img, 0, 0, out_w, out_h)
                if encoded is None:
                    return ("skip", None)
                _crop, data = encoded
                return ("ok", ((tx, ty, tw, th), data))

            if self._stamps:
                # The top grid row bands its bottom edge when that is the clean
                # placement (see _resolve_top_row_band_edge); every other row and
                # every single-row grid keeps the top edge.
                bottom = bool(
                    self._stamp_bottom_top_row and ty == self._top_stamp_ty)
                # Home-tile split: an exemplar fully contained in THIS tile is
                # sent as an in-situ box pointing at the real object (no paste);
                # every other exemplar pastes as usual.
                paste, insitu_boxes = self._split_stamps_for_tile(
                    tx, ty, tw, th, bottom)
                out = composite_tile_with_stamps(
                    tile_img, src_x, src_y, tw, th, paste, bottom=bottom)
                if out is None:
                    return ("skip", None)
                (_sx, _sy, cw, ch), data, ex_boxes, stamp_norm = out
                # Server cap: at most 8 exemplars per request. Pasted-crop
                # boxes must all stay (their pixels are in the image); the
                # in-situ/region tail is trimmed if it ever overflows.
                boxes = (ex_boxes + insitu_boxes)[:8]
                if boxes:
                    self._tile_exemplars[tile_idx] = boxes
                # stamp_norm only ever covers PASTED pixels: a tile that pasted
                # nothing must record NO stamp region at all, so the drop
                # filter can never discard a detection of the in-situ object.
                if stamp_norm:
                    self._tile_stamp_norm[tile_idx] = stamp_norm
                    # A non-empty stamp region means pixels were actually
                    # pasted, so the sent tile differs from the raw render.
                    # Capture the pre-stamp image (clean_image) so a replay can
                    # reconstruct the un-stamped input. Only when client_meta
                    # opted in, so an ordinary run pays no extra encode.
                    if self._client_meta is not None:
                        clean = encode_tile_png(tile_img, src_x, src_y, tw, th)
                        if clean is not None:
                            self._tile_clean_image[tile_idx] = (
                                tile_png_to_base64(clean[1]))
                return ("ok", ((tx, ty, cw, ch), data))
            encoded = encode_tile_png(tile_img, src_x, src_y, tw, th)
            if encoded is None:
                return ("skip", None)
            (_sx, _sy, cw, ch), data = encoded
            return ("ok", ((tx, ty, cw, ch), data))
        except Exception as exc:
            logger.warning("AutoDetectionWorker: tile encode failed at (%d,%d): %s",
                           tx, ty, exc)
            return ("skip", None)

    def _submit_batch(self, batch: list) -> list:
        """Submit a batch of (tile_idx, tile_spec, png_bytes) CONCURRENTLY.

        Returns one outcome tuple per item, in order, each one of:
          ("ok", request_id, poll_interval, max_wait, tile_transform)
          ("completed_inline", response, tile_transform)
                                     - sync fast path: masks already in the
                                       submit response (status=completed), no poll
          ("skip",)                  - drop the tile (caller counts it done)
          ("retry", retry_after_s)   - rate-limit/transient; caller requeues
          ("exhausted", remaining)   - credits/quota ran out
          ("fatal", code)            - non-retryable submit error

        Where the old _submit_tile uploaded one tile and slept on retries, this
        sends every tile in one batched round-trip and never sleeps: the caller
        requeues retryable tiles and paces with a single coalesced back-off, so a
        rate-limited batch can't block the thread mid-upload.
        """
        from ..core.cloud_detection import mask_scale_field, tile_png_to_base64

        run_mask_scale = mask_scale_field(self._mask_scale)
        submissions = []
        transforms = []
        for tile_idx, tile_spec, png_bytes in batch:
            tile_x, tile_y, tile_w, tile_h = tile_spec
            tile_transform = self._make_tile_transform(tile_x, tile_y, tile_w, tile_h)
            bbox_native = tile_transform["bbox_native"]
            transforms.append(tile_transform)
            submission = {
                "run_id": self._run_id,
                "prompt": self._prompt,
                "image_b64": tile_png_to_base64(png_bytes),
                "tile_index": tile_idx,
                "crs_authid": self._crs_authid,
                "tile_bbox_wgs84": None,
                "tile_bbox_native": {
                    "xmin": bbox_native[0],
                    "ymin": bbox_native[1],
                    "xmax": bbox_native[2],
                    "ymax": bbox_native[3],
                },
                "pixel_size_m": self._tile_pixel_size_m(bbox_native, png_bytes),
                "max_masks": self._max_masks,
                "threshold": self._detection_threshold,
                "mask_threshold": None,
                # Per-tile example boxes (where the stamps were pasted on THIS
                # tile); None/[] for text-only runs.
                "exemplars": self._tile_exemplars.get(tile_idx) or None,
                # Re-split quadrants carry their parent so the server can bill
                # the parent once for the whole re-scan (older servers ignore).
                "parent_tile_index": self._billed_ancestor_of(tile_idx),
            }
            # Additive, optional: ask for the coverage map only when the run
            # opted in (map-like text prompt + server dial on). Absent = today.
            if self._return_semantic:
                submission["return_semantic"] = True
            # Additive, optional: the run's coarser mask grid (2 or absent);
            # one value for the whole run so every tile shares the same grid.
            if run_mask_scale is not None:
                submission["mask_scale"] = run_mask_scale
            # Additive, optional per-run provenance + benchmark fields (None-safe:
            # absent client_meta leaves the payload byte-identical to today).
            self._apply_client_meta(submission)
            submissions.append(submission)

        responses = self._client.submit_detection_many(
            submissions, self._auth, should_abort=self._should_abort)
        outcomes = []
        for (tile_idx, _spec, _png), response, tile_transform in zip(
            batch, responses, transforms
        ):
            outcomes.append(
                self._classify_submit_response(tile_idx, response, tile_transform)
            )
        return outcomes

    def _classify_submit_response(self, tile_idx: int, response: dict, tile_transform: dict):
        """Map one /predict submit response to an outcome tuple (see _submit_batch).
        Pure: no network, no sleeping."""
        code = response.get("code", "")
        if "error" in response:
            if code in EXHAUSTED_CODES:
                return ("exhausted", _as_int(response.get("credits_remaining"), 0))
            if code == "RATE_LIMITED":
                try:
                    delay = float(response.get("retry_after", 0) or 0)
                except (TypeError, ValueError):
                    delay = 0.0
                # A queue-aware server also sends its waiting-room numbers
                # (queue_position/queue_depth/eta_seconds) so the UI can show
                # "you're in line, N ahead" instead of a silent stall. An old
                # server / plain rate limit has none -> generic busy (-1).
                position = _as_int(response.get("queue_position"))
                depth = _as_int(response.get("queue_depth"))
                eta = _as_int(response.get("eta_seconds"))
                self._note_busy(position, depth, eta)
                # 4th slot = code: the caller resets the offline fast-fail on a
                # busy answer (the server was reached, so the run is not offline).
                return ("retry", delay if delay > 0 else 5.0, True, "RATE_LIMITED")
            if code in TRANSIENT_CODES:
                if code == "SERVICE_WARMING":
                    # Submit deadline passed with no answer: the backend
                    # instance is cold-starting or busy. Surface the waiting
                    # room and retry on the queue TIME budget, so the tile is
                    # never skipped mid-boot by an attempt cap. is_busy
                    # (4th slot True) also resets the offline fast-fail: the
                    # link is fine, the service is just warming.
                    self._note_busy(-1, -1, 0)
                    return ("retry", 5.0, True, code)
                if code in ("SERVER_ERROR", "TIMEOUT"):
                    # Likely overload/cold start (the inference service's own 429/503 has a
                    # non-JSON body and lands here as SERVER_ERROR): tell the
                    # user the server is busy rather than staying silent.
                    self._note_busy(-1, -1, 0)
                else:
                    # Connectivity-side blip (no internet/DNS/proxy/SSL): a
                    # stale "spot reserved" label would mislead; restore the
                    # plain progress text while the silent retry runs.
                    self._note_flowing()
                # 4th slot = code: the caller feeds the offline fast-fail counter,
                # which advances only on hard-connectivity codes (DNS/refused/proxy)
                # and resets on a pure timeout / server-busy.
                return ("retry", 2.0, False, code)
            if code in BACKEND_UNAVAILABLE_CODES:
                # A cold instance's auth backend is not reachable yet (HTTP 503,
                # pre-charge / fail-closed): retry rather than skip, because it
                # stabilizes within seconds. Surface the waking-up state (same
                # as a SERVICE_WARMING answer) and hand a NON-busy retry to
                # _retry_decision, which paces it on the small backend-unavailable
                # attempt count. Not fed to the offline fast-fail (the link is
                # fine, only the service is warming).
                self._note_busy(-1, -1, 0)
                return ("retry", _BACKEND_UNAVAILABLE_DELAY_S, False, code)
            # Non-retryable error. Run-level codes end the run; anything else
            # rejects only THIS tile (the caller skips it and keeps going,
            # with a consecutive-rejection guard for systematic failures).
            if code in RUN_FATAL_CODES:
                return ("fatal", code)
            return ("tile_fatal", code)

        # Synchronous fast path: the server ran inference inline and returned the
        # masks directly in the submit response (status="completed"), so this tile
        # is already done - zero polls. A cold-start submit still returns
        # status="pending" and falls through to the polled path below. The
        # completed body has the same shape as a /status completed body, so the
        # caller decodes it with the same _handle_completed path.
        if response.get("status") == "completed":
            return ("completed_inline", response, tile_transform)

        request_id = response.get("request_id", "")
        if not request_id:
            self.warning.emit(
                f"Tile {tile_idx}: submit response missing request_id; skipping"
            )
            return ("skip",)

        poll_interval = _as_float(response.get("poll_interval"), self._poll_interval_s)
        max_wait = _as_float(response.get("max_wait"), self._poll_max_wait_s)
        # Defensive: an early server build shipped max_wait in milliseconds.
        # A ceiling above one hour can only be ms; normalize to seconds.
        if max_wait > 3600:
            max_wait = max_wait / 1000.0
        return ("ok", request_id, poll_interval, max_wait, tile_transform)

    def _make_tile_transform(
        self, tile_x: int, tile_y: int, tile_w: int, tile_h: int
    ) -> dict:
        """Build a tile_transform dict for polygon_exporter compatibility.

        geo_transform["bbox"] is (minx, miny, maxx, maxy) standard order.
        tile_transform["bbox"] is (minx, maxx, miny, maxy) -- polygon_exporter
        convention (existing code reads bbox[0]=minx, bbox[1]=maxx,
        bbox[2]=miny, bbox[3]=maxy).
        tile_transform["bbox_native"] is (minx, miny, maxx, maxy) standard order
        for the API payload.
        """
        src_bbox = self._geo_transform.get("bbox", (0.0, 0.0, 1.0, 1.0))
        img_shape = self._geo_transform.get("img_shape", (1, 1))
        img_h, img_w = img_shape[0], img_shape[1]

        # Avoid division by zero.
        img_w = max(img_w, 1)
        img_h = max(img_h, 1)

        src_minx, src_miny, src_maxx, src_maxy = src_bbox

        px_w = (src_maxx - src_minx) / img_w
        px_h = (src_maxy - src_miny) / img_h

        tile_minx = src_minx + tile_x * px_w
        tile_maxx = src_minx + (tile_x + tile_w) * px_w
        # y increases downward in pixel space, upward in map space.
        tile_miny = src_maxy - (tile_y + tile_h) * px_h
        tile_maxy = src_maxy - tile_y * px_h

        return {
            # polygon_exporter convention: (minx, maxx, miny, maxy)
            "bbox": (tile_minx, tile_maxx, tile_miny, tile_maxy),
            # standard order: (minx, miny, maxx, maxy) -- for API payload
            "bbox_native": (tile_minx, tile_miny, tile_maxx, tile_maxy),
            "img_shape": (tile_h, tile_w),
            "crs": self._crs_authid,
        }

    def _maybe_subdivide(self, tile_idx: int) -> bool:
        """Queue a saturated tile's 2x2 quadrants for detection at 2x scale.

        Returns True when the quadrants were queued (the parent's truncation
        will be retried finer), False when the ladder ends here (no budget, max
        depth, exemplar run, stop requested, or a degenerate/outside parent).
        Runs on the worker thread; the specs are drained into the submit queue
        by the run loops via _drain_subtiles. Exemplar runs never re-split:
        their stamps are rendered for the RUN scale, and re-stamping them into
        a 2x quadrant would reintroduce the apparent-scale mismatch.
        """
        from ..core.tile_manager import subdivide_quadrants

        if self._stop_requested or self._stamps:
            return False
        if self._resplit_time_spent():
            return False
        depth = self._tile_depth.get(tile_idx, 0)
        if depth >= self._subdiv_max_depth or self._subdivide_budget < 4:
            return False
        try:
            tx, ty, tw, th = self._tiles[tile_idx]
        except (IndexError, ValueError):
            return False
        quads = subdivide_quadrants(
            tx, ty, tw, th,
            overlap_fraction=self._subdiv_overlap,
            min_parent_px=self._subdiv_min_parent_px,
        )
        if not quads:
            return False
        quads = [q for q in quads if self._quad_intersects_zone(q)]
        if not quads or len(quads) > self._subdivide_budget:
            return False
        self._subdivide_budget -= len(quads)
        self.tiles_subdivided += 1
        for spec in quads:
            self._pending_subtiles.append((spec, depth + 1, tile_idx))
        self._mark_rescanning(tile_idx, len(quads))
        logger.debug(
            "AutoDetectionWorker: tile %d saturated, re-split into %d "
            "quadrant(s) at depth %d", tile_idx, len(quads), depth + 1,
        )
        return True

    def _quad_intersects_zone(self, spec) -> bool:
        """True if the quadrant's ground bbox touches the drawn zone (so a
        quadrant of a boundary parent that lies fully outside is never
        submitted or billed). No clip polygon (rectangle/MCP path) = True."""
        if self._clip_geom is None:
            return True
        try:
            from qgis.core import QgsGeometry, QgsRectangle

            bbox = self._make_tile_transform(*spec)["bbox_native"]
            rect = QgsGeometry.fromRect(
                QgsRectangle(bbox[0], bbox[1], bbox[2], bbox[3]))
            return bool(self._clip_geom.intersects(rect))
        except Exception:  # noqa: BLE001 - keep the quadrant on any doubt
            return True

    def _mark_rescanning(self, tile_idx: int, quads: int) -> None:
        """Book ``quads`` more inferences against the BASE tile this re-split
        belongs to, and tell the GUI to mark that ground the first time.

        Counted on the base ancestor, not on the tile that was just re-split: a
        depth-2 re-split is more work inside ground the GUI already marked, so
        it must extend that mark rather than draw a second one inside it."""
        root = self._billed_ancestor_of(tile_idx)
        if root is None:
            root = tile_idx
        first = root not in self._rescanning
        self._rescanning[root] = self._rescanning.get(root, 0) + quads
        if not first:
            return
        try:
            tx, ty, tw, th = self._tiles[root]
            bbox = self._make_tile_transform(tx, ty, tw, th)["bbox_native"]
            self.rescan_state.emit(root, bbox, True)
        except (IndexError, ValueError, KeyError, RuntimeError):
            self._rescanning.pop(root, None)

    def _settle_rescanning(self, tile_idx: int) -> None:
        """One quadrant is in: drop it from its base tile's outstanding count and
        clear the mark once that ground has been fully re-read. A quadrant that
        never settles (skip, fatal, stop) leaves its count short, which the
        terminal's clear-all covers."""
        self._settle_rescanning_root(self._billed_ancestor_of(tile_idx))

    def _settle_rescanning_root(self, root: int | None) -> None:
        """Same count-down, keyed on the BASE tile directly. A quadrant dropped
        before it ever got a tile index has no index to resolve, so it settles
        through the base its _mark_rescanning counted against."""
        if root is None or root not in self._rescanning:
            return
        left = self._rescanning[root] - 1
        if left > 0:
            self._rescanning[root] = left
            return
        del self._rescanning[root]
        try:
            self.rescan_state.emit(root, None, False)
        except RuntimeError:
            pass

    def _billed_ancestor_of(self, tile_idx: int) -> int | None:
        """The BASE-grid ancestor of a re-split quadrant, or None for a base
        tile. A depth-2 quadrant's direct parent is itself a quadrant, so the
        chain is walked to the root: the root is the tile whose charge the
        server can verify when deciding a quadrant rides that charge."""
        parent = self._parent_of.get(tile_idx)
        while parent is not None and self._parent_of.get(parent) is not None:
            parent = self._parent_of.get(parent)
        return parent

    def _drop_unsent_quadrants(self, pending: deque) -> int:
        """Take the not-yet-submitted free quadrants out of the submit deque.

        Only over ground NO finer read has landed on. Such a parent falls back
        cleanly to its own coarse read, which the terminal flush puts back
        (_flush_withheld). Ground that is already part re-read keeps its
        remaining quadrants: its coarse read is superseded for good, so
        dropping the rest would leave that part of the tile empty.

        The "already part re-read" test is on the item's DIRECT parent, the key
        the withhold bookkeeping uses (_settle_converted). On a depth-2 ladder
        the billed ancestor is the base tile, shared by every branch, so
        testing it lets one delivering quadrant keep every sibling's quadrants
        and the time cap stops bounding the free tail.

        Paid base tiles are never touched: they have no ancestor, and a run may
        not drop work the user was charged for.
        """
        if not pending:
            return 0
        kept: deque = deque()
        dropped = 0
        while pending:
            item = pending.popleft()
            if self._billed_ancestor_of(item[0]) is None:
                kept.append(item)  # paid base tile
                continue
            if self._parent_of.get(item[0]) in self._parents_with_child_results:
                kept.append(item)
                continue
            dropped += 1
            self._settle_rescanning(item[0])
        pending.extend(kept)
        if dropped:
            self._resplit_dropped += dropped
            logger.debug(
                "AutoDetectionWorker: re-split time budget spent, dropped %d "
                "unsent quadrant(s)", dropped)
        return dropped

    def _resplit_time_spent(self) -> bool:
        """True once the free re-split tail has used its share of the run.

        The share is a multiple of what the PAID grid took, so it scales with
        the zone instead of being a number that is generous on a small run and
        absurd on a big one. Before the paid grid is done there is no deadline:
        quadrants queued then are interleaved with tiles the user is paying for
        anyway."""
        deadline = self._resplit_deadline
        return bool(deadline) and time.monotonic() > deadline

    def _drain_subtiles(self, pending: deque) -> int:
        """Move queued quadrant specs into the submit deque as NEW tiles (fresh
        indices appended to self._tiles, so transforms/progress bookkeeping stay
        index-consistent) and return how many were added, or MINUS the number
        dropped once the tail's clock is spent (the caller adds the return to
        the run total either way). The quadrant renders
        at 2x its rect size (out_w/out_h), i.e. 2x finer ground resolution per
        re-split depth. Called from both run loops between cycles."""
        if self._resplit_time_spent():
            # Out of time: drop what is still queued rather than making the user
            # wait for it. Counted and logged, never silently.
            if self._pending_subtiles:
                self._resplit_dropped += len(self._pending_subtiles)
                logger.debug(
                    "AutoDetectionWorker: re-split time budget spent, dropped %d "
                    "queued quadrant(s)", len(self._pending_subtiles))
                # Owed before the drop, exactly as _drop_unsent_quadrants does
                # it: a quadrant that will never be sent still holds a slot in
                # its base tile's re-scan count, and that count is what keeps
                # the parent's objects withheld until the terminal flush.
                for _spec, _depth, parent_idx in self._pending_subtiles:
                    root = self._billed_ancestor_of(parent_idx)
                    self._settle_rescanning_root(
                        parent_idx if root is None else root)
                self._pending_subtiles.clear()
            # Draining alone does not bound the tail: quadrants queued WHILE the
            # paid grid ran are already in the submit deque before the clock
            # exists, and on a dense zone that is most of them. Drop those too,
            # and give the caller back a NEGATIVE count: the run total already
            # counted them, so the progress readout would otherwise wait on
            # tiles that will never answer.
            return -self._drop_unsent_quadrants(pending)
        added = 0
        while self._pending_subtiles:
            spec, depth, parent_idx = self._pending_subtiles.pop()
            idx = len(self._tiles)
            self._tiles.append(spec)
            self._tile_depth[idx] = depth
            self._parent_of[idx] = parent_idx
            _tx, _ty, tw, th = spec
            self._tile_outsize[idx] = (tw * 2, th * 2)
            pending.append((idx, spec))
            added += 1
        return added

    def _emit_completed(
        self,
        response: dict,
        tile_idx: int,
        tile_w: int,
        tile_h: int,
        tile_transform: dict,
    ) -> bool:
        """Decode a completed tile and emit its detections INLINE. True on success.

        The per-tile decode + geometry pipeline (RLE decode, clip
        intersection(), suppress_redundant_hypotheses) can throw on ONE
        malformed tile (a bad score/width, a GEOS/numpy fault). Guarded here so
        that bad tile becomes a skip (a warning, still counted done) exactly
        like a submit-side tile_fatal, instead of propagating to run()'s
        last-resort net and aborting the whole PAID run, which would lose every
        later tile and leave the failed tile mis-accounted. The tile is already
        billed server-side, so skipping only forfeits its geometry; the caller
        still counts it and advances progress.

        This is the synchronous composition of the three halves below, kept for
        the batched path and the tests. The streaming path calls them
        separately so the middle one runs off the loop that drives the sockets.
        """
        job = self._plan_completed(
            response, tile_idx, tile_w, tile_h, tile_transform)
        try:
            detections = self._convert_completed(job)
        except Exception as exc:  # noqa: BLE001 - one bad tile must never kill the run
            return self._settle_converted(False, job, exc)
        return self._settle_converted(True, job, detections)

    def _plan_completed(
        self,
        response: dict,
        tile_idx: int,
        tile_w: int,
        tile_h: int,
        tile_transform: dict,
    ) -> dict:
        """WORKER-THREAD half of a finished tile: everything that touches run
        state or the run's own tile queue, so the conversion that follows can
        run on any thread. Cheap: it reads mask COUNTS, never mask pixels.

        Returns the job the conversion needs. Never raises on a malformed
        payload: the validation stays inside the conversion, where the existing
        one-bad-tile guard already catches it.
        """
        from ..core.cloud_detection import detection_mask_count

        # Re-split quadrants render at 2x their grid rect (see _drain_subtiles),
        # so their real SENT image size lives in _tile_outsize, not the grid-rect
        # size the caller carries in tile_spec. Use it as the RLE decode fallback;
        # a base tile has no _tile_outsize entry, so the rect size is kept. Server
        # responses carry width/height, so the fallback only matters if one omits
        # them (mapping a mask at the wrong size shifts it toward its tile origin).
        tile_w, tile_h = self._tile_outsize.get(tile_idx, (tile_w, tile_h))
        # The tile got its answer, so it will never be re-submitted: release the
        # pre-stamp image kept for its submission.
        self._release_tile_clean_image(tile_idx)
        # Results are flowing again: clear any "in line / server busy" UI state.
        self._note_flowing()

        # The count is read from the entries without decoding any of them, so a
        # saturated tile never holds its whole mask set at once.
        decoded_count = detection_mask_count(response, self._score_threshold)
        # Paid-grid accounting: the free tail's clock starts the moment the grid
        # the user was charged for is fully answered.
        if self._tile_depth.get(tile_idx, 0) == 0:
            self._paid_tiles_done += 1
            paid_grid_done = not self._resplit_deadline and self._paid_tiles_done >= self._paid_tiles_total
            if paid_grid_done and self._resplit_time_ratio > 0:
                spent = max(0.0, time.monotonic() - self._run_started_at)
                self._resplit_deadline = (
                    time.monotonic() + spent * self._resplit_time_ratio)
        # Tile at (or brushing) the per-inference ceiling => the scene likely
        # had more objects than one inference can emit; flag it so the run end
        # can hint "raise Detail".
        resplit = False
        if decoded_count >= self._mask_cap_trigger:
            self._hit_mask_cap = True
            self.tiles_mask_capped += 1
            # Re-split ladder: with budget + depth headroom, queue this tile's
            # quadrants at 2x scale so the truncated objects get their own
            # inference slots. Only tiles that stay capped at the end of the
            # ladder count as residual truncation for the review hint. Decided
            # HERE because it queues quadrants into the run's tile deque, which
            # only the worker thread may touch.
            resplit = self._maybe_subdivide(tile_idx)
            if not resplit:
                self.tiles_capped_final += 1
        return {
            "response": response,
            "tile_idx": tile_idx,
            "tile_w": tile_w,
            "tile_h": tile_h,
            "transform": tile_transform,
            "count": decoded_count,
            "resplit": resplit,
        }

    def _convert_completed(self, job: dict) -> list:
        """ANY-THREAD half: masks -> ready geometry (WKB). Pure CPU.

        Runs on a converter thread during a streaming run, so it must not touch
        the run's queues, counters or Qt signals. The two run-wide accumulators
        it does feed are taken under a lock (see _detections_to_geoms), and the
        prepared clip engine it needs is built per thread (_clip_for_thread).

        Streaming decode: the iterator validates the payload and resolves the
        decode dimensions eagerly, but builds each mask only when it is asked
        for, so at most ONE full-tile grid per converter thread is alive.
        """
        from ..core.cloud_detection import iter_detection_masks

        mask_iter = iter_detection_masks(
            job["response"], job["tile_w"], job["tile_h"], self._score_threshold
        )
        return self._detections_to_geoms(
            self._iter_kept_masks(
                mask_iter, job["response"], job["tile_idx"],
                job["tile_w"], job["tile_h"], job["count"],
            ),
            job["transform"],
        )

    def _settle_converted(self, ok: bool, job: dict, payload) -> bool:
        """WORKER-THREAD half: fold one conversion back into the run and emit
        it. True when the tile's geometry was delivered.

        ``payload`` is the detection list when ``ok``, else the exception the
        conversion raised.
        """
        tile_idx = job["tile_idx"]
        # Owed before anything else: a quadrant that failed to convert is still
        # a quadrant that will never come, and the ground must not stay marked
        # for it.
        self._settle_rescanning(tile_idx)
        if not ok:
            logger.warning(
                "AutoDetectionWorker: tile %d decode/convert failed: %s",
                tile_idx, payload,
            )
            self.warning.emit(
                f"Tile {tile_idx}: could not process result; skipping"
            )
            return False

        detections = payload
        logger.debug(
            "AutoDetectionWorker: tile %d completed with %d detection(s)",
            tile_idx, len(detections),
        )
        if job["resplit"]:
            # This tile's quadrants will re-read the same ground 2x finer:
            # withhold its coarse detections so they never union-bridge the
            # quadrants' cleanly separated objects (flushed at the terminal
            # only if every quadrant fails; see _flush_withheld). The tile still
            # emits, with nothing in it, exactly as it did inline.
            self._withheld[tile_idx] = detections
            detections = []
        else:
            parent = self._parent_of.get(tile_idx)
            if parent is not None and detections:
                # A quadrant delivered: its parent's withheld coarse read is
                # permanently superseded.
                self._parents_with_child_results.add(parent)

        self.tile_completed.emit(tile_idx, detections)
        return True

    def _iter_kept_masks(
        self, mask_iter, response: dict, tile_idx: int, tile_w: int, tile_h: int,
        instance_count: int,
    ):
        """Yield this tile's (mask, score) one at a time, in the server's order,
        then the coverage-rescue mask when it applies.

        Same values and same order as the list this used to build; the point of
        the generator is that the consumer converts each mask to geometry and
        lets it go, so a saturated tile never holds every full-tile boolean grid
        at once.

        Composite-per-tile: drop any detection whose centroid lands on the
        stamped example region (the example itself, not a real object).
        stamp_norm is the REAL pasted rectangle ([nx0,ny0,nx1,ny1] normalized),
        along the top edge or, for the first grid row, the bottom edge. Prefer
        the server box ([cx,cy,w,h] normalized); fall back to the mask centroid
        so the example is still dropped even if the server omits a box.
        """
        stamp = self._tile_stamp_norm.get(tile_idx)
        for mask, score, box in mask_iter:
            if stamp and self._centroid_in_stamp(box, mask, stamp):
                continue
            yield (mask, score)

        # Coverage-map zero-instance rescue (policy-gated, map-like text prompts
        # only): when the per-instance pass returned nothing for this tile, a
        # coverage mask above the floor is converted like an instance mask so a
        # continuous feature is not left empty. A tile with any instance keeps
        # today's result untouched (the rescue no-ops on it).
        yield from self._semantic_rescue_masks(
            response, instance_count, tile_w, tile_h)

    def _semantic_rescue_masks(
        self, response: dict, instance_count: int, tile_w: int, tile_h: int,
    ) -> list:
        """Coverage-map zero-instance rescue for map-like prompts (policy-gated).

        When the run opted in and the per-instance pass returned nothing for
        this tile, decode the coverage mask (present and at or above the floor)
        like an instance mask and return it as a single (mask, score) with the
        coverage as the score, so the review confidence slider still filters it.
        Returns [] when the rescue was not requested, the fields are absent, or
        the coverage is below the floor (a missing field is never an error).
        """
        if not self._return_semantic:
            return []
        from ..core.cloud_detection import (
            decode_rle_to_mask,
            parse_semantic_fields,
            should_rescue_with_semantic,
        )

        rle, coverage, _presence = parse_semantic_fields(response)
        if not should_rescue_with_semantic(
            instance_count, coverage, rle is not None,
            self._return_semantic, self._semantic_coverage_floor,
        ):
            return []
        # Decode at the SAME server-reported dimensions decode_detection_response
        # uses, so the coverage mask maps back to ground on the same pixel grid
        # as the instance masks (a wrong size would shift it toward the origin).
        srv_w = response.get("width")
        srv_h = response.get("height")
        decode_w = int(srv_w) if srv_w is not None else tile_w
        decode_h = int(srv_h) if srv_h is not None else tile_h
        mask = decode_rle_to_mask(rle, decode_h, decode_w)
        if not mask.any():
            return []
        return [(mask, float(coverage))]

    def _make_clip_pair(self):
        """Build a (geometry, prepared engine) pair from the run's copied WKB.

        Returns (None, None) for the rectangle/MCP path where clip_polygon_wkb
        is None, matching the GUI's old behaviour.
        """
        if not self._clip_polygon_wkb:
            return None, None
        from qgis.core import QgsGeometry

        geom = QgsGeometry()
        geom.fromWkb(self._clip_polygon_wkb)
        if geom.isEmpty():
            return None, None
        try:
            engine = QgsGeometry.createGeometryEngine(geom.constGet())
            engine.prepareGeometry()
        except Exception:  # noqa: BLE001 - fall back to plain intersection()
            engine = None
        return geom, engine

    def _build_clip_engine(self) -> None:
        """Rebuild the worker thread's own zone clip geometry + prepared engine.

        Kept as the worker thread's entry point because _quad_intersects_zone
        reads _clip_geom directly on that thread.
        """
        self._clip_geom, self._clip_engine = self._make_clip_pair()

    def _clip_for_thread(self):
        """The calling thread's own (clip geometry, prepared engine).

        A prepared GEOS engine caches state inside the geometry instance it was
        built from, so one pair can never be shared by the converter threads and
        the worker thread at once. Each thread builds its own once and keeps it;
        the geometry is a zone polygon, so the duplication is a few kB per
        thread.
        """
        if not self._clip_polygon_wkb:
            return None, None
        pair = getattr(self._clip_local, "pair", None)
        if pair is None:
            pair = self._make_clip_pair()
            self._clip_local.pair = pair
        return pair

    def _detections_to_geoms(self, kept, tile_transform) -> list:
        """Turn (mask, score) detections into ready (geom_wkb: bytes, score) on
        the worker thread: refine -> polygonize -> clip-to-zone -> repair -> WKB.

        This is the verbatim per-detection pipeline that used to run on the GUI
        thread, minus the merge, which now runs on the live stitcher thread (see
        workers/live_stitch_thread.py). Every op here is value-class QgsGeometry
        or pure numpy/scipy, safe off the main thread; no QgsProject, no layer
        edits, no area measurement.
        """
        import numpy as np

        from ..core.cloud_detection import (
            mask_cell_size,
            pinhole_fill_limit_px,
            tile_simplify_tolerance,
        )
        from ..core.layer_conventions import repair_polygon, to_multipolygon
        from ..core.polygon_exporter import (
            fill_small_holes,
            masks_to_polygons_packed,
            suppress_redundant_hypotheses,
        )

        # Light sub-cell simplification trims the staircase off every mask as
        # it is built; the post-run refine simplifies further, so this never
        # costs final-shape fidelity. The tolerance (and the pinhole ceiling
        # below) is derived PER MASK from the grid the mask actually came back
        # on: the service may return masks coarser than the sent tile, and the
        # staircase step is the mask's own cell, not the native pixel (see
        # cloud_detection.tile_simplify_tolerance; unchanged when they match).
        # Anti-sliver floor: on uniform texture the cloud model returns sub-pixel noise
        # fragments (~0.1 m2) that clutter the output. Drop any detection whose
        # ground area is below a small square tied to pixel size. k=1.5 means a
        # detection smaller than ~1.5 px on a side is noise, not an object: at
        # gsd 0.4 m/px the floor is (1.5*0.4)^2 = 0.36 m2, which drops 0.1 m2
        # slivers while keeping a 2x2 m tree/car. gsd<=0 (no metric scale) =>
        # floor 0, no drop.
        min_keep_area = (
            max((self._min_keep_px * self._gsd) ** 2, self._min_keep_floor_m2)
            if self._gsd > 0 else 0.0
        )
        # Tile ground size, for the observed mask-resolution bookkeeping below.
        # tile_transform["bbox"] uses the polygon_exporter (minx, maxx, miny,
        # maxy) convention.
        bbox = tile_transform.get("bbox", (0.0, 1.0, 0.0, 1.0))
        ground_w = float(bbox[1] - bbox[0])
        ground_h = float(bbox[3] - bbox[2])
        # This runs on a converter thread during a streaming run, so the clip
        # pair is the CALLING thread's own (see _clip_for_thread) and the two
        # run-wide accumulators below are merged once, under a lock, instead of
        # being read-modify-written per mask from several threads at once.
        clip_geom, clip_engine = self._clip_for_thread()
        observed_cell = 0.0
        # Masks the whole-tile blob guard threw away, by which of its three
        # tests fired. Folded run-wide under _stat_lock beside
        # raw_detections_total, for the same reason: several converter threads
        # reach this and += would lose counts. The guard drops a mask the user
        # PAID for and left no trace at all, so a legitimate parcel that fills
        # its tile vanished into a tile-shaped hole with nothing to read.
        n_blob_hard = 0
        n_blob_span = 0
        n_blob_shape = 0
        n_blob_kept_map = 0
        n_blob_map_lowscore = 0
        # Scores of every whole-tile mask this tile produced in MAP mode, kept
        # or cut. The floor above can only be chosen from the spread.
        map_cover_scores: list[float] = []
        out = []
        # Prepared crops, in the order the masks arrived: (crop, (row0, col0)),
        # keyed by the polygonize parameters they share, plus a parallel list of
        # the per-mask facts the geometry tail needs. Polygonizing is batched
        # per key (see masks_to_polygons_packed): rasterio's shapes() costs a few
        # ms per CALL whatever it is handed, and a dense tile called it once per
        # mask. Only the crops are held, never the full-tile grids, so a
        # saturated tile still costs about one tile's worth of pixels.
        pending_crops: dict = {}
        pending_meta: dict = {}
        for mask, score in kept:
            # Crop the mask to the object's bounding box BEFORE the per-pixel work
            # (scipy fill-holes + rasterio polygonize). A dense run returns ~130
            # masks/tile, each a full HxW (1024^2) array with one small object;
            # scanning the whole grid per mask was ~17s/batch of idle time.
            # Cropping makes both ops proportional to the object, not the tile, and
            # mask_to_polygons offsets the geo-transform by (col0,row0) so the
            # output stays pixel-exact. full_shape keeps the px->ground scale.
            full_h, full_w = mask.shape
            # The mask's OWN ground cell (the polygon staircase step): tile
            # ground size / returned mask size. Recorded run-wide (max, so
            # partial boundary tiles with a finer ratio never understate it)
            # for the review's px<->ground refine, and used per mask below to
            # key the staircase simplify + pinhole fill to the grid this
            # polygon is actually built on.
            cell = mask_cell_size(ground_w, ground_h, full_w, full_h)
            if cell > observed_cell:
                observed_cell = cell
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            row0, col0 = int(ys.min()), int(xs.min())
            row1, col1 = int(ys.max()), int(xs.max())
            # whole-tile "everything" masks (near-whole-tile blobs on uniform texture)
            # must not reach the merger in SEPARATE/count mode. But coverage
            # alone cannot tell a texture blob from a REAL large building that
            # fills the tile, so >55% only ARMS a compactness check on the
            # resulting geometry (below): a solid rectangular object survives,
            # an irregular whole-tile blob is dropped. Above the hard cap the
            # mask is a fill-everything failure regardless of shape. Skipped in
            # MAP mode so a real whole-tile lake/field always survives.
            coverage = ys.size / float(full_h * full_w)
            blob_check = False
            # Raw-collect mode keeps every fragment (gates OFF): the client
            # applies them later if it re-merges as SEPARATE.
            if self._merge_separate and not self._collect_raw and coverage > self._max_tile_coverage:
                if coverage > self._hard_tile_coverage:
                    n_blob_hard += 1
                    continue
                # A mask the TILE bounds is not an object: the grid drew its
                # outline, and in count mode it has no edge of its own to be
                # counted by. The compactness check below cannot see this,
                # because such a mask fills its oriented box perfectly - it IS
                # the solid rectangle that check exists to keep. So the span
                # test runs first, and the rescue never gets to look at it.
                span = self._tile_span_fraction
                if (col1 - col0 + 1 >= span * full_w and row1 - row0 + 1 >= span * full_h):
                    n_blob_span += 1
                    continue
                blob_check = True
            elif coverage > self._max_tile_coverage and (
                    self._map_cover_score_floor > 0.0
                    and float(score) < self._map_cover_score_floor):
                # MAP mode, the one cut that survives the "it might be a real
                # lake" objection: the bigger the claim, the more confident it
                # has to be. A mask over most of a tile is the largest claim the
                # model can make there, and when it is also among the least
                # confident it is the fill-everything answer, not the lake.
                # Nothing about this is class-specific, so it holds for every
                # continuous prompt (forest, water, road, grass) at once.
                # OFF at 0.0, which is the behaviour that shipped.
                n_blob_map_lowscore += 1
                map_cover_scores.append(float(score))
                continue
            elif coverage > self._max_tile_coverage:
                # MAP mode, where the guard above is off on purpose so a real
                # whole-tile lake or field survives. COUNT ONLY, nothing is
                # dropped: the same mask that is a genuine lake here is also how
                # a "fill everything" answer looks, and in MAP the merger then
                # unions it with the correct outlines from its neighbours, so
                # one bad tile can swallow a whole run into a single object that
                # no review filter can take apart. Whether to arm anything here
                # is a behaviour change on a paid path, and this number is what
                # it should be decided on. Their SCORES ride along: picking the
                # floor above needs the spread, not the count.
                n_blob_kept_map += 1
                map_cover_scores.append(float(score))
            sub = mask[row0:row1 + 1, col0:col1 + 1]
            # Pad 1px of background on every side BEFORE fill_holes. binary_fill_holes
            # floods inward from the array border: an object touching the crop edge
            # (always true - the crop IS its bbox) would let a concavity that opens
            # onto that edge connect to the border and fill differently than in the
            # full tile. The 1px background margin restores the full-array result
            # exactly. The offset shifts by 1 to keep the geo-reference pixel-exact.
            sub = np.pad(sub, 1, constant_values=False)
            # Fill interior PINHOLES only: mask staircase and compression
            # noise, plus rooftop texture pits. Real interior holes
            # (courtyards, ring roads, islands) are kept so the review's "Fill
            # holes" toggle stays meaningful; an unconditional fill here would
            # export every courtyard building as a solid block and make that
            # toggle a no-op. The ground ceiling converts to pixels of the
            # MASK's own grid (see pinhole_fill_limit_px) so a coarser
            # returned grid keeps the same ground meaning instead of silently
            # doubling it.
            sub = fill_small_holes(
                sub, pinhole_fill_limit_px(self._gsd, cell, self._pinhole_m))
            # Queue instead of polygonizing now. The key is everything the
            # polygonizer needs to be identical for two masks to share a call:
            # the grid they were returned on and the staircase tolerance of that
            # grid. In practice one tile yields one key.
            key = (
                (full_h, full_w),
                tile_simplify_tolerance(
                    self._gsd, cell, self._tile_simplify_mult),
            )
            pending_crops.setdefault(key, []).append((sub, (row0 - 1, col0 - 1)))
            pending_meta.setdefault(key, []).append((float(score), blob_check))

        for key, crops in pending_crops.items():
            full_shape, simplify_tolerance = key
            polygon_lists = masks_to_polygons_packed(
                crops, tile_transform, full_shape,
                simplify_tolerance=simplify_tolerance,
            )
            for (score, blob_check), geoms in zip(pending_meta[key], polygon_lists):
                for geom in geoms:
                    if geom is None or geom.isEmpty():
                        continue
                    # Confine results to the DRAWN polygon: a boundary tile is
                    # rectangular and overflows the shape. A prepared-engine
                    # contains() skips the clip for the interior majority; only
                    # boundary-crossing detections pay for intersection().
                    if clip_geom is not None:
                        inside = False
                        if clip_engine is not None:
                            try:
                                inside = clip_engine.contains(geom.constGet())
                            except Exception:  # noqa: BLE001 - fall back to clip
                                inside = False
                        if not inside:
                            geom = geom.intersection(clip_geom)
                        if geom is None or geom.isEmpty() or geom.area() <= 0:
                            continue
                    # Coerce to a polygon-only MultiPolygon at the SOURCE: the
                    # clip intersection can yield a GeometryCollection that a
                    # MultiPolygon layer would later reject.
                    geom = to_multipolygon(repair_polygon(geom) or geom)
                    if geom is None or geom.isEmpty():
                        continue
                    # Drop sub-pixel noise slivers (computed once above). Placed
                    # AFTER the clip + repair so a detection trimmed to a tiny
                    # boundary sliver is also dropped, not just intrinsically
                    # tiny ones.
                    if min_keep_area > 0.0 and geom.area() < min_keep_area:
                        continue
                    # Armed by the >55% coverage gate above: keep a compact
                    # (solid, rectangular-ish) large object, drop an irregular
                    # texture blob.
                    if blob_check and not self._is_compact_shape(
                            geom, self._compact_min_fill):
                        n_blob_shape += 1
                        continue
                    out.append((geom, score))
        # Raw (pre-NMS/pre-merge) detection count, for the run-summary log: the
        # MAP pre-merge below shrinks what the GUI receives, so the GUI-side
        # fold counter alone would under-report the model's raw output. Merged
        # with the run's coarsest observed mask cell under one lock: several
        # converter threads reach both, and `+=` is a read-modify-write that
        # loses counts when two land together.
        with self._stat_lock:
            self.raw_detections_total += len(out)
            self.masks_dropped_whole_tile += (
                n_blob_hard + n_blob_span + n_blob_shape)
            self.masks_dropped_hard_cover += n_blob_hard
            self.masks_dropped_tile_span += n_blob_span
            self.masks_dropped_not_compact += n_blob_shape
            self.masks_whole_tile_kept_map += n_blob_kept_map
            self.masks_dropped_map_lowscore += n_blob_map_lowscore
            self.map_cover_scores.extend(map_cover_scores)
            if observed_cell > self.observed_mask_gsd:
                self.observed_mask_gsd = observed_cell
        # SEPARATE/count mode: resolve the model's overlapping same-region
        # hypotheses (whole-complex vs per-roof vs roof-section) by SELECTION
        # before the merger ever sees them. Without this, the merger's IoS
        # dedup UNIONS the granularities into one mega-object that inherits the
        # max constituent score, so a low-score complex-wide mask (shadow
        # fringe included) surfaces at its best roof's confidence. MAP mode is
        # left untouched: coverage there is the union of hypotheses by design.
        # The same NMS, in MAP mode, when the server switch is on. OFF ships the
        # behaviour above unchanged. It is the one test that separates the two
        # readings of a whole-tile mask WITHOUT a threshold: a tile that returned
        # precise outlines AND a mask covering the same ground is a hypothesis
        # stack, and NMS picks one; a tile that is genuinely all lake returns the
        # big mask ALONE, so there is nothing to suppress and it survives. A
        # coverage cut cannot tell those apart, and re-arming one here would
        # punch a hole in the middle of every real lake, since the neighbours
        # only reach one overlap band into a dropped tile.
        if self._merge_separate or self._collect_raw or self._map_hypothesis_nms:
            # NMS also runs in raw-collect mode: it resolves the model's
            # overlapping same-region hypothesis stacks (not distinct instances),
            # so the fragments the client keeps are already free of that
            # redundancy while still un-gated and un-premerged.
            ms = self._merge_scalars
            sup_kwargs = {k: ms[k] for k in (
                "ios_threshold", "dup_ios_floor", "dup_centroid_frac") if k in ms}
            out = suppress_redundant_hypotheses(out, **sup_kwargs)
        if not (self._merge_separate or self._collect_raw):
            # MAP/continuous mode has no per-tile hypothesis NMS (coverage is
            # the union of hypotheses by design), so a dense continuous prompt
            # (roads, forest) ships hundreds of raw overlapping fragments per
            # tile. Folding those one-by-one into the GUI merger's ever-growing
            # keepers is quadratic and freezes QGIS at end-of-run on dense
            # continuous runs. Pre-stitch THIS tile's fragments here on the
            # worker thread instead: union is commutative, so the GUI merger
            # still produces the same final objects from a handful of
            # per-tile keepers.
            out = self._premerge_map_fragments(out)
        return [(bytes(geom.asWkb()), score) for geom, score in out]

    def _premerge_map_fragments(self, out: list) -> list:
        """Fold one tile's MAP-mode fragments into a LOCAL IncrementalMerger
        carrying the run merger's exact policy knobs (plain-union mode, the
        size-aware seam gate, the run gsd) and return its (geom, score)
        keepers. Runs on the worker thread: geometries are value-class and
        tile-bounded, so the union cost stays small and overlaps network I/O.
        Score semantics are unchanged (a keeper carries the MAX of its
        fragments, exactly what the GUI merger computes when folding the raw
        fragments itself)."""
        if len(out) < 2:
            return out
        from ..core.polygon_exporter import IncrementalMerger

        ms = self._merge_scalars
        merge_kwargs = {k: ms[k] for k in (
            "merge_ios", "dedup_ios", "dup_ios_floor", "dup_centroid_frac",
            "seam_span_ios", "seam_span_tol", "jitter_area_frac",
            "score_floor_frac") if k in ms}
        merger = IncrementalMerger(
            seam_min_dim=self._seam_min_dim,
            select_duplicates=False,
            gsd=self._gsd,
            **merge_kwargs,
        )
        for geom, score in out:
            merger.add(geom, float(score))
        return merger.result_scored()

    @staticmethod
    def _is_compact_shape(geom, min_fill: float = _COMPACT_MIN_FILL) -> bool:
        """True if geom fills at least ``min_fill`` of its oriented minimum
        bounding box.

        Used to rescue a REAL large object (warehouse, big roof) from the
        whole-tile "everything"-blob drop: man-made large objects are solid and
        near-rectangular, texture blobs (canopy, bare soil) are ragged. Any
        failure counts as not-compact, which restores the old drop behaviour.
        """
        try:
            _obb, obb_area, _angle, _w, _h = geom.orientedMinimumBoundingBox()
            if obb_area and obb_area > 0.0:
                return geom.area() / obb_area >= min_fill
        except Exception:  # noqa: BLE001 -- best-effort rescue, never fatal  # nosec B110
            pass
        return False

    @staticmethod
    def _centroid_in_stamp(box, mask, stamp) -> bool:
        """True if a detection's normalized centroid lies in the stamp rectangle
        ``[nx0, ny0, nx1, ny1]`` (the real pasted region, top OR bottom edge).
        Prefers the server box ([cx, cy, w, h] normalized) when present and
        non-degenerate, else falls back to the mask's pixel centroid so the
        stamped example is dropped even when the server omits a box."""
        nx0, ny0, nx1, ny1 = stamp[0], stamp[1], stamp[2], stamp[3]
        if box and len(box) == 4 and (box[2] > 0 or box[3] > 0):
            cx, cy = box[0], box[1]
            return nx0 <= cx <= nx1 and ny0 <= cy <= ny1
        try:
            import numpy as np
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                return False
            h = max(1, mask.shape[0])
            w = max(1, mask.shape[1])
            cx = float(xs.mean()) / w
            cy = float(ys.mean()) / h
            return nx0 <= cx <= nx1 and ny0 <= cy <= ny1
        except Exception:  # noqa: BLE001 -- best-effort filter, never fatal
            return False
