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
- All QgsGeometry / QGIS API calls happen in slots on the main thread;
  the worker only touches numpy arrays and plain Python objects.
- Never use ThreadPoolExecutor inside this QThread.

HTTP stack: everything goes through the QGIS network layer (TerraLabClient's
QgsBlockingNetworkRequest paths, plus QgsNetworkAccessManager for the
streaming path), so proxy/TLS settings configured in QGIS are inherited.
No raw requests/urllib transport in this module.
"""
from __future__ import annotations

import logging
import re
import random
import time
import uuid
from collections import deque

from qgis.PyQt.QtCore import (
    QMutex, QObject, QThread, QWaitCondition, pyqtSignal, pyqtSlot,
)

from .adaptive_concurrency import AdaptiveConcurrency, OfflineFastFail

logger = logging.getLogger(__name__)

# Transient error codes that trigger a per-tile retry (bounded by the same
# _MAX_RATE_LIMIT_RETRIES=8 ceiling as rate-limit retries; see _run_streaming /
# _run_batched, where both requeue a "retry" until n exceeds that bound).
_TRANSIENT_CODES = {
    "NO_INTERNET", "TIMEOUT", "DNS_ERROR", "PROXY_ERROR",
    "SSL_ERROR", "SERVER_ERROR", "CONNECTION_REFUSED",
    # A submit whose deadline passed with no HTTP answer at all: the backend
    # instance is cold-starting or busy. Retried on the queue TIME budget (not
    # an attempt count) and shown as the "your spot is held" waiting room,
    # never blamed on the user's connection. See terralab_client._classify_qt_error.
    "SERVICE_WARMING",
}

# Sentinel codes for exhausted credits: a clean end-of-run, not a tile failure.
# QUOTA_EXCEEDED is the Pro-tier exhaustion code (monthly cap reached mid-run);
# treating it as exhausted stops the run gracefully instead of as a fatal error.
_EXHAUSTED_CODES = {"CREDITS_EXHAUSTED", "FREE_DETECTIONS_EXHAUSTED", "QUOTA_EXCEEDED"}

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
# AHEAD of need (async jobs, overlapped with the in-flight inference). Kept
# small so the pipeline stays fed without stacking main-thread render jobs or
# holding more than a few tile images alive.
_PREFETCH_DEPTH = 2
# On a USER cancel we stop firing new tiles at once, then wait this long for
# the handful ALREADY in flight to land so their billed masks are kept, not
# thrown away. Bounded so one hung reply can never hold the stop open: past it
# the stragglers are aborted (and refunded server side as non-completed). The
# in-flight set is <= max_concurrent and each direct tile is ~1s, so a real
# cancel drains in well under this ceiling; kept short so Cancel feels prompt
# (a tile still computing past it is aborted rather than making the user wait).
_STOP_DRAIN_BUDGET_S = 2.5
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
_MASK_CAP_TRIGGER = int(_MASK_CAP_TRIGGER_FRAC * _MAX_MASKS_PER_TILE)
# Saturated-tile re-split recursion ceiling. Depth 1 quarters the object count
# per inference; depth 2 covers extreme dense scenes. Past that the quadrants
# are too small/interpolated to add signal.
_SUBDIV_MAX_DEPTH = 2
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

# Sentinel "fatal" code raised by the offline fast-fail so the terminal error
# carries a clear connectivity message instead of a raw code (see
# _submit_error_message and adaptive_concurrency.OfflineFastFail).
_OFFLINE_STOP_CODE = "NO_CONNECTION"

# Run-level fatal codes: stop the whole run at once (an auth or subscription
# problem fails every tile identically, retrying per tile only burns time).
# Any OTHER non-retryable code is a PER-TILE rejection: that one tile is
# skipped and the run continues, because one bad tile (a 4xx for its image,
# a code this client version does not know) must never kill a paid
# multi-tile run. A streak of consecutive per-tile rejections with no
# success in between still aborts the run, so a NEW run-level code the
# server introduces later costs at most a handful of requests.
_RUN_FATAL_CODES = {"AUTH_ERROR", "INVALID_KEY", "SUBSCRIPTION_INACTIVE"}
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

# Mid-run offline abort threshold. After the first successful tile the
# offline fast-fail no longer trips at its small pre-success threshold; a
# brief wifi blip must be absorbed by the normal retry backoff. But a link
# that stays dead used to grind every remaining tile's full retry budget
# (minutes of "Detecting...") before the run gave up. This streak of
# CONSECUTIVE hard-connectivity failures (roughly 1-2 minutes of continuous
# outage across the in-flight window) ends the run instead; the billed
# partials are salvaged into the review either way.
_MIDRUN_OFFLINE_STREAK = 30


def _as_int(value, default: int = -1) -> int:
    """Lenient int coercion for optional server-sent queue fields."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sanitize_filename_part(text: str, max_len: int = 40) -> str:
    """Remove characters unsafe for filenames and truncate."""
    safe = re.sub(r"[^\w\- ]", "", text).strip()
    safe = safe.replace(" ", "_")
    return safe[:max_len] if safe else "detection"


class TileRenderBridge(QObject):
    """Main-thread render bridge for the per-tile JIT path.

    Lives on (and is moved to) the GUI thread. QgsMapRendererParallelJob is
    GUI-thread only, so the worker thread can never render directly; instead it
    asks this bridge to render a tile via a QueuedConnection signal and blocks on
    a QWaitCondition until the bridge stores the result and wakes it.

    The bridge owns the layer + global geo_transform: given a tile rect
    (tx, ty, tw, th) it computes that tile's bbox_native from the SAME global
    geo_transform the worker uses for georeferencing, then renders ONLY that
    sub-extent to a tw x th QImage. The render thus matches slicing one big zone
    render exactly (same CRS, same map units per pixel, same tile origin).
    """

    _render_requested = pyqtSignal(int, int, int, int, int, int, int)

    def __init__(self, layer, geo_transform: dict, parent=None):
        super().__init__(parent)
        self._layer = layer
        self._geo_transform = geo_transform
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        # request_seq -> QImage|None once the render is done; None means not ready.
        self._results: dict[int, object] = {}
        self._done: set[int] = set()
        self._seq = 0
        self._cancelled = False
        # The bridge's slot must run on the bridge's (main) thread even when the
        # signal is emitted from the worker thread: a queued connection marshals
        # the call onto the bridge's event loop. AutoConnection already does this
        # across threads, but we pin QueuedConnection to be explicit.
        from qgis.PyQt.QtCore import Qt as _Qt
        self._render_requested.connect(
            self._on_render_requested, _Qt.ConnectionType.QueuedConnection)

    def _tile_extent(self, tx: int, ty: int, tw: int, th: int):
        """Build the tile's bbox_native as a QgsRectangle in the layer CRS,
        from the global geo_transform. Identical math to
        AutoDetectionWorker._make_tile_transform (bbox_native), so the two never
        diverge."""
        from qgis.core import QgsRectangle

        src_bbox = self._geo_transform.get("bbox", (0.0, 0.0, 1.0, 1.0))
        img_shape = self._geo_transform.get("img_shape", (1, 1))
        img_h, img_w = max(img_shape[0], 1), max(img_shape[1], 1)
        src_minx, src_miny, src_maxx, src_maxy = src_bbox
        px_w = (src_maxx - src_minx) / img_w
        px_h = (src_maxy - src_miny) / img_h
        tile_minx = src_minx + tx * px_w
        tile_maxx = src_minx + (tx + tw) * px_w
        tile_miny = src_maxy - (ty + th) * px_h
        tile_maxy = src_maxy - ty * px_h
        return QgsRectangle(tile_minx, tile_miny, tile_maxx, tile_maxy)

    @pyqtSlot(int, int, int, int, int, int, int)
    def _on_render_requested(
        self, seq: int, tx: int, ty: int, tw: int, th: int,
        out_w: int, out_h: int,
    ) -> None:
        """Start this tile's render as an ASYNC job on the main thread.

        The slot returns immediately; the job's completion stores the QImage
        and wakes the worker. Because the slot never blocks, several queued
        requests start OVERLAPPING render jobs - that concurrency is the whole
        point of the worker-side prefetch (the serialized per-tile render is
        the bottleneck of large runs, not the network or the detection host).
        (out_w, out_h) is the OUTPUT pixel size; when it differs from (tw, th)
        the same ground extent renders at a finer scale (the saturated-tile
        re-split path). 0 means "same as the rect"."""
        from ..core.cloud_detection import start_tile_render_job

        started = False
        try:
            extent = self._tile_extent(tx, ty, tw, th)
            started = start_tile_render_job(
                self._layer, extent, out_w or tw, out_h or th,
                lambda img, s=seq: self._store_result(s, img),
            )
        except Exception as exc:  # noqa: BLE001 - never break the handshake
            logger.warning("TileRenderBridge: render failed at (%d,%d): %s", tx, ty, exc)
            started = False
        if not started:
            self._store_result(seq, None)

    def _store_result(self, seq: int, img) -> None:
        """Store one render's outcome and wake every waiting collector."""
        self._mutex.lock()
        try:
            self._results[seq] = img
            self._done.add(seq)
            self._cond.wakeAll()
        finally:
            self._mutex.unlock()

    def cancel(self) -> None:
        """Called FROM THE WORKER THREAD (via request_stop). Mark the bridge
        cancelled and wake any render_tile blocked on the condition AT ONCE, so a
        stop never waits out the condition timeout. This is the deadlock guard:
        on unload the main thread blocks in worker.wait(); if the worker were
        parked in render_tile waiting for a main-thread render that can no longer
        run, only this immediate wake lets it exit so wait() returns cleanly."""
        self._mutex.lock()
        try:
            self._cancelled = True
            self._cond.wakeAll()
        finally:
            self._mutex.unlock()

    def request_render(self, tx: int, ty: int, tw: int, th: int,
                       out_w: int = 0, out_h: int = 0) -> int | None:
        """Called FROM THE WORKER THREAD. Post a render request WITHOUT
        blocking and return its collect token (or None when cancelled). The
        render runs as an async job on the main thread, so several requested
        tiles render side by side while the worker keeps driving the network
        window - this is the prefetch half of the pipeline."""
        self._mutex.lock()
        try:
            if self._cancelled:
                return None
            seq = self._seq
            self._seq += 1
        finally:
            self._mutex.unlock()
        # Emit OUTSIDE the lock so the queued slot can run on the main thread.
        self._render_requested.emit(seq, tx, ty, tw, th, out_w, out_h)
        return seq

    def collect_render(self, seq: int):
        """Called FROM THE WORKER THREAD. Block until the requested render is
        done and return its QImage (or None). Returns at once when the result
        already landed, and unblocks the moment cancel() is called even if the
        render never ran (the unload deadlock guard)."""
        self._mutex.lock()
        try:
            while seq not in self._done and not self._cancelled:
                # Block until the main thread stores this render OR cancel()
                # wakes us. The timeout is only a safety net (a render job
                # itself times out at 60s); cancel() wakes immediately.
                self._cond.wait(self._mutex, 30000)
            img = self._results.pop(seq, None)
            self._done.discard(seq)
            return img
        finally:
            self._mutex.unlock()

    def render_tile(self, tx: int, ty: int, tw: int, th: int,
                    out_w: int = 0, out_h: int = 0):
        """Called FROM THE WORKER THREAD. Request a main-thread render of this
        tile and block until it is done, returning the QImage (or None).
        Composition of request_render + collect_render, kept as the simple
        synchronous entry point (non-prefetched tiles, batched path, tests)."""
        seq = self.request_render(tx, ty, tw, th, out_w, out_h)
        if seq is None:
            return None
        return self.collect_render(seq)


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
    """

    tile_completed = pyqtSignal(int, list)
    all_tiles_finished = pyqtSignal(list)
    progress = pyqtSignal(int, int)
    warning = pyqtSignal(str)
    error = pyqtSignal(str)
    credits_exhausted = pyqtSignal(int)
    cancelled = pyqtSignal()
    # Server-load feedback: (queue_position, queue_depth, eta_seconds).
    # position >= 1 -> a queue-aware server told us our place in line;
    # position == -1 -> server busy but no position known (old server / platform
    # 429 / cold start); (0, 0, 0) -> flowing again, clear any busy UI.
    queue_state = pyqtSignal(int, int, int)

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
        tile_renderer=None,
        parent=None,
    ):
        """Initialise the worker.

        Args:
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
                             Drives the per-tile simplify tolerance (0.75 * gsd).
            parent:          Optional Qt parent.
        """
        super().__init__(parent)

        # Per-tile JIT render source (main-thread bridge callback): _encode_tile
        # renders each tile on demand. Always set on every live path.
        self._tile_renderer = tile_renderer
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
        # Inter-tile overlap span (ground units), the run merger's size-aware
        # anti-over-merge gate. Used by the MAP-mode per-tile pre-merge below
        # so its local merger applies the exact same policy as the GUI's.
        self._seam_min_dim = seam_min_dim
        # Merge/dedup scalars resolved by the plugin (server policy or generic
        # fallback). Passed to the per-tile merge helpers below; absent keys fall
        # through to those callees' own signature defaults.
        self._merge_scalars = merge_scalars if isinstance(merge_scalars, dict) else {}
        self._clip_geom = None
        self._clip_engine = None
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
        self._mask_cap_trigger = int(
            _dp.mask_cap_trigger_frac(_MASK_CAP_TRIGGER_FRAC)
            * _MAX_MASKS_PER_TILE)
        self._subdiv_max_depth = _dp.subdiv_max_depth(_SUBDIV_MAX_DEPTH)
        self._max_tile_coverage = _dp.max_tile_coverage(_MAX_TILE_COVERAGE)
        self._hard_tile_coverage = _dp.hard_tile_coverage(_HARD_TILE_COVERAGE)
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
        self.tiles_subdivided = 0                  # parents re-split (plain int)
        # Tiles still at the ceiling AFTER the re-split ladder (depth/budget
        # exhausted, or re-split disabled): the residual truncation the review
        # dense hint reports. Plain ints, GIL-safe.
        self.tiles_capped_final = 0

        # AIMD in-flight width controller: opens narrow (_AIMD_START), grows per
        # clean cycle up to max_concurrent, halves on a timeout/latency setback.
        # Drives effective_cap (_run_batched) and the window (_run_streaming).
        self._aimd = AdaptiveConcurrency(
            start=_AIMD_START, minimum=_AIMD_MIN, maximum=self._max_concurrent,
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
        # Tiles that reached server status "completed" (zero-mask included,
        # they consume a credit). Failed/timed-out tiles are refunded server
        # side and excluded, so this is the billable tile count.
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
        # Tiles whose render returned nothing (provider error, WMS/WMTS timeout,
        # coverage hole): also never submitted or billed. Surfaced at run end as
        # "N tiles could not be loaded" so a slow-server run's blank regions are
        # not a silent coverage gap.
        self.tiles_render_failed = 0
        self._completed_idx: set[int] = set()
        # Set True if any tile came back at the per-inference ceiling
        # (_MAX_MASKS_PER_TILE masks): the model emits at most 200 object
        # queries per forward pass, so a tile at exactly that count was likely
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

    def remaining_tiles(self) -> list[tuple[int, int, int, int]]:
        """Input tile rects not billed as completed, in original order.

        Read after the run stops to resume a cancelled or credits-exhausted
        run: failed/timed-out/skipped tiles are refunded server side, so
        every non-completed tile is safe to resubmit under the same run id.
        Returns (x, y, w, h) specs; the resumed worker re-encodes them lazily
        off the GUI thread, same as the original run.
        """
        return [t for i, t in enumerate(self._tiles) if i not in self._completed_idx]

    def _emit_progress(self, completed: int, total: int) -> None:
        """Emit progress shifted by the resume offset (a resumed run keeps
        counting from where the original run stopped)."""
        shown_total = self._progress_total if self._progress_total else total
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
        from ..core.power_inhibit import begin_activity, end_activity

        activity = begin_activity("AI Segmentation cloud detection")
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
                from ..core.telemetry import report_exception
                report_exception(exc, stage="segment", module="auto_detection_worker")
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self.error.emit("Detection stopped unexpectedly: {msg}".format(msg=exc))
            except Exception:  # noqa: BLE001 - signal delivery must never re-raise here  # nosec B110
                pass
        finally:
            end_activity(activity)
            client = getattr(self, "_client", None)
            if client is not None:
                try:
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
        total = len(self._tiles)

        if total == 0:
            self.all_tiles_finished.emit([])
            return

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
            self._run_streaming(total)
        else:
            self._run_batched(total)

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
        finished normally."""
        self._flush_withheld()
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
        if code == _OFFLINE_STOP_CODE:
            return "No internet connection. Check your connection and try again."
        return "Tile submit failed: {code}".format(code=code)

    def _run_batched(self, total: int) -> None:
        # Process tiles in bounded concurrent batches.
        # We maintain a queue of tiles and a dict of in-flight requests.
        # "Concurrent" here means overlapping submit+poll cycles, all on
        # this single thread via a cooperative polling loop.
        # (tile_idx, (x, y, w, h)) -- encoded lazily when its slot is filled.
        pending: deque = deque(enumerate(self._tiles))
        # Tiles to RE-submit (already encoded) after a rate-limit/transient error,
        # so a retry never re-encodes. submit_attempts bounds the retries per tile.
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
        # (see _RUN_FATAL_CODES / _MAX_CONSECUTIVE_TILE_FATALS).
        fatal_streak = 0

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
                if resubmit:
                    batch.append(resubmit.popleft())  # already encoded
                    continue
                tile_idx, spec = pending.popleft()
                # Encode off the GUI thread, just before submit. A blank/failed
                # render is deferred for a re-render (the worker keeps it);
                # only a permanent skip counts toward progress here.
                status, payload = self._encode_or_defer(tile_idx, spec)
                if status == "defer":
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
                            "Tile {idx}: rejected ({code}); skipping".format(
                                idx=tile_idx, code=bad_code))
                        completed += 1
                        self._emit_progress(completed, total)
                        if batch_stop is None and fatal_streak >= _MAX_CONSECUTIVE_TILE_FATALS:
                            batch_stop = ("fatal", bad_code)
                    elif kind == "retry":
                        # Requeue (no inline sleep) and let one coalesced back-off
                        # pace the next cycle. Queue-busy retries burn a TIME
                        # budget (waiting in line is expected under launch load);
                        # transient network errors keep the attempt ceiling.
                        delay, is_busy = outcome[1], outcome[2]
                        retry_code = outcome[3] if len(outcome) > 3 else ""
                        now = time.monotonic()
                        if is_busy:
                            first = busy_since.setdefault(tile_idx, now)
                            give_up = (now - first) > _QUEUE_RETRY_BUDGET_S
                            delay = max(1.0, delay) * random.uniform(*_BUSY_JITTER)  # nosec B311 - jitter, not crypto
                            # A busy/queue answer means the server was reached, so
                            # it is not an offline run and not a link setback.
                            self._fastfail.reset()
                        else:
                            n = submit_attempts.get(tile_idx, 0) + 1
                            submit_attempts[tile_idx] = n
                            give_up = n > _MAX_RATE_LIMIT_RETRIES
                            # Exponential-ish with jitter so transient blips
                            # don't retry in synchronized waves.
                            delay = min(30.0, delay * (2 ** min(n - 1, 4)))
                            delay *= random.uniform(0.5, 1.0)  # nosec B311 - jitter, not crypto
                            # Transient network error: an AIMD setback, and a
                            # hard-connectivity code advances the offline
                            # fast-fail counter (pre-first-success it trips at
                            # the small default threshold; mid-run only at the
                            # much larger _MIDRUN_OFFLINE_STREAK).
                            cycle_setback = True
                            self._fastfail.record(retry_code)
                        if give_up:
                            self.warning.emit(
                                "Tile {idx}: submit retries exhausted; skipping".format(
                                    idx=tile_idx
                                )
                            )
                            completed += 1
                            self._emit_progress(completed, total)
                        else:
                            resubmit.append((tile_idx, tile_spec, png_bytes))
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

            # Offline fast-fail: a run that only ever sees hard-connectivity
            # errors (DNS / connection refused / proxy) is offline. Before the
            # first success it aborts within a few failures; after it, only a
            # long unbroken streak (a link that stays dead, not a blip) ends
            # the run, and the billed partials are salvaged into the review.
            offline_streak = self.tiles_succeeded == 0 or self._fastfail.streak >= _MIDRUN_OFFLINE_STREAK
            if batch_stop is None and self._fastfail.tripped and offline_streak:
                batch_stop = ("fatal", _OFFLINE_STOP_CODE)

            # Process the rest of the batch (so already-charged "ok" tiles are kept)
            # before halting. Mirrors the old behaviour: in-flight tiles are not
            # drained after a terminal stop; the plugin finalizes partial results.
            if batch_stop is not None:
                # Flush withheld re-split parents BEFORE the terminal signal:
                # the main thread finalizes (nulls the merger) on that signal,
                # so a parent's late tile_completed would race the finalize and
                # its billed detections could be dropped. _flush_withheld clears
                # _withheld, so the later call in _emit_terminal is a no-op.
                self._flush_withheld()
                if batch_stop[0] == "exhausted":
                    self.credits_exhausted.emit(batch_stop[1])
                    self._stop_reason = "exhausted"
                else:
                    self.error.emit(self._submit_error_message(batch_stop[1]))
                    self._stop_reason = "error"
                self._stop_requested = True

            if self._stop_requested:
                break

            # Nothing in flight to poll and we only have re-submits waiting: back
            # off once (rate-limit retry_after) before looping to re-send them.
            if submit_backoff is not None and not in_flight:
                self._settle_concurrency(cycle_setback, cycle_progress)
                self._interruptible_sleep(
                    min(max(submit_backoff, _MIN_POLL_BACKOFF_S), 60.0)
                )
                continue

            if not in_flight:
                self._settle_concurrency(cycle_setback, cycle_progress)
                # Only render-retry tiles waiting out their delay: pace the
                # loop instead of spinning hot until they mature.
                if self._render_deferred and not pending and not resubmit:
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
                        "Tile {idx} failed: {err}".format(idx=tile_idx, err=err)
                    )
                    completed += 1
                    self._emit_progress(completed, total)
                    finished_ids.append(request_id)

                elif status == "pending":
                    retry_after = float(resp.get("retry_after", poll_interval))
                    if time.monotonic() > deadline:
                        self.warning.emit(
                            "Tile {idx} timed out after {s}s".format(
                                idx=tile_idx, s=int(max_wait)
                            )
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
                    if code in _TRANSIENT_CODES:
                        # Transient poll error. Enforce the same deadline as the
                        # pending branch so a tile whose status keeps failing
                        # transiently times out instead of looping forever.
                        if time.monotonic() > deadline:
                            self.warning.emit(
                                "Tile {idx} timed out after {s}s".format(
                                    idx=tile_idx, s=int(max_wait)
                                )
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
                            "Tile {idx}: unexpected poll response code={code}".format(
                                idx=tile_idx, code=code
                            )
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
                    min(max(next_backoff, _MIN_POLL_BACKOFF_S), 5.0)
                )

        self._emit_terminal()

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
            if width_m <= 0:
                return None
            return round(width_m / size[0], 4)
        except Exception:  # nosec B110 -- best-effort analytics field
            return None

    def _build_submission(self, tile_idx: int, tile_spec, png_bytes) -> tuple[dict, dict]:
        """Build the (submission, tile_transform) pair for one encoded tile,
        shared by the batched and streaming paths."""
        from ..core.cloud_detection import tile_png_to_base64

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
            "max_masks": _MAX_MASKS_PER_TILE,
            "threshold": self._detection_threshold,
            "mask_threshold": None,
            "exemplars": self._tile_exemplars.get(tile_idx) or None,
            # Set for re-split quadrants only: lets the server bill the parent
            # once and treat its finer re-scan as part of the same paid work.
            # Older servers ignore the field (the quadrant is billed normally).
            "parent_tile_index": self._billed_ancestor_of(tile_idx),
        }
        return submission, tile_transform

    def _run_streaming(self, total: int) -> None:
        """Continuous sliding-window detection for the synchronous direct
        endpoint. Keeps up to max_concurrent /predict posts in flight at all
        times; the instant any reply finishes it is converted + emitted and a new
        tile is fired, so the service never idles between batches and tiles stream in
        one-by-one. No poll phase (the direct endpoint answers each post with the
        finished masks). A user cancel stops firing new tiles at once, then
        drains the already-in-flight set (bounded by _STOP_DRAIN_BUDGET_S) so
        their billed masks are kept before sockets are released. Falls back to
        requeue+retry on transient/rate-limit codes and stops cleanly on
        exhausted/fatal, mirroring the batched path."""
        from qgis.core import QgsNetworkAccessManager
        from qgis.PyQt.QtCore import QEventLoop, QCoreApplication

        nam = QgsNetworkAccessManager.instance()
        pending: deque = deque(enumerate(self._tiles))
        # (tile_idx, tile_spec, png_bytes, not_before) - already encoded; not_before
        # is the monotonic instant the retry may be re-posted (0.0 = immediately).
        # Without it a busy server was re-hammered with zero delay, exactly the
        # synchronized-retry storm the queue is meant to absorb.
        resubmit: deque = deque()
        in_flight: dict = {}       # reply -> (tile_idx, tile_spec, tile_transform, png_bytes)
        submit_attempts: dict[int, int] = {}
        busy_since: dict[int, float] = {}  # see _run_batched: time-budget busy retries
        completed = 0
        # Consecutive per-tile rejections with no success in between
        # (see _RUN_FATAL_CODES / _MAX_CONSECUTIVE_TILE_FATALS).
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
                    status, payload = self._encode_or_defer(tile_idx, spec)
                    if status == "defer":
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
                in_flight[reply] = (tile_idx, tile_spec, tile_transform, png_bytes)
                # This tile's inference just started: use the wait for its
                # result to render the NEXT tiles concurrently on the main thread.
                self._request_render_prefetch(pending)
                return True
            return False

        # Prime the window at the current adaptive (AIMD) width, not the full
        # max_concurrent: it opens narrow and grows per clean cycle.
        while len(in_flight) < self._aimd.cap and fire_next():
            pass

        while (
            in_flight or resubmit or pending or self._render_deferred
        ) and not self._stop_requested:
            if not in_flight:
                # Everything in flight drained while retries wait out their
                # back-off (a fully busy server can reach this): pace with an
                # interruptible sleep and try to refill, instead of exiting the
                # run with tiles still owed.
                self._interruptible_sleep(0.25)
                while len(in_flight) < self._aimd.cap and fire_next():
                    pass
                continue
            # Block until a network event arrives (or 250ms), so this loop never
            # busy-spins; a cancel registers within one slice.
            QCoreApplication.processEvents(_wait, 250)
            if self._stop_requested:
                break

            done = [r for r in in_flight if r.isFinished()]
            if not done:
                continue

            # Per-cycle AIMD signals for this drain (see _run_batched).
            cycle_setback = False
            cycle_progress = False
            stop_payload = None
            for reply in done:
                tile_idx, tile_spec, tile_transform, png_bytes = in_flight.pop(reply)
                response = self._client.parse_reply(reply)
                reply.deleteLater()
                outcome = self._classify_submit_response(tile_idx, response, tile_transform)
                kind = outcome[0]
                if kind == "completed_inline":
                    _, resp, ttf = outcome
                    _, _, tile_w, tile_h = tile_spec
                    if self._emit_completed(resp, tile_idx, tile_w, tile_h, ttf):
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
                    # Same policy as _run_batched: queue-busy retries burn a
                    # TIME budget with jittered server-suggested delays;
                    # transient errors keep the attempt ceiling.
                    delay, is_busy = outcome[1], outcome[2]
                    retry_code = outcome[3] if len(outcome) > 3 else ""
                    now = time.monotonic()
                    if is_busy:
                        first = busy_since.setdefault(tile_idx, now)
                        give_up = (now - first) > _QUEUE_RETRY_BUDGET_S
                        delay = max(1.0, delay) * random.uniform(*_BUSY_JITTER)  # nosec B311 - jitter, not crypto
                        self._fastfail.reset()  # server reached: not offline
                    else:
                        n = submit_attempts.get(tile_idx, 0) + 1
                        submit_attempts[tile_idx] = n
                        give_up = n > _MAX_RATE_LIMIT_RETRIES
                        delay = min(30.0, delay * (2 ** min(n - 1, 4)))
                        delay *= random.uniform(0.5, 1.0)  # nosec B311 - jitter, not crypto
                        cycle_setback = True
                        # Fed regardless of past successes: pre-first-success
                        # it trips at the small default threshold; mid-run
                        # only at the larger _MIDRUN_OFFLINE_STREAK.
                        self._fastfail.record(retry_code)
                    if give_up:
                        self.warning.emit(
                            "Tile {idx}: submit retries exhausted; skipping".format(idx=tile_idx)
                        )
                        completed += 1
                        self._emit_progress(completed, total)
                    else:
                        resubmit.append((tile_idx, tile_spec, png_bytes, now + delay))
                elif kind in ("ok",):
                    # A "pending" reply on the direct path is unexpected (the
                    # endpoint is synchronous). Treat as a skip so the run never
                    # hangs waiting to poll a path that does not exist here.
                    self.warning.emit(
                        "Tile {idx}: unexpected pending on direct path; skipping".format(idx=tile_idx)
                    )
                    completed += 1
                    self._emit_progress(completed, total)
                elif kind == "skip":
                    completed += 1
                    self._emit_progress(completed, total)
                elif kind == "tile_fatal":
                    # Per-tile rejection: skip this tile, keep the run alive.
                    # A streak with no success in between is systematic (see
                    # _run_batched) and stops the run with that code.
                    bad_code = outcome[1] or "UNKNOWN"
                    fatal_streak += 1
                    self.warning.emit(
                        "Tile {idx}: rejected ({code}); skipping".format(
                            idx=tile_idx, code=bad_code))
                    completed += 1
                    self._emit_progress(completed, total)
                    if stop_payload is None and fatal_streak >= _MAX_CONSECUTIVE_TILE_FATALS:
                        stop_payload = ("fatal", bad_code)
                elif stop_payload is None:  # "exhausted" or "fatal"
                    stop_payload = outcome

            # Offline fast-fail: abort a run that only ever sees hard-connectivity
            # errors (see _run_batched for the rationale and the mid-run rule).
            offline_streak = self.tiles_succeeded == 0 or self._fastfail.streak >= _MIDRUN_OFFLINE_STREAK
            if stop_payload is None and self._fastfail.tripped and offline_streak:
                stop_payload = ("fatal", _OFFLINE_STOP_CODE)

            if stop_payload is not None:
                # Flush withheld re-split parents BEFORE the terminal signal:
                # the main thread finalizes (nulls the merger) on that signal,
                # so a parent's late tile_completed would race the finalize and
                # its billed detections could be dropped. _flush_withheld clears
                # _withheld, so the later call in _emit_terminal is a no-op.
                self._flush_withheld()
                if stop_payload[0] == "exhausted":
                    self.credits_exhausted.emit(stop_payload[1])
                    self._stop_reason = "exhausted"
                else:
                    self.error.emit(self._submit_error_message(stop_payload[1]))
                    self._stop_reason = "error"
                self._stop_requested = True
                break

            # Fold this drain cycle into the adaptive width, then refill to it so
            # the window stays full (continuous, no barrier). Quadrants queued by
            # saturated tiles in THIS cycle join the deque first, so the refill
            # can fire them and the loop guard sees them before exiting.
            total += self._drain_subtiles(pending)
            self._settle_concurrency(cycle_setback, cycle_progress)
            # Do not start new tiles once a stop is pending: a cancel that landed
            # during this cycle's reply-processing would otherwise fire fresh work
            # (a blocking main-thread render + a POST) that then has to be drained
            # or aborted, delaying the wind-down. A tile not fired is not billed.
            while (not self._stop_requested
                   and len(in_flight) < self._aimd.cap and fire_next()):  # noqa: W503
                pass
            # Keep the render pipeline fed even on cycles where no slot freed
            # (e.g. only deferred retries matured): the prefetch is what hides
            # the per-tile render behind the in-flight inference.
            if not self._stop_requested:
                self._request_render_prefetch(pending)

        # User cancel: before releasing sockets, drain the tiles ALREADY in
        # flight. The server bills a tile when it processes the request, so a
        # reply we abort unread is a detection paid for and thrown away. New
        # tiles stopped firing the instant Cancel was pressed (the real cost +
        # time driver), so this only waits out the small in-flight set already
        # sent, and only up to _STOP_DRAIN_BUDGET_S so a hung reply can never
        # hold the stop open. Errors/exhausted skip this: there is no user
        # result to salvage and the wall is already hit.
        # Only drain when the service has proven responsive (at least one tile
        # already landed): then the handful still in flight are worth the short
        # wait to keep their billed masks. If nothing has EVER succeeded, the
        # in-flight replies are a cold-start / dead-service hang that will not
        # land in the drain window anyway, so abort at once and let Cancel feel
        # instant (the aborted replies are refunded server side as non-completed).
        if self._stop_reason == "user" and in_flight and self.tiles_succeeded > 0:
            drain_deadline = time.monotonic() + _STOP_DRAIN_BUDGET_S
            while in_flight and time.monotonic() < drain_deadline:
                QCoreApplication.processEvents(_wait, 100)
                for reply in [r for r in in_flight if r.isFinished()]:
                    tile_idx, tile_spec, tile_transform, png_bytes = in_flight.pop(reply)
                    response = self._client.parse_reply(reply)
                    reply.deleteLater()
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
                    # A non-completed reply on a user stop is not retried: it
                    # drops to the abort/refund path below.

        # On stop, abort any still-in-flight replies so sockets are released.
        for reply in list(in_flight.keys()):
            try:
                if not reply.isFinished():
                    reply.abort()
                reply.deleteLater()
            except (RuntimeError, AttributeError):
                pass
        in_flight.clear()

        self._emit_terminal()

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
            if crop is None or crop.isNull():
                continue
            valid.append((crop, int(label), obj_box, full_box))
        if not valid:
            return

        # Size cap for the RUN-TOTAL exemplar count, even though a tile may
        # paste only a subset (home-tile exemplars are sent in-situ instead):
        # any subset then always fits the band, and a crop keeps ONE apparent
        # scale on every tile it is pasted on. Small crops keep their native
        # crispness; only larger ones are downsized here (smooth filter).
        cap = stamp_size_cap(len(valid))
        for crop, label, obj_box, full_box in valid:
            if max(crop.width(), crop.height()) > cap:
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
        from ..core.cloud_detection import in_situ_exemplar_box
        from ..core.tile_manager import OVERLAP_FRACTION, TILE_SIZE

        paste: list = []
        in_situ: list = []  # [(stamp, tile-local box)]
        for i, stamp in enumerate(self._stamps):
            full_box = (self._stamp_full_boxes[i]
                        if i < len(self._stamp_full_boxes) else None)
            local = in_situ_exemplar_box(full_box, tx, ty, tw, th)
            if local is None:
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
                    paste.append(stamp)
                else:
                    kept.append((stamp, local))
            in_situ = kept
        boxes = [
            {"box": [float(v) for v in local], "label": int(stamp[1])}
            for stamp, local in in_situ
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
        band_content_h = max(int(crop.height()) for crop, _l, _b in self._stamps)
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
        for tile_idx, spec in list(pending)[:_PREFETCH_DEPTH]:
            if len(self._prefetched) >= _PREFETCH_DEPTH:
                return
            if tile_idx in self._prefetched:
                continue
            tx, ty, tw, th = spec
            out_w, out_h = self._tile_outsize.get(tile_idx, (0, 0))
            seq = self._render_request(tx, ty, tw, th, out_w, out_h)
            if seq is None:
                return
            self._prefetched[tile_idx] = seq

    def _encode_or_defer(self, tile_idx: int, spec) -> tuple:
        """Encode one tile with the blank/failed-render retry ladder.

        Returns one of:
          ("ok", (tile_spec, image_bytes))  - ready to submit
          ("defer", None)                   - re-render queued (worker keeps it)
          ("skip", None)                    - permanently dropped (caller counts
                                              it done, exactly the old None path)

        A blank or empty render is retried up to _RENDER_RETRY_MAX times with a
        PER-ATTEMPT DOUBLING delay from _RENDER_RETRY_DELAY_S (an online
        basemap usually just had not fetched that area yet, and a multi-second
        provider hiccup must not swallow the whole ladder); only when the
        ladder is exhausted is the tile counted in tiles_skipped_blank /
        tiles_render_failed and skipped for good. A failure also holds the
        render PREFETCH off for _PREFETCH_HOLDOFF_S so concurrent fetches stop
        piling into a struggling provider.
        """
        tx, ty, tw, th = spec
        status, payload = self._encode_tile(tile_idx, tx, ty, tw, th)
        if status == "ok":
            self._render_attempts.pop(tile_idx, None)
            return ("ok", payload)
        if status in ("blank", "render"):
            self._prefetch_holdoff_until = time.monotonic() + _PREFETCH_HOLDOFF_S
        if status in ("blank", "render") and not self._stop_requested:
            attempts = self._render_attempts.get(tile_idx, 0)
            if attempts < _RENDER_RETRY_MAX:
                self._render_attempts[tile_idx] = attempts + 1
                delay = _RENDER_RETRY_DELAY_S * (2 ** attempts)
                self._render_deferred.append(
                    (time.monotonic() + delay, tile_idx, spec))
                return ("defer", None)
        # Ladder exhausted (or a non-retryable failure): count + skip.
        if status == "blank":
            self.tiles_skipped_blank += 1
        elif status == "render":
            self.tiles_render_failed += 1
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
          ("blank", None) when the render is a uniform/nodata fill;
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
            composite_tile_with_stamps, encode_tile_png, tile_is_blank,
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
            if prefetch_seq is not None and self._render_collect is not None:
                tile_img = self._render_collect(prefetch_seq)
            else:
                tile_img = self._tile_renderer(tx, ty, tw, th, out_w, out_h)
            if tile_img is None or tile_img.isNull():
                # The render produced nothing: a provider/WMS hole or timeout,
                # not real ground. Retryable (the caller's ladder decides when
                # it becomes a counted coverage hole); never submitted, so
                # never billed.
                return ("render", None)
            # Blank/nodata tile: on an online basemap this is usually imagery
            # that was not downloaded yet, so it is retryable too. Once the
            # ladder is exhausted the caller skips it before submit, so an
            # empty region inside the zone (mosaic gap, out-of-footprint
            # corner) never spends a credit to return nothing. Checked on the
            # raw render, before any example stamp is composited in.
            if tile_is_blank(tile_img):
                return ("blank", None)
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
                boxes = ex_boxes + insitu_boxes
                if boxes:
                    self._tile_exemplars[tile_idx] = boxes
                # stamp_norm only ever covers PASTED pixels: a tile that pasted
                # nothing must record NO stamp region at all, so the drop
                # filter can never discard a detection of the in-situ object.
                if stamp_norm:
                    self._tile_stamp_norm[tile_idx] = stamp_norm
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
        from ..core.cloud_detection import tile_png_to_base64

        submissions = []
        transforms = []
        for tile_idx, tile_spec, png_bytes in batch:
            tile_x, tile_y, tile_w, tile_h = tile_spec
            tile_transform = self._make_tile_transform(tile_x, tile_y, tile_w, tile_h)
            bbox_native = tile_transform["bbox_native"]
            transforms.append(tile_transform)
            submissions.append({
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
                "max_masks": _MAX_MASKS_PER_TILE,
                "threshold": self._detection_threshold,
                "mask_threshold": None,
                # Per-tile example boxes (where the stamps were pasted on THIS
                # tile); None/[] for text-only runs.
                "exemplars": self._tile_exemplars.get(tile_idx) or None,
                # Re-split quadrants carry their parent so the server can bill
                # the parent once for the whole re-scan (older servers ignore).
                "parent_tile_index": self._billed_ancestor_of(tile_idx),
            })

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
            if code in _EXHAUSTED_CODES:
                return ("exhausted", int(response.get("credits_remaining", 0)))
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
            if code in _TRANSIENT_CODES:
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
            # Non-retryable error. Run-level codes end the run; anything else
            # rejects only THIS tile (the caller skips it and keeps going,
            # with a consecutive-rejection guard for systematic failures).
            if code in _RUN_FATAL_CODES:
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
                "Tile {idx}: submit response missing request_id; skipping".format(
                    idx=tile_idx
                )
            )
            return ("skip",)

        poll_interval = float(response.get("poll_interval", _DEFAULT_POLL_INTERVAL_S))
        max_wait = float(response.get("max_wait", _DEFAULT_MAX_WAIT_S))
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
        depth = self._tile_depth.get(tile_idx, 0)
        if depth >= self._subdiv_max_depth or self._subdivide_budget < 4:
            return False
        try:
            tx, ty, tw, th = self._tiles[tile_idx]
        except (IndexError, ValueError):
            return False
        quads = subdivide_quadrants(tx, ty, tw, th)
        if not quads:
            return False
        quads = [q for q in quads if self._quad_intersects_zone(q)]
        if not quads or len(quads) > self._subdivide_budget:
            return False
        self._subdivide_budget -= len(quads)
        self.tiles_subdivided += 1
        for spec in quads:
            self._pending_subtiles.append((spec, depth + 1, tile_idx))
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

    def _billed_ancestor_of(self, tile_idx: int) -> int | None:
        """The BASE-grid ancestor of a re-split quadrant, or None for a base
        tile. A depth-2 quadrant's direct parent is itself a quadrant, so the
        chain is walked to the root: the root is the tile whose charge the
        server can verify when deciding a quadrant rides that charge."""
        parent = self._parent_of.get(tile_idx)
        while parent is not None and self._parent_of.get(parent) is not None:
            parent = self._parent_of.get(parent)
        return parent

    def _drain_subtiles(self, pending: deque) -> int:
        """Move queued quadrant specs into the submit deque as NEW tiles (fresh
        indices appended to self._tiles, so transforms/progress bookkeeping stay
        index-consistent) and return how many were added. The quadrant renders
        at 2x its rect size (out_w/out_h), i.e. 2x finer ground resolution per
        re-split depth. Called from both run loops between cycles."""
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
        """Decode a completed tile and emit its detections. True on success.

        The per-tile decode + geometry pipeline (_handle_completed ->
        decode_detection_response RLE decode, clip intersection(),
        suppress_redundant_hypotheses) can throw on ONE malformed tile (a bad
        score/width, a GEOS/numpy fault). Guarded here so that bad tile becomes
        a skip (a warning, still counted done) exactly like a submit-side
        tile_fatal, instead of propagating to run()'s last-resort net and
        aborting the whole PAID run, which would lose every later tile and leave
        the failed tile mis-accounted. The tile is already billed server-side,
        so skipping only forfeits its geometry; the caller still counts it and
        advances progress.
        """
        # Re-split quadrants render at 2x their grid rect (see _drain_subtiles),
        # so their real SENT image size lives in _tile_outsize, not the grid-rect
        # size the caller carries in tile_spec. Use it as the RLE decode fallback;
        # a base tile has no _tile_outsize entry, so the rect size is kept. Server
        # responses carry width/height, so the fallback only matters if one omits
        # them (mapping a mask at the wrong size shifts it toward its tile origin).
        tile_w, tile_h = self._tile_outsize.get(tile_idx, (tile_w, tile_h))
        try:
            detections = self._handle_completed(
                response, tile_idx, tile_w, tile_h, tile_transform
            )
            self.tile_completed.emit(tile_idx, detections)
            return True
        except Exception as exc:  # noqa: BLE001 - one bad tile must never kill the run
            logger.warning(
                "AutoDetectionWorker: tile %d decode/convert failed: %s",
                tile_idx, exc,
            )
            self.warning.emit(
                "Tile {idx}: could not process result; skipping".format(idx=tile_idx)
            )
            return False

    def _handle_completed(
        self,
        response: dict,
        tile_idx: int,
        tile_w: int,
        tile_h: int,
        tile_transform: dict,
    ) -> list:
        """Decode a completed status response.

        Returns a list of (mask, score, box, tile_transform) for the
        tile_completed signal. Masks are NOT retained past this call.
        """
        from ..core.cloud_detection import decode_detection_response

        # Results are flowing again: clear any "in line / server busy" UI state.
        self._note_flowing()

        decoded = decode_detection_response(
            response, tile_w, tile_h, self._score_threshold
        )
        # Tile at (or brushing) the per-inference ceiling => the scene likely
        # had more objects than one inference can emit; flag it so the run end
        # can hint "raise Detail".
        resplit = False
        if len(decoded) >= self._mask_cap_trigger:
            self._hit_mask_cap = True
            self.tiles_mask_capped += 1
            # Re-split ladder: with budget + depth headroom, queue this tile's
            # quadrants at 2x scale so the truncated objects get their own
            # inference slots. Only tiles that stay capped at the end of the
            # ladder count as residual truncation for the review hint.
            resplit = self._maybe_subdivide(tile_idx)
            if not resplit:
                self.tiles_capped_final += 1

        # Per-tile detections are emitted to the main thread immediately, which
        # converts them to geometry and drops the masks. We deliberately do NOT
        # accumulate masks for the whole run: that was a multi-GB memory hazard,
        # and the all_tiles_finished payload is unused by every consumer (the
        # main slot and the MCP loop both ignore it, so it stays an empty list).
        # Composite-per-tile: drop any detection whose centroid lands on the
        # stamped example region (the example itself, not a real object).
        # stamp_norm is the REAL pasted rectangle ([nx0,ny0,nx1,ny1] normalized),
        # along the top edge or, for the first grid row, the bottom edge.
        # Prefer the server box ([cx,cy,w,h] normalized); fall back to the mask
        # centroid so the example is still dropped even if the server omits a box.
        stamp = self._tile_stamp_norm.get(tile_idx)
        kept = []
        for mask, score, box in decoded:
            if stamp and self._centroid_in_stamp(box, mask, stamp):
                continue
            kept.append((mask, score))

        # Convert masks -> ready geometry (WKB) on THIS worker thread, so the GUI
        # only rehydrates + merges. This is the per-tile work that used to freeze
        # the GUI at end-of-run; doing it here overlaps it with network I/O.
        detections = self._detections_to_geoms(kept, tile_transform)

        logger.debug(
            "AutoDetectionWorker: tile %d completed with %d detection(s)",
            tile_idx, len(detections),
        )

        if resplit:
            # This tile's quadrants will re-read the same ground 2x finer:
            # withhold its coarse detections so they never union-bridge the
            # quadrants' cleanly separated objects (flushed at the terminal
            # only if every quadrant fails; see _flush_withheld).
            self._withheld[tile_idx] = detections
            return []

        parent = self._parent_of.get(tile_idx)
        if parent is not None and detections:
            # A quadrant delivered: its parent's withheld coarse read is
            # permanently superseded.
            self._parents_with_child_results.add(parent)

        return detections

    def _build_clip_engine(self) -> None:
        """Rebuild the zone clip geometry + a prepared GEOS engine on the worker
        thread from the copied WKB. No-op (clip stays None) for the rectangle/MCP
        path where clip_polygon_wkb is None, matching the GUI's old behaviour."""
        if not self._clip_polygon_wkb:
            return
        from qgis.core import QgsGeometry

        geom = QgsGeometry()
        geom.fromWkb(self._clip_polygon_wkb)
        if geom.isEmpty():
            return
        self._clip_geom = geom
        try:
            engine = QgsGeometry.createGeometryEngine(geom.constGet())
            engine.prepareGeometry()
            self._clip_engine = engine
        except Exception:  # noqa: BLE001 - fall back to plain intersection()
            self._clip_engine = None

    def _detections_to_geoms(self, kept, tile_transform) -> list:
        """Turn (mask, score) detections into ready (geom_wkb: bytes, score) on
        the worker thread: refine -> polygonize -> clip-to-zone -> repair -> WKB.

        This is the verbatim per-detection pipeline that used to run on the GUI
        thread in the plugin's _process_auto_queue, minus the scored-store/merger
        writes (those stay on the GUI). Every op here is value-class QgsGeometry
        or pure numpy/scipy, safe off the main thread; no QgsProject, no layer
        edits, no area measurement.
        """
        import numpy as np
        from ..core.polygon_exporter import (
            fill_small_holes,
            mask_to_polygons,
            suppress_redundant_hypotheses,
        )
        from ..core.layer_conventions import repair_polygon, to_multipolygon

        # Light sub-pixel simplification (0.75x GSD) trims the staircase off every
        # mask as it is built; the post-run refine simplifies further (2.5x GSD),
        # so this never costs final-shape fidelity.
        tile_simplify_tol = 0.75 * self._gsd if self._gsd > 0 else 0.0
        # Anti-sliver floor: on uniform texture the cloud model returns sub-pixel noise
        # fragments (~0.1 m2) that clutter the output. Drop any detection whose
        # ground area is below a small square tied to pixel size. k=1.5 means a
        # detection smaller than ~1.5 px on a side is noise, not an object: at
        # gsd 0.4 m/px the floor is (1.5*0.4)^2 = 0.36 m2, which drops 0.1 m2
        # slivers while keeping a 2x2 m tree/car. gsd<=0 (no metric scale) =>
        # floor 0, no drop.
        min_keep_area = (1.5 * self._gsd) ** 2 if self._gsd > 0 else 0.0
        # Tile ground size, for the observed mask-resolution bookkeeping below.
        # tile_transform["bbox"] uses the polygon_exporter (minx, maxx, miny,
        # maxy) convention.
        bbox = tile_transform.get("bbox", (0.0, 1.0, 0.0, 1.0))
        ground_w = float(bbox[1] - bbox[0])
        ground_h = float(bbox[3] - bbox[2])
        out = []
        for mask, score in kept:
            # Crop the mask to the object's bounding box BEFORE the per-pixel work
            # (scipy fill-holes + rasterio polygonize). A dense run returns ~130
            # masks/tile, each a full HxW (1024^2) array with one small object;
            # scanning the whole grid per mask was ~17s/batch of idle time.
            # Cropping makes both ops proportional to the object, not the tile, and
            # mask_to_polygons offsets the geo-transform by (col0,row0) so the
            # output stays pixel-exact. full_shape keeps the px->ground scale.
            full_h, full_w = mask.shape
            # Record the mask's OWN ground resolution (the polygon staircase
            # step): tile ground size / returned mask size. Max across the run
            # so partial boundary tiles (finer ratio) never understate it.
            if full_w > 0 and full_h > 0 and ground_w > 0:
                gsd_mask = max(ground_w / full_w, ground_h / full_h)
                if gsd_mask > self.observed_mask_gsd:
                    self.observed_mask_gsd = gsd_mask
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
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
                    continue
                blob_check = True
            row0, col0 = int(ys.min()), int(xs.min())
            sub = mask[row0:int(ys.max()) + 1, col0:int(xs.max()) + 1]
            # Pad 1px of background on every side BEFORE fill_holes. binary_fill_holes
            # floods inward from the array border: an object touching the crop edge
            # (always true - the crop IS its bbox) would let a concavity that opens
            # onto that edge connect to the border and fill differently than in the
            # full tile. The 1px background margin restores the full-array result
            # exactly. The offset shifts by 1 to keep the geo-reference pixel-exact.
            sub = np.pad(sub, 1, constant_values=False)
            # Fill interior PINHOLES only (mask staircase / compression noise,
            # rooftop texture pits up to ~2.5 ground-meters, i.e. ~6 m2). Real
            # interior holes (courtyards, ring roads, islands: the smallest
            # real patios run ~3x3 m and up) are kept so the review's "Fill
            # holes" toggle stays meaningful; an unconditional fill here
            # exported every courtyard building as a solid block and made that
            # toggle a no-op. 1.2 m proved too tight: textured industrial
            # roofs kept fields of 2-5 m2 pits that read as noise.
            max_hole_px = int((2.5 / self._gsd) ** 2) if self._gsd > 0 else 36
            sub = fill_small_holes(sub, max(9, max_hole_px))
            for geom in mask_to_polygons(
                sub, tile_transform, simplify_tolerance=tile_simplify_tol,
                pixel_offset=(col0 - 1, row0 - 1), full_shape=(full_h, full_w),
            ):
                if geom is None or geom.isEmpty():
                    continue
                # Confine results to the DRAWN polygon: a boundary tile is
                # rectangular and overflows the shape. A prepared-engine
                # contains() skips the clip for the interior majority; only
                # boundary-crossing detections pay for intersection().
                if self._clip_geom is not None:
                    inside = False
                    if self._clip_engine is not None:
                        try:
                            inside = self._clip_engine.contains(geom.constGet())
                        except Exception:  # noqa: BLE001 - fall back to clip
                            inside = False
                    if not inside:
                        geom = geom.intersection(self._clip_geom)
                    if geom is None or geom.isEmpty() or geom.area() <= 0:
                        continue
                # Coerce to a polygon-only MultiPolygon at the SOURCE: the clip
                # intersection can yield a GeometryCollection that a MultiPolygon
                # layer would later reject.
                geom = to_multipolygon(repair_polygon(geom) or geom)
                if geom is None or geom.isEmpty():
                    continue
                # Drop sub-pixel noise slivers (computed once above). Placed AFTER
                # the clip + repair so a detection trimmed to a tiny boundary
                # sliver is also dropped, not just intrinsically tiny ones.
                if min_keep_area > 0.0 and geom.area() < min_keep_area:
                    continue
                # Armed by the >55% coverage gate above: keep a compact (solid,
                # rectangular-ish) large object, drop an irregular texture blob.
                if blob_check and not self._is_compact_shape(geom):
                    continue
                out.append((geom, float(score)))
        # Raw (pre-NMS/pre-merge) detection count, for the run-summary log: the
        # MAP pre-merge below shrinks what the GUI receives, so the GUI-side
        # fold counter alone would under-report the model's raw output. Plain
        # int increment, GIL-safe; read by the GUI only at a worker terminal.
        self.raw_detections_total += len(out)
        # SEPARATE/count mode: resolve the model's overlapping same-region
        # hypotheses (whole-complex vs per-roof vs roof-section) by SELECTION
        # before the merger ever sees them. Without this, the merger's IoS
        # dedup UNIONS the granularities into one mega-object that inherits the
        # max constituent score, so a low-score complex-wide mask (shadow
        # fringe included) surfaces at its best roof's confidence. MAP mode is
        # left untouched: coverage there is the union of hypotheses by design.
        if self._merge_separate or self._collect_raw:
            # NMS also runs in raw-collect mode: it resolves the model's
            # overlapping same-region hypothesis stacks (not distinct instances),
            # so the fragments the client keeps are already free of that
            # redundancy while still un-gated and un-premerged.
            ms = self._merge_scalars
            sup_kwargs = {k: ms[k] for k in (
                "ios_threshold", "dup_ios_floor", "dup_centroid_frac") if k in ms}
            out = suppress_redundant_hypotheses(out, **sup_kwargs)
        else:
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
            "seam_span_ios") if k in ms}
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
    def _is_compact_shape(geom) -> bool:
        """True if geom fills >=85% of its oriented minimum bounding box.

        Used to rescue a REAL large object (warehouse, big roof) from the
        whole-tile "everything"-blob drop: man-made large objects are solid and
        near-rectangular, texture blobs (canopy, bare soil) are ragged. Any
        failure counts as not-compact, which restores the old drop behaviour.
        """
        try:
            _obb, obb_area, _angle, _w, _h = geom.orientedMinimumBoundingBox()
            if obb_area and obb_area > 0.0:
                return geom.area() / obb_area >= 0.85
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
