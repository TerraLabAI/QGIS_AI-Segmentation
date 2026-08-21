"""Off-GUI stitching of arriving tile geometry into the live preview set.

The converter pool turns each finished tile's masks into fragment geometry on a
worker thread. Everything AFTER that used to run on the GUI thread: folding the
fragments into the run's merger, measuring each merged object, applying the
run's shape preset to it, and diffing the result onto the preview layer. On a
dense run that work does not fit in the crumbs of event loop a live canvas
leaves, so the preview fell behind the service and QGIS stuttered while it
caught up.

This thread does all of it instead. It owns the run's ``IncrementalMerger`` for
the duration, and hands the GUI a queue of ready-to-draw deltas: one entry per
object that appeared, changed shape or score, or left the visible set. A repaint
then costs the event loop a provider write and nothing else.

Ownership is a baton, not a share. The GUI creates the merger, hands it over
with ``start()``, and must not read it again until ``join_run()`` returns.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

from qgis.PyQt.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

#: Upper bound on retained per-tile fragments (the count-vs-map re-group and the
#: review override read them). Past it the list is dropped and the override is
#: marked unavailable; the map-likeness counters keep counting either way, so
#: the automatic decision still runs. A run this dense is continuous cover in
#: practice.
RAW_FRAGMENT_RETAIN_CAP = 40000

#: How long a teardown waits for the thread to leave its current tile. Folding
#: one tile is bounded work with no network in it, so this only ever has to
#: cover the slowest single fold.
STITCH_JOIN_TIMEOUT_MS = 8000

#: Relative change in the detection pixel size that is worth rebuilding every
#: shown shape for. Below it the reported size is only creeping (it is a running
#: maximum over the tiles answered so far) and a rebuild would cost one refine
#: per shown object, per tile.
_RESCALE_MIN_CHANGE = 0.05

#: How many drain cycles an object may keep growing before it is shaped anyway.
#: An object on a tile boundary is touched by every tile that overlaps it, and
#: shaping it each time is the work this ceiling bounds; a long road that never
#: stops growing still gets its shape within this many tiles.
_SHAPE_MAX_WAIT = 3

#: Delta kinds. The GUI maps them straight onto provider operations.
DELTA_ADD = "add"
DELTA_UPDATE = "update"
DELTA_REMOVE = "remove"


def _build_area_measurer(crs_authid: str):
    """The run's geodesic area measurer, or None to fall back to planar area.

    Reads the project, so it belongs on the GUI thread.
    """
    from qgis.core import QgsCoordinateReferenceSystem

    from ..core.layer_conventions import make_area_measurer

    try:
        return make_area_measurer(QgsCoordinateReferenceSystem(crs_authid))
    except Exception:  # noqa: BLE001 - planar area is a usable fallback
        return None


def _detached(geom):
    """A geometry that shares no data with ``geom``.

    ``QgsGeometry(other)`` is a shallow copy: both wrappers point at the same
    abstract geometry, so it is not enough to hand one across a thread. Cloning
    the inner geometry is.
    """
    from qgis.core import QgsGeometry

    inner = geom.constGet()
    if inner is None:
        return QgsGeometry(geom)
    return QgsGeometry(inner.clone())


class LiveStitchThread(QThread):
    """Folds tile fragments into the run's merger and emits drawable deltas.

    Created and started by the GUI at the top of a run, fed one list of
    ``(geom_wkb, score)`` per finished tile, and joined at the terminal before
    anything reads the merger again.
    """

    #: Emitted after a tile produced at least one delta. Carries no payload: the
    #: GUI drains ``take_deltas()`` on its own coalesced repaint tick, so a burst
    #: of tiles becomes one provider write.
    batch_ready = pyqtSignal()

    def __init__(
        self,
        merger,
        params: dict,
        pixel_size: float,
        metres_per_unit: float,
        crs_authid: str,
        unit_aspect: float = 1.0,
        retain_fragments: bool = False,
        retain_coverage: bool = False,
        tile_ground_area: float = 0.0,
        hard_coverage: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._merger = merger
        self._params = dict(params or {})
        self._pixel_size = float(pixel_size or 0.0)
        self._metres_per_unit = float(metres_per_unit or 1.0)
        # Measured by the GUI, for the same reason as the area measurer below:
        # it reads the project. 1.0 is right for a projected run and leaves the
        # squaring reading raw coordinates, which is wrong in a geographic one.
        self._unit_aspect = float(unit_aspect or 1.0)
        self._crs_authid = crs_authid or "EPSG:4326"
        self._retain_fragments = bool(retain_fragments)
        self._retain_coverage = bool(retain_coverage)
        self._tile_ground_area = float(tile_ground_area or 0.0)
        self._hard_coverage = float(hard_coverage or 0.0)

        self._inbox: queue.Queue = queue.Queue()
        self._outbox: list = []
        self._lock = threading.Lock()
        self._aborted = False
        self._stopping = False
        # fid -> (score, visible) for every object the GUI has been told about,
        # so a delta says add / update / remove without the GUI having to guess.
        self._shown: dict[int, float] = {}
        # fid -> how many drain cycles it has been waiting for its shape. An
        # object still being merged is drawn raw and shaped once it settles.
        # Only an object that is ON the map and not yet carrying the run's shape
        # is queued here: the settle pass reads the same geometry and the same
        # score the raw pass just read, so for anything else it can only reach
        # the same answer twice.
        self._pending_shape: dict[int, int] = {}
        # fid -> the merger keeper geometry the object's drawn shape was built
        # from. A tile that only re-reads an object it already has reports it as
        # changed and hands back the reading already on the map, and rebuilding
        # that is the largest avoidable cost in a dense fold. The refine is a
        # pure function of the geometry and the run's dials, and the dials move
        # only on a rescale, which clears this, so the same geometry means the
        # same shape.
        #
        # Compared with ``is``, never by value: it is an identity test, not a
        # geometry test. Holding the reference is what makes that sound, because
        # a geometry the merger has retired could otherwise be freed and a new
        # one land on its address.
        self._drawn_from: dict[int, object] = {}
        # fid -> (keeper geometry, its ground area). One cycle gates the same
        # object twice, once for the raw draw and once for the shaped one, and
        # the geodesic measure walks every vertex.
        self._area_of: dict[int, tuple] = {}
        self._refiner = None
        # Built HERE, on the GUI thread, because make_area_measurer reads the
        # project's transform context and ellipsoid. The result is a value
        # class and travels; the read does not.
        self._measurer = _build_area_measurer(self._crs_authid)
        self._rescaled = False

        # Observability, read after the join: how long the fold actually took
        # and whether the run ever had to rebuild its shapes. A stitcher that
        # falls behind shows up here before it shows up as a laggy preview.
        self.fold_ms = 0.0
        self.tiles_folded = 0
        self.rescales = 0

        # The shape this thread built for each object it showed, keyed by merger
        # fid, plus the two run scalars it built them at. The finalize applies
        # the SAME refine to the SAME bases, so without this it re-runs the whole
        # GEOS shape pass on the GUI thread while the user waits on
        # "Almost done". Read after the join only, like the counters above, and
        # only when the two scalars still match what the finalize resolved.
        self.shaped_geoms: dict[int, object] = {}
        self.shape_pixel_size = float(pixel_size or 0.0)
        self.shape_metres_per_unit = self._metres_per_unit

        # Run-wide accumulators the finalize reads back after the join. They
        # live here because this thread is the only place that sees every
        # fragment once the fold moved off the GUI.
        self.raw_count = 0
        self.raw_n_total = 0
        self.raw_coverage_sum = 0.0
        self.raw_coverage_sq_sum = 0.0
        self.raw_fragments: list | None = [] if retain_fragments else None

    # ---- GUI-thread API -----------------------------------------------------

    def submit(self, detections: list, pixel_size: float = 0.0) -> None:
        """Hand one finished tile over. Returns at once.

        ``pixel_size`` is the run's current detection pixel size. It moves once,
        from the render resolution to the returned-mask grid, the first time a
        tile comes back; a change re-refines everything already shown, so the
        preview never mixes two scales.
        """
        if self._aborted:
            return
        self._inbox.put((list(detections or ()), float(pixel_size or 0.0)))

    def take_deltas(self) -> list:
        """Every delta produced since the last call, oldest first.

        Each entry is ``(kind, fid, geometry, score)``; ``geometry`` is None for
        a remove. Called from the GUI thread only.
        """
        with self._lock:
            out = self._outbox
            self._outbox = []
        return out

    def pending(self) -> int:
        """Tiles handed over but not folded yet."""
        return self._inbox.qsize()

    def finish(self) -> None:
        """Ask the thread to stop once it has folded everything queued.

        The run's terminal calls this, then waits with ``join_run`` so the
        merger is complete before finalize reads it. Idempotent: the cooperative
        wait calls it once per slice.
        """
        if self._stopping:
            return
        self._stopping = True
        self._inbox.put(None)

    def abort(self) -> None:
        """Drop whatever is queued and stop now (teardown, cancel, unload).

        The merger is left as it stands: a hard teardown throws it away, and a
        soft cancel keeps only what was already folded, which is what the user
        was billed for.
        """
        self._aborted = True
        if self._stopping:
            return
        self._stopping = True
        self._inbox.put(None)

    def join_run(self, timeout_ms: int = STITCH_JOIN_TIMEOUT_MS) -> bool:
        """Block until the thread has exited. True when it did.

        Never called from the thread itself. A False return means the fold is
        wedged, which the caller must treat as "do not touch the merger".
        """
        if not self.isRunning():
            return True
        return bool(self.wait(timeout_ms))

    # ---- Stitch-thread body -------------------------------------------------

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self._build_tools()
            while True:
                item = self._inbox.get()
                if item is None:
                    break
                if self._aborted:
                    continue
                detections, pixel_size = item
                started = time.perf_counter()
                try:
                    self._fold_tile(detections, pixel_size)
                except Exception:  # noqa: BLE001 - one bad tile must not end the run
                    logger.warning(
                        "LiveStitchThread: tile fold failed", exc_info=True)
                finally:
                    self.fold_ms += (time.perf_counter() - started) * 1000.0
                    self.tiles_folded += 1
            if not self._aborted:
                # Last word: shape everything that was still growing when the
                # tiles ran out, so the preview the review opens over is not a
                # mix of shaped and raw outlines.
                self._publish(self._drain_merger_changes(settle_all=True))
        except Exception:  # noqa: BLE001 - the thread body must never raise out
            logger.warning("LiveStitchThread: stopped on error", exc_info=True)

    def _build_tools(self) -> None:
        """Build the per-run refiner on THIS thread.

        It resolves every server dial the run's shape preset needs, once,
        instead of once per object. It reads nothing shared, so building it
        here keeps the whole geometry path on one thread. The area measurer
        cannot follow: it reads the project, so __init__ builds it.
        """
        from ..core.live_refine import LiveRefiner

        if self._measurer is None:
            self._measurer = _build_area_measurer(self._crs_authid)
        self._refiner = LiveRefiner(
            self._params, self._pixel_size, self._metres_per_unit,
            self._unit_aspect)

    def _wants_rescale(self, pixel_size: float) -> bool:
        """Whether this pixel size is worth rebuilding the shown shapes for.

        The run starts on the render resolution and learns its true mask grid
        from the first answered tile, which is ONE real change. After that the
        reported size only creeps, because it is a running maximum over the
        tiles seen so far, and every creep would re-refine the whole shown set:
        O(objects) work per tile, late in the run, when the set is at its
        largest. So a rescale needs a material change, and the run gets one.
        """
        if pixel_size <= 0 or self._rescaled:
            return False
        widest = max(pixel_size, self._pixel_size)
        if widest <= 0:
            return False
        return abs(pixel_size - self._pixel_size) / widest > _RESCALE_MIN_CHANGE

    def _rescale(self, pixel_size: float) -> None:
        """Adopt the run's true detection pixel size and re-refine what is shown.

        Once per run (see _wants_rescale): the shapes are px-scaled, so the
        first answered tile invalidates everything built on the pre-run
        estimate, and the set is still small when it lands.
        """
        from ..core.live_refine import LiveRefiner

        self._pixel_size = pixel_size
        self.shape_pixel_size = float(pixel_size)
        self._rescaled = True
        self.rescales += 1
        # Everything shaped at the old scale is about to be rebuilt, and until
        # it is there is nothing here worth handing on. _drawn_from goes with
        # it: the new refiner answers differently on the same geometry, so a
        # match against it would hold the old scale on the map.
        self.shaped_geoms.clear()
        self._drawn_from.clear()
        self._refiner = LiveRefiner(
            self._params, self._pixel_size, self._metres_per_unit,
            self._unit_aspect)
        self._merger.mark_changed(list(self._shown))

    def _fold_tile(self, detections: list, pixel_size: float) -> None:
        """Merge one tile's fragments, then queue the deltas they produced."""
        from qgis.core import QgsGeometry

        if self._wants_rescale(pixel_size):
            self._rescale(pixel_size)

        for wkb, score in detections:
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            if geom.isEmpty():
                continue
            self.raw_count += 1
            if self._retain_coverage:
                self._note_coverage(geom)
            if self.raw_fragments is not None:
                if len(self.raw_fragments) >= RAW_FRAGMENT_RETAIN_CAP:
                    # Overflow: free the list and mark the review override
                    # unavailable. The map-likeness counters keep running.
                    self.raw_fragments = None
                else:
                    self.raw_fragments.append((wkb, float(score)))
            # Fold EVERY fragment confidence-agnostic, carrying its score: a
            # seam half that scored low still unions with its strong other half,
            # and the merged object keeps the MAX score. Filtering happens below
            # on whole objects, never on fragments.
            self._merger.add(geom, float(score))

        self._publish(self._drain_merger_changes())

    def _publish(self, deltas: list) -> None:
        """Hand a batch of deltas to the GUI, if there is anything in it."""
        if not deltas:
            return
        with self._lock:
            self._outbox.extend(deltas)
        self.batch_ready.emit()

    def _note_coverage(self, geom) -> None:
        """Accumulate the area-weighted tile coverage an exemplar-only run reads
        to decide count versus map. Near-whole-tile texture fills are excluded so
        a handful cannot fake continuous cover."""
        if self._tile_ground_area <= 0:
            self.raw_n_total += 1
            return
        self.raw_n_total += 1
        cov = geom.area() / self._tile_ground_area
        if 0.0 < cov <= self._hard_coverage:
            self.raw_coverage_sum += cov
            self.raw_coverage_sq_sum += cov * cov

    def _drain_merger_changes(self, settle_all: bool = False) -> list:
        """Turn the merger's change log into drawable deltas.

        Two passes, because shaping an object costs about a hundred times what
        drawing its raw outline does, and an object sitting on a tile boundary
        is touched again by every tile that overlaps it:

        1. everything that changed this cycle is drawn AT ONCE from its raw
           merged outline, so a detection never waits to appear;
        2. everything that changed in an earlier cycle and has now stopped
           moving gets the run's shape preset and is redrawn.

        So each object is shaped about once instead of once per tile that
        touched it, and the user still sees it the moment it is found. Objects
        that keep growing are shaped anyway after _SHAPE_MAX_WAIT cycles, so a
        long road never waits for the end of the run. ``settle_all`` shapes
        everything still pending, for the last drain before the join.

        Only an object with something left to draw is queued for pass 2, and
        pass 1 says nothing about one whose reading has not moved since it was
        drawn (see _wants_shape and _already_drawn). Most of what the merger
        reports as changed on a dense run is a later tile re-reading an object
        that is already right on the map, and rebuilding those is work whose
        result the user cannot tell from what is already there.
        """
        changed, removed = self._merger.drain_changes()
        out: list = []
        for fid in removed:
            self._forget(fid)
            if self._shown.pop(fid, None) is not None:
                out.append((DELTA_REMOVE, fid, None, 0.0))
        hot = set(changed)
        for fid in changed:
            waited = self._pending_shape.get(fid, 0) + 1
            if waited >= _SHAPE_MAX_WAIT or settle_all:
                self._pending_shape.pop(fid, None)
                out.append(self._object_delta(fid, shaped=True))
            else:
                out.append(self._object_delta(fid, shaped=False))
                if self._wants_shape(fid):
                    self._pending_shape[fid] = waited
                else:
                    self._pending_shape.pop(fid, None)
        for fid in [f for f in self._pending_shape if settle_all or f not in hot]:
            self._pending_shape.pop(fid, None)
            out.append(self._object_delta(fid, shaped=True))
        return [d for d in out if d is not None]

    def _wants_shape(self, fid: int) -> bool:
        """Whether a later cycle still has a shape to build for this object.

        The settle pass runs only on an object the merger has NOT touched since,
        so it reads back the same geometry and the same score the raw pass just
        read. An object off the map fails the same gate again, and one already
        carrying the run's shape has nothing left to build; queueing either only
        buys the same answer twice.
        """
        return fid in self._shown and fid not in self.shaped_geoms

    def _forget(self, fid: int) -> None:
        """Drop everything this thread remembers about one object.

        Every retirement and every drop goes through here, so a fid the merger
        has given up on stops holding its geometry alive.
        """
        self._pending_shape.pop(fid, None)
        self._drawn_from.pop(fid, None)
        self._area_of.pop(fid, None)
        self.shaped_geoms.pop(fid, None)

    def _object_delta(self, fid: int, shaped: bool):
        """One object's delta, or None when it has nothing to say.

        Never raises. By the time this runs the merger's change log has already
        been consumed, so an exception escaping to the per-tile guard would
        discard every delta of this tile, including the ones already built, and
        nothing would ever re-report them: those objects would stay invisible
        for the rest of the run.
        """
        try:
            return self._build_object_delta(fid, shaped)
        except Exception:  # noqa: BLE001 - one bad object, not the whole batch
            logger.warning(
                "LiveStitchThread: no delta for object %s", fid, exc_info=True)
            return None

    def _build_object_delta(self, fid: int, shaped: bool):
        """The delta itself. ``shaped`` picks the run's full shape preset over
        the raw merged outline. Either way the object goes through the same
        visible-set gate, so a raw draw and a shaped draw never disagree about
        whether it belongs on the map."""
        from ..core.layer_conventions import to_multipolygon

        geom, score = self._merger.keeper(fid)
        if geom is None or geom.isEmpty():
            return self._drop(fid)
        if not self._passes_filters(score, self._object_area(fid, geom)):
            return self._drop(fid)
        if self._already_drawn(fid, geom, shaped):
            return self._rescore_delta(fid, geom, score)
        shape = None
        if shaped:
            try:
                shape = self._refiner.refine(geom)
            except Exception:  # noqa: BLE001 - draw the unshaped outline
                shape = None
        refined = shaped and shape is not None and not shape.isEmpty()
        if not refined:
            # Either this object is still growing and gets its raw outline, or
            # the refine failed on it. Either way what is about to be drawn is
            # not the run's shape, so anything held for this fid is stale.
            self.shaped_geoms.pop(fid, None)
        if shape is None or shape.isEmpty():
            # The raw outline IS the merger's live keeper, and the GUI is about
            # to hand it to a provider index and the map render thread while
            # this thread keeps merging into it. A QgsGeometry copy still
            # shares its data, and the bounding box is cached lazily on first
            # read, so three threads would write that cache at once. Detach.
            shape = _detached(geom)
        shape = to_multipolygon(shape)
        if shape is None or shape.isEmpty():
            return self._drop(fid)
        kind = DELTA_UPDATE if fid in self._shown else DELTA_ADD
        self._shown[fid] = float(score)
        self._drawn_from[fid] = geom
        if refined:
            # The held copy is this thread's alone and the GUI gets a detached
            # one, because a later cycle may draw this shape again for a score
            # that moved. Handing out the held instance twice would put the map
            # render thread and this thread on one lazily cached bounding box,
            # the same trap the raw draw detaches to avoid.
            self.shaped_geoms[fid] = shape
            shape = _detached(shape)
        return (kind, fid, shape, float(score))

    def _already_drawn(self, fid: int, geom, shaped: bool) -> bool:
        """Whether the map already carries what this delta would build.

        True when the merger handed back the very geometry the drawn shape came
        from AND that shape is the one being asked for. A shaped pass over an
        object drawn raw is not covered: that object still owes its shape.
        """
        if fid not in self._shown or self._drawn_from.get(fid) is not geom:
            return False
        return (not shaped) or fid in self.shaped_geoms

    def _rescore_delta(self, fid: int, geom, score: float):
        """Report an object whose shape is already right, or nothing.

        Only the score can have moved: the merger takes the max over what an
        object absorbs, so a re-reading can raise it without touching the
        outline. The redraw then carries a copy of the shape already built,
        never a second build of it.
        """
        from ..core.layer_conventions import to_multipolygon

        if self._shown.get(fid) == float(score):
            return None
        held = self.shaped_geoms.get(fid)
        # Detached for the same reason as every other draw: what crosses to the
        # GUI must share nothing with what this thread keeps.
        shape = to_multipolygon(_detached(held if held is not None else geom))
        if shape is None or shape.isEmpty():
            return self._drop(fid)
        self._shown[fid] = float(score)
        return (DELTA_UPDATE, fid, shape, float(score))

    def _drop(self, fid: int):
        """Take one object off the map, or nothing when it was never on it.

        The area memo is deliberately kept: the object is still in the merger,
        under the same geometry, and a gate it fails on its score alone can be
        asked again on the next tile.
        """
        self._pending_shape.pop(fid, None)
        self._drawn_from.pop(fid, None)
        self.shaped_geoms.pop(fid, None)
        if self._shown.pop(fid, None) is None:
            return None
        return (DELTA_REMOVE, fid, None, 0.0)

    def _object_area(self, fid: int, geom) -> float:
        """Geodesic ground area (m2), the same number the export writes, so the
        live size gate agrees with the saved layer.

        Memoized against the geometry it was measured on, because one cycle
        gates the same object twice. The memo holds that geometry, which is also
        what keeps the identity test in _already_drawn sound.
        """
        held = self._area_of.get(fid)
        if held is not None and held[0] is geom:
            return held[1]
        area = 0.0
        try:
            if self._measurer is not None:
                area = float(self._measurer.measureArea(geom))
            else:
                area = float(geom.area())
        except (RuntimeError, AttributeError):
            try:
                area = float(geom.area())
            except (RuntimeError, AttributeError):
                return 0.0
        self._area_of[fid] = (geom, area)
        return area

    def _passes_filters(self, score: float, area: float) -> bool:
        """The whole-object confidence plus min/max size gate, shared with the
        review so what streams in is what the review opens on."""
        from ..core.review_defaults import object_passes_review_gates
        return object_passes_review_gates(score, area, self._params)
