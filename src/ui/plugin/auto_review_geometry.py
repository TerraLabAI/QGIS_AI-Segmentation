"""The review's geometry work: ground-metre to CRS-unit conversions, the
per-object shape refine and its cache, and shared borders over the whole set.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
)

from ...core.live_refine import LiveRefiner, points_dial_fraction

# How many run refiners the review keeps alive at once. One covers a whole
# reslice; the spare room absorbs the Correct step's per-shape overrides, which
# hand this mixin a different params dict for one object inside a pass.
_LIVE_REFINER_MEMO_MAX = 8

# How many SUPERSEDED shape keys keep their refined geometry, and how many
# geometries all of them together may hold. Moving a Shape control and moving
# it back is the commonest gesture in the step, so the way back should cost
# nothing; the geometry budget is what stops a huge run from paying for that in
# memory instead.
_RESLICE_PARKED_KEYS_MAX = 2
_RESLICE_PARKED_GEOMS_MAX = 40000

# How long the safety-net export may spend shaping objects the Confidence
# cutoff was hiding. Those were never refined, so a dense review can owe
# thousands of GEOS passes at the exact moment the user is closing QGIS, on a
# path with no event loop to yield to. Windows calls a window that stops
# answering for a few seconds "not responding", and a user who force-quits
# there loses the whole rescue, so past this budget the remaining objects are
# saved with their traced outline instead. Losing the polish beats losing the
# detection.
_RESCUE_REFINE_BUDGET_S = 3.0


def _geom_centre_xy(geom) -> tuple[float, float]:
    """Where a per-CRS ground measure is taken for ONE object: the centre of
    its bounding box, or the CRS origin when it cannot be read."""
    try:
        centre = geom.boundingBox().center()
        return float(centre.x()), float(centre.y())
    except (AttributeError, RuntimeError, TypeError):
        return 0.0, 0.0


def _geoms_centre(geoms: list) -> tuple[float, float] | None:
    """Centre of the combined bounding box of ``geoms``, or None when it
    cannot be read. Used as the position a per-CRS ground measure is taken
    at, so the whole set converts on one factor."""
    box = None
    try:
        for geom in geoms:
            if geom is None or geom.isEmpty():
                continue
            bbox = geom.boundingBox()
            if box is None:
                box = bbox
            else:
                box.combineExtentWith(bbox)
        if box is None:
            return None
        centre = box.center()
        return float(centre.x()), float(centre.y())
    except (RuntimeError, AttributeError, TypeError):
        return None


class AutoReviewGeometryMixin:
    """Refine one object, cache the result, then snap the assembled set."""

    def _auto_crs_metres_per_unit(self, ref_x: float, ref_y: float) -> float:
        """Ground metres per unit of the RUN CRS, near (ref_x, ref_y).

        A run happens in the layer CRS, which is often Web Mercator, where one
        unit is well under a metre away from the equator (and a geographic CRS
        counts in degrees). Any setting expressed in ground metres has to cross
        that gap before it touches a geometry. Measured on the ellipsoid, so a
        projected metre CRS answers ~1.0 whatever its zone. Answers 1.0 on any
        failure, which reads the metre setting as CRS units: the behaviour
        before this conversion existed.

        Memoized per CRS and per coarse position: the factor moves with
        latitude, but not across one drawn zone.
        """
        authid = self._auto_crs_authid or "EPSG:4326"
        try:
            crs = QgsCoordinateReferenceSystem(authid)
            if not crs.isValid():
                return 1.0
            geographic = bool(crs.isGeographic())
        except Exception:  # noqa: BLE001 -- an unusable CRS means no conversion
            return 1.0
        # Degrees move fast, projected units do not: bucket each at the scale
        # where the factor is still flat (about 1 km / 10 km on the ground).
        bucket = round(ref_y, 2) if geographic else round(ref_y / 10000.0)
        key = (authid, bucket)
        cached = getattr(self, "_auto_crs_mpu", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        factor = 1.0
        try:
            from ...core.layer_conventions import make_area_measurer
            step = 0.001 if geographic else 1.0
            metres = float(make_area_measurer(crs).measureLine(
                QgsPointXY(ref_x, ref_y), QgsPointXY(ref_x + step, ref_y)))
            if metres > 0:
                factor = metres / step
        except Exception:  # noqa: BLE001 -- never block a refine on a measure
            factor = 1.0
        self._auto_crs_mpu = (key, factor)
        return factor

    def _auto_crs_unit_aspect(self, ref_x: float, ref_y: float) -> float:
        """How much longer one y unit of the RUN CRS is than one x unit, near
        (ref_x, ref_y). See core.layer_conventions.ground_unit_aspect: 1.0 in a
        projected CRS, above 1 in a geographic one, where squaring a footprint
        on raw coordinates leaves every corner tilted. Memoized like the ground
        measure above, and on the same coarse position."""
        authid = self._auto_crs_authid or "EPSG:4326"
        try:
            crs = QgsCoordinateReferenceSystem(authid)
            if not crs.isValid():
                return 1.0
            geographic = bool(crs.isGeographic())
        except Exception:  # noqa: BLE001 -- an unusable CRS means no correction
            return 1.0
        bucket = round(ref_y, 2) if geographic else round(ref_y / 10000.0)
        key = (authid, bucket)
        cached = getattr(self, "_auto_crs_aspect", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            from ...core.layer_conventions import ground_unit_aspect
            aspect = ground_unit_aspect(crs, ref_x, ref_y)
        except Exception:  # noqa: BLE001 -- never block a refine on a measure
            aspect = 1.0
        self._auto_crs_aspect = (key, aspect)
        return aspect

    def _refine_geom_for_review(self, base, params: dict, pixel_size: float):
        """Apply the review shape-refine controls to a base object geometry.
        simplify/expand are px in the UI, converted to ground units by the run's
        detection pixel size (_auto_refine_pixel_size). Returns a NEW geometry
        (base is never mutated).

        Every dial behind those controls is constant for a run, so they are all
        resolved once into a core.live_refine.LiveRefiner and reused: this
        method only picks the right refiner and hands it the object. The
        refiner carries no plugin state, so the same work can also run off the
        GUI thread.
        """
        # _review_shape_key covers exactly the params the refiner reads
        # (confidence and the size filters change no shape), so it doubles as
        # the refiner's identity; the run CRS joins it because the ground
        # measure below depends on it. Read off the class, not off self: the
        # tests bind this method onto bare stubs.
        key = (getattr(self, "_auto_crs_authid", None),
               AutoReviewGeometryMixin._review_shape_key(params, pixel_size))
        refiners = getattr(self, "_review_live_refiners", None)
        if not isinstance(refiners, dict):
            refiners = {}
            self._review_live_refiners = refiners
        refiner = refiners.get(key)
        if refiner is None:
            if len(refiners) >= _LIVE_REFINER_MEMO_MAX:
                refiners.clear()
            # The ground measure is taken at the FIRST object the refiner sees
            # and then reused for the run: _auto_crs_metres_per_unit is itself
            # memoized per coarse position, and one drawn zone sits inside one
            # bucket, so the factor was already the same for every object.
            ref_x, ref_y = _geom_centre_xy(base)
            refiner = LiveRefiner(
                params, pixel_size,
                self._auto_crs_metres_per_unit(ref_x, ref_y),
                self._auto_crs_unit_aspect(ref_x, ref_y))
            refiners[key] = refiner
        return refiner.refine(base)

    @staticmethod
    def _review_shape_key(params: dict, pixel_size: float) -> tuple:
        """Hashable identity of the SHAPE portion of the review params (+ the
        px scale they are applied at). Confidence and Min/Max size are filters,
        not shape ops, so they stay OUT: two reslices that differ only by a
        filter share every refined geometry through _auto_reslice_cache."""
        fill_holes = bool(params.get("fill_holes", False))
        return (
            round(float(params.get("simplify_px", 0) or 0.0), 4),
            bool(params.get("smooth", False)),
            round(float(params.get("expand_px", 0) or 0.0), 4),
            fill_holes,
            # Without this the reslice cache serves geometry refined at the old
            # threshold and the Fill-under spinbox reads as a dead control.
            # Reported as 0 while Fill holes is off, because the refine does not
            # read it then: a key that moved anyway threw away every refined
            # object for a change that cannot alter one.
            round(float(params.get("fill_max_m2", 0) or 0.0), 4) if fill_holes
            else 0.0,
            round(float(params.get("open_px", 0) or 0.0), 4),
            bool(params.get("ortho", False)),
            round(float(pixel_size or 0.0), 6),
            # The point budget changes every cached outline, so a run whose
            # spacing or Points dial differs must not be served the previous
            # geometry.
            round(float(params.get("vertex_spacing_m", 0) or 0.0), 4),
            round(points_dial_fraction(params), 4),
            # Shared borders is applied to the WHOLE set, after this cache, so
            # it changes no cached per-object geometry. It still belongs in the
            # key: the key is also the provenance stamp the incremental push
            # diffs on, and without it a toggle would leave the previous,
            # unsnapped shapes on the layer as "already current".
            bool(params.get("snap_boundaries", False)),
        )

    def _boundary_snap_offered(self) -> bool:
        """Whether this run may show the shared-borders control: a land-cover
        object, and few enough shapes for one pass. The prompt comes from the
        run context (the review's own prompt as a fallback), so a restored run
        answers the same way a fresh one does. Fail-closed on any error: the
        control simply does not appear."""
        try:
            from ...core.boundary_snap import boundary_snap_offered
            prompt = str((self._auto_run_ctx or {}).get("prompt") or "").strip()
            if not prompt:
                prompt = str((self._auto_review or {}).get("prompt") or "")
            return boundary_snap_offered(prompt, len(self._auto_objects))
        except Exception:  # noqa: BLE001 -- a gate failure hides the control
            return False

    def _boundary_snap_tolerance_units(self, geoms: list) -> float:
        """The shared-borders tolerance for this set, in RUN CRS UNITS.

        The dial is a ground distance (see core.boundary_snap) and a run often
        works in Web Mercator, where one unit is well under a metre: measure
        the metres one unit spans in the middle of the set, then convert. A set
        whose position cannot be read falls back to the metre value, which is
        exact in any metric projection.
        """
        from ...core.boundary_snap import boundary_snap_tolerance_units
        ref = _geoms_centre(geoms)
        if ref is None:
            return boundary_snap_tolerance_units(0.0)
        return boundary_snap_tolerance_units(
            self._auto_crs_metres_per_unit(ref[0], ref[1]))

    def _apply_boundary_snap(self, geoms: list, params: dict) -> list:
        """Give the VISIBLE set exact shared borders, once, as a whole set.

        Every other review control acts on ONE object, so it lives in the
        per-object refine (_refine_geom_for_review). Sharing a border needs
        every neighbour at once, so it runs here instead: on the assembled
        visible set, right before the review adopts it, which is also the set
        the Export commits.

        Off unless the user ticked the control, and the control itself is only
        offered on a land-cover run small enough for one pass. Returns the
        input list when anything is off, so a caller can always use the
        result; the length and the order always match the input, which is what
        keeps the parallel score and id lists aligned.
        """
        if not params.get("snap_boundaries"):
            return geoms
        if not isinstance(geoms, list) or len(geoms) < 2:
            return geoms
        from ...core.boundary_snap import boundary_snap_max_objects, snap_boundaries
        cap = boundary_snap_max_objects()
        if cap > 0 and len(geoms) > cap:
            # The control is not offered above the cap, so this is a safety
            # net (a set that grew after a batch fold): skip quietly.
            QgsMessageLog.logMessage(
                f"Auto review: shared borders skipped, {len(geoms)} shapes "
                f"over the {cap} limit",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            return geoms
        tolerance = self._boundary_snap_tolerance_units(geoms)
        if tolerance <= 0:
            return geoms
        import time as _t
        t0 = _t.monotonic()
        try:
            out = snap_boundaries(
                geoms, tolerance, crs=self._auto_crs_authid or None)
        except Exception as exc:  # noqa: BLE001 -- keep the unsnapped shapes
            # A whole-set GEOS failure here must degrade to "no shared
            # borders", never end the finalize/reslice chain that called it.
            QgsMessageLog.logMessage(
                f"Auto review: shared borders failed, keeping the unsnapped "
                f"shapes ({exc})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return geoms
        if not isinstance(out, list) or len(out) != len(geoms):
            return geoms
        took_ms = int((_t.monotonic() - t0) * 1000)
        # Log only a slow snap: this runs on every reslice while Shared borders
        # is on, so logging each cheap pass would spam the main-thread message
        # log during a confidence drag.
        if took_ms > 50:
            QgsMessageLog.logMessage(
                f"Auto review: shared borders over {len(geoms)} shape(s) "
                f"in {took_ms} ms",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
        return out

    def _reset_review_refine_cache(self) -> None:
        """Drop the reslice refine cache. Must run whenever _auto_objects is
        rebuilt or cleared (finalize build, exemplar re-group, export, discard,
        run start): the cache is keyed by object INDEX, so a stale entry after
        a rebuild would render the wrong object. The review provider map goes
        with it (det_ids from a rebuilt object set can collide with the old
        ones), which forces the next push to a full rebuild of the layer.

        The run refiners go with it: they hold the run's pixel size and its
        ground measure, and a rebuild can be the start of a different run. So
        do the parked shape keys, for the same index-collision reason as the
        current one."""
        self._auto_reslice_cache = {"key": None, "geoms": {}, "parked": {}}
        self._review_fid_map = {}
        self._review_live_refiners = {}

    def _invalidate_review_refine(self, indices) -> None:
        """Drop only ``indices`` from the reslice refine cache, keeping every
        other object's cached geometry, and force the next push to re-diff.

        A hand edit (merge / split) rewrites just a few rows and, by the index
        discipline in core.shape_edits, never shifts the others: a merge
        overwrites its target row, a split overwrites the target and APPENDS the
        extra pieces. So the untouched objects keep their cached shape instead
        of being re-refined, which matters because the right-angle squaring runs
        at a few ms per object and a full reset re-squared the whole set on
        every click.

        The provider diff map keeps every other entry: only the rewritten rows
        get their stamp voided, which is what makes the diff rewrite exactly
        their geometry. Clearing the whole map instead sent the next push down
        the full truncate + re-add path, rebuilding a review layer of tens of
        thousands of features after a single click (and silencing the
        progressive mid-reslice pushes, which need the diff). Voiding rather
        than dropping the entry matters: a dropped det_id reads as an ADD and
        would leave the pre-edit feature orphaned on the layer.

        Falls back to nothing when the cache is absent (the next reslice
        rebuilds it)."""
        cache = getattr(self, "_auto_reslice_cache", None)
        touched = [int(idx) for idx in indices]
        if isinstance(cache, dict):
            # The parked keys go with the current one. They exist so that
            # returning to a previous setting is free, and a hand edit rewrote
            # the object: without this, moving a control back would redraw the
            # shape the user just edited away.
            entries = [cache.get("geoms")]
            entries.extend((cache.get("parked") or {}).values())
            for geoms in entries:
                if isinstance(geoms, dict):
                    for idx in touched:
                        geoms.pop(idx, None)
        fid_map = getattr(self, "_review_fid_map", None)
        if not touched or not isinstance(fid_map, dict) or not fid_map:
            return
        for idx in touched:
            det_id = self._object_fid_for(idx)
            rec = fid_map.get(det_id)
            if rec is not None:
                # (provider_fid, stamp, is_full, score): a None stamp can never
                # equal the next push's, so the diff reports a geometry change.
                fid_map[det_id] = (rec[0], None, rec[2], rec[3])

    def _adopt_reslice_shape_key(self, cache: dict, key: tuple) -> None:
        """Make ``key`` the cache's current shape key, parking the outgoing one.

        A Shape control is used by moving it, looking, and moving it back or
        on. With one key held, every one of those steps re-ran the GEOS refine
        over the whole visible set, including the step that returns to a shape
        already computed a second earlier. Parking the superseded keys makes
        the way back a dict lookup.

        Bounded twice, because a review can hold tens of thousands of objects:
        by how many keys are parked, and by how many geometries they hold in
        total. Over the budget the oldest go first, so a big run degrades to
        today's behaviour instead of growing without limit.
        """
        parked = cache.get("parked")
        if not isinstance(parked, dict):
            parked = {}
            cache["parked"] = parked
        old_key, old_geoms = cache.get("key"), cache.get("geoms")
        if old_key is not None and old_geoms:
            # Re-inserting moves it to the end: dicts keep insertion order, so
            # "the oldest" below is the least recently current key.
            parked.pop(old_key, None)
            parked[old_key] = old_geoms
        revived = parked.pop(key, None)
        held = sum(len(g) for g in parked.values())
        while parked and (len(parked) > _RESLICE_PARKED_KEYS_MAX or held > _RESLICE_PARKED_GEOMS_MAX):
            held -= len(parked.pop(next(iter(parked))))
        cache["key"] = key
        cache["geoms"] = revived if revived is not None else {}

    @staticmethod
    def _plain_outline_geom(base):
        """``base`` repaired and coerced to MultiPolygon, with no shape control
        applied: the object exactly as the run traced it.

        The same normalize the refine cache applies at fill time, so an object
        that takes this path is byte-comparable with a refined one everywhere
        downstream. Used where a shaped outline is wanted but cannot be had:
        the budgeted rescue export, and a refine that failed or emptied
        (_review_refined_geom). Returns None when the repair empties it."""
        from ...core.layer_conventions import repair_polygon, to_multipolygon
        try:
            g = to_multipolygon(repair_polygon(base) or QgsGeometry(base))
        except Exception:  # noqa: BLE001 -- keep the outline as traced
            g = QgsGeometry(base)
        return g if (g is not None and not g.isEmpty()) else None

    def _review_refined_geom(self, det_idx: int, base, params: dict,
                             pixel_size: float):
        """Refined + normalized (repaired, MultiPolygon-coerced) geometry for
        one canonical object, memoized in _auto_reslice_cache. A cache hit is a
        dict lookup, so filter-only reslices (Confidence / Min / Max size) never
        re-run the GEOS refine on objects whose shape params did not change.

        A refine that fails or empties falls back to the object as the run
        traced it, and only a base that will not repair returns None. Shaping is
        cosmetic (right angles, simplify, fill holes) and the live preview drew
        the raw outline, so dropping the object instead made a paid detection
        disappear at the end of the run over a step that was never load-bearing.
        Right angles are on by default for built classes, which is exactly where
        the regularizer is most likely to empty a shape.

        Callers must not mutate the returned geometry (copy first)."""
        cache = self._auto_reslice_cache
        key = self._review_shape_key(params, pixel_size)
        if cache.get("key") != key:
            self._adopt_reslice_shape_key(cache, key)
        geoms = cache["geoms"]
        if det_idx in geoms:
            return geoms[det_idx]
        # No provenance check here, on purpose. Every edit the Correct step
        # makes (AI reshape, native vertex work, a hand-drawn add, a merge, a
        # split) becomes the object's canonical base, and the Shapes step reads
        # those bases. So one outline can never sit outside the settings the
        # user is moving, whatever produced it.
        # One object may carry its own Simplify / Right angles (the Correct
        # step's per-shape settings). Merged HERE so every consumer of the
        # visible set, the export included, sees the same shape; a change to an
        # override drops that object's cache entry, so the shared key stays a
        # whole-set identity.
        _per_shape = getattr(self, "_shape_params_for_object", None)
        if _per_shape is not None:
            params = _per_shape(det_idx, params)
        # Guarded: one pathological geometry (self-intersecting, NaN coords)
        # raising out of the GEOS refine/repair must cost only ITSELF, never
        # the finalize/reslice chain. The result is cached either way, so a
        # retried pass does not repeat the failure.
        try:
            g = self._refine_geom_for_review(base, params, pixel_size)
            if g is not None and not g.isEmpty():
                # Normalize ONCE at cache-fill time (repair + MultiPolygon
                # coerce), so every later push of this geometry skips both.
                from ...core.layer_conventions import repair_polygon, to_multipolygon
                g = to_multipolygon(repair_polygon(g) or g)
        except Exception as exc:  # noqa: BLE001 -- this object keeps its outline
            try:
                QgsMessageLog.logMessage(
                    f"Auto review: refine failed on object {det_idx}, "
                    f"kept as traced ({exc})",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            except Exception:  # nosec B110
                pass
            g = None
        if g is None or g.isEmpty():
            g = self._plain_outline_geom(base)
        result = g if (g is not None and not g.isEmpty()) else None
        geoms[det_idx] = result
        return result

    def _stitch_shapes_are_reusable(self, pixel_size: float,
                                    objects: list) -> bool:
        """Whether the shapes the live stitch thread built still describe THIS
        set, at the scale the finalize is about to refine it at.

        Three things can make them a different shape from what the review would
        compute, and all three are silent: the run's detection pixel size moves
        (the thread latches its rescale once, the finalize reads a running
        maximum), the ground metres-per-CRS-unit factor is resolved at a
        different place, and the object under a fid changes after the thread has
        stopped. The last one is flagged by its two causes; the first two are
        compared here. Any doubt returns False and the refine runs as before.
        """
        if getattr(self, "_auto_stitch_shapes_stale", True):
            return False
        stitch_px = float(getattr(self, "_auto_stitch_shape_px", 0.0) or 0.0)
        if abs(stitch_px - float(pixel_size or 0.0)) > 1e-9:
            return False
        factor = float(getattr(self, "_auto_stitch_shape_mpu", 0.0) or 0.0)
        if factor <= 0.0:
            return False
        # The review resolves its own factor from ONE object's position and
        # reuses it for the run, memoized per coarse position. Check the two
        # ends of the set: a zone that straddles a memo bucket is the case where
        # the review's single factor would not be the thread's.
        for geom, _score, _area in (objects[0], objects[-1]):
            try:
                centre = geom.centroid().asPoint()
                here = self._auto_crs_metres_per_unit(centre.x(), centre.y())
            except (RuntimeError, AttributeError, TypeError, ValueError):
                return False
            if abs(float(here) - factor) > factor * 1e-6:
                return False
        return True

    def _seed_review_refine_cache(self, params: dict, pixel_size: float,
                                  objects: list, object_fids: list) -> int:
        """Fill the refine cache with the shapes the live stitch thread already
        built, and return how many landed.

        That thread applies the run's shape preset to every object it settles,
        off the GUI thread, while the run is still going. The filter phase then
        ran the same refine over the same bases on the GUI thread with an empty
        cache, which is most of the arithmetic behind "Almost done" and all of
        it twice.

        Object by object, so one that cannot be matched costs only itself:
        anything not seeded falls through to the refine exactly as before, and
        the worst case is the behaviour this replaces.
        """
        shapes = getattr(self, "_auto_stitch_shapes", None)
        if not shapes or not objects or len(object_fids) != len(objects):
            return 0
        if not self._stitch_shapes_are_reusable(pixel_size, objects):
            return 0
        from ...core.layer_conventions import repair_polygon, to_multipolygon
        cache = self._auto_reslice_cache
        key = self._review_shape_key(params, pixel_size)
        if cache.get("key") != key:
            self._adopt_reslice_shape_key(cache, key)
        geoms = cache["geoms"]
        per_shape = getattr(self, "_shape_params_for_object", None)
        seeded = 0
        for det_idx, fid in enumerate(object_fids):
            if det_idx in geoms:
                continue
            shape = shapes.get(fid)
            if shape is None:
                continue
            # An outline carrying its own Shape settings is not what the run-wide
            # preset produced. It cannot exist on a fresh finalize, which is the
            # only caller today, so this is a guard for a future one.
            if per_shape is not None and per_shape(det_idx, params) is not params:
                continue
            try:
                # The same normalize the refine path applies at cache-fill time.
                # The thread coerces to MultiPolygon but does not repair, so
                # without this a seeded object and a refined one would not be
                # byte-identical.
                g = to_multipolygon(repair_polygon(shape) or shape)
            except Exception:  # noqa: BLE001 -- fall through to the refine  # nosec B112
                continue
            if g is None or g.isEmpty():
                continue
            geoms[det_idx] = g
            seeded += 1
        return seeded

    def _compute_visible_objects(
        self, params: dict, pixel_size: float, with_scores: bool = False,
        refine_budget_s: float = 0.0,
    ) -> list | tuple[list, list]:
        """Synchronous filter+refine of _auto_objects into the visible geom set:
        whole-object confidence + size filter, then the shape refine. Used by the
        headless path; the interactive path does the same cooperatively.
        ``with_scores`` also returns the parallel per-object score list, so the
        headless export can fill the layer's `score` field like the review path
        does (it silently exported NULL scores before).

        ``refine_budget_s`` bounds the time spent shaping objects that are NOT
        already in the reslice cache (see _RESCUE_REFINE_BUDGET_S). A cache hit
        is a dict lookup and never touches the budget, so the set the user was
        actually looking at always comes out fully shaped; only the cohort the
        Confidence cutoff had hidden can fall back to its traced outline. 0
        means no bound, which is what the headless path wants."""
        import time as _t

        removed = self._review_removed_fids()
        # Settle the cache's shape key BEFORE the loop so the peek below reads
        # the dict that _review_refined_geom will fill, not one it is about to
        # swap out.
        cache = self._auto_reslice_cache
        shape_key = self._review_shape_key(params, pixel_size)
        if cache.get("key") != shape_key:
            self._adopt_reslice_shape_key(cache, shape_key)
        cached_geoms = cache["geoms"]
        deadline = (_t.monotonic() + refine_budget_s) if refine_budget_s > 0 else None
        unshaped = 0
        unrepairable = 0
        out = []
        out_scores = []
        for det_idx, (base, score, area) in enumerate(self._auto_objects):
            if det_idx in removed or base is None or base.isEmpty():
                continue
            if not (self._object_is_manual(det_idx) or self._passes_review_filters(score, area, params)):
                continue
            if deadline is not None and det_idx not in cached_geoms and _t.monotonic() >= deadline:
                g = self._plain_outline_geom(base)
                unshaped += 1
            else:
                g = self._review_refined_geom(det_idx, base, params, pixel_size)
            if g is not None:
                out.append(g)
                out_scores.append(float(score))
            else:
                unrepairable += 1
        if unshaped:
            QgsMessageLog.logMessage(
                f"Auto review: rescue export ran out of its {_RESCUE_REFINE_BUDGET_S:.0f}s "
                f"shaping budget; {unshaped} hidden object(s) saved with their "
                f"traced outline", "AI Segmentation", level=Qgis.MessageLevel.Info)
        if unrepairable:
            # A base that will not repair is the one case that loses a billed
            # object, so it is counted and said rather than dropped in silence.
            QgsMessageLog.logMessage(
                f"Auto review: {unrepairable} object(s) left out; their geometry "
                f"could not be repaired", "AI Segmentation",
                level=Qgis.MessageLevel.Warning)
        # The set is complete here, which is the only point a whole-set
        # operation can run: shared borders needs every neighbour at once.
        out = self._apply_boundary_snap(out, params)
        if with_scores:
            return out, out_scores
        return out
