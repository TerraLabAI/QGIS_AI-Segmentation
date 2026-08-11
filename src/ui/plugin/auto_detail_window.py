"""How far the Precision slider may travel for the object the user named.

The control is labelled "Precision" on screen and ``detail`` in code (see
``ui/dock/auto_detail_level.py``). Its travel used to be a machine bound only:
every level from 1 up to whatever the tile cap and the source's native
resolution allowed, the same range for a car and for a farm parcel.

That range lies to the user. One end of a slider labelled Precision reads as
"better", so a user chasing a cleaner result drags it up. For a small object
that works. For a large one it does the opposite: each tile is a separate look
at the ground, and once the tile stops being much wider than the object, the
model is judging a parcel it can only see a corner of. The run comes back in
fragments, and it cost more credits than the good one would have.

So the travel is now the band that is worth offering for THIS object:

- the fine end stops where the object starts taking up too much of a tile for
  the model to still see it whole,
- the coarse end stops where the object stops being big enough to spot,
- and the band always contains the level the plugin itself recommends, so a
  class whose objects are large simply has no travel past that level.

Both ends are computed from the shared ``_grid_for_detail`` math, so they
cannot disagree with the credit estimate, the grid preview or the seed.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); methods are plain
mixin members and state lives on the instance.
"""
from __future__ import annotations


def _positive_number(value: object) -> float:
    """``value`` as a float when it is a real positive number, else 0.0.

    Server payloads reach here unvalidated, and ``bool`` is an ``int`` in
    Python, so ``True`` would otherwise pass as the number 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    return value if value > 0 and value == value and value != float("inf") else 0.0


class AutoDetailWindowMixin:
    """The Precision slider's object-aware travel: its two ends and their memo."""

    def _detail_window_for_object(
        self, layer, zone_in_layer, object_class: str
    ) -> tuple[int, int]:
        """(coarsest, finest) Precision level worth offering for this object.

        Falls back to the plain machine bound (1 .. `_max_useful_detail`) when
        nothing says how big the object is: no word the tables carry and no
        example drawn. That is the range the slider has always had.

        Memoised on the inputs that move it, because the credit estimate
        re-asks on every debounced slider tick and each walk costs an
        ellipsoidal ground measurement per level.
        """
        machine_max = self._max_useful_detail(layer, zone_in_layer)
        obj = (object_class or "").strip()

        obj_m, floor_m = self._detail_window_profile(obj, layer, zone_in_layer)
        if not obj and obj_m <= 0:
            return 1, machine_max
        try:
            key = (
                layer.id(), obj.lower(), machine_max, obj_m, floor_m,
                self._free_run_tile_cap(),
                zone_in_layer.xMinimum(), zone_in_layer.yMinimum(),
                zone_in_layer.xMaximum(), zone_in_layer.yMaximum(),
            )
        except (RuntimeError, AttributeError):
            key = None
        cached = getattr(self, "_detail_window_cache", None)
        if key is not None and cached is not None and cached[0] == key:
            return cached[1]

        try:
            window = self._walk_detail_window(
                layer, zone_in_layer, obj, obj_m, floor_m, machine_max)
        except (RuntimeError, AttributeError, ValueError, ZeroDivisionError):
            return 1, machine_max
        if key is not None:
            self._detail_window_cache = (key, window)
        return window

    def _detail_window_profile(
        self, object_class: str, layer=None, zone_in_layer=None
    ) -> tuple[float, float]:
        """(object ground size m, minimum tile ground side m) for the window.

        The server run plan wins over the shipped tier table when one was
        fetched for this exact prompt, exactly like the seed appliers, so the
        travel can never contradict the level the seed picked inside it.

        A drawn example then RAISES that size and never lowers it. It is a
        measurement of the thing the user is actually after, taken off their own
        imagery, so it covers what no word list can: a spelling or a language
        the tables do not carry, and a run with no text at all. Raising only,
        because a word that resolved to a class carries a size for the whole
        class, which one drawing has no standing to contradict downwards.
        """
        from ...core.detection_policy import object_profile, object_tile_floor_m

        obj_m = 0.0
        floor_m = 0.0
        if object_class:
            obj_m, _target_mupp = object_profile(object_class)
            floor_m = object_tile_floor_m(object_class)
            plan = self._active_run_plan(object_class)
            plan_obj_m = _positive_number(plan.get("object_size_m")) if plan else 0.0
            if plan_obj_m:
                # The floor is the guard-rail on the size, so the two come from
                # ONE source or neither does. The plan carries the floor of the
                # same tier it took the size from, and omits it when that tier
                # names none, so an absent key means no floor rather than "ask
                # the blob": pairing a served size with a floor from another
                # reader is exactly what object_tile_floor_m forbids.
                obj_m = plan_obj_m
                floor_m = _positive_number(plan.get("min_tile_ground_m"))
        if layer is not None and zone_in_layer is not None:
            obj_m = max(obj_m, self._exemplar_object_size_m(layer, zone_in_layer))
        return obj_m, floor_m

    def _exemplar_object_size_m(self, layer, zone_in_layer) -> float:
        """Ground size (m) of the examples drawn on the canvas, 0.0 with none.

        One example measures as its longer side, so an object traced at an
        angle counts for what it spans rather than for how tidy its outline is.
        Across several, the median: one loose draw beside careful ones must not
        decide the band on its own.

        Exclude marks and correction regions are skipped. Neither is a drawing
        of the object, and an exclude is often drawn around something bigger.
        """
        from ...core.exemplar_store import LABEL_POSITIVE

        store = getattr(self, "_auto_exemplar_store", None)
        if store is None:
            return 0.0
        sides: list[float] = []
        for ex in store.list():
            if ex.label != LABEL_POSITIVE or ex.region:
                continue
            try:
                rect = self._reproject_zone_to_run_crs(ex.map_rect, layer)
                side = max(rect.width(), rect.height())
            except (RuntimeError, AttributeError):
                continue
            if side <= 0:
                continue
            # Same conversion the grid math uses, so a drawn size and a tile
            # size are measured by one ruler and cannot disagree.
            ground = self._mupp_to_meters(layer, zone_in_layer, side)
            if ground > 0:
                sides.append(ground)
        if not sides:
            return 0.0
        sides.sort()
        mid = len(sides) // 2
        if len(sides) % 2:
            return sides[mid]
        return (sides[mid - 1] + sides[mid]) / 2.0

    def _walk_detail_window(
        self, layer, zone_in_layer, object_class: str,
        obj_m: float, floor_m: float, machine_max: int,
    ) -> tuple[int, int]:
        """The coarse -> fine walk behind `_detail_window_for_object`.

        One pass collects all three levels the two ends are built from: the
        finest level whose tile still holds the object with room around it, the
        coarsest level at which the object still spans the minimum pixels, and
        the finest level a free run can still afford.
        """
        from ...core.detection_policy import (
            detail_max_object_tile_frac,
            object_min_px,
        )
        from ...core.tile_manager import TILE_SIZE

        # The fine end is the share of a tile an object may take before the
        # model starts judging it from a fragment. NOT the split-risk line the
        # amber warning draws: that one marks where the pieces stop being
        # stitchable, which is later, and stopping the slider there would leave
        # the whole stretch in between, where the run is already paying more
        # credits for a worse answer.
        frac = detail_max_object_tile_frac()
        if obj_m > 0 and 0 < frac <= 1:
            floor_m = max(floor_m, obj_m / frac)
        min_px = object_min_px()
        free_cap = self._free_run_tile_cap()

        finest = 0
        coarsest = 0
        affordable = 0
        for n in range(1, machine_max + 1):
            sized = self._grid_for_detail(layer, zone_in_layer, n)
            if sized is None:
                break
            _pixel_w, _pixel_h, mupp, tiles = sized
            if tiles == -1:
                break
            ground_mupp = self._mupp_to_meters(layer, zone_in_layer, mupp)
            if ground_mupp <= 0:
                break
            if free_cap is None or tiles <= free_cap:
                affordable = n
            if not coarsest and obj_m > 0 and obj_m / ground_mupp >= min_px:
                coarsest = n
            if floor_m > 0 and TILE_SIZE * ground_mupp < floor_m:
                break
            finest = n

        recommended = self._recommended_detail_now(
            layer, zone_in_layer, object_class)
        # The band must contain the plugin's own pick. A small zone can be
        # narrower than the object's tile floor, so the fine end computed above
        # comes back below the recommendation (or at 0); a slider that stops
        # under the level the plugin recommends is broken, whatever the object.
        finest = min(max(1, finest, recommended), machine_max)
        # Never past the recommendation: raising the COARSE end raises what the
        # cheapest run costs, and the seed is the most a default may spend.
        coarsest = min(coarsest or 1, recommended, finest)
        # And never past what a free run covers, so lowering Precision always
        # stays a way out of the premium gate rather than a dead end.
        if free_cap is not None and affordable >= 1:
            coarsest = min(coarsest, affordable)
        return max(1, coarsest), finest
