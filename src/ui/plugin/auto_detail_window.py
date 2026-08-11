





























from __future__ import annotations

import math


def _positive_number(value: object) -> float:





    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    return value if value > 0 and not math.isnan(value) and value != float("inf") else 0.0


class AutoDetailWindowMixin:


    def _detail_window_for_object(
        self, layer, zone_in_layer, object_class: str
    ) -> tuple[int, int]:










        from ...core.detection_policy import (
            detail_coarse_travel_ratio,
            detail_fine_travel_ratio,
            seed_headroom_levels,
        )

        machine_max = self._max_useful_detail(layer, zone_in_layer)
        obj = (object_class or "").strip()

        obj_m, floor_m = self._detail_window_profile(obj, layer, zone_in_layer)
        if not obj and obj_m <= 0:
            return 1, machine_max
        try:
            key = (
                layer.id(), obj.lower(), machine_max, obj_m, floor_m,
                self._free_run_tile_cap(), seed_headroom_levels(),
                detail_coarse_travel_ratio(), detail_fine_travel_ratio(),
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













        from ...core.detection_policy import object_profile, object_tile_floor_m

        obj_m = 0.0
        floor_m = 0.0
        if object_class:
            obj_m, _target_mupp = object_profile(object_class)
            floor_m = object_tile_floor_m(object_class)


        plan = self._active_run_plan(object_class) if (object_class or self._run_plan_from_exemplar()) else None
        if plan is not None:
            plan_obj_m = _positive_number(plan.get("object_size_m"))
            if plan_obj_m:






                obj_m = plan_obj_m
                floor_m = _positive_number(plan.get("min_tile_ground_m"))
        if layer is not None and zone_in_layer is not None:
            obj_m = max(obj_m, self._exemplar_object_size_m(layer, zone_in_layer))
        return obj_m, floor_m

    def _exemplar_object_size_m(self, layer, zone_in_layer) -> float:










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

    def _exemplar_seed_target_mupp(self, layer, zone_in_layer) -> float:
















        from ...core.detection_policy import drawn_object_tile_frac, zone_seed_mupp
        from ...core.tile_manager import TILE_SIZE

        try:
            drawn_m = float(self._exemplar_object_size_m(layer, zone_in_layer))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return 0.0
        if drawn_m <= 0:
            return 0.0
        try:
            frac = drawn_object_tile_frac()
            floor_tile_m = zone_seed_mupp() * TILE_SIZE
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return 0.0
        if frac <= 0 or floor_tile_m <= 0:
            return 0.0
        return max(floor_tile_m, drawn_m / frac) / TILE_SIZE

    def _walk_detail_window(
        self, layer, zone_in_layer, object_class: str,
        obj_m: float, floor_m: float, machine_max: int,
    ) -> tuple[int, int]:







        from ...core.detection_policy import (
            detail_coarse_travel_ratio,
            detail_fine_travel_ratio,
            object_min_px,
            object_tile_ceiling_m,
            seed_headroom_levels,
        )
        from ...core.tile_manager import TILE_SIZE







        recommended = self._recommended_detail_now(
            layer, zone_in_layer, object_class)
        rec_tile_m = 0.0
        sized = self._grid_for_detail(layer, zone_in_layer, recommended)
        if sized is not None and sized[3] != -1:
            rec_tile_m = TILE_SIZE * self._mupp_to_meters(
                layer, zone_in_layer, sized[2])
        coarse_m = rec_tile_m * detail_coarse_travel_ratio()
        if rec_tile_m > 0:
            floor_m = max(floor_m, rec_tile_m / detail_fine_travel_ratio())
        min_px = object_min_px()



        ceiling_m = object_tile_ceiling_m(object_class) if object_class else 0.0
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
            tile_m = TILE_SIZE * ground_mupp
            if free_cap is None or tiles <= free_cap:
                affordable = n
            over_ceiling = ceiling_m > 0 and tile_m > ceiling_m
            px_ok = obj_m <= 0 or obj_m / ground_mupp >= min_px
            travel_ok = coarse_m <= 0 or tile_m <= coarse_m
            if (not coarsest and not over_ceiling and px_ok and travel_ok
                    and (obj_m > 0 or coarse_m > 0)):
                coarsest = n
            if floor_m > 0 and tile_m < floor_m:
                break
            finest = n





        finest = min(max(1, finest, recommended + seed_headroom_levels()),
                     machine_max)


        coarsest = min(coarsest or 1, recommended, finest)


        if free_cap is not None and affordable >= 1:
            coarsest = min(coarsest, affordable)
        return max(1, coarsest), finest
