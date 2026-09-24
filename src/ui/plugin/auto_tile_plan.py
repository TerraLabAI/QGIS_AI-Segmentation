














from __future__ import annotations

from ...core.tile_manager import TILE_SIZE



_FLOOR_PROBE_GROUND_M = 0.01


class AutoTilePlanMixin:


    def _tile_plan_active(self) -> bool:

        try:
            from ...core.detection_policy import tile_plan_enabled
            return tile_plan_enabled()
        except Exception:  # noqa: BLE001
            return False

    def _tile_plan_metres_per_unit(self, layer, zone_in_layer) -> float:

        side = max(zone_in_layer.width(), zone_in_layer.height())
        if side <= 0:
            return 0.0
        ref = side / TILE_SIZE
        metres = self._mupp_to_meters(layer, zone_in_layer, ref)
        return metres / ref if metres > 0 else 0.0

    def _tile_plan_source_floor_m(self, layer, zone_in_layer, per_unit: float) -> float:


        if per_unit <= 0:
            return 0.0
        ref = _FLOOR_PROBE_GROUND_M / per_unit
        sized = self._grid_for_run_mupp(layer, zone_in_layer, ref)
        if sized is None or sized[2] <= ref * 1.0001:
            return 0.0
        return float(sized[2]) * per_unit

    def _tile_plan_prior(self, layer, zone_in_layer):



        from ...core.detection_policy import (
            exemplar_band_enabled,
            object_min_px,
            object_profile,
            object_tile_ceiling_m,
            object_tile_floor_m,
            run_plan_tile_block,
            tile_fit_object_frac,
            zone_seed_mupp,
        )
        from ...core.tile_plan import (
            REASON_PRIOR_DEFAULT,
            REASON_PRIOR_EXEMPLAR,
            REASON_PRIOR_PLAN,
            REASON_PRIOR_TIER,
            TilePrior,
        )
        from .auto_detail_window import _positive_number

        token = self._resolved_auto_object_class()
        default_m = zone_seed_mupp() * TILE_SIZE
        frac = tile_fit_object_frac()
        size_m, floor_m = self._detail_window_profile(token, layer, zone_in_layer)
        ceiling_m = object_tile_ceiling_m(token) if token else 0.0


        drawn_m = 0.0
        if exemplar_band_enabled():
            try:
                drawn_m = float(self._exemplar_object_size_m(layer, zone_in_layer))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                drawn_m = 0.0
        drawn = {"drawn_m": drawn_m, "drawn_min_px": float(object_min_px()),
                 "drawn_only": not token} if drawn_m > 0 else {}
        plan = (self._active_run_plan(token)
                if (token or self._run_plan_from_exemplar()) else None)
        if plan is not None:
            block = run_plan_tile_block(plan)
            tile_m = _positive_number(block.get("tile_ground_m"))
            if not tile_m:
                tile_m = _positive_number(plan.get("target_mupp")) * TILE_SIZE
            if tile_m:
                band = block.get("band_m")
                band_pair = None
                if isinstance(band, (list, tuple)) and len(band) == 2:
                    band_pair = (_positive_number(band[0]), _positive_number(band[1]))
                route = _positive_number(block.get("route_floor_m")) or floor_m
                size = _positive_number(block.get("size_m")) or size_m
                ceiling = _positive_number(block.get("ceiling_m")) or ceiling_m
                return TilePrior(
                    tile_ground_m=tile_m, source=REASON_PRIOR_PLAN,
                    band_m=band_pair, size_m=size, fit_frac=frac,
                    route_floor_m=route, ceiling_m=ceiling,
                    default_tile_ground_m=default_m, **drawn)
        if token:
            _obj_m, target_mupp = object_profile(token)
            return TilePrior(
                tile_ground_m=target_mupp * TILE_SIZE, source=REASON_PRIOR_TIER,
                size_m=size_m, fit_frac=frac,
                route_floor_m=object_tile_floor_m(token), ceiling_m=ceiling_m,
                default_tile_ground_m=default_m, **drawn)
        exemplar_mupp = self._exemplar_seed_target_mupp(layer, zone_in_layer)
        if exemplar_mupp > 0:
            return TilePrior(
                tile_ground_m=exemplar_mupp * TILE_SIZE,
                source=REASON_PRIOR_EXEMPLAR, size_m=0.0, fit_frac=frac,
                default_tile_ground_m=default_m, **drawn)
        return TilePrior(
            tile_ground_m=default_m, source=REASON_PRIOR_DEFAULT,
            fit_frac=frac, default_tile_ground_m=default_m)

    def _tile_plan_count_at(self, layer, zone_in_layer, side_m: float,
                            per_unit: float, cap: int) -> int:



        if per_unit <= 0 or side_m <= 0:
            return 0
        mupp = side_m / TILE_SIZE / per_unit
        sized = self._grid_for_run_mupp(layer, zone_in_layer, mupp)
        if sized is None:
            return 0
        pixel_w, pixel_h, grid_mupp, bbox_tiles = sized
        if 0 <= bbox_tiles <= cap:
            return bbox_tiles
        grid = self._tile_manager.compute_grid(pixel_w, pixel_h, apply_cap=False)
        if not grid:
            return bbox_tiles if bbox_tiles >= 0 else 10 ** 9
        minx, maxy = self._tile_plan_grid_origin(
            zone_in_layer, pixel_w, pixel_h, grid_mupp)
        bbox = (minx, maxy - pixel_h * grid_mupp,
                minx + pixel_w * grid_mupp, maxy)
        kept = self._tiles_in_polygon(
            grid, bbox, pixel_w, pixel_h, layer, self._grid_crs_authid(layer))
        return len(kept) if kept else bbox_tiles

    def _tile_plan_now(self, layer, zone_in_layer, density_side_m: float = 0.0):






        from ...core.detection_policy import (
            get_detection_policy,
            tile_plan_half_steps,
            tile_plan_step_ratio,
        )
        from ...core.tile_plan import (
            TileCaps,
            TileDensity,
            TileSource,
            TileZone,
            resolve_tile_ground,
        )

        if density_side_m <= 0:
            density_side_m = self._density_forced_side_m()

        if self._tile_manager is None:
            self._setup_auto_mode()
        per_unit = self._tile_plan_metres_per_unit(layer, zone_in_layer)
        if per_unit <= 0:
            return None
        prior = self._tile_plan_prior(layer, zone_in_layer)
        machine_cap = int(self._auto_zone_tile_cap())
        seed_cap = min(machine_cap, int(self._seed_tile_cap_for_plan()))
        try:
            key = (
                layer.id(), id(get_detection_policy()), prior, machine_cap,
                seed_cap, round(per_unit, 9), round(density_side_m, 3),
                tile_plan_half_steps(), tile_plan_step_ratio(),
                zone_in_layer.xMinimum(), zone_in_layer.yMinimum(),
                zone_in_layer.xMaximum(), zone_in_layer.yMaximum(),
            )
        except (RuntimeError, AttributeError):
            key = None
        cached = getattr(self, "_tile_plan_cache", None)
        if key is not None and cached is not None and cached[0] == key:
            return cached[1]

        floor_m = self._tile_plan_source_floor_m(layer, zone_in_layer, per_unit)
        counts: dict = {}

        def count(side_m: float, cap: int = seed_cap) -> int:
            memo_key = (round(side_m, 3), cap)
            if memo_key not in counts:
                counts[memo_key] = self._tile_plan_count_at(
                    layer, zone_in_layer, side_m, per_unit, cap)
            return counts[memo_key]

        plan = resolve_tile_ground(
            prior,
            TileSource(native_m=floor_m, allowance=1.0),
            TileZone(width_m=zone_in_layer.width() * per_unit,
                     height_m=zone_in_layer.height() * per_unit,
                     count_tiles=count),
            TileCaps(max_tiles=seed_cap),
            TileDensity(tile_ground_m=density_side_m) if density_side_m > 0 else None,
        )
        window = self._tile_plan_window_for(plan, machine_cap, count)
        value = (plan, window)
        if key is not None:
            self._tile_plan_cache = (key, value)
        return value

    @staticmethod
    def _tile_plan_window_for(plan, machine_cap: int, count) -> tuple[int, int, int]:

        from ...core.detection_policy import tile_plan_half_steps, tile_plan_step_ratio
        from ...core.tile_plan import tile_step_window

        return tile_step_window(
            plan, machine_cap, lambda side: count(side, machine_cap),
            tile_plan_half_steps(), tile_plan_step_ratio())

    def _tile_plan_centre_step(self) -> int:

        from ...core.detection_policy import tile_plan_half_steps
        return tile_plan_half_steps() + 1

    def _tile_plan_step_run_mupp(self, layer, zone_in_layer, step: int) -> float:


        from ...core.detection_policy import tile_plan_half_steps, tile_plan_step_ratio
        from ...core.tile_plan import tile_step_ground_m

        try:
            resolved = self._tile_plan_now(layer, zone_in_layer)
            if resolved is None:
                return 0.0
            per_unit = self._tile_plan_metres_per_unit(layer, zone_in_layer)
            if per_unit <= 0:
                return 0.0
            side_m = tile_step_ground_m(
                resolved[0], step, tile_plan_half_steps(), tile_plan_step_ratio())
            return side_m / TILE_SIZE / per_unit
        except (RuntimeError, AttributeError, TypeError, ValueError,
                ZeroDivisionError):
            return 0.0

    def _tile_plan_window(self, layer, zone_in_layer) -> tuple[int, int, int] | None:

        try:
            resolved = self._tile_plan_now(layer, zone_in_layer)
        except (RuntimeError, AttributeError, TypeError, ValueError,
                ZeroDivisionError):
            return None
        return None if resolved is None else resolved[1]

    def _tile_plan_warning_now(self, layer, zone_in_layer, rendered_m: float) -> str:


        from ...core.detection_policy import tile_plan_step_ratio
        from ...core.tile_plan import TILE_WARNING_NONE, tile_plan_warning

        try:
            resolved = self._tile_plan_now(layer, zone_in_layer)
        except (RuntimeError, AttributeError, TypeError, ValueError,
                ZeroDivisionError):
            return TILE_WARNING_NONE
        if resolved is None:
            return TILE_WARNING_NONE
        return tile_plan_warning(resolved[0], rendered_m, tile_plan_step_ratio())

    def _tile_plan_top_reason(self, layer, zone_in_layer) -> str:



        from ...core.detection_policy import tile_plan_half_steps, tile_plan_step_ratio
        from ...core.tile_plan import REASON_CAP_TILES, tile_step_ground_m

        try:
            resolved = self._tile_plan_now(layer, zone_in_layer)
        except (RuntimeError, AttributeError, TypeError, ValueError,
                ZeroDivisionError):
            return ""
        if resolved is None:
            return ""
        plan, window = resolved
        half = tile_plan_half_steps()
        machine = window[2]
        if machine >= 2 * half + 1:
            return ""
        if window[1] < machine:
            return ""
        side = tile_step_ground_m(plan, machine + 1, half, tile_plan_step_ratio())
        if plan.floor_m and side < plan.floor_m * 0.9999:
            return "cap" if REASON_CAP_TILES in plan.reasons else ""
        if side < plan.band_m[0] * 0.9999:
            return ""
        return "cap"

    @staticmethod
    def _tile_plan_grid_origin(zone_in_layer, pixel_w: int, pixel_h: int,
                               mupp: float) -> tuple[float, float]:



        minx = zone_in_layer.xMinimum()
        maxy = zone_in_layer.yMaximum()
        grid_w = pixel_w * mupp
        grid_h = pixel_h * mupp
        if grid_w > zone_in_layer.width():
            minx -= (grid_w - zone_in_layer.width()) / 2.0
        if grid_h > zone_in_layer.height():
            maxy += (grid_h - zone_in_layer.height()) / 2.0
        return minx, maxy

    def _tile_plan_run_props(self, layer, zone_in_layer, gsd_m: float) -> dict:



        if not self._tile_plan_active():
            return {"tile_plan": False}
        props: dict = {"tile_plan": True}
        if gsd_m > 0:
            props["tile_ground_m"] = int(round(TILE_SIZE * gsd_m))
        resolved = None
        if zone_in_layer is not None:
            try:
                resolved = self._tile_plan_now(layer, zone_in_layer)
            except (RuntimeError, AttributeError, TypeError, ValueError,
                    ZeroDivisionError):
                resolved = None
        if resolved is not None:
            plan = resolved[0]
            props["tile_prior_m"] = int(round(plan.prior_m))
            reasons = list(plan.reasons)
            step = self._get_auto_detail_level()
            if step != self._tile_plan_centre_step():
                reasons.append("user_step")
            if gsd_m > 0 and TILE_SIZE * gsd_m > plan.tile_ground_m * 1.05 \
                    and step == self._tile_plan_centre_step():
                reasons.append("imagery_floor")
            props["tile_reasons"] = ",".join(reasons)

            from ...core.detection_policy import tile_plan_step_ratio
            from ...core.tile_plan import tile_plan_warning
            props["tile_warning"] = tile_plan_warning(
                plan, TILE_SIZE * gsd_m if gsd_m > 0 else 0.0,
                tile_plan_step_ratio())
            try:
                from qgis.core import Qgis, QgsMessageLog
                QgsMessageLog.logMessage(
                    f"Auto detection: tile plan T={plan.tile_ground_m:.0f} m "
                    f"(rendered {props.get('tile_ground_m', 0)} m, step {step}, "
                    f"prior {plan.prior_m:.0f} m, tiles {plan.tiles}) "
                    f"reasons={props['tile_reasons']}",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            except (ImportError, RuntimeError):
                pass  # nosec B110
        return props
