







from __future__ import annotations


class AutoFlowDetailMixin:




    def _get_auto_detail_level(self) -> int:

        if self.dock_widget is None:
            return 1
        try:
            return max(1, int(self.dock_widget.auto_detail_slider.value()))
        except (RuntimeError, AttributeError):
            return 1

    def _free_run_tile_cap(self) -> int | None:













        from ...core.credit_gate import free_run_tile_cap
        from ...core.detection_policy import free_run_fraction

        usage = self._last_usage or {}
        if not usage.get("is_free_tier", True):
            return None



        return free_run_tile_cap(usage.get("free_detections_total"), free_run_fraction())

    def _seed_tile_cap_for_plan(self) -> int:


        from ...core.detection_policy import seed_tile_cap

        cap = seed_tile_cap()
        free_cap = self._free_run_tile_cap()
        return cap if free_cap is None else min(cap, free_cap)

    def _default_detail_for_zone(self, layer, zone_in_layer) -> int:








        from ...core.detection_policy import zone_seed_mupp

        return self._finest_level_reaching(
            layer, zone_in_layer, zone_seed_mupp())

    def _default_detail_from_examples(self, layer, zone_in_layer) -> int:













        target_mupp = self._exemplar_seed_target_mupp(layer, zone_in_layer)
        if target_mupp <= 0:
            return self._default_detail_for_zone(layer, zone_in_layer)
        return self._finest_level_reaching(layer, zone_in_layer, target_mupp)

    def _finest_level_reaching(
        self, layer, zone_in_layer, target_mupp: float, floor_m: float = 0.0,
    ) -> int:


































        from ...core.detection_policy import seed_headroom_levels
        from ...core.tile_manager import TILE_SIZE

        cap = self._max_useful_detail(layer, zone_in_layer)
        ceiling = max(1, cap - seed_headroom_levels())
        tile_cap = self._seed_tile_cap_for_plan()
        best = 1
        for n in range(1, ceiling + 1):
            sized = self._grid_for_detail(layer, zone_in_layer, n)
            if sized is None:
                break
            _pw, _ph, mupp, tiles = sized
            if tiles != -1:
                tiles = self._tiles_after_cull_confirmed(
                    layer, zone_in_layer, n, tiles, tile_cap)
            if tiles == -1 or tiles > tile_cap:
                break
            ground_mupp = self._mupp_to_meters(layer, zone_in_layer, mupp)
            if ground_mupp <= 0:
                continue
            if floor_m > 0 and TILE_SIZE * ground_mupp < floor_m:
                break
            best = n
            if ground_mupp <= target_mupp:
                return n
        return best

    def _object_detail_profile(self, object_class: str) -> tuple[float, float]:







        from ...core.detection_policy import object_profile

        return object_profile(object_class)

    def _auto_detail_for_object(self, layer, zone_in_layer, object_class) -> int:














        try:
            obj_m, target_mupp = self._object_detail_profile(object_class)
            return self._auto_detail_for_target(
                layer, zone_in_layer, obj_m, target_mupp,
                self._detail_window_profile(object_class)[1])
        except (RuntimeError, AttributeError, ValueError):
            return self._default_detail_for_zone(layer, zone_in_layer)

    def _auto_detail_for_target(
        self, layer, zone_in_layer, obj_m: float, target_mupp: float,
        floor_m: float = 0.0,
    ) -> int:











        return self._finest_level_reaching(
            layer, zone_in_layer, target_mupp, floor_m)

    def _current_auto_object_class(self) -> str:

        if self.dock_widget is None:
            return ""
        try:
            return self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            return ""

    def _resolve_object_token(self, raw: str) -> str:










        raw = (raw or "").strip()
        if not raw:
            return ""
        cache = getattr(self, "_auto_token_cache", None)
        if cache is None:
            cache = {}
            self._auto_token_cache = cache
        key = raw.lower()
        if key in cache:
            return cache[key]
        try:
            from ..dock.prompt_guard import resolve_object_token
            token = resolve_object_token(raw)
        except Exception:  # noqa: BLE001
            token = raw
        cache[key] = token
        return token

    def _resolved_auto_object_class(self) -> str:



        return self._resolve_object_token(self._current_auto_object_class())

    def _seed_auto_detail_value(self, detail: int) -> None:







        self.dock_widget.set_auto_detail_value(detail)
        self._auto_detail_seeded = self._get_auto_detail_level()

    def _apply_default_detail(self, zone_rect) -> None:









        self._auto_detail_user_locked = False
        self._auto_detail_lock_prompt = ""


        self._auto_detail_seeded = None
        if not self.dock_widget:
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            zone_in_layer = self._reproject_zone_to_run_crs(zone_rect, layer)
            object_class = self._resolved_auto_object_class()
            if object_class:
                detail = self._auto_detail_for_object(
                    layer, zone_in_layer, object_class)
            else:
                detail = self._default_detail_from_examples(
                    layer, zone_in_layer)
            self._seed_auto_detail_value(detail)
        except (RuntimeError, AttributeError):
            pass

    def _reseed_auto_detail_for_object(self, object_class: str = "") -> None:










        raw = (object_class or "").strip()
        token = self._resolve_object_token(raw)
        self._reseed_auto_detail_from_blob(token)
        self._fetch_auto_run_plan(token)



        if raw and token == raw:
            self._fetch_auto_token(raw)






        self._maybe_warmup_auto()


        self._refresh_rerun_guard()

    def _reseed_auto_detail_from_blob(self, object_class: str = "") -> None:

















        if not self.dock_widget or self._auto_zone is None:
            return

        if self._auto_worker is not None or self._auto_review is not None:
            return
        object_class = (object_class or "").strip()
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
        except (RuntimeError, AttributeError):
            return


        target_mupp = 0.0
        if not object_class:
            target_mupp = self._exemplar_seed_target_mupp(layer, zone_in_layer)
            if target_mupp <= 0:
                return
        if self._auto_detail_user_locked:
            locked_for = getattr(self, "_auto_detail_lock_prompt", "")
            if object_class.lower() == locked_for:
                return
            self._auto_detail_user_locked = False
            self._auto_detail_lock_prompt = ""
        try:
            if object_class:
                detail = self._auto_detail_for_object(
                    layer, zone_in_layer, object_class)
            else:
                detail = self._finest_level_reaching(
                    layer, zone_in_layer, target_mupp)
            self._seed_auto_detail_value(detail)
            self._update_credit_estimate()
        except (RuntimeError, AttributeError):
            pass

    def _reseed_auto_detail_from_plan(self, prompt: str, plan: dict) -> None:




        if not self.dock_widget or self._auto_zone is None:
            return
        if self._auto_worker is not None or self._auto_review is not None:
            return
        prompt = (prompt or "").strip()


        if not prompt and not self._run_plan_from_exemplar():
            return
        if prompt.lower() != self._resolved_auto_object_class().strip().lower():
            return
        if self._auto_detail_user_locked and getattr(self, "_auto_detail_lock_prompt", "") == prompt.lower():
            return
        target_mupp = plan.get("target_mupp")
        if not isinstance(target_mupp, (int, float)) or isinstance(target_mupp, bool) or target_mupp <= 0:
            return
        obj_m = plan.get("object_size_m")
        is_valid_obj_m = isinstance(obj_m, (int, float)) and not isinstance(obj_m, bool) and obj_m > 0
        obj_m = float(obj_m) if is_valid_obj_m else 10.0
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)



            detail = self._auto_detail_for_target(
                layer, zone_in_layer, obj_m, float(target_mupp),
                self._detail_window_profile(prompt)[1])
            self._seed_auto_detail_value(detail)
            self._update_credit_estimate()
        except (RuntimeError, AttributeError):
            pass

    def _push_detail_feedback(self, layer, zone_in_layer, ground_mupp: float) -> None:

















        if self.dock_widget is None:
            return
        if ground_mupp <= 0:
            self.dock_widget.set_auto_detail_feedback(None, "")
            return
        obj = self._current_auto_object_class()



        obj_token = self._resolve_object_token(obj)
        has_reference = False
        try:
            has_reference = self._auto_exemplar_store.count() > 0
        except (RuntimeError, AttributeError):
            pass
        if not obj and not has_reference:

            self.dock_widget.set_auto_detail_feedback(None, "")
            return
        try:
            from ...core.detection_policy import object_min_px

            obj_m, target_mupp = self._object_detail_profile(obj_token)
            value = self._get_auto_detail_level()
            recommended = self._recommended_detail_now(layer, zone_in_layer, obj_token)
            target_met = ground_mupp <= target_mupp
            if obj_m / ground_mupp < object_min_px():
                state = "coarse"
            elif value > recommended:
                if self._detail_splits_objects(
                        layer, zone_in_layer, ground_mupp, obj_m,
                        target_mupp, recommended):
                    state = "over"
                else:


                    state = "above" if target_met else "helps"
            elif value < recommended:
                state = "below"
            else:
                state = "recommended"
            self.dock_widget.set_auto_detail_feedback(state, obj)
        except (RuntimeError, AttributeError, ValueError, ZeroDivisionError):

            pass

    def _detail_splits_objects(
        self, layer, zone_in_layer, ground_mupp: float, obj_m: float,
        target_mupp: float, recommended: int,
    ) -> bool:














        from ...core.detection_policy import (
            detail_over_ratio,
            detail_over_ratio_free,
            split_risk_tile_frac,
        )
        from ...core.tile_manager import TILE_SIZE

        if ground_mupp <= 0 or obj_m <= 0:
            return False
        if obj_m < split_risk_tile_frac() * TILE_SIZE * ground_mupp:
            return False

        is_subscriber = bool(getattr(
            self.dock_widget, "_auto_is_subscriber", False))
        over_ratio = (detail_over_ratio() if is_subscriber
                      else detail_over_ratio_free())
        rec_mupp = self._ground_mupp_for_detail(
            layer, zone_in_layer, recommended)



        anchor = min(rec_mupp, target_mupp) if rec_mupp > 0 else target_mupp
        return ground_mupp < anchor * over_ratio

    def _recommended_detail_now(self, layer, zone_in_layer, object_class: str) -> int:






        plan = self._active_run_plan(object_class) if object_class else None
        if plan:
            target_mupp = plan.get("target_mupp")
            if isinstance(target_mupp, (int, float)) and not isinstance(target_mupp, bool) and target_mupp > 0:
                obj_m = plan.get("object_size_m")
                is_valid_obj_m = isinstance(obj_m, (int, float)) and not isinstance(obj_m, bool) and obj_m > 0
                obj_m = float(obj_m) if is_valid_obj_m else 10.0
                return self._auto_detail_for_target(
                    layer, zone_in_layer, obj_m, float(target_mupp),
                    self._detail_window_profile(object_class)[1])
        if object_class:
            return self._auto_detail_for_object(layer, zone_in_layer, object_class)
        return self._default_detail_for_zone(layer, zone_in_layer)
