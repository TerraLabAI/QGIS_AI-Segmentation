







from __future__ import annotations

from ...core.i18n import tr
from .shared import clip_served_hint




_DETECT_PLAN_WAIT_MS = 3000


class AutoFlowRunPlanMixin:




    def _active_run_plan(self, prompt: str) -> dict | None:



        rp = getattr(self, "_auto_run_plan", None)
        if not isinstance(rp, dict):
            return None
        if (rp.get("prompt") or "").strip().lower() != (prompt or "").strip().lower():
            return None
        plan = rp.get("plan")
        return plan if isinstance(plan, dict) else None

    def _fetch_auto_run_plan(self, prompt: str) -> None:






        prompt = (prompt or "").strip()


        if self._auto_worker is not None or self._auto_review is not None:
            return
        if getattr(self, "_auto_imagery_probe", None) is not None:
            return




        rewritten_from = self._plan_rewritten_from(prompt)


        self._auto_run_plan = None
        self._auto_attribute_filters = []
        self._cancel_task("_auto_run_plan_task")
        if not self.dock_widget:
            return



        exemplar_size_m = self._exemplar_size_for_plan()
        if not prompt and exemplar_size_m is None:
            return
        from ...core.activation_manager import get_auth_header, is_plugin_activated
        if not is_plugin_activated():
            return
        auth = get_auth_header()
        if not auth:
            return
        zone_area_m2, native_mupp = self._auto_run_plan_inputs()
        try:
            from qgis.core import QgsApplication

            from ...api.terralab_client import TerraLabClient
            from ...workers.generic_request_task import GenericRequestTask
            client = TerraLabClient()
            task = GenericRequestTask(
                tr("Planning AI Segmentation run"),
                lambda: client.get_seg_run_plan(
                    prompt, zone_area_m2, native_mupp, auth=auth,
                    exemplar_size_m=exemplar_size_m,
                    rewritten_from=rewritten_from),
                hidden=True,
            )
            task.succeeded.connect(
                lambda plan, p=prompt, x=exemplar_size_m, rf=rewritten_from:
                    self._on_auto_run_plan_ready(
                        p, plan, exemplar_size_m=x, rewritten_from=rf))
            task.failed.connect(
                lambda *_a, p=prompt: self._on_auto_run_plan_failed(p))
            self._auto_run_plan_task = task
            self._auto_run_plan_task_prompt = prompt
            QgsApplication.taskManager().addTask(task)
        except Exception:  # noqa: BLE001
            self._auto_run_plan_task = None  # nosec B110

    def _plan_rewritten_from(self, prompt: str) -> str | None:


        rp = getattr(self, "_auto_run_plan", None)
        if not isinstance(rp, dict):
            return None
        source = rp.get("rewritten_from")
        if not isinstance(source, str) or not source.strip():
            return None
        if (rp.get("prompt") or "").strip().lower() != (prompt or "").strip().lower():
            return None
        return source.strip()

    def _exemplar_size_for_plan(self) -> float | None:



        if self._auto_zone is None:
            return None
        try:
            layer = self._get_active_raster_layer()
            if layer is None:
                return None
            zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
            size = float(self._exemplar_object_size_m(layer, zone_in_layer))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return None
        return size if size > 0 else None

    def _run_plan_from_exemplar(self) -> bool:


        rp = getattr(self, "_auto_run_plan", None)
        return isinstance(rp, dict) and rp.get("exemplar_size_m") is not None

    def _auto_run_plan_inputs(self) -> tuple[float | None, float | None]:




        zone_area_m2: float | None = None
        native_mupp: float | None = None
        try:
            layer = self._get_active_raster_layer()
        except (RuntimeError, AttributeError):
            layer = None
        if layer is None:
            return zone_area_m2, native_mupp
        zone_in_layer = None
        if self._auto_zone is not None:
            try:
                zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
            except (RuntimeError, AttributeError):
                zone_in_layer = None
        if zone_in_layer is not None:
            try:
                from qgis.core import QgsGeometry

                from ...core.layer_conventions import make_area_measurer



                da = make_area_measurer(self._run_crs_now(layer) or layer.crs())
                area = da.measureArea(QgsGeometry.fromRect(zone_in_layer))
                if area and area > 0:
                    zone_area_m2 = float(area)
            except Exception:  # noqa: BLE001  # nosec B110
                zone_area_m2 = None
        try:
            layer_w = layer.width()
            layer_h = layer.height()
            ext = layer.extent()




            ref = zone_in_layer if zone_in_layer is not None else ext
            to_run_x, to_run_y = self._layer_units_to_run_units(layer, ref)
            if self._needs_canvas_render(layer) or layer_w <= 0 or layer_h <= 0:
                native_units = self._online_native_mupp(layer) * max(to_run_x, to_run_y)
            elif ext.width() > 0 and ext.height() > 0:
                native_units = max(ext.width() / layer_w * to_run_x,
                                   ext.height() / layer_h * to_run_y)
            else:
                native_units = 0.0
            if native_units and native_units > 0:
                meters = self._mupp_to_meters(layer, ref, native_units)
                if meters and meters > 0:
                    native_mupp = float(meters)
        except (RuntimeError, AttributeError, ValueError):
            native_mupp = None
        return zone_area_m2, native_mupp

    def _on_auto_run_plan_ready(
        self, prompt: str, plan: object, exemplar_size_m: float | None = None,
        rewritten_from: str | None = None,
    ) -> None:








        self._auto_run_plan_task = None
        self._auto_run_plan_task_prompt = ""
        try:
            self._store_auto_run_plan(
                prompt, plan, exemplar_size_m, rewritten_from=rewritten_from)
        finally:
            self._resume_detect_after_plan(prompt)

    def _store_auto_run_plan(
        self, prompt: str, plan: object, exemplar_size_m: float | None,
        rewritten_from: str | None = None,
    ) -> None:

        if not isinstance(plan, dict) or plan.get("error"):
            return




        from ...core.run_eta import own_pace_seconds_per_tile
        own_pace = own_pace_seconds_per_tile(plan)
        if own_pace is not None and self.dock_widget is not None:
            try:
                self.dock_widget.set_auto_own_pace(own_pace)
            except (RuntimeError, AttributeError):
                pass



        if self._auto_worker is not None or self._auto_review is not None:
            return
        prompt = (prompt or "").strip()
        if not prompt and exemplar_size_m is None:
            return
        if prompt.lower() != self._resolved_auto_object_class().strip().lower():
            return
        self._auto_run_plan = {
            "prompt": prompt, "plan": plan, "exemplar_size_m": exemplar_size_m}


        if rewritten_from:
            self._auto_run_plan["rewritten_from"] = rewritten_from
        self._reseed_auto_detail_from_plan(prompt, plan)



        if self._apply_prompt_rewrite(prompt, plan):
            return




        hint = plan.get("hint")
        if isinstance(hint, str) and self.dock_widget is not None:
            hint = clip_served_hint(hint)
            if hint:
                try:
                    if self.dock_widget.show_auto_prompt_hint(hint):
                        from ...core import telemetry_run_events
                        telemetry_run_events.track_auto_prompt_hint_shown(
                            kind="plan_hint", prompt=prompt)
                except Exception:  # noqa: BLE001
                    pass  # nosec B110

    def _apply_prompt_rewrite(self, prompt: str, plan: dict) -> bool:












        from ...core.prompt_rewrite import parse_prompt_rewrite

        action, payload, filters = parse_prompt_rewrite(plan.get("prompt_rewrite"))


        self._auto_attribute_filters = filters
        dock = self.dock_widget
        if dock is None:
            return False
        if action == "rewrite":



            try:
                swapped = bool(dock.apply_prompt_swap(payload, "server_rewrite"))
            except (RuntimeError, AttributeError):
                return False
            if swapped:



                rp = getattr(self, "_auto_run_plan", None)
                if isinstance(rp, dict) and rp.get("plan") is plan:
                    self._auto_run_plan = dict(
                        rp, prompt=payload, rewritten_from=prompt)
            return swapped
        if action == "decline":


            try:
                return bool(dock.show_auto_prompt_decline(payload))
            except (RuntimeError, AttributeError):
                return False
        return False

    def _on_auto_run_plan_failed(self, prompt: str = "") -> None:


        self._auto_run_plan_task = None
        self._auto_run_plan_task_prompt = ""
        self._resume_detect_after_plan(prompt)

    def _headless_prompt_rewrite(self, object_class: str, plan: dict) -> str:







        from ...core.prompt_rewrite import parse_prompt_rewrite

        object_class = (object_class or "").strip()
        if not object_class:
            return ""
        action, payload, filters = parse_prompt_rewrite(plan.get("prompt_rewrite"))
        if action != "rewrite" or payload.lower() == object_class.lower():
            return ""
        dock = self.dock_widget
        if dock is None:
            return ""
        try:
            if not dock.apply_prompt_swap(payload, "server_rewrite"):
                return ""
        except (RuntimeError, AttributeError):
            return ""
        self._auto_attribute_filters = filters
        rp = getattr(self, "_auto_run_plan", None)
        if isinstance(rp, dict):
            self._auto_run_plan = dict(
                rp, prompt=payload, rewritten_from=object_class)
        return payload



    def _run_plan_gate_at_detect(self) -> bool:










        resumed = bool(getattr(self, "_auto_plan_detect_resumed", False))
        wait = getattr(self, "_auto_plan_detect_wait", None)
        if wait is not None and not resumed:
            if self._detect_plan_wait_matches_box(wait):
                return False
            self._drop_detect_plan_wait()
        dock = self.dock_widget
        prompt = self._current_auto_object_class()
        if dock is None or not prompt:
            return True
        token = self._resolve_object_token(prompt)
        plan = self._active_run_plan(token)
        if plan is not None:
            try:
                self._apply_prompt_rewrite(token, plan)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            return True
        if resumed:
            return True
        return not self._hold_detect_for_plan(prompt, token)

    def _hold_detect_for_plan(self, prompt: str, token: str) -> bool:


        in_flight = (
            getattr(self, "_auto_run_plan_task", None) is not None
            and (getattr(self, "_auto_run_plan_task_prompt", "") or "").lower()
            == token.lower())
        if not in_flight:
            self._fetch_auto_run_plan(token)
            in_flight = getattr(self, "_auto_run_plan_task", None) is not None
        if not in_flight:
            return False
        generation = int(getattr(self, "_auto_plan_detect_generation", 0)) + 1
        self._auto_plan_detect_generation = generation
        self._auto_plan_detect_wait = {
            "generation": generation, "box": prompt, "token": token}
        dock = self.dock_widget
        try:
            from ...core.qt_compat import safe_single_shot
            try:
                from ...core.server_dials import dial_in_range
                plan_wait_ms = dial_in_range(
                    "tuning.auto.detect_plan_wait_ms", _DETECT_PLAN_WAIT_MS, 500, 10000)
            except Exception:  # noqa: BLE001
                plan_wait_ms = _DETECT_PLAN_WAIT_MS
            safe_single_shot(
                plan_wait_ms, dock,
                lambda g=generation: self._resume_detect_after_plan(
                    "", generation=g))
            dock._set_prompt_info(dock._prompt_lookup_note(), tip=True)
            dock._set_prompt_lookup_busy(True)
        except (RuntimeError, AttributeError, ImportError):

            self._drop_detect_plan_wait()
            return False
        return True

    def _detect_plan_wait_matches_box(self, wait: dict) -> bool:


        box = self._current_auto_object_class().lower()
        if box == (wait.get("box") or "").lower():
            return True
        rp = getattr(self, "_auto_run_plan", None)
        return (isinstance(rp, dict)
                and (rp.get("rewritten_from") or "").lower()
                == (wait.get("token") or "").lower()
                and box == (rp.get("prompt") or "").lower())

    def _drop_detect_plan_wait(self) -> None:


        if getattr(self, "_auto_plan_detect_wait", None) is None:
            return
        self._auto_plan_detect_wait = None
        dock = self.dock_widget
        if dock is None:
            return
        try:
            dock._set_prompt_lookup_busy(False)
            if getattr(dock, "_prompt_info_kind", None) == "tip":
                dock._set_prompt_info()
        except (RuntimeError, AttributeError):
            pass

    def _resume_detect_after_plan(self, prompt: str, generation: int = 0) -> None:




        wait = getattr(self, "_auto_plan_detect_wait", None)
        if not isinstance(wait, dict):
            return
        if generation:
            if generation != wait.get("generation"):
                return
        elif (prompt or "").strip().lower() != (wait.get("token") or "").lower():
            return
        matches = self._detect_plan_wait_matches_box(wait)
        self._drop_detect_plan_wait()
        if not matches or self.dock_widget is None:
            return
        if self._auto_worker is not None or self._auto_review is not None:
            return
        self._auto_plan_detect_resumed = True
        try:
            self._on_auto_detect_requested()
        finally:
            self._auto_plan_detect_resumed = False



    def _fetch_auto_token(self, raw: str) -> bool:












        raw = (raw or "").strip()
        if not raw or not self.dock_widget:
            return False
        if self._auto_worker is not None or self._auto_review is not None:
            return False
        try:
            from ..dock.prompt_guard import server_lookup_wanted, validate_prompt
            if not server_lookup_wanted(raw, *validate_prompt(raw)):
                return False
        except Exception:  # noqa: BLE001  # nosec B110
            pass

        self._cancel_task("_auto_token_task")
        try:
            from qgis.core import QgsApplication

            from ...api.prompt_translation import resolve_english_prompt
            from ...workers.generic_request_task import GenericRequestTask
            task = GenericRequestTask(
                tr("Resolving object name"),
                lambda: {"token": resolve_english_prompt(raw)},
                hidden=True,
            )
            task.succeeded.connect(
                lambda res, r=raw: self._on_auto_token_ready(r, res))
            task.failed.connect(lambda *_a, r=raw: self._on_auto_token_failed(r))
            self._auto_token_task = task
            QgsApplication.taskManager().addTask(task)
        except Exception:  # noqa: BLE001
            self._auto_token_task = None  # nosec B110
            return False
        return True

    def _on_auto_token_ready(self, raw: str, result: object) -> None:




        self._auto_token_task = None
        if self._auto_worker is not None or self._auto_review is not None:
            return
        raw = (raw or "").strip()
        if not raw or raw.lower() != self._current_auto_object_class().strip().lower():
            return
        token = result.get("token") if isinstance(result, dict) else None
        try:
            from ..dock.prompt_guard import vet_server_token
            token = vet_server_token(token if isinstance(token, str) else None)
        except Exception:  # noqa: BLE001
            token = token if isinstance(token, str) else None
        if not token or token.strip().lower() == raw.lower():


            self._fetch_auto_run_plan(self._resolve_object_token(raw))
            return
        cache = getattr(self, "_auto_token_cache", None)
        if cache is None:
            cache = {}
            self._auto_token_cache = cache
        cache[raw.lower()] = token

        self._reseed_auto_detail_from_blob(token)
        self._fetch_auto_run_plan(token)

    def _on_auto_token_failed(self, raw: str = "") -> None:


        self._auto_token_task = None
        raw = (raw or "").strip()
        if raw and raw.lower() == self._current_auto_object_class().strip().lower():
            self._fetch_auto_run_plan(self._resolve_object_token(raw))
