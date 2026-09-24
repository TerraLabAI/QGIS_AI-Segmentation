








from __future__ import annotations

from typing import Callable

from qgis.core import (
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)

from ...core.interaction_dials import cancel_watchdog_ms
from .auto_run_cancel import (
    _CANCEL_WATCHDOG_MS,
)





_HEADLESS_CANCEL_POLL_MS = 250




_HEADLESS_CANCEL_GRACE_MS = _CANCEL_WATCHDOG_MS * 3


def _headless_cancel_grace_ms() -> float:


    return cancel_watchdog_ms(_CANCEL_WATCHDOG_MS) * 3


def _headless_cancel_poll_ms() -> int:


    from ...core.server_dials import dial_in_range
    return dial_in_range(
        "tuning.auto.headless_cancel_poll_ms", _HEADLESS_CANCEL_POLL_MS, 50, 2000)


class AutoRunHeadlessMixin:


    def _store_auto_zone_from_geometry(self, geom, active_layer) -> QgsRectangle:









        from ...core.qt_compat import PolygonGeometry, geometry_op_succeeded

        shape = QgsGeometry(geom)


        zone_crs = None
        if active_layer is not None:
            try:
                layer_crs = active_layer.crs()
                canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
                if (layer_crs.isValid() and canvas_crs.isValid()
                        and layer_crs != canvas_crs):
                    xform = QgsCoordinateTransform(
                        layer_crs, canvas_crs, QgsProject.instance())
                    if not geometry_op_succeeded(shape.transform(xform)):
                        raise ValueError("zone transform failed")
            except Exception:  # noqa: BLE001


                shape = QgsGeometry(geom)
                zone_crs = active_layer.crs()
        rect = QgsRectangle(shape.boundingBox())
        self._auto_zone_polygon = None
        if not shape.isEmpty() and shape.type() == PolygonGeometry:


            from ...core.layer_conventions import repair_polygon
            repaired = repair_polygon(QgsGeometry(shape))
            self._auto_zone_polygon = (
                repaired if repaired is not None and not repaired.isEmpty()
                else shape)




        self._store_auto_zone(rect, crs=zone_crs)
        return rect

    def _arm_headless_cancel_poll(self, loop, should_cancel, state: dict):









        state["asked"] = False
        state["deadline"] = 0.0
        if should_cancel is None:
            return None

        import time as _t

        from qgis.PyQt.QtCore import QTimer

        timer = QTimer(self.dock_widget)
        timer.setInterval(_headless_cancel_poll_ms())

        def _tick():
            if state["asked"]:
                if (self._last_auto_result is not None
                        or _t.monotonic() >= state["deadline"]):
                    loop.quit()
                return
            try:
                wants_stop = bool(should_cancel())
            except Exception:  # noqa: BLE001

                timer.stop()
                return
            if not wants_stop:
                return
            state["asked"] = True
            state["deadline"] = _t.monotonic() + _headless_cancel_grace_ms() / 1000.0
            self._on_auto_cancel_clicked()

        timer.timeout.connect(_tick)
        timer.start()
        return timer

    @staticmethod
    def _disarm_headless_cancel_poll(timer) -> None:





        if timer is None:
            return
        try:
            timer.stop()
            timer.timeout.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            timer.deleteLater()
        except RuntimeError:
            pass

    @staticmethod
    def _resolve_headless_raster(name_or_id: str):








        project = QgsProject.instance()
        by_id = project.mapLayer(name_or_id)
        if isinstance(by_id, QgsRasterLayer):
            return by_id, False
        matches = [
            lyr for lyr in project.mapLayers().values()
            if isinstance(lyr, QgsRasterLayer) and lyr.name() == name_or_id
        ]
        if len(matches) > 1:
            return None, True
        return (matches[0] if matches else None), False

    def _restore_dock_after_headless(self, mode_before, shown_layer_id: str) -> None:








        if shown_layer_id:
            try:
                node = QgsProject.instance().layerTreeRoot().findLayer(shown_layer_id)
                if node is not None:
                    node.setItemVisibilityChecked(False)
            except (RuntimeError, AttributeError):
                pass
        if mode_before is None:
            return
        worker = getattr(self, "_auto_worker", None)
        if worker is not None and worker.isRunning():
            return
        if getattr(self, "_auto_review", None) is not None:
            return
        try:
            dock = self.dock_widget
            if dock is not None and dock._mode != mode_before:
                dock._on_mode_selected(mode_before)
        except (RuntimeError, AttributeError):
            pass

    @staticmethod
    def _headless_english_token(object_class: str) -> str:






        try:
            from ..dock.prompt_guard import (
                server_lookup_wanted,
                validate_prompt,
                vet_server_token,
            )

            ok, reason, suggestion = validate_prompt(object_class)
            if server_lookup_wanted(object_class, ok, reason, suggestion):
                from ...api.prompt_translation import resolve_english_prompt

                token = vet_server_token(resolve_english_prompt(object_class))
                if token:
                    return token
            if ok and reason in ("translated", "plural", "alias") and suggestion:
                return suggestion
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return ""

    def _run_auto_detect_headless(
        self,
        zone_wkt: str,
        object_class: str,
        layer_name: str | None = None,
        timeout_s: int = 280,
        exemplars: list[dict] | None = None,
        detail: int | None = None,
        confidence: float | None = None,
        refine: dict | None = None,
        should_cancel: Callable[[], bool] | None = None,
        instance_colors: bool = False,
        wait: bool = True,
    ) -> dict:























































        from qgis.PyQt.QtCore import QEventLoop, QTimer

        from ..ai_segmentation_dockwidget import Mode




        live = getattr(self, "_auto_worker", None)
        if ((live is not None and live.isRunning())
                or getattr(self, "_auto_finalize_state", None) is not None
                or getattr(self, "_auto_start_in_progress", False)):
            return {
                "_error": (
                    "A zone detection is already running. Poll it with "
                    "auto_detect_status(), or stop it with cancel_auto(), "
                    "before starting another."
                ),
                "busy": True,
            }
        if getattr(self, "_auto_review", None) is not None:
            return {
                "_error": "Export or exit the current detection review before starting another run.",
                "busy": True,
            }


        self._ensure_dock_widget()
        if self._tile_manager is None:
            self._setup_auto_mode()




        mode_before = None
        shown_layer_id = ""



        hard_stopped = False






        try:
            try:
                dock = self.dock_widget
                if dock and dock._mode != Mode.AUTOMATIC:
                    mode_before = dock._mode
                    if dock._on_mode_selected(Mode.AUTOMATIC) is False:
                        return {"_error": "The panel refused to switch to automatic mode."}
            except (RuntimeError, AttributeError):
                pass


            if layer_name:
                target_layer, ambiguous = self._resolve_headless_raster(layer_name)
                if ambiguous:
                    return {"_error": (
                        f"More than one raster layer is called '{layer_name}'. "
                        "Pass the layer id instead, so the run reads the one you "
                        "mean.")}
                if target_layer is None:




                    from ...mcp_api import (
                        LAYER_NAME_ARGUMENT_NOTE,
                        not_found_error,
                    )
                    available = sorted(
                        lyr.name()
                        for lyr in QgsProject.instance().mapLayers().values()
                        if isinstance(lyr, QgsRasterLayer)
                    )
                    return not_found_error(
                        "raster layer", layer_name, available,
                        note=LAYER_NAME_ARGUMENT_NOTE,
                    )


                try:
                    if self.dock_widget and hasattr(self.dock_widget, "layer_combo"):
                        self.dock_widget.layer_combo.setLayer(target_layer)
                    if self.dock_widget and hasattr(self.dock_widget, "auto_layer_combo"):
                        self.dock_widget.auto_layer_combo.setLayer(target_layer)
                except (RuntimeError, AttributeError):
                    pass






                active = self._get_active_raster_layer()
                if active is None or active.id() != target_layer.id():
                    try:
                        node = QgsProject.instance().layerTreeRoot().findLayer(
                            target_layer.id())
                        if node is not None and not node.itemVisibilityChecked():
                            node.setItemVisibilityChecked(True)
                            shown_layer_id = target_layer.id()
                        for combo_name in ("layer_combo", "auto_layer_combo"):
                            combo = getattr(self.dock_widget, combo_name, None)
                            if combo is None:
                                continue
                            refresh = getattr(combo, "_refresh", None)
                            if callable(refresh):
                                refresh()
                            combo.setLayer(target_layer)
                    except (RuntimeError, AttributeError):
                        pass
                    active = self._get_active_raster_layer()
                    if active is None or active.id() != target_layer.id():
                        return {"_error": (
                            f"Raster layer '{layer_name}' exists but could not be selected "
                            "(hidden in the layer tree or filtered out). Make it "
                            "visible and retry.")}


            if zone_wkt and zone_wkt.strip():
                geom = QgsGeometry.fromWkt(zone_wkt)
                if geom is None or geom.isEmpty():
                    return {"_error": "Invalid zone WKT"}



                active_layer = self._get_active_raster_layer()





                zone_crs = active_layer.crs() if active_layer is not None else None
                free_fit = self._fit_zone_to_free_budget(geom, crs=zone_crs)
                if free_fit is not None:
                    if free_fit.geom is None:
                        try:
                            from ...core import telemetry_run_events
                            telemetry_run_events.track_auto_zone_too_large(
                                area_km2=free_fit.requested_km2)
                        except Exception:
                            pass  # nosec B110
                        from .shared import zone_over_free_cap_message
                        return {"_error": zone_over_free_cap_message(
                            free_fit.requested_km2)}
                    geom = QgsGeometry(free_fit.geom)


                self._store_auto_zone_from_geometry(geom, active_layer)
                if free_fit is not None:
                    self._record_free_zone_fit(free_fit, notify=False)
                try:
                    if self.dock_widget:
                        self.dock_widget.set_auto_zone_state("zone_set")
                except (RuntimeError, AttributeError):
                    pass
            else:
                self._store_auto_zone(None)
                self._auto_zone_polygon = None
                try:
                    if self.dock_widget:
                        self.dock_widget.set_auto_zone_state("idle")
                except (RuntimeError, AttributeError):
                    pass




            translated_from = ""
            if object_class:
                english = self._headless_english_token(object_class)
                if english and english.lower() != object_class.lower():
                    translated_from, object_class = object_class, english


            try:
                if self.dock_widget:
                    self.dock_widget.set_prompt_text(object_class)
            except (RuntimeError, AttributeError):
                pass





            self._clear_exemplars()
            if exemplars:


                self._sync_exemplar_store_tier()
                ex_layer = self._get_active_raster_layer()
                ex_xform = None
                if ex_layer is not None:
                    try:
                        l_crs = ex_layer.crs()
                        c_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
                        if l_crs.isValid() and c_crs.isValid() and l_crs != c_crs:
                            ex_xform = QgsCoordinateTransform(l_crs, c_crs, QgsProject.instance())
                    except (RuntimeError, AttributeError):
                        ex_xform = None
                for ex in exemplars:
                    try:
                        bb = ex.get("bbox") or ex.get("box")
                        if not bb or len(bb) < 4:
                            continue
                        rect = QgsRectangle(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                        if ex_xform is not None:
                            rect = ex_xform.transformBoundingBox(rect)
                        self._auto_exemplar_store.add(rect, int(ex.get("label", 1)))
                    except (RuntimeError, AttributeError, ValueError, TypeError):
                        continue
                self._refresh_exemplar_chips()







            if detail is not None:
                try:
                    if self.dock_widget:
                        self.dock_widget.set_auto_detail_value(max(1, int(detail)))


                    self._auto_detail_seeded = None
                except (RuntimeError, AttributeError, ValueError, TypeError):

                    pass
            elif object_class and self._auto_zone is not None:
                try:

                    self._auto_detail_user_locked = False
                    self._auto_detail_lock_prompt = ""
                    self._reseed_auto_detail_from_blob(object_class)
                except (RuntimeError, AttributeError):
                    pass














            self._auto_run_plan = None
            prompt_rewritten = None
            exemplar_size_m = None
            try:
                exemplar_size_m = self._exemplar_size_for_plan()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                exemplar_size_m = None
            if object_class or exemplar_size_m is not None:
                try:
                    from ...core.activation_manager import get_auth_header
                    plan_auth = get_auth_header()
                    if plan_auth:
                        from ...api.terralab_client import TerraLabClient
                        zone_area_m2, native_mupp = self._auto_run_plan_inputs()
                        plan = TerraLabClient().get_seg_run_plan(
                            object_class, zone_area_m2, native_mupp, auth=plan_auth,
                            exemplar_size_m=exemplar_size_m)
                        if isinstance(plan, dict) and not plan.get("error"):
                            self._auto_run_plan = {
                                "prompt": object_class, "plan": plan,
                                "exemplar_size_m": exemplar_size_m}




                            rewritten = self._headless_prompt_rewrite(
                                object_class, plan)
                            if rewritten:
                                prompt_rewritten = {
                                    "from": object_class, "to": rewritten}
                                typed = object_class
                                object_class = rewritten





                                replan = TerraLabClient().get_seg_run_plan(
                                    rewritten, zone_area_m2, native_mupp,
                                    auth=plan_auth,
                                    exemplar_size_m=exemplar_size_m,
                                    rewritten_from=typed)
                                if isinstance(replan, dict) and not replan.get("error"):
                                    plan = replan
                                    self._auto_run_plan = {
                                        "prompt": rewritten, "plan": plan,
                                        "exemplar_size_m": exemplar_size_m,
                                        "rewritten_from": typed}


                            if detail is None:
                                try:
                                    self._auto_detail_user_locked = False
                                    self._reseed_auto_detail_from_plan(object_class, plan)
                                except (RuntimeError, AttributeError, TypeError, ValueError):

                                    pass
                except Exception:  # noqa: BLE001
                    pass  # nosec B110
            if translated_from:


                prompt_rewritten = {"from": translated_from, "to": object_class}




            if confidence is not None:
                try:
                    spin = getattr(self.dock_widget, "auto_confidence_spin", None)
                    if spin is not None:
                        spin.setValue(max(0.05, min(0.95, float(confidence))))
                except (RuntimeError, AttributeError, TypeError, ValueError):

                    pass




            self._auto_review_preset_overrides = (
                dict(refine) if isinstance(refine, dict) and refine else None)



            self._auto_headless_run = True


            self._headless_error = None
            worker_before = self._auto_worker
            self._start_auto_detection()

            if self._auto_worker is None or self._auto_worker is worker_before:



                reason = getattr(self, "_headless_error", None)
                refused = {
                    "_error": reason or (
                        "Detection did not start. Check the AI Segmentation log: "
                        "missing raster, not signed in, zone too large, or feature disabled."
                    )
                }
                if worker_before is not None and self._auto_worker is worker_before:
                    refused["busy"] = True
                return refused







            if not wait:





                shown_layer_id = ""
                return {
                    "started": True,
                    "running": True,
                    "object_class": object_class,
                    **({"prompt_rewritten": prompt_rewritten}
                       if prompt_rewritten else {}),
                    "hint": ("The sweep is running in the AI Segmentation panel, which shows the "
                             "tiles, the progress and the cost. Call auto_detect_status(wait_s=45) "
                             "for the outcome: it answers as soon as the run ends. Nothing else is "
                             "needed to keep it going."),
                }




            loop = QEventLoop()

            worker = self._auto_worker

            def _on_finished(_results):
                loop.quit()

            def _on_error(_msg):
                loop.quit()

            def _on_exhausted(_remaining):
                loop.quit()

            def _on_cancelled():
                loop.quit()

            worker.all_tiles_finished.connect(_on_finished)
            worker.error.connect(_on_error)
            worker.credits_exhausted.connect(_on_exhausted)
            worker.cancelled.connect(_on_cancelled)





            cancel_state = {"asked": False, "deadline": 0.0}
            poll = self._arm_headless_cancel_poll(loop, should_cancel, cancel_state)


            deadline_timer = QTimer(self.dock_widget)
            deadline_timer.setSingleShot(True)
            deadline_timer.timeout.connect(loop.quit)
            try:
                deadline_timer.start(timeout_s * 1000)
                loop.exec()
            finally:
                deadline_timer.stop()
                self._disarm_headless_cancel_poll(poll)



            for sig, slot in (
                (worker.all_tiles_finished, _on_finished),
                (worker.error, _on_error),
                (worker.credits_exhausted, _on_exhausted),
                (worker.cancelled, _on_cancelled),
            ):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass

            if self._last_auto_result is None:
                if cancel_state["asked"]:




                    return {"_error": "Cancelled", "cancelled": True}


                salvaged = self._salvage_headless_timeout()
                timed_out = f"Detection timed out after {timeout_s}s"
                if salvaged is not None:
                    salvaged = dict(salvaged)
                    salvaged["_error"] = timed_out
                    return salvaged
                self._stop_auto_detection()
                hard_stopped = True
                return {"_error": timed_out}

            result = self._last_auto_result
            status = result.get("status")

            if status == "completed":




                if self._auto_review is not None:
                    exported = self._export_auto_review()
                    if exported and exported[0]:
                        result["layer_name"] = exported[0]
                    else:



                        return {"_error": "The detections could not be written "
                                          "to a layer. Check the project folder "
                                          "is writable and export again."}




                done = {
                    "instances": result.get("instances", 0),
                    "tiles_processed": result.get("tiles_processed", 0),
                    "layer_name": result.get("layer_name"),
                }
                if prompt_rewritten:
                    done["prompt_rewritten"] = prompt_rewritten
                return self._color_saved_objects_apart(done, instance_colors)
            if status == "error":
                return {"_error": result.get("message", "Unknown error")}
            if status == "credits_exhausted":
                out = {
                    "_error": "Credits exhausted",
                    "credits_remaining": result.get("credits_remaining", 0),
                }



                if result.get("layer_name"):
                    out["layer_name"] = result.get("layer_name")
                    out["instances"] = result.get("instances", 0)
                    out["tiles_processed"] = result.get("tiles_processed", 0)
                return self._color_saved_objects_apart(out, instance_colors)
            if status in ("cancelled", "stalled"):
                out = {"_error": (
                    "Detection stopped responding" if status == "stalled"
                    else "Cancelled")}


                if status == "cancelled":
                    out["cancelled"] = True




                if result.get("layer_name"):
                    out["layer_name"] = result.get("layer_name")
                    out["instances"] = result.get("instances", 0)
                    out["tiles_processed"] = result.get("tiles_processed", 0)
                return self._color_saved_objects_apart(out, instance_colors)

            return {"_error": f"Unexpected result status: {status}"}
        finally:











            live = getattr(self, "_auto_worker", None)
            if hard_stopped or live is None or not live.isRunning():
                self._auto_headless_run = False
                self._auto_review_preset_overrides = None
            self._restore_dock_after_headless(mode_before, shown_layer_id)




            if getattr(self, "_unload_deferred", False):
                self._unload_deferred = False
                try:
                    self.unload()
                except Exception:  # noqa: BLE001
                    pass  # nosec B110




    def _color_saved_objects_apart(self, out: dict, wanted: bool) -> dict:





        if not wanted or not out.get("layer_name"):
            return out
        try:
            from ...core.instance_symbology import report_instance_colors
            layer_id = getattr(self, "_auto_export_layer_id", "") or ""
            layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
            return report_instance_colors(out, layer)
        except Exception:  # noqa: BLE001
            out["instance_colors"] = False
            out["instance_colors_note"] = (
                "The objects are saved. Colouring them one by one did not work, "
                "so the layer keeps its export style.")
            return out
