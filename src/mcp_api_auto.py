





from __future__ import annotations

import math
from typing import Callable

from qgis.core import Qgis, QgsGeometry

from .mcp_api_guard import gui_thread_only



_CONFIDENCE_BOUNDS = (0.05, 0.95)


def _confidence_bounds_in_force() -> tuple[float, float]:







    try:
        from .core.server_dials import dial_pair

        low, high = dial_pair("agent.confidence_bounds", _CONFIDENCE_BOUNDS)
        if 0.0 < low <= high < 1.0:
            return (low, high)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return _CONFIDENCE_BOUNDS





_STATUS_WAIT_CAP_S = 50.0
_STATUS_WAIT_RECOMMENDED_S = 45


_STATUS_POLL_AFTER_S = 30
_STATUS_POLL_AFTER_MIN_S = 5

_STATUS_WAIT_TICK_MS = 250


def _status_wait_seconds(wait_s) -> tuple[float, dict | None]:





    if wait_s is None:
        return 0.0, None
    if isinstance(wait_s, bool):
        return 0.0, {"_error": f"wait_s must be a number of seconds, got {wait_s!r}."}
    try:
        seconds = float(wait_s)
    except (TypeError, ValueError):
        return 0.0, {"_error": f"wait_s must be a number of seconds, got {wait_s!r}."}
    if not math.isfinite(seconds) or seconds < 0:
        return 0.0, {"_error": f"wait_s must be 0 or more, got {wait_s!r}."}
    return min(seconds, _STATUS_WAIT_CAP_S), None


def _auto_eta_seconds(plugin, dock, progress: dict | None) -> int | None:







    if progress is None:
        return None
    try:
        pace = getattr(dock, "_auto_run_pace", None) if dock is not None else None
        seconds = pace.seconds_left() if pace is not None else None
        if seconds is not None and math.isfinite(seconds):
            return max(0, int(round(seconds)))
        from .core.run_eta import run_seconds

        total = int((getattr(plugin, "_auto_run_ctx", None) or {}).get("total") or 0)
        own_pace = dock._auto_quote_pace() if dock is not None else None
        typical, _slow = run_seconds(total, seconds_per_tile=own_pace)
        if typical <= 0:
            return None
        percent = progress.get("percent")
        share_left = 1.0 - (float(percent) / 100.0 if percent is not None else 0.0)
        return max(0, int(round(typical * max(0.0, share_left))))
    except Exception:  # noqa: BLE001
        return None


def _poll_after_seconds(eta_s: int | None) -> int:

    if eta_s is None:
        return _STATUS_POLL_AFTER_S
    return max(_STATUS_POLL_AFTER_MIN_S, min(_STATUS_POLL_AFTER_S, int(eta_s)))


def _auto_progress_snapshot(dock) -> dict | None:













    if dock is None:
        return None
    try:
        found = int(getattr(dock, "_auto_found_count", 0) or 0)
        phase = getattr(dock, "_auto_progress_phase", None)
        bar = getattr(dock, "auto_tile_progress", None)
        percent = None
        if bar is not None and bar.maximum():
            percent = int(round(100.0 * bar.value() / bar.maximum()))
        clock = getattr(dock, "auto_progress_clock", None)
        elapsed = clock.text() if clock is not None and hasattr(clock, "text") else None
        note = None
        label = getattr(dock, "auto_progress_label", None)
        if label is not None and label.isVisible():
            note = label.text() or None

        left = None
        try:
            from .core.run_eta import friendly_time_left

            pace = getattr(dock, "_auto_run_pace", None)
            seconds = pace.seconds_left() if pace is not None else None
            if seconds is not None:
                left = friendly_time_left(seconds)
        except Exception:  # noqa: BLE001
            left = None

        return {
            "found_so_far": found,
            "object": (getattr(dock, "_auto_run_object_word", "") or "") or None,
            "percent": percent,
            "phase": phase,
            "elapsed": elapsed,
            "time_left": left,
            "note": note,
        }
    except (RuntimeError, AttributeError, TypeError, ZeroDivisionError):
        return None


class SegmentationAutoMixin:


    def detect_auto(
        self,
        zone_wkt: str,
        object_class: str,
        layer_name: str | None = None,
        exemplars: list[dict] | None = None,
        detail: int | None = None,
        confidence: float | None = None,
        refine: dict | None = None,
        timeout_s: int | None = None,
        should_cancel: Callable[[], bool] | None = None,
        instance_colors: bool = False,
        wait: bool = True,
    ) -> dict:

















































































































































        plugin = self._plugin

        from .core.detect_gate import can_detect

        has_text = bool(object_class and object_class.strip())



        positives = 0
        for ex in (exemplars or []):
            try:
                if int(ex.get("label", 1)) == 1:
                    positives += 1
            except (TypeError, ValueError, AttributeError):
                positives += 1
        if not can_detect(has_text, positives):
            return {"_error": (
                "a run needs an object: pass a non-empty object_class, or at "
                "least one positive exemplar. A word is the tighter of the two."
            )}

        if not hasattr(plugin, "_run_auto_detect_headless"):
            return {
                "_error": (
                    "Automatic detection not available in this plugin version. "
                    "Upgrade to AI Segmentation 1.3.0+."
                )
            }

        conf, conf_err = self._confidence_in_range(confidence)
        if conf_err:
            return conf_err

        detail, detail_err = self._detail_in_range(detail)
        if detail_err:
            return detail_err
        exemplar_err = self._exemplars_refused(exemplars)
        if exemplar_err:
            return exemplar_err

        if refine is not None and not isinstance(refine, dict):
            return {"_error": f"refine must be a dict of settings, or None, got {refine!r}."}
        if should_cancel is not None and not callable(should_cancel):
            return {"_error": "should_cancel must be a callable taking no arguments, or None."}


        if not isinstance(instance_colors, bool):
            return {"_error": f"instance_colors must be True or False, got {instance_colors!r}."}
        if timeout_s is not None:
            try:
                timeout_s = int(timeout_s)
            except (TypeError, ValueError):
                return {"_error": f"timeout_s must be a whole number of seconds, got {timeout_s!r}."}
            if timeout_s <= 0:
                return {"_error": f"timeout_s must be greater than 0, got {timeout_s!r}."}




        if not (zone_wkt and str(zone_wkt).strip()):
            extent_wkt = self._full_extent_wkt_over_free_budget(layer_name)
            if extent_wkt is not None:
                zone_wkt = extent_wkt



        runner = plugin._run_auto_detect_headless
        kwargs = {
            "zone_wkt": zone_wkt,
            "object_class": (object_class or "").strip(),
            "layer_name": layer_name,
            "exemplars": exemplars,
            "detail": detail,
        }


        import inspect
        try:
            accepted = inspect.signature(runner).parameters
        except (TypeError, ValueError):
            accepted = {}



        dropped: list[str] = []
        if "confidence" in accepted:
            kwargs["confidence"] = conf
        elif conf is not None:
            dropped.append("confidence")
        if "refine" in accepted:
            kwargs["refine"] = self._refine_overrides_from(refine)
        elif refine:
            dropped.append("refine")
        if timeout_s is not None:
            if "timeout_s" in accepted:
                kwargs["timeout_s"] = timeout_s
            else:
                dropped.append("timeout_s")
        if should_cancel is not None:
            if "should_cancel" in accepted:
                kwargs["should_cancel"] = should_cancel
            else:
                dropped.append("should_cancel")
        if instance_colors:
            if "instance_colors" in accepted:
                kwargs["instance_colors"] = True
            else:
                dropped.append("instance_colors")
        if not wait:



            if "wait" in accepted:
                kwargs["wait"] = False
            else:
                dropped.append("wait")



        if not self._agent_zone_run_in_flight():
            self._agent_stamp_last_result()
        id_before = str(getattr(plugin, "_auto_run_id", "") or "")
        try:
            result = self._with_dropped_options(
                self._with_auto_hint(runner(**kwargs)), dropped)
            return self._with_agent_run_facts(result, id_before)
        except Exception as e:
            import traceback

            from qgis.core import QgsMessageLog
            QgsMessageLog.logMessage(
                f"MCP detect_auto failed: {e}\n{traceback.format_exc()}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical
            )
            return {"_error": f"Automatic detection failed: {str(e)}"}

    def _with_agent_run_facts(self, result, id_before: str):






        if not isinstance(result, dict):
            return result
        plugin = self._plugin
        run_id = str(getattr(plugin, "_auto_run_id", "") or "")
        if not run_id or run_id == id_before:
            return result
        result["run_id"] = run_id
        if not self._agent_zone_run_in_flight():
            self._agent_stamp_last_result()
        if result.get("started") and result.get("running"):
            typical, slow = 0.0, 0.0
            try:
                from .core.run_eta import run_seconds

                dock = getattr(plugin, "dock_widget", None)
                total = int((getattr(plugin, "_auto_run_ctx", None) or {}).get("total") or 0)
                own_pace = dock._auto_quote_pace() if dock is not None else None
                typical, slow = run_seconds(total, seconds_per_tile=own_pace)
            except Exception:  # noqa: BLE001
                typical, slow = 0.0, 0.0
            eta = int(round(typical)) if typical > 0 else None
            if eta is not None:
                result["eta_s"] = eta
                result["eta_slow_s"] = int(round(slow))
            result["poll_after_s"] = _poll_after_seconds(eta)
            calls = (f", about {max(1, math.ceil(typical / _STATUS_WAIT_RECOMMENDED_S))} "
                     "calls for a run this size" if eta is not None else "")
            result["hint"] = (
                "The sweep is running in the AI Segmentation panel, which shows "
                "the tiles, the progress and the cost. Call "
                f"auto_detect_status(run_id='{run_id}', "
                f"wait_s={_STATUS_WAIT_RECOMMENDED_S}) until running and "
                f"finishing are both False{calls}. Never start it again "
                "meanwhile: a second run is billed again.")
        elif str(result.get("_error") or "").startswith("Detection timed out"):
            result["hint"] = (
                "A blocking call gives up at its timeout and stops the run, "
                "keeping the tiles already paid for. For a zone this size, "
                "call detect_auto(..., wait=False) and wait with "
                f"auto_detect_status(wait_s={_STATUS_WAIT_RECOMMENDED_S}).")
        return result




    def _with_auto_hint(self, result):

        if not isinstance(result, dict) or "_error" in result or "hint" in result:
            return result
        if "instances" not in result:
            return result
        from .core.server_dials import dial_text

        if int(result.get("instances") or 0) > 0:
            fallback = (
                "The objects are saved in the layer named above. Call "
                "refine_settings() to read the shape cleanup a run starts "
                "from, and pass refine= to detect_auto() to change it."
            )
            hint_id = "auto_found_instances"
        else:
            fallback = (
                "Nothing matched here. Call detect_auto() again with a plainer "
                "object_class word, a lower confidence, or a detail that suits "
                "the size of the objects."
            )
            hint_id = "auto_found_nothing"
        result["hint"] = dial_text("tuning.agent.hints", hint_id, 400) or fallback
        return result

    def _with_dropped_options(self, result, dropped: list[str]):






        if not dropped or not isinstance(result, dict):
            return result
        result["dropped_options"] = list(dropped)
        result["dropped_options_note"] = (
            "This plugin version does not take " + ", ".join(dropped)
            + " on a zone run, so the run went ahead without it. Update the "
            "plugin to use it."
        )
        return result

    def _detail_in_range(self, detail):






        if detail is None:
            return None, None
        from .core.tile_manager import MAX_DETAIL_LEVEL

        if isinstance(detail, bool) or not isinstance(detail, (int, float)):
            return None, {"_error": (
                f"detail must be a whole number from 1 to {MAX_DETAIL_LEVEL}, "
                f"or None to let the run choose, got {detail!r}.")}
        if float(detail) != int(detail) or not 1 <= int(detail) <= MAX_DETAIL_LEVEL:
            return None, {"_error": (
                f"detail must be a whole number from 1 to {MAX_DETAIL_LEVEL}, "
                f"got {detail!r}.")}
        return int(detail), None

    def _exemplars_refused(self, exemplars):






        if exemplars is None:
            return None
        if not isinstance(exemplars, (list, tuple)):
            return {"_error": (
                "exemplars must be a list of "
                "{'bbox': [xmin, ymin, xmax, ymax], 'label': 1 or 0}.")}
        for index, item in enumerate(exemplars):
            if not isinstance(item, dict):
                return {"_error": (
                    f"exemplars[{index}] must be a dict with 'bbox' and "
                    f"'label', got {item!r}.")}
            box = item.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return {"_error": (
                    f"exemplars[{index}]['bbox'] must be "
                    f"[xmin, ymin, xmax, ymax], got {box!r}.")}
            try:
                xmin, ymin, xmax, ymax = (float(value) for value in box)
            except (TypeError, ValueError):
                return {"_error": (
                    f"exemplars[{index}]['bbox'] must hold four numbers, "
                    f"got {box!r}.")}
            if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
                return {"_error": (
                    f"exemplars[{index}]['bbox'] must hold finite numbers, "
                    f"got {box!r}.")}
            if xmin >= xmax or ymin >= ymax:
                return {"_error": (
                    f"exemplars[{index}]['bbox'] has no area: xmin must be "
                    f"below xmax and ymin below ymax, got {box!r}.")}
            label = item.get("label", 1)
            if isinstance(label, bool) or label not in (0, 1):
                return {"_error": (
                    f"exemplars[{index}]['label'] must be 1 (find similar) or "
                    f"0 (exclude), got {label!r}.")}
        return None

    def _confidence_in_range(self, confidence):

        if confidence is None:
            return None, None
        low, high = _confidence_bounds_in_force()
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            return None, {"_error": (
                f"confidence must be a number in [{low:g}, {high:g}], got {confidence!r}.")}
        if not low <= conf <= high:
            return None, {"_error": (
                f"confidence must be in [{low:g}, {high:g}], got {confidence!r}.")}
        return conf, None

    @gui_thread_only
    def set_mode(self, mode: str) -> dict:











        plugin = self._plugin
        if mode is not None and not isinstance(mode, str):
            return {"_error": "mode must be a string, 'interactive' or 'automatic'"}
        mode_lower = mode.strip().lower() if mode else ""
        if mode_lower not in ("interactive", "automatic"):


            from .mcp_api import not_found_error
            return not_found_error(
                "mode", mode_lower, ["interactive", "automatic"],
                note="The panel labels them Semi-Auto and Automatic.",
            )

        try:
            plugin._ensure_dock_widget()
        except Exception:  # nosec B110
            pass

        try:
            from .ui.ai_segmentation_dockwidget import Mode
            target = Mode.AUTOMATIC if mode_lower == "automatic" else Mode.INTERACTIVE
            dock = getattr(plugin, "dock_widget", None)
            if dock is None:
                return {"_error": "Dock widget not available"}



            accepted = dock._on_mode_selected(target)
            if accepted is False:
                return {"_error": f"The panel refused to switch to {mode_lower} mode."}
            if target == Mode.AUTOMATIC:
                try:
                    if plugin._tile_manager is None:
                        plugin._setup_auto_mode()
                except (RuntimeError, AttributeError):
                    pass
                try:
                    plugin._refresh_auto_credits()
                except (RuntimeError, AttributeError):
                    pass
            return {"mode": mode_lower}
        except Exception as e:
            return {"_error": f"Failed to switch mode: {str(e)}"}

    @gui_thread_only
    def set_auto_zone(self, zone_wkt: str | None) -> dict:























        plugin = self._plugin

        if zone_wkt is not None and not isinstance(zone_wkt, str):
            return {"_error": "zone_wkt must be a WKT string, or None to clear the zone"}

        if not zone_wkt or not zone_wkt.strip():
            plugin._store_auto_zone(None)
            plugin._auto_zone_polygon = None
            try:
                dock = getattr(plugin, "dock_widget", None)
                if dock:
                    dock.set_auto_zone_state("idle")
            except (RuntimeError, AttributeError):
                pass
            return {"zone_set": False}

        geom = QgsGeometry.fromWkt(zone_wkt)
        if geom is None or geom.isEmpty():
            return {"_error": "Invalid zone WKT"}



        active_layer = None
        try:
            active_layer = plugin._get_active_raster_layer()
        except (RuntimeError, AttributeError):
            pass





        try:
            zone_crs = active_layer.crs() if active_layer is not None else None
            free_fit = plugin._fit_zone_to_free_budget(geom, crs=zone_crs)
        except (RuntimeError, AttributeError):
            free_fit = None
        if free_fit is not None:
            if free_fit.geom is None:
                try:
                    from .core import telemetry_run_events
                    telemetry_run_events.track_auto_zone_too_large(
                        area_km2=free_fit.requested_km2)
                except Exception:
                    pass  # nosec B110
                from .ui.plugin.shared import zone_over_free_cap_message
                return {"_error": zone_over_free_cap_message(free_fit.requested_km2)}
            geom = QgsGeometry(free_fit.geom)




        bbox = plugin._store_auto_zone_from_geometry(geom, active_layer)
        if free_fit is not None:
            plugin._record_free_zone_fit(free_fit, notify=False)
        try:
            dock = getattr(plugin, "dock_widget", None)
            if dock:
                dock.set_auto_zone_state("zone_set")
        except (RuntimeError, AttributeError):
            pass

        result = {
            "zone_set": True,
            "xmin": bbox.xMinimum(),
            "ymin": bbox.yMinimum(),
            "xmax": bbox.xMaximum(),
            "ymax": bbox.yMaximum(),
        }
        if free_fit is not None:

            result["free_fit"] = {
                "km2_requested": round(free_fit.requested_km2, 4),
                "km2_processed": round(free_fit.processed_km2, 4),
            }
        result.update(self._agent_zone_cost_facts())
        return result

    def _agent_zone_cost_facts(self) -> dict:








        plugin = self._plugin
        dock = getattr(plugin, "dock_widget", None)
        facts: dict = {}
        try:


            plugin._update_credit_estimate()
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        if getattr(plugin, "_auto_zone", None) is None:
            return facts
        try:
            km2 = float(plugin._auto_zone_area_km2())
        except Exception:  # noqa: BLE001
            km2 = 0.0
        if km2 > 0:
            facts["km2"] = round(km2, 4)
        try:
            tiles = getattr(dock, "_auto_est_credits", None) if dock is not None else None
            if isinstance(tiles, int) and tiles > 0:
                from .core.run_eta import run_seconds

                typical, slow = run_seconds(tiles, seconds_per_tile=dock._auto_quote_pace())
                if typical > 0:
                    facts["estimate"] = {
                        "tiles": tiles,
                        "duration_s": [int(round(typical)), int(round(slow))],
                        "note": ("At the panel's current detail. The object "
                                 "named at detect_auto picks its own grid, so "
                                 "the run can take more or less."),
                    }
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        try:
            if getattr(plugin, "_last_usage", None):
                left = dock._auto_km2_left() if dock is not None else None
                allowance = {
                    "km2_left": round(left, 4) if left is not None else None,
                    "credits_remaining": getattr(dock, "_auto_credits", None),
                    "is_subscriber": bool(getattr(dock, "_auto_is_subscriber", False)),
                }
                if left is not None and km2 > 0:
                    allowance["fits"] = km2 <= left
                facts["allowance"] = allowance
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return facts

    def auto_detect_status(self, wait_s: float = 0, run_id: str | None = None) -> dict:







































        wait, wait_err = _status_wait_seconds(wait_s)
        if wait_err:
            return wait_err
        if run_id is not None and not isinstance(run_id, str):
            return {"_error": f"run_id must be the string detect_auto returned, or None, got {run_id!r}."}

        if run_id and not self._agent_run_id_known(run_id):
            out = self._agent_status_snapshot()
            latest = out.get("run_id")
            out["_error"] = (
                f"No run with id '{run_id}' is known here: it is neither running "
                "nor the last one to finish. " + (
                    f"The latest run is '{latest}'." if latest
                    else "No run has been recorded since QGIS started."))
            return out

        if wait > 0 and self._agent_zone_run_in_flight():
            self._agent_wait_for_zone_run(wait)
        return self._agent_status_snapshot()

    def _agent_status_snapshot(self) -> dict:

        plugin = self._plugin

        running = False
        try:
            worker = plugin._auto_worker
            running = worker is not None and worker.isRunning()
        except (RuntimeError, AttributeError):
            pass

        dock = None
        try:
            dock = getattr(plugin, "dock_widget", None)
        except (RuntimeError, AttributeError):
            pass

        mode_str = None
        try:
            if dock and hasattr(dock, "_mode"):
                mode_str = dock._mode.value
        except (RuntimeError, AttributeError):
            pass

        in_flight = self._agent_zone_run_in_flight()
        if not in_flight:
            self._agent_stamp_last_result()
        last = getattr(plugin, "_last_auto_result", None)
        current_id = str(getattr(plugin, "_auto_run_id", "") or "") or None

        out = {
            "running": running,
            "last_result": last,
            "mode": mode_str,
            "progress": _auto_progress_snapshot(dock) if running else None,
            "run_id": current_id if in_flight else (
                last.get("run_id") if isinstance(last, dict) else None),
            "finishing": bool(
                in_flight and not running and last is None
                and getattr(plugin, "_auto_finalize_state", None) is not None),
        }
        if in_flight:
            eta = _auto_eta_seconds(plugin, dock, out["progress"])
            if eta is not None:
                out["eta_s"] = eta
            out["poll_after_s"] = _poll_after_seconds(eta)
            out["hint"] = (
                "Still going. Call auto_detect_status(wait_s="
                f"{_STATUS_WAIT_RECOMMENDED_S}): it answers as soon as the run "
                "ends. Never start a second run meanwhile: it would be billed "
                "again.")
        return out

    def _agent_zone_run_in_flight(self) -> bool:






        plugin = self._plugin
        try:
            worker = getattr(plugin, "_auto_worker", None)
            if worker is not None and worker.isRunning():
                return True
        except (RuntimeError, AttributeError):
            pass
        if getattr(plugin, "_auto_start_in_progress", False):
            return True
        return (getattr(plugin, "_auto_finalize_state", None) is not None
                and getattr(plugin, "_last_auto_result", None) is None)

    def _agent_stamp_last_result(self) -> None:







        plugin = self._plugin
        last = getattr(plugin, "_last_auto_result", None)
        current_id = str(getattr(plugin, "_auto_run_id", "") or "")
        if isinstance(last, dict) and current_id and "run_id" not in last:
            last["run_id"] = current_id

    def _agent_run_id_known(self, run_id: str) -> bool:

        plugin = self._plugin
        if self._agent_zone_run_in_flight():
            return run_id == str(getattr(plugin, "_auto_run_id", "") or "")
        self._agent_stamp_last_result()
        last = getattr(plugin, "_last_auto_result", None)
        return isinstance(last, dict) and last.get("run_id") == run_id

    def _agent_wait_for_zone_run(self, seconds: float) -> None:








        import time

        deadline = time.monotonic() + seconds
        from .mcp_api_guard import _on_the_gui_thread

        if not _on_the_gui_thread():
            while time.monotonic() < deadline and self._agent_zone_run_in_flight():
                time.sleep(_STATUS_WAIT_TICK_MS / 1000.0)
            return

        from qgis.PyQt.QtCore import QEventLoop, QTimer

        loop = QEventLoop()
        timer = QTimer(loop)
        timer.setInterval(_STATUS_WAIT_TICK_MS)

        def _tick():
            try:
                done = not self._agent_zone_run_in_flight()
            except Exception:  # noqa: BLE001
                done = True
            if done or time.monotonic() >= deadline:
                loop.quit()

        timer.timeout.connect(_tick)
        try:
            timer.start()
            loop.exec()
        finally:
            try:
                timer.stop()
                timer.timeout.disconnect()
            except (RuntimeError, TypeError):
                pass

    def cancel_auto(self) -> dict:



















        plugin = self._plugin



        salvaged = 0
        was_running = False
        try:
            worker = getattr(plugin, "_auto_worker", None)
            if worker is not None:
                was_running = bool(worker.isRunning())
                salvaged = int(getattr(worker, "tiles_succeeded", 0) or 0)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            salvaged = 0

        if not was_running:
            return {
                "cancelled": False,
                "was_running": False,
                "tiles_salvaged": salvaged,
                "_error": "No run to cancel.",
            }

        try:
            if hasattr(plugin, "_on_auto_cancel_clicked"):
                plugin._on_auto_cancel_clicked()
            else:
                plugin._stop_auto_detection()
        except (RuntimeError, AttributeError):
            pass
        return {"cancelled": True, "was_running": True, "tiles_salvaged": salvaged}

    def _full_extent_wkt_over_free_budget(self, layer_name: str | None):






        plugin = self._plugin
        try:
            layer = self._resolve_raster_layer(layer_name)
            if layer is None:
                return None
            extent = layer.extent()
            if extent is None or extent.isEmpty():
                return None
            shape = QgsGeometry.fromRect(extent)
            fit = plugin._fit_zone_to_free_budget(shape, crs=layer.crs())
        except (RuntimeError, AttributeError, TypeError):
            return None
        if fit is None:
            return None
        return shape.asWkt()
