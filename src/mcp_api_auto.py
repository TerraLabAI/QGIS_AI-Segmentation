"""Zone detection for the public API: every instance of a class in an area.

Part of `SegmentationMCPAPI` (see `mcp_api.py`), split out so one concern sits
in one file. These calls reach TerraLab's servers, take minutes and spend the
account's monthly allowance, so each one says so in its docstring.
"""
from __future__ import annotations

import math
from typing import Callable

from qgis.core import Qgis, QgsGeometry

# The confidence band detect_auto accepts. The shipped pair is the fallback;
# the band in force comes from _confidence_bounds_in_force().
_CONFIDENCE_BOUNDS = (0.05, 0.95)


def _confidence_bounds_in_force() -> tuple[float, float]:
    """The (low, high) confidence band in force, resolved at call time.

    Every sentence naming the band formats off this pair, so what the refusal
    says and what the check does can never drift apart. A cutoff only means
    anything strictly inside 0 and 1, so a served pair outside that is
    ignored.
    """
    try:
        from .core.server_dials import dial_pair

        low, high = dial_pair("agent.confidence_bounds", _CONFIDENCE_BOUNDS)
        if 0.0 < low <= high < 1.0:
            return (low, high)
    except Exception:  # noqa: BLE001 - the shipped band always works  # nosec B110
        pass
    return _CONFIDENCE_BOUNDS


class SegmentationAutoMixin:
    """Start, watch and stop a zone run, and set the panel up for one."""

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
    ) -> dict:
        """Run an Automatic (cloud) detection over a zone.

        Parameters
        ----------
        zone_wkt : str
            Well-known text (WKT) geometry in the raster layer's CRS defining
            the detection zone. Use POLYGON or MULTIPOLYGON. If empty string,
            the full raster extent is used.
        object_class : str
            Class of objects to detect, e.g. "Building", "Tree", "Car". May be
            empty ONLY when at least TWO positive exemplars are given: a single
            reference detects poorly, so the example-only path needs a pair (the
            cloud model needs either a text prompt or two visual examples).
        layer_name : str | None
            Optional raster layer name. If None, uses the currently selected
            layer.
        exemplars : list[dict] | None
            Optional visual exemplars ("draw one example, find all"). Each item
            is {"bbox": [xmin, ymin, xmax, ymax], "label": 1|0} in the raster
            layer's CRS (same CRS as zone_wkt), where label 1 = positive
            (find similar) and 0 = exclude. An exemplar run uses single-image
            mode (the whole zone is one query image). Additive: omit for the
            text-only behaviour.
        detail : int | None
            Tiles along the zone's longer side. Many small objects want a high
            value, one large object a low one (1 or 2) so it is not cut across
            tile edges. None lets the run pick from the prompt.
        confidence : float | None
            Cutoff applied to this run's results, inside the band the product
            accepts; a value outside it is refused with the band named. Lower
            keeps more and shows more false positives; higher keeps less.
            None uses the value the product resolves for this object class.
            Additive. A value equal to the product default reads as "no
            choice made".
        refine : dict | None
            Shape cleanup applied to this run's objects, as plain keys:
            ``simplify_px`` (float), ``smooth`` (bool, round the corners),
            ``expand_px`` (int, positive grows and negative shrinks),
            ``fill_holes`` (bool), ``fill_holes_max_m2`` (float, 0 = every
            hole), ``clean_px`` (float, trim spikes), ``ortho`` (bool, square
            the corners), ``min_size_m2`` (float, drop anything smaller) and
            ``points_pct`` (int 1-100, share of outline points kept). Unknown
            keys are ignored. Additive: omit to keep the settings the product
            picks for this object class.
        timeout_s : int | None
            Seconds this call may block before it gives up on the run. None
            keeps the plugin's own ceiling. Additive.
        should_cancel : Callable[[], bool] | None
            Asked every few hundred milliseconds while the call blocks. The
            first True stops the run the same way the panel's Cancel button
            does, so the tiles the account already paid for are kept and come
            back in the result under "cancelled". None (the default) polls
            nothing. A callable that raises is read as "not cancelled" and is
            never asked again. Additive.
        instance_colors : bool
            Give every object found its own colour on the saved layer, instead
            of one colour for the whole run. Buildings that touch then read as
            separate shapes rather than one block. False (the default) keeps
            the run's single export colour, so nothing changes for a caller
            that does not ask. Past the count where one colour per object stops
            being readable, the layer keeps the export colour and the result
            says so under "instance_colors_note". Additive.

        Returns
        -------
        dict with keys:
            "instances"     -- int, number of polygons detected
            "tiles_processed" -- int, imagery tiles the run got an answer
                               for. A measure of work done, NOT of cost: an
                               Automatic run is charged for the surface of
                               the zone it covers, whatever it finds, and
                               the plugin never learns the final figure.
                               To measure the real cost, read get_status()
                               ["auto_credits_remaining"] before the run
                               and again a few seconds after it ends (the
                               balance refreshes in the background).
            "cancelled"     -- bool, present and True when the run was stopped
                               on request. The keys above then describe what
                               was kept, and "_error" reads "Cancelled".
            "layer_name"    -- str, name of the output vector layer created.
                               Treat as opaque: it is a human-friendly name
                               like "Buildings (3 Jul)". Results are saved as
                               a table inside the project's
                               ai_segmentation.gpkg.
            "busy"          -- bool, present and True when another run is
                               already going. Poll auto_detect_status() rather
                               than calling again.
            "hint"          -- str, present on a run that worked: one sentence
                               naming the next call worth making.
            "instance_colors" -- bool, present only when instance_colors was
                               asked for: True when the saved layer wears one
                               colour per object, False when it kept the
                               export colour.
            "instance_colors_note" -- str, present only when
                               "instance_colors" is False: one sentence saying
                               why, for example that the run found more
                               objects than the ceiling allows.
            "dropped_options" -- list[str], present only when this plugin
                               version could not apply an argument that was
                               passed. The run went ahead without it.
            "dropped_options_note" -- str, the same fact in one sentence.
            "_error"        -- str, present only on failure

        Cost
        ----
        Billed by the area swept, and it can take minutes. Only one run at a
        time: a second call while one is going is refused with ``busy``, never
        queued and never silently attached to the first.
        """
        plugin = self._plugin

        from .core.detect_gate import can_detect

        has_text = bool(object_class and object_class.strip())
        # A run needs the word. Exemplars sharpen it and never replace it, the
        # same floor the panel's Detect button reads, so an agent and a person
        # are refused for the same reason (see core/detect_gate.can_detect).
        positives = 0
        for ex in (exemplars or []):
            try:
                if int(ex.get("label", 1)) == 1:
                    positives += 1
            except (TypeError, ValueError, AttributeError):
                positives += 1  # malformed label defaults to positive
        if not can_detect(has_text, positives):
            return {"_error": (
                "object_class must be a non-empty string. Exemplars sharpen a "
                "run, they cannot stand in for the word."
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

        if should_cancel is not None and not callable(should_cancel):
            return {"_error": "should_cancel must be a callable taking no arguments, or None."}
        # A bool and nothing else: bool("false") is True, so a string here
        # would silently turn the option on for a caller trying to turn it off.
        if not isinstance(instance_colors, bool):
            return {"_error": f"instance_colors must be True or False, got {instance_colors!r}."}
        if timeout_s is not None:
            try:
                timeout_s = int(timeout_s)
            except (TypeError, ValueError):
                return {"_error": f"timeout_s must be a whole number of seconds, got {timeout_s!r}."}
            if timeout_s <= 0:
                return {"_error": f"timeout_s must be greater than 0, got {timeout_s!r}."}

        # A blank zone means the whole raster, and the run caps only a zone it
        # was handed, so the free-tier cap is applied here to the extent that
        # blank stands for. Otherwise the API starts a run the panel refuses.
        if not (zone_wkt and str(zone_wkt).strip()):
            over_cap = self._full_extent_over_free_cap(layer_name)
            if over_cap is not None:
                return over_cap

        # _run_auto_detect_headless switches mode itself; no need to refuse
        # just because the dock was in Interactive mode.
        runner = plugin._run_auto_detect_headless
        kwargs = {
            "zone_wkt": zone_wkt,
            "object_class": (object_class or "").strip(),
            "layer_name": layer_name,
            "exemplars": exemplars,
            "detail": detail,
        }
        # Read the signature rather than catching a TypeError from the call: a
        # TypeError raised INSIDE a run would then start a second, billable one.
        import inspect
        try:
            accepted = inspect.signature(runner).parameters
        except (TypeError, ValueError):
            accepted = {}
        # An option this build's runner does not take is dropped, and a run
        # that ignores what the caller asked for while charging for the zone
        # has to say so. Named here, reported on the result below.
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

        try:
            return self._with_dropped_options(
                self._with_auto_hint(runner(**kwargs)), dropped)
        except Exception as e:
            import traceback

            from qgis.core import QgsMessageLog
            QgsMessageLog.logMessage(
                f"MCP detect_auto failed: {e}\n{traceback.format_exc()}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical
            )
            return {"_error": f"Automatic detection failed: {str(e)}"}

    # A run through this API saves itself and leaves no review open, so the
    # next move is never a review call. It is the next run, and what to change
    # in it depends only on whether this one found anything.
    def _with_auto_hint(self, result):
        """Add the one sentence saying what to call next, on a run that worked."""
        if not isinstance(result, dict) or "_error" in result or "hint" in result:
            return result
        if "instances" not in result:
            return result
        if int(result.get("instances") or 0) > 0:
            result["hint"] = (
                "The objects are saved in the layer named above. Call "
                "refine_settings() to read the shape cleanup a run starts "
                "from, and pass refine= to detect_auto() to change it."
            )
        else:
            result["hint"] = (
                "Nothing matched here. Call detect_auto() again with a plainer "
                "object_class word, a lower confidence, or a detail that suits "
                "the size of the objects."
            )
        return result

    def _with_dropped_options(self, result, dropped: list[str]):
        """Name the options this build could not apply, on the run's own answer.

        Silence here reads as "applied": a caller that asked for a tighter
        confidence and got a loose run had no way to tell that its argument
        never reached anything.
        """
        if not dropped or not isinstance(result, dict):
            return result
        result["dropped_options"] = list(dropped)
        result["dropped_options_note"] = (
            "This plugin version does not take " + ", ".join(dropped) +
            " on a zone run, so the run went ahead without it. Update the "
            "plugin to use it."
        )
        return result

    def _detail_in_range(self, detail):
        """Validate an optional detail level, as (value_or_None, error_or_None).

        Detail is tiles along the zone's longer side, so zero and a fraction
        name no grid at all. Both used to reach the run, where zero read as
        "one tile" and quietly changed what the zone was scanned at.
        """
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
        """Error dict when the drawn examples are unusable, else None.

        An exemplar the run cannot read is worse than none: the run starts, the
        zone is charged, and the examples the caller drew are silently absent
        from what it paid for.
        """
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
        """Validate an optional confidence, as (value_or_None, error_or_None)."""
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

    def set_mode(self, mode: str) -> dict:
        """Switch the dock between interactive and automatic modes.

        Parameters
        ----------
        mode : str
            "interactive" or "automatic" (case-insensitive).

        Returns
        -------
        dict with key "mode" (new mode string) or "_error".
        """
        plugin = self._plugin
        if mode is not None and not isinstance(mode, str):
            return {"_error": "mode must be a string, 'interactive' or 'automatic'"}
        mode_lower = mode.strip().lower() if mode else ""
        if mode_lower not in ("interactive", "automatic"):
            # Imported here, not at the top: mcp_api assembles this mixin, so
            # the import can only run once that module is built.
            from .mcp_api import not_found_error
            return not_found_error(
                "mode", mode_lower, ["interactive", "automatic"],
                note="The panel labels them Manual and Automatic.",
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
            dock._on_mode_selected(target)
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

    def set_auto_zone(self, zone_wkt: str | None) -> dict:
        """Set the detection zone for automatic mode.

        The WKT must be in the raster layer's CRS. Pass None or empty string
        to clear the zone (use full raster extent).

        Returns
        -------
        dict with key "zone_set" (bool) and bbox keys when a zone is set,
        or "_error".
        """
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

        # Convert from layer CRS to canvas CRS (same transform as in
        # _run_auto_detect_headless; _start_auto_detection reprojects back).
        active_layer = None
        try:
            active_layer = plugin._get_active_raster_layer()
        except (RuntimeError, AttributeError):
            pass

        # Free-trial zone cap: mirror the interactive draw guard (additive,
        # explicit error; subscribers are never capped). The WKT is in the
        # layer CRS (canvas CRS when no layer is resolved).
        try:
            zone_crs = active_layer.crs() if active_layer is not None else None
            cap_area = plugin._free_zone_cap_exceeded_km2(geom, crs=zone_crs)
        except (RuntimeError, AttributeError):
            cap_area = None
        if cap_area is not None:
            try:
                from .core import telemetry_run_events
                telemetry_run_events.track_auto_zone_too_large(area_km2=cap_area)
            except Exception:
                pass  # nosec B110
            from .ui.plugin.shared import zone_over_free_cap_message
            return {"_error": zone_over_free_cap_message(cap_area)}

        # Keeps the polygon beside the rectangle, so the cap that measured the
        # polygon and the run that tiles the zone agree on one shape, and the
        # run reports the surface it is billed on.
        bbox = plugin._store_auto_zone_from_geometry(geom, active_layer)
        try:
            dock = getattr(plugin, "dock_widget", None)
            if dock:
                dock.set_auto_zone_state("zone_set")
        except (RuntimeError, AttributeError):
            pass

        return {
            "zone_set": True,
            "xmin": bbox.xMinimum(),
            "ymin": bbox.yMinimum(),
            "xmax": bbox.xMaximum(),
            "ymax": bbox.yMaximum(),
        }

    def auto_detect_status(self) -> dict:
        """Return the current automatic detection status.

        Returns
        -------
        dict with keys:
            "running"      -- bool, True if a worker is currently active.
            "last_result"  -- dict or None, result of the most recent run.
            "mode"         -- str ("interactive" or "automatic") or None.
        """
        plugin = self._plugin

        running = False
        try:
            worker = plugin._auto_worker
            running = worker is not None and worker.isRunning()
        except (RuntimeError, AttributeError):
            pass

        mode_str = None
        try:
            dock = getattr(plugin, "dock_widget", None)
            if dock and hasattr(dock, "_mode"):
                mode_str = dock._mode.value
        except (RuntimeError, AttributeError):
            pass

        return {
            "running": running,
            "last_result": getattr(plugin, "_last_auto_result", None),
            "mode": mode_str,
        }

    def cancel_auto(self) -> dict:
        """Cancel any running automatic detection, keeping what it already paid for.

        Takes the same route as the panel's Cancel button: the worker is asked
        to stop and the tiles it already delivered go on into the review. The
        hard teardown keeps nothing, so cancelling a nearly finished run there
        threw away every tile the account had been billed for.

        The tiles are salvaged asynchronously, so the count below is what the
        worker had delivered at the moment of the call.

        Returns
        -------
        dict with keys:
            "cancelled"      -- bool, always True.
            "tiles_salvaged" -- int, tiles already delivered and kept.
        """
        plugin = self._plugin

        # Read the count before cancelling: the soft path drops the worker
        # reference once it winds down.
        salvaged = 0
        try:
            worker = getattr(plugin, "_auto_worker", None)
            if worker is not None:
                salvaged = int(getattr(worker, "tiles_succeeded", 0) or 0)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            salvaged = 0

        try:
            if hasattr(plugin, "_on_auto_cancel_clicked"):
                plugin._on_auto_cancel_clicked()
            else:
                plugin._stop_auto_detection()
        except (RuntimeError, AttributeError):
            pass
        return {"cancelled": True, "tiles_salvaged": salvaged}

    def _full_extent_over_free_cap(self, layer_name: str | None):
        """Error dict when the full raster is over the free-tier zone cap, else None.

        A blank zone means the whole raster, and the run guard caps only a zone
        it was handed, so the derived extent is measured here instead.
        """
        plugin = self._plugin
        try:
            layer = self._resolve_raster_layer(layer_name)
            if layer is None:
                return None
            extent = layer.extent()
            if extent is None or extent.isEmpty():
                return None
            cap_area = plugin._free_zone_cap_exceeded_km2(
                QgsGeometry.fromRect(extent), crs=layer.crs())
        except (RuntimeError, AttributeError, TypeError):
            return None
        if cap_area is None:
            return None
        try:
            from .core import telemetry_run_events
            telemetry_run_events.track_auto_zone_too_large(area_km2=cap_area)
        except Exception:
            pass  # nosec B110
        from .ui.plugin.shared import zone_over_free_cap_message
        return {"_error": zone_over_free_cap_message(cap_area)}
