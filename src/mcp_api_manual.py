"""Point-and-click detection for the public API: one object at a time.

Part of `SegmentationMCPAPI` (see `mcp_api.py`), split out so one concern sits
in one file. `detect` takes a single point; `detect_points` takes a whole
prompt, positive points that grow the outline and negative points that cut
parts off it. Both run the same crop, predict, vectorize and save path, so they
return the same keys.
"""
from __future__ import annotations

import math

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
)

# How close to the ceiling a window has to land to count as held there. The
# ceiling is one exact number and a window that reaches it was clamped, so
# anything but floating-point slack would be a coincidence.
_CEILING_SLACK = 0.01

# How much tighter the second look is when the first was held at the ceiling.
# Eight times narrower turns a window covering a district into one covering a
# street, which is the frame a single object is actually found in.
_CAPPED_RETRY_FACTOR = 8.0


class SegmentationManualMixin:
    """Outline one object from points a caller supplies, and save it."""

    def detect(
        self,
        x: float,
        y: float,
        layer_name: str | None = None,
        discard_unsaved: bool = False,
        output_dir: str | None = None,
    ) -> dict:
        """Outline the object under a map point. Returns the polygon, or an error.

        This call exports what it detects, so the export IS the save and the
        save rule applies: one object saved while TerraLab's servers answer the
        clicks costs one credit. On an empty balance the call refuses with
        ``_error`` and writes nothing. A session that runs on this computer
        costs nothing and is never refused.

        Loads the model itself when it is not loaded, which costs nothing and
        adds seconds to the first call of a session; the answer then carries
        ``model_loaded_by_this_call``.

        The answer also carries ``crop_ground_width_m``, how much ground the
        click was read from. A map zoomed out past what a click can work at
        gets a tighter second look, reported as ``reframed``; when even that
        stays at the ceiling, ``crop_capped`` says so and the ``hint`` asks
        for the map to be moved onto the object.

        Parameters
        ----------
        x, y : float
            Point in the canvas CRS. Both must be finite numbers.
        layer_name : str | None
            The imagery layer to read, by name or by layer id. None uses the
            live session's layer, or the first raster in the project. Two
            layers sharing a name are refused, and the refusal lists their ids.
        discard_unsaved : bool
            Switching to another layer restarts the session, which throws away
            every polygon a person saved but has not exported. The call refuses
            instead, unless this is True. Additive, default False.
        output_dir : str | None
            Folder for the GeoPackage this call writes. It has to exist
            already. None keeps the project folder. Additive, default None.
        """
        # A string, a None or a NaN reaches QgsPointXY and the transform below
        # as an exception the caller cannot read, so refuse it by name here.
        try:
            px, py = float(x), float(y)
        except (TypeError, ValueError):
            return {"_error": f"x and y must be numbers, got ({x!r}, {y!r})."}
        if not (math.isfinite(px) and math.isfinite(py)):
            return {"_error": f"x and y must be finite numbers, got ({x}, {y})."}

        return self._detect_from_points(
            [(px, py)], [], layer_name, discard_unsaved, output_dir)

    def detect_points(
        self,
        positive: list[list[float]],
        negative: list[list[float]] | None = None,
        layer_name: str | None = None,
        discard_unsaved: bool = False,
        output_dir: str | None = None,
    ) -> dict:
        """Outline one object from several points, and save it.

        This is how a person actually works. The first positive point picks the
        object. Every extra positive point tells the model to include a part it
        missed. Every negative point tells it to drop a part it wrongly took in,
        which is the usual fix when an outline swallows a neighbour.

        All the points describe ONE object. To find several objects, call this
        once per object, or use :meth:`detect_auto` for a whole zone.

        Same save rule, same self-loading model and same framing keys as
        :meth:`detect`.

        Parameters
        ----------
        positive : list[list[float]]
            At least one [x, y] pair in the canvas CRS. The first pair is the
            point the image region is cut around, so make it the point most
            clearly on the object.
        negative : list[list[float]] | None
            Optional [x, y] pairs in the canvas CRS marking what to leave out.
        layer_name : str | None
            The imagery layer to read, by name or by layer id. None uses the
            live session's layer, or the first raster in the project.
        discard_unsaved : bool
            Same meaning as on :meth:`detect`: refuse rather than throw away
            polygons a person saved in an open session. Default False.
        output_dir : str | None
            Folder for the GeoPackage this call writes. It has to exist
            already. None keeps the project folder.

        Returns
        -------
        dict
            The same keys as :meth:`detect` (``detected``, ``score``,
            ``polygon_wkt``, ``polygon_count``, ``crs``, ``mask_pixels``,
            ``exported_layer``, ``exported_file``, ``hint``), plus
            ``points_used`` counting the positive and negative points that were
            applied. On failure, a single ``_error`` key.

        Cost
        ----
        The same as :meth:`detect`: it saves the object it outlines, so one
        object saved while TerraLab's servers answer the clicks costs one
        credit. On an empty balance the call refuses and writes nothing. A
        session running on this computer costs nothing.
        """
        pos, err = self._points_as_pairs(positive, "positive")
        if err:
            return err
        if not pos:
            return {"_error": "positive needs at least one [x, y] point."}
        neg, err = self._points_as_pairs(negative or [], "negative")
        if err:
            return err
        return self._detect_from_points(
            pos, neg, layer_name, discard_unsaved, output_dir)

    def _points_as_pairs(self, points, label: str):
        """Validate a list of [x, y] pairs, as (pairs, error_dict_or_None)."""
        if points is None:
            return [], None
        if not isinstance(points, (list, tuple)):
            return None, {"_error": f"{label} must be a list of [x, y] pairs."}
        out: list[tuple[float, float]] = []
        for item in points:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                return None, {"_error": (
                    f"Each {label} point must be an [x, y] pair, got {item!r}.")}
            try:
                px, py = float(item[0]), float(item[1])
            except (TypeError, ValueError):
                return None, {"_error": (
                    f"{label} coordinates must be numbers, got {item!r}.")}
            if not (math.isfinite(px) and math.isfinite(py)):
                return None, {"_error": (
                    f"{label} coordinates must be finite numbers, got {item!r}.")}
            out.append((px, py))
        return out, None

    def _detect_from_points(
        self,
        positive: list[tuple[float, float]],
        negative: list[tuple[float, float]],
        layer_name: str | None,
        discard_unsaved: bool,
        output_dir: str | None,
    ) -> dict:
        """Crop, predict, vectorize and save one object from a point prompt.

        The shared body behind :meth:`detect` and :meth:`detect_points`. The
        first positive point is the one the image region is cut around; every
        other point is converted into that region's pixel space.
        """
        plugin = self._plugin

        loaded_here = False
        if plugin.predictor is None:
            # The model unloads between sessions, and this call cannot work
            # without it. Loading is local, free and takes seconds, so the
            # answer is to do it rather than to hand the caller a step it
            # would take anyway. The result says it happened, because a
            # detect that quietly costs half a minute reads as a hang.
            outcome = self.load_model()
            if plugin.predictor is None:
                detail = outcome.get("_error") or "The model did not load."
                return {"_error": (
                    f"{detail} A person does the same by opening the AI "
                    "Segmentation panel and clicking 'Start Semi-Auto AI "
                    "Segmentation'."
                )}
            loaded_here = True

        raster_layer, err = self._ensure_session(layer_name, discard_unsaved)
        if err:
            return err

        px, py = positive[0]

        # Enter headless mode
        plugin._headless = True
        plugin._headless_error = None
        try:
            raster_pt = plugin._transform_to_raster_crs(QgsPointXY(px, py))
            if raster_pt is None:
                # No image in the raster CRS: the point sits outside the
                # projection domain. Guard both layer kinds here, the extent
                # check below only runs for file-based layers.
                return {
                    "_error": f"Point ({px}, {py}) cannot be projected into the raster CRS "
                    f"({raster_layer.crs().authid()}). Pick a point closer to the imagery."
                }

            # Check bounds for file-based layers
            is_online = getattr(plugin, "_is_online_layer", False)
            if not is_online and hasattr(plugin, "_is_point_in_raster_extent"):
                if not plugin._is_point_in_raster_extent(raster_pt):
                    ext = raster_layer.extent()
                    return {
                        "_error": f"Point ({px}, {py}) is outside the raster extent. "
                        f"Extent: xmin={ext.xMinimum():.2f}, ymin={ext.yMinimum():.2f}, "
                        f"xmax={ext.xMaximum():.2f}, ymax={ext.yMaximum():.2f} "
                        f"(CRS: {raster_layer.crs().authid()})."
                    }

            facts, err = self._crop_and_predict(
                raster_layer, raster_pt, positive, negative)
            if err:
                return err

            # A window held at its ceiling means the map was zoomed out past
            # anything a click can work at, and one object in it is a few
            # pixels. Rather than hand that back and ask the caller to move
            # the map, read the same point again in a window a fraction of the
            # size and keep whichever answer the model believes more. Only the
            # tiled path can be capped, and only there is the override a
            # ground step rather than a factor on the source's own pixels.
            retried = False
            if facts["capped"] and getattr(plugin, "_is_online_layer", False):
                finer, finer_err = self._crop_and_predict(
                    raster_layer, raster_pt, positive, negative,
                    force_step=facts["step"] / _CAPPED_RETRY_FACTOR)
                if finer_err is None:
                    retried = True
                    if finer["score"] > facts["score"]:
                        facts = finer

            mask = facts["mask"]
            score = facts["score"]
            crop_width_m = facts["width_m"]
            still_capped = facts["capped"]
            minx, miny, maxx, maxy = facts["bounds"]
            img_height, img_width = facts["img_shape"]

            points_used = {"positive": len(positive), "negative": len(negative)}

            if mask.sum() == 0:
                out = {"detected": False, "score": score,
                       "message": "No object detected at this point.",
                       "points_used": points_used}
                self._add_run_facts(out, loaded_here, crop_width_m,
                                    retried, still_capped)
                return out

            # Vectorize mask
            from .core.polygon_exporter import mask_to_polygons

            crs_authid = raster_layer.crs().authid() if raster_layer.crs().isValid() else "EPSG:4326"
            transform_info = {
                "bbox": (minx, maxx, miny, maxy),
                "img_shape": (img_height, img_width),
                "crs": crs_authid,
            }

            polygons = mask_to_polygons(mask, transform_info)
            if not polygons:
                out = {"detected": True, "score": score,
                       "message": "Object detected but vectorization failed.",
                       "points_used": points_used}
                self._add_run_facts(out, loaded_here, crop_width_m,
                                    retried, still_capped)
                return out

            if len(polygons) == 1:
                combined = polygons[0]
            else:
                combined = QgsGeometry.unaryUnion(polygons)

            wkt = combined.asWkt()

            # This call exports as it detects, so the export IS the save and the
            # same rule applies: an object whose click TerraLab's servers
            # answered costs one credit. Refused before the export rather than
            # after, so nothing is written that the account did not pay for.
            billing_id = plugin._next_handoff_det_id()
            if self._save_refused_for_credits_quiet(billing_id):
                return {
                    "_error": "Monthly cloud objects used up. Saving an object "
                              "spends one while TerraLab's servers answer the "
                              "clicks. Turn cloud processing off in the panel to "
                              "work on this computer, or upgrade to Pro."
                }

            # Auto-export
            export_result = self.export_polygon(
                wkt, crs_authid, raster_layer.name(), output_dir)
            if export_result and "_error" not in export_result:
                # The GeoPackage is on disk by now. Never report a failure past
                # this line: the caller would retry, write the object a second
                # time and pay for it a second time. So the charge carries its
                # own handler and stays out of the outer one.
                try:
                    plugin._charge_manual_saved_object(
                        billing_id, geom=combined, crs_authid=crs_authid)
                    ledger = getattr(plugin, "_manual_credit_ledger", None)
                    if ledger is not None:
                        ledger.start_next_object()
                except Exception as charge_err:  # noqa: BLE001
                    from qgis.core import QgsMessageLog
                    QgsMessageLog.logMessage(
                        f"MCP detect: the object charge did not go out ({charge_err})",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning
                    )

            result = {
                "detected": True,
                "score": score,
                "polygon_wkt": wkt,
                "polygon_count": len(polygons),
                "crs": crs_authid,
                "mask_pixels": int(mask.sum()),
                "points_used": points_used,
            }
            if export_result and "_error" not in export_result:
                result["exported_layer"] = export_result.get("layer_name")
                result["exported_file"] = export_result.get("file_path")
                # The outline is on disk, so the only move left is the shape
                # itself, and the one lever this path has is another point.
                result["hint"] = (
                    "Outline too big or too small? Call detect_points() with "
                    "the same positive point plus a negative one on the part "
                    "to cut off."
                )
            elif export_result:
                # The detection stands and the caller can still read its WKT,
                # so this is a key beside the result, not an _error over it.
                result["export_error"] = export_result["_error"]
                result["hint"] = (
                    "The outline is in polygon_wkt and nothing was written: "
                    "call export_polygon() with it once the folder is writable."
                )

            # Last, so a window that framed the object badly overrides the
            # advice above: it is the one thing worth fixing first.
            self._add_run_facts(result, loaded_here, crop_width_m,
                                retried, still_capped)
            return result

        except Exception as e:
            import traceback

            # Qgis comes from the module import: rebinding it here would make
            # it a local for the whole method, including the charge handler.
            from qgis.core import QgsMessageLog
            QgsMessageLog.logMessage(
                f"MCP detect failed: {e}\n{traceback.format_exc()}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical
            )
            return {"_error": f"Detection failed: {str(e)}"}
        finally:
            plugin._headless = False

    @staticmethod
    def _ground_width_in_metres(raster_layer, width_in_raster_units: float) -> float:
        """A width measured in the raster's own units, in metres. 0.0 when the
        units cannot be told, which reads as "not measured" everywhere below.
        """
        from qgis.core import QgsUnitTypes

        # The metre constant moved from QgsUnitTypes onto Qgis in 3.30, and the
        # floor this plugin supports is older than that.
        metres = getattr(getattr(Qgis, "DistanceUnit", None), "Meters", None)
        if metres is None:
            metres = getattr(QgsUnitTypes, "DistanceMeters", None)
        if metres is None:
            return 0.0
        try:
            per_metre = QgsUnitTypes.fromUnitToUnitFactor(
                metres, raster_layer.crs().mapUnits())
            width = float(width_in_raster_units)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0.0
        if not per_metre or per_metre <= 0 or not math.isfinite(per_metre):
            return 0.0
        if not math.isfinite(width) or width <= 0:
            return 0.0
        return round(width / per_metre, 1)

    def _add_run_facts(self, result: dict, loaded_here: bool,
                       crop_width_m: float, retried: bool,
                       still_capped: bool) -> None:
        """Say what the click was actually answered from.

        Three things a caller cannot see and would otherwise blame on the
        model. The model load, because a call that quietly takes half a minute
        reads as a hang. The window, because how much ground it covered is
        what decides whether one object was ever more than a few pixels. And
        the second look, taken when the map was zoomed out past what a click
        can work at, because the answer then came from a frame the caller
        never set.
        """
        if loaded_here:
            result["model_loaded_by_this_call"] = True
        if crop_width_m <= 0:
            return
        result["crop_ground_width_m"] = crop_width_m
        if retried:
            result["reframed"] = True
        if not still_capped:
            return
        result["crop_capped"] = True
        result["hint"] = (
            f"The map is zoomed out, so this click was read from the widest "
            f"window allowed, {crop_width_m:.0f} m across, and one object in "
            f"it is a few pixels. A tighter second look did no better. Move "
            f"the map onto the object, then click again."
        )

    def _crop_and_predict(self, raster_layer, raster_pt, positive, negative,
                          force_step=None):
        """Read the window around a click and ask the model for one mask.

        Returns ``(facts, error)``, exactly one of which is None. ``facts``
        carries the mask, the score the model gave it, the window it was cut
        from and how much ground that window covered.

        ``force_step`` reads a fresh window at a named ground step instead of
        the one the map implies, which is how the same click is asked a second
        time in a tighter frame.
        """
        import numpy as np

        plugin = self._plugin

        if force_step is not None:
            if not plugin._extract_and_encode_crop(
                    raster_pt, mupp_override=force_step):
                detail = plugin._headless_error or "Failed to encode image region."
                return None, {"_error": f"Crop encoding failed: {detail}"}
        else:
            crop_status = plugin._check_crop_status(raster_pt)
            if crop_status != "ok":
                if not plugin._handle_reencode(crop_status, raster_pt):
                    detail = plugin._headless_error or "Failed to encode image region."
                    return None, {"_error": f"Crop encoding failed: {detail}"}

        crop_info = plugin._current_crop_info
        if crop_info is None:
            return None, {"_error": (
                "No image region encoded. Try again or check the raster layer.")}
        img_height, img_width = crop_info["img_shape"]
        minx, miny, maxx, maxy = crop_info["bounds"]

        to_pixel = self._crop_pixel_mapper(
            minx, miny, maxx, maxy, img_width, img_height)

        coords: list[list[float]] = [to_pixel(raster_pt)]
        labels: list[int] = [1]
        # Every other point rides the SAME crop, so a point far outside it
        # lands off the image and the model ignores it rather than moving
        # the crop under the first point.
        for group, label in ((positive[1:], 1), (negative, 0)):
            for gx, gy in group:
                extra_pt = plugin._transform_to_raster_crs(QgsPointXY(gx, gy))
                if extra_pt is None:
                    return None, {"_error": (
                        f"Point ({gx}, {gy}) cannot be projected into the "
                        f"raster CRS ({raster_layer.crs().authid()}).")}
                coords.append(to_pixel(extra_pt))
                labels.append(label)

        masks, scores, _low_res = plugin.predictor.predict(
            point_coords=np.array(coords),
            point_labels=np.array(labels),
            multimask_output=True,
        )
        if plugin._headless_error:
            return None, {"_error": plugin._headless_error}

        # Pick the best mask, avoiding the ones that took the whole window.
        total_pixels = masks[0].shape[0] * masks[0].shape[1]
        areas = [int(m.sum()) for m in masks]
        small_enough = [i for i in range(len(scores))
                        if 0 < areas[i] < 0.8 * total_pixels]
        if small_enough:
            best = max(small_enough, key=lambda i: scores[i])
        else:
            best = min(range(len(scores)), key=lambda i: areas[i])

        width_m = self._ground_width_in_metres(raster_layer, maxx - minx)
        return {
            # Only the real image area: reflect padding at a raster edge would
            # otherwise leak mirrored polygons outside the raster.
            "mask": masks[best][:img_height, :img_width],
            "score": float(scores[best]),
            "bounds": (minx, miny, maxx, maxy),
            "img_shape": (img_height, img_width),
            "step": (maxx - minx) / float(img_width),
            "width_m": width_m,
            "capped": self._window_is_at_ceiling(width_m),
        }, None

    @staticmethod
    def _window_is_at_ceiling(width_m: float) -> bool:
        """Was this window held at the widest ground one crop may cover?

        The ceiling is one exact number, so a window that reaches it was
        clamped there; anything but floating-point slack would be a
        coincidence.
        """
        if width_m <= 0:
            return False
        from .core.crop_window import MAX_CROP_GROUND_WIDTH_M
        from .core.server_dials import dial

        ceiling = dial("manual.max_crop_ground_width_m", MAX_CROP_GROUND_WIDTH_M)
        return width_m >= ceiling * (1.0 - _CEILING_SLACK)

    def _crop_pixel_mapper(self, minx, miny, maxx, maxy, img_width, img_height):
        """Return a function turning a raster-CRS point into crop pixel [col, row]."""
        try:
            from rasterio import transform as rio_transform
            from rasterio.transform import from_bounds as transform_from_bounds

            clip = transform_from_bounds(minx, miny, maxx, maxy, img_width, img_height)

            def _mapper(point):
                row, col = rio_transform.rowcol(clip, point.x(), point.y())
                return [float(col), float(row)]

            return _mapper
        except ImportError:
            def _mapper(point):
                return [
                    (point.x() - minx) / (maxx - minx) * img_width,
                    (maxy - point.y()) / (maxy - miny) * img_height,
                ]

            return _mapper

    def _ensure_session(self, layer_name: str | None = None,
                        discard_unsaved: bool = False):
        """Ensure plugin has an active session. Returns (layer, error_dict_or_None).

        ``discard_unsaved`` allows a restart on another layer while a person
        has polygons saved but not exported. Default False, which refuses: the
        panel asks the user before throwing that work away, and this path has
        nobody to ask.
        """
        plugin = self._plugin

        # One resolution for the whole call, before anything looks at the open
        # session. A layer id first, then the name, and two layers sharing a
        # name are refused rather than guessed at, including when the session
        # already runs on one of them: an open session is no reason to read a
        # name one way here and another way everywhere else. The refusal also
        # covers the trap in the argument itself: it names the imagery to read,
        # and a caller who reads it as the output name learns why nothing was
        # found. Imported here, not at the top: mcp_api assembles this mixin,
        # so the import can only run once that module is built.
        target_layer = None
        if layer_name:
            from .mcp_api import raster_layer_by_id_or_name
            target_layer, layer_err = raster_layer_by_id_or_name(layer_name)
            if layer_err:
                return None, layer_err

        # Already active on the layer the caller means?
        current = getattr(plugin, "_current_layer", None)
        if current is not None:
            try:
                if target_layer is None or target_layer.id() == current.id():
                    # A session opened by a path that started no ledger would
                    # pass the save gate and export cloud-answered objects for
                    # free, so this path checks for one as well.
                    self._open_manual_ledger_if_missing()
                    return current, None
            except RuntimeError:
                pass

        # Find target layer
        if target_layer is None:
            dock = getattr(plugin, "dock_widget", None)
            if dock and hasattr(dock, "layer_combo"):
                target_layer = dock.layer_combo.currentLayer()
            if target_layer is None:
                for lyr in QgsProject.instance().mapLayers().values():
                    if isinstance(lyr, QgsRasterLayer):
                        target_layer = lyr
                        break

        if target_layer is None:
            return None, {"_error": "No raster layer available. The user needs to load one first."}

        # Starting a session clears every polygon the live one saved, and their
        # rubber bands with them. The panel asks before doing that; refuse here
        # instead, because the caller may be working over someone's shoulder.
        if not discard_unsaved and getattr(plugin, "saved_polygons", None):
            return None, {"_error": (
                f"{len(plugin.saved_polygons)} polygon(s) saved in the open "
                "session would be lost by starting a new one. Export them "
                "first, or call again with discard_unsaved=True."
            )}

        # Setup session programmatically (no UI)
        try:
            layer_name_safe = target_layer.name().replace(" ", "_")
            # RAW source, same as the UI start path (manual_workflow.
            # _on_start_segmentation): normcase lowercases and flips
            # separators, which destroys a GDAL URI source on Windows
            # (/vsicurl/, /vsizip/, GPKG:...:layer, NETCDF:"...":var).
            raster_path = target_layer.source()

            if hasattr(plugin, "_reset_session"):
                plugin._reset_session()
            # Same pairing as the button path: the reset leaves the refine
            # values on the shipped defaults, the panel on the user's own.
            dock = getattr(plugin, "dock_widget", None)
            if dock is not None and hasattr(dock, "publish_refine_settings"):
                dock.publish_refine_settings()

            plugin._current_layer = target_layer
            plugin._current_layer_name = layer_name_safe
            plugin._is_online_layer = plugin._needs_canvas_render(target_layer)

            if hasattr(plugin, "_is_layer_georeferenced"):
                plugin._is_non_georeferenced_mode = (
                    not plugin._is_online_layer and not plugin._is_layer_georeferenced(target_layer)
                )

            plugin._current_raster_path = raster_path

            # Headless QGIS has no iface and so no canvas CRS to convert from:
            # the caller's coordinates are then the raster's own.
            from qgis.utils import iface
            plugin._canvas_to_raster_xform = None
            plugin._raster_to_canvas_xform = None
            if iface is not None:
                canvas_crs = iface.mapCanvas().mapSettings().destinationCrs()
                raster_crs = target_layer.crs()
                if raster_crs and canvas_crs.isValid() and raster_crs.isValid():
                    if canvas_crs != raster_crs:
                        plugin._canvas_to_raster_xform = QgsCoordinateTransform(
                            canvas_crs, raster_crs, QgsProject.instance())
                        plugin._raster_to_canvas_xform = QgsCoordinateTransform(
                            raster_crs, canvas_crs, QgsProject.instance())

        except Exception as e:
            return None, {"_error": f"Failed to start session: {str(e)}"}

        if getattr(plugin, "_current_layer", None) is None:
            return None, {"_error": "Session failed to start."}

        self._open_manual_ledger_if_missing()

        return plugin._current_layer, None

    def _open_manual_ledger_if_missing(self) -> None:
        """Open the session's billing ledger, and only when there is none.

        This path builds its session by hand rather than through the panel, so
        it has to open the ledger itself. Without one a click routed to
        TerraLab's servers exports an object nobody paid for. Opening a second
        one over a live session would zero the spend and make every object it
        already charged billable again, so an open ledger is left alone. Opens
        nothing when the predictor in the slot is the on-device one.
        """
        plugin = self._plugin
        try:
            if getattr(plugin, "_manual_credit_ledger", None) is not None:
                return
            plugin._start_manual_credit_session()
        except Exception:  # nosec B110 -- a missing ledger never breaks a call
            pass

    def _save_refused_for_credits_quiet(self, billing_id) -> bool:
        """The panel's Save refusal, without the warning it puts on screen.

        The panel path ends in a message-bar warning and a full rebuild of the
        dock. A machine caller reads the refusal in ``_error``, so an agent's
        call must not make a warning pop up on someone's screen.
        """
        plugin = self._plugin
        try:
            from .core.manual_object_credit import save_affordable

            if not plugin._manual_save_is_billable(billing_id):
                return False
            if save_affordable(plugin._manual_credit_balance()):
                return False
        except (RuntimeError, AttributeError, ImportError):
            # Nothing to judge on quietly: take the panel gate rather than let
            # a billable save through unpaid.
            try:
                return bool(plugin._manual_save_refused_for_credits(billing_id))
            except (RuntimeError, AttributeError):
                return False
        # The balance behind this refusal can be minutes old, and the usual
        # reason it is wrong is the user having just paid. Read it again so the
        # next call is judged on a fresh one.
        try:
            plugin._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass
        return True

    def undo_last_point(self) -> dict:
        """Take back the last click of a session opened in the panel.

        Mirrors the panel's "Undo last point" button and Ctrl+Z: it removes the
        last point of the open click session, then the last frozen part, then
        the last object a person deleted.

        It has nothing to undo after :meth:`detect` or :meth:`detect_points`,
        because those two save the object as they make it and open no click
        session. This exists so an agent can take back a point a PERSON placed
        in the panel, on request.

        Returns
        -------
        dict
            ``{"undone": True}`` when the plugin ran its undo, or ``_error``
            when this build has no undo to run. Costs nothing.
        """
        plugin = self._plugin
        undo = getattr(plugin, "_on_undo", None)
        if not callable(undo):
            return {"_error": "This build has no undo for the click session."}
        try:
            undo()
        except Exception as err:  # noqa: BLE001 - the API never raises
            return {"_error": f"Undo failed: {err}"}
        return {"undone": True}
