







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

from .mcp_api_guard import gui_thread_only




_CEILING_SLACK = 0.01






_CAPPED_RETRY_FACTOR = 8.0






_DETECT_RESPONSE_FORMATS = ("detailed", "concise")


def _detect_response_format(value) -> tuple[str | None, dict | None]:

    wanted = value.strip().lower() if isinstance(value, str) else ""
    if wanted in _DETECT_RESPONSE_FORMATS:
        return wanted, None
    from .mcp_api import not_found_error
    return None, not_found_error(
        "response format", str(value), list(_DETECT_RESPONSE_FORMATS))


def _capped_retry_factor() -> float:

    from .core.server_dials import dial_in_range

    return dial_in_range(
        "tuning.agent.capped_retry_factor", _CAPPED_RETRY_FACTOR, 2.0, 20.0)


class SegmentationManualMixin:


    @gui_thread_only
    def detect(
        self,
        x: float,
        y: float,
        layer_name: str | None = None,
        discard_unsaved: bool = False,
        output_dir: str | None = None,
        response_format: str = "detailed",
    ) -> dict:












































        try:
            px, py = float(x), float(y)
        except (TypeError, ValueError):
            return {"_error": f"x and y must be numbers, got ({x!r}, {y!r})."}
        if not (math.isfinite(px) and math.isfinite(py)):
            return {"_error": f"x and y must be finite numbers, got ({x}, {y})."}



        from .mcp_api import coerce_bool_param
        discard_unsaved, bool_err = coerce_bool_param("discard_unsaved", discard_unsaved)
        if bool_err:
            return bool_err
        answer_format, format_err = _detect_response_format(response_format)
        if format_err:
            return format_err

        return self._detect_from_points(
            [(px, py)], [], layer_name, discard_unsaved, output_dir,
            response_format=answer_format)

    @gui_thread_only
    def detect_points(
        self,
        positive: list[list[float]],
        negative: list[list[float]] | None = None,
        layer_name: str | None = None,
        discard_unsaved: bool = False,
        output_dir: str | None = None,
        response_format: str = "detailed",
    ) -> dict:



















































        pos, err = self._points_as_pairs(positive, "positive")
        if err:
            return err
        if not pos:
            return {"_error": "positive needs at least one [x, y] point."}
        neg, err = self._points_as_pairs(negative or [], "negative")
        if err:
            return err


        from .mcp_api import coerce_bool_param
        discard_unsaved, bool_err = coerce_bool_param("discard_unsaved", discard_unsaved)
        if bool_err:
            return bool_err
        answer_format, format_err = _detect_response_format(response_format)
        if format_err:
            return format_err
        return self._detect_from_points(
            pos, neg, layer_name, discard_unsaved, output_dir,
            response_format=answer_format)

    def _points_as_pairs(self, points, label: str):

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
        response_format: str = "detailed",
    ) -> dict:






        plugin = self._plugin

        if (getattr(plugin, "_encoding_in_progress", False)
                or getattr(plugin, "_headless", False)):
            return {"_error": "A Manual detection or image preparation is still running.", "busy": True}
        busy = self._review_mutation_error()
        if busy:
            return busy

        loaded_here = False
        if plugin.predictor is None:





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


        plugin._headless = True
        plugin._headless_error = None
        try:
            raster_pt = plugin._transform_to_raster_crs(QgsPointXY(px, py))
            if raster_pt is None:



                return {
                    "_error": f"Point ({px}, {py}) cannot be projected into the raster CRS "
                    f"({raster_layer.crs().authid()}). Pick a point closer to the imagery."
                }


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








            retried = False
            if facts["capped"] and getattr(plugin, "_is_online_layer", False):
                finer, finer_err = self._crop_and_predict(
                    raster_layer, raster_pt, positive, negative,
                    force_step=facts["step"] / _capped_retry_factor())
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





            billing_id = plugin._next_handoff_det_id()
            if self._save_refused_for_credits_quiet(billing_id):
                return {
                    "_error": "Monthly cloud objects used up. Saving an object "
                              "spends one while TerraLab's servers answer the "
                              "clicks. Turn cloud processing off in the panel to "
                              "work on this computer, or upgrade to Pro."
                }


            export_result = self.export_polygon(
                wkt, crs_authid, raster_layer.name(), output_dir)
            if export_result and "_error" not in export_result:




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


                from .core.server_dials import dial_text

                result["hint"] = dial_text(
                    "tuning.agent.hints", "manual_detected", 400) or (
                    "Outline too big or too small? Call detect_points() with "
                    "the same positive point plus a negative one on the part "
                    "to cut off."
                )
            elif export_result:


                result["export_error"] = export_result["_error"]
                from .core.server_dials import dial_text

                result["hint"] = dial_text(
                    "tuning.agent.hints", "manual_export_failed", 400) or (
                    "The outline is in polygon_wkt and nothing was written: "
                    "call export_polygon() with it once the folder is writable."
                )



            self._add_run_facts(result, loaded_here, crop_width_m,
                                retried, still_capped)
            if response_format == "concise":
                self._make_detect_answer_concise(result, combined, crs_authid)
            return result

        except Exception as e:
            import traceback



            from qgis.core import QgsMessageLog
            QgsMessageLog.logMessage(
                f"MCP detect failed: {e}\n{traceback.format_exc()}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical
            )
            return {"_error": f"Detection failed: {str(e)}"}
        finally:
            plugin._headless = False
            if getattr(plugin, "_unload_deferred", False):
                plugin._unload_deferred = False
                try:
                    plugin.unload()
                except Exception:  # noqa: BLE001
                    pass  # nosec B110

    @staticmethod
    def _make_detect_answer_concise(result: dict, geom, crs_authid: str) -> None:






        from qgis.core import QgsCoordinateReferenceSystem

        crs = QgsCoordinateReferenceSystem(crs_authid)
        digits = 7 if crs.isValid() and crs.isGeographic() else 2
        try:
            box = geom.boundingBox()
            result["bbox"] = [round(box.xMinimum(), digits), round(box.yMinimum(), digits),
                              round(box.xMaximum(), digits), round(box.yMaximum(), digits)]
            result["vertex_count"] = int(geom.constGet().nCoordinates())
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass
        try:
            from .core.layer_conventions import make_area_measurer
            result["area_m2"] = round(float(make_area_measurer(crs).measureArea(geom)), 2)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        if result.get("exported_layer") and "bbox" in result:
            result.pop("polygon_wkt", None)

    @staticmethod
    def _ground_width_in_metres(raster_layer, width_in_raster_units: float) -> float:



        from qgis.core import QgsUnitTypes



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






        if width_m <= 0:
            return False
        from .core.crop_window import MAX_CROP_GROUND_WIDTH_M
        from .core.server_dials import dial

        ceiling = dial("manual.max_crop_ground_width_m", MAX_CROP_GROUND_WIDTH_M)
        return width_m >= ceiling * (1.0 - _CEILING_SLACK)

    def _crop_pixel_mapper(self, minx, miny, maxx, maxy, img_width, img_height):

        try:
            from rasterio import transform as rio_transform
            from rasterio.transform import from_bounds as transform_from_bounds

            clip = transform_from_bounds(minx, miny, maxx, maxy, img_width, img_height)
            rio_transform.rowcol(clip, minx, maxy)

            def _mapper(point):
                row, col = rio_transform.rowcol(clip, point.x(), point.y())
                return [float(col), float(row)]

            return _mapper
        except Exception:  # noqa: BLE001
            def _mapper(point):
                return [
                    (point.x() - minx) / (maxx - minx) * img_width,
                    (maxy - point.y()) / (maxy - miny) * img_height,
                ]

            return _mapper

    def _ensure_session(self, layer_name: str | None = None,
                        discard_unsaved: bool = False):







        plugin = self._plugin










        target_layer = None
        if layer_name:
            from .mcp_api import raster_layer_by_id_or_name
            target_layer, layer_err = raster_layer_by_id_or_name(layer_name)
            if layer_err:
                return None, layer_err


        current = getattr(plugin, "_current_layer", None)
        if current is not None:
            try:
                if target_layer is None or target_layer.id() == current.id():



                    self._open_manual_ledger_if_missing()
                    return current, None
            except RuntimeError:
                pass


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




        unsaved_count = len(getattr(plugin, "saved_polygons", None) or [])
        if (getattr(plugin, "current_mask", None) is not None
                or getattr(plugin, "_frozen_sessions", None)
                or getattr(plugin, "_unfrozen_display_polygon", None) is not None):
            unsaved_count += 1
        if not discard_unsaved and unsaved_count:
            return None, {"_error": (
                f"{unsaved_count} polygon(s) or selections in the open "
                "session would be lost by starting a new one. Export them "
                "first, or call again with discard_unsaved=True."
            )}


        try:
            layer_name_safe = target_layer.name().replace(" ", "_")




            raster_path = target_layer.source()

            if hasattr(plugin, "_reset_session"):
                plugin._reset_session()


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









        plugin = self._plugin
        try:
            if getattr(plugin, "_manual_credit_ledger", None) is not None:
                return
            plugin._start_manual_credit_session()
        except Exception:  # nosec B110
            pass

    def _save_refused_for_credits_quiet(self, billing_id) -> bool:






        plugin = self._plugin
        try:
            from .core.manual_object_credit import save_affordable

            if not plugin._manual_save_is_billable(billing_id):
                return False
            if save_affordable(plugin._manual_credit_balance()):
                return False
        except (RuntimeError, AttributeError, ImportError):


            try:
                return bool(plugin._manual_save_refused_for_credits(billing_id))
            except (RuntimeError, AttributeError):
                return False



        try:
            plugin._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass
        return True

    def undo_last_point(self) -> dict:

















        plugin = self._plugin
        undo = getattr(plugin, "_on_undo", None)
        if not callable(undo):
            return {"_error": "This build has no undo for the click session."}
        try:
            undo()
        except Exception as err:  # noqa: BLE001
            return {"_error": f"Undo failed: {err}"}
        return {"undone": True}
