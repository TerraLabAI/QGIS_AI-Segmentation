"""Programmatic API for MCP integration. No UI, no dialogs, no cursors.

This module provides a stable public interface for AI agents to control
AI Segmentation without touching the human UI.
"""
from __future__ import annotations

import os
from datetime import datetime

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from .core.qt_compat import field_type_double, field_type_string

# QgsField type args (QGIS 4 rejects raw int, #25/#36): resolved once in
# qt_compat (QVariant on QGIS 3, QMetaType on QGIS 4).
_FIELD_TYPE_STRING = field_type_string()
_FIELD_TYPE_DOUBLE = field_type_double()

AISEG_KEYS = ["AI_Segmentation", "QGIS_AI-Segmentation", "QGIS_AI-Segmentation-Team"]
AISEG_REGISTER_URL = "https://terra-lab.ai/ai-segmentation?utm_source=qgis&utm_medium=mcp&utm_campaign=ai-agent"


def _find_plugin():
    import qgis.utils
    for key in AISEG_KEYS:
        plugin = qgis.utils.plugins.get(key)
        if plugin is not None:
            return plugin
    return None


class SegmentationMCPAPI:
    """Public API for MCP/headless access to AI Segmentation."""

    def __init__(self, plugin):
        self._plugin = plugin

    def get_status(self) -> dict:
        """Check plugin readiness without touching UI."""
        plugin = self._plugin

        status = {"installed": True}

        # Check model downloaded
        model_downloaded = False
        try:
            from .core.checkpoint_manager import checkpoint_exists
            model_downloaded = checkpoint_exists()
        except Exception:
            pass  # nosec B110

        predictor_loaded = plugin.predictor is not None

        if not model_downloaded and not predictor_loaded:
            status.update({
                "ready": False,
                "state": "MODEL_NOT_DOWNLOADED",
                "action_required": "Download the SAM model. Open the AI Segmentation panel and click Download.",
                "register_url": AISEG_REGISTER_URL,
            })
            return status

        if not predictor_loaded:
            status.update({
                "ready": False,
                "state": "MODEL_NOT_LOADED",
                "action_required": "Open the AI Segmentation panel and click 'Load Model'.",
            })
            return status

        status["model_loaded"] = True

        # Check raster layer
        raster_layer = getattr(plugin, "_current_layer", None)
        if raster_layer is None:
            dock = getattr(plugin, "dock_widget", None)
            if dock and hasattr(dock, "layer_combo"):
                raster_layer = dock.layer_combo.currentLayer()

        if raster_layer is None:
            available = []
            for lyr in QgsProject.instance().mapLayers().values():
                if isinstance(lyr, QgsRasterLayer):
                    available.append(lyr.name())
            if available:
                status.update({
                    "ready": False,
                    "state": "NO_RASTER_LAYER",
                    "model_loaded": True,
                    "action_required": (
                        f"No raster layer selected. Available: {', '.join(available)}."
                        " Pass layer_name to ai_segment_detect or select one in the panel."
                    ),
                    "available_raster_layers": available,
                })
            else:
                status.update({
                    "ready": False,
                    "state": "NO_RASTER_LAYER",
                    "model_loaded": True,
                    "action_required": "No raster layer in the project. The user needs to load one first.",
                    "available_raster_layers": [],
                })
            return status

        status.update({
            "ready": True,
            "state": "READY",
            "raster_layer": raster_layer.name(),
            "raster_extent": {
                "xmin": raster_layer.extent().xMinimum(),
                "ymin": raster_layer.extent().yMinimum(),
                "xmax": raster_layer.extent().xMaximum(),
                "ymax": raster_layer.extent().yMaximum(),
            },
            "raster_crs": raster_layer.crs().authid(),
        })

        # Additive fields: mode + credits. Never remove or rename existing keys.
        try:
            from .core.activation_manager import is_plugin_activated
            if is_plugin_activated():
                dock = getattr(plugin, "dock_widget", None)
                status["mode"] = "interactive"
                if dock and hasattr(dock, "_mode"):
                    status["mode"] = dock._mode.value
                if dock and hasattr(dock, "_auto_credits") and dock._auto_credits is not None:
                    status["auto_credits_remaining"] = dock._auto_credits
                if dock and hasattr(dock, "_auto_is_subscriber"):
                    status["auto_is_subscriber"] = dock._auto_is_subscriber
        except Exception:
            pass  # nosec B110 -- additive fields; never break existing behavior

        return status

    def detect(self, x: float, y: float, layer_name: str | None = None) -> dict:
        """Run SAM detection at a map point. Returns structured result or error."""
        plugin = self._plugin

        if plugin.predictor is None:
            return {"_error": "SAM model not loaded. Open the AI Segmentation panel and click 'Load Model'."}

        # Ensure session
        raster_layer, err = self._ensure_session(layer_name)
        if err:
            return err

        point = QgsPointXY(x, y)
        raster_pt = plugin._transform_to_raster_crs(point)

        # Check bounds for file-based layers
        is_online = getattr(plugin, "_is_online_layer", False)
        if not is_online and hasattr(plugin, "_is_point_in_raster_extent"):
            if not plugin._is_point_in_raster_extent(raster_pt):
                ext = raster_layer.extent()
                return {
                    "_error": f"Point ({x}, {y}) is outside the raster extent. "
                    f"Extent: xmin={ext.xMinimum():.2f}, ymin={ext.yMinimum():.2f}, "
                    f"xmax={ext.xMaximum():.2f}, ymax={ext.yMaximum():.2f} "
                    f"(CRS: {raster_layer.crs().authid()})."
                }

        # Enter headless mode
        plugin._headless = True
        plugin._headless_error = None
        try:
            # Check/encode crop
            crop_status = plugin._check_crop_status(raster_pt)
            if crop_status != "ok":
                encode_ok = plugin._handle_reencode(crop_status, raster_pt)
                if not encode_ok:
                    err_detail = plugin._headless_error or "Failed to encode image region."
                    return {"_error": f"Crop encoding failed: {err_detail}"}

            if plugin._current_crop_info is None:
                return {"_error": "No image region encoded. Try again or check the raster layer."}

            # Convert to pixel coords and predict
            import numpy as np
            crop_info = plugin._current_crop_info
            crop_bounds = crop_info["bounds"]
            img_shape = crop_info["img_shape"]
            img_height, img_width = img_shape
            minx, miny, maxx, maxy = crop_bounds

            try:
                from rasterio import transform as rio_transform
                from rasterio.transform import from_bounds as transform_from_bounds
                img_clip_transform = transform_from_bounds(minx, miny, maxx, maxy, img_width, img_height)
                row, col = rio_transform.rowcol(img_clip_transform, raster_pt.x(), raster_pt.y())
                point_coords = np.array([[col, row]])
            except ImportError:
                px_x = (raster_pt.x() - minx) / (maxx - minx) * img_width
                px_y = (maxy - raster_pt.y()) / (maxy - miny) * img_height
                point_coords = np.array([[px_x, px_y]])

            point_labels = np.array([1])

            masks, scores, low_res_masks = plugin.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

            if plugin._headless_error:
                return {"_error": plugin._headless_error}

            # Select best mask (avoid full-crop masks)
            total_pixels = masks[0].shape[0] * masks[0].shape[1]
            mask_areas = [int(m.sum()) for m in masks]
            small_enough = [i for i in range(len(scores)) if 0 < mask_areas[i] < 0.8 * total_pixels]
            if small_enough:
                best_idx = max(small_enough, key=lambda i: scores[i])
            else:
                best_idx = min(range(len(scores)), key=lambda i: mask_areas[i])

            # Keep only the real image area: reflect padding at raster edges
            # would otherwise leak mirrored polygons outside the raster.
            mask = masks[best_idx][:img_height, :img_width]
            score = float(scores[best_idx])

            if mask.sum() == 0:
                return {"detected": False, "score": score, "message": "No object detected at this point."}

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
                return {"detected": True, "score": score, "message": "Object detected but vectorization failed."}

            if len(polygons) == 1:
                combined = polygons[0]
            else:
                combined = QgsGeometry.unaryUnion(polygons)

            wkt = combined.asWkt()

            # Auto-export
            export_result = self.export_polygon(wkt, crs_authid, raster_layer.name())

            result = {
                "detected": True,
                "score": score,
                "polygon_wkt": wkt,
                "polygon_count": len(polygons),
                "crs": crs_authid,
                "mask_pixels": int(mask.sum()),
            }
            if export_result and "_error" not in export_result:
                result["exported_layer"] = export_result.get("layer_name")
                result["exported_file"] = export_result.get("file_path")

            return result

        except Exception as e:
            import traceback

            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                f"MCP detect failed: {e}\n{traceback.format_exc()}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical
            )
            return {"_error": f"Detection failed: {str(e)}"}
        finally:
            plugin._headless = False

    def export_polygon(self, geometry_wkt: str, crs: str, raster_name: str) -> dict:
        """Export a polygon to GeoPackage layer in the project."""
        try:
            crs_obj = QgsCoordinateReferenceSystem(crs)
            geom = QgsGeometry.fromWkt(geometry_wkt)
            if geom is None or geom.isEmpty():
                return {"_error": "Invalid geometry WKT"}

            # Determine output directory
            output_dir = QgsProject.instance().absolutePath()
            if not output_dir:
                output_dir = os.path.expanduser("~")

            # Find existing segmentation layer to append to
            seg_group_name = f"{raster_name} (AI Segmentation)"
            root = QgsProject.instance().layerTreeRoot()

            existing_layer = None
            for lyr in QgsProject.instance().mapLayers().values():
                if isinstance(lyr, QgsVectorLayer) and lyr.name().startswith("mask_"):
                    node = root.findLayer(lyr.id())
                    if node and node.parent() and node.parent().name() == seg_group_name:
                        existing_layer = lyr
                        break

            from .core.layer_conventions import (
                apply_output_conventions,
                attribute_values_for_fields,
                geodesic_area_m2,
                make_committed_renderer,
                repair_polygon,
                to_multipolygon,
            )

            timestamp = datetime.now().isoformat(timespec="seconds")

            if existing_layer and existing_layer.dataProvider():
                try:
                    g = QgsGeometry(geom)
                    g = repair_polygon(g) or g
                    # Coerce to polygon-only MultiPolygon so a collection can
                    # never reach the layer provider (it would be rejected).
                    g = to_multipolygon(g) or g
                    feature = QgsFeature(existing_layer.fields())
                    feature.setGeometry(g)
                    # Match the layer's schema by field name so appending
                    # works on layers created by any plugin version.
                    feature.setAttributes(attribute_values_for_fields(
                        existing_layer.fields(), g, existing_layer.crs(),
                        raster_name, timestamp,
                    ))
                    existing_layer.dataProvider().addFeatures([feature])
                    existing_layer.updateExtents()
                    existing_layer.triggerRepaint()
                    return {
                        "layer_name": existing_layer.name(),
                        "file_path": existing_layer.source().split("|")[0],
                        "appended": True,
                    }
                except Exception as e:
                    from qgis.core import QgsMessageLog
                    QgsMessageLog.logMessage(
                        f"Failed to append mask to existing layer, creating a new one: {e}",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning
                    )

            # Create new layer
            mask_num = 1
            for lyr in QgsProject.instance().mapLayers().values():
                if lyr.name().startswith("mask_"):
                    try:
                        num = int(lyr.name().split("_")[1])
                        mask_num = max(mask_num, num + 1)
                    except (IndexError, ValueError):
                        pass

            layer_name = f"mask_{mask_num}"
            gpkg_path = os.path.join(output_dir, f"{layer_name}.gpkg")
            counter = 1
            while os.path.exists(gpkg_path):
                gpkg_path = os.path.join(output_dir, f"{layer_name}_{counter}.gpkg")
                counter += 1

            temp_layer = QgsVectorLayer("MultiPolygon", layer_name, "memory")
            temp_layer.setCrs(crs_obj)
            # Minimal per-feature schema (editable label + geodesic measure);
            # run-level provenance goes in the layer metadata, not per row.
            pr = temp_layer.dataProvider()
            pr.addAttributes([
                QgsField("label", _FIELD_TYPE_STRING),
                QgsField("area_m2", _FIELD_TYPE_DOUBLE),
            ])
            temp_layer.updateFields()

            g = QgsGeometry(geom)
            g = repair_polygon(g) or g
            # Coerce to polygon-only MultiPolygon so a collection can never
            # reach the layer provider (it would be rejected).
            g = to_multipolygon(g) or g
            feature = QgsFeature(temp_layer.fields())
            feature.setGeometry(g)
            feature.setAttributes(["", geodesic_area_m2(g, crs_obj)])
            pr.addFeatures([feature])
            temp_layer.updateExtents()

            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "GPKG"
            options.fileEncoding = "UTF-8"
            from .core.layer_conventions import write_vector_layer
            error = write_vector_layer(
                temp_layer, gpkg_path, options,
                QgsProject.instance().transformContext(),
            )
            if error[0] != QgsVectorFileWriter.WriterError.NoError:
                return {"_error": f"Failed to save GeoPackage: {error[1]}"}

            # Open the table by its explicit name (a GPKG table defaults to
            # the file stem): a bare path leaves the sublayer choice to the
            # provider, which some GDAL/QGIS builds resolve differently and
            # then report the freshly written file as invalid.
            table = os.path.splitext(os.path.basename(gpkg_path))[0]
            result_layer = QgsVectorLayer(
                f"{gpkg_path}|layername={table}", layer_name, "ogr")
            if not result_layer.isValid():
                result_layer = QgsVectorLayer(gpkg_path, layer_name, "ogr")
            if not result_layer.isValid():
                return {"_error": "Created GeoPackage but layer is invalid"}

            result_layer.setRenderer(make_committed_renderer())
            # Style + provenance stored with the .gpkg (survives reloads).
            apply_output_conventions(result_layer, raster_name)

            group = root.findGroup(seg_group_name)
            if group is None:
                group = root.insertGroup(0, seg_group_name)

            QgsProject.instance().addMapLayer(result_layer, False)
            group.addLayer(result_layer)

            return {"layer_name": layer_name, "file_path": gpkg_path}

        except Exception as e:
            return {"_error": f"Export failed: {str(e)}"}

    def detect_auto(
        self,
        zone_wkt: str,
        object_class: str,
        layer_name: str | None = None,
        exemplars: list[dict] | None = None,
        detail: int | None = None,
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

        Returns
        -------
        dict with keys:
            "instances"     -- int, number of polygons detected
            "credits_used"  -- int, credits consumed
            "layer_name"    -- str, name of the output vector layer created.
                               Treat as opaque: it is a human-friendly name
                               like "Buildings (3 Jul)". Results are saved as
                               a table inside the project's
                               ai_segmentation.gpkg.
            "_error"        -- str, present only on failure
        """
        plugin = self._plugin

        from .core.detect_gate import can_detect

        has_text = bool(object_class and object_class.strip())
        has_exemplars = bool(exemplars)
        if not has_text and not has_exemplars:
            return {"_error": "object_class must be a non-empty string (or pass exemplars)."}
        # Reference-image detection needs at least two positive exemplars when
        # there is no text prompt: a single one detects poorly. Reject the weak
        # one-positive-no-text call up front with a clear error rather than
        # running (and billing) a poor detection. The run guard in
        # _start_auto_detection enforces the same rule as a backstop.
        positives = 0
        for ex in (exemplars or []):
            try:
                if int(ex.get("label", 1)) == 1:
                    positives += 1
            except (TypeError, ValueError, AttributeError):
                positives += 1  # malformed label defaults to positive
        if not can_detect(has_text, positives):
            return {"_error": (
                "Reference-image detection needs at least two positive exemplars "
                "(label 1) when object_class is empty. Add another example, or "
                "pass object_class."
            )}

        if not hasattr(plugin, "_run_auto_detect_headless"):
            return {
                "_error": (
                    "Automatic detection not available in this plugin version. "
                    "Upgrade to AI Segmentation 1.3.0+."
                )
            }

        # _run_auto_detect_headless switches mode itself; no need to refuse
        # just because the dock was in Interactive mode.
        return plugin._run_auto_detect_headless(
            zone_wkt=zone_wkt,
            object_class=(object_class or "").strip(),
            layer_name=layer_name,
            exemplars=exemplars,
            detail=detail,
        )

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
        mode_lower = mode.strip().lower() if mode else ""
        if mode_lower not in ("interactive", "automatic"):
            return {"_error": "mode must be 'interactive' or 'automatic'"}

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
            return {"_error": "Failed to switch mode: {}".format(str(e))}

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

        if not zone_wkt or not zone_wkt.strip():
            plugin._auto_zone = None
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

        bbox = geom.boundingBox()

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
                from .core import telemetry
                telemetry.track_auto_zone_too_large(area_km2=cap_area)
            except Exception:
                pass  # nosec B110
            from .ui.plugin.shared import free_zone_cap_km2
            return {"_error": (
                "Zone is {:.1f} km2; free trial zones go up to {:g} km2. "
                "Use a smaller zone, or subscribe to segment areas of "
                "any size.".format(cap_area, free_zone_cap_km2())
            )}

        if active_layer is not None:
            try:
                from qgis.utils import iface as _iface
                layer_crs = active_layer.crs()
                canvas_crs = _iface.mapCanvas().mapSettings().destinationCrs()
                if layer_crs.isValid() and canvas_crs.isValid() and layer_crs != canvas_crs:
                    xform = QgsCoordinateTransform(layer_crs, canvas_crs, QgsProject.instance())
                    bbox = xform.transformBoundingBox(bbox)
            except Exception:  # nosec B110 -- antimeridian, invalid CRS
                pass

        plugin._auto_zone = bbox
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
        """Cancel any running automatic detection.

        Returns
        -------
        dict with key "cancelled" (True).
        """
        plugin = self._plugin
        try:
            plugin._stop_auto_detection()
        except (RuntimeError, AttributeError):
            pass
        return {"cancelled": True}

    def _ensure_session(self, layer_name: str | None = None):
        """Ensure plugin has an active session. Returns (layer, error_dict_or_None)."""
        plugin = self._plugin

        # Already active on the right layer?
        current = getattr(plugin, "_current_layer", None)
        if current is not None:
            try:
                current.id()
                if layer_name and current.name() != layer_name:
                    pass  # need different layer
                else:
                    return current, None
            except RuntimeError:
                pass

        # Find target layer
        target_layer = None
        if layer_name:
            for lyr in QgsProject.instance().mapLayers().values():
                if isinstance(lyr, QgsRasterLayer) and lyr.name() == layer_name:
                    target_layer = lyr
                    break
            if target_layer is None:
                return None, {"_error": f"Raster layer '{layer_name}' not found."}
        else:
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

        # Setup session programmatically (no UI)
        try:
            layer_name_safe = target_layer.name().replace(" ", "_")
            # Resolve to a clean readable file path (decodes GDAL URI options /
            # subdatasets, "" for pathless layers so the canvas-render fallback
            # kicks in). Mirrors the UI start path.
            if hasattr(plugin, "_resolve_raster_file_path"):
                raster_path = plugin._resolve_raster_file_path(target_layer)
            else:
                raster_path = os.path.normcase(target_layer.source())

            if hasattr(plugin, "_reset_session"):
                plugin._reset_session()

            plugin._current_layer = target_layer
            plugin._current_layer_name = layer_name_safe
            plugin._is_online_layer = plugin._is_online_provider(target_layer)

            if hasattr(plugin, "_is_layer_georeferenced"):
                plugin._is_non_georeferenced_mode = (
                    not plugin._is_online_layer and not plugin._is_layer_georeferenced(target_layer)
                )

            plugin._current_raster_path = raster_path

            from qgis.utils import iface
            canvas_crs = iface.mapCanvas().mapSettings().destinationCrs()
            raster_crs = target_layer.crs()
            plugin._canvas_to_raster_xform = None
            plugin._raster_to_canvas_xform = None
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

        return plugin._current_layer, None
