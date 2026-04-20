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
    QgsFillSymbol,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

# QGIS 4 rejects raw int for QgsField type arg (#25, #36); pick by version.
if getattr(Qgis, "QGIS_VERSION_INT", 0) >= 40000:
    from qgis.PyQt.QtCore import QMetaType as _QMetaType
    _FIELD_TYPE_STRING = _QMetaType.Type.QString
    _FIELD_TYPE_DOUBLE = _QMetaType.Type.Double
else:
    from qgis.PyQt.QtCore import QVariant as _QVariant
    _FIELD_TYPE_STRING = _QVariant.String
    _FIELD_TYPE_DOUBLE = _QVariant.Double

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
            pass

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
            small_enough = [i for i in range(len(scores)) if mask_areas[i] < 0.8 * total_pixels]
            if small_enough:
                best_idx = max(small_enough, key=lambda i: scores[i])
            else:
                best_idx = min(range(len(scores)), key=lambda i: mask_areas[i])

            mask = masks[best_idx]
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

            timestamp = datetime.now().isoformat(timespec="seconds")

            if existing_layer and existing_layer.dataProvider():
                try:
                    g = QgsGeometry(geom)
                    if not g.isMultipart():
                        g.convertToMultiType()
                    feature = QgsFeature(existing_layer.fields())
                    feature.setGeometry(g)
                    feature.setAttributes(["", g.area(), raster_name, timestamp])
                    existing_layer.dataProvider().addFeatures([feature])
                    existing_layer.updateExtents()
                    existing_layer.triggerRepaint()
                    return {
                        "layer_name": existing_layer.name(),
                        "file_path": existing_layer.source().split("|")[0],
                        "appended": True,
                    }
                except Exception:
                    pass

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
            pr = temp_layer.dataProvider()
            pr.addAttributes([
                QgsField("label", _FIELD_TYPE_STRING),
                QgsField("area", _FIELD_TYPE_DOUBLE),
                QgsField("raster_source", _FIELD_TYPE_STRING),
                QgsField("created_at", _FIELD_TYPE_STRING),
            ])
            temp_layer.updateFields()

            g = QgsGeometry(geom)
            if not g.isMultipart():
                g.convertToMultiType()
            feature = QgsFeature(temp_layer.fields())
            feature.setGeometry(g)
            feature.setAttributes(["", g.area(), raster_name, timestamp])
            pr.addFeatures([feature])
            temp_layer.updateExtents()

            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "GPKG"
            options.fileEncoding = "UTF-8"
            error = QgsVectorFileWriter.writeAsVectorFormatV3(
                temp_layer, gpkg_path,
                QgsProject.instance().transformContext(), options
            )
            if error[0] != QgsVectorFileWriter.NoError:
                return {"_error": f"Failed to save GeoPackage: {error[1]}"}

            result_layer = QgsVectorLayer(gpkg_path, layer_name, "ogr")
            if not result_layer.isValid():
                return {"_error": "Created GeoPackage but layer is invalid"}

            symbol = QgsFillSymbol.createSimple({
                "color": "0,0,0,0",
                "outline_color": "220,0,0,255",
                "outline_width": "0.5",
            })
            result_layer.setRenderer(QgsSingleSymbolRenderer(symbol))

            group = root.findGroup(seg_group_name)
            if group is None:
                group = root.insertGroup(0, seg_group_name)

            QgsProject.instance().addMapLayer(result_layer, False)
            group.addLayer(result_layer)

            return {"layer_name": layer_name, "file_path": gpkg_path}

        except Exception as e:
            return {"_error": f"Export failed: {str(e)}"}

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
