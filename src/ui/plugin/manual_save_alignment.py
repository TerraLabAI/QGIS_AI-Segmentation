





















from __future__ import annotations

from qgis.core import Qgis, QgsCoordinateReferenceSystem, QgsMessageLog


def align_manual_saved_shape(controller, combined):


    try:
        if combined is None or combined.isEmpty():
            return combined
        if not getattr(controller, "_refine_ortho", False):
            return combined
        if getattr(controller, "_is_non_georeferenced_mode", False):

            return combined
        from ...core.detection_policy import manual_save_alignment_settings
        settings = manual_save_alignment_settings()
        if settings is None:
            return combined


        if combined.isMultipart() and len(combined.asMultiPolygon() or []) != 1:
            return combined
        crs = _manual_session_crs(controller)
        if crs is None:
            return combined
        from ...core.footprint_alignment import align_saved_footprint
        from ...core.layer_conventions import make_area_measurer
        aligned = align_saved_footprint(
            combined,
            _saved_neighbour_geoms(controller),
            settings,
            _ground_pixel_m(controller, combined),
            crs,
            make_area_measurer(crs))
        if aligned is None or aligned.isEmpty():
            return combined
        if not getattr(controller, "_save_alignment_logged", False):
            controller._save_alignment_logged = True
            QgsMessageLog.logMessage(
                "Semi-Auto save: footprints align to the session grid",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
        return aligned
    except Exception:  # noqa: BLE001
        return combined


def _manual_session_crs(controller) -> QgsCoordinateReferenceSystem | None:






    try:
        for definition in (_live_session_crs_definition(controller),
                           controller._manual_saved_crs_definition()):
            if definition:
                crs = QgsCoordinateReferenceSystem(definition)
                if crs.isValid():
                    return crs
        layer = getattr(controller, "_current_layer", None)
        if layer is not None and layer.crs().isValid():
            return layer.crs()
    except Exception:  # noqa: BLE001
        return None
    return None


def _live_session_crs_definition(controller) -> str:

    info = getattr(controller, "current_transform_info", None) or {}
    value = info.get("crs")
    return value.strip() if isinstance(value, str) else ""


def _ground_pixel_m(controller, combined) -> float:


    try:
        info = getattr(controller, "current_transform_info", None)
        if not info:
            return 0.0
        px_units = controller._crop_pixel_size_units(info)
        if px_units <= 0:
            return 0.0
        centre = combined.boundingBox().center()
        factor = controller._manual_metres_per_unit(centre.x(), centre.y())
        if factor is None or factor <= 0:
            return 0.0
        return px_units * factor
    except Exception:  # noqa: BLE001
        return 0.0


def _saved_neighbour_geoms(controller) -> list:





    session_crs = _live_session_crs_definition(controller)
    out = []
    for entry in getattr(controller, "saved_polygons", None) or []:
        geom = None
        try:
            entry_crs = (entry.get("transform_info") or {}).get("crs")
            if (session_crs and isinstance(entry_crs, str)
                    and entry_crs.strip() and entry_crs.strip() != session_crs):
                continue
            geom = controller._entry_geom(entry)
        except Exception:  # noqa: BLE001
            geom = None
        if geom is not None and not geom.isEmpty():
            out.append(geom)
    return out
