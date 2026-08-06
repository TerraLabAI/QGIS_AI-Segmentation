









from __future__ import annotations

from datetime import datetime



_LANGUAGE = "ENG"
_RESOURCE_TYPE = "dataset"


def output_timestamp_iso() -> str:





    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_iso_from_epoch(epoch: float) -> str:


    try:
        return datetime.fromtimestamp(float(epoch)).astimezone().isoformat(
            timespec="seconds")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _crs_authid(crs) -> str:

    try:
        if crs is None or not crs.isValid():
            return ""
        return str(crs.authid() or "")
    except (RuntimeError, AttributeError):
        return ""


def _measures_in_ground_metres(crs) -> bool:

    try:
        from .layer_conventions import crs_measures_in_ground_metres

        return bool(crs_measures_in_ground_metres(crs))
    except Exception:  # noqa: BLE001
        return False


def coordinate_lines(layer_crs, source_crs_authid: str = "") -> list[str]:







    lines: list[str] = []
    authid = _crs_authid(layer_crs)
    source = str(source_crs_authid or "")
    if authid and source and source != authid:
        lines.append(f"Coordinates: {authid} (reprojected from {source}).")
    elif authid:
        lines.append(f"Coordinates: {authid}.")
    lines.append(
        "Areas and perimeters are ellipsoidal (geodesic) measures, not "
        "planar. A perimeter is the length of every ring of the object, "
        "its interior rings (holes) included."
    )
    if authid and not _measures_in_ground_metres(layer_crs):
        lines.append(
            "One unit of this CRS is not one metre on the ground, so a "
            "length read off the coordinates will not match the measure "
            "columns."
        )
    return lines


def build_abstract_lines(
    *,
    layer,
    source_raster_name: str,
    prompt: str,
    detail,
    confidence,
    created: str,
    plugin_version: str,
    basemap_label: str,
    source_crs_authid: str,
    overlapping_pairs,
) -> list[str]:

    lines = ["Polygons digitized with AI Segmentation (TerraLab)."]
    if prompt:
        lines.append(f"Object: {prompt}.")
    if source_raster_name:
        lines.append(f"Source raster: {source_raster_name}.")
    if basemap_label:
        lines.append(f"Imagery: {basemap_label}.")
    lines.append(f"Created: {created}.")
    if detail is not None:
        lines.append(f"Detail level: {detail}.")
    if confidence is not None:
        lines.append(f"Confidence cutoff: {confidence}.")
    try:
        count = int(layer.featureCount())
        if count >= 0:
            lines.append(f"Detections: {count}.")
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    if overlapping_pairs is not None:
        lines.append(f"Overlapping pairs: {int(overlapping_pairs)}.")
    try:
        lines.extend(coordinate_lines(layer.crs(), source_crs_authid))
    except (RuntimeError, AttributeError):  # nosec B110
        pass
    if plugin_version:
        lines.append(f"Plugin version: {plugin_version}.")
    return lines


def apply_layer_metadata(
    layer,
    *,
    source_raster_name: str = "",
    prompt: str = "",
    detail=None,
    confidence=None,
    created: str = "",
    plugin_version: str = "",
    basemap_label: str = "",
    source_crs_authid: str = "",
    overlapping_pairs=None,
) -> None:





    md = layer.metadata()
    md.setTitle(layer.name())
    md.setAbstract("\n".join(build_abstract_lines(
        layer=layer,
        source_raster_name=source_raster_name,
        prompt=prompt,
        detail=detail,
        confidence=confidence,
        created=created,
        plugin_version=plugin_version,
        basemap_label=basemap_label,
        source_crs_authid=source_crs_authid,
        overlapping_pairs=overlapping_pairs,
    )))
    keywords = [k for k in ("AI Segmentation", prompt, source_raster_name) if k]
    md.addKeywords("AI Segmentation", keywords)
    history = list(md.history())
    detected = f"detected '{prompt}'" if prompt else "segmented"
    history.append(
        f"{created} {detected} from '{source_raster_name}'"
        if source_raster_name
        else f"{created} {detected}"
    )
    md.setHistory(history)
    _set_identification(md, layer, basemap_label, created)
    layer.setMetadata(md)


def _set_identification(md, layer, basemap_label: str, created: str) -> None:






    try:
        md.setType(_RESOURCE_TYPE)
        md.setLanguage(_LANGUAGE)
    except (RuntimeError, AttributeError, TypeError):  # nosec B110
        pass
    if basemap_label:
        try:
            md.setRights([f"Imagery: {basemap_label}."])
        except (RuntimeError, AttributeError, TypeError):  # nosec B110
            pass
    try:
        md.setCrs(layer.crs())
    except (RuntimeError, AttributeError, TypeError):  # nosec B110
        pass
    _set_spatial_extent(md, layer)
    _set_creation_date(md, created)


def _set_spatial_extent(md, layer) -> None:

    try:
        from qgis.core import QgsBox3d, QgsLayerMetadata

        rect = layer.extent()
        if rect is None or rect.isEmpty():
            return
        spatial = QgsLayerMetadata.SpatialExtent()
        spatial.extentCrs = layer.crs()
        spatial.bounds = QgsBox3d(
            rect.xMinimum(), rect.yMinimum(), 0.0,
            rect.xMaximum(), rect.yMaximum(), 0.0,
        )
        extent = QgsLayerMetadata.Extent()
        extent.setSpatialExtents([spatial])
        md.setExtent(extent)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _set_creation_date(md, created: str) -> None:

    if not created:
        return
    try:
        from qgis.core import Qgis
        from qgis.PyQt.QtCore import QDateTime, Qt

        setter = getattr(md, "setDateTime", None)
        date_type = getattr(
            getattr(Qgis, "MetadataDateType", None), "Created", None)
        if setter is None or date_type is None:
            return
        stamp = QDateTime.fromString(created, Qt.DateFormat.ISODate)
        if stamp.isValid():
            setter(date_type, stamp)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


_DETECTION_LINE_PREFIX = "Detections: "


def refresh_detection_count(layer) -> bool:







    try:
        md = layer.metadata()
        abstract = str(md.abstract() or "")
        if _DETECTION_LINE_PREFIX not in abstract:
            return False
        count = int(layer.featureCount())
        if count < 0:
            return False
        lines = [
            f"{_DETECTION_LINE_PREFIX}{count}."
            if line.startswith(_DETECTION_LINE_PREFIX) else line
            for line in abstract.split("\n")
        ]
        md.setAbstract("\n".join(lines))
        layer.setMetadata(md)
        return True
    except Exception:  # noqa: BLE001
        return False
