"""Output-layer conventions: geodesic measure, geometry repair, style
persistence and layer-level provenance metadata.

Mirrors the AI Edit plugin's conventions so both TerraLab plugins hand
professionals the same kind of deliverable: real square metres whatever
the CRS, valid geometries, a style that travels inside the GeoPackage,
and run-level provenance in Layer Properties > Metadata instead of being
repeated identically on every row.
"""
from __future__ import annotations

import time

from qgis.core import (
    Qgis,
    QgsDistanceArea,
    QgsGeometry,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)


def write_vector_layer(layer, file_path: str, options, transform_context=None):
    """Write a layer to disk, picking the newest writer the QGIS version has.

    writeAsVectorFormatV3 only exists since QGIS 3.20 (our current floor); on
    any older build it falls back to writeAsVectorFormatV2, which has the same
    leading (error_code, error_message) tuple shape. Keeping the branch means
    the code stays correct if the minimum version is lowered further later.
    Returns the writer's result tuple unchanged so callers keep reading
    result[0] / result[1].
    """
    if transform_context is None:
        transform_context = QgsProject.instance().transformContext()
    if Qgis.QGIS_VERSION_INT >= 32000:
        return QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, file_path, transform_context, options)
    return QgsVectorFileWriter.writeAsVectorFormatV2(
        layer, file_path, transform_context, options)


def geodesic_area_m2(geom: QgsGeometry, crs) -> float:
    """True square metres whatever the layer CRS (degrees included).

    A plain geom.area() returns degrees squared on EPSG:4326 rasters,
    which is meaningless to a surveyor. Main-thread only (QgsProject).
    """
    try:
        measurer = QgsDistanceArea()
        project = QgsProject.instance()
        if crs is not None and crs.isValid():
            measurer.setSourceCrs(crs, project.transformContext())
        measurer.setEllipsoid(project.ellipsoid() or "EPSG:7030")
        return float(measurer.measureArea(geom))
    except Exception:
        return float(geom.area())


def repair_polygon(geom: QgsGeometry) -> QgsGeometry | None:
    """Return a valid (multi)polygon, repairing instead of dropping.

    Simplify/smooth can produce self-intersecting rings. Invisible on
    screen, but they break area math and downstream geoprocessing.
    makeValid may return a collection with stray lines/points: keep only
    the polygon parts.
    """
    if geom is None or geom.isEmpty():
        return None
    if geom.isGeosValid():
        return geom
    fixed = geom.makeValid()
    if fixed is None or fixed.isEmpty():
        return geom
    parts = [
        part
        for part in (fixed.asGeometryCollection() if fixed.isMultipart() else [fixed])
        if not part.isEmpty()
        and part.type() == QgsWkbTypes.GeometryType.PolygonGeometry  # noqa: W503
    ]
    if not parts:
        return geom
    combined = QgsGeometry.collectGeometry(parts)
    if combined is None or combined.isEmpty():
        return geom
    return combined


def apply_output_conventions(layer: QgsVectorLayer, source_raster_name: str) -> None:
    """Run-level provenance on the LAYER (Properties > Metadata) plus the
    style stored inside the GeoPackage, so the file opens styled and
    documented in any QGIS, with or without this plugin."""
    try:
        md = layer.metadata()
        md.setTitle(layer.name())
        md.setAbstract(
            "Polygons digitized with AI Segmentation (TerraLab)."
            + (f" Source raster: {source_raster_name}." if source_raster_name else "")
        )
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        history = list(md.history())
        history.append(
            f"{created} segmented from '{source_raster_name}'"
            if source_raster_name
            else f"{created} segmented"
        )
        md.setHistory(history)
        layer.setMetadata(md)
    except Exception:  # nosec B110
        pass  # metadata is cosmetic, never block an export
    try:
        layer.saveStyleToDatabase(layer.name(), "AI Segmentation", True, "")
    except Exception:  # nosec B110
        pass  # style persistence is cosmetic, never block an export


def attribute_values_for_fields(fields, geom: QgsGeometry, crs, raster_name: str, timestamp: str) -> list:
    """Attribute list matching an existing layer's schema by field NAME, so
    appends keep working on layers created by any plugin version (older
    layers carry area/raster_source/created_at columns)."""
    values = []
    for field in fields:
        name = field.name().lower()
        if name == "area_m2":
            values.append(geodesic_area_m2(geom, crs))
        elif name == "area":
            values.append(float(geom.area()))
        elif name == "raster_source":
            values.append(raster_name)
        elif name == "created_at":
            values.append(timestamp)
        elif name == "label":
            values.append("")
        else:  # fid and any user-added column: let the provider default it
            values.append(None)
    return values
