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
    QgsFillSymbol,
    QgsGeometry,
    QgsProject,
    QgsSingleSymbolRenderer,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtGui import QColor

from .qt_compat import PolygonGeometry, WkbMultiPolygon, WkbPolygon

# Result-polygon color semantic, shared by both modes and the MCP path so the
# meaning lives in one place: RED = committed/saved deliverable, BLUE = a
# temporary, still-editable selection under review. Blue is BRAND_BLUE #1e88e5
# (30,136,229), matching the manual live selection and the Automatic zone outline.
_COMMITTED_RED = "220,0,0,255"
# Under-review fill + outline, matching the Manual selection (mask_rubber_band:
# fill 0,120,255,100 / stroke 0,80,200). A light-blue FILL plus a visible outline
# is essential when a dense run packs hundreds of objects together - a thin
# outline alone is unreadable there.
_REVIEW_FILL = "0,120,255,100"
_REVIEW_OUTLINE = "0,80,200,255"


def make_committed_renderer(
    outline_width: str = "0.5", color: QColor | None = None
) -> QgsSingleSymbolRenderer:
    """Renderer for a saved/committed result layer.

    Without a color: the historical red thin outline, no fill (kept intact so
    the MCP path and layers made by older flows stay byte-identical). With a
    color (the per-prompt committed color from output_store): a solid outline
    of that hue plus a light same-hue fill, readable over imagery at any zoom
    (the light fill marks coverage when zoomed out, the imagery stays
    judgeable through it when zoomed in).
    """
    if color is None:
        symbol = QgsFillSymbol.createSimple({
            "color": "0,0,0,0",
            "style": "no",
            "outline_color": _COMMITTED_RED,
            "outline_width": outline_width,
        })
        return QgsSingleSymbolRenderer(symbol)
    outline = color.darker(115)
    fill = QColor(color)
    fill.setAlpha(64)
    symbol = QgsFillSymbol.createSimple({
        "color": f"{fill.red()},{fill.green()},{fill.blue()},{fill.alpha()}",
        "style": "solid",
        "outline_color": f"{outline.red()},{outline.green()},{outline.blue()},255",
        "outline_width": "0.66",
        "outline_style": "solid",
    })
    return QgsSingleSymbolRenderer(symbol)


def make_review_renderer() -> QgsSingleSymbolRenderer:
    """Renderer for a temporary, still-editable selection: light-blue fill + a
    visible blue outline, the same design as the Manual selection mask.

    Blue marks "under review", reserving red for the saved export (see
    make_committed_renderer). The fill is what makes a dense run (hundreds of
    objects) readable; render-time simplification + the provider spatial index
    (applied by the caller) keep it smooth.
    """
    symbol = QgsFillSymbol.createSimple({
        "color": _REVIEW_FILL,
        "style": "solid",
        "outline_color": _REVIEW_OUTLINE,
        "outline_width": "0.4",
        "outline_style": "solid",
    })
    return QgsSingleSymbolRenderer(symbol)


def make_area_measurer(crs) -> QgsDistanceArea:
    """Build a QgsDistanceArea configured once for repeated measureArea() calls.

    setEllipsoid() loads ellipsoid parameters from the SRS database, so building
    a fresh measurer per feature (as geodesic_area_m2 does) costs seconds across
    thousands of polygons. Build this ONCE and reuse it in an export loop.
    Main-thread only (QgsProject).
    """
    measurer = QgsDistanceArea()
    project = QgsProject.instance()
    if crs is not None and crs.isValid():
        measurer.setSourceCrs(crs, project.transformContext())
    measurer.setEllipsoid(project.ellipsoid() or "EPSG:7030")
    return measurer


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

    A plain geom.area() returns degrees squared on EPSG:4326 rasters, which is
    meaningless to a surveyor. Main-thread only (QgsProject). For many features
    in a loop, build one make_area_measurer(crs) and call measureArea directly
    instead of paying this per-call ellipsoid load each time.
    """
    try:
        return float(make_area_measurer(crs).measureArea(geom))
    except Exception:
        return float(geom.area())


def _polygon_parts(geom: QgsGeometry) -> list[QgsGeometry]:
    """Flatten a geometry to its non-empty polygonal parts only.

    A GeometryCollection (e.g. from an ``intersection`` clip that grazes a
    boundary) can mix a polygon with stray lines/points; this keeps only the
    polygon parts so nothing non-areal survives downstream.
    """
    members = geom.asGeometryCollection() if geom.isMultipart() else [geom]
    return [
        part
        for part in members
        if not part.isEmpty() and part.type() == PolygonGeometry
    ]


def repair_polygon(geom: QgsGeometry) -> QgsGeometry | None:
    """Return a valid polygon-only geometry, repairing instead of dropping.

    Simplify/smooth can produce self-intersecting rings. Invisible on
    screen, but they break area math and downstream geoprocessing.
    makeValid (and an intersection clip) may return a collection with stray
    lines/points: keep only the polygon parts. The return is ALWAYS a
    (multi)polygon or None, never a collection/line/point, so a caller can
    feed it straight to a MultiPolygon layer.
    """
    if geom is None or geom.isEmpty():
        return None
    flat = QgsWkbTypes.flatType(geom.wkbType())
    pure_polygon = flat in (
        WkbPolygon,
        WkbMultiPolygon,
    )
    # A geometry collection can be GEOS-valid yet still carry non-polygon parts,
    # so the validity short-circuit only applies when the geometry is ALREADY a
    # pure (multi)polygon. Everything else is forced through part extraction.
    if pure_polygon and geom.isGeosValid():
        return geom
    fixed = geom if geom.isGeosValid() else geom.makeValid()
    if fixed is None or fixed.isEmpty():
        fixed = geom
    parts = _polygon_parts(fixed)
    if not parts:
        return None
    combined = QgsGeometry.collectGeometry(parts)
    if combined is None or combined.isEmpty():
        return None
    return combined


def to_multipolygon(geom: QgsGeometry) -> QgsGeometry | None:
    """Coerce any geometry into a MultiPolygon for a MultiPolygon layer.

    This is the LAST-line guard right before ``setGeometry``: a MultiPolygon
    provider rejects a feature whose geometry is a GeometryCollection (the
    "Could not add feature with geometry type GeometryCollection" error), so
    every detection is funneled through here. Drops non-polygonal parts and
    returns None when nothing areal remains (the caller skips that feature).
    """
    if geom is None or geom.isEmpty():
        return None
    flat = QgsWkbTypes.flatType(geom.wkbType())
    if flat == WkbMultiPolygon:
        return geom
    if flat == WkbPolygon:
        promoted = QgsGeometry(geom)
        promoted.convertToMultiType()
        return promoted
    parts = _polygon_parts(geom)
    if not parts:
        return None
    combined = QgsGeometry.collectGeometry(parts)
    if combined is None or combined.isEmpty():
        return None
    if not combined.isMultipart():
        combined.convertToMultiType()
    return combined


def apply_output_conventions(
    layer: QgsVectorLayer,
    source_raster_name: str,
    *,
    prompt: str = "",
    detail: int | None = None,
    confidence: float | None = None,
    created_iso: str = "",
    plugin_version: str = "",
) -> None:
    """Run-level provenance on the LAYER (Properties > Metadata) plus the
    style stored inside the GeoPackage, so the file opens styled and
    documented in any QGIS, with or without this plugin.

    The keyword args carry the run parameters that used to be encoded in the
    filename (prompt, detail, confidence cutoff); they land in the metadata
    abstract/keywords so the tree name can stay human-friendly. All optional:
    old call sites keep today's behavior unchanged. Metadata stays English.
    """
    created = created_iso or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        md = layer.metadata()
        md.setTitle(layer.name())
        lines = ["Polygons digitized with AI Segmentation (TerraLab)."]
        if prompt:
            lines.append(f"Object: {prompt}.")
        if source_raster_name:
            lines.append(f"Source raster: {source_raster_name}.")
        lines.append(f"Created: {created}.")
        if detail is not None:
            lines.append(f"Detail level: {detail}.")
        if confidence is not None:
            lines.append(f"Confidence cutoff: {confidence}.")
        try:
            count = int(layer.featureCount())
            if count >= 0:
                lines.append(f"Detections: {count}.")
        except Exception:  # nosec B110
            pass
        if plugin_version:
            lines.append(f"Plugin version: {plugin_version}.")
        md.setAbstract("\n".join(lines))
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
        elif name == "perimeter_m":
            try:
                values.append(
                    float(make_area_measurer(crs).measurePerimeter(geom)))
            except Exception:
                values.append(None)
        elif name == "area":
            values.append(float(geom.area()))
        elif name == "raster_source":
            values.append(raster_name)
        elif name == "created_at":
            values.append(timestamp)
        elif name == "label":
            values.append("")
        # "class"/"score" are per-detection facts of an Automatic run; a
        # manually appended polygon has neither, so they stay NULL (honest
        # unknown) via the fallthrough below.
        else:  # fid and any user-added column: let the provider default it
            values.append(None)
    return values
