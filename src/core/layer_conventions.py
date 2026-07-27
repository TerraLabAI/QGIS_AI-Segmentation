"""Output-layer conventions: geodesic measure, geometry repair, style
persistence and layer-level provenance metadata.

Mirrors the AI Edit plugin's conventions so both TerraLab plugins hand
professionals the same kind of deliverable: real square metres whatever
the CRS, coordinates written in a CRS where a metre is a ground metre,
valid geometries, a style that travels inside the GeoPackage, and
run-level provenance in Layer Properties > Metadata instead of being
repeated identically on every row.
"""
from __future__ import annotations

import time

from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsDistanceArea,
    QgsFillSymbol,
    QgsGeometry,
    QgsProject,
    QgsRendererCategory,
    QgsSingleSymbolRenderer,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtGui import QColor

from . import class_symbology
from .i18n import tr
from .qt_compat import (
    DistanceMeters,
    PolygonGeometry,
    WkbMultiPolygon,
    WkbPolygon,
)

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

# Measures are rounded when they are written, not when they are displayed: a
# GeoPackage REAL column carries no precision, so OGR reports 0 decimals for
# every double and a QgsField precision set on the memory layer is gone the
# moment the saved table is reloaded. Rounding at write time is the only form
# that survives into CSV, DXF and a colleague's spreadsheet. 1 cm2 and 1 cm are
# already finer than the imagery any run segments.
MEASURE_DECIMALS = 2

# Column headers a reader sees in place of the raw field names. English on
# purpose: the file travels to whoever the deliverable is for, and the layer
# metadata written beside it is English too.
EXPORT_FIELD_ALIASES = {
    "label": "Label",
    "class": "Class",
    "score": "Confidence",
    "area_m2": "Area (m²)",
    "perimeter_m": "Perimeter (m)",
}


def crs_measures_in_ground_metres(crs) -> bool:
    """Whether a straight coordinate difference in this CRS is a ground metre.

    False for a geographic CRS (the numbers are degrees) and for any Mercator,
    Pseudo-Mercator included: Mercator stretches by 1/cos(latitude), so at 49
    degrees north a 4 m car is stored 1.53 times too long. A deliverable
    measured or sent to CAD in those coordinates is wrong, silently.
    """
    try:
        if crs is None or not crs.isValid() or crs.isGeographic():
            return False
        if str(crs.projectionAcronym()).lower() == "merc":
            return False
        return crs.mapUnits() == DistanceMeters
    except (RuntimeError, AttributeError, TypeError):
        return False


def pick_output_crs(source_crs, extent, project_crs=None):
    """The CRS a saved layer should be written in.

    Keeps the source CRS whenever it already measures in ground metres, so a
    user working in a national grid gets their own grid back untouched. Only a
    run off a web basemap (Pseudo-Mercator) or a degrees raster is moved, to
    the project CRS when that one is metric, otherwise to the UTM zone under
    the data. Returns the source CRS unchanged if nothing better can be built.

    ``extent`` is a QgsRectangle in ``source_crs``; its centre picks the zone.
    """
    if crs_measures_in_ground_metres(source_crs):
        return source_crs
    if project_crs is None:
        try:
            project_crs = QgsProject.instance().crs()
        except (RuntimeError, AttributeError):
            project_crs = None
    if crs_measures_in_ground_metres(project_crs):
        return project_crs
    return _utm_crs_for_extent(source_crs, extent) or source_crs


def _utm_crs_for_extent(source_crs, extent):
    """The WGS84 UTM zone covering the centre of ``extent``, or None.

    UTM is the fallback because it exists everywhere and its scale error stays
    around 1 part in 2500, which is finer than the imagery this plugin reads.
    """
    from qgis.core import QgsCoordinateTransform, QgsPointXY

    try:
        if extent is None or extent.isEmpty():
            return None
        centre = QgsPointXY(extent.center())
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        if source_crs is not None and source_crs.isValid() and source_crs != wgs84:
            centre = QgsCoordinateTransform(
                source_crs, wgs84, QgsProject.instance()).transform(centre)
        lon, lat = centre.x(), centre.y()
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            return None
        zone = int((lon + 180.0) / 6.0) + 1
        zone = min(max(zone, 1), 60)
        code = (32600 if lat >= 0 else 32700) + zone
        utm = QgsCoordinateReferenceSystem(f"EPSG:{code}")
        return utm if utm.isValid() else None
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None


def round_measure(value) -> float | None:
    """One rounding rule for every area and perimeter written to a file.

    None in, None out, so an unknown measure stays an honest NULL instead of
    becoming 0.0.
    """
    if value is None:
        return None
    try:
        return round(float(value), MEASURE_DECIMALS)
    except (TypeError, ValueError):
        return None


def apply_export_field_aliases(layer) -> None:
    """Readable attribute-table headers on a saved layer.

    Aliases live in the layer style, so saveStyleToDatabase carries them into
    the GeoPackage and the file opens with readable columns in any QGIS, with
    or without this plugin. Must run BEFORE the style is saved. Unknown field
    names (a user-added column) keep their own name.
    """
    try:
        fields = layer.fields()
    except (RuntimeError, AttributeError):
        return
    for index, field in enumerate(fields):
        alias = EXPORT_FIELD_ALIASES.get(field.name().lower())
        if not alias:
            continue
        try:
            layer.setFieldAlias(index, alias)
        except (RuntimeError, AttributeError, TypeError):
            pass


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


def _class_category(value: str, hex_color: str, label: str) -> QgsRendererCategory:
    """One category symbol: light fill + a darker outline of the same hue,
    the same look make_committed_renderer uses for its per-prompt color."""
    color = QColor(hex_color)
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
    return QgsRendererCategory(value, symbol, label)


def make_class_categorized_renderer(
    layer: QgsVectorLayer, field: str = "class",
    *, max_categories: int | None = None,
) -> QgsCategorizedSymbolRenderer | None:
    """Categorized renderer keyed on field, one color per distinct value.

    Returns None when the field is missing or holds fewer than 2 distinct
    values, so the caller falls back to make_committed_renderer (the common
    one-prompt run keeps today's single-color look, unchanged). Values past
    max_categories share one fallback color instead of growing the legend
    without bound; that fallback is one "all other values" category when the
    running QGIS build supports it, else one category per folded value (still
    correctly colored, just more legend rows on that older-build path).
    """
    field_index = layer.fields().indexOf(field)
    if field_index < 0:
        return None
    try:
        raw_values = [str(v) for v in layer.uniqueValues(field_index) if v is not None]
    except Exception:  # nosec B110
        return None
    if not class_symbology.needs_categorized_renderer(raw_values):
        return None
    mapping = class_symbology.class_color_mapping(raw_values, max_categories=max_categories)
    # The fold color is read once so the mapping and the legend row for it can
    # never come from two different configurations.
    other_color = class_symbology.legend_other_color()
    kept = [v for v, c in mapping.items() if c != other_color]
    folded = [v for v, c in mapping.items() if c == other_color]
    categories = [_class_category(value, mapping[value], value) for value in kept]
    if folded:
        supports_else = hasattr(QgsRendererCategory, "setElseValue")
        if supports_else:
            other = _class_category("", other_color, tr("Other"))
            other.setElseValue(True)
            categories.append(other)
        else:
            categories.extend(
                _class_category(value, other_color, tr("Other"))
                for value in folded
            )
    return QgsCategorizedSymbolRenderer(field, categories)


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
    # project.ellipsoid() returns the truthy string "NONE" (not "") when
    # ellipsoidal measurement is disabled; both "NONE" and "" mean "measure
    # planar", which on a geographic CRS yields degrees^2 written as area_m2.
    # Fall back to a real ellipsoid so the geodesic areas stay truly metric.
    ellipsoid = project.ellipsoid()
    if not ellipsoid or ellipsoid.upper() == "NONE":
        ellipsoid = "EPSG:7030"
    measurer.setEllipsoid(ellipsoid)
    return measurer


def write_vector_layer(layer, file_path: str, options, transform_context=None):
    """Write a layer to disk with the one writer every supported build has.

    writeAsVectorFormatV3 exists since QGIS 3.20, below the declared floor, so
    there is nothing to fall back to. Returns the writer's result tuple
    unchanged so callers keep reading result[0] / result[1].
    """
    if transform_context is None:
        transform_context = QgsProject.instance().transformContext()
    return QgsVectorFileWriter.writeAsVectorFormatV3(
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
        if promoted.convertToMultiType():
            return promoted
        # In-place promotion refused: collect the part instead. Handing a
        # single Polygon back would let the provider reject the feature, and
        # this guard exists so that never happens.
        promoted = QgsGeometry.collectGeometry([geom])
        if promoted is None or promoted.isEmpty() or not promoted.isMultipart():
            return None
        return promoted
    parts = _polygon_parts(geom)
    if not parts:
        return None
    combined = QgsGeometry.collectGeometry(parts)
    if combined is None or combined.isEmpty():
        return None
    if not combined.isMultipart() and not combined.convertToMultiType():
        return None
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
    # Before the style is saved: aliases are part of the style, so this is the
    # only point where they still reach the GeoPackage.
    apply_export_field_aliases(layer)
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
        # setMetadata() only fills the in-memory layer property, which dies with
        # the QGIS session. Write it down as well so the provenance is still
        # there when the file is opened on its own, outside this project.
        save_metadata = getattr(layer, "saveDefaultMetadata", None)
        if save_metadata is not None:
            save_metadata()
    except Exception:  # nosec B110
        pass  # a read-only output keeps the in-memory metadata
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
            values.append(round_measure(geodesic_area_m2(geom, crs)))
        elif name == "perimeter_m":
            try:
                values.append(round_measure(
                    make_area_measurer(crs).measurePerimeter(geom)))
            except Exception:
                values.append(None)
        elif name == "area":
            values.append(round_measure(geom.area()))
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
