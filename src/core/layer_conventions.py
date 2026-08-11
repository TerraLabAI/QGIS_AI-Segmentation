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
    Qgis,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsDistanceArea,
    QgsFillSymbol,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
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

# How far apart the two axes of a CRS may measure before ground_unit_aspect
# reports the difference at all. A correction this small moves a corner by well
# under a degree and a threshold by a quarter of a percent, neither of which any
# output shows, so a projected CRS reports a flat 1.0 and its shapes stay
# exactly as they were.
GROUND_ASPECT_DEAD_BAND = 0.01

# Bounds on the zone a run may move into a LOCAL projected CRS. A projected
# zone is true to the ground near its central meridian and drifts away from it,
# so a zone wider than this, or one this close to a pole, cannot use one.
RUN_CRS_MAX_SPAN_DEG = 3.0
RUN_CRS_MAX_LATITUDE = 80.0

# Where the UTM grid itself stops being defined. Asymmetric, and not the same
# question as the run bound above: this one says whether a UTM zone exists at
# all, that one says whether a run's tiles may ride in one.
UTM_MAX_LATITUDE_N = 84.0
UTM_MIN_LATITUDE_S = -80.0

# The frame a geographic run falls back to when no local projected CRS holds
# over its zone. Mercator stretches both axes by the same factor, so one of its
# pixels covers the same ground distance each way: the render is square on the
# ground, which is the only property the image needs. Its coordinates are NOT
# ground metres, and nothing reads them as such - every measure the run reports
# is taken on the ellipsoid. Without this a wide or polar geographic zone stays
# in degrees and hands the model an image whose north-south resolution is
# 1/cos(latitude) coarser than the user asked for.
RUN_CRS_SQUARE_PIXEL_FALLBACK = "EPSG:3857"

# How far one unit of a candidate run CRS may measure from one ground metre
# before the candidate is rejected. This catches a project CRS that measures in
# metres but belongs to another part of the world, which would otherwise be
# accepted on its units alone and stretch the whole run.
RUN_CRS_SCALE_TOLERANCE = 0.02

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

# -- what the server may change about an export -----------------------------
# Which CRS a run is written in is the decision behind every length and area a
# user reads off the file, and the three bounds below are what decides it. They
# are also the decision most likely to be wrong somewhere we have not looked:
# one report of a stretched or squashed export in an unusual part of the world
# is a value edit here, not a release everybody has to install. The aliases are
# on the same footing, because a column header can break the spreadsheet or the
# CAD import it lands in.
#
# Each reader is bounded, cache-only and fails to the shipped constant, so an
# absent or damaged configuration writes exactly what the plugin writes today.
# The bounds are not decoration: outside them the value stops describing a
# usable CRS at all, and an export written in a wrong CRS is silently wrong
# rather than visibly broken.

# Alias overrides are NOT served copy. Copy arrives in the user's language, and
# a header is deliberately English whatever the user's locale, because the file
# travels to whoever the deliverable is for. So the map is read raw, per key,
# and only for a field the plugin already writes.
_MAX_ALIAS_CHARS = 48


def measure_decimals() -> int:
    """Decimals a written measure is rounded to. 0 is legal, it writes whole
    units."""
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("export_policy.measure_decimals", MEASURE_DECIMALS, 0, 9))
    except Exception:  # noqa: BLE001 -- export policy is best-effort  # nosec B110
        return MEASURE_DECIMALS


def ground_aspect_dead_band() -> float:
    """How far apart two axes may measure before the difference is reported."""
    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "export_policy.ground_aspect_dead_band", GROUND_ASPECT_DEAD_BAND, 0.0, 0.5))
    except Exception:  # noqa: BLE001 -- export policy is best-effort  # nosec B110
        return GROUND_ASPECT_DEAD_BAND


def run_crs_enabled() -> bool:
    """Whether a run may move off the CRS its layer came in with.

    Off sends every run back to the layer's own CRS, which is what the plugin
    did before the move existed. The three dials below only narrow the move;
    this is the one that stops it, so a fleet-wide problem with it is a deploy
    rather than a release.
    """
    try:
        from .server_dials import feature_enabled

        return feature_enabled("run_crs")
    except Exception:  # noqa: BLE001 -- a kill switch is best-effort
        return True


def run_crs_max_span_deg() -> float:
    """Widest zone, in degrees, still moved into a projected CRS."""
    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "export_policy.run_crs.max_span_deg", RUN_CRS_MAX_SPAN_DEG, 0.1, 60.0))
    except Exception:  # noqa: BLE001 -- export policy is best-effort  # nosec B110
        return RUN_CRS_MAX_SPAN_DEG


def run_crs_max_latitude() -> float:
    """Latitude past which a zone keeps the CRS it came in with."""
    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "export_policy.run_crs.max_latitude", RUN_CRS_MAX_LATITUDE, 0.0, 90.0))
    except Exception:  # noqa: BLE001 -- export policy is best-effort  # nosec B110
        return RUN_CRS_MAX_LATITUDE


def run_crs_scale_tolerance() -> float:
    """How far one unit of a candidate CRS may measure from one ground metre
    before the candidate is refused."""
    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "export_policy.run_crs.scale_tolerance", RUN_CRS_SCALE_TOLERANCE, 0.0, 0.5))
    except Exception:  # noqa: BLE001 -- export policy is best-effort  # nosec B110
        return RUN_CRS_SCALE_TOLERANCE


def export_field_alias(field_name: str) -> str | None:
    """The column header for one field, served or shipped, None when neither.

    A served entry replaces the shipped header for that field alone, so one bad
    entry costs one column. A field the plugin does not write is ignored, which
    keeps the served map from adding headers to a schema it cannot see.
    """
    key = (field_name or "").lower()
    shipped = EXPORT_FIELD_ALIASES.get(key)
    if shipped is None:
        return None
    try:
        from .server_dials import read_value

        served = read_value("export_policy.field_aliases")
        if isinstance(served, dict):
            value = served.get(key)
            if isinstance(value, str):
                value = value.strip()[:_MAX_ALIAS_CHARS].strip()
                if value:
                    return value
    except Exception:  # noqa: BLE001 -- export policy is best-effort  # nosec B110
        pass
    return shipped


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

    The project CRS is taken only when its metre is a ground metre HERE, the
    same test ``pick_run_crs`` applies. A national grid belonging to another
    part of the world still reports metres, and reading its numbers under this
    data gives an area and a length that the file's own ``$area`` contradicts.

    ``extent`` is a QgsRectangle in ``source_crs``; its centre picks the zone.
    """
    if crs_measures_in_ground_metres(source_crs):
        return source_crs
    if project_crs is None:
        try:
            project_crs = QgsProject.instance().crs()
        except (RuntimeError, AttributeError):
            project_crs = None
    if (crs_measures_in_ground_metres(project_crs)
            and _crs_holds_ground_scale(project_crs, source_crs, extent)):
        return project_crs
    return _utm_crs_for_extent(source_crs, extent) or source_crs


def _utm_crs_for_extent(source_crs, extent):
    """The WGS84 UTM zone covering ``extent``, or None.

    UTM is the fallback because it exists everywhere and its scale error stays
    around 1 part in 2500, which is finer than the imagery this plugin reads.
    That only holds near the zone's own meridian, so the same two bounds
    ``pick_run_crs`` puts on a local projected zone apply here: a zone wider
    than the span bound, or one this close to a pole, gets no UTM CRS at all.
    Transverse Mercator runs away from the ground far from its meridian, and a
    rectangle drawn across the antimeridian arrives spanning the long way
    round, which is what the width test catches: without it the zone comes from
    a centre on the far side of the planet and the coordinates land nowhere.

    ``_wgs84_bounds`` transforms every source, geographic ones included,
    because a geographic CRS need not count degrees from Greenwich.
    """
    bounds = _wgs84_bounds(source_crs, extent)
    if bounds is None:
        return None
    lon_min, lat_min, lon_max, lat_max = bounds
    if (lon_max - lon_min) > run_crs_max_span_deg():
        return None
    # UTM's OWN validity, not the run bound. A run's tiles are held to a tighter
    # latitude than this, because a run measures ground per pixel and a saved
    # file measures area; borrowing that tighter bound here refused UTM between
    # 80 and 84 north, where UTM is defined, and left the file in the degrees or
    # the Mercator it came in with.
    if lat_max > UTM_MAX_LATITUDE_N or lat_min < UTM_MIN_LATITUDE_S:
        return None
    try:
        lon = (lon_min + lon_max) / 2.0
        lat = (lat_min + lat_max) / 2.0
        zone = int((lon + 180.0) / 6.0) + 1
        zone = min(max(zone, 1), 60)
        code = (32600 if lat >= 0 else 32700) + zone
        utm = QgsCoordinateReferenceSystem(f"EPSG:{code}")
        return utm if utm.isValid() else None
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None


def pick_run_crs(source_crs, extent, project_crs=None):
    """The CRS one detection run renders, tiles and measures in.

    A run held in a geographic CRS renders pixels that are square in degrees,
    which on the ground are rectangles: a degree of longitude shrinks towards
    the poles while a degree of latitude does not. The image handed to the
    model is squashed by that ratio, so a square roof arrives as a rectangle,
    and every distance the run measures is right on one axis and wrong on the
    other. Rendering into a ground-metre CRS instead makes the pixel square on
    the ground, which costs one reprojection QGIS already does on the fly.

    Only a CRS whose two axes disagree on the ground is moved, which in
    practice means a geographic one. Mercator is left alone on purpose: it
    stretches both axes by the same factor, so its pixels are already square on
    the ground and its image needs nothing. Only the meaning of its numbers is
    off, and every ground figure the run reports is measured, not read off the
    coordinates. Moving it would buy no image and cost a reprojection on every
    basemap tile.

    Returns ``source_crs`` unchanged when its axes already agree on the ground,
    which is the equator and every projected CRS: there is nothing to correct,
    so nothing moves and the grid, the tile count and the bill are what they
    were.

    A zone too wide or too polar for a local projected CRS does NOT keep its
    degrees. No local zone stays true across it, but Mercator is square on the
    ground everywhere it is valid, so the run moves there instead: the image is
    what the model is judged on, and a squashed image is the one outcome worth
    avoiding. Only a zone this cannot be proved for either - no usable bounds,
    or ground outside Mercator's own valid band - keeps the CRS it came in with.

    ``extent`` is a QgsRectangle in ``source_crs``, and should be the zone the
    run covers rather than the whole layer: both the width test and the scale
    test answer for the ground actually being rendered.

    Switched off (``features.run_crs``), every run keeps the CRS it came in
    with. That is the behaviour the plugin shipped for months, so the switch
    has a known-good off state: it costs the square-pixel image back and
    nothing else. It is read here, at the one place the decision is made, and
    the caller memoizes the answer per zone, so a configuration refresh cannot
    move a run that is already estimated or under way.
    """
    if not run_crs_enabled():
        return source_crs
    if crs_measures_in_ground_metres(source_crs):
        return source_crs
    try:
        centre = extent.center()
    except (RuntimeError, AttributeError):
        return source_crs
    if ground_unit_aspect(source_crs, centre.x(), centre.y()) == 1.0:
        return source_crs
    bounds = _wgs84_bounds(source_crs, extent)
    if bounds is None:
        # Where the zone sits is unknown, so neither candidate can be proved
        # here. Nothing moves.
        return source_crs
    lon_min, lat_min, lon_max, lat_max = bounds
    # Every branch below is a refusal of the LOCAL zone, and each one falls
    # back to the square-pixel frame rather than to degrees. It is computed at
    # the point of refusal so the ordinary path pays nothing for it.
    if (lon_max - lon_min) > run_crs_max_span_deg():
        return _square_pixel_crs(source_crs, bounds) or source_crs
    if max(abs(lat_min), abs(lat_max)) > run_crs_max_latitude():
        return _square_pixel_crs(source_crs, bounds) or source_crs
    candidate = pick_output_crs(source_crs, extent, project_crs)
    try:
        if candidate is None or not candidate.isValid() or candidate == source_crs:
            return _square_pixel_crs(source_crs, bounds) or source_crs
        # The run carries its CRS to the worker and into the archive as an
        # authority code and nothing else, so a CRS that has none cannot be
        # rebuilt on the far side. A project CRS defined by a bare WKT is the
        # real case. Taking it here would render the tiles in one CRS while
        # every extent stayed in another, and every tile would still be billed.
        if not candidate.authid():
            return _square_pixel_crs(source_crs, bounds) or source_crs
    except (RuntimeError, AttributeError):
        return _square_pixel_crs(source_crs, bounds) or source_crs
    if not _crs_holds_ground_scale(candidate, source_crs, extent):
        return _square_pixel_crs(source_crs, bounds) or source_crs
    return candidate


def _square_pixel_crs(source_crs, bounds):
    """A frame with square GROUND pixels for a zone no local projected CRS fits.

    ``bounds`` is the zone as (lon_min, lat_min, lon_max, lat_max) in WGS84.
    Returns None when the frame cannot be PROVED square over that zone, which
    leaves the caller with the CRS it came in with.

    Mercator is the frame because it is valid worldwide and stretches both axes
    by the same factor, so a pixel of it covers the same ground distance each
    way. Its units are not ground metres and nothing here pretends they are:
    the run measures on the ellipsoid, exactly as it does for the web basemap
    runs that are already in this CRS.

    Two things are checked rather than assumed. The zone must sit inside the
    frame's own declared bounds, because Mercator runs to infinity at the poles
    and a transform past its band still answers, with a clamped coordinate that
    is not the zone. And the frame must measure square at the zone's centre,
    which is the property being bought.
    """
    from qgis.core import QgsCoordinateTransform

    lon_min, lat_min, lon_max, lat_max = bounds
    try:
        candidate = QgsCoordinateReferenceSystem(RUN_CRS_SQUARE_PIXEL_FALLBACK)
        if not candidate.isValid() or candidate == source_crs:
            return None
        if not candidate.authid():
            return None
        valid = candidate.bounds()
        if valid is None or valid.isEmpty():
            return None
        if not (valid.xMinimum() <= lon_min and lon_max <= valid.xMaximum()
                and valid.yMinimum() <= lat_min and lat_max <= valid.yMaximum()):
            return None
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        centre = QgsCoordinateTransform(
            wgs84, candidate, QgsProject.instance()
        ).transform(QgsPointXY((lon_min + lon_max) / 2.0,
                               (lat_min + lat_max) / 2.0))
        if ground_unit_aspect(candidate, centre.x(), centre.y()) != 1.0:
            return None
        return candidate
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None


def _wgs84_bounds(source_crs, extent):
    """``extent`` as (lon_min, lat_min, lon_max, lat_max), or None.

    None whenever the answer would be a guess: no extent, a failed transform,
    or coordinates off the globe. A rectangle cannot straddle the antimeridian
    (it would need xMin above xMax), so a zone drawn across it arrives here
    spanning the long way round and is rejected by its width.

    Every source is transformed, geographic ones included, because a CRS being
    geographic says nothing about its numbers being degrees counted from
    Greenwich: some national CRS count grads, or count from Paris or Rome.
    """
    from qgis.core import QgsCoordinateTransform

    try:
        if extent is None or extent.isEmpty():
            return None
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        rect = extent
        if source_crs is not None and source_crs.isValid() and source_crs != wgs84:
            rect = QgsCoordinateTransform(
                source_crs, wgs84, QgsProject.instance()
            ).transformBoundingBox(extent)
        lon_min, lon_max = rect.xMinimum(), rect.xMaximum()
        lat_min, lat_max = rect.yMinimum(), rect.yMaximum()
        if not (-180.0 <= lon_min <= lon_max <= 180.0):
            return None
        if not (-90.0 <= lat_min <= lat_max <= 90.0):
            return None
        return lon_min, lat_min, lon_max, lat_max
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None


def _crs_holds_ground_scale(candidate, source_crs, extent) -> bool:
    """Whether one unit of ``candidate`` is one ground metre over ``extent``.

    Asks the candidate itself rather than trusting its declared units, which is
    what catches a metric CRS belonging to another part of the world: it still
    reports metres, and its metres are still the wrong length here.
    """
    from qgis.core import QgsCoordinateTransform

    try:
        centre = QgsPointXY(extent.center())
        if source_crs is not None and source_crs.isValid() and source_crs != candidate:
            centre = QgsCoordinateTransform(
                source_crs, candidate, QgsProject.instance()).transform(centre)
        along_x, along_y = ground_unit_metres(candidate, centre.x(), centre.y())
        tolerance = run_crs_scale_tolerance()
        return (abs(along_x - 1.0) <= tolerance
                and abs(along_y - 1.0) <= tolerance)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return False


def round_measure(value) -> float | None:
    """One rounding rule for every area and perimeter written to a file.

    None in, None out, so an unknown measure stays an honest NULL instead of
    becoming 0.0.
    """
    if value is None:
        return None
    try:
        return round(float(value), measure_decimals())
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
        alias = export_field_alias(field.name())
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


def ground_unit_metres(crs, ref_x: float, ref_y: float) -> tuple[float, float]:
    """Ground metres covered by one x unit and one y unit of ``crs``, near (x, y).

    (1.0, 1.0) on any failure, and on a CRS whose units are already ground
    metres the two come back as 1.0 to within measurement noise. Both are
    measured on the ellipsoid, so the answer holds wherever the point is.
    """
    try:
        if crs is None or not crs.isValid():
            return 1.0, 1.0
        step = 0.001 if crs.isGeographic() else 1.0
        measurer = make_area_measurer(crs)
        along_x = float(measurer.measureLine(
            QgsPointXY(ref_x, ref_y), QgsPointXY(ref_x + step, ref_y)))
        along_y = float(measurer.measureLine(
            QgsPointXY(ref_x, ref_y), QgsPointXY(ref_x, ref_y + step)))
        if along_x > 0.0 and along_y > 0.0:
            return along_x / step, along_y / step
    except Exception:  # noqa: BLE001 -- an unusable CRS answers for nothing  # nosec B110
        pass
    return 1.0, 1.0


def ground_unit_aspect(crs, ref_x: float, ref_y: float) -> float:
    """How much longer one y unit of ``crs`` is than one x unit, near (x, y).

    1.0 wherever the two axes cover the same ground distance, which is every
    projected CRS this plugin meets. In a geographic CRS it climbs with
    latitude, because a degree of longitude shrinks towards the poles while a
    degree of latitude does not: around 1.6 at 52 degrees, over 2 in northern
    Scandinavia. A local image supplied in degrees is common, so anything that
    reads raw coordinates as if they were distances (squaring a footprint,
    buffering by a width) is wrong there by exactly this factor.

    Exactly 1.0 whenever the two axes measure within GROUND_ASPECT_DEAD_BAND of
    each other, so a projected CRS costs its callers no correction at all: the
    ellipsoidal measure lands a hair off 1 in Web Mercator, and a hair is not a
    reason to move anybody's shapes. Also 1.0 on any failure, which is the
    behaviour of code that never asked.
    """
    along_x, along_y = ground_unit_metres(crs, ref_x, ref_y)
    if along_x > 0.0 and along_y > 0.0:
        aspect = along_y / along_x
        if abs(aspect - 1.0) >= ground_aspect_dead_band():
            return aspect
    return 1.0


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

    RETURNS THE ARGUMENT ITSELF when it is already a MultiPolygon, so the
    result is NOT the caller's to mutate. A caller that transforms, translates
    or otherwise edits what comes back must clone the inner geometry first
    (``QgsGeometry(g.constGet().clone())``; the plain ``QgsGeometry(g)`` copy is
    shallow and shares the same abstract geometry). The autosave broke this
    once: it reprojected the result and so moved the merger's live keepers into
    another CRS, which the rest of the run then measured in the CRS it still
    declared.
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


def _log_convention_failure(step: str, err: Exception) -> None:
    """One Warning line for a convention step that could not be applied.

    Every step here is best-effort, so none of them stops the export. Written
    down all the same: a GeoPackage delivered without its column headers or its
    style otherwise looks like the plugin never tried.
    """
    try:
        QgsMessageLog.logMessage(
            f"Export conventions: {step} failed: {err}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
    except Exception:  # noqa: BLE001 -- logging must never raise  # nosec B110
        pass


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
    except Exception as err:  # noqa: BLE001 -- metadata never blocks an export
        # Never blocks the export, but never silent either: a delivered file
        # that lost its provenance has to leave a trace somewhere.
        _log_convention_failure("provenance metadata", err)
    try:
        # setMetadata() only fills the in-memory layer property, which dies with
        # the QGIS session. Write it down as well so the provenance is still
        # there when the file is opened on its own, outside this project.
        save_metadata = getattr(layer, "saveDefaultMetadata", None)
        if save_metadata is not None:
            save_metadata()
    except Exception as err:  # noqa: BLE001 -- a read-only output keeps the in-memory copy
        _log_convention_failure("saving the metadata into the file", err)
    try:
        layer.saveStyleToDatabase(layer.name(), "AI Segmentation", True, "")
    except Exception as err:  # noqa: BLE001 -- style persistence never blocks an export
        _log_convention_failure("saving the style into the file", err)


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
