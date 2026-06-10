









from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsDistanceArea,
    QgsField,
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
    field_type_double,
)





_COMMITTED_RED = "220,0,0,255"




_REVIEW_FILL = "0,120,255,100"
_REVIEW_OUTLINE = "0,80,200,255"







MEASURE_DECIMALS = 2






GROUND_ASPECT_DEAD_BAND = 0.01




RUN_CRS_MAX_SPAN_DEG = 3.0
RUN_CRS_MAX_LATITUDE = 80.0




UTM_MAX_LATITUDE_N = 84.0
UTM_MIN_LATITUDE_S = -80.0









RUN_CRS_SQUARE_PIXEL_FALLBACK = "EPSG:3857"





RUN_CRS_SCALE_TOLERANCE = 0.02




EXPORT_FIELD_ALIASES = {
    "det_id": "ID",
    "class": "Class",
    "confidence": "Confidence",
    "area_m2": "Area (m²)",
    "perimeter_m": "Perimeter (m)",


    "perim_m": "Perimeter (m)",


    "score": "Confidence",
    "label": "Label",
}




















_MAX_ALIAS_CHARS = 48


def measure_decimals() -> int:


    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("export_policy.measure_decimals", MEASURE_DECIMALS, 0, 9))
    except Exception:  # noqa: BLE001  # nosec B110
        return MEASURE_DECIMALS


def ground_aspect_dead_band() -> float:

    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "export_policy.ground_aspect_dead_band", GROUND_ASPECT_DEAD_BAND, 0.0, 0.5))
    except Exception:  # noqa: BLE001  # nosec B110
        return GROUND_ASPECT_DEAD_BAND


def run_crs_enabled() -> bool:







    try:
        from .server_dials import feature_enabled

        return feature_enabled("run_crs")
    except Exception:  # noqa: BLE001
        return True


def run_crs_max_span_deg() -> float:

    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "export_policy.run_crs.max_span_deg", RUN_CRS_MAX_SPAN_DEG, 0.1, 60.0))
    except Exception:  # noqa: BLE001  # nosec B110
        return RUN_CRS_MAX_SPAN_DEG


def run_crs_max_latitude() -> float:

    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "export_policy.run_crs.max_latitude", RUN_CRS_MAX_LATITUDE, 0.0, 90.0))
    except Exception:  # noqa: BLE001  # nosec B110
        return RUN_CRS_MAX_LATITUDE


def run_crs_scale_tolerance() -> float:


    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "export_policy.run_crs.scale_tolerance", RUN_CRS_SCALE_TOLERANCE, 0.0, 0.5))
    except Exception:  # noqa: BLE001  # nosec B110
        return RUN_CRS_SCALE_TOLERANCE


def export_field_alias(field_name: str) -> str | None:






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
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return shipped






MIN_LAYER_GROUND_SPAN_M = 1.0


def crs_disagrees_with_extent(crs, extent) -> str:
















    try:
        if crs is None or not crs.isValid() or extent is None or extent.isEmpty():
            return ""
        width = abs(float(extent.width()))
        height = abs(float(extent.height()))
        if width <= 0.0 or height <= 0.0:
            return ""
        looks_geographic = (
            abs(float(extent.xMinimum())) <= 180.0
            and abs(float(extent.xMaximum())) <= 180.0
            and abs(float(extent.yMinimum())) <= 90.0
            and abs(float(extent.yMaximum())) <= 90.0
        )
        if crs.isGeographic():



            return "" if looks_geographic else "metres_in_geographic_crs"
        if not crs_measures_in_ground_metres(crs):



            return ""
        if min(width, height) >= MIN_LAYER_GROUND_SPAN_M:
            return ""
        return "degrees_in_metric_crs" if looks_geographic else ""
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return ""


def crs_measures_in_ground_metres(crs) -> bool:







    try:
        if crs is None or not crs.isValid() or crs.isGeographic():
            return False
        if str(crs.projectionAcronym()).lower() == "merc":
            return False
        return crs.mapUnits() == DistanceMeters
    except (RuntimeError, AttributeError, TypeError):
        return False


def _project_transform_context(transform_context=None):






    if transform_context is not None:
        return transform_context
    return QgsProject.instance().transformContext()


def pick_output_crs(source_crs, extent, project_crs=None,
                    transform_context=None, ellipsoid=None):



















    if crs_measures_in_ground_metres(source_crs):
        return source_crs
    if project_crs is None:
        try:
            project_crs = QgsProject.instance().crs()
        except (RuntimeError, AttributeError):
            project_crs = None
    if (crs_measures_in_ground_metres(project_crs)
            and _crs_holds_ground_scale(project_crs, source_crs, extent,
                                        transform_context, ellipsoid)):
        note_output_crs_choice(source_crs, project_crs)
        return project_crs
    chosen = _utm_crs_for_extent(
        source_crs, extent, transform_context) or source_crs
    note_output_crs_choice(source_crs, chosen)
    return chosen




_logged_output_crs_pairs: set[tuple[str, str]] = set()


def note_output_crs_choice(source_crs, chosen_crs) -> None:









    try:
        source = str(source_crs.authid() or "") if source_crs else ""
        chosen = str(chosen_crs.authid() or "") if chosen_crs else ""
    except (RuntimeError, AttributeError):
        return
    if not source or not chosen:
        return
    key = (source, chosen)
    if key in _logged_output_crs_pairs:
        return
    _logged_output_crs_pairs.add(key)
    stayed = source == chosen
    message = (
        f"Output CRS: {source} kept, no ground-metre CRS covers this extent; "
        "coordinates are not ground metres"
        if stayed
        else f"Output CRS: coordinates written in {chosen}, read from {source}"
    )
    try:
        QgsMessageLog.logMessage(
            message, "AI Segmentation",
            level=(Qgis.MessageLevel.Warning if stayed
                   else Qgis.MessageLevel.Info),
        )
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _utm_crs_for_extent(source_crs, extent, transform_context=None):















    bounds = _wgs84_bounds(source_crs, extent, transform_context)
    if bounds is None:
        return None
    lon_min, lat_min, lon_max, lat_max = bounds
    if (lon_max - lon_min) > run_crs_max_span_deg():
        return None





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


        return source_crs
    lon_min, lat_min, lon_max, lat_max = bounds



    if (lon_max - lon_min) > run_crs_max_span_deg():
        return _square_pixel_crs(source_crs, bounds) or source_crs
    if max(abs(lat_min), abs(lat_max)) > run_crs_max_latitude():
        return _square_pixel_crs(source_crs, bounds) or source_crs
    candidate = pick_output_crs(source_crs, extent, project_crs)
    try:
        if candidate is None or not candidate.isValid() or candidate == source_crs:
            return _square_pixel_crs(source_crs, bounds) or source_crs





        if not candidate.authid():
            return _square_pixel_crs(source_crs, bounds) or source_crs
    except (RuntimeError, AttributeError):
        return _square_pixel_crs(source_crs, bounds) or source_crs
    if not _crs_holds_ground_scale(candidate, source_crs, extent):
        return _square_pixel_crs(source_crs, bounds) or source_crs
    return candidate


def _square_pixel_crs(source_crs, bounds):


















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


def _wgs84_bounds(source_crs, extent, transform_context=None):











    from qgis.core import QgsCoordinateTransform

    try:
        if extent is None or extent.isEmpty():
            return None
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        rect = extent
        if source_crs is not None and source_crs.isValid() and source_crs != wgs84:
            rect = QgsCoordinateTransform(
                source_crs, wgs84, _project_transform_context(transform_context)
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


def _crs_holds_ground_scale(candidate, source_crs, extent,
                            transform_context=None, ellipsoid=None) -> bool:






    from qgis.core import QgsCoordinateTransform

    try:
        centre = QgsPointXY(extent.center())
        if source_crs is not None and source_crs.isValid() and source_crs != candidate:
            centre = QgsCoordinateTransform(
                source_crs, candidate,
                _project_transform_context(transform_context)).transform(centre)
        along_x, along_y = ground_unit_metres(
            candidate, centre.x(), centre.y(), transform_context, ellipsoid)
        tolerance = run_crs_scale_tolerance()
        return (abs(along_x - 1.0) <= tolerance
                and abs(along_y - 1.0) <= tolerance)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return False


def round_measure(value) -> float | None:





    if value is None:
        return None
    try:
        return round(float(value), measure_decimals())
    except (TypeError, ValueError):
        return None







_MEASURE_FIELD_INTEGER_DIGITS = 17


def measure_field(name: str, decimals: int | None = None):













    if decimals is None:
        decimals = measure_decimals()
    decimals = max(0, min(int(decimals), 9))
    return QgsField(
        name, field_type_double(), "double",
        _MEASURE_FIELD_INTEGER_DIGITS + decimals + 1, decimals,
    )


def apply_export_field_aliases(layer) -> None:







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








    symbol = QgsFillSymbol.createSimple({
        "color": _REVIEW_FILL,
        "style": "solid",
        "outline_color": _REVIEW_OUTLINE,
        "outline_width": "0.4",
        "outline_style": "solid",
    })
    return QgsSingleSymbolRenderer(symbol)


def make_area_measurer(crs, transform_context=None,
                       ellipsoid=None) -> QgsDistanceArea:









    measurer = QgsDistanceArea()
    if crs is not None and crs.isValid():
        measurer.setSourceCrs(crs, _project_transform_context(transform_context))
    if ellipsoid is None:
        ellipsoid = QgsProject.instance().ellipsoid()




    if not ellipsoid or str(ellipsoid).upper() == "NONE":
        ellipsoid = "EPSG:7030"
    measurer.setEllipsoid(ellipsoid)
    return measurer


def ground_unit_metres(crs, ref_x: float, ref_y: float,
                       transform_context=None,
                       ellipsoid=None) -> tuple[float, float]:






    try:
        if crs is None or not crs.isValid():
            return 1.0, 1.0
        step = 0.001 if crs.isGeographic() else 1.0
        measurer = make_area_measurer(crs, transform_context, ellipsoid)
        along_x = float(measurer.measureLine(
            QgsPointXY(ref_x, ref_y), QgsPointXY(ref_x + step, ref_y)))
        along_y = float(measurer.measureLine(
            QgsPointXY(ref_x, ref_y), QgsPointXY(ref_x, ref_y + step)))
        if along_x > 0.0 and along_y > 0.0:
            return along_x / step, along_y / step
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return 1.0, 1.0


def ground_unit_aspect(crs, ref_x: float, ref_y: float) -> float:
















    along_x, along_y = ground_unit_metres(crs, ref_x, ref_y)
    if along_x > 0.0 and along_y > 0.0:
        aspect = along_y / along_x
        if abs(aspect - 1.0) >= ground_aspect_dead_band():
            return aspect
    return 1.0


def write_vector_layer(layer, file_path: str, options, transform_context=None):






    if transform_context is None:
        transform_context = QgsProject.instance().transformContext()
    return QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, file_path, transform_context, options)




_planar_area_fallback = {"logged": False}


def note_planar_area_fallback() -> None:





    if _planar_area_fallback["logged"]:
        return
    _planar_area_fallback["logged"] = True
    try:
        QgsMessageLog.logMessage(
            "Area: the ellipsoidal measure refused this CRS, areas are "
            "written in layer units for the rest of the session",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def geodesic_area_m2(geom: QgsGeometry, crs) -> float:











    try:
        return float(make_area_measurer(crs).measureArea(geom))
    except Exception:
        note_planar_area_fallback()
        return float(geom.area())


def _polygon_parts(geom: QgsGeometry) -> list[QgsGeometry]:






    members = geom.asGeometryCollection() if geom.isMultipart() else [geom]
    return [
        part
        for part in members
        if not part.isEmpty() and part.type() == PolygonGeometry
    ]


def repair_polygon(geom: QgsGeometry) -> QgsGeometry | None:









    if geom is None or geom.isEmpty():
        return None
    flat = QgsWkbTypes.flatType(geom.wkbType())
    pure_polygon = flat in (
        WkbPolygon,
        WkbMultiPolygon,
    )



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

















    if geom is None or geom.isEmpty():
        return None
    flat = QgsWkbTypes.flatType(geom.wkbType())
    if flat == WkbMultiPolygon:
        return geom
    if flat == WkbPolygon:
        promoted = QgsGeometry(geom)
        if promoted.convertToMultiType():
            return promoted



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






    try:
        QgsMessageLog.logMessage(
            f"Export conventions: {step} failed: {err}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _layer_still_writable(layer) -> bool:





    try:
        from qgis.PyQt import sip

        if sip.isdeleted(layer) is True:
            return False
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    try:
        return bool(layer.isValid())
    except (RuntimeError, AttributeError):
        return False


def _write_conventions_into_the_file(layer, metadata: bool, style: bool) -> None:

    if not _layer_still_writable(layer):
        return
    if metadata:
        try:
            save_metadata = getattr(layer, "saveDefaultMetadata", None)
            if save_metadata is not None:
                save_metadata()
        except Exception as err:  # noqa: BLE001
            _log_convention_failure("saving the metadata into the file", err)
    if style:
        try:
            layer.saveStyleToDatabase(layer.name(), "AI Segmentation", True, "")
        except Exception as err:  # noqa: BLE001
            _log_convention_failure("saving the style into the file", err)


def _reread_file_after_write(layer) -> None:














    if not _layer_still_writable(layer):
        return
    try:
        if layer.providerType() == "ogr" and not layer.isEditable():
            layer.dataProvider().reloadData()
    except (RuntimeError, AttributeError):  # nosec B110
        pass
    try:
        from .run_autosave import repaint_layers_over

        repaint_layers_over(layer.source())
    except Exception:  # noqa: BLE001  # nosec B110
        pass




_pending_file_writes: dict[str, dict] = {}


def persist_layer_to_file_later(layer, *, metadata: bool = False,
                                style: bool = False) -> None:












    if layer is None or not (metadata or style):
        return






    try:
        key = str(layer.id())
    except (RuntimeError, AttributeError):
        key = ""
    if key:
        pending = _pending_file_writes.get(key)
        if pending is not None:
            pending["metadata"] = pending["metadata"] or metadata
            pending["style"] = pending["style"] or style
            return
        _pending_file_writes[key] = {"metadata": metadata, "style": style}

    def _write() -> None:
        flags = _pending_file_writes.pop(key, None) if key else None
        want_metadata = flags["metadata"] if flags else metadata
        want_style = flags["style"] if flags else style
        _write_conventions_into_the_file(layer, want_metadata, want_style)
        _reread_file_after_write(layer)

    try:
        from qgis.core import QgsApplication
        from qgis.PyQt.QtCore import QThread

        app = QgsApplication.instance()
        on_gui_thread = app is None or QThread.currentThread() == app.thread()
    except (RuntimeError, AttributeError):
        on_gui_thread = True
    if not on_gui_thread:
        _write()
        return
    try:
        from .qt_compat import safe_single_shot

        safe_single_shot(0, layer, _write)
    except Exception:  # noqa: BLE001
        _write()


def apply_output_conventions(
    layer: QgsVectorLayer,
    source_raster_name: str,
    *,
    prompt: str = "",
    detail: int | None = None,
    confidence: float | None = None,
    created_iso: str = "",
    plugin_version: str = "",
    basemap_label: str = "",
    source_crs_authid: str = "",
    overlapping_pairs: int | None = None,
    store_style: bool = True,
) -> None:















    from .output_metadata import apply_layer_metadata, output_timestamp_iso

    created = created_iso or output_timestamp_iso()


    apply_export_field_aliases(layer)
    apply_output_display_expression(layer)
    try:
        apply_layer_metadata(
            layer,
            source_raster_name=source_raster_name,
            prompt=prompt,
            detail=detail,
            confidence=confidence,
            created=created,
            plugin_version=plugin_version,
            basemap_label=basemap_label,
            source_crs_authid=source_crs_authid,
            overlapping_pairs=overlapping_pairs,
        )
    except Exception as err:  # noqa: BLE001


        _log_convention_failure("provenance metadata", err)



    persist_layer_to_file_later(layer, metadata=True, style=store_style)




_DISPLAY_FIELD_ORDER = ("class", "det_id", "label", "area_m2")


def apply_output_display_expression(layer) -> None:







    try:
        names = {f.name().lower() for f in layer.fields()}
    except (RuntimeError, AttributeError):
        return
    present = [c for c in _DISPLAY_FIELD_ORDER if c in names]
    if not present:
        return



    quoted = ", ".join(f'"{name}"' for name in present)
    expression = quoted if len(present) == 1 else f"coalesce({quoted})"
    try:
        layer.setDisplayExpression(expression)
    except (RuntimeError, AttributeError, TypeError):  # nosec B110
        pass


def attribute_values_for_fields(
    fields, geom: QgsGeometry, crs, raster_name: str, timestamp: str,
    *, det_id=None, object_class: str = "", confidence: float | None = None,
    measurer: QgsDistanceArea | None = None,
) -> list:











    if measurer is None:
        measurer = make_area_measurer(crs)
    values = []
    for field in fields:
        name = field.name().lower()
        if name == "det_id":
            values.append(None if det_id is None else str(det_id))
        elif name == "class":
            values.append(object_class or None)
        elif name in ("confidence", "score"):
            values.append(None if confidence is None
                          else round(float(confidence), 3))
        elif name in ("area_m2", "perimeter_m", "perim_m"):
            values.append(_measured_value(measurer, name, geom, crs))
        elif name == "area":
            values.append(round_measure(geom.area()))
        elif name == "raster_source":



            values.append(raster_name)
        elif name == "created_at":
            values.append(timestamp)
        else:
            values.append(None)
    return values


def _measured_value(measurer: QgsDistanceArea, name: str, geom: QgsGeometry,
                    crs) -> float | None:

    try:
        if name == "area_m2":
            return round_measure(measurer.measureArea(geom))
        return round_measure(measurer.measurePerimeter(geom))
    except Exception:  # noqa: BLE001
        if name == "area_m2":
            note_planar_area_fallback()
        return None
