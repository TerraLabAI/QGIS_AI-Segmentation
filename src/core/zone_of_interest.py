# SPDX-FileCopyrightText: 2026 TerraLab <yvann.barbot@terra-lab.ai>
# SPDX-License-Identifier: GPL-2.0-or-later

































from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsDistanceArea,
    QgsFeature,
    QgsFillSymbol,
    QgsGeometry,
    QgsProject,
    QgsRectangle,
    QgsVectorLayer,
    QgsWkbTypes,
)




ZONE_PROPERTY = "terralab/zone_of_interest"
ZONE_SCOPE = "TerraLab"
ZONE_LAYER_ENTRY = "/ZoneOfInterestLayerId"
ZONE_LABEL_ENTRY = "/ZoneOfInterestLabel"



ZONE_SHAPES_PROPERTY = ZONE_PROPERTY + "/shapes_wkt"




ZONE_LAYER_NAME = "Area of interest"







ZONE_LAYER_NAMES = {
    "de": "Interessengebiet",
    "es": "Área de interés",
    "fr": "Zone d’intérêt",
    "id": "Area of interest",
    "it": "Area di interesse",
    "ja": "関心域",
    "nl": "Interessegebied",
    "pl": "Obszar zainteresowania",
    "pt": "Área de interesse",
    "zh_CN": "关注区域",
    "zh_TW": "關注區域",
}




LEGACY_ZONE_LAYER_NAMES = ("Zone of interest",)



ZONE_COLOR = "#f5a623"
ZONE_OUTLINE_WIDTH_MM = 0.8





MAX_UNION_FEATURES = 5000


def zone_layer_name() -> str:

    try:
        locale = str(QgsApplication.locale() or "")
    except Exception:  # noqa: BLE001
        locale = ""
    locale = locale.replace("-", "_")
    lang = locale.split("_")[0].lower()
    if lang == "zh":
        hant = any(tag in locale.upper() for tag in ("TW", "HK", "MO", "HANT"))
        return ZONE_LAYER_NAMES["zh_TW" if hant else "zh_CN"]
    return ZONE_LAYER_NAMES.get(lang, ZONE_LAYER_NAME)


def is_default_zone_name(name: str) -> bool:


    text = str(name or "").strip().casefold()
    defaults = (ZONE_LAYER_NAME, *ZONE_LAYER_NAMES.values(), *LEGACY_ZONE_LAYER_NAMES)
    return bool(text) and any(text == d.casefold() for d in defaults)


def polygon_geometry_type() -> Any:

    scoped = getattr(QgsWkbTypes, "GeometryType", None)
    member = getattr(scoped, "PolygonGeometry", None)
    if member is None:
        member = getattr(QgsWkbTypes, "PolygonGeometry", None)
    return member


def _transform_succeeded(outcome: Any) -> bool:







    if outcome is None:
        return True
    if isinstance(outcome, bool):
        return outcome
    try:
        return int(outcome) == 0
    except (TypeError, ValueError):
        return True


@dataclass
class Zone:



    geometry: QgsGeometry
    crs: QgsCoordinateReferenceSystem
    label: str = ""
    layer_id: str = ""


    approximate: bool = False

    def area_km2(self) -> float:
        return area_km2(self.geometry, self.crs)

    def wkt_in(self, crs: QgsCoordinateReferenceSystem, project: Any = None) -> str:

        moved = to_crs(self.geometry, self.crs, crs, project)
        return "" if moved is None else moved.asWkt()


@dataclass
class ZoneSource:



    kind: str
    label: str
    layer_id: str = ""
    feature_count: int = 0
    area_km2: float = 0.0





def _project(project: Any = None) -> QgsProject:
    return project if project is not None else QgsProject.instance()


def zone_layer(project: Any = None) -> QgsVectorLayer | None:






    proj = _project(project)
    try:
        held, _ok = proj.readEntry(ZONE_SCOPE, ZONE_LAYER_ENTRY, "")
    except Exception:  # noqa: BLE001
        held = ""
    if held:
        layer = proj.mapLayer(held)
        if isinstance(layer, QgsVectorLayer) and layer.isValid():
            return layer
    for layer in proj.mapLayers().values():
        if not isinstance(layer, QgsVectorLayer) or not layer.isValid():
            continue
        if str(layer.customProperty(ZONE_PROPERTY, "") or ""):
            return layer
    return None


def read_zone(project: Any = None) -> Zone | None:





    layer = zone_layer(project)
    if layer is None:
        return None
    parts = _zone_parts(layer)
    if not parts and _refill_zone_layer(layer):

        parts = _zone_parts(layer)
    if not parts:
        return None
    geom = parts[0] if len(parts) == 1 else union(parts)
    if geom is None or geom.isEmpty():
        return None
    proj = _project(project)
    try:
        label, _ok = proj.readEntry(ZONE_SCOPE, ZONE_LABEL_ENTRY, "")
    except Exception:  # noqa: BLE001
        label = ""
    return Zone(geometry=geom, crs=layer.crs(), label=str(label or ""),
                layer_id=layer.id())


def _zone_parts(layer: QgsVectorLayer) -> list[QgsGeometry]:
    parts = []
    for feature in layer.getFeatures():
        geom = feature.geometry()
        if geom is not None and not geom.isEmpty():
            parts.append(QgsGeometry(geom))
    return parts





def store_zone_shapes(layer: QgsVectorLayer) -> None:



    if layer is None:
        return
    shapes = []
    try:
        for feature in layer.dataProvider().getFeatures():
            geom = feature.geometry()
            if geom is not None and not geom.isEmpty():
                shapes.append(geom.asWkt())
        layer.setCustomProperty(ZONE_SHAPES_PROPERTY, json.dumps(shapes))
    except Exception:  # noqa: BLE001
        return


def stored_zone_shapes(layer: QgsVectorLayer) -> list[QgsGeometry] | None:


    try:
        raw = layer.customProperty(ZONE_SHAPES_PROPERTY, "")
        items = json.loads(str(raw)) if raw else None
    except (TypeError, ValueError):
        return None
    if not isinstance(items, list):
        return None
    shapes = []
    for wkt in items:
        geom = QgsGeometry.fromWkt(str(wkt))
        if geom is not None and not geom.isEmpty():
            shapes.append(geom)
    return shapes


def store_project_zone_shapes(project: Any = None) -> None:







    layer = zone_layer(project)
    if layer is not None:
        store_zone_shapes(layer)


def _refill_zone_layer(layer: QgsVectorLayer) -> bool:



    try:
        provider = layer.dataProvider()
        if provider is None or provider.name() != "memory" or layer.isEditable():
            return False
        if provider.featureCount() > 0:
            return False
    except Exception:  # noqa: BLE001
        return False
    shapes = stored_zone_shapes(layer)
    if not shapes:
        return False
    features = []
    for geom in shapes:
        feature = QgsFeature(layer.fields())
        feature.setGeometry(geom)
        features.append(feature)
    try:
        provider.addFeatures(features)
        layer.updateExtents()
        layer.triggerRepaint()
    except Exception:  # noqa: BLE001
        return False
    return provider.featureCount() > 0


def restore_zone_layer(project: Any = None) -> bool:






    layer = zone_layer(project)
    if layer is None:
        return False
    return _refill_zone_layer(layer)





def _same_shape(held: Zone, geometry: QgsGeometry,
                crs: QgsCoordinateReferenceSystem) -> bool:

    if held.crs != crs:
        moved = to_crs(geometry, crs, held.crs)
        if moved is None:
            return False
        geometry = moved
    try:
        return bool(held.geometry.equals(geometry))
    except Exception:  # noqa: BLE001
        return False


def write_zone(geometry: QgsGeometry, crs: QgsCoordinateReferenceSystem,
               label: str | None = None, project: Any = None,
               approximate: bool = False) -> QgsVectorLayer | None:













    if geometry is None or geometry.isEmpty():
        return None
    if crs is None or not crs.isValid():
        return None
    proj = _project(project)
    if label is None:
        held = read_zone(proj)
        label = held.label if (held is not None and _same_shape(held, geometry, crs)) else ""
    layer = zone_layer(proj)
    if layer is not None and layer.crs() != crs:



        try:
            proj.removeMapLayer(layer.id())
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        layer = None
    fresh = layer is None
    if fresh:
        layer = QgsVectorLayer(f"Polygon?crs={crs.authid() or crs.toWkt()}",
                               zone_layer_name(), "memory")
        if not layer.isValid():
            return None
        style_zone_layer(layer)
    elif is_default_zone_name(layer.name()) and layer.name() != zone_layer_name():


        layer.setName(zone_layer_name())
    provider = layer.dataProvider()
    existing = [f.id() for f in layer.getFeatures()]
    if existing:
        provider.deleteFeatures(existing)
    feature = QgsFeature(layer.fields())
    feature.setGeometry(QgsGeometry(geometry))
    provider.addFeatures([feature])
    layer.updateExtents()
    layer.setCustomProperty(ZONE_PROPERTY, "1")
    store_zone_shapes(layer)
    if approximate:
        layer.setCustomProperty(ZONE_PROPERTY + "/approximate", "1")
    else:
        layer.removeCustomProperty(ZONE_PROPERTY + "/approximate")
    if fresh:
        proj.addMapLayer(layer)
    try:
        proj.writeEntry(ZONE_SCOPE, ZONE_LAYER_ENTRY, layer.id())
        proj.writeEntry(ZONE_SCOPE, ZONE_LABEL_ENTRY, str(label or ""))
    except Exception:  # noqa: BLE001
        pass  # nosec B110
    layer.triggerRepaint()
    return layer


def clear_zone(project: Any = None) -> bool:

    proj = _project(project)
    layer = zone_layer(proj)
    try:
        proj.writeEntry(ZONE_SCOPE, ZONE_LAYER_ENTRY, "")
        proj.writeEntry(ZONE_SCOPE, ZONE_LABEL_ENTRY, "")
    except Exception:  # noqa: BLE001
        pass  # nosec B110
    if layer is None:
        return False
    try:
        proj.removeMapLayer(layer.id())
    except Exception:  # noqa: BLE001
        return False
    return True


def style_zone_layer(layer: QgsVectorLayer) -> None:

    try:
        symbol = QgsFillSymbol.createSimple({
            "style": "no",
            "outline_color": ZONE_COLOR,
            "outline_width": str(ZONE_OUTLINE_WIDTH_MM),
            "outline_width_unit": "MM",
            "outline_style": "dash",
        })
        layer.renderer().setSymbol(symbol)
    except Exception:  # noqa: BLE001
        return





def outline_of_layer(layer: Any, selected_only: bool = False
                     ) -> tuple[QgsGeometry, QgsCoordinateReferenceSystem, bool] | None:







    if layer is None:
        return None
    crs = getattr(layer, "crs", None)
    crs = crs() if callable(crs) else None
    if crs is None or not crs.isValid():
        return None
    if not isinstance(layer, QgsVectorLayer):

        extent = layer.extent() if hasattr(layer, "extent") else None
        if extent is None or extent.isEmpty():
            return None
        return QgsGeometry.fromRect(QgsRectangle(extent)), crs, True
    if selected_only:
        features = list(layer.selectedFeatures())
        if not features:
            return None
    else:
        count = layer.featureCount()
        if count is not None and count > MAX_UNION_FEATURES:
            extent = layer.extent()
            if extent is None or extent.isEmpty():
                return None
            return QgsGeometry.fromRect(QgsRectangle(extent)), crs, True
        features = list(layer.getFeatures())
    polygons = []
    for feature in features:
        geom = feature.geometry()
        if geom is None or geom.isEmpty():
            continue
        if geom.type() == polygon_geometry_type():
            polygons.append(QgsGeometry(geom))
    if polygons:
        united = union(polygons)
        if united is not None and not united.isEmpty():
            return united, crs, False

    box = QgsRectangle()
    for feature in features:
        geom = feature.geometry()
        if geom is not None and not geom.isEmpty():
            box.combineExtentWith(geom.boundingBox())
    if box.isEmpty():
        return None
    return QgsGeometry.fromRect(box), crs, True


def zone_sources(project: Any = None, limit: int = 30) -> list[ZoneSource]:







    proj = _project(project)
    out: list[ZoneSource] = []
    held = read_zone(proj)
    zone_id = held.layer_id if held is not None else ""
    if held is not None:
        out.append(ZoneSource(kind="zone", label=held.label or zone_layer_name(),
                              layer_id=held.layer_id, feature_count=1,
                              area_km2=held.area_km2()))
    layers = [layer for layer in proj.mapLayers().values()
              if isinstance(layer, QgsVectorLayer) and layer.isValid()
              and layer.id() != zone_id
              and layer.geometryType() == polygon_geometry_type()]
    for layer in layers:
        selected = layer.selectedFeatureCount()
        if not selected:
            continue
        found = outline_of_layer(layer, selected_only=True)
        if found is None:
            continue
        surface = area_km2(found[0], found[1])
        if surface <= 0:
            continue
        out.append(ZoneSource(
            kind="selection", label=layer.name(), layer_id=layer.id(),
            feature_count=selected, area_km2=surface))
    for layer in layers:
        found = outline_of_layer(layer)
        if found is None:
            continue
        surface = area_km2(found[0], found[1])
        if surface <= 0:
            continue
        out.append(ZoneSource(
            kind="layer", label=layer.name(), layer_id=layer.id(),
            feature_count=layer.featureCount() or 0, area_km2=surface))
    return out[:limit]





def area_km2(geometry: QgsGeometry, crs: QgsCoordinateReferenceSystem) -> float:







    if geometry is None or geometry.isEmpty() or crs is None or not crs.isValid():
        return 0.0
    try:
        measure = QgsDistanceArea()
        measure.setSourceCrs(crs, QgsProject.instance().transformContext())
        measure.setEllipsoid(QgsProject.instance().ellipsoid() or "WGS84")
        return float(measure.measureArea(geometry)) / 1_000_000.0
    except Exception:  # noqa: BLE001
        return 0.0


def to_crs(geometry: QgsGeometry, source: QgsCoordinateReferenceSystem,
           target: QgsCoordinateReferenceSystem, project: Any = None
           ) -> QgsGeometry | None:


    if geometry is None or geometry.isEmpty():
        return None
    if source is None or target is None or not source.isValid() or not target.isValid():
        return None
    if source == target:
        return QgsGeometry(geometry)
    moved = QgsGeometry(geometry)
    try:
        transform = QgsCoordinateTransform(source, target, _project(project))
        if not _transform_succeeded(moved.transform(transform)):
            return None
    except Exception:  # noqa: BLE001
        return None
    return None if moved.isEmpty() else moved


def union(parts: list[QgsGeometry]) -> QgsGeometry | None:

    if not parts:
        return None
    if len(parts) == 1:
        return QgsGeometry(parts[0])
    unary = getattr(QgsGeometry, "unaryUnion", None)
    if callable(unary):
        try:
            found = unary(parts)
            if found is not None and not found.isEmpty():
                return found
        except Exception:  # noqa: BLE001
            pass  # nosec B110
    found = QgsGeometry(parts[0])
    for part in parts[1:]:
        try:
            found = found.combine(part)
        except Exception:  # noqa: BLE001
            return found
    return None if found is None or found.isEmpty() else found
