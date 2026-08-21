






from __future__ import annotations

import math

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
)

from .core import run_recipe
from .core.review_defaults import AUTO_DEFAULT_CONFIDENCE
from .mcp_api_auto import _confidence_bounds_in_force


class SegmentationRecipeMixin:


    def export_recipe(
        self,
        zone_wkt: str,
        object_class: str,
        layer_name: str | None = None,
        detail: int = 1,
        confidence: float | None = None,
        refine: dict | None = None,
    ) -> dict:








































        if zone_wkt is not None and not isinstance(zone_wkt, str):
            return {"_error": "zone_wkt must be a WKT string"}
        if not zone_wkt or not zone_wkt.strip():
            return {"_error": "zone_wkt is required to export a recipe"}
        geom = QgsGeometry.fromWkt(zone_wkt)
        if geom is None or geom.isEmpty():
            return {"_error": "Invalid zone WKT"}

        layer = self._resolve_raster_layer(layer_name)
        if layer_name and layer is None:
            return {"_error": "The requested raster layer was not found."}
        src_crs = layer.crs() if layer is not None else QgsCoordinateReferenceSystem("EPSG:4326")
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        try:
            ring = self._exterior_ring_in_crs(geom, src_crs, wgs84)
        except Exception as err:  # nosec B110
            return {"_error": f"Could not reproject zone to lon/lat: {err}"}
        if len(ring) < 3:
            return {"_error": "zone must be a polygon with at least 3 points"}

        if confidence is None:
            conf = AUTO_DEFAULT_CONFIDENCE
        else:
            try:
                conf = float(confidence)
            except (TypeError, ValueError):
                return {"_error": f"confidence must be a number in [0, 1], got {confidence!r}"}
            if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
                return {"_error": f"confidence must be in [0, 1], got {confidence!r}"}



        from .core.tile_manager import MAX_DETAIL_LEVEL
        try:
            detail_level = int(detail or 1)
        except (TypeError, ValueError):
            return {"_error": f"detail must be a whole number, got {detail!r}"}
        detail_level = max(1, min(MAX_DETAIL_LEVEL, detail_level))

        try:
            token = run_recipe.encode(
                run_recipe.RunRecipe(
                    prompt=(object_class or "").strip(),
                    detail=detail_level,
                    zone_lonlat=ring,
                    confidence=conf,
                    refine=dict(refine or {}),
                )
            )
        except (run_recipe.RecipeError, TypeError, ValueError, AttributeError) as err:
            return {"_error": f"Could not encode recipe: {err}"}
        return {"recipe": token}

    def run_from_recipe(self, token: str, layer_name: str | None = None) -> dict:





























        try:
            recipe = run_recipe.decode(token)
        except run_recipe.RecipeError as err:
            return {"_error": f"Invalid recipe: {err}"}

        layer = self._resolve_raster_layer(layer_name)
        if layer_name and layer is None:
            return {"_error": "The requested raster layer was not found."}
        dst_crs = layer.crs() if layer is not None else QgsCoordinateReferenceSystem("EPSG:4326")
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        try:
            pts = [QgsPointXY(lon, lat) for lon, lat in recipe.zone_lonlat]
            if dst_crs != wgs84:
                xform = QgsCoordinateTransform(wgs84, dst_crs, QgsProject.instance())
                pts = [xform.transform(pt) for pt in pts]
            zone_wkt = QgsGeometry.fromPolygonXY([pts]).asWkt()
        except Exception as err:  # nosec B110
            return {"_error": f"Could not reproject recipe zone: {err}"}

        normalized = dict(recipe.refine)
        result = self.detect_auto(
            zone_wkt=zone_wkt,
            object_class=recipe.prompt,
            layer_name=layer_name,
            detail=recipe.detail,
            confidence=self._recipe_confidence(recipe.confidence),
            refine=normalized,
        )
        if isinstance(result, dict):
            result["recipe_applied"] = {
                "prompt": recipe.prompt,
                "detail": recipe.detail,
                "confidence": recipe.confidence,
                "refine": normalized,
            }
        return result

    def _recipe_confidence(self, value):





        try:
            conf = float(value)
        except (TypeError, ValueError):
            return None
        low, high = _confidence_bounds_in_force()
        if not math.isfinite(conf) or not low <= conf <= high:
            return None
        return conf

    def _exterior_ring_in_crs(self, geom, src_crs, dst_crs) -> list[tuple[float, float]]:

        if geom.isMultipart():
            polys = geom.asMultiPolygon()
            if len(polys) != 1:
                raise ValueError("A recipe zone must contain one polygon.")
            rings = polys[0]
        else:
            rings = geom.asPolygon()
        if len(rings) != 1:
            raise ValueError("A recipe zone must contain one exterior ring and no holes.")
        ring = rings[0]
        xform = None
        if src_crs != dst_crs:
            xform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
        out: list[tuple[float, float]] = []
        for pt in ring:
            p = xform.transform(pt) if xform is not None else pt
            out.append((p.x(), p.y()))
        return out
