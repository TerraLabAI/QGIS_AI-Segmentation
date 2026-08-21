"""Portable run tokens for the public API.

Part of `SegmentationMCPAPI` (see `mcp_api.py`), split out so one concern sits
in one file. A recipe packs what a zone run was asked to do into one short
string that carries no path, no key and no layer name, so it travels in a bug
report safely.
"""
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


class SegmentationRecipeMixin:
    """Encode a run's intent into a token, and replay it."""

    def export_recipe(
        self,
        zone_wkt: str,
        object_class: str,
        layer_name: str | None = None,
        detail: int = 1,
        confidence: float | None = None,
        refine: dict | None = None,
    ) -> dict:
        """Serialize a run's intent into a short, portable ``aiseg1:`` token.

        A recipe captures WHAT to segment and WHERE (object prompt, drawn zone,
        detail level, review confidence, refine settings) so the same run can be
        reproduced later or on another machine. It is meant for debugging: an
        agent that just called :meth:`detect_auto` can hand the same arguments
        here and get one string that reconstructs the run exactly via
        :meth:`run_from_recipe`, or that a user can paste into a bug report.

        By construction the token holds no raster path, activation key, layer
        name, or URL: what the schema cannot hold, it cannot leak. The zone is
        stored as WGS84 lon/lat, so this reprojects ``zone_wkt`` (given in the
        raster layer's CRS) to lon/lat before encoding.

        Parameters
        ----------
        zone_wkt : str
            Zone polygon in the raster layer's CRS (same CRS as
            :meth:`detect_auto`). POLYGON or MULTIPOLYGON.
        object_class : str
            The object prompt, e.g. "Building". May be empty (an exemplar-only
            run), but such a recipe cannot be re-run headlessly because
            exemplar draws are deliberately not carried in a recipe.
        layer_name : str | None
            Raster layer whose CRS the ``zone_wkt`` is in. None = active layer.
        detail : int
            The detail-slider level used (>= 1).
        confidence : float | None
            Review confidence [0, 1]. None keeps the Automatic default.
        refine : dict | None
            Refine settings that differ from the review defaults (keys like
            ``simplify``, ``smooth``, ``ortho``, ``expand``, ``fill_holes``,
            and ``fill_holes_max`` for the fill-holes size cutoff in ground m2,
            0 = every hole). Unknown keys are ignored, so an older or newer
            reader of the same token still works.

        Returns
        -------
        dict with key "recipe" (the token string) or "_error". Costs nothing.
        """
        if zone_wkt is not None and not isinstance(zone_wkt, str):
            return {"_error": "zone_wkt must be a WKT string"}
        if not zone_wkt or not zone_wkt.strip():
            return {"_error": "zone_wkt is required to export a recipe"}
        geom = QgsGeometry.fromWkt(zone_wkt)
        if geom is None or geom.isEmpty():
            return {"_error": "Invalid zone WKT"}

        layer = self._resolve_raster_layer(layer_name)
        src_crs = layer.crs() if layer is not None else QgsCoordinateReferenceSystem("EPSG:4326")
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        try:
            ring = self._exterior_ring_in_crs(geom, src_crs, wgs84)
        except Exception as err:  # nosec B110 -- invalid CRS / antimeridian
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

        # Detail sets the tile count, and the tile count is what a rerun of this
        # recipe costs, so hold it inside the levels the slider can reach.
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
        """Reproduce an Automatic run from an ``aiseg1:`` recipe token.

        Decodes the token, reprojects its WGS84 lon/lat zone back to the raster
        layer's CRS, and calls :meth:`detect_auto` with the decoded prompt,
        zone, detail and confidence. This gives a deterministic reproduction of
        a user-reported run for debugging.

        The refine settings ride in the returned ``recipe_applied`` block for
        reference: they are post-run client-side shaping, and this path passes
        them straight through to the run's own refine argument.

        Parameters
        ----------
        token : str
            An ``aiseg1:`` recipe string from :meth:`export_recipe`.
        layer_name : str | None
            Raster layer to run against; its CRS is used to place the zone.
            None = active layer.

        Returns
        -------
        dict : the :meth:`detect_auto` result, plus "recipe_applied" (the
            decoded intent), or "_error".

        Cost
        ----
        This starts a real zone run, so it costs exactly what the original run
        cost and takes as long.
        """
        try:
            recipe = run_recipe.decode(token)
        except run_recipe.RecipeError as err:
            return {"_error": f"Invalid recipe: {err}"}

        layer = self._resolve_raster_layer(layer_name)
        dst_crs = layer.crs() if layer is not None else QgsCoordinateReferenceSystem("EPSG:4326")
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        try:
            pts = [QgsPointXY(lon, lat) for lon, lat in recipe.zone_lonlat]
            if dst_crs != wgs84:
                xform = QgsCoordinateTransform(wgs84, dst_crs, QgsProject.instance())
                pts = [xform.transform(pt) for pt in pts]
            zone_wkt = QgsGeometry.fromPolygonXY([pts]).asWkt()
        except Exception as err:  # nosec B110 -- invalid CRS / antimeridian
            return {"_error": f"Could not reproject recipe zone: {err}"}

        normalized = recipe.normalized_refine()
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
        """A recipe's confidence as a value detect_auto accepts, or None.

        A token may hold 0 or 1, which the run's own range refuses, so anything
        outside it reads as "no choice made" rather than failing the replay.
        """
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(conf) or not 0.05 <= conf <= 0.95:
            return None
        return conf

    def _exterior_ring_in_crs(self, geom, src_crs, dst_crs) -> list[tuple[float, float]]:
        """Exterior ring of a (multi)polygon as (x, y) pairs in ``dst_crs``."""
        if geom.isMultipart():
            polys = geom.asMultiPolygon()
            ring = polys[0][0] if polys and polys[0] else []
        else:
            rings = geom.asPolygon()
            ring = rings[0] if rings else []
        xform = None
        if src_crs != dst_crs:
            xform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
        out: list[tuple[float, float]] = []
        for pt in ring:
            p = xform.transform(pt) if xform is not None else pt
            out.append((p.x(), p.y()))
        return out
