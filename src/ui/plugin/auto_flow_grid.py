







from __future__ import annotations

from ...core.qt_compat import DistanceMeters
from .shared import _WEBMERC_MUPP_Z0


def _crs_run_identifier(crs) -> str:








    try:
        if crs is None or not crs.isValid():
            return ""
        return crs.authid() or crs.toWkt()
    except (RuntimeError, AttributeError):
        return ""


class AutoFlowGridMixin:


    def _online_native_mupp(self, layer) -> float:













        if layer is None:
            return 0.0









        try:
            provider = layer.dataProvider()
            if provider is not None:

                native = provider.nativeResolutions()
                finest = self._finest_native_resolution(native)
                if finest > 0:
                    return finest
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass





        try:
            if self._layer_is_web_mercator(layer):
                return self._xyz_zmax_mupp_from_source(layer.source() or "")
        except (RuntimeError, AttributeError, ValueError, TypeError):
            pass
        return 0.0

    @staticmethod
    def _finest_native_resolution(resolutions) -> float:




        best = 0.0
        try:
            items = list(resolutions or [])
        except TypeError:
            return 0.0
        for r in items:
            try:
                v = float(r)
            except (TypeError, ValueError):
                continue
            if v > 0 and (best == 0.0 or v < best):
                best = v
        return best

    @staticmethod
    def _layer_is_web_mercator(layer) -> bool:



        try:
            authid = (layer.crs().authid() or "").upper()
        except (RuntimeError, AttributeError):
            return False
        return authid in (
            "EPSG:3857", "EPSG:900913", "EPSG:102100", "EPSG:102113",
        )

    @staticmethod
    def _xyz_zmax_mupp_from_source(source: str) -> float:







        try:
            import re  # noqa: PLC0415



            m = re.search(r"(?:^|[?&])zmax=(\d+)", source or "")
            if not m:
                return 0.0
            zmax = int(m.group(1))
            if zmax <= 0:
                return 0.0
            return _WEBMERC_MUPP_Z0 / float(2 ** zmax)
        except (ValueError, TypeError):
            return 0.0

    def _served_source_floor(
        self, layer, zone_in_layer, mupp: float, allowance: float
    ) -> float:

















        from ...core.source_resolution import source_floor_mupp_m

        if mupp <= 0 or allowance <= 0:
            return 0.0
        try:
            source = layer.source() or ""
        except (RuntimeError, AttributeError):
            return 0.0
        floor_m = source_floor_mupp_m(source)
        if floor_m <= 0:
            return 0.0
        metres = self._mupp_to_meters(layer, zone_in_layer, mupp)
        if metres <= 0:
            return 0.0


        metres_per_unit = metres / mupp
        if metres_per_unit <= 0:
            return 0.0
        return (floor_m / metres_per_unit) / allowance

    def _layer_units_to_run_units(self, layer, zone_in_run) -> tuple[float, float]:







        from qgis.core import QgsCoordinateTransform, QgsProject

        from ...core.layer_conventions import ground_unit_metres

        run_crs = self._run_crs_now(layer)
        try:
            layer_crs = layer.crs()
            if run_crs is None or run_crs == layer_crs:
                return 1.0, 1.0
            centre_run = zone_in_run.center()




            key = (layer_crs.authid(), run_crs.authid(),
                   round(centre_run.x(), 6), round(centre_run.y(), 6))
            memo = getattr(self, "_auto_unit_factor_memo", None)
            if memo is not None and memo[0] == key:
                return memo[1]
            run_mx, run_my = ground_unit_metres(
                run_crs, centre_run.x(), centre_run.y())
            if run_mx <= 0 or run_my <= 0:
                return 1.0, 1.0
            centre_layer = QgsCoordinateTransform(
                run_crs, layer_crs, QgsProject.instance()).transform(centre_run)
            layer_mx, layer_my = ground_unit_metres(
                layer_crs, centre_layer.x(), centre_layer.y())


            factors = (layer_mx / run_mx, layer_my / run_my)
            self._auto_unit_factor_memo = (key, factors)
            return factors
        except (RuntimeError, AttributeError, TypeError, ValueError,
                ZeroDivisionError):
            return 1.0, 1.0

    def _grid_for_detail(self, layer, zone_in_layer, detail_n: int,
                         mupp_floor: float = 0.0):



























        from ...core.source_resolution import oversample_allowance
        from ...core.tile_manager import OVERLAP_FRACTION, TILE_SIZE

        longer_side = max(zone_in_layer.width(), zone_in_layer.height())
        if longer_side <= 0:
            return None
        try:
            layer_w = layer.width()
            layer_h = layer.height()
            ext = layer.extent()
        except (RuntimeError, AttributeError):
            return None

        use_online = layer_w <= 0 or layer_h <= 0 or self._needs_canvas_render(layer)



        stride = int(TILE_SIZE * (1.0 - OVERLAP_FRACTION))
        target_px = TILE_SIZE + (max(1, detail_n) - 1) * stride
        mupp = longer_side / target_px












        allowance = oversample_allowance()
        to_run_x, to_run_y = self._layer_units_to_run_units(layer, zone_in_layer)
        if not use_online and ext.width() > 0 and ext.height() > 0:
            native_mupp = max(ext.width() / layer_w * to_run_x,
                              ext.height() / layer_h * to_run_y)
            mupp = max(mupp, native_mupp / allowance)
        elif use_online:



            online_mupp = self._online_native_mupp(layer)
            if online_mupp > 0:
                native_mupp = online_mupp * max(to_run_x, to_run_y)
                mupp = max(mupp, native_mupp / allowance)
            mupp = max(mupp, self._served_source_floor(
                layer, zone_in_layer, mupp, allowance))



        if mupp_floor > 0:
            mupp = max(mupp, float(mupp_floor))

        pixel_w = max(1, int(zone_in_layer.width() / mupp))
        pixel_h = max(1, int(zone_in_layer.height() / mupp))











        pixel_w = max(pixel_w, TILE_SIZE)
        pixel_h = max(pixel_h, TILE_SIZE)






        tile_count = self._tile_manager.estimate_credits(pixel_w, pixel_h)
        return pixel_w, pixel_h, mupp, tile_count

    def _max_useful_detail(self, layer, zone_in_layer) -> int:











        from ...core.tile_manager import MAX_DETAIL_LEVEL





        max_tiles = self._auto_zone_tile_cap()




        try:
            key = (
                layer.id(), max_tiles,
                zone_in_layer.xMinimum(), zone_in_layer.yMinimum(),
                zone_in_layer.xMaximum(), zone_in_layer.yMaximum(),
            )
        except (RuntimeError, AttributeError):
            key = None
        cached = getattr(self, "_max_detail_cache", None)
        if key is not None and cached is not None and cached[0] == key:
            return cached[1]

        best = 1
        prev_mupp = None
        for n in range(1, MAX_DETAIL_LEVEL + 1):
            sized = self._grid_for_detail(layer, zone_in_layer, n)
            if sized is None:
                break
            _pw, _ph, mupp, tiles = sized
            if tiles != -1:
                tiles = self._tiles_after_cull_confirmed(
                    layer, zone_in_layer, n, tiles, max_tiles)
            if tiles == -1 or tiles > max_tiles:
                break
            if prev_mupp is not None and mupp >= prev_mupp:
                break
            best = n
            prev_mupp = mupp
        if key is not None:
            self._max_detail_cache = (key, best)
        return best

    def _mupp_to_meters(self, layer, zone_in_layer, mupp: float) -> float:








        try:
            from qgis.core import (
                QgsDistanceArea,
                QgsPointXY,
                QgsProject,
            )

            measure_crs = self._run_crs_now(layer) or layer.crs()
            da = QgsDistanceArea()
            da.setSourceCrs(measure_crs, QgsProject.instance().transformContext())
            da.setEllipsoid("WGS84")
            cx = (zone_in_layer.xMinimum() + zone_in_layer.xMaximum()) / 2.0
            cy = (zone_in_layer.yMinimum() + zone_in_layer.yMaximum()) / 2.0







            dist_x = da.measureLine(QgsPointXY(cx, cy), QgsPointXY(cx + mupp, cy))
            dist_y = da.measureLine(QgsPointXY(cx, cy), QgsPointXY(cx, cy + mupp))
            dist = max(dist_x, dist_y)
            return da.convertLengthMeasurement(dist, DistanceMeters)
        except (RuntimeError, AttributeError, ValueError):
            return 0.0

    def _ground_mupp_for_detail(
        self, layer, zone_in_layer, detail_n: int
    ) -> float:


        sized = self._grid_for_detail(layer, zone_in_layer, detail_n)
        if sized is None:
            return 0.0
        return self._mupp_to_meters(layer, zone_in_layer, sized[2])

    def _compute_auto_grid(self, layer, mupp_floor: float = 0.0) -> dict | None:

































        if self._tile_manager is None:
            self._setup_auto_mode()

        try:
            layer_w = layer.width()
            layer_h = layer.height()
            ext = layer.extent()
        except (RuntimeError, AttributeError):
            return None

        use_online = layer_w <= 0 or layer_h <= 0 or self._needs_canvas_render(layer)

        if self._auto_zone is not None:




            zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
            detail_n = self._get_auto_detail_level()
            sized = self._grid_for_detail(
                layer, zone_in_layer, detail_n, mupp_floor=mupp_floor)
            if sized is None:
                return None
            pixel_w, pixel_h, mupp, _tiles = sized











            minx = zone_in_layer.xMinimum()
            maxy = zone_in_layer.yMaximum()
            maxx = minx + pixel_w * mupp
            miny = maxy - pixel_h * mupp

            run_crs = self._run_crs_now(layer)
            return {
                "pixel_w": pixel_w,
                "pixel_h": pixel_h,
                "zone_x": 0,
                "zone_y": 0,
                "bbox": (minx, miny, maxx, maxy),
                "crs": (_crs_run_identifier(run_crs)
                        or _crs_run_identifier(layer.crs())),
                "online": True,
            }



        if use_online:
            return None
        if ext.width() <= 0:
            return None
        pixel_w, pixel_h = layer_w, layer_h
        zone_x, zone_y = 0, 0
        bbox = (ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum())


        return {
            "pixel_w": pixel_w, "pixel_h": pixel_h,
            "zone_x": zone_x, "zone_y": zone_y,
            "bbox": bbox, "crs": _crs_run_identifier(layer.crs()),
            "online": False,
        }
