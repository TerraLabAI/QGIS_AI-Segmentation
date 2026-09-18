







from __future__ import annotations

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)

from ...core.qt_compat import geometry_op_succeeded


class AutoZoneCrsMixin:





    def _store_auto_zone(self, zone: QgsRectangle | None, crs=None) -> None:
























        self._auto_clip_polygon = None
        self._auto_clip_engine = None
        self._auto_run_ctx = None


        self._auto_free_zone_fit = None
        if zone is None:
            self._auto_zone = None
            self._forget_zone_crs()
            return
        self._auto_zone = zone
        self._record_zone_crs(zone, crs)

    def _record_zone_crs(self, zone: QgsRectangle, crs=None) -> None:







        self._forget_zone_crs()
        try:
            if crs is None:
                crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        except (RuntimeError, AttributeError):
            return
        if crs is None or not crs.isValid():
            return
        self._auto_zone_crs = QgsCoordinateReferenceSystem(crs)
        self._auto_zone_crs_rect = self._zone_rect_key(zone)

    def _forget_zone_crs(self) -> None:

        self._auto_zone_crs = None
        self._auto_zone_crs_rect = None

    @staticmethod
    def _zone_rect_key(rect: QgsRectangle) -> tuple[float, float, float, float]:

        return (rect.xMinimum(), rect.yMinimum(),
                rect.xMaximum(), rect.yMaximum())

    def _zone_source_crs(self, zone: QgsRectangle | None):














        try:
            live = self.iface.mapCanvas().mapSettings().destinationCrs()
        except (RuntimeError, AttributeError):
            return None
        recorded = getattr(self, "_auto_zone_crs", None)
        rect = getattr(self, "_auto_zone_crs_rect", None)
        stored = self._auto_zone
        if (recorded is None or not recorded.isValid() or rect is None
                or stored is None or zone is None):
            return live
        try:
            return recorded if self._zone_rect_key(stored) == rect else live
        except (RuntimeError, AttributeError):
            return live

    def _zone_in_layer_crs(
        self, zone: QgsRectangle, layer: QgsRasterLayer
    ) -> QgsRectangle:








        try:
            zone_crs = self._zone_source_crs(zone)
            layer_crs = layer.crs()
        except (RuntimeError, AttributeError):
            return zone

        if zone_crs is None or zone_crs == layer_crs:
            return zone
        if not zone_crs.isValid() or not layer_crs.isValid():
            return zone

        try:
            xform = QgsCoordinateTransform(zone_crs, layer_crs, QgsProject.instance())
            result = xform.transformBoundingBox(zone)
        except Exception:  # nosec B110
            return zone

        if result.width() <= 0 or result.height() <= 0:
            return zone

        return result

    def _run_crs_for_layer(self, layer: QgsRasterLayer, zone_in_layer: QgsRectangle):










        from ...core.layer_conventions import pick_run_crs

        try:
            layer_crs = layer.crs()
            key = self._run_crs_memo_key(layer_crs, zone_in_layer)
        except (RuntimeError, AttributeError):
            return None
        cached = getattr(self, "_auto_run_crs_memo", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            run_crs = pick_run_crs(layer_crs, zone_in_layer)





            if run_crs is not None and run_crs != layer_crs:
                moved = QgsCoordinateTransform(
                    layer_crs, run_crs, QgsProject.instance()
                ).transformBoundingBox(zone_in_layer)
                if moved.width() <= 0 or moved.height() <= 0:
                    run_crs = layer_crs
        except (RuntimeError, AttributeError, TypeError, ValueError):
            run_crs = layer_crs
        except Exception:  # noqa: BLE001  # nosec B110
            run_crs = layer_crs
        self._auto_run_crs_memo = (key, run_crs)
        return run_crs

    @staticmethod
    def _run_crs_memo_key(layer_crs, zone_in_layer):





        try:
            project_authid = QgsProject.instance().crs().authid()
        except (RuntimeError, AttributeError):
            project_authid = ""
        return (layer_crs.authid(), project_authid,
                zone_in_layer.xMinimum(), zone_in_layer.yMinimum(),
                zone_in_layer.xMaximum(), zone_in_layer.yMaximum())

    def _run_crs_now(self, layer: QgsRasterLayer):







        try:
            layer_crs = layer.crs()
            memo = getattr(self, "_auto_run_crs_memo", None)
            if memo is None:
                return layer_crs
            project_authid = QgsProject.instance().crs().authid()
            if memo[0][0] == layer_crs.authid() and memo[0][1] == project_authid:
                return memo[1]
            return layer_crs
        except (RuntimeError, AttributeError, IndexError, TypeError):
            return None

    def _reproject_zone_to_run_crs(
        self, zone: QgsRectangle, layer: QgsRasterLayer
    ) -> QgsRectangle:

















        anchor = self._auto_zone if self._auto_zone is not None else zone
        anchor_in_layer = self._zone_in_layer_crs(anchor, layer)
        zone_in_layer = (anchor_in_layer if anchor is zone
                         else self._zone_in_layer_crs(zone, layer))
        run_crs = self._run_crs_for_layer(layer, anchor_in_layer)
        try:
            if run_crs is None or run_crs == layer.crs():
                return zone_in_layer
            zone_crs = self._zone_source_crs(zone)
            drawn_in = zone_crs is not None and zone_crs.isValid()
            source_crs = zone_crs if drawn_in else layer.crs()
            xform = QgsCoordinateTransform(source_crs, run_crs, QgsProject.instance())
            source_zone = zone if drawn_in else zone_in_layer
            result = xform.transformBoundingBox(source_zone)
            if result.width() <= 0 or result.height() <= 0:
                raise ValueError("degenerate rectangle in the run CRS")
        except Exception:  # noqa: BLE001
            self._forget_run_crs(layer, anchor_in_layer)
            return zone_in_layer
        return result

    def _forget_run_crs(self, layer: QgsRasterLayer, zone_in_layer: QgsRectangle) -> None:




        try:
            layer_crs = layer.crs()
            self._auto_run_crs_memo = (
                self._run_crs_memo_key(layer_crs, zone_in_layer), layer_crs)
        except (RuntimeError, AttributeError):
            self._auto_run_crs_memo = None

    @staticmethod
    def _prepare_clip_engine(clip_geom):







        if clip_geom is None or clip_geom.isEmpty():
            return None
        try:
            engine = QgsGeometry.createGeometryEngine(clip_geom.constGet())
            engine.prepareGeometry()
            return engine
        except Exception:  # noqa: BLE001
            return None

    def _polygon_in_run_crs(self, layer):






        if self._auto_zone_polygon is None:
            return None
        geom = QgsGeometry(self._auto_zone_polygon)
        try:




            zone_rect = self._auto_zone
            if zone_rect is None:
                zone_rect = geom.boundingBox()
            zone_crs = self._zone_source_crs(zone_rect)
            target_crs = self._run_crs_for_layer(
                layer, self._zone_in_layer_crs(zone_rect, layer))
            if target_crs is None:
                target_crs = layer.crs()
        except (RuntimeError, AttributeError):
            return None
        if zone_crs is None or zone_crs == target_crs:
            return geom
        if not zone_crs.isValid() or not target_crs.isValid():
            return geom
        try:
            xform = QgsCoordinateTransform(zone_crs, target_crs, QgsProject.instance())



            if not geometry_op_succeeded(geom.transform(xform)):
                return None
        except Exception:  # nosec B110
            return None
        return geom

    def _tiles_in_polygon(self, tiles, bbox, pixel_w, pixel_h, layer,
                          crs_authid=None):














        poly = self._polygon_in_run_crs(layer)
        if poly is not None and poly.isEmpty():
            poly = None


        data_bb = None
        if layer is not None and crs_authid:
            data_bb = self._layer_extent_in_run_crs(layer, crs_authid)
            if data_bb is not None and data_bb.isEmpty():
                data_bb = None
        if (poly is None and data_bb is None) or not tiles:
            return tiles
        if pixel_w <= 0 or pixel_h <= 0:
            return tiles
        minx, _miny, maxx, maxy = bbox
        span_x = maxx - minx
        span_y = maxy - bbox[1]





        engine = self._prepare_clip_engine(poly) if poly is not None else None
        pbb = poly.boundingBox() if poly is not None else None
        kept = []
        for tile in tiles:
            tx, ty, tw, th = tile
            gx0 = minx + (tx / pixel_w) * span_x
            gx1 = minx + ((tx + tw) / pixel_w) * span_x
            gy1 = maxy - (ty / pixel_h) * span_y
            gy0 = maxy - ((ty + th) / pixel_h) * span_y
            if data_bb is not None and (
                    gx1 < data_bb.xMinimum() or gx0 > data_bb.xMaximum()
                    or gy1 < data_bb.yMinimum() or gy0 > data_bb.yMaximum()):
                continue
            if pbb is None:
                kept.append(tile)
                continue
            if gx1 < pbb.xMinimum() or gx0 > pbb.xMaximum() or gy1 < pbb.yMinimum() or gy0 > pbb.yMaximum():
                continue
            cell = QgsGeometry.fromRect(QgsRectangle(gx0, gy0, gx1, gy1))
            if engine is not None:
                if engine.intersects(cell.constGet()):
                    kept.append(tile)
            elif cell.intersects(poly):
                kept.append(tile)
        return kept
