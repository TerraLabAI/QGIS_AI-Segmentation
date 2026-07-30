






















from __future__ import annotations

from ...core.i18n import tr






_PROBE_SIDE_PX: int = 256


class AutoImageryGuardMixin:


    def _online_imagery_verdict(self, layer, grid) -> tuple[float, str | None]:















        if not getattr(self, "_auto_source_is_online", False):
            return 0.0, None
        from ...core.server_dials import feature_enabled




        if not feature_enabled("imagery_probe"):
            return 0.0, None
        mupp = self._grid_mupp(grid)
        centres = self._probe_centres(layer, grid)
        if not centres or mupp <= 0:
            return 0.0, None
        floor = 0.0
        try:
            for centre in centres:
                extent = self._imagery_probe_extent(grid, centre)
                if extent is None:
                    continue
                level, refusal = self._walk_source_depth(
                    layer, extent, mupp, grid)


                if refusal is not None:
                    return 0.0, refusal
                floor = max(floor, level)
        except Exception:  # noqa: BLE001
            return 0.0, None
        return floor, None

    def _note_imagery_backoff(self) -> None:









        if getattr(self, "_auto_headless_run", False):
            return
        try:
            from ...core.server_dials import dial_copy

            self.iface.messageBar().pushInfo("AI Segmentation", dial_copy(
                "auto.imagery_backoff",
                tr("This map source has no sharper picture of this area. The "
                   "run uses the sharpest one it has, and costs fewer tiles "
                   "than the estimate.")))
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _walk_source_depth(self, layer, extent, mupp, grid):







        from ...core.cloud_detection import probe_depth_chain, render_verdict
        from ...core.render_depth import (
            LEVEL_SAMPLE_PX,
            agreement_min,
            backoff_steps,
            level_agreement,
        )

        steps = backoff_steps()
        crs = self._probe_render_crs(grid)
        images = probe_depth_chain(
            layer, extent, render_crs=crs, count=steps + 2,
            side_px=None, min_side_px=LEVEL_SAMPLE_PX)
        if len(images) < 2:
            return 0.0, None
        floor_min = agreement_min()
        for step in range(min(steps + 1, len(images) - 1)):
            fine, coarse = images[step], images[step + 1]
            if render_verdict(fine) != "unavailable":
                return self._level_floor(mupp, step), None



            if level_agreement(fine, coarse) >= floor_min:
                return self._level_floor(mupp, step), None
        return 0.0, tr(
            "This layer has no image over your zone at this precision. "
            "The map source answered with an empty tile, so there is nothing "
            "to detect on. Lower Precision, zoom the layer out until the "
            "imagery shows, or pick a layer that covers this area."
        )

    @staticmethod
    def _level_floor(mupp: float, step: int) -> float:





        return 0.0 if step <= 0 else float(mupp) * (2 ** step)

    @staticmethod
    def _grid_mupp(grid) -> float:

        try:
            minx, _miny, maxx, _maxy = (grid or {})["bbox"]
            pixel_w = int((grid or {})["pixel_w"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        if pixel_w <= 0:
            return 0.0
        width = float(maxx) - float(minx)
        return width / pixel_w if width > 0 else 0.0

    @staticmethod
    def _probe_render_crs(grid):


        from qgis.core import QgsCoordinateReferenceSystem

        authid = (grid or {}).get("crs") or ""
        if not authid:
            return None
        crs = QgsCoordinateReferenceSystem(authid)
        return crs if crs.isValid() else None

    def _probe_centres(self, layer, grid) -> list:










        from qgis.core import QgsGeometry, QgsRectangle

        try:
            minx, miny, maxx, maxy = (grid or {})["bbox"]
        except (KeyError, TypeError, ValueError):
            return []
        centre = ((float(minx) + float(maxx)) / 2.0,
                  (float(miny) + float(maxy)) / 2.0)
        try:
            poly = self._polygon_in_run_crs(layer)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            poly = None
        if poly is None or poly.isEmpty():
            return [centre]
        points: list = []
        bb = poly.boundingBox()
        step = bb.width() / 3.0
        for i in range(3):
            band = QgsRectangle(bb.xMinimum() + i * step, bb.yMinimum(),
                                bb.xMinimum() + (i + 1) * step, bb.yMaximum())
            try:
                part = poly.intersection(QgsGeometry.fromRect(band))
                if part is None or part.isEmpty():
                    continue
                on_surface = part.pointOnSurface()
                if on_surface is None or on_surface.isEmpty():
                    continue
                point = on_surface.asPoint()
                points.append((point.x(), point.y()))
            except (RuntimeError, ValueError, TypeError):
                continue
        return points or [centre]

    @staticmethod
    def _imagery_probe_extent(grid, centre=None):










        from qgis.core import QgsRectangle

        try:
            minx, miny, maxx, maxy = (grid or {})["bbox"]
            pixel_w = int((grid or {})["pixel_w"])
            pixel_h = int((grid or {})["pixel_h"])
        except (KeyError, TypeError, ValueError):
            return None
        width = float(maxx) - float(minx)
        height = float(maxy) - float(miny)
        if width <= 0 or height <= 0 or pixel_w <= 0 or pixel_h <= 0:
            return None





        from ...core.shape_policy_dials import imagery_probe_px
        side_px = imagery_probe_px(_PROBE_SIDE_PX)
        half_w = min(width, side_px * width / pixel_w) / 2.0
        half_h = min(height, side_px * height / pixel_h) / 2.0
        if centre is not None:
            cx, cy = float(centre[0]), float(centre[1])
        else:
            cx = (float(minx) + float(maxx)) / 2.0
            cy = (float(miny) + float(maxy)) / 2.0
        return QgsRectangle(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
