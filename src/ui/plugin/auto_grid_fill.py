



















from __future__ import annotations




_PROBE_TILES = 256


class AutoGridFillMixin:


    def _zone_fill_fraction(self, layer, zone_in_layer) -> float:











        key = self._zone_fill_key(layer, zone_in_layer)
        if key is None:
            return 1.0
        cached = getattr(self, "_zone_fill_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        fraction = self._measure_zone_fill(layer, zone_in_layer)
        self._zone_fill_cache = (key, fraction)
        return fraction

    @staticmethod
    def _zone_fill_key(layer, zone_in_layer):


        try:
            return (layer.id(), zone_in_layer.xMinimum(), zone_in_layer.yMinimum(),
                    zone_in_layer.xMaximum(), zone_in_layer.yMaximum())
        except (RuntimeError, AttributeError):
            return None

    def _measure_zone_fill(self, layer, zone_in_layer) -> float:

        from ...core.tile_manager import MAX_DETAIL_LEVEL

        try:
            probe = self._probe_detail_level(layer, zone_in_layer)
            sized = (None if probe is None else self._grid_for_detail(
                layer, zone_in_layer, min(probe, MAX_DETAIL_LEVEL)))
        except (RuntimeError, AttributeError, TypeError, ValueError):



            return 1.0
        if sized is None:
            return 1.0
        pixel_w, pixel_h, mupp, _tiles = sized
        try:
            tiles = self._tile_manager.compute_grid(pixel_w, pixel_h, apply_cap=False)
        except (RuntimeError, AttributeError, TypeError):
            return 1.0
        if not tiles:
            return 1.0
        bbox = (zone_in_layer.xMinimum(),
                zone_in_layer.yMaximum() - pixel_h * mupp,
                zone_in_layer.xMinimum() + pixel_w * mupp,
                zone_in_layer.yMaximum())
        try:
            kept = self._tiles_in_polygon(
                tiles, bbox, pixel_w, pixel_h, layer, self._grid_crs_authid(layer))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return 1.0
        if not kept:



            return 1.0
        return min(1.0, max(len(kept) / len(tiles), 1.0 / len(tiles)))

    def _probe_detail_level(self, layer, zone_in_layer) -> int | None:






        from ...core.tile_manager import MAX_DETAIL_LEVEL

        for n in range(1, MAX_DETAIL_LEVEL + 1):
            sized = self._grid_for_detail(layer, zone_in_layer, n)
            if sized is None:
                return None
            tiles = sized[3]
            if tiles == -1:
                return max(1, n - 1)
            if tiles >= _PROBE_TILES:
                return n
        return MAX_DETAIL_LEVEL

    def _grid_crs_authid(self, layer) -> str | None:





        try:
            from .auto_flow import _crs_run_identifier
            return (_crs_run_identifier(self._run_crs_now(layer))
                    or _crs_run_identifier(layer.crs()))
        except (RuntimeError, AttributeError, ImportError):
            return None

    def _tiles_after_cull(self, layer, zone_in_layer, bbox_tiles: int) -> int:






        if bbox_tiles <= 0:
            return bbox_tiles
        fraction = self._zone_fill_fraction(layer, zone_in_layer)
        if fraction >= 1.0:
            return bbox_tiles
        return max(1, int(round(bbox_tiles * fraction)))

    def _tiles_after_cull_confirmed(
        self, layer, zone_in_layer, detail_n: int, bbox_tiles: int, cap: int,
    ) -> int:
















        estimate = self._tiles_after_cull(layer, zone_in_layer, bbox_tiles)
        if estimate <= cap:
            return estimate
        exact = self._exact_kept_tiles(layer, zone_in_layer, detail_n)
        if exact is None:
            return estimate
        if bbox_tiles > 0:
            self._zone_fill_cache = (self._zone_fill_key(layer, zone_in_layer),
                                     min(1.0, max(exact / bbox_tiles,
                                                  1.0 / bbox_tiles)))
        return exact

    def _exact_kept_tiles(self, layer, zone_in_layer, detail_n: int) -> int | None:


        sized = self._grid_for_detail(layer, zone_in_layer, detail_n)
        if sized is None:
            return None
        pixel_w, pixel_h, mupp, _tiles = sized
        try:
            grid = self._tile_manager.compute_grid(
                pixel_w, pixel_h, apply_cap=False)
        except (RuntimeError, AttributeError, TypeError):
            return None
        if not grid:
            return None
        bbox = (zone_in_layer.xMinimum(),
                zone_in_layer.yMaximum() - pixel_h * mupp,
                zone_in_layer.xMinimum() + pixel_w * mupp,
                zone_in_layer.yMaximum())
        try:
            kept = self._tiles_in_polygon(
                grid, bbox, pixel_w, pixel_h, layer,
                self._grid_crs_authid(layer))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return None
        return len(kept) if kept else None
