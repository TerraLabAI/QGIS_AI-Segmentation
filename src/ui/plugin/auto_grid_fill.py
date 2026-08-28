"""How much of a zone's bounding-box grid the run actually sends.

Every Automatic zone is a hand-drawn polygon. The tile grid is built over its
BOUNDING BOX, then culled: a tile whose ground touches neither the polygon nor
the raster's own extent is never rendered, sent or billed. So two counts exist
for the same detail level, and they are far apart. Over 27 archived user runs
only 26% of the bounding-box grid was inside the zone, and the widest zones
were the worst: a road run kept 10%, a building run over Paris kept 19%.

The seed walk compares a level's cost against a tile cap. Comparing the
bounding-box count stopped it several levels early on exactly the long, thin,
diagonal zones where the gap is largest, and those are the zones people draw
for a corridor, a river bank or an administrative boundary.

This module gives the walk the count the run really pays. The cull itself is
`_tiles_in_polygon` (auto_zone), which is what the run and the on-screen
estimate already use, so all three read one number.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); methods are plain
mixin members and state lives on the instance.
"""
from __future__ import annotations

# Bounding-box tile count the fill probe aims for. Big enough that boundary
# tiles are a small share of the answer, so the fraction it measures is close
# to the one a much finer grid would give; small enough that one cull is cheap.
_PROBE_TILES = 256


class AutoGridFillMixin:
    """The share of a level's bounding-box grid that survives the zone cull."""

    def _zone_fill_fraction(self, layer, zone_in_layer) -> float:
        """Share of a bounding-box grid the drawn zone keeps, 0 < f <= 1.

        Measured once per zone by culling a probe grid of about `_PROBE_TILES`
        tiles, then reused for every level. It is a slight OVER-estimate for
        finer levels, because a coarse grid keeps proportionally more boundary
        tiles than a fine one, so a level's cost is never understated and the
        seed can never propose a run the cap would refuse.

        Returns 1.0 (no adjustment, the behaviour before this existed) when
        there is no polygon, when the probe cannot be built, or on any error.
        """
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
        """Cache key for one zone on one layer, or None when either cannot be
        read (which the callers treat as "no adjustment")."""
        try:
            return (layer.id(), zone_in_layer.xMinimum(), zone_in_layer.yMinimum(),
                    zone_in_layer.xMaximum(), zone_in_layer.yMaximum())
        except (RuntimeError, AttributeError):
            return None

    def _measure_zone_fill(self, layer, zone_in_layer) -> float:
        """One cull of a probe grid. See `_zone_fill_fraction` for the contract."""
        from ...core.tile_manager import MAX_DETAIL_LEVEL

        try:
            probe = self._probe_detail_level(layer, zone_in_layer)
            sized = (None if probe is None else self._grid_for_detail(
                layer, zone_in_layer, min(probe, MAX_DETAIL_LEVEL)))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            # Before the dock opens there is no tile manager to size a grid
            # with. Fail to "no adjustment" rather than let the seed walk
            # unwind into its own fallback.
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
        # _tiles_in_polygon never culls to empty: it hands back the whole grid
        # rather than an empty run. That safety fallback reads here as "no
        # polygon", which is the right answer for a fraction.
        return min(1.0, max(len(kept) / len(tiles), 1.0 / len(tiles)))

    def _probe_detail_level(self, layer, zone_in_layer) -> int | None:
        """Lowest detail level whose bounding-box grid reaches `_PROBE_TILES`.

        Walking up rather than picking a fixed level keeps the probe the same
        size whatever the zone's shape, so a long thin zone and a square one
        both pay one cull of about the same cost.
        """
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
        """Authid of the run CRS, for the cull's raster-extent test.

        None skips that test and leaves the polygon test alone, which is the
        larger effect and the one this module exists for.
        """
        try:
            from .auto_flow import _crs_run_identifier
            return (_crs_run_identifier(self._run_crs_now(layer))
                    or _crs_run_identifier(layer.crs()))
        except (RuntimeError, AttributeError, ImportError):
            return None

    def _tiles_after_cull(self, layer, zone_in_layer, bbox_tiles: int) -> int:
        """A level's bounding-box tile count scaled to what the run will send.

        The walk's cap comparison goes through here, so a zone that fills a
        quarter of its own bounding box is allowed four times the levels it
        used to get. Never returns less than 1.
        """
        if bbox_tiles <= 0:
            return bbox_tiles
        fraction = self._zone_fill_fraction(layer, zone_in_layer)
        if fraction >= 1.0:
            return bbox_tiles
        return max(1, int(round(bbox_tiles * fraction)))

    def _tiles_after_cull_confirmed(
        self, layer, zone_in_layer, detail_n: int, bbox_tiles: int, cap: int,
    ) -> int:
        """Like `_tiles_after_cull`, but exact when the answer decides the walk.

        The probe measures the kept share on a small grid, where boundary tiles
        are a large part of the answer, so it reads HIGH: on a 20 km corridor
        200 m wide it says 18% where the level the walk actually reaches keeps
        3.9%. That only ever costs levels, never spends them, but on a long
        thin zone it costs four or five of them, and those are the zones this
        whole adjustment exists for.

        So when the scaled count says the level is over the cap, cull that
        level exactly before believing it, and keep the measured share for the
        rest of the walk. The share falls as levels get finer, so replacing the
        probe with a real measurement taken further along only ever improves
        it. One cull of 32000 tiles costs 134 ms measured, and the walk breaks
        right after, so it is paid at most a couple of times per zone.
        """
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
        """Tiles one level really sends, culled against the drawn polygon and
        the raster's extent. None when the grid cannot be built or read."""
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
