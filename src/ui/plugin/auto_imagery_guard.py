"""Pre-run guard: does the chosen layer actually show an image over the zone?

An online tile source that holds no picture of a piece of ground at a given
level of detail does not fail the request. It answers with a placeholder card,
a flat grey tile carrying a line of text. Rendered, that card is real pixels,
so every check below the render reads it as ground: the run sends a zone of
grey cards, the model finds nothing in them, and the user pays for it.

This guard renders the same small window at several depths before any billable
work, and reads them together. Two things come out of that which one picture
cannot give:

- A flat grey card and flat grey GROUND look alike. Salt flats, ice and bare
  desert all trip the single-picture test, and a run over them used to be
  refused for imagery that was really there. Ground looks like itself one level
  coarser; a card does not, because one level down the source has a picture.
- When the run's own level holds no picture but a coarser one does, the answer
  is not to stop. The run falls back to the coarsest level that has imagery and
  goes ahead there, which costs the user fewer tiles than they were quoted.

It fails open at every step: a probe that times out, errors, or returns
anything the check cannot name lets the run start exactly as it would have.
"""
from __future__ import annotations

from ...core.i18n import tr

# Ground the probe covers, in run-resolution pixels. It has to render at the
# RUN's resolution to ask the right question (a whole-zone probe renders at a
# coarser zoom, where a source usually does have imagery even when it has none
# at the run's), so the window is kept small instead: enough pixels to
# recognise a card, few enough that the user barely waits for it.
_PROBE_SIDE_PX: int = 256


class AutoImageryGuardMixin:
    """Refuses, or coarsens, an Automatic run whose source shows no picture."""

    def _online_imagery_verdict(self, layer, grid) -> tuple[float, str | None]:
        """``(mupp_floor, refusal message)`` for this run.

        A floor above 0 is a resolution the run must not go finer than: the
        source holds no picture at the one asked for, but does at that one.
        The message is set only when no level the guard tried had imagery.
        ``(0.0, None)`` means run exactly as asked, and is what every failure
        inside here returns.

        Online sources only: a local raster has no placeholder card, and its
        own empty regions are already handled by the zone-outside-layer guard
        and the per-tile blank skip.

        ``grid`` is the dict from ``_compute_auto_grid``: its bbox and pixel
        size give the run's ground resolution, which is the whole point.
        """
        if not getattr(self, "_auto_source_is_online", False):
            return 0.0, None
        from ...core.server_dials import feature_enabled

        # Off = the behaviour that shipped before this guard existed: the run
        # starts and the per-tile skip catches the cards. Nothing is lost, so
        # the switch is safe to pull mid-incident.
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
                # The worst window decides. One arm of the zone with no picture
                # is an arm of grey cards the user pays for.
                if refusal is not None:
                    return 0.0, refusal
                floor = max(floor, level)
        except Exception:  # noqa: BLE001 - a probe must never block a run
            return 0.0, None
        return floor, None

    def _note_imagery_backoff(self) -> None:
        """Tell the user the basemap, not the slider, set the run's detail.

        The Precision warning already on the panel asks for a higher precision
        or a closer zoom, and neither is an action here: the source has no
        sharper picture of this ground whatever the slider says. So this note
        names the source as the limit and says the run went ahead. Served
        through ``dial_copy`` so the wording can be retuned without a release,
        and it never raises: a note is not worth failing a run over.
        """
        if getattr(self, "_auto_headless_run", False):
            return
        try:
            from ...core.server_dials import dial_copy

            self.iface.messageBar().pushInfo("AI Segmentation", dial_copy(
                "auto.imagery_backoff",
                tr("This map source has no sharper picture of this area. The "
                   "run uses the sharpest one it has, and costs fewer tiles "
                   "than the estimate.")))
        except Exception:  # noqa: BLE001 - a note never fails a run  # nosec B110
            pass

    def _walk_source_depth(self, layer, extent, mupp, grid):
        """Render the probe window at each depth in turn and pick a level.

        One window, rendered into half as many pixels each time, so every step
        asks the source for one zoom level less over exactly the same ground.
        Consecutive renders are compared in pairs, which means each render
        serves twice and the whole walk costs one small render per level.
        """
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
            # The flat-and-grey test accused this render. It is only a card if
            # the level below shows a different place; flat ground shows the
            # same one, however little there is in it.
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
        """The resolution floor for a level ``step`` coarser than the run's.

        0.0 at step 0, because the run already renders there and a floor equal
        to its own resolution would be a no-op that still had to travel.
        """
        return 0.0 if step <= 0 else float(mupp) * (2 ** step)

    @staticmethod
    def _grid_mupp(grid) -> float:
        """Run-CRS units per pixel of a computed grid, or 0.0 when unreadable."""
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
        """The run CRS as a QgsCoordinateReferenceSystem, or None to render in
        the layer's own (which is what the run does when it named no CRS)."""
        from qgis.core import QgsCoordinateReferenceSystem

        authid = (grid or {}).get("crs") or ""
        if not authid:
            return None
        crs = QgsCoordinateReferenceSystem(authid)
        return crs if crs.isValid() else None

    def _probe_centres(self, layer, grid) -> list:
        """Points to centre the probe windows on, in the run CRS.

        The zone's bounding-box centre falls OUTSIDE an L-shaped, diagonal or
        corridor zone, so one window there tests ground the run never reads: a
        zone whose centre carries imagery and whose arms do not passed the
        guard, and the user paid for grey cards. These points all sit ON the
        polygon's surface and are spread along it, so no arm goes unlooked at.
        Falls back to the bounding-box centre when there is no drawn polygon
        (the rectangle and headless paths), which is the old behaviour.
        """
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
        """A _PROBE_SIDE_PX square of ground at the run's resolution, centred
        on ``centre``, as a QgsRectangle in the run CRS.

        ``centre`` is an (x, y) pair from :meth:`_probe_centres`; None falls
        back to the middle of the grid's own extent.

        None when the grid carries no usable extent, which fails the guard open.
        Clamped to the zone so a zone smaller than the probe window is not
        probed on ground the run never reads.
        """
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
        # Run-CRS units per pixel, per axis. The automatic grid makes square
        # pixels, so these agree; reading both keeps the probe honest if that
        # ever stops being true.
        # The same served side the probe renders at, so the square of ground
        # and the pixels it lands in agree.
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
