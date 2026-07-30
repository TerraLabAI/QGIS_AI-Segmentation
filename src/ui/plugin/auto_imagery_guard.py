"""Pre-run guard: does the chosen layer actually show an image over the zone?

An online tile source that holds no picture of a piece of ground at a given
level of detail does not fail the request. It answers with a placeholder card,
a flat grey tile carrying a line of text. Rendered, that card is real pixels,
so every check below the render reads it as ground: the run sends a zone of
grey cards, the model finds nothing in them, and the user pays for it.

This guard renders ONE small window at the run's own resolution before any
billable work and refuses the run when what comes back is a card. It fails
open: a probe that times out, errors, or returns anything the check cannot
name lets the run start exactly as before.
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
    """Refuses an Automatic run whose imagery source shows no picture."""

    def _online_imagery_probe_message(self, layer, grid) -> str | None:
        """The message to refuse this run with, or None to let it start.

        Online sources only: a local raster has no placeholder card, and its
        own empty regions are already handled by the zone-outside-layer guard
        and the per-tile blank skip.

        ``grid`` is the dict from ``_compute_auto_grid``: its bbox and pixel
        size give the run's ground resolution, which is the whole point of the
        probe.
        """
        if not getattr(self, "_auto_source_is_online", False):
            return None
        from ...core.server_dials import feature_enabled

        # Off = the behaviour that shipped before this guard existed: the run
        # starts and the per-tile skip catches the cards. Nothing is lost, so
        # the switch is safe to pull mid-incident.
        if not feature_enabled("imagery_probe"):
            return None
        extent = self._imagery_probe_extent(grid)
        if extent is None:
            return None
        from ...core.cloud_detection import probe_imagery_availability

        try:
            crs = self._probe_render_crs(grid)
            verdict = probe_imagery_availability(layer, extent, render_crs=crs)
        except Exception:  # noqa: BLE001 - a probe must never block a run
            return None
        if verdict != "unavailable":
            return None
        return tr(
            "This layer has no image over your zone at this precision. "
            "The map source answered with an empty tile, so there is nothing "
            "to detect on. Lower Precision, zoom the layer out until the "
            "imagery shows, or pick a layer that covers this area."
        )

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

    @staticmethod
    def _imagery_probe_extent(grid):
        """A _PROBE_SIDE_PX square of ground at the run's resolution, centred
        on the zone, as a QgsRectangle in the run CRS.

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
        half_w = min(width, _PROBE_SIDE_PX * width / pixel_w) / 2.0
        half_h = min(height, _PROBE_SIDE_PX * height / pixel_h) / 2.0
        cx = (float(minx) + float(maxx)) / 2.0
        cy = (float(miny) + float(maxy)) / 2.0
        return QgsRectangle(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
