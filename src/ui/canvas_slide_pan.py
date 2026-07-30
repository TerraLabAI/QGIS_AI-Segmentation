"""Pan a map canvas by sliding what is already drawn, with no re-render.

Every map tool in this plugin that lets the user move the map carries its own
pan, because the tool is on the canvas in place of the QGIS pan tool and a
gesture the user makes has to reach somebody. What those pans used to do was
move the extent on each mouse event and ask for a full canvas render on a
throttle. That is the expensive way round: a render redraws every layer, and on
a live basemap it cancels and restarts the tile job on each pixel of motion, so
the map stutters exactly while the user is dragging it.

QGIS's own pan tool never does that. The rendered image and every rubber band
on it live in one graphics scene, so offsetting that scene slides the whole
picture under the cursor for the price of a widget scroll. The extent holds
still for the length of the gesture, which also keeps ``toMapCoordinates`` and
every hit test agreeing with what is on screen. The extent moves once, at the
end, with a single render.

The canvas method QGIS uses for the start of that gesture is not in PyQGIS
(``panActionStart``), so this does the same three steps directly: offset the
scene, put it back square, then move the centre by the distance travelled.

One instance per tool, held for the tool's life. ``begin_at`` on the gesture's
first point, ``slide_to`` on each move, ``commit`` when the user lets go.
Nothing here raises: a pan must never be what breaks a tool.
"""
from __future__ import annotations

from qgis.core import QgsPointXY

# Everything below talks to a canvas that a teardown may already have taken
# away, hence the same guard on every call.
_CANVAS_GONE = (RuntimeError, AttributeError, TypeError)


class CanvasSlidePan:
    """The slide-pan gesture for one map canvas."""

    def __init__(self, canvas) -> None:
        self._canvas = canvas
        self._anchor = None
        self._last = None

    @property
    def sliding(self) -> bool:
        """True between ``begin_at`` and ``commit``."""
        return self._anchor is not None

    def begin_at(self, screen_point) -> None:
        """The gesture starts here (a QPoint in canvas pixels)."""
        self._anchor = screen_point
        self._last = screen_point

    def slide_to(self, screen_point) -> None:
        """The cursor is here now: slide the drawn picture to follow it.

        The offset is measured from the anchor every time, never accumulated,
        so a dropped move event cannot leave the picture lagging the cursor.
        """
        anchor = self._anchor
        if anchor is None:
            self.begin_at(screen_point)
            return
        self._last = screen_point
        try:
            size = self._canvas.viewport().size()
            self._canvas.setSceneRect(
                anchor.x() - screen_point.x(), anchor.y() - screen_point.y(),
                size.width(), size.height())
        except _CANVAS_GONE:
            pass

    def commit(self, screen_point=None) -> None:
        """End the gesture: turn the slid picture into the extent it stands for.

        The scene goes back square FIRST, so the one render that follows lands
        on a canvas sitting straight. A gesture that moved nothing (a Space tap)
        renders nothing: re-rendering there would restart the whole basemap tile
        job for a map that never moved.
        """
        anchor, last = self._anchor, screen_point or self._last
        self._anchor = None
        self._last = None
        if anchor is None:
            return
        self.square_the_scene()
        if last is None or (last.x() == anchor.x() and last.y() == anchor.y()):
            return
        try:
            transform = self._canvas.getCoordinateTransform()
            start = transform.toMapCoordinates(anchor.x(), anchor.y())
            end = transform.toMapCoordinates(last.x(), last.y())
            center = self._canvas.center()
            self._canvas.setCenter(QgsPointXY(
                center.x() + start.x() - end.x(),
                center.y() + start.y() - end.y(),
            ))
            self._canvas.refresh()
        except _CANVAS_GONE:
            pass

    def abandon(self) -> None:
        """Drop the gesture and leave the map where it was. For a tool that is
        going away with nothing to commit to."""
        self._anchor = None
        self._last = None
        self.square_the_scene()

    def square_the_scene(self) -> None:
        """Put the scene offset back to zero. Safe to call at any time, and the
        safety net for a gesture whose end the tool never sees, which would
        otherwise leave the map parked off-centre."""
        try:
            size = self._canvas.viewport().size()
            self._canvas.setSceneRect(0, 0, size.width(), size.height())
        except _CANVAS_GONE:
            pass
