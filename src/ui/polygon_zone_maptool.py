"""Point-by-point polygon zone selection for Automatic detection.

Designed to be obvious, following the standard polygon-drawing UX (Leaflet.draw,
Mapbox GL Draw, Google Maps): you click to drop points, each one shows as a clear
blue dot, a solid line connects every placed point as you go, the interior fills
translucent blue and rubber-bands live with the cursor from three points on, and a
dashed segment trails the cursor. Finish in whatever feels natural:

  - double-click, or
  - press Enter, or
  - click the first point again (it lights up green when you are close).

Right-click is deliberately inert (it used to finish, which fired on accidental
context-clicks). Ctrl+Z (or Backspace) removes the last point. Escape clears the
in-progress points; with none placed it asks to go back a step. Everything is the
brand blue so it matches the committed zone outline. Works in the map canvas CRS
and emits the finished polygon as a QgsGeometry.

Rendering uses three separate rubber bands on purpose: a single polygon band only
draws once it has 3+ points, which makes the first edges and the fill seem to
"disappear". A dedicated line band for the edges shows every segment from the
second point on; a polygon band handles the live fill.
"""
from __future__ import annotations

import logging

from qgis.core import QgsGeometry, QgsPointXY
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor

from ..core.qt_compat import LineGeometry, PolygonGeometry, VertexIconCircle, event_pos
from .canvas_palette import (
    CHROME_BLUE,
    CLOSE_DOT_OK,
    CLOSE_DOT_WARN,
    ZONE_DRAW_FILL,
    ZONE_DRAW_LINE,
)

logger = logging.getLogger("AISegmentation")

# Brand blue, matching the committed zone outline (_show_zone_polygon_band).
_BLUE = CHROME_BLUE
_BLUE_LINE = ZONE_DRAW_LINE
_BLUE_FILL = ZONE_DRAW_FILL
_TRANSPARENT = QColor(0, 0, 0, 0)
# "Click here to close" highlight on the first point when the cursor is near it.
_CLOSE_GREEN = CLOSE_DOT_OK


class PolygonZoneMapTool(QgsMapTool):
    """Draw a polygon zone one point at a time, with live visual feedback.

    Emits ``zone_selected(QgsGeometry)`` (polygon, canvas CRS) on finish,
    ``zone_cleared()`` on cancel / too-few-points, and ``vertices_changed(int)``
    every time the point count changes so the dock can show live guidance.
    """

    zone_selected = pyqtSignal(QgsGeometry)
    zone_cleared = pyqtSignal()
    vertices_changed = pyqtSignal(int)
    # Ctrl+Z / Backspace / Escape on an EMPTY canvas (no points yet) means
    # "go back" a step, not "undo a point" or "clear". The owner turns this
    # into leaving the draw step (zone) or disarming (exemplar).
    back_requested = pyqtSignal()

    MIN_VERTICES: int = 3
    # Ignore a click within this many screen pixels of the last point: it is a
    # double-click's second hit or an accidental jitter, not a new vertex.
    MIN_STEP_PX: int = 6
    # Cursor within this many pixels of the first point closes the polygon.
    CLOSE_PX: int = 14

    def __init__(self, canvas, color: "QColor | None" = None):
        super().__init__(canvas)
        self._canvas = canvas
        self._points: list[QgsPointXY] = []
        # Colors: brand blue for the segmentation zone (default), or a caller
        # color (e.g. green) for an example polygon, so the same point-by-point
        # UX is reused with a different accent.
        base = color or _BLUE
        self._c_main = base
        self._c_line = QColor(base.red(), base.green(), base.blue(), 235)
        self._c_fill = QColor(base.red(), base.green(), base.blue(), 55)
        # Close-highlight on the first point: green reads well over blue, but
        # would vanish on a green/red example polygon, so use amber when a
        # custom color is supplied.
        self._c_close = _CLOSE_GREEN if color is None else CLOSE_DOT_WARN
        # Translucent interior, drawn under the edges. No stroke of its own: the
        # edges band is the visible outline, so the fill never doubles it.
        self._fill_band = QgsRubberBand(canvas, PolygonGeometry)
        self._fill_band.setColor(self._c_fill)
        self._fill_band.setStrokeColor(_TRANSPARENT)
        self._fill_band.setWidth(0)
        # Solid edges through the placed points: visible from the second point on.
        self._edges_band = QgsRubberBand(canvas, LineGeometry)
        self._edges_band.setColor(self._c_line)
        self._edges_band.setWidth(3)
        # Dashed segment from the last point to the cursor (the next edge).
        self._preview_band = QgsRubberBand(canvas, LineGeometry)
        self._preview_band.setColor(self._c_line)
        self._preview_band.setWidth(2)
        self._preview_band.setLineStyle(Qt.PenStyle.DashLine)
        self._markers: list[QgsVertexMarker] = []
        self._can_close = False
        self._space_panning = False
        self._pan_last = None
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    # -- ShortcutFilter space-pan hooks (compatible with the old tool) --
    def start_space_pan(self) -> None:
        self._space_panning = True
        self._pan_last = self._canvas.mapFromGlobal(QCursor.pos())
        self._canvas.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def stop_space_pan(self) -> None:
        self._space_panning = False
        self._pan_last = None
        self._canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def set_snap_context(self, *args, **kwargs) -> None:
        """No-op (kept so the shared activation path can call it blindly)."""
        return

    # -- Mouse --
    def canvasMoveEvent(self, event):  # noqa: N802 (Qt API)
        if self._space_panning:
            self._do_pan(event_pos(event))
            return
        if not self._points:
            return
        pos = event_pos(event)
        self._can_close = self._near_first(pos)
        cursor = self._points[0] if self._can_close else self.toMapCoordinates(pos)
        # Dashed edge to the cursor (snaps to the first point when closable).
        self._preview_band.setToGeometry(
            QgsGeometry.fromPolylineXY([self._points[-1], cursor]), None)
        # Live fill that rubber-bands with the cursor (an area from 3 vertices,
        # counting the cursor as the next one, i.e. from the 2nd placed point).
        self._draw_fill(self._points + [cursor])
        self._highlight_first(self._can_close)

    def canvasReleaseEvent(self, event):  # noqa: N802 (Qt API)
        if self._space_panning:
            return
        if event.button() == Qt.MouseButton.RightButton:
            # Right-click is deliberately inert mid-draw (it used to finish,
            # which fired on accidental context-clicks). Consume the event so
            # it cannot reach any other handler.
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._can_close and len(self._points) >= self.MIN_VERTICES:
            self._finish()
            return
        pos = event_pos(event)
        self._add_point(self.toMapCoordinates(pos), pos)

    def canvasDoubleClickEvent(self, event):  # noqa: N802 (Qt API)
        # The two single clicks of the double-click are deduped by MIN_STEP_PX,
        # so by here the real vertices are already in; just finish.
        if event.button() == Qt.MouseButton.LeftButton:
            self._finish()

    def keyPressEvent(self, event):  # noqa: N802 (Qt API)
        key = event.key()
        if key == Qt.Key.Key_Escape:
            # Runs only when no dock shortcut / canvas filter consumed the
            # press first; mirrors _route_escape's two-stage decision so one
            # Escape press does exactly one stage on every channel.
            self._cancel_or_back()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish()
        elif key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self._undo_or_back()
        elif key == Qt.Key.Key_Z and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._undo_or_back()
        else:
            super().keyPressEvent(event)

    def _undo_or_back(self) -> None:
        """Undo the last point, or (when none are placed yet) ask to go back a
        step. Ctrl+Z maps to Cmd+Z on macOS via Qt's ControlModifier."""
        if self._points:
            self._undo_last()
        else:
            self.back_requested.emit()

    def _cancel_or_back(self) -> None:
        """Escape: clear the in-progress points and stay armed, or (when none
        are placed) ask to go back a step. The owner decides what "back" means
        (zone draw: exit the flow; exemplar draw: disarm)."""
        if self._points:
            self._cancel()
        else:
            self.back_requested.emit()

    def has_points(self) -> bool:
        """True while at least one vertex of an in-progress draw is placed."""
        return bool(self._points)

    # -- Internals --
    def _add_point(self, map_pt: QgsPointXY, screen_pt) -> None:
        if self._points:
            last_screen = self.toCanvasCoordinates(self._points[-1])
            if (abs(screen_pt.x() - last_screen.x()) < self.MIN_STEP_PX
                    and abs(screen_pt.y() - last_screen.y()) < self.MIN_STEP_PX):  # noqa: W503
                return  # duplicate / double-click second hit
        self._points.append(map_pt)
        self._add_marker(map_pt, first=len(self._points) == 1)
        self._draw_edges()
        self._draw_fill(self._points)
        self.vertices_changed.emit(len(self._points))

    def _undo_last(self) -> None:
        if not self._points:
            return
        self._points.pop()
        if self._markers:
            self._canvas.scene().removeItem(self._markers.pop())
        self._restyle_markers()
        self._draw_edges()
        self._draw_fill(self._points)
        if not self._points:
            self._preview_band.reset(LineGeometry)
        self.vertices_changed.emit(len(self._points))

    def _finish(self) -> None:
        pts = list(self._points)
        if len(pts) < self.MIN_VERTICES:
            return  # not enough to be a shape yet; keep drawing
        geom = QgsGeometry.fromPolygonXY([pts])
        self._reset_visuals()
        self._points = []
        if geom.isEmpty():
            self.vertices_changed.emit(0)
            self.zone_cleared.emit()
            return
        self.vertices_changed.emit(0)
        self.zone_selected.emit(geom)

    def _cancel(self) -> None:
        self._reset_visuals()
        self._points = []
        self.vertices_changed.emit(0)
        self.zone_cleared.emit()

    def clear_selection(self) -> None:
        self._cancel()

    def _draw_edges(self) -> None:
        # setToGeometry (not reset + addPoint) so the band is shown + repainted:
        # the addPoint(doUpdate=False) path leaves a rubber band hidden, which is
        # why the lines used to "disappear". Geometry is in canvas CRS (layer=None).
        if len(self._points) >= 2:
            self._edges_band.setToGeometry(
                QgsGeometry.fromPolylineXY(self._points), None)
        else:
            self._edges_band.reset(LineGeometry)

    def _draw_fill(self, pts: list) -> None:
        if len(pts) >= 3:
            self._fill_band.setToGeometry(
                QgsGeometry.fromPolygonXY([list(pts)]), None)
        else:
            self._fill_band.reset(PolygonGeometry)

    def _add_marker(self, pt: QgsPointXY, first: bool) -> None:
        m = QgsVertexMarker(self._canvas)
        m.setCenter(pt)
        m.setIconType(VertexIconCircle)
        m.setColor(self._c_main)
        try:
            m.setFillColor(QColor(255, 255, 255))
        except (AttributeError, TypeError):
            pass  # older builds: outline-only marker is still clearly visible
        m.setPenWidth(3)
        # The first point is a touch bigger: it is the one you click to close.
        m.setIconSize(13 if first else 10)
        m.setZValue(1000)
        self._markers.append(m)

    def _restyle_markers(self) -> None:
        for i, m in enumerate(self._markers):
            m.setIconSize(13 if i == 0 else 10)
            m.setColor(self._c_main)

    def _highlight_first(self, hot: bool) -> None:
        if not self._markers or len(self._points) < self.MIN_VERTICES:
            return
        first = self._markers[0]
        first.setColor(self._c_close if hot else self._c_main)
        first.setIconSize(18 if hot else 13)

    def _near_first(self, screen_pt) -> bool:
        if len(self._points) < self.MIN_VERTICES:
            return False
        first_screen = self.toCanvasCoordinates(self._points[0])
        return (abs(screen_pt.x() - first_screen.x()) <= self.CLOSE_PX
                and abs(screen_pt.y() - first_screen.y()) <= self.CLOSE_PX)  # noqa: W503

    def _do_pan(self, pos) -> None:
        if self._pan_last is None:
            self._pan_last = pos
            return
        start = self.toMapCoordinates(self._pan_last)
        end = self.toMapCoordinates(pos)
        center = self._canvas.center()
        self._canvas.setCenter(QgsPointXY(
            center.x() + (start.x() - end.x()),
            center.y() + (start.y() - end.y()),
        ))
        self._canvas.refresh()
        self._pan_last = pos

    def _reset_visuals(self) -> None:
        self._edges_band.reset(LineGeometry)
        self._fill_band.reset(PolygonGeometry)
        self._preview_band.reset(LineGeometry)
        for m in self._markers:
            try:
                self._canvas.scene().removeItem(m)
            except (RuntimeError, AttributeError):
                pass
        self._markers = []
        self._can_close = False

    def deactivate(self) -> None:
        self._reset_visuals()
        self._points = []
        super().deactivate()
