"""Hold the picture the map already shows while a new layer is drawn onto it.

Committing a run hands the map over from the live in-memory layer to the saved
GeoPackage layer. Drawing that saved layer takes as long as its polygon count
says, and while it draws, the canvas replaces the finished picture on screen
with the half-drawn one a few times a second. On a dense result that reads as
the detections vanishing right after the user asked to save them, then filling
back in band by band.

Parking the canvas update interval for the length of that one redraw keeps the
polygons the user is already looking at on screen until the new layer is fully
drawn, so the handover is a single swap. The hold is lifted the moment the
redraw completes, the moment the user moves the map, or once the timeout
:func:`hold_timeout_ms` reports has passed, whichever comes first: a still map
is only better than a filling one for as long as the fill would have taken.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QTimer

# Longer than any redraw, so no partial picture reaches the screen while the
# hold is on. The hold is always lifted below, never by this interval elapsing.
_PARKED_UPDATE_INTERVAL_MS = 3_600_000

# A redraw still running after this stops being a handover and starts being a
# frozen map, so past it the canvas goes back to filling in as it draws.
# Shipped value, and the fallback the getter below returns.
HOLD_TIMEOUT_MS = 8_000

# One hold per canvas: a second commit while the first is still drawing must
# not save the parked interval as if it were the real one.
_holds: dict[int, _CanvasPictureHold] = {}


def hold_timeout_ms() -> int:
    """How long the map may stay held, in milliseconds.

    Bounded on both sides: below the floor the hold expires inside the redraw
    it exists to cover, above the ceiling a stuck redraw leaves the map frozen
    for a full minute. Cache-only and never raises, so it is safe to read on
    the commit path.
    """
    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("ui.canvas_hold_timeout_ms", HOLD_TIMEOUT_MS, 500, 60_000))
    except Exception:  # noqa: BLE001 -- the timeout is best-effort  # nosec B110
        return HOLD_TIMEOUT_MS


def hold_map_picture_during_redraw(canvas) -> None:
    """Keep what the canvas shows until the redraw that follows is complete.

    Call right before the layer change whose redraw would otherwise blank the
    map. Best-effort: any failure leaves the canvas exactly as it was.
    """
    if canvas is None:
        return
    existing = _holds.get(id(canvas))
    if existing is not None:
        if existing.is_live():
            return
        # The canvas that hold belonged to is gone and Python handed its id to
        # this one: the entry is stale, not a hold on this canvas.
        _holds.pop(id(canvas), None)
    try:
        _holds[id(canvas)] = _CanvasPictureHold(canvas)
    except (RuntimeError, AttributeError, TypeError):
        _holds.pop(id(canvas), None)


def release_map_picture_hold(canvas) -> None:
    """Lift the hold on this canvas now, if one is on. Called from teardown so
    a canvas is never left parked by a plugin that is going away."""
    hold = _holds.get(id(canvas)) if canvas is not None else None
    if hold is not None:
        hold.release()


class _CanvasPictureHold:
    def __init__(self, canvas) -> None:
        self._canvas = canvas
        self._saved_interval = canvas.mapUpdateInterval()
        self._armed = False
        canvas.setMapUpdateInterval(_PARKED_UPDATE_INTERVAL_MS)
        # A redraw already running when the hold goes on is NOT the one we are
        # waiting for, and its completion must not lift the hold: arm on the
        # next redraw to start, which is the one carrying the new layer.
        canvas.renderStarting.connect(self._on_render_starting)
        canvas.mapCanvasRefreshed.connect(self._on_map_refreshed)
        canvas.extentsChanged.connect(self.release)
        self._timer = QTimer(canvas)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.release)
        self._timer.start(hold_timeout_ms())

    def is_live(self) -> bool:
        """True while this hold still has a canvas with a C++ side."""
        if self._canvas is None:
            return False
        try:
            self._canvas.mapUpdateInterval()
        except (RuntimeError, AttributeError):
            return False
        return True

    def _on_render_starting(self) -> None:
        self._armed = True

    def _on_map_refreshed(self) -> None:
        # isDrawing means the layer change split into more than one redraw and
        # the screen is still waiting on the later one.
        if not self._armed:
            return
        try:
            if self._canvas is not None and self._canvas.isDrawing():
                return
        except (RuntimeError, AttributeError):
            pass
        self.release()

    def release(self) -> None:
        canvas, self._canvas = self._canvas, None
        if canvas is None:
            return
        _holds.pop(id(canvas), None)
        try:
            self._timer.stop()
        except RuntimeError:  # nosec B110 - timer died with its parent canvas
            pass
        for signal, slot in ((canvas.renderStarting, self._on_render_starting),
                             (canvas.mapCanvasRefreshed, self._on_map_refreshed),
                             (canvas.extentsChanged, self.release)):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):  # nosec B110 - already gone
                pass
        try:
            canvas.setMapUpdateInterval(self._saved_interval)
        except (RuntimeError, AttributeError):  # nosec B110
            pass
