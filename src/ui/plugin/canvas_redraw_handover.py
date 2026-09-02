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

A live run has the same problem on every frame, not once. Each tile that lands
drops the live layer's cached picture, and every redraw after that (a zoom, a
basemap tile arriving, the run's own repaint) paints the polygons back in
band by band, so they blink out and fill in again under the user's eyes. The
run hold below parks the same interval for the whole run: the canvas only
ever swaps one complete picture for the next, and a zoom shows the last full
picture stretched until the new one is drawn. One frame that outlives the
run hold's timeout gets its fill back, so a slow redraw never reads as a
frozen map, and the next frame is held again.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QTimer

# Longer than any redraw, so no partial picture reaches the screen while the
# hold is on. The hold is always lifted below, never by this interval elapsing.
_PARKED_UPDATE_INTERVAL_MS = 3_600_000

# A redraw still running after this stops being a handover and starts being a
# frozen map, so past it the canvas goes back to filling in as it draws.
# Shipped value, and the fallback the getter below returns.
#
# 2 s, not the 8 s this shipped with. The hold is nearly always lifted by the
# redraw finishing, so the number only ever shows itself when something has
# gone wrong, and then it is the whole of what the user feels: a map that will
# not answer. A normal redraw finishes well inside 2 s even on a dense layer,
# and a stuck one now costs two seconds instead of eight.
HOLD_TIMEOUT_MS = 2_000

# One hold per canvas: a second commit while the first is still drawing must
# not save the parked interval as if it were the real one.
_holds: dict[int, _CanvasPictureHold] = {}

# One run hold per canvas, kept apart from the commit hold above: the two
# overlap at commit (the run hold is lifted by the same teardown that follows
# the commit's own hold), and each must restore the interval the canvas had
# before EITHER went on, never the parked value the other one left.
_run_holds: dict[int, _LiveRunPictureHold] = {}

# A live run's frame may be held longer than a commit's: the polygons on screen
# are the run's whole feedback, and losing them on every zoom is what the hold
# exists to stop. Shipped value, and the fallback the getter returns.
LIVE_HOLD_TIMEOUT_MS = 3 * HOLD_TIMEOUT_MS


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


def live_hold_timeout_ms() -> int:
    """How long ONE frame may stay held during a live run, in milliseconds.

    Same bounds as :func:`hold_timeout_ms`. Cache-only and never raises.
    """
    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("ui.live_canvas_hold_timeout_ms", LIVE_HOLD_TIMEOUT_MS, 500, 60_000))
    except Exception:  # noqa: BLE001 -- the timeout is best-effort  # nosec B110
        return LIVE_HOLD_TIMEOUT_MS


def _live_hold_for(registry: dict, canvas):
    """The hold this registry has on ``canvas``, dropping a stale entry whose
    canvas is gone (Python may hand a dead canvas's id to a new one)."""
    hold = registry.get(id(canvas))
    if hold is None:
        return None
    if hold.is_live():
        return hold
    registry.pop(id(canvas), None)
    return None


def _interval_before_any_hold(canvas) -> int:
    """The update interval the canvas had before any hold parked it. Read from
    the hold already on, if there is one: the canvas itself only ever reports
    the parked value while a hold is on."""
    for registry in (_holds, _run_holds):
        other = _live_hold_for(registry, canvas)
        if other is not None:
            return other.saved_interval
    return canvas.mapUpdateInterval()


def _restore_interval(canvas, saved: int) -> None:
    """Give the canvas its interval back, unless another hold is still on, in
    which case the picture stays parked for that one."""
    for registry in (_holds, _run_holds):
        if _live_hold_for(registry, canvas) is not None:
            canvas.setMapUpdateInterval(_PARKED_UPDATE_INTERVAL_MS)
            return
    canvas.setMapUpdateInterval(saved)


def hold_map_picture_for_live_run(canvas) -> None:
    """Keep every picture the canvas shows complete until the run hold is
    lifted. Call when the live preview starts painting; lift it with
    :func:`release_live_run_picture_hold` on every run teardown. Best-effort:
    any failure leaves the canvas exactly as it was."""
    if canvas is None:
        return
    if _live_hold_for(_run_holds, canvas) is not None:
        return
    try:
        _run_holds[id(canvas)] = _LiveRunPictureHold(canvas)
    except (RuntimeError, AttributeError, TypeError):
        _run_holds.pop(id(canvas), None)


def release_live_run_picture_hold(canvas) -> None:
    """Lift the run hold on this canvas now, if one is on."""
    hold = _live_hold_for(_run_holds, canvas) if canvas is not None else None
    if hold is not None:
        hold.release()


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
        self.saved_interval = _interval_before_any_hold(canvas)
        self._armed = False
        self._timer = None
        canvas.setMapUpdateInterval(_PARKED_UPDATE_INTERVAL_MS)
        try:
            # A redraw already running when the hold goes on is NOT the one we
            # are waiting for, and its completion must not lift the hold: arm
            # on the next redraw to start, which is the one carrying the new
            # layer.
            canvas.renderStarting.connect(self._on_render_starting)
            canvas.mapCanvasRefreshed.connect(self._on_map_refreshed)
            canvas.extentsChanged.connect(self.release)
            self._timer = QTimer(canvas)
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self.release)
            self._timer.start(hold_timeout_ms())
        except BaseException:
            # The interval is already parked, and the caller drops the hold
            # when this constructor raises, so nothing else would ever give it
            # back: the canvas would stop repainting for the rest of the
            # session, plugin reload included.
            self.release()
            raise

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
        timer, self._timer = self._timer, None
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):  # nosec B110 - timer died with its parent canvas
                pass
        # The signal lookup sits INSIDE the guard: reading canvas.renderStarting
        # on a canvas whose C++ half is gone raises by itself, and that skipped
        # the interval restore below, which is the whole point of this method.
        for name, slot in (("renderStarting", self._on_render_starting),
                           ("mapCanvasRefreshed", self._on_map_refreshed),
                           ("extentsChanged", self.release)):
            try:
                getattr(canvas, name).disconnect(slot)
            except (TypeError, RuntimeError, AttributeError):  # nosec B110 - already gone
                pass
        try:
            _restore_interval(canvas, self.saved_interval)
        except (RuntimeError, AttributeError):  # nosec B110
            pass


class _LiveRunPictureHold:
    """The whole-run variant: parked until released, with a per-frame timeout
    instead of a single one. Frames are held one at a time: a frame that runs
    past the timeout gets the normal fill back, and the frame after it is
    held again."""

    def __init__(self, canvas) -> None:
        self._canvas = canvas
        self.saved_interval = _interval_before_any_hold(canvas)
        self._frame_timer = None
        canvas.setMapUpdateInterval(_PARKED_UPDATE_INTERVAL_MS)
        try:
            canvas.renderStarting.connect(self._on_render_starting)
            canvas.mapCanvasRefreshed.connect(self._on_map_refreshed)
            self._frame_timer = QTimer(canvas)
            self._frame_timer.setSingleShot(True)
            self._frame_timer.timeout.connect(self._on_frame_timeout)
            if canvas.isDrawing():
                self._frame_timer.start(live_hold_timeout_ms())
        except BaseException:
            # Same reason as the commit hold: the interval is parked already,
            # and nothing else would ever give it back.
            self.release()
            raise

    def is_live(self) -> bool:
        if self._canvas is None:
            return False
        try:
            self._canvas.mapUpdateInterval()
        except (RuntimeError, AttributeError):
            return False
        return True

    def _on_render_starting(self) -> None:
        # Every frame starts held. A frame that follows one that timed out
        # re-parks here, so one slow redraw costs one fill, not the run.
        try:
            self._canvas.setMapUpdateInterval(_PARKED_UPDATE_INTERVAL_MS)
            self._frame_timer.start(live_hold_timeout_ms())
        except (RuntimeError, AttributeError):  # nosec B110 - canvas or timer gone
            pass

    def _on_frame_timeout(self) -> None:
        # This frame is now a frozen map, not a handover: let it fill in.
        try:
            if self._canvas.isDrawing():
                self._canvas.setMapUpdateInterval(self.saved_interval)
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def _on_map_refreshed(self) -> None:
        try:
            self._frame_timer.stop()
            self._canvas.setMapUpdateInterval(_PARKED_UPDATE_INTERVAL_MS)
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def release(self) -> None:
        canvas, self._canvas = self._canvas, None
        if canvas is None:
            return
        _run_holds.pop(id(canvas), None)
        timer, self._frame_timer = self._frame_timer, None
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):  # nosec B110 - timer died with its parent canvas
                pass
        for name, slot in (("renderStarting", self._on_render_starting),
                           ("mapCanvasRefreshed", self._on_map_refreshed)):
            try:
                getattr(canvas, name).disconnect(slot)
            except (TypeError, RuntimeError, AttributeError):  # nosec B110 - already gone
                pass
        try:
            _restore_interval(canvas, self.saved_interval)
        except (RuntimeError, AttributeError):  # nosec B110
            pass
