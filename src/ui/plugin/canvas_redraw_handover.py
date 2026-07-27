

























from __future__ import annotations

from qgis.PyQt.QtCore import QTimer



_PARKED_UPDATE_INTERVAL_MS = 3_600_000










HOLD_TIMEOUT_MS = 2_000



_holds: dict[int, _CanvasPictureHold] = {}





_run_holds: dict[int, _LiveRunPictureHold] = {}




LIVE_HOLD_TIMEOUT_MS = 3 * HOLD_TIMEOUT_MS


def hold_timeout_ms() -> int:







    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("ui.canvas_hold_timeout_ms", HOLD_TIMEOUT_MS, 500, 60_000))
    except Exception:  # noqa: BLE001  # nosec B110
        return HOLD_TIMEOUT_MS


def live_hold_timeout_ms() -> int:




    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("ui.live_canvas_hold_timeout_ms", LIVE_HOLD_TIMEOUT_MS, 500, 60_000))
    except Exception:  # noqa: BLE001  # nosec B110
        return LIVE_HOLD_TIMEOUT_MS


def _live_hold_for(registry: dict, canvas):


    hold = registry.get(id(canvas))
    if hold is None:
        return None
    if hold.is_live():
        return hold
    registry.pop(id(canvas), None)
    return None


def _interval_before_any_hold(canvas) -> int:



    for registry in (_holds, _run_holds):
        other = _live_hold_for(registry, canvas)
        if other is not None:
            return other.saved_interval
    return canvas.mapUpdateInterval()


def _restore_interval(canvas, saved: int) -> None:


    for registry in (_holds, _run_holds):
        if _live_hold_for(registry, canvas) is not None:
            canvas.setMapUpdateInterval(_PARKED_UPDATE_INTERVAL_MS)
            return
    canvas.setMapUpdateInterval(saved)


def hold_map_picture_for_live_run(canvas) -> None:




    if canvas is None:
        return
    if _live_hold_for(_run_holds, canvas) is not None:
        return
    try:
        _run_holds[id(canvas)] = _LiveRunPictureHold(canvas)
    except (RuntimeError, AttributeError, TypeError):
        _run_holds.pop(id(canvas), None)


def release_live_run_picture_hold(canvas) -> None:

    hold = _live_hold_for(_run_holds, canvas) if canvas is not None else None
    if hold is not None:
        hold.release()


def hold_map_picture_during_redraw(canvas) -> None:





    if canvas is None:
        return
    existing = _holds.get(id(canvas))
    if existing is not None:
        if existing.is_live():
            return


        _holds.pop(id(canvas), None)
    try:
        _holds[id(canvas)] = _CanvasPictureHold(canvas)
    except (RuntimeError, AttributeError, TypeError):
        _holds.pop(id(canvas), None)


def release_map_picture_hold(canvas) -> None:


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




            canvas.renderStarting.connect(self._on_render_starting)
            canvas.mapCanvasRefreshed.connect(self._on_map_refreshed)
            canvas.extentsChanged.connect(self.release)
            self._timer = QTimer(canvas)
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self.release)
            self._timer.start(hold_timeout_ms())
        except BaseException:




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
        self._armed = True

    def _on_map_refreshed(self) -> None:


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
            except (RuntimeError, AttributeError):  # nosec B110
                pass



        for name, slot in (("renderStarting", self._on_render_starting),
                           ("mapCanvasRefreshed", self._on_map_refreshed),
                           ("extentsChanged", self.release)):
            try:
                getattr(canvas, name).disconnect(slot)
            except (TypeError, RuntimeError, AttributeError):  # nosec B110
                pass
        try:
            _restore_interval(canvas, self.saved_interval)
        except (RuntimeError, AttributeError):  # nosec B110
            pass


class _LiveRunPictureHold:





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


        try:
            self._canvas.setMapUpdateInterval(_PARKED_UPDATE_INTERVAL_MS)
            self._frame_timer.start(live_hold_timeout_ms())
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def _on_frame_timeout(self) -> None:

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
            except (RuntimeError, AttributeError):  # nosec B110
                pass
        for name, slot in (("renderStarting", self._on_render_starting),
                           ("mapCanvasRefreshed", self._on_map_refreshed)):
            try:
                getattr(canvas, name).disconnect(slot)
            except (TypeError, RuntimeError, AttributeError):  # nosec B110
                pass
        try:
            _restore_interval(canvas, self.saved_interval)
        except (RuntimeError, AttributeError):  # nosec B110
            pass
