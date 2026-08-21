"""The outline that lights up under the cursor, before the click.

Semi-Auto with the cloud engine only. The whole feature is one loop: the cursor
rests, the crop under it is named to the service, one preview comes back, and
its shape is painted over the ground it describes. Click and the ordinary path
takes over, unchanged, from the first line.

What is painted is the FINAL shape that click would save, and nothing else:
the answer's mask runs the same refine tail as a click (through
_manual_outline_for), under whatever refine settings are live at answer time.
An answer that chain cannot turn into an outline draws nothing at all. A ghost
of the raw mask promises a shape the click does not return, and a promise the
click breaks is worse than no promise.

The loop also keeps the last few answers for the crop it is working on. Sweep
back onto an object the session already asked about and its shape comes
straight back, drawn through that same chain, with nothing sent and nothing
waited for. The memory belongs to one crop: new ground, new memory.

Everything the loop needs lives on the controller here, so the click path keeps
its own state and gains four hooks and nothing else. That split is the point:
two sessions editing one click file collide on every change, and a preview has
no business inside a click.

Four rules it never breaks:

- It never pays. No credit, no charge, no note that the network answered. A
  preview is not an object and never becomes one.
- It never writes. No point, no seed, no history, no mask on the session. The
  click that follows behaves exactly as it does with previews switched off, and
  it asks the service again for the answer it keeps.
- It never speaks. Every failure draws nothing at all.
- It never waits. The reply lands in a callback, so the map keeps drawing, and
  an answer nobody is holding a place for any more is dropped unread.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); the mixin at the
bottom is the wiring and holds no state of its own.
"""
from __future__ import annotations

import re
import time

from qgis.core import QgsGeometry, QgsPointXY, QgsRectangle
from qgis.PyQt.QtCore import QEvent, QObject, QTimer

from ..canvas_palette import HOVER_PREVIEW_OUTLINE_WIDTH

# How many times the loop may come back to a crop that is not ready yet before
# it gives up and waits for the next real move. The crop is read and handed
# over off the GUI thread, so the wait is real, and a cursor resting on ground
# the session has never prepared must not poll for ever. Read with the tick
# below: the two together are the ten seconds a preview waits for its pixels.
_WAIT_TICKS_MAX = 100

# How long between those attempts. Nothing is moving, so this keeps up with
# nothing; it is how late an answer can be once the crop IS ready, and every
# millisecond of it is dead time the cursor spends on a shape it could have.
_WAIT_TICK_MS = 100

# How far, in crop pixels, the cursor may sit from the pixel a preview was
# asked about and still be read as resting on the same spot. An answer whose
# mask misses its own prompt point is drawn inside this drift and dropped
# outside it, so a shape never lands behind a cursor that moved on.
_STILL_NEAR_PX = 32

# How many answered shapes the loop keeps for the crop it is on, so a cursor
# coming back to an object it already asked about is served from hand instead
# of from the wire. Small on purpose: a hand sweeping the map goes back and
# forth over the few objects in view, never over all of them, and every answer
# held holds a crop-sized mask with it. A newer answer pushes out the oldest.
_RECENT_ANSWERS_KEPT = 6

# The share of the crop under which a remembered mask counts as one object,
# and any point inside it is the same question. On a car, a house or a pool
# every point names the same thing, so the answer already in hand is the right
# one. A mask wider than this is a stretch of ground rather than an object, a
# road or a field, and its two ends are genuinely different questions the
# service would answer differently. Above the line, covering the point stops
# being enough on its own: the cursor has to be resting near the pixel that
# answer was asked about too.
_RECALL_ONE_OBJECT_COVERAGE = 0.1

# The most one answer may spend, on the GUI thread, turning its mask into the
# final click shape (vectorize plus the refine tail). A stall guard, not a
# normal-case bar: the chain runs in single-digit milliseconds on ordinary
# ground, so nothing healthy comes near this, and an answer that does bust it
# still shows the shape it paid for. What the guard buys is that the SAME mask
# is never shaped a second time, which is what dragging a refine slider over a
# pathological ghost would ask for. Every new answer starts clear, so one slow
# mask never condemns the ones after it.
_SHAPE_BUDGET_MS = 500.0

# How many shaping costs the controller remembers, for reading off a live
# session. Diagnostics only: nothing in the loop reads past the last one.
_SHAPE_COSTS_KEPT = 50

# The share of the crop a mask may cover and still be worth shaping. Above it
# the answer is not an object anybody is hovering, it is most of the scene, and
# vectorizing it busts the budget every time. Nothing is drawn instead, so one
# pathological answer neither stalls the cursor nor promises an outline the
# click would not return. A stall guard, not a quality rule: the number is
# deliberately generous.
_SHAPE_MAX_COVERAGE = 0.6

# How long the route half of the gates (the served switch and the account
# store) may be answered from the last look, and how long one auth header is
# reused. Short enough that signing out stops previews in the same breath,
# long enough that a hand sweeping the map does not read the store per move.
_ROUTE_MEMO_MS = 3000.0

# The most failure classes one session ever writes a line about. The code comes
# off the wire, so the set it feeds has to have a ceiling.
_REPORTED_MAX = 20

# The longest a failure code may be once it is cleaned up for the log.
_CODE_CHARS_MAX = 32

# Everything a failure code may be made of. The rest is dropped: the code is
# server-controlled text on its way into a log line and into a set.
_CODE_ALLOWED = re.compile(r"[^A-Z0-9_]")


class HoverPreviewLeaveWatch(QObject):
    """Tells the controller when the pointer leaves the map.

    Installed on the canvas viewport, never on the application: an application
    filter sees every event in QGIS and this one needs exactly two.
    """

    def __init__(self, controller) -> None:
        super().__init__()
        self._controller = controller

    def eventFilter(self, _obj, event):  # noqa: N802 (Qt API)
        try:
            if event.type() in (QEvent.Type.Leave, QEvent.Type.FocusOut):
                self._controller.stop("cursor left the map")
        except Exception:  # noqa: BLE001 -- a filter must never abort an event  # nosec B110
            pass
        return False


class HoverPreviewController:
    """Owns every piece of preview state: the wait, the request and the shape."""

    def __init__(self, plugin) -> None:
        self._plugin = plugin
        self._overlay = None
        self._timer: QTimer | None = None
        self._leave_watch: HoverPreviewLeaveWatch | None = None
        self._canvas = None
        self._call = None
        # Which preview the controller is holding a place for. Every request
        # captures it and every answer checks it, so a newer hover drops the
        # older answer rather than drawing it late.
        self._serial = 0
        self._pending_point: QgsPointXY | None = None
        self._wait_ticks = 0
        # The mask on the map right now, with the crop window it was asked
        # about: (bounds, shape, mask array, the shaped geometry in the raster
        # CRS, which is the thing actually drawn). Read for two things: a
        # cursor resting INSIDE the drawn shape asks nothing (the answer would
        # be the same object), and it goes down whenever the picture does.
        self._shown: tuple | None = None
        # What a click landing on the drawn shape would need to stand on it
        # rather than ask the same question again: (bounds, shape, the pixel
        # that was asked, mask, score, logits). Beside _shown and not inside
        # it, because the picture goes down on events that leave the answer
        # perfectly good, the click itself first of all.
        self._answer: tuple | None = None
        # The last few answers for ONE crop: the key they belong to, then
        # (asked pixel, mask, score, logits, spans the crop) each, oldest
        # first. Only answers that actually reached the map go in, so serving
        # one back can never promise a shape the map never showed.
        self._recent_answers: list = []
        self._recent_answers_key: tuple | None = None
        # The one mask whose shaping busted its budget, held by identity so
        # that same mask is not put through the chain again while it is on the
        # map. Every answer brings its own mask and gets its own attempt.
        self._slow_mask = None
        # One shaped answer, kept by mask identity and by every setting it was
        # shaped under: (mask, served snapshot, settings, outline). The click
        # session's own memo only holds the ACTIVE mask, so a hover answer and
        # a refine tick would otherwise pay the whole chain every time.
        self._hover_mask_memo: tuple | None = None
        # The address a preview goes to, read once on attach.
        self._url: str | None = None
        # The headers one preview travels with, and when they were read.
        self._auth: dict | None = None
        self._auth_at = 0.0
        # The last shaping costs in ms, newest last. Diagnostics only.
        self._shape_costs: list = []
        # One line per failure class, so a service that refuses every preview
        # costs one line and not one per mouse move.
        self._reported: set[str] = set()
        self._torn_down = False

    # -- lifecycle ---------------------------------------------------------

    def attach(self) -> None:
        """Arm the loop. Idempotent, so a second session costs nothing."""
        if self._torn_down or self._timer is not None:
            return
        try:
            canvas = self._plugin.iface.mapCanvas()
        except Exception:  # noqa: BLE001 -- no canvas, no previews
            return
        if canvas is None:
            return
        self._canvas = canvas
        # Read once: it is the click route's own address, and resolving it
        # builds a client, which is not work a mouse move should pay for.
        try:
            from ...core.hover_preview_client import preview_refine_url

            self._url = preview_refine_url()
        except Exception:  # noqa: BLE001 -- no address, no preview
            self._url = None
        # Parented to the canvas: the plugin controller is not a QObject, and a
        # timer with no parent outlives the session that made it.
        self._timer = QTimer(canvas)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timer)
        try:
            canvas.extentsChanged.connect(self._on_extents_changed)
        except Exception:  # noqa: BLE001 -- a preview simply stops tracking  # nosec B110
            pass
        try:
            # A CRS swap redraws every pixel the picture was stretched over.
            canvas.destinationCrsChanged.connect(self._on_extents_changed)
        except Exception:  # noqa: BLE001 -- a preview simply stops tracking  # nosec B110
            pass
        try:
            viewport = canvas.viewport()
            if viewport is not None:
                self._leave_watch = HoverPreviewLeaveWatch(self)
                viewport.installEventFilter(self._leave_watch)
        except Exception:  # noqa: BLE001 -- without it the shape lives one move longer
            self._leave_watch = None

    def detach(self) -> None:
        """Give everything back. Called from unload and from a mode leaving."""
        self._torn_down = True
        self.stop("torn down")
        timer = self._timer
        self._timer = None
        if timer is not None:
            try:
                timer.stop()
                timer.timeout.disconnect(self._on_timer)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            try:
                # Parented to the canvas, so without this it outlives the
                # session that made it for as long as the canvas lives.
                timer.deleteLater()
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        canvas = self._canvas
        if canvas is not None:
            try:
                canvas.extentsChanged.disconnect(self._on_extents_changed)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            try:
                canvas.destinationCrsChanged.disconnect(self._on_extents_changed)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            if self._leave_watch is not None:
                try:
                    viewport = canvas.viewport()
                    if viewport is not None:
                        viewport.removeEventFilter(self._leave_watch)
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
        self._leave_watch = None
        overlay = self._overlay
        self._overlay = None
        if overlay is not None:
            try:
                overlay.clear_preview()
                scene = canvas.scene() if canvas is not None else None
                if scene is not None:
                    scene.removeItem(overlay)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        self._canvas = None
        self._plugin = None

    def stop(self, reason: str = "") -> None:
        """Take the shape off the map and drop anything in flight.

        The one way out, used by every clearing event: a click, a mode change,
        a pan, a tool swap, a failure. Never raises.
        """
        self._serial += 1
        self._pending_point = None
        self._wait_ticks = 0
        self._shown = None
        self._answer = None
        self._recent_answers = []
        self._recent_answers_key = None
        self._slow_mask = None
        self._hover_mask_memo = None
        try:
            if self._timer is not None:
                self._timer.stop()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        call = self._call
        self._call = None
        if call is not None:
            try:
                call.abandon()
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        overlay = self._overlay
        if overlay is not None:
            try:
                overlay.clear_preview()
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        if (reason and reason not in self._reported
                and len(self._reported) < _REPORTED_MAX):
            self._reported.add(reason)

    # -- the loop ----------------------------------------------------------

    def on_cursor_moved(self, point) -> None:
        """One plain move of the cursor over the map, in the canvas own CRS."""
        if self._torn_down or self._timer is None:
            return
        # The cheap half only. The route half reads the served switch and the
        # account store, and this runs on every single move of the mouse.
        if not self._gates_open():
            # Anything the loop still holds (a shape on the map, a request on
            # the wire) describes a state that just ended.
            if self._call is not None or (
                    self._overlay is not None and self._overlay.has_preview()):
                self.stop("gate closed")
            return
        here = QgsPointXY(point)
        # A cursor that jiggles on the same spot is still resting there, so the
        # cap in _wait_again has to survive it: reset only on a real move, or a
        # hand sitting on ground the session never prepares polls for ever.
        if not self._near_pending_point(here):
            self._wait_ticks = 0
        self._pending_point = here
        try:
            from ...core.hover_preview_client import hover_preview_debounce_ms

            self._timer.start(hover_preview_debounce_ms())
        except Exception:  # noqa: BLE001 -- an unreadable dial is no preview  # nosec B110
            pass

    def _near_pending_point(self, point) -> bool:
        """Whether ``point`` rests on the same spot as the one already pending.

        Judged in canvas pixels, through the same drift an answer is allowed,
        so the two agree on what "still here" means at any zoom.
        """
        previous = self._pending_point
        if previous is None or self._canvas is None:
            return False
        try:
            units = float(self._canvas.mapSettings().mapUnitsPerPixel())
            if units <= 0:
                return False
            span = _STILL_NEAR_PX * units
            return (abs(point.x() - previous.x()) <= span
                    and abs(point.y() - previous.y()) <= span)
        except Exception:  # noqa: BLE001 -- an unreadable canvas rests nowhere
            return False

    def _on_extents_changed(self) -> None:
        """A pan or a zoom: the picture describes ground that is being redrawn."""
        if self._torn_down:
            return
        self.stop("the map moved")

    def _on_timer(self) -> None:
        """The cursor has rested long enough, or a crop was being prepared."""
        if self._torn_down:
            return
        try:
            self._ask_for_preview()
        except Exception:  # noqa: BLE001 -- a preview never reaches the user
            self.stop("preview failed")

    def _ask_for_preview(self) -> None:
        point = self._pending_point
        if point is None or not self._gates_open():
            return
        # The expensive half, asked once per rest instead of once per move.
        if not self._route_gates_open():
            return
        plugin = self._plugin
        raster_pt = plugin._transform_to_raster_crs(point)
        if raster_pt is None or not plugin._is_point_in_raster_extent(raster_pt):
            self.stop("off the imagery")
            return
        if plugin._check_crop_status(raster_pt) != "ok":
            # Nothing to ask about yet. The warm-up lane rides the same cursor
            # signal and is already preparing this neighbourhood if its own
            # gates allow it (it stays speculative there, so a real click can
            # abandon it). This loop only ever waits for the result.
            self._wait_again()
            return
        if getattr(plugin, "_encoding_in_progress", False):
            # A worker owns the predictor pipe, so the crop it holds and the
            # window this session describes may be mid-swap. Asking now could
            # name one crop and place the answer on another's ground.
            self._wait_again()
            return
        handle = self._preview_handle()
        if handle is None:
            # The pixels are here and the service has not named them yet. That
            # hand-over runs off the GUI thread, so waiting is the answer.
            self._wait_again()
            return
        token, (crop_h, crop_w) = handle
        crop_info = getattr(plugin, "_current_crop_info", None)
        if not isinstance(crop_info, dict):
            return
        bounds = crop_info.get("bounds")
        shape = crop_info.get("img_shape")
        if bounds is None or shape is None:
            return
        if (int(shape[0]), int(shape[1])) != (int(crop_h), int(crop_w)):
            # The pixels the service holds are not the pixels this session is
            # working on. Asking would draw one crop's shape on another.
            return
        from ...core.crop_window import crop_pixel_of_point

        pixel = crop_pixel_of_point(tuple(bounds), (int(shape[0]), int(shape[1])),
                                    raster_pt.x(), raster_pt.y())
        if pixel is None:
            return
        if self._rests_on_shown_mask(tuple(bounds),
                                     (int(shape[0]), int(shape[1])), pixel,
                                     raster_pt):
            # The drawn shape already answers this spot: the service would hand
            # back the same object, so the shape holds still and nothing is
            # asked. This is what makes sweeping a large object feel free.
            return
        crop_key = (str(token), tuple(bounds),
                    (int(shape[0]), int(shape[1])))
        if self._serve_remembered_answer(crop_key, pixel):
            # This crop has already answered for this spot. The shape comes
            # back from hand: no request, no wait, no crop hand-over.
            return
        self._send(token, tuple(bounds), (int(shape[0]), int(shape[1])),
                   col=pixel[1], row=pixel[0])

    def _serve_remembered_answer(self, key: tuple, pixel) -> bool:
        """Draw a shape this crop already answered for this spot, or say no.

        The bet is the one the drawn shape already makes for the last answer,
        made once more for the few before it: a pixel resting inside a small
        answer's own mask would be answered by that same object again. Being
        near a remembered answer is not enough on its own, because the answer
        to one prompt pixel is not the answer to another. Nor is covering the
        point enough on its own once the mask spans the crop: see
        ``_remembered_answer_at``.

        Newest first, so where two remembered shapes cover the spot the one
        the user last saw there is the one that comes back.

        The shape goes up through the ordinary drawing path, so the size
        window and whatever refine settings are live right now apply exactly
        as they do to an answer off the wire, and the click can take it. Any
        doubt at all, and the service is asked as before.

        Answers whether a remembered shape reached the map.
        """
        try:
            found = self._remembered_answer_at(key, pixel)
            if found is None:
                return False
            asked, mask, score, logits, _wide = found
            bounds, shape = key[1], key[2]
            if not self._draw_answer_mask(mask, bounds, shape):
                return False
            self._answer = (bounds, shape, asked, mask, score, logits)
        except Exception:  # noqa: BLE001 -- an unreadable memory asks the service
            return False
        # Served from hand, so anything still on the wire describes a spot the
        # cursor has left: its answer is dropped unread rather than drawn over
        # the shape that is on the map now.
        self._serial += 1
        call = self._call
        self._call = None
        if call is not None:
            try:
                call.abandon()
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        self._wait_ticks = 0
        return True

    def _remembered_answer_at(self, key: tuple, pixel):
        """The newest answer held for ``key`` that still answers ``pixel``.

        Covering the pixel is the whole rule while the mask is one object.
        Once it spans the crop the cursor must also be resting near the pixel
        that answer was asked about, through the drift this file calls "still
        here" everywhere else, because the far end of a road is a different
        question from the near end and the service would answer it differently.

        None for another crop, for an empty memory, or for a spot no
        remembered answer still speaks for.
        """
        if not key or self._recent_answers_key != key:
            return None
        row, col = int(pixel[0]), int(pixel[1])
        for entry in reversed(self._recent_answers):
            if self._answer_speaks_for(entry, row, col):
                return entry
        return None

    @staticmethod
    def _answer_speaks_for(entry: tuple, row: int, col: int) -> bool:
        """Whether one remembered answer still answers this pixel.

        The two halves of the recall rule, and nothing else: the mask has to
        cover the pixel, and once the mask spans the crop the pixel has to be
        near the one that answer was asked about as well. Its own method so
        the loop above reads as the search it is, and so an unreadable entry
        is a no rather than a break in the walk.
        """
        try:
            asked, mask = entry[0], entry[1]
            if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]
                    and mask[row, col]):
                return False
            if not entry[4]:
                return True
            return (abs(row - int(asked[0])) <= _STILL_NEAR_PX
                    and abs(col - int(asked[1])) <= _STILL_NEAR_PX)
        except Exception:  # noqa: BLE001 -- an unreadable mask answers nothing
            return False

    def _remember_answer(self, key: tuple, asked: tuple, mask, score,
                         logits) -> None:
        """Keep one drawn answer under the crop it describes.

        A key that is not the one in hand empties the memory before anything
        is kept: an answer describes ground, and a new crop token, a new crop
        window or a new crop size all mean the ground moved.

        The share of the crop the mask covers is measured here, once, rather
        than on every rest: it decides which recall rule the answer lives
        under, and it never changes after the answer arrives.
        """
        try:
            if self._recent_answers_key != key:
                self._recent_answers_key = key
                self._recent_answers = []
            wide = self._mask_spans_crop(mask, key[2])
            self._recent_answers.append((asked, mask, score, logits, wide))
            del self._recent_answers[:-_RECENT_ANSWERS_KEPT]
        except Exception:  # noqa: BLE001 -- a memory that refuses holds nothing
            self._recent_answers = []
            self._recent_answers_key = None

    def _rests_on_shown_mask(self, bounds: tuple, shape: tuple, pixel,
                             raster_pt) -> bool:
        """Whether the shape on the map already answers the resting point.

        Judged on what is actually drawn, which is always the shaped polygon.
        It and the raw mask can disagree at the rim (Right angles moves it),
        and pinning to the mask would re-ask about ground the user sees
        covered.

        Covering the point is not enough on its own. A shape holds a whole
        building, and a click in the middle of it does not ask the service what
        a click at its edge asked: the answer to one prompt pixel is not the
        answer to another, so a ghost held right across an object promises a
        shape the click does not return. It is held over the same spot instead,
        through the drift this file calls "still here" everywhere else.
        """
        shown = self._shown
        overlay = self._overlay
        if shown is None or overlay is None or not overlay.has_preview():
            return False
        if shown[0] != bounds or shown[1] != shape:
            return False
        answer = self._answer
        if answer is None:
            return False
        asked = answer[2]
        if (abs(int(pixel[0]) - int(asked[0])) > _STILL_NEAR_PX
                or abs(int(pixel[1]) - int(asked[1])) > _STILL_NEAR_PX):
            return False
        shaped = shown[3] if len(shown) > 3 else None
        if shaped is None:
            return False
        try:
            return bool(shaped.contains(QgsPointXY(raster_pt)))
        except Exception:  # noqa: BLE001 -- an unreadable shape answers nothing
            return False

    def _wait_again(self) -> None:
        """Come back to a crop that is still being prepared, up to the cap."""
        self._wait_ticks += 1
        if self._wait_ticks > _WAIT_TICKS_MAX or self._timer is None:
            return
        try:
            self._timer.start(_WAIT_TICK_MS)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _auth_headers(self) -> dict | None:
        """The headers one preview travels with, re-read a few seconds apart.

        The read reaches the settings store and the auth database, which is
        more than a hover should pay for on every rest.
        """
        now = time.monotonic() * 1000.0
        if self._auth and now - self._auth_at < _ROUTE_MEMO_MS:
            return self._auth
        try:
            from ...core.activation_manager import get_auth_header

            auth = get_auth_header()
        except Exception:  # noqa: BLE001 -- no key, no preview
            return None
        if not auth:
            self._auth = None
            return None
        self._auth = auth
        self._auth_at = now
        return auth

    def _send(self, token: str, bounds: tuple, shape: tuple, col, row) -> None:
        from ...core.hover_preview_client import HoverPreviewCall, build_preview_body

        url = self._url
        if not url:
            return
        auth = self._auth_headers()
        if not auth:
            return
        # The crop is ready and the wait is over. Left standing, the tick count
        # from that wait was still there for the NEXT crop, which then gave up
        # early or at once.
        self._wait_ticks = 0
        # A newer hover supersedes the one in flight: its answer is dropped
        # unread rather than drawn behind the cursor.
        if self._call is not None:
            try:
                self._call.abandon()
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            self._call = None
        self._serial += 1
        serial = self._serial

        def _answer(answer: dict) -> None:
            self._on_answer(serial, bounds, shape, (int(row), int(col)),
                            answer, token)

        call = HoverPreviewCall(url, build_preview_body(token, col, row),
                                auth, _answer)
        if call.send():
            self._call = call

    def _on_answer(self, serial: int, bounds: tuple, shape: tuple,
                   asked: tuple, answer: dict, token: str | None = None) -> None:
        """Draw one answer, or draw nothing. Never speaks to the user."""
        if self._torn_down or serial != self._serial:
            return
        self._call = None
        if not isinstance(answer, dict) or answer.get("error") or answer.get("code"):
            self._note_failure(str((answer or {}).get("code") or "refused"))
            return
        # Both halves here: an answer is once per rest, so the route half costs
        # what it did before the split, and a session that signed out or lost
        # the feature while this was on the wire must still draw nothing.
        if not self._gates_open() or not self._route_gates_open():
            return
        try:
            from ...core.hover_preview_client import read_preview_answer

            read = read_preview_answer(answer, shape[0], shape[1])
            if read is None:
                return
            mask, score, logits = read
            if not self._cursor_still_on(bounds, shape, asked, mask):
                # The cursor travelled on while this answer was on the wire.
                # The shape it describes is behind the gesture now, so the map
                # keeps whatever it is showing and the newer rest answers.
                return
            # Kept only once the map actually shows it: an answer recorded
            # before a draw that fell through is a promise to the click that
            # nothing on screen backs.
            if self._draw_answer_mask(mask, bounds, shape):
                self._answer = (bounds, shape, asked, mask, score, logits)
                # Kept under the same test, for the same reason: a shape the
                # map showed once can be shown again for free.
                self._remember_answer((str(token or ""), bounds, shape),
                                      asked, mask, score, logits)
        except Exception:  # noqa: BLE001 -- a preview never reaches the user
            self._note_failure("draw")

    def _draw_answer_mask(self, mask, bounds: tuple, shape: tuple) -> bool:
        """Put one answered mask on the map as the FINAL click shape, or take
        the map back to nothing.

        There is one form and no other. An answer the click chain cannot turn
        into an outline (it falls outside the user's own Min/Max size window,
        the refine settings leave it empty, the mask is too wide to shape, the
        chain busted its budget) leaves the map clear and no answer behind it.
        Drawing the raw mask instead was the one thing the ghost could do that
        the click would not honour.

        Answers whether anything reached the map."""
        ground = self._ground_rectangle(bounds)
        if ground is None:
            return False
        overlay = self._ensure_overlay()
        if overlay is None:
            return False
        shaped = self._shaped_answer_geometry(mask, bounds, shape)
        if shaped is not None:
            on_canvas = QgsGeometry(shaped)
            self._plugin._transform_geometry_to_canvas_crs(on_canvas)
            if not on_canvas.isEmpty():
                # One update per answered preview, and never a scheduled
                # second one: a canvas item told faster than the compositor
                # draws stops showing the shape the cursor is on.
                overlay.show_preview_polygon(on_canvas, ground)
                self._shown = (bounds, shape, mask, shaped)
                return True
        overlay.clear_preview()
        self._shown = None
        self._answer = None
        return False

    def _shaped_answer_geometry(self, mask, bounds: tuple, shape: tuple):
        """The shape a click here would save, or None.

        The exact click chain, through _manual_outline_for: mask refinement,
        polygonize, the shared geometry tail with whatever refine settings are
        live right now, then the size window. Timed, because it runs on the
        GUI thread between two cursor moves: an answer that busts the budget
        keeps the shape it paid for, and that one mask is refused a second run
        rather than stalling the cursor again. The mask that arrives with the
        next answer is a different object and gets its own attempt.
        """
        if mask is self._slow_mask or self._mask_too_wide_to_shape(mask, shape):
            return None
        plugin = self._plugin
        try:
            info = {
                "bbox": (float(bounds[0]), float(bounds[2]),
                         float(bounds[1]), float(bounds[3])),
                "img_shape": (int(shape[0]), int(shape[1])),
                "crs": None,
            }
            try:
                layer = getattr(plugin, "_current_layer", None)
                if layer is not None and layer.crs().isValid():
                    info["crs"] = layer.crs().authid()
            except Exception:  # noqa: BLE001 -- the tail reads the layer, not this  # nosec B110
                pass
            config, settings = self._hover_shape_key(info)
            memo = self._hover_mask_memo
            if (memo is not None and memo[0] is mask and memo[1] is config
                    and memo[2] == settings):
                # The same mask under the same settings: a refine tick that
                # lands back on a value already shaped costs nothing.
                return memo[3]
            started = time.monotonic()
            outline = plugin._manual_outline_for(mask, info)
            cost_ms = (time.monotonic() - started) * 1000.0
            self._shape_costs.append(cost_ms)
            del self._shape_costs[:-_SHAPE_COSTS_KEPT]
            if cost_ms > _SHAPE_BUDGET_MS:
                self._slow_mask = mask
            if outline is None or outline.isEmpty():
                outline = None
            # Holding the mask itself keeps the identity check honest: a live
            # reference cannot have its address reused under the memo.
            self._hover_mask_memo = (mask, config, settings, outline)
            return outline
        except Exception:  # noqa: BLE001 -- an unshapeable answer draws nothing
            return None

    @staticmethod
    def _mask_too_wide_to_shape(mask, shape: tuple) -> bool:
        """Whether this answer covers too much of the crop to be worth shaping.

        One pass over the mask, against a full vectorize plus the refine tail.
        A stall guard only: an answer this wide is the scene, not the object
        under the cursor, and nothing is drawn for it.
        """
        try:
            total = int(shape[0]) * int(shape[1])
            if total <= 0:
                return False
            return int(mask.sum()) > _SHAPE_MAX_COVERAGE * total
        except Exception:  # noqa: BLE001 -- an unmeasurable mask is not too wide
            return False

    @staticmethod
    def _mask_spans_crop(mask, shape: tuple) -> bool:
        """Whether this mask is a stretch of ground rather than one object.

        One pass over the mask, paid once when the answer is kept. Unreadable
        counts as a stretch, which is the careful answer: the stricter recall
        rule then applies and a doubtful shape is never served far from where
        it was asked.
        """
        try:
            total = int(shape[0]) * int(shape[1])
            if total <= 0:
                return True
            return int(mask.sum()) > _RECALL_ONE_OBJECT_COVERAGE * total
        except Exception:  # noqa: BLE001 -- an unmeasurable mask gets the strict rule
            return True

    def _hover_shape_key(self, info: dict) -> tuple:
        """The served snapshot and every live setting the shaped ghost reads.

        Built from the click path's own key builders, so a control a click
        obeys can never be missing here and leave a stale ghost on the map.
        """
        from ...core.config_cache import get_config
        from ...core.detection_policy import manual_simplify_multiple_of_px

        plugin = self._plugin
        multiple = manual_simplify_multiple_of_px()
        tolerance = (multiple * plugin._crop_pixel_size_units(info)
                     if multiple > 0 else 0.0)
        fill_holes, max_hole_px = plugin._fill_holes_arguments(info)
        mask_stage = (plugin._refine_expand, fill_holes, plugin._refine_min_area,
                      max_hole_px, tolerance)
        return get_config(), plugin._manual_outline_key(mask_stage)

    def take_shown_answer(self):
        """The answer under the ghost, handed over once and forgotten here.

        ``(bounds, shape, asked pixel, mask, score, logits, drawn outline)``,
        or None. The outline is the shape the user was aiming at, in the raster
        CRS, and it travels WITH the answer because the click clears the ghost
        the moment it takes it: read later, there is nothing left to read.

        For the click that is replacing the ghost, and only where the service
        answers a preview through the click's own chain: while the two chains
        differ, the ghost is a promise the click improves on. Hands over a
        record, not a decision. Which crop it belongs to and whether the click
        landed on the shape are the caller's to check.
        """
        answer = self._answer
        shown = self._shown
        self._answer = None
        if answer is None or shown is None or self._torn_down:
            return None
        try:
            from ...core.hover_preview_client import hover_preview_reuse_offered

            if not hover_preview_reuse_offered():
                return None
        except Exception:  # noqa: BLE001 -- an unreadable switch is a closed one
            return None
        drawn = shown[3] if len(shown) > 3 else None
        if drawn is not None:
            # A copy: the caller must not be able to move the record the ghost
            # is still drawing from.
            try:
                drawn = QgsGeometry(drawn)
            except Exception:  # noqa: BLE001 -- an uncopyable shape is no shape
                drawn = None
        return tuple(answer) + (drawn,)

    def reshape_shown(self) -> None:
        """A refine control moved while a ghost is up: re-shape the mask still
        in hand, now and without a request. Never leaves the old styling: on
        any failure the ghost comes down rather than lying about the click."""
        if self._torn_down:
            return
        shown = self._shown
        overlay = self._overlay
        if shown is None or overlay is None or not overlay.has_preview():
            return
        try:
            self._draw_answer_mask(shown[2], shown[0], shown[1])
        except Exception:  # noqa: BLE001 -- a stale-styled ghost must not stay
            self.stop("reshape failed")

    def _cursor_still_on(self, bounds: tuple, shape: tuple, asked: tuple,
                         mask) -> bool:
        """Whether the cursor is still over the object this answer describes.

        Still inside the answered mask counts, and so does resting near the
        pixel that was asked, because a mask does not always cover its own
        prompt point.
        """
        point = self._pending_point
        if point is None:
            return False
        try:
            raster_pt = self._plugin._transform_to_raster_crs(point)
            if raster_pt is None:
                return False
            # crop_pixel_of_point clamps to the grid, so a cursor that left the
            # crop entirely would read as resting on its edge. Asked here.
            if not (bounds[0] <= raster_pt.x() <= bounds[2]
                    and bounds[1] <= raster_pt.y() <= bounds[3]):
                return False
            from ...core.crop_window import crop_pixel_of_point

            pixel = crop_pixel_of_point(bounds, shape,
                                        raster_pt.x(), raster_pt.y())
            if pixel is None:
                return False
            row, col = int(pixel[0]), int(pixel[1])
            if 0 <= row < shape[0] and 0 <= col < shape[1] and mask[row, col]:
                return True
            return (abs(row - asked[0]) <= _STILL_NEAR_PX
                    and abs(col - asked[1]) <= _STILL_NEAR_PX)
        except Exception:  # noqa: BLE001 -- an unplaceable cursor left the object
            return False

    def _note_failure(self, code: str) -> None:
        """One line per failure class, at most. Silent to the user.

        The code comes off the wire, and it lands in a log line and in a set
        that lives as long as the session. So it is cut down to the shape a
        code has, cut short, and the set has a ceiling.
        """
        from ...core.hover_preview_client import PREVIEW_BUSY_CODE, log_preview_note

        named = _CODE_ALLOWED.sub("", (code or "").strip().upper())[:_CODE_CHARS_MAX]
        if not named:
            named = "REFUSED"
        if named == PREVIEW_BUSY_CODE or named in self._reported:
            return
        if len(self._reported) >= _REPORTED_MAX:
            return
        self._reported.add(named)
        log_preview_note(f"Hover preview: nothing drawn ({named})")

    # -- pieces ------------------------------------------------------------

    def _ensure_overlay(self):
        if self._overlay is not None:
            return self._overlay
        try:
            from ..hover_preview_overlay import HoverPreviewOverlay

            self._overlay = HoverPreviewOverlay(self._canvas)
        except Exception:  # noqa: BLE001 -- no item, no preview
            self._overlay = None
        return self._overlay

    def _ground_rectangle(self, bounds: tuple) -> QgsRectangle | None:
        """The crop window in the CRS the canvas draws in, or None.

        The whole box is projected, not two of its corners: on a curved
        projection the extreme sits along an edge rather than at a corner, and
        the sliver that is missed is where the ghost gets clipped. Then grown
        by the rim's own width, because the dashed pen is cosmetic and strokes
        outside the shape, while the item never draws past this rectangle.
        """
        try:
            plugin = self._plugin
            rect = QgsRectangle(float(bounds[0]), float(bounds[1]),
                                float(bounds[2]), float(bounds[3]))
            transform = getattr(plugin, "_raster_to_canvas_xform", None)
            if transform is not None:
                rect = transform.transformBoundingBox(rect)
            rect.normalize()
            if rect.isEmpty():
                return None
            slack = self._pen_slack_units()
            if slack > 0:
                rect.grow(slack)
            return rect
        except Exception:  # noqa: BLE001 -- an unplaceable window is no preview
            return None

    def _pen_slack_units(self) -> float:
        """Map units the dashed rim needs outside the crop window.

        The pen is cosmetic, so its width is a count of screen pixels whatever
        the zoom, and half of it falls outside the shape it strokes.
        """
        try:
            units = float(self._canvas.mapSettings().mapUnitsPerPixel())
            return max(0.0, units * HOVER_PREVIEW_OUTLINE_WIDTH)
        except Exception:  # noqa: BLE001 -- an unreadable canvas gets no slack
            return 0.0

    def _preview_handle(self):
        """The service's name for the crop in hand, plus its size, or None."""
        predictor = getattr(self._plugin, "predictor", None)
        getter = getattr(predictor, "hover_preview_handle", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:  # noqa: BLE001 -- an unreadable predictor names nothing
            return None

    def _gates_open(self) -> bool:
        """The conditions a mouse move can afford to check, on every move.

        Attribute reads and one isinstance, nothing that reaches a settings
        store, a database or the served configuration. The half that does is
        ``_route_gates_open``, asked once per rest.

        Fail-closed throughout. A preview sends imagery, so anything unreadable
        here means no preview rather than a guess.
        """
        plugin = self._plugin
        if plugin is None or self._torn_down:
            return False
        try:
            if getattr(plugin, "_headless", False):
                return False
            from ..ai_segmentation_dockwidget import Mode

            dock = plugin.dock_widget
            if dock is None or getattr(dock, "_mode", None) != Mode.INTERACTIVE:
                return False
            # A panel that is closed or tabbed away cannot show what the shape
            # under the cursor is, and a preview sends imagery to draw it.
            if not dock.isVisible():
                return False
            canvas = plugin.iface.mapCanvas()
            if canvas is None or canvas.mapTool() is not plugin.map_tool:
                return False
            tool = plugin.map_tool
            if tool is None or not tool.isActive() or tool.is_space_panning():
                return False
            # The review's fix session has its own click loop and its own
            # imagery. Previews belong to base Semi-Auto only.
            if getattr(plugin, "_refine_handoff_active", False):
                return False
            if getattr(plugin, "_auto_review", None) is not None:
                return False
            # A click session in progress owns the crop: previews stop until
            # the object is saved or cleared, so a preview can never fight the
            # refinement the user is steering.
            if getattr(plugin, "current_mask", None) is not None:
                return False
            if getattr(plugin, "_active_crop_points_positive", None):
                return False
            if getattr(plugin, "_active_crop_points_negative", None):
                return False
            if getattr(plugin, "_frozen_sessions", None):
                return False
            if getattr(plugin, "_unfrozen_display_polygon", None) is not None:
                return False
            # The predictor in hand is this mode's remote route. One isinstance,
            # so it stays on the cheap side.
            return bool(plugin._manual_cloud_predictor_active())
        except Exception:  # noqa: BLE001 -- an unreadable gate is a closed one
            return False

    def _route_gates_open(self) -> bool:
        """The engine card is on Cloud AI, the option is still offered, and the
        account is one the service will accept.

        The expensive half: the served switch and the account store. Answered
        through the plugin's own memo, so one rest costs one look and a hand
        sweeping the map costs none.
        """
        plugin = self._plugin
        if plugin is None or self._torn_down:
            return False
        try:
            return bool(plugin._hover_preview_could_run())
        except Exception:  # noqa: BLE001 -- an unreadable gate is a closed one
            return False


class ManualHoverPreviewMixin:
    """The four hooks the click path hands the preview loop."""

    def _hover_preview_controller(self):
        """The controller, built on first use. None once the plugin unloaded."""
        controller = getattr(self, "_hover_preview", None)
        if controller is None:
            controller = HoverPreviewController(self)
            controller.attach()
            self._hover_preview = controller
        return controller

    def _on_hover_cursor_moved(self, point) -> None:
        """Cursor motion over the map. Cheap: the gates answer before anything."""
        try:
            controller = getattr(self, "_hover_preview", None)
            if controller is None:
                # Built only where previews could run at all, so a session on
                # the on-device engine never makes one.
                if not self._hover_preview_could_run():
                    return
                controller = self._hover_preview_controller()
            controller.on_cursor_moved(point)
        except Exception:  # noqa: BLE001 -- a preview never breaks the map tool  # nosec B110
            pass

    def _hover_preview_could_run(self) -> bool:
        """Whether the route behind previews is open at all.

        The served switch plus the account store, so it reaches settings and
        the auth database. Held for a few seconds: a cursor crossing the map
        asks it hundreds of times, and signing out or a withdrawn feature
        still stops previews within that window.
        """
        if getattr(self, "_headless", False):
            return False
        now = time.monotonic() * 1000.0
        memo = getattr(self, "_hover_route_memo", None)
        if memo is not None and now - memo[0] < _ROUTE_MEMO_MS:
            return memo[1]
        try:
            from ...core.hover_preview_client import hover_preview_offered

            answer = bool(hover_preview_offered()
                          and self._manual_cloud_route_ready())
        except Exception:  # noqa: BLE001 -- an unreadable gate is a closed one
            answer = False
        self._hover_route_memo = (now, answer)
        return answer

    def _take_hover_preview_answer(self):
        """The answer under the ghost a click is about to replace, or None.

        Taken by the click BEFORE it clears the ghost, because clearing drops
        it. What comes back is checked against the crop and the click pixel in
        ``_run_prediction``, which is the one place that knows both.
        """
        controller = getattr(self, "_hover_preview", None)
        if controller is None:
            return None
        try:
            return controller.take_shown_answer()
        except Exception:  # noqa: BLE001 -- a preview never breaks a click
            return None

    def _reshape_hover_preview(self) -> None:
        """A refine setting moved: the ghost re-shapes the mask it already
        holds, immediately and offline. No-op without a visible ghost."""
        controller = getattr(self, "_hover_preview", None)
        if controller is None:
            return
        try:
            controller.reshape_shown()
        except Exception:  # noqa: BLE001 -- a preview never breaks the panel  # nosec B110
            pass

    def _stop_hover_preview(self, reason: str = "") -> None:
        """Take any preview off the map now. Safe to call at any time."""
        controller = getattr(self, "_hover_preview", None)
        if controller is None:
            return
        try:
            controller.stop(reason)
        except Exception:  # noqa: BLE001 -- clearing must never raise  # nosec B110
            pass

    def _teardown_hover_preview(self) -> None:
        """Unload: drop the timer, the filter, the item and anything in flight."""
        controller = getattr(self, "_hover_preview", None)
        self._hover_preview = None
        self._hover_route_memo = None
        if controller is None:
            return
        try:
            controller.detach()
        except Exception:  # noqa: BLE001 -- teardown must never raise  # nosec B110
            pass
