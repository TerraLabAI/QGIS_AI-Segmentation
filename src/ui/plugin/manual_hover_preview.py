





































from __future__ import annotations

import re
import time

from qgis.core import QgsGeometry, QgsPointXY, QgsRectangle
from qgis.PyQt.QtCore import QEvent, QObject, QTimer

from ...core.interaction_dials import (
    hover_recall_one_object_coverage,
    hover_refusal_quiet_s,
    hover_shape_budget_ms,
    hover_shape_max_coverage,
    hover_transient_quiet_s,
    route_memo_ms,
)
from ..canvas_palette import HOVER_PREVIEW_OUTLINE_WIDTH







_WAIT_TICKS_MAX = 16




_WAIT_TICK_MS = 100




_WAIT_TICKS_FAST = 10
_WAIT_TICK_SLOW_MS = 250





_STILL_NEAR_PX = 32






_RECENT_ANSWERS_KEPT = 6









_RECALL_ONE_OBJECT_COVERAGE = 0.1









_SHAPE_BUDGET_MS = 500.0





_REFUSAL_QUIET_S = 20.0









_TRANSIENT_QUIET_S = 2.5




_TRANSIENT_CODES = frozenset({
    "PREVIEW_BUSY", "TIMEOUT", "TIMED_OUT", "NETWORK", "CONNECTION",
    "UNAVAILABLE", "COLD", "STARTING", "WARMING", "OVERLOADED",
    "TOO_MANY_REQUESTS", "RETRY", "REFUSED",
})







_SHAPE_MAX_COVERAGE = 0.6





_ROUTE_MEMO_MS = 3000.0



_REPORTED_MAX = 20


def _reported_max() -> int:
    try:
        from ...core.server_dials import dial_in_range
        return int(dial_in_range("tuning.hover.reported_reasons_max", _REPORTED_MAX, 5, 50))
    except Exception:  # noqa: BLE001
        return _REPORTED_MAX



_CODE_CHARS_MAX = 32



_CODE_ALLOWED = re.compile(r"[^A-Z0-9_]")


class HoverPreviewLeaveWatch(QObject):






    def __init__(self, controller) -> None:
        super().__init__()
        self._controller = controller

    def eventFilter(self, _obj, event):  # noqa: N802
        try:
            if event.type() in (QEvent.Type.Leave, QEvent.Type.FocusOut):
                self._controller.stop("cursor left the map")
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return False


class HoverPreviewController:


    def __init__(self, plugin) -> None:
        self._plugin = plugin
        self._overlay = None
        self._timer: QTimer | None = None
        self._leave_watch: HoverPreviewLeaveWatch | None = None
        self._canvas = None
        self._call = None



        self._serial = 0
        self._pending_point: QgsPointXY | None = None
        self._wait_ticks = 0





        self._shown: tuple | None = None





        self._answer: tuple | None = None




        self._recent_answers: list = []
        self._recent_answers_key: tuple | None = None



        self._slow_mask = None




        self._hover_mask_memo: tuple | None = None

        self._url: str | None = None
        self._url_at = 0.0

        self._debounce_ms = 0
        self._debounce_at = 0.0


        self._interactive_mode = None

        self._quiet_until = 0.0

        self._auth: dict | None = None
        self._auth_at = 0.0


        self._reported: set[str] = set()
        self._torn_down = False



    def attach(self) -> None:

        if self._torn_down or self._timer is not None:
            return
        try:
            canvas = self._plugin.iface.mapCanvas()
        except Exception:  # noqa: BLE001
            return
        if canvas is None:
            return
        self._canvas = canvas

        self._quiet_until = 0.0
        try:
            from ..ai_segmentation_dockwidget import Mode

            self._interactive_mode = Mode.INTERACTIVE
        except Exception:  # noqa: BLE001
            self._interactive_mode = None


        self._timer = QTimer(canvas)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timer)
        try:
            canvas.extentsChanged.connect(self._on_extents_changed)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        try:

            canvas.destinationCrsChanged.connect(self._on_extents_changed)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        try:
            viewport = canvas.viewport()
            if viewport is not None:
                self._leave_watch = HoverPreviewLeaveWatch(self)
                viewport.installEventFilter(self._leave_watch)
        except Exception:  # noqa: BLE001
            self._leave_watch = None

    def detach(self) -> None:

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
                and len(self._reported) < _reported_max()):
            self._reported.add(reason)



    def on_cursor_moved(self, point) -> None:

        if self._torn_down or self._timer is None:
            return


        if not self._gates_open():


            if self._call is not None or (
                    self._overlay is not None and self._overlay.has_preview()):
                self.stop("gate closed")
            return
        here = QgsPointXY(point)



        if not self._near_pending_point(here):
            self._wait_ticks = 0
        self._pending_point = here
        debounce = self._debounce_ms_cached()
        if debounce <= 0:
            return
        try:
            self._timer.start(debounce)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _debounce_ms_cached(self) -> int:





        now = time.monotonic() * 1000.0
        if self._debounce_ms and now - self._debounce_at < route_memo_ms(_ROUTE_MEMO_MS):
            return self._debounce_ms
        try:
            from ...core.hover_preview_client import hover_preview_debounce_ms

            value = int(hover_preview_debounce_ms())
        except Exception:  # noqa: BLE001
            value = 0
        self._debounce_ms = value
        self._debounce_at = now
        return value

    def _near_pending_point(self, point) -> bool:





        previous = self._pending_point
        if previous is None or self._canvas is None:
            return False
        try:
            from ...core.server_dials import dial_in_range

            still_near_px = dial_in_range(
                "tuning.hover.still_near_px", _STILL_NEAR_PX, 4, 100)
            units = float(self._canvas.mapSettings().mapUnitsPerPixel())
            if units <= 0:
                return False
            span = still_near_px * units
            return (abs(point.x() - previous.x()) <= span
                    and abs(point.y() - previous.y()) <= span)
        except Exception:  # noqa: BLE001
            return False

    def _on_extents_changed(self) -> None:

        if self._torn_down:
            return
        self.stop("the map moved")

    def _on_timer(self) -> None:

        if self._torn_down:
            return
        try:
            self._ask_for_preview()
        except Exception:  # noqa: BLE001
            self.stop("preview failed")

    def _ask_for_preview(self) -> None:
        point = self._pending_point
        if point is None or not self._gates_open():
            return
        if time.monotonic() < self._quiet_until:


            return

        if not self._route_gates_open():
            return
        plugin = self._plugin
        raster_pt = plugin._transform_to_raster_crs(point)
        if raster_pt is None or not plugin._is_point_in_raster_extent(raster_pt):
            self.stop("off the imagery")
            return
        if plugin._check_crop_status(raster_pt) != "ok":




            self._wait_again()
            return
        if getattr(plugin, "_encoding_in_progress", False):



            self._wait_again()
            return
        handle = self._preview_handle()
        if handle is None:


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


            return
        from ...core.crop_window import crop_pixel_of_point

        pixel = crop_pixel_of_point(tuple(bounds), (int(shape[0]), int(shape[1])),
                                    raster_pt.x(), raster_pt.y())
        if pixel is None:
            return
        if self._rests_on_shown_mask(tuple(bounds),
                                     (int(shape[0]), int(shape[1])), pixel,
                                     raster_pt):



            return
        crop_key = (str(token), tuple(bounds),
                    (int(shape[0]), int(shape[1])))
        if self._serve_remembered_answer(crop_key, pixel):


            return
        self._send(token, tuple(bounds), (int(shape[0]), int(shape[1])),
                   col=pixel[1], row=pixel[0])

    def _serve_remembered_answer(self, key: tuple, pixel) -> bool:




















        try:
            found = self._remembered_answer_at(key, pixel)
            if found is None:
                return False
            asked, mask, score, logits, _wide = found[:5]
            if len(found) > 5 and found[5]:



                return True
            bounds, shape = key[1], key[2]
            if not self._draw_answer_mask(mask, bounds, shape):
                return False
            self._answer = (bounds, shape, asked, mask, score, logits)
        except Exception:  # noqa: BLE001
            return False



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











        if not key or self._recent_answers_key != key:
            return None
        row, col = int(pixel[0]), int(pixel[1])
        for entry in reversed(self._recent_answers):
            if self._answer_speaks_for(entry, row, col):
                return entry
        return None

    @staticmethod
    def _answer_speaks_for(entry: tuple, row: int, col: int) -> bool:








        try:
            from ...core.server_dials import dial_in_range

            still_near_px = dial_in_range(
                "tuning.hover.still_near_px", _STILL_NEAR_PX, 4, 100)
            asked, mask = entry[0], entry[1]
            if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]
                    and mask[row, col]):
                return False
            if not entry[4]:
                return True
            return (abs(row - int(asked[0])) <= still_near_px
                    and abs(col - int(asked[1])) <= still_near_px)
        except Exception:  # noqa: BLE001
            return False

    def _remember_answer(self, key: tuple, asked: tuple, mask, score,
                         logits, shaped_to_nothing: bool = False) -> None:










        try:
            from ...core.server_dials import dial_in_range

            kept = dial_in_range(
                "tuning.hover.recent_answers_kept", _RECENT_ANSWERS_KEPT, 1, 20)
            if self._recent_answers_key != key:
                self._recent_answers_key = key
                self._recent_answers = []
            wide = self._mask_spans_crop(mask, key[2])
            self._recent_answers.append(
                (asked, mask, score, logits, wide, bool(shaped_to_nothing)))
            del self._recent_answers[:-kept]
        except Exception:  # noqa: BLE001
            self._recent_answers = []
            self._recent_answers_key = None

    def _rests_on_shown_mask(self, bounds: tuple, shape: tuple, pixel,
                             raster_pt) -> bool:














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
        from ...core.server_dials import dial_in_range

        still_near_px = dial_in_range(
            "tuning.hover.still_near_px", _STILL_NEAR_PX, 4, 100)
        if (abs(int(pixel[0]) - int(asked[0])) > still_near_px
                or abs(int(pixel[1]) - int(asked[1])) > still_near_px):
            return False
        shaped = shown[3] if len(shown) > 3 else None
        if shaped is None:
            return False
        try:
            return bool(shaped.contains(QgsPointXY(raster_pt)))
        except Exception:  # noqa: BLE001
            return False

    def _wait_again(self) -> None:

        from ...core.server_dials import dial_in_range

        self._wait_ticks += 1
        wait_ticks_max = dial_in_range(
            "tuning.hover.wait_ticks_max", _WAIT_TICKS_MAX, 1, 100)
        if self._wait_ticks > wait_ticks_max or self._timer is None:
            return
        wait_ticks_fast = dial_in_range(
            "tuning.hover.wait_ticks_fast", _WAIT_TICKS_FAST, 0, 100)


        wait_tick_ms = dial_in_range(
            "tuning.hover.wait_tick_ms", _WAIT_TICK_MS, 50, 2000)
        wait_tick_slow_ms = dial_in_range(
            "tuning.hover.wait_tick_slow_ms", _WAIT_TICK_SLOW_MS, 50, 5000)
        interval = (wait_tick_ms if self._wait_ticks <= wait_ticks_fast
                    else wait_tick_slow_ms)
        try:
            self._timer.start(interval)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _auth_headers(self) -> dict | None:





        now = time.monotonic() * 1000.0
        if self._auth and now - self._auth_at < route_memo_ms(_ROUTE_MEMO_MS):
            return self._auth
        try:
            from ...core.activation_manager import get_auth_header

            auth = get_auth_header()
        except Exception:  # noqa: BLE001
            return None
        if not auth:
            self._auth = None
            return None
        self._auth = auth
        self._auth_at = now
        return auth

    def _preview_url(self) -> str | None:






        now = time.monotonic() * 1000.0
        if self._url and now - self._url_at < route_memo_ms(_ROUTE_MEMO_MS):
            return self._url
        try:
            from ...core.hover_preview_client import preview_refine_url

            url = preview_refine_url()
        except Exception:  # noqa: BLE001
            url = None
        self._url = url or None
        self._url_at = now
        return self._url

    def _send(self, token: str, bounds: tuple, shape: tuple, col, row) -> None:
        from ...core.hover_preview_client import HoverPreviewCall, build_preview_body

        url = self._preview_url()
        if not url:
            return
        auth = self._auth_headers()
        if not auth:
            return



        self._wait_ticks = 0


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

        if self._torn_down or serial != self._serial:
            return
        self._call = None
        if not isinstance(answer, dict) or answer.get("error") or answer.get("code"):
            self._note_failure(str((answer or {}).get("code") or "refused"))
            return



        if not self._gates_open() or not self._route_gates_open():
            return
        try:
            from ...core.hover_preview_client import read_preview_answer

            read = read_preview_answer(answer, shape[0], shape[1])
            if read is None:
                return
            mask, score, logits = read
            if not self._cursor_still_on(bounds, shape, asked, mask):



                return



            if self._draw_answer_mask(mask, bounds, shape):
                self._answer = (bounds, shape, asked, mask, score, logits)


                self._remember_answer((str(token or ""), bounds, shape),
                                      asked, mask, score, logits)
            else:




                self._remember_answer((str(token or ""), bounds, shape),
                                      asked, mask, score, logits,
                                      shaped_to_nothing=True)
        except Exception:  # noqa: BLE001
            self._note_failure("draw")

    def _draw_answer_mask(self, mask, bounds: tuple, shape: tuple) -> bool:











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



                overlay.show_preview_polygon(on_canvas, ground)
                self._shown = (bounds, shape, mask, shaped)
                return True
        overlay.clear_preview()
        self._shown = None
        self._answer = None
        return False

    def _shaped_answer_geometry(self, mask, bounds: tuple, shape: tuple):










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
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            config, settings = self._hover_shape_key(info)
            memo = self._hover_mask_memo
            if (memo is not None and memo[0] is mask and memo[1] is config
                    and memo[2] == settings):


                return memo[3]
            started = time.monotonic()
            outline = plugin._manual_outline_for(mask, info)
            cost_ms = (time.monotonic() - started) * 1000.0
            if cost_ms > hover_shape_budget_ms(_SHAPE_BUDGET_MS):
                self._slow_mask = mask
            if outline is None or outline.isEmpty():
                outline = None


            self._hover_mask_memo = (mask, config, settings, outline)
            return outline
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _mask_too_wide_to_shape(mask, shape: tuple) -> bool:






        try:
            total = int(shape[0]) * int(shape[1])
            if total <= 0:
                return False
            return int(mask.sum()) > hover_shape_max_coverage(_SHAPE_MAX_COVERAGE) * total
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _mask_spans_crop(mask, shape: tuple) -> bool:







        try:
            total = int(shape[0]) * int(shape[1])
            if total <= 0:
                return True
            return int(mask.sum()) > hover_recall_one_object_coverage(_RECALL_ONE_OBJECT_COVERAGE) * total
        except Exception:  # noqa: BLE001
            return True

    def _hover_shape_key(self, info: dict) -> tuple:





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













        answer = self._answer
        shown = self._shown
        self._answer = None
        if answer is None or shown is None or self._torn_down:
            return None
        try:
            from ...core.hover_preview_client import hover_preview_reuse_offered

            if not hover_preview_reuse_offered():
                return None
        except Exception:  # noqa: BLE001
            return None
        drawn = shown[3] if len(shown) > 3 else None
        if drawn is not None:


            try:
                drawn = QgsGeometry(drawn)
            except Exception:  # noqa: BLE001
                drawn = None
        return tuple(answer) + (drawn,)

    def reshape_shown(self) -> None:



        if self._torn_down:
            return



        self._slow_mask = None
        self._recent_answers = [entry for entry in self._recent_answers
                                if not (len(entry) > 5 and entry[5])]
        shown = self._shown
        overlay = self._overlay
        if shown is None or overlay is None or not overlay.has_preview():
            return
        try:
            self._draw_answer_mask(shown[2], shown[0], shown[1])
        except Exception:  # noqa: BLE001
            self.stop("reshape failed")

    def _cursor_still_on(self, bounds: tuple, shape: tuple, asked: tuple,
                         mask) -> bool:






        point = self._pending_point
        if point is None:
            return False
        try:
            raster_pt = self._plugin._transform_to_raster_crs(point)
            if raster_pt is None:
                return False


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
            from ...core.server_dials import dial_in_range

            still_near_px = dial_in_range(
                "tuning.hover.still_near_px", _STILL_NEAR_PX, 4, 100)
            return (abs(row - asked[0]) <= still_near_px
                    and abs(col - asked[1]) <= still_near_px)
        except Exception:  # noqa: BLE001
            return False

    def _note_failure(self, code: str) -> None:






        from ...core.hover_preview_client import PREVIEW_BUSY_CODE, log_preview_note

        named = _CODE_ALLOWED.sub("", (code or "").strip().upper())[:_CODE_CHARS_MAX]
        if not named:
            named = "REFUSED"






        transient = named in _TRANSIENT_CODES
        quiet = (hover_transient_quiet_s(_TRANSIENT_QUIET_S) if transient
                 else hover_refusal_quiet_s(_REFUSAL_QUIET_S))
        self._quiet_until = time.monotonic() + quiet
        if transient:







            try:
                self._plugin._maybe_warmup_auto()
            except (RuntimeError, AttributeError):
                pass  # nosec B110
        if named == PREVIEW_BUSY_CODE or named in self._reported:
            return
        if len(self._reported) >= _reported_max():
            return
        self._reported.add(named)
        log_preview_note(f"Hover preview: nothing drawn ({named})")



    def _ensure_overlay(self):
        if self._overlay is not None:
            return self._overlay
        try:
            from ..hover_preview_overlay import HoverPreviewOverlay

            self._overlay = HoverPreviewOverlay(self._canvas)
        except Exception:  # noqa: BLE001
            self._overlay = None
        return self._overlay

    def _ground_rectangle(self, bounds: tuple) -> QgsRectangle | None:








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
        except Exception:  # noqa: BLE001
            return None

    def _pen_slack_units(self) -> float:





        try:
            units = float(self._canvas.mapSettings().mapUnitsPerPixel())
            return max(0.0, units * HOVER_PREVIEW_OUTLINE_WIDTH)
        except Exception:  # noqa: BLE001
            return 0.0

    def _preview_handle(self):

        predictor = getattr(self._plugin, "predictor", None)
        getter = getattr(predictor, "hover_preview_handle", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:  # noqa: BLE001
            return None

    def _gates_open(self) -> bool:









        plugin = self._plugin
        if plugin is None or self._torn_down:
            return False
        try:
            if getattr(plugin, "_headless", False):
                return False
            mode = self._interactive_mode
            dock = plugin.dock_widget
            if mode is None or dock is None or getattr(dock, "_mode", None) != mode:
                return False


            if not dock.isVisible():
                return False
            canvas = plugin.iface.mapCanvas()
            if canvas is None or canvas.mapTool() is not plugin.map_tool:
                return False
            tool = plugin.map_tool
            if tool is None or not tool.isActive() or tool.is_space_panning():
                return False


            if getattr(plugin, "_refine_handoff_active", False):
                return False
            if getattr(plugin, "_auto_review", None) is not None:
                return False



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


            return bool(plugin._manual_cloud_predictor_active())
        except Exception:  # noqa: BLE001
            return False

    def _route_gates_open(self) -> bool:







        plugin = self._plugin
        if plugin is None or self._torn_down:
            return False
        try:
            return bool(plugin._hover_preview_could_run())
        except Exception:  # noqa: BLE001
            return False


class ManualHoverPreviewMixin:


    def _hover_preview_controller(self):

        controller = getattr(self, "_hover_preview", None)
        if controller is None:
            controller = HoverPreviewController(self)
            controller.attach()
            self._hover_preview = controller
        return controller

    def _on_hover_cursor_moved(self, point) -> None:

        try:
            controller = getattr(self, "_hover_preview", None)
            if controller is None:


                if not self._hover_preview_could_run():
                    return
                controller = self._hover_preview_controller()
            controller.on_cursor_moved(point)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _hover_preview_could_run(self) -> bool:







        if getattr(self, "_headless", False):
            return False
        now = time.monotonic() * 1000.0
        memo = getattr(self, "_hover_route_memo", None)
        if memo is not None and now - memo[0] < route_memo_ms(_ROUTE_MEMO_MS):
            return memo[1]
        try:
            from ...core.hover_preview_client import hover_preview_offered

            answer = bool(hover_preview_offered()
                          and self._manual_cloud_route_ready())
        except Exception:  # noqa: BLE001
            answer = False
        self._hover_route_memo = (now, answer)
        return answer

    def _take_hover_preview_answer(self):






        controller = getattr(self, "_hover_preview", None)
        self._hover_click_shape = None
        if controller is None:
            return None
        try:
            answer = controller.take_shown_answer()
        except Exception:  # noqa: BLE001
            return None
        if answer is not None:


            self._hover_click_shape = getattr(controller, "_hover_mask_memo", None)
        return answer

    def _reshape_hover_preview(self) -> None:


        controller = getattr(self, "_hover_preview", None)
        if controller is None:
            return
        try:
            controller.reshape_shown()
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _stop_hover_preview(self, reason: str = "") -> None:

        controller = getattr(self, "_hover_preview", None)
        if controller is None:
            return
        try:
            controller.stop(reason)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _teardown_hover_preview(self) -> None:

        controller = getattr(self, "_hover_preview", None)
        self._hover_preview = None
        self._hover_route_memo = None
        if controller is None:
            return
        try:
            controller.detach()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
