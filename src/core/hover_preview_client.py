




















from __future__ import annotations

import json
import math
import threading
import time
from collections import deque

from qgis.core import Qgis, QgsMessageLog, QgsNetworkAccessManager
from qgis.PyQt.QtCore import QByteArray, QTimer

from .network_busy import begin as _busy_begin
from .network_busy import end as _busy_end
from .qt_compat import reply_http_status
from .server_dials import dial_bool, dial_in_range



HOVER_PREVIEW_FEATURE = "hover_preview"







HOVER_PREVIEW_REUSE_FEATURE = "hover_preview_click_reuse"




PREVIEW_BUSY_CODE = "PREVIEW_BUSY"




_DEBOUNCE_DEFAULT_MS = 110
_DEBOUNCE_FLOOR_MS = 60
_DEBOUNCE_CEILING_MS = 500



_TIMEOUT_DEFAULT_MS = 8_000
_TIMEOUT_FLOOR_MS = 1_000
_TIMEOUT_CEILING_MS = 30_000



_live_calls: set = set()


def log_preview_note(message: str) -> None:


    QgsMessageLog.logMessage(message, "AI Segmentation", level=Qgis.MessageLevel.Info)


def hover_preview_offered() -> bool:










    try:
        from .config_cache import config_source
        from .served_config import served_config_ready

        if config_source() != "live" or not served_config_ready():
            return False
        return dial_bool(f"features.{HOVER_PREVIEW_FEATURE}", False)
    except Exception:  # noqa: BLE001  # nosec B110
        return False


def hover_preview_reuse_offered() -> bool:






    try:
        from .config_cache import config_source

        if config_source() != "live":
            return False
        return dial_bool(f"features.{HOVER_PREVIEW_REUSE_FEATURE}", False)
    except Exception:  # noqa: BLE001  # nosec B110
        return False









_RECENT_TRIPS_MAX = 32
_MIN_TRIPS_TO_JUDGE = 32
_recent_trips_ms: deque = deque(maxlen=_RECENT_TRIPS_MAX)
_trips_lock = threading.Lock()




_SLOW_LINK_MS_DEFAULT = 0
_SLOW_DEBOUNCE_DEFAULT_MS = 0


def note_preview_round_trip(elapsed_ms: float) -> None:

    try:
        if math.isfinite(elapsed_ms) and elapsed_ms >= 0:
            with _trips_lock:
                _recent_trips_ms.append(float(elapsed_ms))
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def median_preview_round_trip_ms() -> float | None:

    window = int(dial_in_range("tuning.hover.trips_window", _RECENT_TRIPS_MAX,
                               1, _RECENT_TRIPS_MAX))
    least = int(dial_in_range("tuning.hover.trips_min", _MIN_TRIPS_TO_JUDGE,
                              1, _RECENT_TRIPS_MAX))
    with _trips_lock:
        trips = sorted(list(_recent_trips_ms)[-window:])
    if len(trips) < least:
        return None
    mid = len(trips) // 2
    return trips[mid] if len(trips) % 2 else (trips[mid - 1] + trips[mid]) / 2.0


def forget_preview_round_trips() -> None:

    with _trips_lock:
        _recent_trips_ms.clear()


def hover_preview_debounce_ms() -> int:






    base = dial_in_range("ui.hover_preview_debounce_ms", _DEBOUNCE_DEFAULT_MS,
                         _DEBOUNCE_FLOOR_MS, _DEBOUNCE_CEILING_MS)
    slow_after = dial_in_range("tuning.hover.slow_link_ms",
                               _SLOW_LINK_MS_DEFAULT, 200, 10_000)
    if slow_after <= 0:
        return base
    median = median_preview_round_trip_ms()
    if median is None or median <= slow_after:
        return base
    return max(base, dial_in_range("tuning.hover.slow_link_debounce_ms",
                                   _SLOW_DEBOUNCE_DEFAULT_MS, 100, 5_000))


def hover_preview_timeout_ms() -> int:

    return dial_in_range("ui.hover_preview_timeout_ms", _TIMEOUT_DEFAULT_MS,
                         _TIMEOUT_FLOOR_MS, _TIMEOUT_CEILING_MS)


def build_preview_body(crop_token: str, col: float, row: float,
                       opening_candidate: int | None = None) -> dict:























    from .cloud_click_predictor import LOW_RES_NAMED

    body = {
        "crop": None,
        "crop_shape": None,
        "crop_format": "raw",
        "crop_token": crop_token,
        "points": [[float(col), float(row)]],
        "labels": [1],
        "mask_input": None,
        "mask_input_shape": None,
        "multimask_output": True,
        "preview": True,
        "low_res": LOW_RES_NAMED,
    }
    if opening_candidate is not None:
        body["opening_candidate"] = int(opening_candidate)
        del body["low_res"]
    return body


def preview_refine_url() -> str | None:





    try:
        from ..api.terralab_client import TerraLabClient

        return TerraLabClient().refine_endpoint_url()
    except Exception:  # noqa: BLE001
        return None


def preview_frame_holds_crop(frame, crop) -> bool:





    try:
        fh, fw = int(frame[0]), int(frame[1])
        ch, cw = int(crop[0]), int(crop[1])
    except (TypeError, ValueError, IndexError):
        return False
    return 0 < ch <= fh and 0 < cw <= fw


def frame_stand_in_logits(mask, frame, side: int):




    import numpy as np

    from .cloud_click_predictor import mask_stand_in_logits

    mask = np.asarray(mask, dtype=bool)
    fh, fw = (int(frame[0]), int(frame[1])) if frame is not None else mask.shape
    if (fh, fw) != mask.shape:
        full = np.zeros((fh, fw), dtype=bool)
        full[:mask.shape[0], :mask.shape[1]] = mask[:fh, :fw]
        mask = full
    return mask_stand_in_logits(mask, side)


def read_preview_answer(answer: dict, height: int, width: int, crop_shape=None):
















    try:
        import numpy as np

        from .cloud_detection import decode_rle_to_mask
    except Exception:  # noqa: BLE001
        return None
    try:
        if not isinstance(answer, dict) or "error" in answer:
            return None
        shape = answer.get("masks_shape")
        rles = answer.get("masks")
        if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                or not isinstance(rles, (list, tuple)) or not rles):
            return None
        from .cloud_click_predictor import _MAX_MASK_COUNT, _MAX_MASK_SIDE

        if (any(isinstance(v, bool) or not isinstance(v, int) for v in shape)
                or not 0 < shape[0] <= _MAX_MASK_COUNT or len(rles) != shape[0]
                or any(not 0 < v <= _MAX_MASK_SIDE for v in shape[1:])
                or shape[1] != height or shape[2] != width):
            return None
        scores = answer.get("scores")
        scored = (isinstance(scores, (list, tuple))
                  and len(scores) == len(rles))
        if not scored or any(not math.isfinite(float(score)) for score in scores):



            return None
        index = 0
        decoded = [decode_rle_to_mask(r, int(height), int(width), strict=True)
                   for r in rles]
        if len(rles) > 1:




            from .multimask_pick import pick_multimask_index

            total = int(height) * int(width)
            areas = [int(np.count_nonzero(m)) for m in decoded]
            index = pick_multimask_index(areas, scores, total)
        if crop_shape is not None:
            ch, cw = int(crop_shape[0]), int(crop_shape[1])
            decoded = [np.array(m[:ch, :cw], copy=True) for m in decoded]
        mask = decoded[index]
        if not np.any(mask):
            return None
        score = float(scores[index])
        if len(decoded) > 1:
            note_preview_alternatives(mask, [
                (decoded[i], float(scores[i]))
                for i in range(len(decoded)) if i != index],
                frame=(int(height), int(width)))
        return mask, score, _preview_logits_row(
            answer, index, mask, frame=(int(height), int(width)))
    except Exception:  # noqa: BLE001
        return None





_PREVIEW_ALTERNATIVES: list = []


def note_preview_alternatives(mask, alternatives, frame=None) -> None:


    import numpy as np

    kept = dial_in_range("tuning.hover.recent_answers_kept", 6, 1, 20)
    packed = [(np.packbits(np.asarray(m, dtype=bool)), tuple(np.shape(m)), float(sc))
              for m, sc in alternatives]
    _PREVIEW_ALTERNATIVES.append((mask, packed, frame))
    del _PREVIEW_ALTERNATIVES[:-kept]


def preview_alternatives_for(mask) -> list:

    import numpy as np

    for held, packed, _frame in reversed(_PREVIEW_ALTERNATIVES):
        if held is mask:
            out = []
            for bits, shape, score in packed:
                count = int(np.prod(shape))
                out.append((np.unpackbits(bits, count=count).astype(bool).reshape(shape),
                            score))
            return out
    return []


def preview_alternatives_frame(mask):


    for held, _packed, frame in reversed(_PREVIEW_ALTERNATIVES):
        if held is mask:
            return frame
    return None


def forget_preview_alternatives(keep_newest: bool = False) -> None:


    del _PREVIEW_ALTERNATIVES[:-1 if keep_newest else None]


def _preview_logits_row(answer: dict, index: int, mask=None, frame=None):











    try:
        from .cloud_click_predictor import (
            note_preview_seed,
            unpack_float16_payload,
        )

        shape = answer.get("low_res_masks_shape")
        payload = answer.get("low_res_masks")
        if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                or any(isinstance(v, bool) or not isinstance(v, int) for v in shape)):
            return None
        if payload is None:
            seed_id = answer.get("seed_id")
            token = answer.get("crop_token")
            pick = answer.get("low_res_pick", 0)
            if (mask is None or not isinstance(seed_id, str) or not seed_id
                    or not isinstance(token, str) or not token
                    or isinstance(pick, bool) or pick != index
                    or shape[0] != 1 or shape[1] != shape[2]):
                return None
            stand_in = frame_stand_in_logits(mask, frame, int(shape[1]))
            note_preview_seed(token, seed_id, stand_in)
            return stand_in
        if not isinstance(payload, str) or not payload:
            return None
        dims = tuple(shape)
        if dims[0] != len(answer.get("masks", [])) or not 0 <= index < dims[0]:
            return None
        return unpack_float16_payload(payload, dims)[index:index + 1]
    except Exception:  # noqa: BLE001
        return None


class HoverPreviewCall:








    def __init__(self, url: str, body: dict, auth: dict, on_answer) -> None:
        self._url = url
        self._body = body
        self._auth = auth or {}
        self._on_answer = on_answer
        self._reply = None
        self._timer = None
        self._done = False
        self._sent_at = 0.0
        self._busy_token = None

    def send(self) -> bool:

        if self._done or self._reply is not None:
            return False
        try:
            from ..api.json_request import build_json_request
            from ..api.terralab_client import TerraLabClient

            TerraLabClient._reject_cleartext_remote(self._url)
            manager = QgsNetworkAccessManager.instance()
            if manager is None:
                return False
            payload = json.dumps(self._body, allow_nan=False).encode("utf-8")


            request = build_json_request(self._url, self._auth, hover_preview_timeout_ms())
            reply = manager.post(request, QByteArray(payload))
            if reply is None:
                return False
        except Exception:  # noqa: BLE001
            return False




        self._reply = reply
        try:
            reply.finished.connect(self._on_finished)
            self._timer = QTimer()
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._on_timeout)
            self._timer.start(hover_preview_timeout_ms())
        except Exception:  # noqa: BLE001
            self.abandon()
            return False
        self._sent_at = time.monotonic()
        self._busy_token = _busy_begin("hover")
        _live_calls.add(self)
        if reply.isFinished():
            QTimer.singleShot(0, self._on_finished)
        return True

    def abandon(self) -> None:

        self._done = True
        self._on_answer = None


        token, self._busy_token = self._busy_token, None
        _busy_end(token)
        timer, self._timer = self._timer, None
        if timer is not None:
            try:
                timer.stop()
                timer.deleteLater()
            except RuntimeError:
                pass  # nosec B110
        reply = self._reply
        self._reply = None
        _live_calls.discard(self)
        if reply is None:
            return
        try:
            reply.finished.disconnect(self._on_finished)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        try:
            if not reply.isFinished():
                reply.abort()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        try:
            reply.deleteLater()
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _on_timeout(self) -> None:
        if not self._done:
            self._deliver({"error": "preview timed out", "code": "TIMEOUT"})

    def _deliver(self, answer: dict) -> None:
        handler = self._on_answer
        if self._sent_at and not self._done:


            note_preview_round_trip((time.monotonic() - self._sent_at) * 1000.0)
        self.abandon()
        if handler is not None:
            try:
                handler(answer)
            except Exception:  # noqa: BLE001
                log_preview_note("Hover preview: the answer could not be drawn")

    def _on_finished(self) -> None:

        if self._done:
            return
        answer: dict = {}
        reply = self._reply
        try:
            if reply is not None:
                raw = bytes(reply.readAll())
                status = reply_http_status(reply)
                parsed = None
                if raw:
                    try:
                        parsed = json.loads(raw.decode("utf-8", "replace"))
                    except Exception:  # noqa: BLE001
                        parsed = None
                if isinstance(parsed, dict):
                    answer = parsed
                from ..api.terralab_client_primitives import _NoError

                if status is not None and status >= 400:
                    answer = dict(answer, error=answer.get("error") or "refused",
                                  code=answer.get("code") or f"HTTP_{status}")
                elif reply.error() != _NoError:
                    answer = {"error": "preview transfer failed", "code": "NO_ANSWER"}
                if not answer:
                    answer = {"error": "no answer", "code": "NO_ANSWER"}
        except Exception:  # noqa: BLE001
            answer = {"error": "unreadable", "code": "UNREADABLE"}
        self._deliver(answer)
