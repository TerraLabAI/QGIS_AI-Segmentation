




















from __future__ import annotations

import json
import math

from qgis.core import Qgis, QgsMessageLog, QgsNetworkAccessManager
from qgis.PyQt.QtCore import QByteArray, QTimer, QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from .qt_compat import resolve_qt_enum
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

_HTTP_STATUS = resolve_qt_enum(QNetworkRequest, "Attribute", "HttpStatusCodeAttribute")





_REDIRECT_ATTR = getattr(getattr(QNetworkRequest, "Attribute", QNetworkRequest),
                         "RedirectPolicyAttribute",
                         getattr(QNetworkRequest, "RedirectPolicyAttribute", None))
_RedirectPolicy = getattr(QNetworkRequest, "RedirectPolicy", QNetworkRequest)
_SAME_ORIGIN_REDIRECT = getattr(_RedirectPolicy, "SameOriginRedirectPolicy",
                                getattr(QNetworkRequest, "SameOriginRedirectPolicy", None))
_NO_LESS_SAFE_REDIRECT = getattr(_RedirectPolicy, "NoLessSafeRedirectPolicy",
                                 getattr(QNetworkRequest, "NoLessSafeRedirectPolicy", None))



_live_calls: set = set()


def log_preview_note(message: str) -> None:


    QgsMessageLog.logMessage(message, "AI Segmentation", level=Qgis.MessageLevel.Info)


def hover_preview_offered() -> bool:










    try:
        from .config_cache import config_source

        if config_source() != "live":
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


def hover_preview_debounce_ms() -> int:

    return dial_in_range("ui.hover_preview_debounce_ms", _DEBOUNCE_DEFAULT_MS,
                         _DEBOUNCE_FLOOR_MS, _DEBOUNCE_CEILING_MS)


def hover_preview_timeout_ms() -> int:

    return dial_in_range("ui.hover_preview_timeout_ms", _TIMEOUT_DEFAULT_MS,
                         _TIMEOUT_FLOOR_MS, _TIMEOUT_CEILING_MS)


def build_preview_body(crop_token: str, col: float, row: float) -> dict:


















    from .cloud_sam_predictor import LOW_RES_NAMED

    return {
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


def preview_refine_url() -> str | None:





    try:
        from ..api.terralab_client import TerraLabClient

        return TerraLabClient().refine_endpoint_url()
    except Exception:  # noqa: BLE001
        return None


def read_preview_answer(answer: dict, height: int, width: int):












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
        from .cloud_sam_predictor import _MAX_MASK_COUNT, _MAX_MASK_SIDE

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




            total = int(height) * int(width)
            areas = [int(np.count_nonzero(m)) for m in decoded]
            small_enough = [i for i in range(len(decoded))
                            if 0 < areas[i] < 0.8 * total]
            if small_enough:
                index = max(small_enough, key=lambda i: float(scores[i]))
            else:
                index = min(range(len(decoded)), key=lambda i: areas[i])
        mask = decoded[index]
        if not np.any(mask):
            return None
        score = float(scores[index])
        return mask, score, _preview_logits_row(answer, index, mask)
    except Exception:  # noqa: BLE001
        return None


def _preview_logits_row(answer: dict, index: int, mask=None):











    try:
        from .cloud_sam_predictor import (
            mask_stand_in_logits,
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
            stand_in = mask_stand_in_logits(mask, int(shape[1]))
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

    def send(self) -> bool:

        if self._done or self._reply is not None:
            return False
        try:
            from ..api.terralab_client import TerraLabClient

            TerraLabClient._reject_cleartext_remote(self._url)
            manager = QgsNetworkAccessManager.instance()
            if manager is None:
                return False
            payload = json.dumps(self._body, allow_nan=False).encode("utf-8")
            request = QNetworkRequest(QUrl(self._url))
            request.setRawHeader(b"Content-Type", b"application/json")
            if _REDIRECT_ATTR is not None:
                policy = (_SAME_ORIGIN_REDIRECT if self._auth
                          else _NO_LESS_SAFE_REDIRECT)
                if policy is not None:
                    request.setAttribute(_REDIRECT_ATTR, policy)

            if hasattr(request, "setTransferTimeout"):
                request.setTransferTimeout(hover_preview_timeout_ms())
            for key, value in self._auth.items():
                request.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))
            reply = manager.post(request, QByteArray(payload))
            if reply is None:
                return False
        except Exception:  # noqa: BLE001
            return False



        self._reply = reply
        try:
            reply.finished.connect(self._on_finished)
            reply.destroyed.connect(self._on_destroyed)
            self._timer = QTimer()
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._on_timeout)
            self._timer.start(hover_preview_timeout_ms())
        except Exception:  # noqa: BLE001
            self.abandon()
            return False
        _live_calls.add(self)
        if reply.isFinished():
            QTimer.singleShot(0, self._on_finished)
        return True

    def abandon(self) -> None:

        self._done = True
        self._on_answer = None
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
            reply.destroyed.disconnect(self._on_destroyed)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
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

    def _on_destroyed(self, *_args) -> None:
        if not self._done:
            self._deliver({"error": "preview ended", "code": "NO_ANSWER"})

    def _deliver(self, answer: dict) -> None:
        handler = self._on_answer
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
                status = reply.attribute(_HTTP_STATUS)
                status = int(status) if status is not None else None
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
