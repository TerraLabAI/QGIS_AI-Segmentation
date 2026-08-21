"""One hover preview asked of the service, without ever holding the window.

A preview is what the user sees before they click: the outline under the
cursor, drawn while they are still deciding. It rides the same route as a
click, on the same crop, with the same headers, and differs in three ways that
all follow from nobody having asked for it.

- It never sends pixels. A preview is only ever asked about a crop the service
  already holds by name, so a cursor crossing ground the session has not
  prepared costs one local check and no request at all.
- It never waits. The reply is read from a callback, so the map keeps drawing,
  the cursor keeps moving, and a preview still travelling when the next one
  starts is simply dropped.
- It never speaks. Every failure ends in nothing being drawn: a busy service, a
  timeout, a refusal and a cold start all read the same to a user who has not
  asked for anything.

The switch is fail-CLOSED, like every other route that only exists if a server
answers: no configuration means no previews, and the plugin behaves exactly as
it did before this file existed.
"""
from __future__ import annotations

import json

from qgis.core import Qgis, QgsMessageLog, QgsNetworkAccessManager
from qgis.PyQt.QtCore import QByteArray, QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from .qt_compat import resolve_qt_enum
from .server_dials import dial_bool, dial_in_range

# The fleet-wide switch, so the whole feature can be withdrawn with one deploy
# and no plugin release.
HOVER_PREVIEW_FEATURE = "hover_preview"

# The switch for committing the ghost itself when the click lands on it, so the
# user pays one round trip for an object instead of two. Its own switch, and
# not the one above, because it may only be served while the service answers a
# preview through the SAME chain it answers a click through. Where the two
# chains differ the ghost is a promise the click improves on, and committing it
# would spend that improvement to save a wait.
HOVER_PREVIEW_REUSE_FEATURE = "hover_preview_click_reuse"

# The service is already answering as many previews as it will take right now.
# The one refusal that is expected in normal use, and the one the caller must
# be silent about: another preview is a mouse move away.
PREVIEW_BUSY_CODE = "PREVIEW_BUSY"

# How long the cursor rests before a preview is asked for. Long enough that
# crossing the map costs nothing, short enough that resting on an object feels
# like an answer rather than a wait.
_DEBOUNCE_DEFAULT_MS = 110
_DEBOUNCE_FLOOR_MS = 60
_DEBOUNCE_CEILING_MS = 500

# The longest one preview may stay on the wire. Nobody is waiting on it, so a
# preview that outlives the gesture that asked for it is only holding a socket.
_TIMEOUT_DEFAULT_MS = 8_000
_TIMEOUT_FLOOR_MS = 1_000
_TIMEOUT_CEILING_MS = 30_000

_HTTP_STATUS = resolve_qt_enum(QNetworkRequest, "Attribute", "HttpStatusCodeAttribute")

# The calls on the wire, held only so Python does not collect one mid-request.
# Cleared as each finishes, exactly like the connection warm in click_transport.
_live_calls: set = set()


def log_preview_note(message: str) -> None:
    """One line about the preview loop. Shared with the controller, so the
    whole feature writes under one name and stays greppable."""
    QgsMessageLog.logMessage(message, "AI Segmentation", level=Qgis.MessageLevel.Info)


def hover_preview_offered() -> bool:
    """Whether previews are offered at all. Fail-CLOSED.

    Same shape and same reason as ``manual_cloud_route_offered``: the feature
    exists only where a server answers a preview, so a plugin that has never
    heard from one must never start sending imagery on a mouse move.

    This session's own fetch, and nothing older. The disk copy an earlier
    session left says what the server said then, so a feature withdrawn today
    would keep answering off it until the next fetch lands.
    """
    try:
        from .config_cache import config_source

        if config_source() != "live":
            return False
        return dial_bool(f"features.{HOVER_PREVIEW_FEATURE}", False)
    except Exception:  # noqa: BLE001 -- no configuration means no previews  # nosec B110
        return False


def hover_preview_reuse_offered() -> bool:
    """Whether a click may stand on the ghost's own answer. Fail-CLOSED.

    Same shape and same reason as ``hover_preview_offered``, and read as its
    own switch: a plugin that has never heard from a server must send the click
    it has always sent.
    """
    try:
        from .config_cache import config_source

        if config_source() != "live":
            return False
        return dial_bool(f"features.{HOVER_PREVIEW_REUSE_FEATURE}", False)
    except Exception:  # noqa: BLE001 -- no configuration means one round trip  # nosec B110
        return False


def hover_preview_debounce_ms() -> int:
    """How long the cursor must rest before a preview goes out."""
    return dial_in_range("ui.hover_preview_debounce_ms", _DEBOUNCE_DEFAULT_MS,
                         _DEBOUNCE_FLOOR_MS, _DEBOUNCE_CEILING_MS)


def hover_preview_timeout_ms() -> int:
    """The transfer deadline one preview carries."""
    return dial_in_range("ui.hover_preview_timeout_ms", _TIMEOUT_DEFAULT_MS,
                         _TIMEOUT_FLOOR_MS, _TIMEOUT_CEILING_MS)


def build_preview_body(crop_token: str, col: float, row: float) -> dict:
    """The request for one point on a crop the service already holds.

    Deliberately the click body with the pixels left out and ``preview`` added,
    so the far side reads one shape whichever asked. No seed travels and none
    is named: a preview stands alone and must never enter the click session's
    chain of answers.

    ``multimask_output`` is True because that is what the click this preview
    stands for will send: a ghost only ever shows before the first click of a
    session, and a first click asks for the candidate set (see
    ``_run_prediction``). Asked with False, the service answered from a single
    candidate, picked differently, and the committed shape was not the shape
    the ghost had promised.
    """
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
    }


def preview_refine_url() -> str | None:
    """The address a preview is sent to, or None when it cannot be resolved.

    The same address a click uses, read off the one client that owns the host
    split, so a preview can never travel to a host the clicks do not.
    """
    try:
        from ..api.terralab_client import TerraLabClient

        return TerraLabClient().refine_endpoint_url()
    except Exception:  # noqa: BLE001 -- no address means no preview
        return None


def read_preview_answer(answer: dict, height: int, width: int):
    """``(mask, score, logits)`` for the one candidate a preview stands on.

    None when there is nothing to draw. Strict on the way in: the mask has to
    be the size of the crop it was asked about, or it would be stretched across
    the wrong ground. Nothing here is reported to the user, so a refusal is
    simply no picture.

    ``score`` is 0.0 and ``logits`` is None when the answer carried neither.
    Both are what a click that stands on this answer keeps: the score is the
    number the unsure hint reads, and the logits are the seed the NEXT click of
    the same object refines from.
    """
    try:
        import numpy as np

        from .cloud_detection import decode_rle_to_mask
    except Exception:  # noqa: BLE001 -- no decoder, no preview
        return None
    try:
        shape = answer.get("masks_shape")
        rles = answer.get("masks")
        if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                or not isinstance(rles, (list, tuple)) or not rles):
            return None
        if int(shape[1]) != int(height) or int(shape[2]) != int(width):
            return None
        scores = answer.get("scores")
        scored = (isinstance(scores, (list, tuple))
                  and len(scores) == len(rles))
        index = 0
        decoded = [decode_rle_to_mask(r, int(height), int(width), strict=True)
                   for r in rles]
        if scored and len(rles) > 1:
            # More than one candidate came back: choose exactly as the click
            # does in _run_prediction, or the ghost shows one candidate and
            # the click commits another. Area is read off the decoded masks,
            # the same masks the ghost would draw.
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
        score = float(scores[index]) if scored else 0.0
        return mask, score, _preview_logits_row(answer, index)
    except Exception:  # noqa: BLE001 -- an unreadable answer is no picture
        return None


def _preview_logits_row(answer: dict, index: int):
    """The chosen candidate's low-resolution logits, shaped ``(1, s, s)``.

    None whenever they cannot be read. They are only ever the seed for a
    following click, and a click with no seed still answers, so this never
    costs the preview its picture.
    """
    try:
        from .cloud_sam_predictor import unpack_float16_payload

        shape = answer.get("low_res_masks_shape")
        payload = answer.get("low_res_masks")
        if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                or not isinstance(payload, str) or not payload):
            return None
        dims = tuple(int(v) for v in shape)
        if not 0 <= index < dims[0]:
            return None
        return unpack_float16_payload(payload, dims)[index:index + 1]
    except Exception:  # noqa: BLE001 -- an unreadable seed is no seed
        return None


class HoverPreviewCall:
    """One preview on the wire, answered through a callback.

    The caller holds it to abandon it: a newer hover supersedes this one, and
    the answer that lands afterwards must never be drawn. Abandoning ends the
    reply and drops the callback, so a controller torn down mid-flight leaves
    nothing behind to fire.
    """

    def __init__(self, url: str, body: dict, auth: dict, on_answer) -> None:
        self._url = url
        self._body = body
        self._auth = auth or {}
        self._on_answer = on_answer
        self._reply = None
        self._done = False

    def send(self) -> bool:
        """Put the preview on the wire. False when nothing went out."""
        try:
            manager = QgsNetworkAccessManager.instance()
            if manager is None:
                return False
            payload = json.dumps(self._body, allow_nan=False).encode("utf-8")
            request = QNetworkRequest(QUrl(self._url))
            request.setRawHeader(b"Content-Type", b"application/json")
            # Absent below Qt 5.15, same guard as the download path.
            if hasattr(request, "setTransferTimeout"):
                request.setTransferTimeout(hover_preview_timeout_ms())
            for key, value in self._auth.items():
                request.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))
            reply = manager.post(request, QByteArray(payload))
            if reply is None:
                return False
        except Exception:  # noqa: BLE001 -- a preview that cannot start is no preview
            return False
        # The reply is a child of the manager, so a manager that dies takes it
        # with it. Connected BEFORE it is held, so nothing can sit in the set
        # for the life of the process with nothing left to take it out.
        self._reply = reply
        try:
            reply.finished.connect(self._on_finished)
        except Exception:  # noqa: BLE001 -- an unconnectable reply is abandoned
            self.abandon()
            return False
        _live_calls.add(self)
        return True

    def abandon(self) -> None:
        """End this preview for good. Never raises: teardown calls it."""
        self._done = True
        self._on_answer = None
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

    def _on_finished(self) -> None:
        """Read the answer and hand it on. Runs on the thread that drew."""
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
                    except Exception:  # noqa: BLE001 -- an unreadable body is a refusal
                        parsed = None
                if isinstance(parsed, dict):
                    answer = parsed
                if status is not None and status >= 400 and "code" not in answer:
                    answer = {"error": "refused", "code": f"HTTP_{status}"}
                if not answer:
                    answer = {"error": "no answer", "code": "NO_ANSWER"}
        except Exception:  # noqa: BLE001 -- a preview never reports its own failure
            answer = {"error": "unreadable", "code": "UNREADABLE"}
        handler = self._on_answer
        self.abandon()
        if handler is None:
            return
        try:
            handler(answer)
        except Exception:  # noqa: BLE001 -- a preview must never abort its caller
            log_preview_note("Hover preview: the answer could not be drawn")
