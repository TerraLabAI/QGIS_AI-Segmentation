"""Point-refine predictor that answers over the network.

Same three members the on-device predictor exposes to the click path
(``is_image_set``, ``set_image``, ``predict``), plus the two no-ops that make
it a drop-in (``reset_image``, ``cleanup``). A caller holding one of these
cannot tell which it has, and must not try: that is the whole point, because
the remote route exists so a user of Automatic mode never installs anything.

Two properties are worth stating, because they are the reason this class is not
simply "the same calls with a socket in the middle":

- ``set_image`` sends the crop ahead of the click, on the caller's own worker
  thread, and a failure there is not an error: the pixels simply travel with
  the first click instead. The user spends that time deciding where to click,
  so the click itself is left with a few hundred bytes to send.
- The remote side keeps the encoded crop behind a token, and the logits of the
  answer it just gave behind a name. Every later click sends both names instead
  of a megabyte and a half of picture and a quarter of a megabyte of logits.

Manual mode never comes here. It stays on-device and offline by product rule.

Only the venv-free part of numpy is used, so this module is importable with no
local install at all.
"""
from __future__ import annotations

import base64
import time
import zlib

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .log_scrub import scrub_sensitive
from .sam_predictor import SamWorkerError

# The remote side answers with this code when the crop behind a token has been
# dropped. It is the one error the caller never sees: the next request carries
# the pixels again and the click goes through.
CROP_EXPIRED_CODE = "CROP_EXPIRED"

# Same idea one level down: the crop is still there, only the logits behind
# their name are gone. Recovering costs a quarter of a megabyte, not a whole
# picture, so the two are told apart.
SEED_EXPIRED_CODE = "SEED_EXPIRED"

# The far side refused the request itself. On a click carrying a hand-built
# seed this is almost always the seed's own side, which the caller cannot know
# before an answer has told it.
INVALID_INPUT_CODE = "INVALID_INPUT"

# The far side could not read the request body at all. On a request whose crop
# travelled as webp this is what a server that predates the format answers, and
# the recovery is to send the same pixels as PNG, which every server reads.
INVALID_REQUEST_CODE = "INVALID_REQUEST"


# What the caller has to do about a refusal. Three answers, because they need
# three different sentences: top up, sign in, or try again.
REFUSAL_CREDITS = "CREDITS"
REFUSAL_SIGN_IN = "SIGN_IN"
REFUSAL_OTHER = "OTHER"


def click_refusal_class(code: str) -> str:
    """Which of the three answers a refusal code deserves.

    The code sets live in ``error_policy``, where the server can add to them,
    so a code we start sending tomorrow reaches the right sentence without a
    plugin release.
    """
    named = (code or "").strip().upper()
    if not named:
        return REFUSAL_OTHER
    try:
        from .error_policy import EXHAUSTED_CODES, RUN_FATAL_CODES

        if named in EXHAUSTED_CODES:
            return REFUSAL_CREDITS
        if named in RUN_FATAL_CODES:
            return REFUSAL_SIGN_IN
    except Exception:  # noqa: BLE001 -- an unreadable set is not a refusal class
        return REFUSAL_OTHER
    return REFUSAL_OTHER


class RefineRefusedError(SamWorkerError):
    """The far side refused the request, and said which refusal it was.

    ``code`` is the backend's own code, carried whole. Without it every
    refusal reaches the click path as one sentence, and an empty balance, a
    session that signed out and a service fault all read the same, which means
    two of the three tell the user to do the wrong thing.
    """

    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.code = code

    def refusal_class(self) -> str:
        """One of ``REFUSAL_CREDITS``, ``REFUSAL_SIGN_IN``, ``REFUSAL_OTHER``."""
        return click_refusal_class(self.code)


class RefineSupersededError(SamWorkerError):
    """The crop moved on while its answer was still in flight.

    The wait for an answer keeps the window alive, so timers and queued signals
    run inside it: a new crop, a session end or a sign-out can all land there.
    The answer that follows belongs to a crop nobody holds any more, and
    writing its token, its logits or its mask into the session that replaced it
    would put the wrong picture under the user's next click.
    """


# Masks come back at the crop's own size, so a malformed shape would allocate
# whatever the answer asked for. The ceiling is far above any crop the plugin
# sends and well below anything that could hurt.
_MAX_MASK_SIDE = 8192
_MAX_MASK_COUNT = 8


def _log(message: str, level=Qgis.MessageLevel.Info) -> None:
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


def _as_ms(value) -> int:
    """A reported duration as whole milliseconds, or 0 when it is not a number."""
    return int(value) if isinstance(value, (int, float)) else 0


def pack_float16_payload(array: np.ndarray) -> str:
    """float array -> base64 of its zlib-compressed float16 bytes.

    Halving the width before compressing is what keeps a seed mask small
    enough to ride along with the click. The precision lost is far below what a
    mask logit carries.
    """
    half = np.ascontiguousarray(array, dtype=np.float16)
    return base64.b64encode(zlib.compress(half.tobytes(), 6)).decode("ascii")


def unpack_float16_payload(payload: str, shape: tuple[int, ...]) -> np.ndarray:
    """The inverse of ``pack_float16_payload``, widened back to float32.

    float32 because that is what the on-device predictor hands back, and the
    caller feeds it straight into the next request as a seed.
    """
    raw = zlib.decompress(base64.b64decode(payload.encode("ascii")))
    flat = np.frombuffer(raw, dtype=np.float16)
    return np.ascontiguousarray(flat.reshape(shape), dtype=np.float32)


# How hard Qt is asked to squeeze the crop. Qt spells this as a "quality"
# number, but PNG has no lossy setting: it only picks the compression effort,
# and the pixels come back identical whatever it is. The default effort spends
# several times longer than this one to shave a few per cent off a picture that
# is about to travel anyway, and that time is paid on the click path. Never
# swap PNG for a lossy format: the mask has to come back on exactly the pixels
# the caller saw.
#
# It trades local encode time against bytes on the wire, and which way that
# trade falls depends on the user's connection, so it is a dial
# (``network.click_png_effort``). The constant below is the fallback.
_PNG_EFFORT = 70

# Qt's own range for the argument. 0 is legal and means "spend nothing", so a
# served value is bounded rather than required to be positive.
_PNG_EFFORT_MIN = 0
_PNG_EFFORT_MAX = 100


def click_png_effort() -> int:
    """Qt's PNG compression effort for one click crop, 0 to 100.

    Read off the served network policy, with the shipped constant standing for
    an absent, out-of-range or malformed value. Never raises and never
    networks: a click path calls it.
    """
    try:
        from .detection_policy import network_policy

        value = network_policy().get("click_png_effort")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and _PNG_EFFORT_MIN <= value <= _PNG_EFFORT_MAX):
            return int(value)
    except Exception:  # noqa: BLE001 -- a dial must never break a click  # nosec B110
        pass
    return _PNG_EFFORT


def encode_crop_png(crop: np.ndarray) -> bytes | None:
    """A contiguous (H, W, 3) uint8 crop as PNG bytes, or None when Qt cannot.

    PNG rather than the raw bytes because the raw form of a large crop nearly
    fills the request-body ceiling, and lossless because the mask has to come
    back on exactly the pixels the caller saw. Qt does the encoding, never an
    imaging library: the click path has to work for a user whose local install
    carries nothing but the light packages.

    Returns None instead of raising, so a click never fails on a compression
    step: the caller sends the raw bytes when it gets None.
    """
    try:
        from qgis.PyQt.QtCore import QBuffer, QByteArray
        from qgis.PyQt.QtGui import QImage

        from .qt_compat import FormatRGB888, WriteOnly

        height, width = int(crop.shape[0]), int(crop.shape[1])
        # Two things QImage will not do for you. It does not copy the buffer, so
        # `raw` has to outlive it (hence the copy() before `raw` goes out of
        # scope), and a 3-byte-per-pixel row is not 4-byte aligned, so without
        # an explicit bytesPerLine Qt reads past the end of every row and the
        # picture comes out sheared.
        raw = crop.tobytes()
        image = QImage(raw, width, height, width * 3, FormatRGB888).copy()
        if image.isNull():
            return None
        payload = QByteArray()
        buffer = QBuffer(payload)
        buffer.open(WriteOnly)
        written = image.save(buffer, "PNG", click_png_effort())
        buffer.close()
        if not written or payload.isEmpty():
            return None
        return bytes(payload)
    except Exception as err:  # noqa: BLE001 -- the raw bytes still go out
        _log(f"Crop PNG encode failed, sending raw pixels: {err}", Qgis.MessageLevel.Warning)
        return None


def crop_webp_allowed() -> bool:
    """Whether the served flag lets this client send WebP. Off when unserved.

    The dial defaults off because the format is a wire contract: every server
    has to accept webp before any client sends it. Wrapped so a broken dial
    read costs PNG, never the click.
    """
    try:
        from .server_dials import crop_webp_enabled

        return crop_webp_enabled()
    except Exception:  # noqa: BLE001 -- a dial must never break a click  # nosec B110
        return False


def encode_crop_webp(crop: np.ndarray) -> bytes | None:
    """A contiguous (H, W, 3) uint8 crop as LOSSLESS WebP bytes, or None.

    WebP because it packs the same pixels about 30% smaller than PNG on aerial
    imagery, and quality 100 because that is the one value at which Qt's WEBP
    plugin writes lossless: every other value is lossy and would bring the
    mask back on pixels the caller never sent, so no other value is ever
    passed here. Returns None instead of raising (a Qt build may carry no
    webp plugin at all), and the caller falls back to PNG: a click never
    fails on a compression step.
    """
    try:
        from qgis.PyQt.QtCore import QBuffer, QByteArray
        from qgis.PyQt.QtGui import QImage

        from .qt_compat import FormatRGB888, WriteOnly

        height, width = int(crop.shape[0]), int(crop.shape[1])
        # Same two QImage traps as the PNG path above: the buffer is not
        # copied (hence the copy() while `raw` is alive), and a 3-byte-per-
        # pixel row needs its bytesPerLine spelled out or the rows shear.
        raw = crop.tobytes()
        image = QImage(raw, width, height, width * 3, FormatRGB888).copy()
        if image.isNull():
            return None
        payload = QByteArray()
        buffer = QBuffer(payload)
        buffer.open(WriteOnly)
        written = image.save(buffer, "WEBP", 100)
        buffer.close()
        if not written or payload.isEmpty():
            return None
        return bytes(payload)
    except Exception:  # noqa: BLE001 -- PNG still goes out  # nosec B110
        return None


def _accepts_cancel_check(call) -> bool:
    """Whether this client's refine call takes the cancellation hook.

    Asked once per request and before it goes out. A stand-in client in a
    script may carry the older two-argument form, and the answer decides how
    the call is made rather than being read off a failure.
    """
    try:
        import inspect

        return "cancel_check" in inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False


def _crop_identity(image_np: np.ndarray) -> tuple:
    """Cheap content key for a crop, so a token is dropped when the pixels move.

    A checksum rather than a comparison: the previous crop is not kept once its
    token is held, and reading 3 MB is fast next to any network call.
    """
    return (image_np.shape, zlib.crc32(image_np.tobytes()))


class CloudSamPredictor:
    """The click path's predictor, answered by the remote refine route."""

    def __init__(self, client=None, auth=None) -> None:
        """``auth`` is either the headers themselves or something that returns
        them. Hand over the callable whenever the session can outlive one key:
        a dict is read once and never again, so a key that rotates mid-session
        leaves every later click failing over to the machine."""
        self._client = client
        self._auth = auth
        # Which client the cancellable-signature answer was resolved for. It
        # cannot change for a given object, and asking costs more than the
        # answer on a path that runs up to three times per click.
        self._cancel_client = None
        self._cancel_supported = False
        self.is_image_set = False
        self.original_size: tuple[int, int] | None = None
        # Only the on-device SAM1 path ever sets this. Kept so a caller that
        # reads it off either predictor gets an answer instead of an
        # AttributeError.
        self.input_size = None
        self._crop: np.ndarray | None = None
        self._crop_key: tuple | None = None
        self._crop_token: str | None = None
        # The crop already turned into the bytes that travel, kept beside the
        # pixels it came from. Encoding a crop takes long enough to be felt, and
        # the paths that send the pixels rather than the token (the hand-over
        # did not land, or the far side dropped it) run on the thread that
        # draws. Doing that work once is the difference between those paths
        # costing a network wait and costing a visible pause.
        self._crop_body: tuple[str, str] | None = None
        # Set the first time a server refuses a webp body, and never unset: the
        # server behind this session is what it is, and asking it again with
        # the same form would cost every later object its first answer.
        self._webp_refused = False
        # The last answer's logits and the name the far side filed them under.
        # A click that seeds from one of these rows sends the name instead.
        self._seed_id: str = ""
        self._seed_logits: np.ndarray | None = None
        # The side of the low-resolution logits the far side works in. It is not
        # the same on every route, and a seed built at the wrong side is refused
        # outright, so a caller that makes one from a polygon has to ask rather
        # than assume. None until an answer has said, and every answer says.
        self.low_res_side: int | None = None
        # This one is the network, always. Read by anything calibrated on one
        # model and not the other, the unsure hint first.
        self.last_answer_was_remote = True
        # Which crop the session is on. Every request captures it before it
        # goes out and checks it on the way back, because the wait for an
        # answer runs the event loop and a new crop can be set inside it.
        self._generation = 0

    # -- the on-device predictor's shape -----------------------------------

    def warm_up(self) -> bool:
        """Nothing to start. True so a caller's warm-up branch stays honest."""
        return True

    def session_generation(self) -> int:
        """Which crop this predictor is on. Moves whenever the pixels behind it
        change or the session ends, so a caller can tell whether the answer it
        is waiting for still belongs to anything."""
        return self._generation

    def reset_image(self) -> None:
        self._generation += 1
        self.is_image_set = False
        self.original_size = None
        self._crop = None
        self._crop_key = None
        self._crop_token = None
        self._crop_body = None
        self._forget_seed()

    def _forget_seed(self) -> None:
        """Drop the named logits. Called whenever they could no longer be the
        ones the far side is holding."""
        self._seed_id = ""
        self._seed_logits = None

    def cleanup(self) -> None:
        self.reset_image()

    def set_image(self, image_np: np.ndarray) -> None:
        """Take the crop and send it ahead of the first click.

        Validated exactly like the on-device path, so a malformed crop fails
        here with a sentence rather than deep inside the answer. The send that
        follows is best effort by design: the caller runs this on a worker
        thread while the user is still choosing where to click, and if it does
        not land the pixels simply ride along with that click as before.
        """
        if (not isinstance(image_np, np.ndarray) or image_np.ndim != 3
                or image_np.shape[2] != 3):
            shape = getattr(image_np, "shape", None)
            raise SamWorkerError(
                f"Invalid image for encoding: expected (H, W, 3), got shape {shape}")
        if image_np.shape[0] == 0 or image_np.shape[1] == 0:
            raise SamWorkerError(
                f"Invalid image for encoding: empty crop {image_np.shape}")
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)

        key = _crop_identity(image_np)
        if key != self._crop_key:
            # Different pixels: anything still travelling for the old ones is
            # no longer this session's, so move the generation before the new
            # crop is in place.
            self._generation += 1
            self._crop_token = None
            self._crop_body = None
            self._forget_seed()
        self._crop = image_np
        self._crop_key = key
        self.original_size = (int(image_np.shape[0]), int(image_np.shape[1]))
        self.is_image_set = True
        if self._crop_token is None:
            self._register_crop(image_np)

    def _encoded_crop(self, crop: np.ndarray) -> tuple[str, str]:
        """The crop ready to travel: (body text, the form it is in).

        Held for as long as the pixels are, because both the hand-over and any
        later request that has to carry the pixels want the same bytes, and
        producing them is the slowest step on this side of the wire.
        """
        held = self._crop_body
        if held is not None:
            return held
        started = time.monotonic()
        # WebP first when the server says the fleet is ready for it, PNG when
        # it is not or when this Qt cannot write it. Both are lossless, so the
        # far side answers on exactly the pixels the user saw either way.
        packed = (encode_crop_webp(crop)
                  if crop_webp_allowed() and not self._webp_refused else None)
        if packed is not None:
            made = (base64.b64encode(packed).decode("ascii"), "webp")
        else:
            png = encode_crop_png(crop)
            if png is None:
                # Raw pixels are what the far side assumes when nothing says
                # otherwise, and they are the answer whenever Qt cannot compress.
                made = (base64.b64encode(crop.tobytes()).decode("ascii"), "raw")
            else:
                made = (base64.b64encode(png).decode("ascii"), "png")
        self._crop_body = made
        # The one local step slow enough to matter, so it gets its own number
        # instead of hiding inside the round trip's.
        _log(f"Remote refine: crop packed as {made[1]} ({len(made[0]) // 1024} KB) "
             f"in {int((time.monotonic() - started) * 1000)} ms")
        return made

    def _register_crop(self, crop: np.ndarray) -> None:
        """Hand the pixels over now, so the click carries only its points.

        Swallows everything. A crop that does not register is not a failure to
        report: the click path still holds the pixels and sends them itself.
        """
        started = time.monotonic()
        generation = self._generation
        payload, form = self._encoded_crop(crop)
        body = {"crop": payload, "crop_format": form,
                "crop_shape": list(crop.shape)}
        try:
            answer = self._resolve_client().submit_refine_register(
                body, self._resolve_auth())
            if (form == "webp" and generation == self._generation
                    and (answer or {}).get("code") == INVALID_REQUEST_CODE):
                # This server cannot read the webp form (it predates it, or a
                # pinned route does). Repack as PNG, which every server reads,
                # and hand the pixels over again, still ahead of the click.
                # Pinned for the session, so the cost is one wasted upload,
                # once, and never a click.
                _log("Remote refine: the server refused the webp crop, "
                     "sending it as png from now on", Qgis.MessageLevel.Warning)
                self._webp_refused = True
                self._crop_body = None
                payload, form = self._encoded_crop(crop)
                body = {"crop": payload, "crop_format": form,
                        "crop_shape": list(crop.shape)}
                answer = self._resolve_client().submit_refine_register(
                    body, self._resolve_auth())
            if generation != self._generation:
                # The crop changed while its pixels were on their way. This
                # token names the old picture, and writing it here would send
                # the next click at imagery the user has left.
                return
            token = (answer or {}).get("crop_token")
            if not isinstance(token, str) or not token:
                raise ValueError((answer or {}).get("error") or "no token")
            self._crop_token = token
            _log("Remote refine: crop sent ahead of the click, {} KB in {} ms".format(
                len(body["crop"]) // 1024, int((time.monotonic() - started) * 1000)))
        except Exception as err:  # noqa: BLE001 -- the click sends the crop itself
            if generation == self._generation:
                self._crop_token = None
            # Scrubbed: a refusal body is echoed straight into this text and
            # can carry the address it came from.
            _log("Remote refine: crop not sent ahead, the click will carry it: "
                 f"{scrub_sensitive(str(err))}", Qgis.MessageLevel.Info)

    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_image_set or self._crop is None:
            raise RuntimeError("Image has not been set. Call set_image first.")
        if mask_input is not None and (
                mask_input.ndim != 3 or mask_input.shape[0] != 1):
            raise ValueError(
                "Invalid mask seed for prediction: expected (1, H, W), "
                f"got shape {tuple(mask_input.shape)}")

        started = time.monotonic()
        generation = self._generation
        answer = self._post(self._build_body(
            point_coords, point_labels, mask_input, multimask_output,
            send_crop=self._crop_token is None, name_seed=True), generation)
        self._refuse_late_answer(generation)

        if answer.get("code") in (CROP_EXPIRED_CODE, SEED_EXPIRED_CODE):
            # A name the far side no longer honours. Nothing the user did, and
            # nothing they should read: send the thing itself and try once more.
            # A lost crop costs the picture, a lost seed only the logits.
            crop_gone = answer.get("code") == CROP_EXPIRED_CODE
            if crop_gone:
                self._crop_token = None
            self._forget_seed()
            answer = self._post(self._build_body(
                point_coords, point_labels, mask_input, multimask_output,
                send_crop=crop_gone, name_seed=False), generation)
            self._refuse_late_answer(generation)

        if mask_input is not None and answer.get("code") == INVALID_INPUT_CODE:
            # The seed itself is what the far side would not read, and the only
            # seed a caller can get wrong is one it built by hand: the models
            # behind this route work in different low-resolution sides, and a
            # polygon seed is made before any answer has said which. Ask again
            # with the points alone rather than let the click fail over to the
            # machine, which would answer this one click with the other model
            # and never say so. The answer that comes back names the side, so
            # the next seed is built right.
            #
            # The pixels travel again whenever there is still no token to name
            # them by. Dropping the seed does not conjure a crop the far side
            # never received, and a retry with neither is a round trip that
            # cannot succeed.
            answer = self._post(self._build_body(
                point_coords, point_labels, None, multimask_output,
                send_crop=self._crop_token is None, name_seed=False),
                generation)
            self._refuse_late_answer(generation)

        if (answer.get("code") == INVALID_REQUEST_CODE
                and self._crop_body is not None
                and self._crop_body[1] == "webp"):
            # The far side could not read the body, and the one negotiable
            # thing in it is the webp form of the crop: a server behind this
            # route may predate the format. Repack the same pixels as PNG and
            # try once more, so a served flag ahead of its servers costs one
            # retry and never the click. Pinned for the session.
            _log("Remote refine: the server refused the webp crop, the click "
                 "retries as png", Qgis.MessageLevel.Warning)
            self._webp_refused = True
            self._crop_body = None
            self._crop_token = None
            self._forget_seed()
            answer = self._post(self._build_body(
                point_coords, point_labels, mask_input, multimask_output,
                send_crop=True, name_seed=False), generation)
            self._refuse_late_answer(generation)

        if "error" in answer:
            # The transport hands a refusal back as data, not as a raise, so
            # this is where a server-side failure becomes a click failure. The
            # code travels with it, because an empty balance, a session that
            # signed out and a service fault each need their own sentence and
            # the text alone cannot tell them apart. Scrubbed: a refusal body
            # can echo the address it came from.
            code = str(answer.get("code") or "")
            detail = scrub_sensitive(str(answer.get("error") or code or "refused"))
            raise RefineRefusedError(f"Refine failed: {detail}", code=code)

        token = answer.get("crop_token")
        if isinstance(token, str) and token:
            self._crop_token = token

        masks, scores, low_res_masks = self._decode(answer)
        # Every answer states the side it works in, so one click is all it takes
        # for a hand-built seed to be the right size from then on.
        if low_res_masks.ndim == 3 and low_res_masks.shape[1] == low_res_masks.shape[2]:
            self.low_res_side = int(low_res_masks.shape[1])
        seed_id = answer.get("seed_id")
        if isinstance(seed_id, str) and seed_id:
            self._seed_id = seed_id
            self._seed_logits = low_res_masks
        else:
            self._forget_seed()
        # Built from values the decode above already checked, so there is no
        # catch here: a shape fault is the one thing worth hearing about, and a
        # bare except was swallowing exactly that.
        count, mask_h, mask_w = masks.shape
        _log(
            "Remote refine: {} masks {}x{}, {} points, {} ms total "
            "(far-side encode {} ms, decode {} ms, crop reused {})".format(
                count, mask_h, mask_w,
                0 if point_labels is None else int(np.size(point_labels)),
                int((time.monotonic() - started) * 1000),
                _as_ms(answer.get("encode_ms")),
                _as_ms(answer.get("decode_ms")),
                bool(answer.get("cached_crop")),
            )
        )
        return masks, scores, low_res_masks

    # -- request -----------------------------------------------------------

    def _build_body(
        self,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        mask_input: np.ndarray | None,
        multimask_output: bool,
        send_crop: bool,
        name_seed: bool,
    ) -> dict:
        crop = self._crop
        body: dict = {
            "crop": None,
            "crop_shape": None,
            # Raw pixels are what the far side assumes when nothing says
            # otherwise, so this line only ever moves to "png".
            "crop_format": "raw",
            "crop_token": None if send_crop else self._crop_token,
            "points": (
                [] if point_coords is None
                else np.asarray(point_coords, dtype=float).tolist()
            ),
            "labels": (
                [] if point_labels is None
                else np.asarray(point_labels, dtype=int).tolist()
            ),
            "mask_input": None,
            "mask_input_shape": None,
            "multimask_output": bool(multimask_output),
        }
        if send_crop and crop is not None:
            body["crop"], body["crop_format"] = self._encoded_crop(crop)
            # Kept on both branches: it costs nothing and the request stays
            # self-describing whichever form the pixels travelled in.
            body["crop_shape"] = list(crop.shape)
        if mask_input is not None:
            index = self._named_seed_index(mask_input) if name_seed else None
            if index is None:
                body["mask_input"] = pack_float16_payload(mask_input)
                body["mask_input_shape"] = list(mask_input.shape)
            else:
                body["seed_id"] = self._seed_id
                body["seed_index"] = index
        return body

    def _named_seed_index(self, mask_input: np.ndarray) -> int | None:
        """Which row of the last answer this seed is, or None if it is not one.

        The caller picks a row out of the logits the last answer returned, so
        the usual case is a hit. Comparing 65k floats costs microseconds next
        to uploading them, and a mismatch just means the pixels travel.
        """
        held = self._seed_logits
        if not self._seed_id or held is None or held.ndim != 3:
            return None
        if mask_input.shape[1:] != held.shape[1:]:
            return None
        for index in range(held.shape[0]):
            if np.array_equal(mask_input[0], held[index]):
                return index
        return None

    def _refuse_late_answer(self, generation: int) -> None:
        """Stop here when the crop moved while the answer was travelling.

        Called on the way back from every request. Nothing of the answer is
        read or stored past this point, so the session that replaced this one
        keeps its own crop token, its own logits and its own picture.
        """
        if generation != self._generation:
            raise RefineSupersededError(
                "The crop changed while its answer was on the way")

    def _post(self, body: dict, generation: int | None = None) -> dict:
        """One round trip. Never raises for a server-side refusal: the caller
        reads the code so it can retry a dropped crop before giving up.

        ``generation`` is the crop the request belongs to. The wait hands it a
        way to ask whether that is still the current one, and ends the round
        trip as soon as it is not.
        """
        if generation is None:
            generation = self._generation
        try:
            client = self._resolve_client()
            auth = self._resolve_auth()
            if self._client_accepts_cancel(client):
                answer = client.submit_refine(
                    body, auth,
                    cancel_check=lambda: self._generation != generation)
            else:
                # A stand-in client without the cancellable signature. Asked
                # BEFORE the call, never after a TypeError: a second attempt on
                # a request that may already have gone out is the one thing
                # this path must never do.
                answer = client.submit_refine(body, auth)
        except Exception as err:  # noqa: BLE001 -- reported as a click failure
            raise SamWorkerError(f"Refine request failed: {err}") from err
        if not isinstance(answer, dict):
            raise SamWorkerError("Refine answer was not readable")
        return answer

    def _client_accepts_cancel(self, client) -> bool:
        """Whether this client's refine call takes the cancellation hook.

        Resolved once per client and held. The answer cannot change for a given
        object, and reading a signature on every post was work the click paid
        for three times over.
        """
        if client is not self._cancel_client:
            self._cancel_client = client
            self._cancel_supported = _accepts_cancel_check(client.submit_refine)
        return self._cancel_supported

    def _resolve_client(self):
        if self._client is None:
            from ..api.terralab_client import TerraLabClient

            self._client = TerraLabClient()
        return self._client

    def _resolve_auth(self) -> dict:
        """The headers for THIS request.

        A callable is asked every time, so a key that changes during a long
        session reaches the very next click. Headers handed over as a dict are
        used as given, which is what a caller that resolved them itself asked
        for.
        """
        held = self._auth
        if callable(held):
            try:
                return held() or {}
            except Exception:  # noqa: BLE001 -- read them ourselves instead
                from .activation_manager import get_auth_header

                # Never written back: replacing the callable with one answer is
                # the frozen key this branch exists to avoid.
                return get_auth_header()
        if held is None:
            from .activation_manager import get_auth_header

            self._auth = get_auth_header()
            return self._auth
        return held

    # -- answer ------------------------------------------------------------

    def _decode(self, answer: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masks = self._decode_masks(answer)
        scores = np.array([float(s) for s in answer.get("scores") or []])
        low_res_masks = self._decode_low_res(answer)
        if masks.shape[0] != scores.shape[0]:
            raise SamWorkerError(
                f"Refine answer is inconsistent: {masks.shape[0]} masks against "
                f"{scores.shape[0]} scores")
        return masks, scores, low_res_masks

    def _decode_masks(self, answer: dict) -> np.ndarray:
        from .cloud_detection import decode_rle_to_mask

        shape = answer.get("masks_shape")
        rles = answer.get("masks")
        if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                or not isinstance(rles, (list, tuple))):
            raise SamWorkerError("Refine answer carried no usable masks")
        count, height, width = (int(shape[0]), int(shape[1]), int(shape[2]))
        if not 0 < count <= _MAX_MASK_COUNT or not 0 < height <= _MAX_MASK_SIDE \
                or not 0 < width <= _MAX_MASK_SIDE:
            raise SamWorkerError(f"Refine answer asked for an unusable mask shape {shape}")
        if len(rles) != count:
            raise SamWorkerError(
                f"Refine answer promised {count} masks and carried {len(rles)}")
        # The mask has to be the size of the crop we sent. Nothing downstream
        # checks it: the polygon builder reads its transform from the mask's own
        # shape, so a smaller mask is stretched across the whole crop and lands
        # as a plausible looking outline in the wrong place. Every helper that
        # would normally trim or merge the answer also gives up on a shape it
        # does not recognise, so this is the one answer that gets no guard at
        # all unless it is refused here.
        if self.original_size is not None and (height, width) != self.original_size:
            raise SamWorkerError(
                f"Refine answer is {height}x{width} for a {self.original_size[0]}"
                f"x{self.original_size[1]} crop")
        try:
            # Strict: an unreadable encoding decodes to an empty mask, and an
            # empty mask reads on screen as "no object found here". A protocol
            # failure that looks like a modelling result is never reported.
            return np.stack([
                decode_rle_to_mask(rle, height, width, strict=True) for rle in rles])
        except ValueError as err:
            raise SamWorkerError("Refine answer's masks were unreadable") from err

    def _decode_low_res(self, answer: dict) -> np.ndarray:
        shape = answer.get("low_res_masks_shape")
        payload = answer.get("low_res_masks")
        if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                or not isinstance(payload, str) or not payload):
            raise SamWorkerError("Refine answer carried no usable mask logits")
        dims = tuple(int(v) for v in shape)
        if not 0 < dims[0] <= _MAX_MASK_COUNT or any(
                not 0 < v <= _MAX_MASK_SIDE for v in dims[1:]):
            raise SamWorkerError(
                f"Refine answer asked for an unusable logit shape {shape}")
        try:
            return unpack_float16_payload(payload, dims)
        except Exception as err:  # noqa: BLE001 -- a bad payload is a failed click
            raise SamWorkerError(f"Refine answer's mask logits were unreadable: {err}") from err
