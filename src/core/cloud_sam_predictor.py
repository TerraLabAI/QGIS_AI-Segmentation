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
  This side remembers the last few crop tokens, not just the current one, so a
  user who works two objects and comes back to the first neighbourhood names
  that crop instead of uploading it a second time.

Manual mode never comes here. It stays on-device and offline by product rule.

Only the venv-free part of numpy is used, so this module is importable with no
local install at all.
"""
from __future__ import annotations

import base64
import hashlib
import time
import zlib
from collections import OrderedDict

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .click_crop_encoding import crop_webp_allowed, encode_crop_png, encode_crop_webp
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
    """Content key for a crop, so a token is dropped when the pixels move.

    A digest rather than a comparison: the previous crop is not kept once its
    token is held, and reading 3 MB is fast next to any network call.

    Wide on purpose. A 32-bit checksum meets itself after tens of thousands of
    crops, and what a meeting costs here is not an error but a wrong answer:
    two different pictures would share one name, so the far side would be asked
    to segment the picture it is still holding rather than the one on screen.
    128 bits puts that past any number of crops a session can produce.
    """
    return (image_np.shape,
            hashlib.blake2b(image_np.tobytes(), digest_size=16).hexdigest())


# How many crops the session remembers a token for. The user works one object,
# moves to the next and comes back, and the grid cell is a fraction of the crop
# width, so the neighbourhood behind an object is reached again and again. The
# far side still holds those pixels, so a remembered name saves the upload
# whole. Small on purpose: each entry is a short string, but a longer memory
# only names crops the far side has had time to drop.
_CROP_TOKEN_MEMORY = 8


class CloudSamPredictor:
    """The click path's predictor, answered by the remote refine route."""

    def __init__(self, client=None, auth=None, on_remote_answer=None) -> None:
        """``auth`` is either the headers themselves or something that returns
        them. Hand over the callable whenever the session can outlive one key:
        a dict is read once and never again, so a key that rotates mid-session
        leaves every later click failing over to the machine.

        ``on_remote_answer`` fires once per click the network really answered,
        same contract as ``CloudFirstPredictor``. A billed lane hangs its charge
        on it, so a click that raised on its way must never fire it."""
        self._client = client
        self._auth = auth
        self._on_remote_answer = on_remote_answer
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
        # One name per crop the far side is still holding, oldest first. Keyed
        # on the pixels themselves, so a user who leaves a neighbourhood and
        # comes back finds the name of the crop already up there instead of
        # sending it a second time. Names only, never the bodies they stand
        # for: eight encoded crops would be tens of megabytes of memory.
        self._crop_tokens: OrderedDict[tuple, str] = OrderedDict()
        # Fingerprint of the account the held names were earned under.
        # None until the first request resolves one.
        self._auth_fingerprint: str | None = None
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
        # Every name goes, not just this crop's. A session end, a predictor
        # swap and a sign-out all come through here, and a name filed under one
        # account must never be sent under the next one.
        self._crop_tokens.clear()
        self._crop_body = None
        self._forget_seed()

    @staticmethod
    def _auth_print(auth) -> str:
        """A fingerprint of the headers, never the key itself."""
        try:
            raw = "\x1f".join(f"{k}={v}" for k, v in sorted((auth or {}).items()))
        except Exception:  # noqa: BLE001 -- unreadable auth is its own identity
            raw = ""
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _auth_changed(self, auth) -> bool:
        """Whether these headers belong to a different account than the names
        this session is holding. False the first time, which only records it."""
        seen = self._auth_print(auth)
        if self._auth_fingerprint is None:
            self._auth_fingerprint = seen
            return False
        if seen == self._auth_fingerprint:
            return False
        self._auth_fingerprint = seen
        return True

    def _forget_tokens_of_another_account(self) -> None:
        """Drop every name the moment the account behind them changes.

        A name is what the far side calls one picture, and it is honoured on
        the strength of the name alone. The session outlives a sign-out, and
        the same predictor can be asked to click again under a different key,
        so without this a name earned by one account would travel under the
        next one.
        """
        try:
            auth = self._resolve_auth()
        except Exception:  # noqa: BLE001 -- unreadable auth means forget them
            auth = None
        if self._auth_changed(auth):
            self._crop_tokens.clear()
            self._forget_seed()

    def _held_crop_token(self) -> str | None:
        """The far side's name for the crop the session is on, or None.

        Reading one keeps it: the entry moves to the young end, so the crops a
        user keeps coming back to are the last ones the memory gives up.
        """
        self._forget_tokens_of_another_account()
        key = self._crop_key
        if key is None:
            return None
        token = self._crop_tokens.get(key)
        if token is not None:
            self._crop_tokens.move_to_end(key)
        return token

    def _hold_crop_token(self, key: tuple | None, token: str,
                         generation: int | None = None) -> None:
        """File the far side's name for one crop, dropping the oldest over the
        ceiling.

        ``generation`` is the session this name was earned in. Checked HERE,
        at the write, not only before it: a session end can land between a
        caller's own check and this call, and a name filed after that belongs
        to a session that no longer exists.
        """
        if key is None or not token:
            return
        if generation is not None and generation != self._generation:
            return
        self._crop_tokens[key] = token
        self._crop_tokens.move_to_end(key)
        while len(self._crop_tokens) > _CROP_TOKEN_MEMORY:
            self._crop_tokens.popitem(last=False)

    def _drop_crop_token(self) -> None:
        """Forget the name of the crop the session is on.

        Called when the far side has said it no longer honours it, so the next
        request carries the pixels. Only this crop's entry goes: the other
        names are still good.
        """
        key = self._crop_key
        if key is not None:
            self._crop_tokens.pop(key, None)

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
            # crop is in place. The names stay, the encoded body does not: the
            # far side is still holding those crops, and one of them may well
            # be this one.
            self._generation += 1
            self._crop_body = None
            self._forget_seed()
        self._crop = image_np
        self._crop_key = key
        self.original_size = (int(image_np.shape[0]), int(image_np.shape[1]))
        self.is_image_set = True
        if self._held_crop_token() is None:
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
        # it is not or when this Qt cannot write it. PNG is always the caller's
        # own pixels; webp is too until the server asks for a lighter one.
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
        key = self._crop_key
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
                # The crop changed while its pixels were on their way. The
                # session has moved, and filing this name now would put it in
                # a memory the user has left, or in the next account's.
                return
            token = (answer or {}).get("crop_token")
            if not isinstance(token, str) or not token:
                raise ValueError((answer or {}).get("error") or "no token")
            self._hold_crop_token(key, token, generation)
            _log("Remote refine: crop sent ahead of the click, {} KB in {} ms".format(
                len(body["crop"]) // 1024, int((time.monotonic() - started) * 1000)))
        except Exception as err:  # noqa: BLE001 -- the click sends the crop itself
            if generation == self._generation:
                self._drop_crop_token()
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
            send_crop=self._held_crop_token() is None, name_seed=True), generation)
        self._refuse_late_answer(generation)

        if answer.get("code") in (CROP_EXPIRED_CODE, SEED_EXPIRED_CODE):
            # A name the far side no longer honours. Nothing the user did, and
            # nothing they should read: send the thing itself and try once more.
            # A lost crop costs the picture, a lost seed only the logits.
            crop_gone = answer.get("code") == CROP_EXPIRED_CODE
            if crop_gone:
                self._drop_crop_token()
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
                send_crop=self._held_crop_token() is None, name_seed=False),
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
            self._drop_crop_token()
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
            # Every path above has already refused a late answer. The
            # session is named again at the write anyway, because a teardown
            # can land between that refusal and this line.
            self._hold_crop_token(self._crop_key, token, generation)

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
        # The network answered. Noted here, past every decode check, so a click
        # that raised on its way is never counted as one the remote side served.
        if self._on_remote_answer is not None:
            try:
                self._on_remote_answer()
            except Exception:  # nosec B110 -- a note never costs a click
                pass
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
            "crop_token": None if send_crop else self._held_crop_token(),
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
            # Checked against the headers THIS request will carry, not against
            # a reading taken earlier. The body was built with whatever name
            # was held then, and an account that changed in between would send
            # one account's name under the other's key. Nothing goes out on a
            # changed account: the click ends here and the next one starts
            # clean, which is the only honest answer to signing out mid-click.
            if self._auth_changed(auth):
                self._crop_tokens.clear()
                self._forget_seed()
                raise RefineSupersededError(
                    "the account changed while the click was being sent")
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
