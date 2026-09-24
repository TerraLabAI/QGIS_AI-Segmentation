


























from __future__ import annotations

import base64
import hashlib
import math
import threading
import time
import zlib
from collections import OrderedDict

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .click_crop_encoding import (
    crop_png_preferred,
    crop_webp_allowed,
    encode_crop_png,
    encode_crop_webp,
    note_crop_upload,
)
from .click_phase_clock import active_click_clock
from .log_scrub import scrub_sensitive
from .sam_predictor import SamWorkerError




CROP_EXPIRED_CODE = "CROP_EXPIRED"




SEED_EXPIRED_CODE = "SEED_EXPIRED"




INVALID_INPUT_CODE = "INVALID_INPUT"




INVALID_REQUEST_CODE = "INVALID_REQUEST"




_CLICK_RETRIES_MAX = 2


def _click_retries_max() -> int:

    from .server_dials import dial_in_range

    return int(dial_in_range(
        "tuning.click.retries_max", _CLICK_RETRIES_MAX, 0, 3))




EMPTY_RESULT_CODE = "EMPTY_RESULT"



MAX_REFINE_POINTS = 64






LOW_RES_NAMED = "named"




_NAMED_SEED_MEMORY = 12


def _named_seed_memory() -> int:
    from .server_dials import dial_in_range

    return dial_in_range("tuning.click.named_seed_memory", _NAMED_SEED_MEMORY, 1, 64)







_PREVIEW_SEED_MEMORY = 8
_preview_seeds: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
_preview_seeds_lock = threading.Lock()


def _preview_seed_memory() -> int:
    from .server_dials import dial_in_range

    return dial_in_range("tuning.hover.preview_seed_memory", _PREVIEW_SEED_MEMORY, 1, 64)





_STAND_IN_LOGIT = 6.0


def _stand_in_logit() -> float:
    from .server_dials import dial_in_range

    return dial_in_range("tuning.click.stand_in_logit", _STAND_IN_LOGIT, 1.0, 20.0)




REFUSAL_CREDITS = "CREDITS"
REFUSAL_SIGN_IN = "SIGN_IN"
REFUSAL_EMPTY = "EMPTY"
REFUSAL_OTHER = "OTHER"


def click_refusal_class(code: str) -> str:






    named = (code or "").strip().upper()
    if not named:
        return REFUSAL_OTHER


    if named == EMPTY_RESULT_CODE:
        return REFUSAL_EMPTY
    try:
        from .error_policy import EXHAUSTED_CODES, RUN_FATAL_CODES

        if named in EXHAUSTED_CODES:
            return REFUSAL_CREDITS
        if named in RUN_FATAL_CODES:
            return REFUSAL_SIGN_IN
    except Exception:  # noqa: BLE001
        return REFUSAL_OTHER
    return REFUSAL_OTHER


class RefineRefusedError(SamWorkerError):








    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.code = code

    def refusal_class(self) -> str:


        return click_refusal_class(self.code)


class RefineSupersededError(SamWorkerError):
    pass












_MAX_MASK_SIDE = 8192
_MAX_MASK_COUNT = 8
_MAX_LOGIT_BYTES = 64 * 1024 * 1024


def _log(message: str, level=Qgis.MessageLevel.Info) -> None:
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


def _as_ms(value) -> int:

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    try:
        return max(0, int(value)) if math.isfinite(value) else 0
    except (OverflowError, ValueError):
        return 0


def capped_refine_points(points: list, labels: list) -> tuple[list, list]:









    if len(points) != len(labels):
        raise ValueError("Every refine point must have one label")
    if len(points) <= MAX_REFINE_POINTS:
        return points, labels
    pairs = list(zip(points, labels))
    positives = [pair for pair in pairs if pair[1]]
    if len(positives) >= MAX_REFINE_POINTS:
        kept = positives[-MAX_REFINE_POINTS:]
    else:
        negatives = [pair for pair in pairs if not pair[1]]
        kept = positives + negatives[-(MAX_REFINE_POINTS - len(positives)):]
    return [pair[0] for pair in kept], [pair[1] for pair in kept]


def pack_float16_payload(array: np.ndarray) -> str:






    with np.errstate(over="ignore", invalid="ignore"):
        half = np.ascontiguousarray(array, dtype=np.float16)
    if not np.isfinite(half).all():
        raise ValueError("Mask seed must contain finite float16 values")
    return base64.b64encode(zlib.compress(half.tobytes(), 6)).decode("ascii")


def unpack_float16_payload(payload: str, shape: tuple[int, ...]) -> np.ndarray:





    if (not isinstance(shape, (tuple, list)) or len(shape) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                   for v in shape)):
        raise ValueError("Mask logits need three integer dimensions")
    if (not 0 < shape[0] <= _MAX_MASK_COUNT
            or any(not 0 < v <= _MAX_MASK_SIDE for v in shape[1:])):
        raise ValueError("Mask logit dimensions are out of bounds")
    expected = math.prod(shape) * np.dtype(np.float16).itemsize
    if expected > _MAX_LOGIT_BYTES:
        raise ValueError("Mask logit array is too large")


    if not isinstance(payload, str) or len(payload) > 4 * ((expected + expected // 1000 + 128 + 2) // 3):
        raise ValueError("Mask logit payload is too large")
    packed = base64.b64decode(payload.encode("ascii"), validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(packed, expected + 1)
    if (len(raw) != expected or not decoder.eof
            or decoder.unconsumed_tail or decoder.unused_data):
        raise ValueError("Mask logit payload does not match its declared shape")
    flat = np.frombuffer(raw, dtype=np.float16)
    if not np.isfinite(flat).all():
        raise ValueError("Mask logits must be finite")
    return np.ascontiguousarray(flat.reshape(shape), dtype=np.float32)


def mask_stand_in_logits(masks: np.ndarray, side: int) -> np.ndarray:









    stack = np.asarray(masks)
    if stack.ndim == 2:
        stack = stack[None]
    if stack.ndim != 3 or not 0 < side <= _MAX_MASK_SIDE:
        raise ValueError("Stand-in logits need (count, H, W) masks and a side")
    height, width = int(stack.shape[1]), int(stack.shape[2])
    rows = np.clip((np.arange(side) * height // side), 0, height - 1)
    cols = np.clip((np.arange(side) * width // side), 0, width - 1)
    sampled = stack[:, rows[:, None], cols[None, :]].astype(bool)
    logit = _stand_in_logit()
    return np.where(sampled, logit, -logit).astype(np.float32)


def note_preview_seed(crop_token: str, seed_id: str, stand_in: np.ndarray) -> None:

    if not crop_token or not seed_id or stand_in is None:
        return
    with _preview_seeds_lock:
        _preview_seeds[(crop_token, seed_id)] = stand_in
        _preview_seeds.move_to_end((crop_token, seed_id))
        while len(_preview_seeds) > _preview_seed_memory():
            _preview_seeds.popitem(last=False)


def forget_preview_seeds() -> None:

    with _preview_seeds_lock:
        _preview_seeds.clear()


def _preview_seed_named(crop_token: str | None,
                        mask_input: np.ndarray) -> str | None:

    if not crop_token:
        return None
    with _preview_seeds_lock:
        held = [(key[1], value) for key, value in _preview_seeds.items()
                if key[0] == crop_token]
    for seed_id, stand_in in reversed(held):
        if stand_in.shape == mask_input.shape and np.array_equal(mask_input, stand_in):
            return seed_id
    return None


def _accepts_cancel_check(call) -> bool:






    try:
        import inspect

        return "cancel_check" in inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False


def _crop_identity(image_np: np.ndarray) -> tuple:











    return (image_np.shape,
            hashlib.blake2b(image_np.tobytes(), digest_size=16).hexdigest())








_CROP_TOKEN_MEMORY = 8


def _crop_token_memory() -> int:
    from .server_dials import dial_in_range

    return dial_in_range("tuning.click.crop_token_memory", _CROP_TOKEN_MEMORY, 1, 64)





SERVICE_LOW_RES_SIDE = 288



_service_low_res_side = SERVICE_LOW_RES_SIDE


def _note_service_side(side: int) -> None:
    global _service_low_res_side
    _service_low_res_side = side


class CloudSamPredictor:


    def __init__(self, client=None, auth=None, on_remote_answer=None,
                 session_id: str | None = None) -> None:












        self._client = client
        self._auth = auth
        self._on_remote_answer = on_remote_answer
        self.session_id: str | None = session_id



        self._cancel_client = None
        self._cancel_supported = False
        self.is_image_set = False
        self.original_size: tuple[int, int] | None = None



        self.input_size = None
        self._crop: np.ndarray | None = None
        self._crop_key: tuple | None = None





        self._crop_tokens: OrderedDict[tuple, str] = OrderedDict()


        self._auth_fingerprint: str | None = None






        self._crop_body: tuple[str, str] | None = None



        self._webp_refused = False




        self._named_seeds: OrderedDict[str, np.ndarray] = OrderedDict()






        self.low_res_side: int | None = _service_low_res_side


        self.last_answer_was_remote = True



        self._generation = 0
        self._register_lock = threading.Lock()
        self._registering: set = set()



    def warm_up(self) -> bool:

        return True

    def session_generation(self) -> int:



        return self._generation

    def set_session_id(self, session_id: str | None) -> None:



        self.session_id = session_id

    def reset_image(self) -> None:
        self._generation += 1
        self.is_image_set = False
        self.original_size = None
        self._crop = None
        self._crop_key = None



        self._crop_tokens.clear()
        self._crop_body = None
        self._forget_seed()
        forget_preview_seeds()

    @staticmethod
    def _auth_print(auth) -> str:

        try:
            raw = "\x1f".join(f"{k}={v}" for k, v in sorted((auth or {}).items()))
        except Exception:  # noqa: BLE001
            raw = ""
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _auth_changed(self, auth) -> bool:


        seen = self._auth_print(auth)
        if self._auth_fingerprint is None:
            self._auth_fingerprint = seen
            return False
        if seen == self._auth_fingerprint:
            return False
        self._auth_fingerprint = seen
        return True

    def _forget_tokens_of_another_account(self) -> None:








        try:
            auth = self._resolve_auth()
        except Exception:  # noqa: BLE001
            auth = None
        if self._auth_changed(auth):
            self._crop_tokens.clear()
            self._forget_seed()
            forget_preview_seeds()

    def _held_crop_token(self) -> str | None:





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








        if key is None or not token:
            return
        if generation is not None and generation != self._generation:
            return
        self._crop_tokens[key] = token
        self._crop_tokens.move_to_end(key)
        while len(self._crop_tokens) > _crop_token_memory():
            self._crop_tokens.popitem(last=False)

    def _drop_crop_token(self) -> None:






        key = self._crop_key
        if key is not None:
            self._crop_tokens.pop(key, None)

    def _forget_seed(self) -> None:




        self._named_seeds = OrderedDict()

    def _remember_named_seed(self, seed_id: str, logits: np.ndarray) -> None:

        held = self._named_seeds
        held[seed_id] = logits
        held.move_to_end(seed_id)
        while len(held) > _named_seed_memory():
            held.popitem(last=False)

    def cleanup(self) -> None:
        self.reset_image()

    def set_image(self, image_np: np.ndarray) -> None:








        if (not isinstance(image_np, np.ndarray) or image_np.ndim != 3
                or image_np.shape[2] != 3):
            shape = getattr(image_np, "shape", None)
            raise SamWorkerError(
                f"Invalid image for encoding: expected (H, W, 3), got shape {shape}")
        if image_np.shape[0] == 0 or image_np.shape[1] == 0:
            raise SamWorkerError(
                f"Invalid image for encoding: empty crop {image_np.shape}")
        if image_np.dtype != np.uint8:
            if not np.issubdtype(image_np.dtype, np.number) or not np.isfinite(image_np).all():
                raise SamWorkerError("Invalid image for encoding: pixels must be finite numbers")
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        image_np = np.array(image_np, dtype=np.uint8, order="C", copy=True)

        key = _crop_identity(image_np)
        if key != self._crop_key:





            self._generation += 1
            self._crop_body = None
            self._forget_seed()
        self._crop = image_np
        self._crop_key = key
        self.original_size = (int(image_np.shape[0]), int(image_np.shape[1]))
        self.is_image_set = True
        if self._held_crop_token() is None:
            self._register_crop(image_np)

    def hover_preview_handle(self) -> tuple[str, tuple[int, int]] | None:







        if not self.is_image_set or self._crop is None:
            return None
        size = self.original_size
        if size is None:
            return None
        token = self._held_crop_token()
        if not token:
            return None
        return token, (int(size[0]), int(size[1]))

    def _encoded_crop(self, crop: np.ndarray) -> tuple[str, str]:






        held = self._crop_body
        if held is not None:
            return held
        started = time.monotonic()



        pixels = int(crop.shape[0]) * int(crop.shape[1]) if crop.ndim == 3 else 0
        packed = (encode_crop_webp(crop)
                  if crop_webp_allowed() and not self._webp_refused
                  and not crop_png_preferred(pixels) else None)
        if packed is not None:
            made = (base64.b64encode(packed).decode("ascii"), "webp")
        else:
            png = encode_crop_png(crop)
            if png is None:


                made = (base64.b64encode(crop.tobytes()).decode("ascii"), "raw")
            else:
                made = (base64.b64encode(png).decode("ascii"), "png")
        self._crop_body = made


        _log(f"Remote refine: crop packed as {made[1]} ({len(made[0]) // 1024} KB) "
             f"in {int((time.monotonic() - started) * 1000)} ms")
        return made

    def _billing_fields(self) -> dict:



        from .request_context import plugin_version

        return {"session_id": self.session_id, "plugin_version": plugin_version()}

    def _register_crop(self, crop: np.ndarray) -> None:





        generation, key = self._generation, self._crop_key
        if crop is not self._crop:
            return
        claim = (generation, key)
        with self._register_lock:
            if claim in self._registering:
                return
            self._registering.add(claim)
        try:
            self._register_crop_once(crop, generation, key)
        finally:
            with self._register_lock:
                self._registering.discard(claim)

    def _register_crop_once(self, crop, generation, key) -> None:
        started = time.monotonic()
        try:
            if generation != self._generation:
                return
            payload, form = self._encoded_crop(crop)
            body = {"crop": payload, "crop_format": form,
                    "crop_shape": list(crop.shape), **self._billing_fields()}
            auth = self._resolve_auth()
            auth_fingerprint = self._auth_print(auth)


            sent_at = time.monotonic()
            answer = self._resolve_client().submit_refine_register(
                body, auth)
            upload_s = time.monotonic() - sent_at
            if generation != self._generation or self._auth_print(self._resolve_auth()) != auth_fingerprint:
                return
            if (form == "webp" and generation == self._generation
                    and (answer or {}).get("code") == INVALID_REQUEST_CODE):





                _log("Remote refine: the server refused the webp crop, "
                     "sending it as png from now on", Qgis.MessageLevel.Warning)
                self._webp_refused = True
                self._crop_body = None
                payload, form = self._encoded_crop(crop)
                body = {"crop": payload, "crop_format": form,
                        "crop_shape": list(crop.shape), **self._billing_fields()}
                sent_at = time.monotonic()
                answer = self._resolve_client().submit_refine_register(
                    body, auth)
                upload_s = time.monotonic() - sent_at
            if generation != self._generation or self._auth_print(self._resolve_auth()) != auth_fingerprint:



                return
            token = (answer or {}).get("crop_token")
            if not isinstance(token, str) or not token:
                raise ValueError((answer or {}).get("error") or "no token")
            note_crop_upload(len(body["crop"]), upload_s)
            self._hold_crop_token(key, token, generation)
            _log("Remote refine: crop sent ahead of the click, {} KB in {} ms".format(
                len(body["crop"]) // 1024, int((time.monotonic() - started) * 1000)))
        except Exception as err:  # noqa: BLE001
            if generation == self._generation:
                self._drop_crop_token()


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
        if mask_input is not None:
            mask_input = np.asarray(mask_input)
        if mask_input is not None and (
                mask_input.ndim != 3 or mask_input.shape[0] != 1
                or any(side <= 0 or side > _MAX_MASK_SIDE for side in mask_input.shape[1:])):
            raise ValueError(
                "Invalid mask seed for prediction: expected (1, H, W), "
                f"got shape {tuple(mask_input.shape)}")

        started = time.monotonic()
        generation = self._generation




        retries_left = _click_retries_max()
        seed = mask_input




        seed_resend_owed = True

        def resend_if_expired(answer: dict) -> dict:



            nonlocal retries_left, seed_resend_owed
            while answer.get("code") in (CROP_EXPIRED_CODE, SEED_EXPIRED_CODE):
                if retries_left > 0:
                    retries_left -= 1
                elif seed_resend_owed and answer.get("code") == SEED_EXPIRED_CODE:
                    seed_resend_owed = False
                else:
                    break
                crop_gone = answer.get("code") == CROP_EXPIRED_CODE
                if crop_gone:
                    self._drop_crop_token()
                self._forget_seed()
                answer = self._post(self._build_body(
                    point_coords, point_labels, seed, multimask_output,
                    send_crop=crop_gone, name_seed=False), generation)
                self._refuse_late_answer(generation)
            return answer

        answer = self._post(self._build_body(
            point_coords, point_labels, seed, multimask_output,
            send_crop=self._held_crop_token() is None, name_seed=True), generation)
        self._refuse_late_answer(generation)
        answer = resend_if_expired(answer)

        if (retries_left > 0 and seed is not None
                and answer.get("code") == INVALID_INPUT_CODE):
            retries_left -= 1
            seed = None













            answer = self._post(self._build_body(
                point_coords, point_labels, None, multimask_output,
                send_crop=self._held_crop_token() is None, name_seed=False),
                generation)
            self._refuse_late_answer(generation)



            answer = resend_if_expired(answer)

        if (retries_left > 0
                and answer.get("code") == INVALID_REQUEST_CODE
                and self._crop_body is not None
                and self._crop_body[1] == "webp"):
            retries_left -= 1





            _log("Remote refine: the server refused the webp crop, the click "
                 "retries as png", Qgis.MessageLevel.Warning)
            self._webp_refused = True
            self._crop_body = None
            self._drop_crop_token()
            self._forget_seed()
            answer = self._post(self._build_body(
                point_coords, point_labels, seed, multimask_output,
                send_crop=True, name_seed=False), generation)
            self._refuse_late_answer(generation)

        if "error" in answer:






            code = str(answer.get("code") or "")
            detail = scrub_sensitive(str(answer.get("error") or code or "refused"))
            raise RefineRefusedError(f"Refine failed: {detail}", code=code)

        masks, scores, low_res_masks = self._decode(answer)
        self._refuse_late_answer(generation)
        token = answer.get("crop_token")
        if isinstance(token, str) and token:



            self._hold_crop_token(self._crop_key, token, generation)



        if low_res_masks.ndim == 3 and low_res_masks.shape[1] == low_res_masks.shape[2]:
            self.low_res_side = int(low_res_masks.shape[1])
            _note_service_side(self.low_res_side)
        seed_id = answer.get("seed_id")
        if isinstance(seed_id, str) and seed_id:
            self._remember_named_seed(seed_id, low_res_masks)
        else:
            self._forget_seed()



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


        if self._on_remote_answer is not None:
            try:
                self._on_remote_answer()
            except Exception:  # nosec B110
                pass
        return masks, scores, low_res_masks



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
        coords = np.empty((0, 2), dtype=float) if point_coords is None else np.asarray(point_coords, dtype=float)
        label_values = np.empty((0,), dtype=int) if point_labels is None else np.asarray(point_labels)
        if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
            raise ValueError("Refine points must be finite coordinate pairs")
        if (label_values.ndim != 1 or label_values.shape[0] != coords.shape[0]
                or not np.isin(label_values, (0, 1)).all()):
            raise ValueError("Every refine point must have a binary label")
        points, labels = capped_refine_points(coords.tolist(), label_values.astype(int).tolist())
        body: dict = {
            "crop": None,
            "crop_shape": None,


            "crop_format": "raw",
            "crop_token": None if send_crop else self._held_crop_token(),
            "points": points,
            "labels": labels,
            "mask_input": None,
            "mask_input_shape": None,
            "multimask_output": bool(multimask_output),
            "low_res": LOW_RES_NAMED,
            **self._billing_fields(),
        }
        if send_crop and crop is not None:
            body["crop"], body["crop_format"] = self._encoded_crop(crop)


            body["crop_shape"] = list(crop.shape)
        if mask_input is not None:
            named = self._named_seed_for(mask_input) if name_seed else None
            if named is None:
                body["mask_input"] = pack_float16_payload(mask_input)
                body["mask_input_shape"] = list(mask_input.shape)
            else:
                body["seed_id"], body["seed_index"] = named
        return body

    def _named_seed_for(self, mask_input: np.ndarray) -> tuple[str, int] | None:








        if mask_input.ndim != 3 or mask_input.shape[0] != 1:
            return None
        for seed_id, held in reversed(list(self._named_seeds.items())):
            if held.ndim != 3 or mask_input.shape[1:] != held.shape[1:]:
                continue
            for index in range(held.shape[0]):
                if np.array_equal(mask_input[0], held[index]):
                    return seed_id, index
        token = self._crop_tokens.get(self._crop_key) if self._crop_key else None
        seed_id = _preview_seed_named(token, mask_input)
        return None if seed_id is None else (seed_id, 0)

    def _refuse_late_answer(self, generation: int) -> None:






        if generation != self._generation:
            raise RefineSupersededError(
                "The crop changed while its answer was on the way")

    def _post(self, body: dict, generation: int | None = None) -> dict:







        if generation is None:
            generation = self._generation
        try:
            client = self._resolve_client()
            auth = self._resolve_auth()






            if self._auth_changed(auth):
                self._crop_tokens.clear()
                self._forget_seed()
                forget_preview_seeds()
                raise RefineSupersededError(
                    "the account changed while the click was being sent")
            if self._client_accepts_cancel(client):
                answer = client.submit_refine(
                    body, auth,
                    cancel_check=lambda: self._generation != generation)
            else:




                answer = client.submit_refine(body, auth)
        except RefineSupersededError:
            raise
        except Exception as err:  # noqa: BLE001
            raise SamWorkerError(f"Refine request failed: {err}") from err
        if self._auth_print(self._resolve_auth()) != self._auth_print(auth):
            self._crop_tokens.clear()
            self._forget_seed()
            forget_preview_seeds()
            self._generation += 1
            raise RefineSupersededError("the account changed while the click was being answered")
        if not isinstance(answer, dict):
            raise SamWorkerError("Refine answer was not readable")
        clock = active_click_clock()
        if clock is not None:
            clock.note_server_ms(answer.get("total_ms"))
        return answer

    def _client_accepts_cancel(self, client) -> bool:






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







        held = self._auth
        if callable(held):
            try:
                return held() or {}
            except Exception:  # noqa: BLE001
                from .activation_manager import get_auth_header



                return get_auth_header()
        if held is None:
            from .activation_manager import get_auth_header

            return get_auth_header()
        return held



    def _decode(self, answer: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masks = self._decode_masks(answer)
        try:
            scores = np.asarray(answer.get("scores"), dtype=float)
        except (TypeError, ValueError, OverflowError) as err:
            raise SamWorkerError("Refine answer carried unreadable scores") from err
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise SamWorkerError("Refine answer carried non-finite or non-vector scores")
        low_res_masks = self._decode_low_res(answer, masks)
        if low_res_masks.shape[0] != masks.shape[0]:
            raise SamWorkerError("Refine answer's logits do not match its masks")
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
        if any(isinstance(v, bool) or not isinstance(v, int) for v in shape):
            raise SamWorkerError("Refine mask dimensions must be integers")
        count, height, width = shape
        if not 0 < count <= _MAX_MASK_COUNT or not 0 < height <= _MAX_MASK_SIDE \
                or not 0 < width <= _MAX_MASK_SIDE:
            raise SamWorkerError(f"Refine answer asked for an unusable mask shape {shape}")
        if len(rles) != count:
            raise SamWorkerError(
                f"Refine answer promised {count} masks and carried {len(rles)}")







        if self.original_size is not None and (height, width) != self.original_size:
            raise SamWorkerError(
                f"Refine answer is {height}x{width} for a {self.original_size[0]}"
                f"x{self.original_size[1]} crop")
        try:



            return np.stack([
                decode_rle_to_mask(rle, height, width, strict=True) for rle in rles])
        except ValueError as err:
            raise SamWorkerError("Refine answer's masks were unreadable") from err

    def _decode_low_res(self, answer: dict, masks: np.ndarray | None = None) -> np.ndarray:








        shape = answer.get("low_res_masks_shape")
        payload = answer.get("low_res_masks")
        named = (payload is None and masks is not None
                 and isinstance(answer.get("seed_id"), str) and answer.get("seed_id"))
        if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                or not (named or (isinstance(payload, str) and payload))):
            raise SamWorkerError("Refine answer carried no usable mask logits")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in shape):
            raise SamWorkerError("Refine logit dimensions must be integers")
        dims = tuple(shape)
        if not 0 < dims[0] <= _MAX_MASK_COUNT or any(
                not 0 < v <= _MAX_MASK_SIDE for v in dims[1:]):
            raise SamWorkerError(
                f"Refine answer asked for an unusable logit shape {shape}")
        if named and masks is not None:
            if dims[1] != dims[2]:
                raise SamWorkerError(
                    f"Refine answer named logits of an unusable shape {shape}")
            return mask_stand_in_logits(masks, dims[1])
        try:
            return unpack_float16_payload(payload, dims)
        except Exception as err:  # noqa: BLE001
            raise SamWorkerError(f"Refine answer's mask logits were unreadable: {err}") from err
