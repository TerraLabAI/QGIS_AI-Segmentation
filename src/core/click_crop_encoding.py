














from __future__ import annotations

import math
import threading
import time

import numpy as np
from qgis.core import Qgis, QgsMessageLog












_PNG_EFFORT = 70



_PNG_EFFORT_MIN = 0
_PNG_EFFORT_MAX = 100







_WEBP_QUALITY_LOSSLESS = 100





_WEBP_QUALITY_MIN = 60
_WEBP_QUALITY_MAX = 100







_WEBP_LOSSLESS_EFFORT = 20











_PNG_UPLINK_FLOOR = 750.0




_UPLINK_FLOOR_MIN = 0.001
_UPLINK_FLOOR_MAX = 1_000_000.0




_UPLINK_HYSTERESIS = 0.25




_UPLINK_SMOOTHING = 0.4




_UPLINK_MIN_SAMPLE_BYTES = 64 * 1024





_link_lock = threading.Lock()
_link_kbytes_s: float | None = None
_link_prefers_png = False




_WIRE_EXPANSION = 4.0 / 3.0






_pack_lock = threading.Lock()
_pack_cost: dict[str, tuple[float, float]] = {}



_PACK_HYSTERESIS = 0.10




_PROFILE_SETTING = "TerraLab/click_crop_profile"







_REMEMBERED_UPLINK_DISCOUNT = 0.5


def _uplink_hysteresis() -> float:


    from .server_dials import dial_in_range

    return dial_in_range(
        "tuning.click.uplink_hysteresis_fraction", _UPLINK_HYSTERESIS, 0.0, 1.0)


def _uplink_smoothing() -> float:

    from .server_dials import dial_in_range

    return dial_in_range(
        "tuning.click.uplink_smoothing_fraction", _UPLINK_SMOOTHING, 0.0, 1.0)


def _pack_hysteresis() -> float:


    from .server_dials import dial_in_range

    return dial_in_range(
        "tuning.click.pack_hysteresis_fraction", _PACK_HYSTERESIS, 0.0, 1.0)


def _remembered_uplink_discount() -> float:

    from .server_dials import dial_in_range

    return dial_in_range(
        "tuning.click.remembered_uplink_discount", _REMEMBERED_UPLINK_DISCOUNT,
        0.0, 1.0)


def _webp_lossless_effort() -> int:

    from .server_dials import dial_in_range

    return int(dial_in_range(
        "tuning.click.webp_lossless_effort", _WEBP_LOSSLESS_EFFORT, 0, 100))


_profile_state = {"loaded": False}


def _load_crop_profile() -> None:








    global _link_kbytes_s
    if _profile_state["loaded"]:
        return
    _profile_state["loaded"] = True
    try:
        import json

        from qgis.core import QgsSettings

        raw = QgsSettings().value(_PROFILE_SETTING, "")
        held = json.loads(raw) if raw else {}
        link = float(held.get("uplink_kbytes_s") or 0.0)
        if math.isfinite(link) and link > 0:
            with _link_lock:
                if _link_kbytes_s is None:
                    _link_kbytes_s = link * _remembered_uplink_discount()
        for form in ("png", "webp"):
            cost = held.get(form)
            if isinstance(cost, list) and len(cost) == 2 and all(
                    math.isfinite(float(part)) and float(part) > 0 for part in cost):
                with _pack_lock:
                    _pack_cost.setdefault(form, (float(cost[0]), float(cost[1])))
    except Exception:  # noqa: BLE001  # nosec B110
        pass






_PROFILE_SAVE_EVERY = 10
_uploads = {"since_save": 0}


def flush_crop_profile() -> None:





    _uploads["since_save"] = 0
    _save_crop_profile()


def _save_crop_profile() -> None:

    try:
        import json

        from qgis.core import QgsSettings

        with _link_lock:
            link = _link_kbytes_s
        with _pack_lock:
            costs = dict(_pack_cost)
        held: dict = {form: list(cost) for form, cost in costs.items()}
        if link:
            held["uplink_kbytes_s"] = link
        QgsSettings().setValue(_PROFILE_SETTING, json.dumps(held, allow_nan=False))
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def note_crop_pack(form: str, pixels: int, packed_bytes: int,
                   elapsed_s: float) -> None:




    try:
        if (form not in ("png", "webp")
                or any(not math.isfinite(v) or v <= 0 for v in (pixels, packed_bytes, elapsed_s))):
            return
        sample = (float(elapsed_s) / pixels, float(packed_bytes) / pixels)
        with _pack_lock:
            held = _pack_cost.get(form)
            if held is None:
                _pack_cost[form] = sample
            else:
                smoothing = _uplink_smoothing()
                _pack_cost[form] = tuple(  # type: ignore[assignment]
                    smoothing * new + (1.0 - smoothing) * old
                    for new, old in zip(sample, held))
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def predicted_crop_wait_s(form: str, pixels: int,
                          uplink_kbytes_s: float) -> float | None:

    with _pack_lock:
        cost = _pack_cost.get(form)
    if cost is None or pixels <= 0 or uplink_kbytes_s <= 0:
        return None
    seconds_per_pixel, bytes_per_pixel = cost
    on_the_wire = bytes_per_pixel * pixels * _WIRE_EXPANSION / 1024.0
    return seconds_per_pixel * pixels + on_the_wire / uplink_kbytes_s


def click_png_effort() -> int:






    try:
        from .detection_policy import network_policy

        value = network_policy().get("click_png_effort")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and _PNG_EFFORT_MIN <= value <= _PNG_EFFORT_MAX):
            return int(value)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return _PNG_EFFORT


def click_webp_quality() -> int:






    try:
        from .detection_policy import network_policy

        value = network_policy().get("click_webp_quality")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and _WEBP_QUALITY_MIN <= value <= _WEBP_QUALITY_MAX):
            return int(value)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return _WEBP_QUALITY_LOSSLESS


def click_png_uplink_floor() -> float:






    try:
        from .detection_policy import network_policy

        value = network_policy().get("click_png_uplink_floor_kbytes_s")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and _UPLINK_FLOOR_MIN <= value <= _UPLINK_FLOOR_MAX):
            return float(value)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return _PNG_UPLINK_FLOOR


def note_crop_upload(sent_bytes: int, elapsed_s: float) -> None:










    global _link_kbytes_s
    try:
        if (not math.isfinite(sent_bytes) or not math.isfinite(elapsed_s)
                or sent_bytes < _UPLINK_MIN_SAMPLE_BYTES or elapsed_s <= 0):
            return
        sample = (float(sent_bytes) / 1024.0) / float(elapsed_s)
        smoothing = _uplink_smoothing()
        with _link_lock:
            held = _link_kbytes_s
            _link_kbytes_s = sample if held is None else (
                smoothing * sample + (1.0 - smoothing) * held)



        _uploads["since_save"] += 1
        if _uploads["since_save"] >= _PROFILE_SAVE_EVERY:
            _uploads["since_save"] = 0
            _save_crop_profile()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


_cheap_webp_writer: bool | None = None


def cheap_webp_writer_available() -> bool:










    global _cheap_webp_writer
    if _cheap_webp_writer is None:
        try:
            from PIL import features

            _cheap_webp_writer = bool(features.check("webp"))
        except Exception:  # noqa: BLE001
            _cheap_webp_writer = False
    return _cheap_webp_writer


def _measured_form_choice(pixels: int, uplink_kbytes_s: float) -> bool | None:





    png = predicted_crop_wait_s("png", pixels, uplink_kbytes_s)
    webp = predicted_crop_wait_s("webp", pixels, uplink_kbytes_s)
    if png is None or webp is None:
        return None
    hysteresis = _pack_hysteresis()
    if _link_prefers_png:
        return not webp < png * (1.0 - hysteresis)
    return png < webp * (1.0 - hysteresis)


def crop_png_preferred(pixels: int = 0) -> bool:
















    global _link_prefers_png
    try:
        if click_webp_quality() < _WEBP_QUALITY_LOSSLESS:
            return False
        floor = click_png_uplink_floor()


        _load_crop_profile()
        cheap_webp = cheap_webp_writer_available()
        with _link_lock:
            seen = _link_kbytes_s
            if seen is None:
                return False
            measured = _measured_form_choice(pixels, seen)
            if measured is not None:
                _link_prefers_png = measured
                return measured
            if cheap_webp:


                _link_prefers_png = False
                return False
            hysteresis = _uplink_hysteresis()
            if _link_prefers_png:
                if seen < floor * (1.0 - hysteresis):
                    _link_prefers_png = False
            elif seen > floor * (1.0 + hysteresis):
                _link_prefers_png = True
            return _link_prefers_png
    except Exception:  # noqa: BLE001  # nosec B110
        return False


def crop_webp_allowed() -> bool:






    try:
        from .server_dials import crop_webp_enabled

        return crop_webp_enabled()
    except Exception:  # noqa: BLE001  # nosec B110
        return False


def encode_crop_png(crop: np.ndarray) -> bytes | None:









    started = time.monotonic()
    try:
        from qgis.PyQt.QtCore import QBuffer, QByteArray
        from qgis.PyQt.QtGui import QImage

        from .qt_compat import FormatRGB888, WriteOnly

        height, width = int(crop.shape[0]), int(crop.shape[1])





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
        note_crop_pack("png", height * width, payload.size(),
                       time.monotonic() - started)
        return bytes(payload)
    except Exception as err:  # noqa: BLE001
        QgsMessageLog.logMessage(
            f"Crop PNG encode failed, sending raw pixels: {err}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return None


def _webp_bytes_via_pillow(crop: np.ndarray) -> bytes | None:














    try:
        import io

        from PIL import Image

        if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8:
            return None
        holder = io.BytesIO()



        Image.fromarray(np.ascontiguousarray(crop)).save(
            holder, format="WEBP", lossless=True, method=0,
            quality=_webp_lossless_effort())
        return holder.getvalue() or None
    except Exception:  # noqa: BLE001  # nosec B110
        return None


def encode_crop_webp(crop: np.ndarray) -> bytes | None:

















    quality = click_webp_quality()
    pixels = int(crop.shape[0]) * int(crop.shape[1]) if crop.ndim == 3 else 0
    started = time.monotonic()
    if quality >= _WEBP_QUALITY_LOSSLESS:



        packed = _webp_bytes_via_pillow(crop)
        if packed is not None:
            note_crop_pack("webp", pixels, len(packed),
                           time.monotonic() - started)
            return packed
    started = time.monotonic()
    try:
        from qgis.PyQt.QtCore import QBuffer, QByteArray
        from qgis.PyQt.QtGui import QImage

        from .qt_compat import FormatRGB888, WriteOnly

        height, width = int(crop.shape[0]), int(crop.shape[1])



        raw = crop.tobytes()
        image = QImage(raw, width, height, width * 3, FormatRGB888).copy()
        if image.isNull():
            return None
        payload = QByteArray()
        buffer = QBuffer(payload)
        buffer.open(WriteOnly)
        written = image.save(buffer, "WEBP", quality)
        buffer.close()
        if not written or payload.isEmpty():
            return None
        note_crop_pack("webp", height * width, payload.size(),
                       time.monotonic() - started)
        return bytes(payload)
    except Exception:  # noqa: BLE001  # nosec B110
        return None
