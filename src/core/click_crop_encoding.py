"""Turning one click crop into the bytes that travel, and the dials that decide.

Split out of ``cloud_sam_predictor`` because it is a concern of its own: three
forms (webp, PNG, raw pixels), the served dials that shape them, and the Qt
traps that come with asking QImage to write any of them. The predictor picks a
form and sends it; this module knows how each one is made, and how fast the
link underneath has been.

Every function here returns None rather than raising. A crop that will not
compress still travels as raw pixels, so a click never fails on a compression
step. Qt is the writer that always answers, because the click path has to work
for a user whose local install carries nothing but the light packages. An
imaging library is asked only when one is already importable, and only where it
does the same lossless job for less of the user's time.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from qgis.core import Qgis, QgsMessageLog

# How hard Qt is asked to squeeze the crop. Qt spells this as a "quality"
# number, but PNG has no lossy setting: it only picks the compression effort,
# and the pixels come back identical whatever it is. The default effort spends
# several times longer than this one to shave a few per cent off a picture that
# is about to travel anyway, and that time is paid on the click path. This
# path stays lossless whatever the number: the one place a crop may lose a
# pixel is the webp quality dial below, and PNG is what answers when it is off.
#
# It trades local encode time against bytes on the wire, and which way that
# trade falls depends on the user's connection, so it is a dial
# (``network.click_png_effort``). The constant below is the fallback.
_PNG_EFFORT = 70

# Qt's own range for the argument. 0 is legal and means "spend nothing", so a
# served value is bounded rather than required to be positive.
_PNG_EFFORT_MIN = 0
_PNG_EFFORT_MAX = 100

# What the WebP writer is asked for. Qt's WEBP plugin reads 100 as lossless and
# EVERY other value as lossy, so this number is not a smooth knob: it either
# keeps the caller's own pixels or hands the far side a picture that has been
# rewritten. The shipped value is the lossless one, so a client that never
# hears from the server sends what it has always sent, and a lighter crop is
# something the server asks for rather than something a cold cache invents.
_WEBP_QUALITY_LOSSLESS = 100

# A served value under this floor is refused whole rather than clamped. Below
# it the rewritten picture moves enough for the mask to land somewhere else,
# and answering a click on a picture the user never saw is worse than spending
# the bytes.
_WEBP_QUALITY_MIN = 60
_WEBP_QUALITY_MAX = 100

# What the lossless writer is told to spend. Under `lossless=True` the library
# reads "quality" as an effort dial and never as picture: every value gives the
# caller's own pixels back. The writer's own default is 80, which spends six
# times the work of 20 to produce a FRACTIONALLY LARGER file, so the default is
# the wrong end of its own curve. Measured on two 1024 px aerial crops: 660 and
# 760 ms at 80 against 108 and 122 ms at 20, for 1117 and 1036 KB against 1111
# and 1030 KB, and the decode is byte-identical to the input at both.
_WEBP_LOSSLESS_EFFORT = 20

# Both forms are lossless, so the choice between them is only ever a trade of
# the user's own time: PNG hands the wire more bytes and costs far less local
# work than webp, so on a fast enough link the whole wait is shorter as PNG and
# on a slow one it is longer. Where the two cross moves with the machine as
# much as with the link, so the crossing point is a dial
# (``network.click_png_uplink_floor_kbytes_s``, in KB/s) and the constant below
# is only the fallback. It sits well clear of the crossing rather than on it,
# because the two mistakes do not cost the same: choosing PNG on a link that is
# too slow makes a struggling user slower, while staying on webp only keeps
# what this client has always sent.
_PNG_UPLINK_FLOOR = 750.0

# Bounds for the served floor. Both ends of this range are usable operating
# points, a near-zero value meaning "always PNG" and a huge one "never", so
# only a value outside it is refused.
_UPLINK_FLOOR_MIN = 0.001
_UPLINK_FLOOR_MAX = 1_000_000.0

# The choice is sticky: it moves only once the estimate is clear of the floor
# by this fraction, so one upload that lands near the crossing point does not
# make every second click change format.
_UPLINK_HYSTERESIS = 0.25

# How closely the estimate follows the link. High enough to notice a user
# moving from an office to a phone tether within a few uploads, low enough that
# one stalled upload does not stand for the whole connection.
_UPLINK_SMOOTHING = 0.4

# Under this many bytes a sample measures the round trip rather than the link,
# so it says nothing about throughput and is dropped. A click crop is always
# far larger than this; a small body is something else.
_UPLINK_MIN_SAMPLE_BYTES = 64 * 1024

# Session state for the link estimate. The crop is handed over from whichever
# thread set the image, so the read-modify-write below is held under a lock:
# uncontended, and on a path that is about to spend hundreds of times longer
# compressing a picture.
_link_lock = threading.Lock()
_link_kbytes_s: float | None = None
_link_prefers_png = False

# What the body carries per byte the packer produced. The crop travels base64
# inside the JSON, so four bytes go out for every three packed, and a crossing
# point worked out on the packed bytes sits below the real one.
_WIRE_EXPANSION = 4.0 / 3.0

# What packing actually cost on THIS machine, per format: seconds of work per
# pixel and packed bytes per pixel, smoothed like the link estimate. The
# crossing point between the two formats moves with the machine as much as with
# the link, and a served number can only know the link, so once a session has
# packed each form once it decides from what it measured instead.
_pack_lock = threading.Lock()
_pack_cost: dict[str, tuple[float, float]] = {}

# The predicted waits have to differ by this fraction before the choice moves,
# for the same reason the floor comparison is sticky.
_PACK_HYSTERESIS = 0.10

# Where the profile is kept between sessions. Never rename the literal: it is a
# key in the user's QGIS profile, and a new spelling silently starts every
# returning user from nothing again.
_PROFILE_SETTING = "TerraLab/click_crop_profile"

# What the remembered uplink is worth as an opening guess. Half, because the two
# mistakes do not cost the same: opening on PNG at a desk when the user has
# moved to a train costs about three seconds on one click, and opening on webp
# at that desk costs about half of one. Half the remembered speed keeps the
# fast-desk case and gives up the borderline one. Packing costs get no discount:
# they are a fact about the machine, and the machine did not move.
_REMEMBERED_UPLINK_DISCOUNT = 0.5

_profile_loaded = False


def _load_crop_profile() -> None:
    """Bring last session's link and packing costs back, once per process.

    Without this every session pays one webp pack before it knows anything,
    which on a fast link is the slowest form on the click a user notices most.

    Never raises. A profile that cannot be read leaves the session measuring
    from nothing, which is what it did before this existed.
    """
    global _profile_loaded, _link_kbytes_s
    if _profile_loaded:
        return
    _profile_loaded = True
    try:
        import json

        from qgis.core import QgsSettings

        raw = QgsSettings().value(_PROFILE_SETTING, "")
        held = json.loads(raw) if raw else {}
        link = float(held.get("uplink_kbytes_s") or 0.0)
        if link > 0:
            with _link_lock:
                if _link_kbytes_s is None:
                    _link_kbytes_s = link * _REMEMBERED_UPLINK_DISCOUNT
        for form in ("png", "webp"):
            cost = held.get(form)
            if isinstance(cost, list) and len(cost) == 2 and all(
                    float(part) > 0 for part in cost):
                with _pack_lock:
                    _pack_cost.setdefault(form, (float(cost[0]), float(cost[1])))
    except Exception:  # noqa: BLE001 -- a stale profile must never break a click  # nosec B110
        pass


def _save_crop_profile() -> None:
    """Write what this session measured, so the next one opens calibrated."""
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
        QgsSettings().setValue(_PROFILE_SETTING, json.dumps(held))
    except Exception:  # noqa: BLE001 -- remembering is never worth a click  # nosec B110
        pass


def note_crop_pack(form: str, pixels: int, packed_bytes: int,
                   elapsed_s: float) -> None:
    """Record what packing one crop cost, so the next crop picks the cheaper form.

    Never raises. A sample that cannot be used leaves the estimate where it was.
    """
    try:
        if pixels <= 0 or packed_bytes <= 0 or elapsed_s <= 0:
            return
        sample = (float(elapsed_s) / pixels, float(packed_bytes) / pixels)
        with _pack_lock:
            held = _pack_cost.get(form)
            if held is None:
                _pack_cost[form] = sample
            else:
                _pack_cost[form] = tuple(  # type: ignore[assignment]
                    _UPLINK_SMOOTHING * new + (1.0 - _UPLINK_SMOOTHING) * old
                    for new, old in zip(sample, held))
    except Exception:  # noqa: BLE001 -- a bad sample must never break a click  # nosec B110
        pass


def predicted_crop_wait_s(form: str, pixels: int,
                          uplink_kbytes_s: float) -> float | None:
    """Packing plus sending one crop in ``form``, or None before it was timed."""
    with _pack_lock:
        cost = _pack_cost.get(form)
    if cost is None or pixels <= 0 or uplink_kbytes_s <= 0:
        return None
    seconds_per_pixel, bytes_per_pixel = cost
    on_the_wire = bytes_per_pixel * pixels * _WIRE_EXPANSION / 1024.0
    return seconds_per_pixel * pixels + on_the_wire / uplink_kbytes_s


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


def click_webp_quality() -> int:
    """Qt's WebP quality for one click crop, the floor to 100.

    Read off the served network policy, with lossless standing for an absent,
    out-of-range or malformed value. Never raises and never networks: a click
    path calls it.
    """
    try:
        from .detection_policy import network_policy

        value = network_policy().get("click_webp_quality")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and _WEBP_QUALITY_MIN <= value <= _WEBP_QUALITY_MAX):
            return int(value)
    except Exception:  # noqa: BLE001 -- a dial must never break a click  # nosec B110
        pass
    return _WEBP_QUALITY_LOSSLESS


def click_png_uplink_floor() -> float:
    """The uplink above which a click crop is better sent as PNG, in KB/s.

    Read off the served network policy, with the shipped constant standing for
    an absent, out-of-range or malformed value. Never raises and never
    networks: a click path calls it.
    """
    try:
        from .detection_policy import network_policy

        value = network_policy().get("click_png_uplink_floor_kbytes_s")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and _UPLINK_FLOOR_MIN <= value <= _UPLINK_FLOOR_MAX):
            return float(value)
    except Exception:  # noqa: BLE001 -- a dial must never break a click  # nosec B110
        pass
    return _PNG_UPLINK_FLOOR


def note_crop_upload(sent_bytes: int, elapsed_s: float) -> None:
    """Record one crop upload, so the next crop is packed for this link.

    ``elapsed_s`` is the time those bytes spent travelling. A window that also
    covers work done before the send is accepted rather than refused: it can
    only pull the estimate down, and a lower estimate keeps the format this
    client has always sent.

    Never raises. This runs beside a click, and a sample that cannot be used is
    not a failure to report: the estimate stays where it was.
    """
    global _link_kbytes_s
    try:
        if sent_bytes < _UPLINK_MIN_SAMPLE_BYTES or elapsed_s <= 0:
            return
        sample = (float(sent_bytes) / 1024.0) / float(elapsed_s)
        with _link_lock:
            held = _link_kbytes_s
            _link_kbytes_s = sample if held is None else (
                _UPLINK_SMOOTHING * sample + (1.0 - _UPLINK_SMOOTHING) * held)
        # One write per crop, and the crop is the slowest thing on this path.
        _save_crop_profile()
    except Exception:  # noqa: BLE001 -- a bad sample must never break a click  # nosec B110
        pass


def observed_uplink_kbytes_s() -> float | None:
    """The smoothed uplink this session has seen, in KB/s. None before any."""
    with _link_lock:
        return _link_kbytes_s


_cheap_webp_writer: bool | None = None


def cheap_webp_writer_available() -> bool:
    """Whether this machine can write lossless WebP for less work than PNG.

    Which of the two lossless forms is cheaper is a fact about the writer, not
    about the format. Qt's WebP writer spends several times its PNG one, and the
    library Pillow reaches spends less than either. So a machine that has the
    cheap writer has nothing to weigh: WebP packs faster AND sends fewer bytes.

    Cached, because the answer cannot change inside a session and the check
    touches a native extension. Never raises.
    """
    global _cheap_webp_writer
    if _cheap_webp_writer is None:
        try:
            from PIL import features

            _cheap_webp_writer = bool(features.check("webp"))
        except Exception:  # noqa: BLE001 -- an absent writer is an answer
            _cheap_webp_writer = False
    return _cheap_webp_writer


def _measured_form_choice(pixels: int, uplink_kbytes_s: float) -> bool | None:
    """PNG or webp from what this machine timed. None until both were timed.

    Called with ``_link_lock`` held, and takes only ``_pack_lock``, so the two
    are always acquired in that order.
    """
    png = predicted_crop_wait_s("png", pixels, uplink_kbytes_s)
    webp = predicted_crop_wait_s("webp", pixels, uplink_kbytes_s)
    if png is None or webp is None:
        return None
    if _link_prefers_png:
        return not webp < png * (1.0 - _PACK_HYSTERESIS)
    return png < webp * (1.0 - _PACK_HYSTERESIS)


def crop_png_preferred(pixels: int = 0) -> bool:
    """Whether PNG beats webp on the whole wait for a crop of ``pixels`` pixels.

    False until an upload has been timed, because an unknown link is exactly
    where the smaller body is the safe answer, and false whenever the server
    has asked for a lighter crop: a lossy webp is a decision about bytes that a
    lossless PNG would undo.

    Once the session has packed each form once, the answer comes from what this
    machine measured rather than from the served crossing point, since a slow
    packer and a fast one cross the link at different speeds. Until then, and
    whenever ``pixels`` is unknown, it falls back to the served floor.

    Sticky on either side, so a click does not change form every time an upload
    lands near the crossing point. Never raises: the format this client has
    always sent is what an unanswerable question gets.
    """
    global _link_prefers_png
    try:
        if click_webp_quality() < _WEBP_QUALITY_LOSSLESS:
            return False
        floor = click_png_uplink_floor()
        # Both before the lock: the first takes both locks itself, in that
        # order, and the second may import a native extension.
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
                # Nothing for the floor to decide: on this machine webp packs
                # faster than PNG and sends fewer bytes, so it wins both ways.
                _link_prefers_png = False
                return False
            if _link_prefers_png:
                if seen < floor * (1.0 - _UPLINK_HYSTERESIS):
                    _link_prefers_png = False
            elif seen > floor * (1.0 + _UPLINK_HYSTERESIS):
                _link_prefers_png = True
            return _link_prefers_png
    except Exception:  # noqa: BLE001 -- the form must never break a click  # nosec B110
        return False


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


def encode_crop_png(crop: np.ndarray) -> bytes | None:
    """A contiguous (H, W, 3) uint8 crop as PNG bytes, or None when Qt cannot.

    PNG rather than the raw bytes because the raw form of a large crop nearly
    fills the request-body ceiling, and lossless because it is the form that
    answers when nothing has asked for a lighter crop.

    Returns None instead of raising, so a click never fails on a compression
    step: the caller sends the raw bytes when it gets None.
    """
    started = time.monotonic()
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
        note_crop_pack("png", height * width, payload.size(),
                       time.monotonic() - started)
        return bytes(payload)
    except Exception as err:  # noqa: BLE001 -- the raw bytes still go out
        QgsMessageLog.logMessage(
            f"Crop PNG encode failed, sending raw pixels: {err}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return None


def _webp_bytes_via_pillow(crop: np.ndarray) -> bytes | None:
    """Lossless WebP at the writer's cheapest setting, or None when unavailable.

    Qt exposes no effort setting for WebP: ``QImageWriter.setCompression`` is
    ignored by that plugin, so Qt only ever writes it the slow way. The library
    behind the format does have a cheap setting that reaches nearly the same
    size for a fraction of the work, and Pillow is how it is asked for. Same
    pixels back either way, so this is a choice about the user's time only.

    Opportunistic. Pillow is not a dependency of this plugin and a cloud-only
    install may carry none, so an absent one is not an error, it is Qt's turn.
    Nothing here puts Pillow on the path either: whatever is already importable
    is used, so this never pulls a native extension into the process that was
    not already loaded there.
    """
    try:
        import io

        from PIL import Image

        if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8:
            return None
        holder = io.BytesIO()
        # No mode argument: Pillow deprecated it, and a (H, W, 3) uint8 array
        # already reads as RGB. Contiguous because the buffer is handed over by
        # pointer, and a no-op when the caller's crop already is one.
        Image.fromarray(np.ascontiguousarray(crop)).save(
            holder, format="WEBP", lossless=True, method=0,
            quality=_WEBP_LOSSLESS_EFFORT)
        return holder.getvalue() or None
    except Exception:  # noqa: BLE001 -- Qt's own writer still answers  # nosec B110
        return None


def encode_crop_webp(crop: np.ndarray) -> bytes | None:
    """A contiguous (H, W, 3) uint8 crop as WebP bytes, or None.

    WebP because it packs the same pixels smaller than PNG on aerial imagery.
    How much smaller is what ``click_webp_quality`` decides: at the lossless
    value the far side answers on exactly the pixels the caller saw, and under
    it the crop is rewritten before it travels, which buys a shorter upload and
    costs the guarantee that the mask came back on the caller's own pixels.
    Lossless stays reachable and is what this client sends until the server
    says otherwise.

    Two writers can produce it. At the lossless setting the cheaper one is
    tried first when it is there; Qt answers otherwise and in every lossy case.

    Returns None instead of raising (a Qt build may carry no webp plugin at
    all), and the caller falls back to PNG: a click never fails on a
    compression step.
    """
    quality = click_webp_quality()
    pixels = int(crop.shape[0]) * int(crop.shape[1]) if crop.ndim == 3 else 0
    started = time.monotonic()
    if quality >= _WEBP_QUALITY_LOSSLESS:
        # Only at the lossless setting. Under it the server has asked for a
        # rewritten picture, and that number is Qt's to honour: the cheap
        # lossless path answers a different question.
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
        written = image.save(buffer, "WEBP", quality)
        buffer.close()
        if not written or payload.isEmpty():
            return None
        note_crop_pack("webp", height * width, payload.size(),
                       time.monotonic() - started)
        return bytes(payload)
    except Exception:  # noqa: BLE001 -- PNG still goes out  # nosec B110
        return None
