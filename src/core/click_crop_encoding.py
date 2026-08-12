"""Turning one click crop into the bytes that travel, and the dials that decide.

Split out of ``cloud_sam_predictor`` because it is a concern of its own: three
forms (webp, PNG, raw pixels), two served dials, and the Qt traps that come
with asking QImage to write any of them. The predictor picks a form and sends
it; this module knows how each one is made.

Every function here returns None rather than raising. A crop that will not
compress still travels as raw pixels, so a click never fails on a compression
step. Qt does the encoding, never an imaging library: the click path has to
work for a user whose local install carries nothing but the light packages.
"""
from __future__ import annotations

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
        QgsMessageLog.logMessage(
            f"Crop PNG encode failed, sending raw pixels: {err}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
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

    Returns None instead of raising (a Qt build may carry no webp plugin at
    all), and the caller falls back to PNG: a click never fails on a
    compression step.
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
        written = image.save(buffer, "WEBP", click_webp_quality())
        buffer.close()
        if not written or payload.isEmpty():
            return None
        return bytes(payload)
    except Exception:  # noqa: BLE001 -- PNG still goes out  # nosec B110
        return None
