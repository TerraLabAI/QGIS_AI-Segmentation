










from __future__ import annotations

import math
import re

MIN_CLASSES = 2
MAX_CLASSES = 12
NAME_MAX_CHARS = 40
IMAGE_MAX_PX = 1024
EVERYTHING_ELSE_COLOR = "#9e9e9e"

FALLBACK_MAX_ZONE_KM2 = 4.0
FALLBACK_WARN_M_PER_PX = 1.0
FALLBACK_MIN_BILLED_KM2 = 0.25

_HEX = re.compile(r"#([0-9a-fA-F]{6})\b")
_SPLIT = re.compile(r"[\t,;]+")


def enabled() -> bool:

    import os

    if os.environ.get("AI_SEGMENTATION_FORCE_MY_CLASSES") == "1":
        return True
    try:
        from .server_dials import feature_switch
        return feature_switch("features.my_classes", False)
    except Exception:  # noqa: BLE001
        return False


def max_classes() -> int:
    from .server_dials import dial_in_range
    return int(dial_in_range("tuning.my_classes.max_classes", MAX_CLASSES, MIN_CLASSES, MAX_CLASSES))


def max_zone_km2() -> float:
    from .server_dials import dial_in_range
    return float(dial_in_range("tuning.my_classes.max_zone_km2", FALLBACK_MAX_ZONE_KM2, 0.01, 1000.0))


def warn_m_per_px() -> float:
    from .server_dials import dial_in_range
    return float(dial_in_range("tuning.my_classes.warn_m_per_px", FALLBACK_WARN_M_PER_PX, 0.01, 1000.0))


def min_billed_km2() -> float:
    from .server_dials import dial_in_range
    return float(dial_in_range("tuning.my_classes.min_billed_km2", FALLBACK_MIN_BILLED_KM2, 0.0, 100.0))


def billed_km2(zone_km2: float) -> float:

    zone = zone_km2 if isinstance(zone_km2, (int, float)) and math.isfinite(zone_km2) else 0.0
    return max(float(zone), min_billed_km2())


def default_colors(count: int) -> list[str]:

    from .class_symbology import interpolate_colors, legend_ramp_anchors
    return [c.lower() for c in interpolate_colors(legend_ramp_anchors(), max(2, count))][:count]


def clean_name(name) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(name or "")).strip()
    return text[:NAME_MAX_CHARS].strip()


def parse_paste_list(text: str) -> list[tuple[str, str | None]]:


    out: list[tuple[str, str | None]] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _HEX.search(line)
        color = None
        if match:
            color = "#" + match.group(1).lower()
            line = (line[:match.start()] + line[match.end():]).strip()
        name = clean_name(_SPLIT.sub(" ", line))
        if name:
            out.append((name, color))
    return out


def validation_error(rows: list[tuple[str, str]], everything_else: str) -> str | None:

    from .i18n import tr

    names = [clean_name(n) for n, _c in rows]
    if len(rows) < MIN_CLASSES:
        return tr("Add at least {n} classes.").format(n=MIN_CLASSES)
    if len(rows) > max_classes():
        return tr("Use at most {n} classes.").format(n=max_classes())
    if any(not n for n in names):
        return tr("Give every class a name.")
    lowered = [n.lower() for n in names]
    if len(set(lowered)) != len(lowered):
        return tr("Two classes have the same name.")
    if everything_else.lower() in lowered:
        return tr('"{name}" is already the last row.').format(name=everything_else)
    return None


def decode_class_raster(png_b64: str, width: int, height: int):


    import base64

    import numpy as np
    from qgis.PyQt.QtGui import QImage

    data = base64.b64decode(png_b64 or "", validate=False)
    image = QImage.fromData(data, "PNG")
    if image.isNull():
        raise ValueError("class raster unreadable")
    image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = image.width(), image.height()
    stride = image.bytesPerLine()
    ptr = image.constBits()
    try:
        ptr.setsize(stride * h)
        buf = bytes(ptr)
    except AttributeError:
        buf = bytes(ptr.asstring(stride * h)) if hasattr(ptr, "asstring") else bytes(ptr)
    grid = np.frombuffer(buf, dtype=np.uint8).reshape(h, stride)[:, :w].copy()
    if (h, w) != (int(height), int(width)):
        rows = np.minimum((np.arange(int(height)) * h) // max(1, int(height)), h - 1)
        cols = np.minimum((np.arange(int(width)) * w) // max(1, int(width)), w - 1)
        grid = grid[rows][:, cols]
    return grid


def image_size_for(extent_w: float, extent_h: float) -> tuple[int, int]:

    if extent_w <= 0 or extent_h <= 0:
        return IMAGE_MAX_PX, IMAGE_MAX_PX
    if extent_w >= extent_h:
        return IMAGE_MAX_PX, max(16, int(round(IMAGE_MAX_PX * extent_h / extent_w)))
    return max(16, int(round(IMAGE_MAX_PX * extent_w / extent_h))), IMAGE_MAX_PX
