"""Turn a geometry into a mask on a crop's pixel grid, with or without rasterio.

Why this exists, and why it is not a helper on a mixin: the click path uses this
mask as the shape a click is allowed to grow from. Every rule that stops a click
shrinking an object reads it, and every one of them gives up quietly when it is
missing. So an install where the rasterize call raises turns all of them off at
once, and the user sees a click replace a house with a roof facet.

That install is not hypothetical. It is the one the cloud click route was built
for: an online basemap, no local model, no torch, and therefore no rasterio.

Qt is always there, in every QGIS, so the fallback paints the polygon into an
image and reads the pixels back. Same mask, no dependency.
"""

from __future__ import annotations

import json

import numpy as np


def rasterize_geometry_to_grid(geom, bounds, shape) -> np.ndarray | None:
    """A bool mask of ``geom`` on a ``shape`` grid covering ``bounds``.

    ``bounds`` is (minx, miny, maxx, maxy) in the geometry's own CRS, ``shape``
    is (height, width). None only when the geometry or the window is unusable,
    never because a package is absent.
    """
    height, width = int(shape[0]), int(shape[1])
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    if width <= 0 or height <= 0 or maxx <= minx or maxy <= miny:
        return None
    if geom is None or geom.isEmpty():
        return None
    mask = _with_rasterio(geom, (minx, miny, maxx, maxy), height, width)
    if mask is not None:
        return mask
    # The Qt half gets the same guard as the rasterio half. Without it a Qt
    # fault escaped to the caller, which answers a raised exception and a
    # returned None the same way: no prior mask. Every anti-shrink rule reads
    # that as "nothing to protect" and switches off at once, so one bad draw
    # turned a click from growing a house into replacing it with a roof facet,
    # silently.
    try:
        return _with_qt(geom, (minx, miny, maxx, maxy), height, width)
    except Exception:  # noqa: BLE001 -- a fallback that raises is not a fallback
        return None


def _with_rasterio(geom, bounds, height, width) -> np.ndarray | None:
    try:
        from rasterio import features
        from rasterio.transform import from_bounds as transform_from_bounds

        minx, miny, maxx, maxy = bounds
        transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)
        painted = features.rasterize(
            [(json.loads(geom.asJson()), 1)], out_shape=(height, width),
            transform=transform, fill=0)
        return painted.astype(bool)
    except Exception:  # noqa: BLE001 -- absent or unhappy, the fallback answers
        return None


def _parts(geom) -> list[list]:
    """Every polygonal part of the geometry, each as its own list of rings.

    Kept apart part by part on purpose. Poured into one odd-even path, two
    parts that overlap cancel each other and leave a hole exactly where two
    objects meet, which is where the click path most needs the mask to be
    solid. Inner rings stay with the part they belong to, and the odd-even rule
    inside that part is what carves a courtyard out.
    """
    try:
        multi = geom.asMultiPolygon()
    except Exception:  # noqa: BLE001
        multi = None
    if multi:
        return [list(polygon) for polygon in multi if polygon]
    try:
        single = geom.asPolygon()
    except Exception:  # noqa: BLE001
        single = None
    return [list(single)] if single else []


def _with_qt(geom, bounds, height, width) -> np.ndarray | None:
    try:
        from qgis.PyQt.QtCore import QPointF, Qt
        from qgis.PyQt.QtGui import (
            QBrush,
            QImage,
            QPainter,
            QPainterPath,
            QPolygonF,
        )

        from .qt_compat import resolve_qt_enum
    except Exception:  # noqa: BLE001 -- no Qt means no QGIS; nothing to draw on
        return None

    # Every enum below goes through the shim, never a flat attribute. Qt6 moved
    # them into scopes, and the plugin repository runs a STATIC checker: a flat
    # access flags the version on upload even when it is only a fallback arm.
    odd_even_fill = resolve_qt_enum(Qt, "FillRule", "OddEvenFill")
    grayscale_8 = resolve_qt_enum(QImage, "Format", "Format_Grayscale8")
    antialiasing = resolve_qt_enum(QPainter, "RenderHint", "Antialiasing")
    no_pen = resolve_qt_enum(Qt, "PenStyle", "NoPen")
    white = resolve_qt_enum(Qt, "GlobalColor", "white")

    parts = _parts(geom)
    if not parts:
        return None

    minx, miny, maxx, maxy = bounds
    scale_x = width / (maxx - minx)
    scale_y = height / (maxy - miny)

    # One path per part, united by drawing them one after the other. Each part
    # carries the odd-even rule that cuts its own courtyards; painting over
    # ground already lit adds nothing, so overlapping parts stay solid.
    paths = []
    for rings in parts:
        path = QPainterPath()
        path.setFillRule(odd_even_fill)
        drawn = False
        for ring in rings:
            if len(ring) < 3:
                continue
            # North is up in the raster and up in the world, so the row axis
            # flips.
            path.addPolygon(QPolygonF([
                QPointF((point.x() - minx) * scale_x,
                        (maxy - point.y()) * scale_y)
                for point in ring
            ]))
            path.closeSubpath()
            drawn = True
        if drawn:
            paths.append(path)
    if not paths:
        return None

    image = QImage(width, height, grayscale_8)
    image.fill(0)
    painter = QPainter(image)
    try:
        # No antialiasing: this is a mask, and a half-lit edge pixel would read
        # as ground the click may grow into.
        painter.setRenderHint(antialiasing, False)
        painter.setPen(no_pen)
        painter.setBrush(QBrush(white))
        for path in paths:
            painter.drawPath(path)
    finally:
        painter.end()

    # Two traps here, both of which have bitten this repo before: the rows are
    # padded to `bytesPerLine`, and a numpy view over Qt's buffer turns into
    # noise the moment Qt frees it. So take the stride into account and copy.
    stride = image.bytesPerLine()
    raw = bytes(image.constBits().asstring(stride * height))
    flat = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)
    return np.array(flat[:, :width] > 127, dtype=bool, copy=True)
