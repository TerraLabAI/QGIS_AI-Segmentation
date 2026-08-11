














from __future__ import annotations

import json

import numpy as np


def rasterize_geometry_to_grid(geom, bounds, shape) -> np.ndarray | None:






    height, width = int(shape[0]), int(shape[1])
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    if width <= 0 or height <= 0 or maxx <= minx or maxy <= miny:
        return None
    if geom is None or geom.isEmpty():
        return None
    mask = _with_rasterio(geom, (minx, miny, maxx, maxy), height, width)
    if mask is not None:
        return mask






    try:
        return _with_qt(geom, (minx, miny, maxx, maxy), height, width)
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        return None


def _parts(geom) -> list[list]:








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
    except Exception:  # noqa: BLE001
        return None




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




    paths = []
    for rings in parts:
        path = QPainterPath()
        path.setFillRule(odd_even_fill)
        drawn = False
        for ring in rings:
            if len(ring) < 3:
                continue


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


        painter.setRenderHint(antialiasing, False)
        painter.setPen(no_pen)
        painter.setBrush(QBrush(white))
        for path in paths:
            painter.drawPath(path)
    finally:
        painter.end()




    stride = image.bytesPerLine()
    raw = bytes(image.constBits().asstring(stride * height))
    flat = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)
    return np.array(flat[:, :width] > 127, dtype=bool, copy=True)
