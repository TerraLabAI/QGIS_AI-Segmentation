



















from __future__ import annotations

from qgis.core import Qgis, QgsGeometry, QgsPointXY

from .layer_conventions import repair_polygon


def _repaired(geom: QgsGeometry | None) -> QgsGeometry | None:







    if geom is None:
        return None
    try:
        if geom.isEmpty():
            return None
        return repair_polygon(geom)
    except Exception:  # noqa: BLE001
        return None


def merge_geometries(geoms: list[QgsGeometry]) -> QgsGeometry | None:






















    if not geoms:
        return None

    parts = [fixed for fixed in (_repaired(g) for g in geoms) if fixed is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    merged = _union_all(parts)
    if merged is None:


        merged = _collected(parts)
    if merged is None:
        return None
    return _repaired(merged) or merged


def _union_all(parts: list[QgsGeometry]) -> QgsGeometry | None:







    try:
        union = QgsGeometry.unaryUnion(parts)
        if union is not None and not union.isEmpty():
            return union
    except Exception:  # noqa: BLE001  # nosec B110
        pass

    merged: QgsGeometry | None = None
    failed = False
    for part in parts:
        if merged is None:
            merged = part
            continue
        try:
            union = merged.combine(part)
        except Exception:  # noqa: BLE001
            union = None
        if union is None or union.isEmpty():
            failed = True
            continue
        merged = union
    if failed or merged is None or merged.isEmpty():
        return None
    return merged


def _collected(parts: list[QgsGeometry]) -> QgsGeometry | None:

    try:
        collected = QgsGeometry.collectGeometry(parts)
    except Exception:  # noqa: BLE001
        return None
    if collected is None or collected.isEmpty():
        return None
    return collected


def split_geometry(
    geom: QgsGeometry, cut_line: list[QgsPointXY]
) -> list[QgsGeometry]:
























    if geom is None:
        return []
    if not cut_line or len(cut_line) < 2:
        return [geom]

    source = _repaired(geom)
    if source is None:
        return [geom]

    work = QgsGeometry(source)
    products = _run_split(work, cut_line)
    if products is None:
        return [geom]

    pieces = [
        fixed
        for fixed in (_repaired(p) for p in [work, *products])
        if fixed is not None
    ]
    if len(pieces) < 2:
        return [geom]
    return _sorted_pieces(pieces)


def _run_split(
    work: QgsGeometry, cut_line: list[QgsPointXY]
) -> list[QgsGeometry] | None:


















    line = list(cut_line)
    try:
        try:
            ret = work.splitGeometry(line, False)
        except TypeError:
            ret = work.splitGeometry(line, False, True)
    except Exception:  # noqa: BLE001
        return None
    return _split_products(ret)


def _split_products(ret) -> list[QgsGeometry]:






    if ret is None:
        return []
    if not isinstance(ret, (tuple, list)):
        return []
    if len(ret) < 2:
        return []
    produced = ret[1]
    if not produced:
        return []
    try:
        return [g for g in produced if isinstance(g, QgsGeometry)]
    except TypeError:
        return []


def _sorted_pieces(pieces: list[QgsGeometry]) -> list[QgsGeometry]:

    def key(piece: QgsGeometry):
        try:
            point = piece.centroid().asPoint()
            return (point.x(), point.y(), -piece.area())
        except Exception:  # noqa: BLE001
            return (0.0, 0.0, 0.0)

    return sorted(pieces, key=key)


def polygon_part_count(geom: QgsGeometry | None) -> int:








    if geom is None:
        return 0
    try:
        if not geom.isMultipart():
            return 1
        return len(geom.asMultiPolygon())
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 1


def bridge_seam_gap(geom: QgsGeometry | None,
                    tolerance: float) -> QgsGeometry | None:









    if geom is None or tolerance <= 0:
        return None
    try:
        grown = _seam_buffer_square_corners(geom, tolerance)
        if grown is None or grown.isEmpty():
            return None
        closed = _seam_buffer_square_corners(grown, -tolerance)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None
    if closed is None or closed.isEmpty():
        return None
    if polygon_part_count(closed) > 1:
        return None
    return _repaired(closed)


def _seam_buffer_square_corners(geom: QgsGeometry,
                                distance: float) -> QgsGeometry | None:








    cap = getattr(getattr(Qgis, "EndCapStyle", None), "Round", None)
    join = getattr(getattr(Qgis, "JoinStyle", None), "Miter", None)
    if cap is None or join is None:
        cap = getattr(QgsGeometry, "CapRound", None)
        join = getattr(QgsGeometry, "JoinStyleMiter", None)
    if cap is not None and join is not None:
        try:
            return geom.buffer(distance, 8, cap, join, 2.0)
        except (TypeError, AttributeError):
            pass
    return geom.buffer(distance, 8)
