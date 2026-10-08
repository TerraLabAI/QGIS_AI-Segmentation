















from __future__ import annotations

import math

from .land_cover import NODATA, fold_small_patches, polygon_parts


BLOCK_PX = 2048
HALO_PX = 256


def _zone_pixel_geometry(zone_wkb: bytes | None, geo: dict, pads: tuple):

    if not zone_wkb:
        return None
    from qgis.core import QgsGeometry
    from qgis.PyQt.QtGui import QTransform

    zone = QgsGeometry()
    zone.fromWkb(zone_wkb)
    if zone.isEmpty():
        return None
    minx, miny, maxx, maxy = geo["bbox"]
    img_h, img_w = geo["img_shape"]
    px_w = (maxx - minx) / max(1, img_w)
    px_h = (maxy - miny) / max(1, img_h)
    top, _bottom, left, _right = pads

    zone.transform(QTransform(1.0 / px_w, 0.0, 0.0, -1.0 / px_h,
                              left - minx / px_w, top + maxy / px_h))
    return zone


def _grid_pads(geo: dict, zone_wkb: bytes | None, max_pad: int = 4) -> tuple:


    if not zone_wkb:
        return (0, 0, 0, 0)
    from qgis.core import QgsGeometry

    zone = QgsGeometry()
    zone.fromWkb(zone_wkb)
    if zone.isEmpty():
        return (0, 0, 0, 0)
    box = zone.boundingBox()
    minx, miny, maxx, maxy = geo["bbox"]
    img_h, img_w = geo["img_shape"]
    px_w = (maxx - minx) / max(1, img_w)
    px_h = (maxy - miny) / max(1, img_h)

    def _pad(reach: float, step: float) -> int:
        return min(max_pad, max(0, math.ceil(reach / step - 1e-6))) if step > 0 else 0

    return (_pad(box.yMaximum() - maxy, px_h), _pad(miny - box.yMinimum(), px_h),
            _pad(minx - box.xMinimum(), px_w), _pad(box.xMaximum() - maxx, px_w))


def _read_window(grid, pads: tuple, r0: int, r1: int, c0: int, c1: int):


    import numpy as np

    top, _bottom, left, _right = pads
    height, width = grid.shape
    rows = np.clip(np.arange(r0, r1) - top, 0, height - 1)
    cols = np.clip(np.arange(c0, c1) - left, 0, width - 1)
    lo_r, hi_r = int(rows[0]), int(rows[-1]) + 1
    lo_c, hi_c = int(cols[0]), int(cols[-1]) + 1
    block = np.asarray(grid[lo_r:hi_r, lo_c:hi_c])
    return block[np.ix_(rows - lo_r, cols - lo_c)]


def _zone_mask(zone_px, r0: int, r1: int, c0: int, c1: int):

    if zone_px is None:
        return None
    from .polygon_masks import rasterize_touched_mask

    return rasterize_touched_mask(bytes(zone_px.asWkb()), r1 - r0, c1 - c0,
                                  (float(c0), 1.0, 0.0, float(r0), 0.0, 1.0))


def _merge_slivers(pieces: list, min_area: float = 1.0) -> list:



    from qgis.core import QgsFeature, QgsGeometry, QgsSpatialIndex

    small = [i for i, (_c, g) in enumerate(pieces) if g.area() < min_area]
    if not small:
        return pieces
    index = QgsSpatialIndex()
    for i, (_cid, geom) in enumerate(pieces):
        feat = QgsFeature(i)
        feat.setGeometry(geom)
        index.addFeature(feat)
    geoms = [g for _c, g in pieces]
    classes = [c for c, _g in pieces]
    alive = [True] * len(pieces)
    small_set = set(small)
    for i in sorted(small, key=lambda k: geoms[k].area()):
        best, best_len = None, 0.0
        for j in index.intersects(geoms[i].boundingBox()):
            if j == i or not alive[j] or j in small_set and geoms[j].area() < min_area:
                continue
            shared = geoms[i].intersection(geoms[j])
            length = shared.length() if shared is not None and not shared.isEmpty() else 0.0
            if length > best_len:
                best, best_len = j, length
        if best is None:
            continue
        joined = QgsGeometry.unaryUnion([geoms[best], geoms[i]])
        if joined is None or joined.isEmpty():
            continue
        parts = polygon_parts(joined)
        if len(parts) != 1:
            continue
        geoms[best] = parts[0]
        alive[i] = False
    return [(classes[k], geoms[k]) for k in range(len(pieces)) if alive[k]]


def build_partition_blocked(grid, geo: dict, zone_wkb: bytes | None,
                            min_patch_m2: float, area_scale: float,
                            block: int = BLOCK_PX, halo: int = HALO_PX,
                            smooth=None) -> dict:


    import numpy as np
    from qgis.core import QgsGeometry
    from qgis.PyQt.QtGui import QTransform

    from .land_cover import class_patches_wkb, smooth_class_edges

    smooth = smooth or smooth_class_edges
    minx, miny, maxx, maxy = geo["bbox"]
    img_h, img_w = geo["img_shape"]
    px_w = (maxx - minx) / max(1, img_w)
    px_h = (maxy - miny) / max(1, img_h)
    px_m2 = abs(px_w * px_h) * max(area_scale, 0.0)
    min_px = (min_patch_m2 / px_m2) if px_m2 > 0 else 0.0
    pads = _grid_pads(geo, zone_wkb)
    top, bottom, left, right = pads
    full_h = int(img_h) + top + bottom
    full_w = int(img_w) + left + right
    zone_px = _zone_pixel_geometry(zone_wkb, geo, pads)
    engine = None
    if zone_px is not None:
        engine = QgsGeometry.createGeometryEngine(zone_px.constGet())
        engine.prepareGeometry()

    to_map = QTransform(px_w, 0.0, 0.0, -px_h,
                        minx - left * px_w, maxy + top * px_h)

    done: list = []
    open_parts: dict = {}
    present: set = set()
    for br0 in range(0, full_h, block):
        for bc0 in range(0, full_w, block):
            br1, bc1 = min(full_h, br0 + block), min(full_w, bc0 + block)
            wr0, wr1 = max(0, br0 - halo), min(full_h, br1 + halo)
            wc0, wc1 = max(0, bc0 - halo), min(full_w, bc1 + halo)
            window = _read_window(grid, pads, wr0, wr1, wc0, wc1)
            inside = _zone_mask(zone_px, wr0, wr1, wc0, wc1)
            if inside is not None:
                window = np.where(inside, window, NODATA).astype(np.uint8)
                del inside
            if not (window != NODATA).any():
                continue
            folded = fold_small_patches(
                window, min_px,
                open_edges=(wr0 > 0, wr1 < full_h, wc0 > 0, wc1 < full_w))
            del window
            core = np.ascontiguousarray(
                folded[br0 - wr0:br1 - wr0, bc0 - wc0:bc1 - wc0])
            del folded
            present.update(int(c) for c in np.unique(core) if c != NODATA)
            pieces = []
            for value, wkb in class_patches_wkb(
                    core, (float(bc0), 1.0, 0.0, float(br0), 0.0, 1.0)):
                geom = QgsGeometry()
                geom.fromWkb(wkb)
                if engine is not None and not engine.contains(geom.constGet()):
                    if not engine.intersects(geom.constGet()):
                        continue
                    geom = geom.intersection(zone_px)
                for part in polygon_parts(geom):
                    pieces.append((int(value), part))
            del core


            pieces = _merge_slivers(smooth(pieces, 1.0))
            for cid, part in pieces:
                box = part.boundingBox()
                touches = ((box.yMinimum() <= br0 and br0 > 0)
                           or (box.yMaximum() >= br1 and br1 < full_h)
                           or (box.xMinimum() <= bc0 and bc0 > 0)
                           or (box.xMaximum() >= bc1 and bc1 < full_w))
                if touches:
                    open_parts.setdefault(cid, []).append(part)
                else:
                    done.append((cid, part))
    for cid, parts in open_parts.items():
        for part in polygon_parts(QgsGeometry.unaryUnion(parts)):
            done.append((cid, part))
    patches = []
    areas: dict = {}
    for cid, piece in done:
        piece.transform(to_map)
        for part in polygon_parts(piece):
            area = part.area() * area_scale
            if area <= 0:
                continue
            areas[cid] = areas.get(cid, 0.0) + area
            patches.append((cid, bytes(part.asWkb())))
    return {"patches": patches, "areas": areas, "min_patch_m2": min_patch_m2,
            "classes_present": sorted(present)}
