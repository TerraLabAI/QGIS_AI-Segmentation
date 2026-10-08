
















from __future__ import annotations

import math
import re

NODATA = 255

GENERIC_MIN_PATCH_M2 = 0.0
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _class_id(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    value = int(value)
    return value if 0 <= value < NODATA else None


def normalize_legend(raw) -> list[dict]:

    out: list[dict] = []
    seen: set[int] = set()
    if not isinstance(raw, (list, tuple)):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        cid = _class_id(entry.get("id"))
        if cid is None or cid in seen:
            continue
        name = entry.get("name")
        color = entry.get("color")
        out.append({
            "id": cid,
            "key": str(entry.get("key") or ""),
            "name": name.strip() if isinstance(name, str) and name.strip() else "",
            "color": color.lower() if isinstance(color, str) and _HEX_RE.match(color) else "",
        })
        seen.add(cid)
    return out


def parse_land_cover_plan(plan) -> dict | None:




    block = plan.get("land_cover") if isinstance(plan, dict) else None
    if not isinstance(block, dict):
        return None
    floor = _number(block.get("min_patch_m2"))
    if floor is None or floor < 0:
        floor = GENERIC_MIN_PATCH_M2
    return {
        "min_patch_m2": floor,
        "class_legend": normalize_legend(block.get("class_legend")),
    }


def response_mask_classes(response: dict) -> list[int | None]:




    masks = response.get("masks") if isinstance(response, dict) else None
    if not isinstance(masks, list):
        return []
    top = response.get("classes")
    top = top if isinstance(top, list) else []
    out: list[int | None] = []
    for i, entry in enumerate(masks):
        cid = _class_id(entry.get("class_id")) if isinstance(entry, dict) else None
        if cid is None and i < len(top):
            cid = _class_id(top[i])
        out.append(cid)
    return out


def decode_tile_labels(response: dict, tile_w: int, tile_h: int):





    import numpy as np

    from .detection_masks import _mask_dimensions, decode_rle_to_mask

    classes = response_mask_classes(response)
    srv_w = response.get("width")
    srv_h = response.get("height")
    dec_h, dec_w = _mask_dimensions(
        srv_h if srv_h is not None else tile_h,
        srv_w if srv_w is not None else tile_w)
    labels = np.full((dec_h, dec_w), NODATA, dtype=np.uint8)
    for entry, cid in zip(response.get("masks") or [], classes):
        if cid is None or not isinstance(entry, dict):
            continue
        mask = decode_rle_to_mask(entry.get("rle", ""), dec_h, dec_w, strict=True)
        labels[mask] = cid
    if (dec_h, dec_w) != (tile_h, tile_w):
        rows = np.minimum((np.arange(tile_h) * dec_h) // max(1, tile_h), dec_h - 1)
        cols = np.minimum((np.arange(tile_w) * dec_w) // max(1, tile_w), dec_w - 1)
        labels = labels[rows][:, cols]
    return labels


def _axis_cuts(intervals: set) -> dict:


    ordered = sorted(intervals)
    out = {}
    for i, (start, end) in enumerate(ordered):
        lo, hi = start, end
        if i > 0:
            prev_end = ordered[i - 1][1]
            if prev_end > start:
                lo = (start + prev_end) // 2
        if i + 1 < len(ordered):
            next_start = ordered[i + 1][0]
            if next_start < end:
                hi = (next_start + end) // 2
        out[(start, end)] = (lo, max(lo, hi))
    return out


class LandCoverMosaic:


    def __init__(self, tiles: list, img_h: int, img_w: int,
                 path: str | None = None) -> None:
        import numpy as np

        self.shape = (int(img_h), int(img_w))
        self.path = path
        if path:


            self.grid = np.memmap(path, dtype=np.uint8, mode="w+", shape=self.shape)
            step = max(1, (64 << 20) // max(1, self.shape[1]))
            for r in range(0, self.shape[0], step):
                self.grid[r:r + step] = NODATA
        else:
            self.grid = np.full(self.shape, NODATA, dtype=np.uint8)
        xs = {(int(x), int(x) + int(w)) for x, _y, w, _h in tiles}
        ys = {(int(y), int(y) + int(h)) for _x, y, _w, h in tiles}
        self._xcut = _axis_cuts(xs)
        self._ycut = _axis_cuts(ys)
        self.tiles_added = 0
        import threading


        self._lock = threading.Lock()
        self._readers = 0
        self._close_pending = False

    def acquire(self):


        with self._lock:
            if self.grid is None or self._close_pending:
                return None
            self._readers += 1
            return self.grid

    def release(self) -> None:

        with self._lock:
            self._readers = max(0, self._readers - 1)
            run_close = self._close_pending and self._readers == 0
        if run_close:
            self._close_now()

    def add_tile(self, rect, labels) -> None:

        if self.grid is None or self._close_pending:
            return
        x, y, w, h = (int(v) for v in rect)
        x0, x1 = self._xcut.get((x, x + w), (x, x + w))
        y0, y1 = self._ycut.get((y, y + h), (y, y + h))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.shape[1]), min(y1, self.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        part = labels[y0 - y:y1 - y, x0 - x:x1 - x]
        self.grid[y0:y0 + part.shape[0], x0:x0 + part.shape[1]] = part
        self.tiles_added += 1

    def core_map_rect(self, rect, geo: dict):

        from qgis.core import QgsRectangle

        x, y, w, h = (int(v) for v in rect)
        x0, x1 = self._xcut.get((x, x + w), (x, x + w))
        y0, y1 = self._ycut.get((y, y + h), (y, y + h))
        minx, miny, maxx, maxy = geo["bbox"]
        img_h, img_w = geo["img_shape"]
        px_w = (maxx - minx) / max(1, img_w)
        px_h = (maxy - miny) / max(1, img_h)
        return QgsRectangle(minx + x0 * px_w, maxy - y1 * px_h,
                            minx + x1 * px_w, maxy - y0 * px_h)

    def close(self) -> None:




        with self._lock:
            if self._readers > 0:
                self._close_pending = True
                return
            self._close_pending = True
        self._close_now()

    def _close_now(self) -> None:
        import os

        grid, self.grid = self.grid, None
        if grid is None:
            return
        mm = getattr(grid, "_mmap", None)
        del grid
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError, OSError):
                return
        if self.path:
            try:
                os.remove(self.path)
            except OSError:  # nosec B110
                pass


def _grid_runs(lab):

    import numpy as np

    height, width = lab.shape
    change = np.empty((height, width), dtype=bool)
    change[:, 0] = True
    np.not_equal(lab[:, 1:], lab[:, :-1], out=change[:, 1:])
    rows, starts = np.nonzero(change)
    del change
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = width
    last_in_row = np.ones(rows.size, dtype=bool)
    last_in_row[:-1] = rows[1:] != rows[:-1]
    ends[last_in_row] = width
    return rows, starts, ends, lab[rows, starts]


def _run_edges(rows, starts, ends, width: int):



    import numpy as np

    n = rows.size
    same_row = rows[1:] == rows[:-1]
    h_a = np.nonzero(same_row)[0]
    h_b = h_a + 1
    gstart = rows * width + starts
    gend = rows * width + ends
    below = np.nonzero(rows > 0)[0]
    up_lo = (rows[below] - 1) * width + starts[below]
    up_hi = (rows[below] - 1) * width + ends[below]
    first = np.searchsorted(gend, up_lo, side="right")
    last = np.searchsorted(gstart, up_hi, side="left") - 1
    count = np.maximum(last - first + 1, 0)
    total = int(count.sum())
    v_b = np.repeat(below, count)
    offsets = np.repeat(np.cumsum(count) - count, count)
    v_a = np.repeat(first, count) + (np.arange(total) - offsets)
    overlap = (np.minimum(ends[v_a], ends[v_b])
               - np.maximum(starts[v_a], starts[v_b]))
    a = np.concatenate([h_a, v_a])
    b = np.concatenate([h_b, v_b])
    length = np.concatenate([np.ones(h_a.size, dtype=np.int64), overlap])
    keep = length > 0
    del n
    return a[keep], b[keep], length[keep]


def _union_components(n: int, a, b):


    import numpy as np

    parent = np.arange(n, dtype=np.int64)
    while True:
        pa, pb = parent[a], parent[b]
        differ = pa != pb
        if not differ.any():
            return parent
        lo = np.minimum(pa[differ], pb[differ])
        hi = np.maximum(pa[differ], pb[differ])
        np.minimum.at(parent, hi, lo)
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped


def fold_small_patches(labels, min_px: float, max_rounds: int = 8,
                       open_edges: tuple = (False, False, False, False)):












    import numpy as np

    if min_px <= 1:
        return labels.copy()
    height, width = labels.shape
    rows, starts, ends, run_class = _grid_runs(labels)
    run_len = (ends - starts).astype(np.int64)
    a, b, length = _run_edges(rows, starts, ends, width)
    top, bottom, left, right = open_edges
    edge_run = np.zeros(rows.size, dtype=bool)
    if top:
        edge_run |= rows == 0
    if bottom:
        edge_run |= rows == height - 1
    if left:
        edge_run |= starts == 0
    if right:
        edge_run |= ends == width
    del rows, starts, ends
    edge_idx = np.nonzero(edge_run)[0]
    del edge_run
    run_class = run_class.astype(np.int16)
    for _round in range(max_rounds):
        valid = (run_class[a] != NODATA) & (run_class[b] != NODATA)
        same = valid & (run_class[a] == run_class[b])
        comp = _union_components(run_class.size, a[same], b[same])
        sizes = np.bincount(comp, weights=run_len, minlength=run_class.size)
        small_comp = (sizes < min_px) & (sizes > 0)
        if edge_idx.size:
            small_comp[comp[edge_idx]] = False
        small_run = small_comp[comp] & (run_class != NODATA)
        if not small_run.any():
            break
        border = valid & ~same
        ea, eb, el = a[border], b[border], length[border]

        src = np.concatenate([comp[ea], comp[eb]])
        dst_class = np.concatenate([run_class[eb], run_class[ea]])
        dst_small = np.concatenate([small_comp[comp[eb]], small_comp[comp[ea]]])
        weight = np.concatenate([el, el])
        keep = small_comp[src]
        src, dst_class, dst_small, weight = (
            src[keep], dst_class[keep], dst_small[keep], weight[keep])
        if src.size == 0:
            break
        target = np.full(run_class.size, -1, dtype=np.int64)


        for prefer_settled in (True, False):
            sel = ~dst_small if prefer_settled else np.ones(src.size, dtype=bool)
            if not sel.any():
                continue
            key = src[sel] * 256 + dst_class[sel]
            uk, inverse = np.unique(key, return_inverse=True)
            sums = np.bincount(inverse, weights=weight[sel])
            ks, kc = uk // 256, uk % 256
            order = np.lexsort((-sums, ks))
            ks, kc = ks[order], kc[order]
            first = np.ones(ks.size, dtype=bool)
            first[1:] = ks[1:] != ks[:-1]
            pick_s, pick_c = ks[first], kc[first]
            free = target[pick_s] < 0
            target[pick_s[free]] = pick_c[free]
        new_class = np.where(small_run & (target[comp] >= 0), target[comp], run_class)
        if np.array_equal(new_class, run_class):
            break
        run_class = new_class.astype(np.int16)
    return np.repeat(run_class.astype(np.uint8), run_len).reshape(height, width)


def class_patches_wkb(labels, geotransform: tuple) -> list[tuple[int, bytes]]:





    import numpy as np

    from .polygon_masks import polygonize_label_raster

    shifted = labels + np.uint8(1)
    return [(value - 1, wkb) for value, wkb in polygonize_label_raster(shifted, geotransform)]


def polygon_parts(geom) -> list:




    from qgis.core import QgsWkbTypes

    try:
        if geom is None or geom.isEmpty():
            return []
        if not geom.isGeosValid():
            fixed = geom.makeValid()
            geom = fixed if fixed is not None and not fixed.isEmpty() else geom
        out = []
        stack = [geom]
        while stack:
            g = stack.pop()
            if g is None or g.isEmpty():
                continue
            parts = g.asGeometryCollection()
            if len(parts) > 1 or (parts and parts[0].isMultipart()):
                stack.extend(parts)
                continue
            flat = QgsWkbTypes.displayString(QgsWkbTypes.flatType(g.wkbType()))
            if flat in ("Polygon", "CurvePolygon") and g.area() > 0:
                out.append(g)
        return out
    except Exception:  # noqa: BLE001
        return []


def smooth_class_edges(patches: list, tolerance: float) -> list:





    from qgis.core import QgsGeometry

    if tolerance <= 0 or len(patches) < 2 or not hasattr(QgsGeometry, "simplifyCoverageVW"):
        return patches
    try:
        parts = [geom for _cid, geom in patches]
        coverage = QgsGeometry.collectGeometry(parts)
        smoothed = coverage.simplifyCoverageVW(tolerance, True)
        pieces = smoothed.asGeometryCollection() if smoothed is not None else []
        if len(pieces) != len(patches):
            return patches
        out = []
        for (cid, _old), piece in zip(patches, pieces):
            if piece is None or piece.isEmpty() or not piece.isGeosValid():
                return patches
            out.append((cid, piece))
        return out
    except Exception:  # noqa: BLE001
        return patches


def legend_rows(class_ids, legend: list[dict], fallback_name: str) -> list[dict]:


    from .class_symbology import interpolate_colors, legend_ramp_anchors

    by_id = {e["id"]: e for e in legend}
    ids = sorted(set(by_id) | {int(c) for c in class_ids})
    ramp = interpolate_colors(legend_ramp_anchors(), max(2, len(ids)))
    rows = []
    for i, cid in enumerate(ids):
        entry = by_id.get(cid, {})
        rows.append({
            "id": cid,
            "key": entry.get("key") or "",
            "name": entry.get("name") or fallback_name.format(n=cid),
            "color": entry.get("color") or ramp[i % len(ramp)],
        })
    return rows


def _area_weights(areas) -> list[float]:

    values = []
    for area in areas:
        try:
            value = float(area)
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        values.append(max(0.0, value) if math.isfinite(value) else 0.0)
    scale = max(values, default=0.0)
    if scale <= 0:
        return [0.0 for _ in values]
    return [a / scale for a in values]


def whole_percent_shares(areas: list) -> list[int]:


    weights = _area_weights(areas)
    total = sum(weights)
    if total <= 0:
        return [0 for _ in weights]

    raw = [100.0 * (a / total) for a in weights]
    floors = [math.floor(v) for v in raw]
    left = 100 - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:left]:
        floors[i] += 1
    return floors



_SUMMARY_GROUPS = (("building", "impervious"), ("low_veg", "tree"), ("water",))


def land_cover_summary(rows: list) -> tuple | None:


    keys = {r.get("key") for r in rows}
    if not all(k in keys for group in _SUMMARY_GROUPS for k in group):
        return None
    weights = _area_weights(r.get("area_m2") for r in rows)
    total = sum(weights)
    if total <= 0:
        return None
    return tuple(
        int(round(100.0 * sum(
            weight for r, weight in zip(rows, weights)
            if r.get("key") in group) / total))
        for group in _SUMMARY_GROUPS)


def table_tsv(rows: list[dict], total_m2: float, header: tuple, total_label: str) -> str:


    import csv
    import io

    shares = whole_percent_shares([float(r["area_m2"]) for r in rows])
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    for row, share in zip(rows, shares):
        writer.writerow((row["name"], f"{row['area_m2']:.0f}", share))
    writer.writerow((total_label, f"{total_m2:.0f}", 100 if sum(shares) else 0))
    return output.getvalue()



GRID_FILE_PREFIX = "aiseg_lc_"
GRID_FILE_SUFFIX = ".grid"


def sweep_stale_grid_files(max_age_s: float = 86400.0) -> int:



    import os
    import tempfile
    import time

    removed = 0
    try:
        folder = tempfile.gettempdir()
        now = time.time()
        for entry in os.scandir(folder):
            name = entry.name
            if not (name.startswith(GRID_FILE_PREFIX) and name.endswith(GRID_FILE_SUFFIX)):
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if now - entry.stat(follow_symlinks=False).st_mtime < max_age_s:
                    continue
                os.remove(os.path.join(folder, name))
                removed += 1
            except OSError:  # nosec B112
                continue
    except OSError:  # nosec B110
        pass
    return removed
