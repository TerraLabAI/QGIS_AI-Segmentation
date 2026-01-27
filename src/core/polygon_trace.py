








































from __future__ import annotations

import struct

import numpy as np



_EAST, _SOUTH, _WEST, _NORTH = 0, 1, 2, 3



_CORNER = np.zeros(16, dtype=bool)
_OUT_DIR = np.full(16, -1, dtype=np.int8)
for _cfg, _dir in ((1, _SOUTH), (2, _WEST), (4, _EAST), (8, _NORTH),
                   (7, _WEST), (11, _NORTH), (13, _SOUTH), (14, _EAST)):
    _CORNER[_cfg] = True
    _OUT_DIR[_cfg] = _dir

_IN_SIDE = np.full(16, -1, dtype=np.int8)
for _cfg, _side in ((1, _EAST), (2, _SOUTH), (4, _NORTH), (8, _WEST),
                    (7, _NORTH), (11, _EAST), (13, _WEST), (14, _SOUTH)):
    _IN_SIDE[_cfg] = _side


_OUTER_START = 1
_HOLE_START = 14







_SADDLE_PASSES = {
    (9, True): ((_WEST, _SOUTH, 0), (_EAST, _NORTH, 0)),
    (9, False): ((_EAST, _SOUTH, _OUTER_START), (_WEST, _NORTH, 0)),
    (6, True): ((_SOUTH, _EAST, _HOLE_START), (_NORTH, _WEST, 0)),
    (6, False): ((_NORTH, _EAST, 0), (_SOUTH, _WEST, 0)),
}


class CropRings:











    __slots__ = ("rings", "outer", "bottom", "mask", "pixels", "boxes", "saddle_free")

    def __init__(self, rings: list, outer: list, bottom: list,
                 mask: np.ndarray | None = None, boxes: list | None = None) -> None:
        self.rings = rings
        self.outer = outer
        self.bottom = bottom
        self.mask = mask

        self.saddle_free = True

        if boxes is None:
            boxes = [(int(r.min()), int(r.max()), int(c.min()), int(c.max()))
                     for r, c in rings]
        self.boxes = boxes



        self.pixels = None

    @property
    def has_holes(self) -> bool:
        return not all(self.outer)

    def after_pinhole_fill(self, max_hole_px, fill) -> CropRings | None:












        if self.mask is None:
            return None
        holes = [ring for ring, o in zip(self.rings, self.outer) if not o]
        outers = [ring for ring, o in zip(self.rings, self.outer) if o]
        if len(outers) > 1:
            boxes = [b for b, o in zip(self.boxes, self.outer) if not o]
            for rows, cols in outers:
                r0, c0 = int(rows[0]), int(cols[0])
                for (lo_r, hi_r, lo_c, hi_c), hole in zip(boxes, holes):
                    if (lo_r <= r0 < hi_r and lo_c <= c0 < hi_c
                            and _inside_ring(r0, c0, hole)):
                        return None
        keep = []
        for (rows, cols), is_outer in zip(self.rings, self.outer):
            if is_outer:
                keep.append(True)
                continue
            keep.append(not _ring_pixel_area(rows, cols) <= max_hole_px)
        if all(keep):
            return self
        out = CropRings([r for r, k in zip(self.rings, keep) if k],
                        [o for o, k in zip(self.outer, keep) if k],
                        [b for b, k in zip(self.bottom, keep) if k],
                        boxes=[b for b, k in zip(self.boxes, keep) if k])
        out.saddle_free = self.saddle_free
        mask = self.mask
        out.pixels = lambda: fill(mask)
        return out

    def polygons(self, skip_box: float = 0.0) -> list | None:














        n_outer = sum(1 for o in self.outer if o)
        if n_outer == 0:
            return []
        outer_ids = [k for k, o in enumerate(self.outer) if o]
        small = set()
        if skip_box > 0.0:
            for k in outer_ids:
                lo_r, hi_r, lo_c, hi_c = self.boxes[k]
                if (hi_r - lo_r) * (hi_c - lo_c) < skip_box:
                    small.add(k)
        if n_outer == 1:
            if small:
                return []
            first = outer_ids[0]
            return [[self.rings[first]] + [
                r for r, o in zip(self.rings, self.outer) if not o]]
        members = {k: [self.rings[k]] for k in outer_ids}
        if self.has_holes:
            boxes = self.boxes
            for ring, is_outer in zip(self.rings, self.outer):
                if is_outer:
                    continue
                r, c = int(ring[0][0]) - 1, int(ring[1][0])
                owner, owner_area = None, None
                for k in outer_ids:
                    lo_r, hi_r, lo_c, hi_c = boxes[k]
                    if not (lo_r <= r < hi_r and lo_c <= c < hi_c):
                        continue
                    if not _inside_ring(r, c, self.rings[k]):
                        continue
                    area = _ring_pixel_area(*self.rings[k])
                    if owner_area is None or area < owner_area:
                        owner, owner_area = k, area
                if owner is None:
                    return None
                members[owner].append(ring)
        kept = [k for k in outer_ids if k not in small]
        bottoms = [self.bottom[k] for k in kept]
        if len(set(bottoms)) == len(bottoms):
            order = sorted(kept, key=self.bottom.__getitem__)
        else:
            mask = self.mask
            if mask is None and self.pixels is not None:
                mask = self.pixels()
            if mask is None:
                return None
            part_of = _gdal_part_ids(mask)
            keys = {k: (self.bottom[k], part_of(int(self.rings[k][0][0]),
                                                int(self.rings[k][1][0])))
                    for k in kept}
            order = sorted(kept, key=keys.__getitem__)
        return [members[k] for k in order]


def _ring_pixel_area(rows: np.ndarray, cols: np.ndarray) -> int:

    twice = (int(np.dot(cols[:-1], rows[1:])) - int(np.dot(rows[:-1], cols[1:]))
             + int(cols[-1]) * int(rows[0]) - int(rows[-1]) * int(cols[0]))
    return abs(twice) // 2


def _inside_ring(row: int, col: int, ring: tuple) -> bool:


    rows, cols = ring
    rows_next = np.roll(rows, -1)
    cols_next = np.roll(cols, -1)
    cy, cx = row + 0.5, col + 0.5
    crossing = ((cols == cols_next) & (cols > cx)
                & (np.minimum(rows, rows_next) < cy) & (cy < np.maximum(rows, rows_next)))
    return bool(np.count_nonzero(crossing) & 1)


def _mask_runs(s: np.ndarray) -> tuple:

    h, w = s.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = s
    flat = padded.ravel()
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = edges[0::2]
    stops = edges[1::2]
    return starts // (w + 2), starts % (w + 2) - 1, stops % (w + 2) - 2


def _part_labels(s: np.ndarray):





    rows, first, last = _mask_runs(s)
    n = int(rows.size)
    parent = np.arange(n)
    if n > 1:



        key_first = rows * (s.shape[1] + 2) + first
        lo = np.searchsorted(key_first, (rows + 1) * (s.shape[1] + 2) + 0, side="left")
        hi = np.searchsorted(key_first, (rows + 1) * (s.shape[1] + 2) + last, side="right")
        counts = hi - lo
        src = np.repeat(np.arange(n), counts)
        dst = (np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
               + np.repeat(lo, counts))
        if src.size:
            touch = last[dst] >= first[src]
            src, dst = src[touch], dst[touch]
        while src.size:
            ra, rb = parent[src], parent[dst]
            moved = ra != rb
            if not moved.any():
                break
            np.minimum.at(parent, np.maximum(ra, rb)[moved], np.minimum(ra, rb)[moved])
            while True:
                nxt = parent[parent]
                if not (nxt != parent).any():
                    break
                parent = nxt
    key = rows * (s.shape[1] + 2) + first

    def part_at(r: int, c: int) -> int:
        k = int(np.searchsorted(key, r * (s.shape[1] + 2) + c, side="right")) - 1
        return int(parent[k])

    return part_at


def _gdal_part_ids(mask: np.ndarray):










    s = np.asarray(mask, dtype=bool)
    rows, first, last = _mask_runs(s)
    run_row = rows.tolist()
    run_a = first.tolist()
    run_b = last.tolist()


    n_runs = len(run_row)
    id_map: list = list(range(n_runs))
    run_id: list = [0] * n_runs
    next_id = 0
    prev_lo = prev_hi = 0
    cur_lo = 0
    row = -1
    for k, (r, a, b) in enumerate(zip(run_row, run_a, run_b)):
        if r != row:
            if r == row + 1:
                prev_lo, prev_hi = cur_lo, k
            else:
                prev_lo = prev_hi = k
            cur_lo = k
            row = r
        rid = -1
        for p in range(prev_lo, prev_hi):
            pb = run_b[p]
            if pb < a:
                continue
            pa = run_a[p]
            if pa > b:
                break
            src = run_id[p]
            if rid < 0:
                if pa <= a:
                    rid = src
                else:
                    rid = next_id
                    next_id += 1


            if (pa if pa > a else a + 1) > (pb if pb < b else b):  # noqa: FURB136
                continue
            src_root = src
            while id_map[src_root] != src_root:
                src_root = id_map[src_root]
            final = rid
            while id_map[final] != final:
                final = id_map[final]
            if src_root == final:
                continue
            cur = rid
            while id_map[cur] != cur:
                nxt = id_map[cur]
                id_map[cur] = final
                cur = nxt
            while id_map[src] != src:
                nxt = id_map[src]
                id_map[src] = final
                src = nxt
            id_map[src] = final
        if rid < 0:
            rid = next_id
            next_id += 1
        run_id[k] = rid

    def root(k: int) -> int:
        while id_map[k] != k:
            k = id_map[k]
        return k

    rows_arr = np.asarray(run_row)
    a_arr = np.asarray(run_a)

    def part_of(r: int, c: int) -> int:
        lo = int(np.searchsorted(rows_arr, r, side="left"))
        hi = int(np.searchsorted(rows_arr, r, side="right"))
        k = lo + int(np.searchsorted(a_arr[lo:hi], c, side="right")) - 1
        return root(run_id[k])

    return part_of


def trace_crop_rings(mask: np.ndarray, saddles: bool = True) -> CropRings | None:







    return trace_crops_rings([mask], saddles)[0]


def _crop_corners(mask, saddles: bool) -> tuple | None:


    s = np.asarray(mask)
    if s.ndim != 2:
        return None
    if s.dtype != np.bool_:
        s = s != 0
    h, w = s.shape
    if h < 3 or w < 3:
        return (s, None, None, None, None) if not s.any() else None
    if s[::h - 1].any() or s[:, ::w - 1].any():
        return None



    above_left, above = s[:-1, :-1], s[:-1, 1:]
    left, here = s[1:, :-1], s[1:, 1:]
    top = above_left ^ above
    saddle = top & ~((above_left ^ here) | (above ^ left))
    has_saddle = bool(saddle.any())
    if has_saddle and not saddles:
        return None
    ri, ci = np.nonzero((top ^ (left ^ here)) | saddle if has_saddle else top ^ (left ^ here))
    s8 = s.view(np.uint8)
    kind = ((s8[ri, ci] << 3) | (s8[ri, ci + 1] << 2)
            | (s8[ri + 1, ci] << 1) | s8[ri + 1, ci + 1])
    passes = None
    if has_saddle:
        part_at = _part_labels(s)
        passes = []
        for k in np.flatnonzero((kind == 6) | (kind == 9)).tolist():
            r, c = int(ri[k]), int(ci[k])
            if kind[k] == 9:
                same = part_at(r, c) == part_at(r + 1, c + 1)
            else:
                same = part_at(r, c + 1) == part_at(r + 1, c)
            passes.append((k, _SADDLE_PASSES[(int(kind[k]), same)]))
    return s, ri, ci, kind, passes


def trace_crops_rings(masks: list, saddles: bool = True) -> list:








    results: list = [None] * len(masks)
    parts = []
    for pos, mask in enumerate(masks):
        corners = _crop_corners(mask, saddles)
        if corners is None:
            continue
        s, ri, ci, kind, passes = corners
        if ri is None or ri.size == 0:
            results[pos] = CropRings([], [], [], s)
            continue
        parts.append((pos, s, ri, ci, kind, passes))
    if not parts:
        return results
    sizes = [p[2].size for p in parts]
    nv = int(sum(sizes))
    vcrop = np.repeat(np.arange(len(parts)), sizes)
    vr = np.concatenate([p[2] for p in parts])
    vc = np.concatenate([p[3] for p in parts])
    vkind = np.concatenate([p[4] for p in parts])


    per_vertex = np.ones(nv, dtype=np.intp)
    node_in = _IN_SIDE[vkind].astype(np.intp)
    node_out = _OUT_DIR[vkind].astype(np.intp)
    node_kind = vkind.astype(np.intp)
    second_in = np.full(nv, -1, dtype=np.intp)
    saddle_rows = []
    offset = 0
    for size, part in zip(sizes, parts):
        if part[5]:
            for k, (first, second) in part[5]:
                saddle_rows.append((offset + k, first, second))
        offset += size
    extra = []
    for v, first, second in saddle_rows:
        per_vertex[v] = 2
        node_in[v], node_out[v], node_kind[v] = first
        second_in[v] = second[0]
        extra.append((v, second))
    first_node = np.concatenate(([0], np.cumsum(per_vertex)[:-1]))
    n = int(per_vertex.sum())
    vertex = np.repeat(np.arange(nv), per_vertex)
    n_in = np.empty(n, dtype=np.intp)
    n_out = np.empty(n, dtype=np.intp)
    n_kind = np.empty(n, dtype=np.intp)
    n_in[first_node] = node_in
    n_out[first_node] = node_out
    n_kind[first_node] = node_kind
    for v, (side_in, dir_out, start_kind) in extra:
        n_in[first_node[v] + 1] = side_in
        n_out[first_node[v] + 1] = dir_out
        n_kind[first_node[v] + 1] = start_kind
    ncrop = vcrop[vertex]
    nr = vr[vertex]
    nc = vc[vertex]
    idx = np.arange(n)



    vidx = np.arange(nv)
    by_col = np.lexsort((vr, vc, vcrop))
    pos_col = np.empty(nv, dtype=np.intp)
    pos_col[by_col] = vidx
    v_here = vertex
    target = v_here + 1
    west = n_out == _WEST
    target[west] = v_here[west] - 1
    south = n_out == _SOUTH
    target[south] = by_col[np.minimum(pos_col[v_here[south]] + 1, nv - 1)]
    north = n_out == _NORTH
    target[north] = by_col[np.maximum(pos_col[v_here[north]] - 1, 0)]
    np.clip(target, 0, nv - 1, out=target)
    arrive = (n_out + 2) % 4
    succ = first_node[target] + (arrive == second_in[target])



    horiz = (n_out == _EAST) | west
    bad = ((vcrop[target] != ncrop) | (n_out < 0)
           | (horiz & (vr[target] != nr)) | (~horiz & (vc[target] != nc))
           | (n_in[succ] != arrive))
    indeg = np.bincount(succ, minlength=n)
    bad |= indeg[succ] != 1
    bad_crop = np.zeros(len(parts), dtype=bool)
    bad_crop[ncrop[bad]] = True


    broken = bad_crop[ncrop]
    succ[broken] = idx[broken]




    longest = int(np.bincount(ncrop).max())
    rep = idx.copy()
    nxt = succ.copy()
    span = 1
    while span < longest:
        rep = np.minimum(rep, rep[nxt])
        nxt = nxt[nxt]
        span *= 2
    start = rep == idx

    pred = np.empty(n, dtype=np.intp)
    pred[succ] = idx
    dist = np.where(start, 0, 1)
    back = np.where(start, idx, pred)
    span = 1
    while span < longest:
        dist = dist + dist[back]
        back = back[back]
        span *= 2
    order = np.lexsort((dist, rep))
    starts = np.flatnonzero(start)
    first_kind = n_kind[starts]
    ring_crop = ncrop[starts]
    bad_crop[ring_crop[(first_kind != _OUTER_START) & (first_kind != _HOLE_START)]] = True
    counts = np.bincount(rep, minlength=n)[starts]
    ring_rows = (nr + 1)[order]
    ring_cols = (nc + 1)[order]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    ring_starts = offsets[:-1]
    top_row = np.minimum.reduceat(ring_rows, ring_starts).tolist()
    low_row = np.maximum.reduceat(ring_rows, ring_starts)

    bottom = (low_row - 1).tolist()
    low_row = low_row.tolist()
    left_col = np.minimum.reduceat(ring_cols, ring_starts).tolist()
    right_col = np.maximum.reduceat(ring_cols, ring_starts).tolist()
    bounds = offsets.tolist()
    outer = (first_kind == _OUTER_START).tolist()
    bad_list = bad_crop.tolist()


    ring_lo = np.searchsorted(ring_crop, np.arange(len(parts)), side="left").tolist()
    ring_hi = np.searchsorted(ring_crop, np.arange(len(parts)), side="right").tolist()
    for cid, part in enumerate(parts):
        if bad_list[cid]:
            continue
        lo_k, hi_k = ring_lo[cid], ring_hi[cid]
        rings = CropRings(
            [(ring_rows[bounds[k]:bounds[k + 1]], ring_cols[bounds[k]:bounds[k + 1]])
             for k in range(lo_k, hi_k)],
            outer[lo_k:hi_k], bottom[lo_k:hi_k], part[1],
            [(top_row[k], low_row[k], left_col[k], right_col[k])
             for k in range(lo_k, hi_k)])
        rings.saddle_free = not part[5]
        results[part[0]] = rings
    return results


def fallback_contours(outline: CropRings) -> list | None:













    if not outline.saddle_free:
        return None
    found = []
    for k, (rows, cols) in enumerate(outline.rings):

        r = np.concatenate((rows[:1], rows[:0:-1]))
        c = np.concatenate((cols[:1], cols[:0:-1]))
        r_next = np.roll(r, -1)
        c_next = np.roll(c, -1)
        dr = r_next - r
        dc = c_next - c
        lengths = np.abs(dr) + np.abs(dc)
        total = int(lengths.sum())
        seg = np.repeat(np.arange(r.size), lengths)
        step = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        pr = r[seg] + np.sign(dr)[seg] * step
        pc = c[seg] + np.sign(dc)[seg] * step
        east = np.flatnonzero(np.sign(dc)[seg] > 0)
        first = east[np.lexsort((pc[east], pr[east]))[0]]
        pr = np.roll(pr, -first)
        pc = np.roll(pc, -first)
        found.append((int(pr[0]), int(pc[0]), pr, pc, k))
    found.sort(key=lambda item: (item[0], item[1]))
    return [(pr, pc, k) for _r0, _c0, pr, pc, k in found]






_SADDLE_OUT = {9: (0, 2), 6: (1, 3)}


def _walk_points(rings: list) -> tuple:




    sizes = np.array([r.size for r, _c in rings], dtype=np.intp)
    first = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    total = int(sizes.sum())
    ring_of = np.repeat(np.arange(sizes.size), sizes)
    local = np.arange(total) - first[ring_of]
    size_of = sizes[ring_of]
    all_r = np.concatenate([r for r, _c in rings]).astype(np.intp)
    all_c = np.concatenate([c for _r, c in rings]).astype(np.intp)

    back = first[ring_of] + (size_of - local) % size_of
    r = all_r[back]
    c = all_c[back]
    ahead = np.where(local == size_of - 1, first[ring_of], np.arange(total) + 1)
    dr = r[ahead] - r
    dc = c[ahead] - c
    lengths = np.abs(dr) + np.abs(dc)
    seg = np.repeat(np.arange(total), lengths)
    step = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    sr = np.sign(dr)[seg]
    sc = np.sign(dc)[seg]
    heading = np.where(sc > 0, 0, np.where(sr > 0, 1, np.where(sc < 0, 2, 3)))
    starts = np.concatenate(([0], np.cumsum(np.add.reduceat(lengths, first))))
    return r[seg] + sr * step, c[seg] + sc * step, heading, starts


def walked_contours(outline: CropRings) -> list | None:





















    import heapq

    mask = outline.mask
    if mask is None:
        return None
    s = np.asarray(mask)
    if s.dtype != np.bool_:
        s = s != 0
    h, w = s.shape
    if h < 3 or w < 3:
        return None
    w1 = w + 1
    plane = (h + 1) * w1
    above_left, above = s[:-1, :-1], s[:-1, 1:]
    left, here = s[1:, :-1], s[1:, 1:]
    kind = np.zeros((h + 1, w + 1), dtype=np.int8)
    kind[1:h, 1:w][above_left & here & ~above & ~left] = 9
    kind[1:h, 1:w][above & left & ~above_left & ~here] = 6

    arcs = []
    out_of: dict = {}
    heap = []
    all_r, all_c, all_head, starts = _walk_points(outline.rings)
    all_key = all_head * plane + all_r * w1 + all_c
    all_sk = kind[all_r, all_c]
    lo = starts[:-1]
    passes = np.add.reduceat(all_sk != 0, lo)

    lens = np.diff(starts)
    smallest = np.minimum.reduceat(all_key, lo)
    at = np.flatnonzero(all_key == np.repeat(smallest, lens))
    if at.size != lens.size:
        return None
    for k in np.flatnonzero(passes == 0).tolist():
        heap.append((int(smallest[k]), 2, k, int(at[k] - lo[k])))
    bounds_all = starts.tolist()
    for k in np.flatnonzero(passes).tolist():
        sl = slice(bounds_all[k], bounds_all[k + 1])
        pr, pc, head, key, sk = all_r[sl], all_c[sl], all_head[sl], all_key[sl], all_sk[sl]
        cut = np.flatnonzero(sk)
        first = int(cut[0])
        if first:
            pr, pc, head, key, sk = (np.roll(a, -first) for a in (pr, pc, head, key, sk))
            cut = cut - first
        n = int(pr.size)
        vert = (pr * w1 + pc).tolist()
        heads = head.tolist()
        kinds = sk.tolist()
        bounds = cut.tolist() + [n]
        for a, b in zip(bounds[:-1], bounds[1:]):
            v, out_h, sad = vert[a], heads[a], kinds[a]
            if out_h not in _SADDLE_OUT[sad]:
                return None
            slot = out_of.get(v)
            if slot is None:
                slot = out_of[v] = {}
                heap.append((_SADDLE_OUT[sad][0] * plane + v, 0, v, 0))
            if out_h in slot:
                return None
            slot[out_h] = len(arcs)
            if b - a > 1:
                j = a + 1 + int(np.argmin(key[a + 1:b]))
                heap.append((int(key[j]), 1, len(arcs), j - a))
            arcs.append((pr[a:b], pc[a:b], vert[b % n], heads[b - 1]))
    heapq.heapify(heap)
    if len(out_of) != int(np.count_nonzero(kind)) or any(len(v) != 2 for v in out_of.values()):
        return None

    def leave(vertex: int, heading_in: int):

        slot = out_of.get(vertex)
        if not slot:
            return None
        right = (heading_in + 1) & 3
        if right in slot:
            return slot.pop(right)
        return slot.pop((heading_in + 3) & 3, None)

    done = [False] * len(arcs)
    contours = []
    while heap:
        _key, what, which, where = heapq.heappop(heap)
        if what == 2:
            o, e = bounds_all[which], bounds_all[which + 1]
            j = o + where
            contours.append((np.concatenate((all_r[j:e], all_r[o:j])),
                             np.concatenate((all_c[j:e], all_c[o:j]))))
            continue
        if what == 0:
            slot = out_of[which]
            if not slot:
                continue
            arc = slot.pop(min(slot))
            stop, head_arc = which, None
            pieces = [arc]
        else:
            if done[which]:
                continue
            arc = which
            stop, head_arc = None, which
            pieces = []
        done[arc] = True
        cur = arc
        for _ in range(len(arcs) + 1):
            end, heading_in = arcs[cur][2], arcs[cur][3]
            if end == stop:
                break
            nxt = leave(end, heading_in)
            if nxt is None:
                return None
            if nxt == head_arc:
                break
            if done[nxt]:
                return None
            done[nxt] = True
            pieces.append(nxt)
            cur = nxt
        else:
            return None
        if what == 0 and out_of[which]:

            heapq.heappush(heap, (_key, 0, which, 0))
        if head_arc is None:
            rows = np.concatenate([arcs[k][0] for k in pieces])
            cols = np.concatenate([arcs[k][1] for k in pieces])
        else:

            tail_r, tail_c = arcs[head_arc][0], arcs[head_arc][1]
            rows = np.concatenate([tail_r[where:]] + [arcs[k][0] for k in pieces]
                                  + [tail_r[:where]])
            cols = np.concatenate([tail_c[where:]] + [arcs[k][1] for k in pieces]
                                  + [tail_c[:where]])
        contours.append((rows, cols))
    if not all(done) or any(out_of.values()):
        return None
    return contours


def exact_vertex_axis(origin, step, index: np.ndarray,
                      fused: bool) -> np.ndarray:











    k = np.asarray(index, dtype=np.float64)
    if not fused:
        return origin + k * step
    a = np.asarray(origin, dtype=np.float64)
    b = np.asarray(step, dtype=np.float64)
    hi = (np.ascontiguousarray(b).view(np.uint64)
          & np.uint64(~((1 << 21) - 1) & ((1 << 64) - 1))).view(np.float64)
    lo = b - hi
    p_hi = k * hi
    p_lo = k * lo

    def two_sum(x, y):
        s = x + y
        bb = s - x
        return s, (x - (s - bb)) + (y - bb)

    s, e1 = two_sum(a, p_hi)
    t, e2 = two_sum(e1, p_lo)
    r, e3 = two_sum(s, t)
    w, e4 = two_sum(e3, e2)
    up = np.nextafter(r, np.inf) - r
    dn = r - np.nextafter(r, -np.inf)
    half_up = up * 0.5
    half_dn = dn * 0.5
    odd = (r.view(np.uint64) & np.uint64(1)).astype(bool)
    go_up = (w > half_up) | ((w == half_up) & ((e4 > 0) | ((e4 == 0) & odd)))
    go_dn = (w < -half_dn) | ((w == -half_dn) & ((e4 < 0) | ((e4 == 0) & odd)))
    out = np.where(go_up, r + up, np.where(go_dn, r - dn, r))

    tiny = np.abs(r) < 1e-200
    if tiny.any():


        from fractions import Fraction

        a_all = np.broadcast_to(a, k.shape)
        b_all = np.broadcast_to(b, k.shape)
        out = out.copy()
        for pos in np.flatnonzero(tiny).tolist():
            out[pos] = float(Fraction(float(a_all[pos]))
                             + int(k[pos]) * Fraction(float(b_all[pos])))
    return out


def vertex_tables(gt: tuple, height: int, width: int, rule: tuple) -> tuple:



    return vertex_tables_many([(gt, height, width)], rule)[0]


def vertex_tables_many(rasters: list, rule: tuple) -> list:


    if not rasters:
        return []
    widths = [w + 1 for _gt, _h, w in rasters]
    heights = [h + 1 for _gt, h, _w in rasters]
    col = np.concatenate([np.arange(n) for n in widths])
    row = np.concatenate([np.arange(n) for n in heights])
    x0 = np.repeat([gt[0] for gt, _h, _w in rasters], widths)
    dx = np.repeat([gt[1] for gt, _h, _w in rasters], widths)
    y0 = np.repeat([gt[3] for gt, _h, _w in rasters], heights)
    dy = np.repeat([gt[5] for gt, _h, _w in rasters], heights)
    xs = np.split(exact_vertex_axis(x0, dx, col, rule[0]), np.cumsum(widths)[:-1])
    ys = np.split(exact_vertex_axis(y0, dy, row, rule[1]), np.cumsum(heights)[:-1])
    return list(zip(xs, ys))


def polygon_wkb(polygon: list, tables: tuple, row0: int, col0: int) -> bytes:





    xs, ys = tables
    parts = [struct.pack("<BII", 1, 3, len(polygon))]
    for rows, cols in polygon:
        n = int(rows.size)
        xy = np.empty((n + 1, 2), dtype="<f8")
        xy[:n, 0] = xs[cols + col0]
        xy[:n, 1] = ys[rows + row0]
        xy[n] = xy[0]
        parts.append(struct.pack("<I", n + 1))
        parts.append(xy.tobytes())
    return b"".join(parts)


_RULE: list = []


def _probe_masks() -> list:





    one = np.zeros((72, 90), dtype=bool)
    for r in range(2, 66):
        one[r, 2:2 + r] = True
    one[40:46, 10:16] = False
    two = np.zeros((40, 40), dtype=bool)
    two[3:20, 4:30] = True
    two[24:37, 8:12] = True
    three = np.zeros((20, 60), dtype=bool)
    three[3, 20:23] = True
    three[4:12, 5:26] = True
    three[7:9, 10:13] = False
    three[3:12, 40:50] = True

    four = np.zeros((26, 44), dtype=bool)
    four[3:6, 3:6] = True
    four[6:9, 6:9] = True
    four[3, 6:11] = True
    four[3:9, 10] = True
    four[8, 9] = True
    four[3:6, 30:33] = True
    four[6:9, 27:30] = True
    four[3, 24:30] = True
    four[3:9, 24] = True
    four[8, 25:27] = True
    four[14:17, 3:6] = True
    four[17:20, 6:9] = True
    four[14:17, 30:33] = True
    four[17:20, 27:30] = True
    return [one, two, three, four]


def gdal_vertex_rule() -> tuple | None:










    if _RULE:
        return _RULE[0]
    rule = None
    try:
        rule = _calibrate()
    except Exception:  # noqa: BLE001
        rule = None
    _RULE.append(rule)
    return rule


def _calibrate() -> tuple | None:
    from rasterio.features import shapes
    from rasterio.transform import from_bounds

    masks = _probe_masks()
    traced_probes = [trace_crop_rings(m) for m in masks]
    if any(t is None for t in traced_probes):
        return None
    px = _discriminating_pixel(masks[0], traced_probes[0], from_bounds)
    if px is None:
        return None
    ok_rules = None
    for mask, traced in zip(masks, traced_probes):
        h, w = mask.shape
        transform = from_bounds(_PROBE_MINX, _PROBE_MAXY - h * px,
                                _PROBE_MINX + w * px, _PROBE_MAXY, w, h)
        gt = transform.to_gdal()
        lab = mask.astype(np.int32)
        got = []
        for geom, _value in shapes(lab, mask=lab > 0, connectivity=4,
                                   transform=transform):
            if geom.get("type") != "Polygon":
                return None
            got.append([[(float(x), float(y)) for x, y in ring]
                        for ring in geom["coordinates"]])
        polygons = traced.polygons()
        if not polygons or len(polygons) != len(got):
            return None
        matching = set()
        for rule in ((True, True), (False, False), (True, False), (False, True)):
            tables = vertex_tables(gt, h, w, rule)
            if all(_wkb_rings(polygon_wkb(poly, tables, 0, 0)) == ref
                   for poly, ref in zip(polygons, got)):
                matching.add(rule)
        ok_rules = matching if ok_rules is None else ok_rules & matching


    if not ok_rules or len(ok_rules) != 1:
        return None
    return next(iter(ok_rules))




_PROBE_MINX = -9929363.51742432
_PROBE_MAXY = 6546923.231844369


def _discriminating_pixel(mask: np.ndarray, traced: CropRings, from_bounds) -> float | None:


    h, w = mask.shape
    rows = np.unique(np.concatenate([r for r, _c in traced.rings]))
    cols = np.unique(np.concatenate([c for _r, c in traced.rings]))
    for step in range(400):
        px = 0.1 + 0.00731 * step
        gt = from_bounds(_PROBE_MINX, _PROBE_MAXY - h * px,
                         _PROBE_MINX + w * px, _PROBE_MAXY, w, h).to_gdal()
        dx = np.count_nonzero(exact_vertex_axis(gt[0], gt[1], cols, True)
                              != exact_vertex_axis(gt[0], gt[1], cols, False))
        dy = np.count_nonzero(exact_vertex_axis(gt[3], gt[5], rows, True)
                              != exact_vertex_axis(gt[3], gt[5], rows, False))
        if dx >= 2 and dy >= 2:
            return px
    return None


def _wkb_rings(blob: bytes) -> list:

    n_rings = struct.unpack_from("<I", blob, 5)[0]
    off = 9
    rings = []
    for _ in range(n_rings):
        count = struct.unpack_from("<I", blob, off)[0]
        off += 4
        pts = np.frombuffer(blob, dtype="<f8", count=2 * count, offset=off)
        off += 16 * count
        rings.append([(float(pts[i]), float(pts[i + 1])) for i in range(0, 2 * count, 2)])
    return rings
