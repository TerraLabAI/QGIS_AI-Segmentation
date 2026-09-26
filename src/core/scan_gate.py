














from __future__ import annotations

from collections.abc import Sequence

from .tile_manager import TILE_SIZE





SCAN_REF_SIDE = TILE_SIZE


def scan_group(
    base: int, max_group: int, cap: float | None, mupp: float | None
) -> int:














    ceiling = max(2, base, max_group)
    if cap is None or cap <= 0:
        return ceiling
    if mupp is None or mupp <= 0:
        return 0


    candidate = int(cap / mupp + 1e-9)
    if candidate < 2:
        return 0
    return min(candidate, ceiling)


def lattice_indexes(values: list[float], tol: float) -> dict[float, int]:






    uniq: list[float] = []
    for v in sorted(values):
        if not uniq or v - uniq[-1] > tol:
            uniq.append(v)
    out: dict[float, int] = {}
    for v in values:
        out[v] = min(range(len(uniq)), key=lambda i: abs(uniq[i] - v))
    return out


def group_tiles(
    tiles: list[tuple[int, int, int, int]], group: int
) -> list[list[tuple[int, int, int]]]:








    if group < 2 or not tiles:
        return []
    tol = max(8.0, min(t[2] for t in tiles) * 0.05)
    col_of = lattice_indexes([float(t[0]) for t in tiles], tol)
    row_of = lattice_indexes([float(t[1]) for t in tiles], tol)
    blocks: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for idx, (x, y, _w, _h) in enumerate(tiles):
        r, c = row_of[float(y)], col_of[float(x)]
        blocks.setdefault((r // group, c // group), []).append(
            (idx, r % group, c % group))
    return [blocks[k] for k in sorted(blocks)]


def effective_min_px(min_px: int, width: int, height: int) -> int:






    scale = (width * height) / float(SCAN_REF_SIDE * SCAN_REF_SIDE)
    return max(1, int(round(min_px * scale)))


def rle_quadrant_pixels(
    rle_str: str, width: int, height: int, group: int
) -> dict[tuple[int, int], int]:







    counts: dict[tuple[int, int], int] = {}
    if not rle_str or width <= 0 or height <= 0 or group < 1:
        return counts
    cell_w = width / group
    cell_h = height / group
    parts = rle_str.split()
    for i in range(0, len(parts) - 1, 2):
        try:
            off = int(parts[i]) - 1
            ln = int(parts[i + 1])
        except ValueError:
            return {}
        while ln > 0:
            y, x = divmod(off, width)
            if y >= height:
                break
            take = min(ln, width - x)
            qc = min(int(x / cell_w), group - 1)
            qr = min(int(y / cell_h), group - 1)

            seam = int((qc + 1) * cell_w)
            step = min(take, max(1, seam - x))
            key = (qr, qc)
            counts[key] = counts.get(key, 0) + step
            off += step
            ln -= step
    return counts


def quadrant_scores(
    masks: list[tuple[str, float]],
    width: int,
    height: int,
    group: int,
    min_px: int,
) -> dict[tuple[int, int], float]:







    floor = effective_min_px(min_px, width, height)
    best: dict[tuple[int, int], float] = {}
    for rle_str, score in masks:
        for key, n in rle_quadrant_pixels(rle_str, width, height, group).items():
            if n >= floor and score > best.get(key, 0.0):
                best[key] = float(score)
    return best


def scan_response_masks(response: dict) -> list[tuple[str, float]]:






    out: list[tuple[str, float]] = []
    masks = response.get("masks")
    if isinstance(masks, list):
        for m in masks:
            if isinstance(m, dict) and isinstance(m.get("rle"), str):
                try:
                    out.append((m["rle"], float(m.get("score", 0.0))))
                except (TypeError, ValueError):
                    continue
        return out
    rles = response.get("rle")
    scores = response.get("scores")
    if isinstance(rles, list) and isinstance(scores, list):
        for rle_str, score in zip(rles, scores):
            if isinstance(rle_str, str):
                try:
                    out.append((rle_str, float(score)))
                except (TypeError, ValueError):
                    continue
    return out


def classify_block(
    block: Sequence[tuple[int, int, int]],
    response: dict,
    group: int,
    min_score: float,
    min_px: int,
) -> tuple[set[int], set[int]]:





    try:
        width = int(response.get("width") or 0)
        height = int(response.get("height") or 0)
        best = quadrant_scores(
            scan_response_masks(response), width, height, group, min_px)
    except Exception:  # noqa: BLE001
        best = None
    skip: set[int] = set()
    keep: set[int] = set()
    for idx, qr, qc in block:
        if best is not None and best.get((qr, qc), 0.0) < min_score:
            skip.add(idx)
        else:
            keep.add(idx)
    return skip, keep
