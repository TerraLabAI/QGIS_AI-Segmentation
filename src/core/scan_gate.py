"""Empty-tile scan gate mechanisms (packed multi-tile scan pass).

A gated Automatic run first sends ONE low-resolution "scan" image per group of
neighboring tiles (each tile downsampled into one quadrant of a single
tile-sized canvas) and only runs the full per-tile detection on the tiles whose
quadrant showed any instance evidence. Empty tiles are settled from the scan
alone, which cuts request count and wall time on sparse scenes.

This module holds the pure mechanisms only: grid grouping, per-quadrant mask
evidence and the keep/skip classification. All tuned dials (which shape
classes, the score threshold, the resolution cap) are server policy, read via
``detection_policy.gate_*``; without that policy the gate is OFF and the run
behaves exactly as today. Pure Python (no Qt, no numpy) so the worker thread
can use it without heavy imports and plain pytest covers it anywhere.
"""
from __future__ import annotations

from .tile_manager import TILE_SIZE

# Reference canvas side (px) the min-pixel dial is expressed against. The scan
# canvas is the run's tile size, so it is that one constant rather than a
# second copy of the number; a server that returns a reduced mask grid scales
# the dial (see effective_min_px).
SCAN_REF_SIDE = TILE_SIZE


def scan_group(
    base: int, max_group: int, cap: float | None, mupp: float | None
) -> int:
    """Effective scan block side for one run; 0 = stand the gate down.

    With a per-class resolution cap and a known run ground resolution the
    packing ADAPTS: the largest block side whose packed scan resolution
    (side times the run resolution, since each tile shrinks into one cell of
    a tile-sized canvas) still respects the cap. Floored at 2 (a 1x1 scan IS
    a detect, it cannot save anything) and ceilinged by the policy
    (``max(base, max_group)``, so a policy that only raises the base group
    keeps its depth). Returns 0 when even the smallest packing would break
    the cap, or when a cap is set but the run resolution is unknown: doubt
    stands the gate down, it never packs blind. Without a cap the policy
    ceiling applies unchanged (the server declared no resolution guard for
    the class).
    """
    ceiling = max(2, base, max_group)
    if cap is None or cap <= 0:
        return ceiling
    if mupp is None or mupp <= 0:
        return 0
    # The tiny epsilon absorbs float-division noise when the cap is an exact
    # multiple of the run resolution, so the boundary side stays reachable.
    candidate = int(cap / mupp + 1e-9)
    if candidate < 2:
        return 0
    return min(candidate, ceiling)


def lattice_indexes(values: list[float], tol: float) -> dict[float, int]:
    """Map each value to its index on the sorted unique lattice.

    Values within ``tol`` of an existing lattice point share its index, so the
    slightly irregular spacing of an edge-aligned grid (the last row/column
    snaps flush to the zone edge) still lands on clean row/column indexes.
    """
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
    """Group grid tiles into ``group`` x ``group`` blocks of neighbors.

    ``tiles`` are the run's (x, y, w, h) pixel rects (row-major, possibly with
    polygon-culled holes). Returns blocks as lists of (tile_index, qr, qc)
    with qr/qc the tile's quadrant row/column inside its block. Blocks follow
    the grid: a partially filled block (culled cells, odd edges) simply has
    fewer members.
    """
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
    """Scale the quadrant-hit pixel floor to the returned mask grid.

    ``min_px`` is defined at the reference scan canvas (SCAN_REF_SIDE); a
    server may return the mask grid at a reduced resolution, in which case the
    same ground evidence covers proportionally fewer pixels.
    """
    scale = (width * height) / float(SCAN_REF_SIDE * SCAN_REF_SIDE)
    return max(1, int(round(min_px * scale)))


def rle_quadrant_pixels(
    rle_str: str, width: int, height: int, group: int
) -> dict[tuple[int, int], int]:
    """Per-quadrant pixel counts of one RLE mask, without materializing it.

    The RLE is the service wire format: space-separated "offset count" pairs,
    offsets ONE-BASED, row-major at ``width`` x ``height``. Each run is split
    at row ends and at vertical quadrant seams, so every counted span lies in
    exactly one quadrant.
    """
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
            # A span may straddle the vertical quadrant seam: cut it there.
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
    """Best instance evidence per quadrant of one packed scan response.

    ``masks`` is (rle, score) per detection. A detection counts for a quadrant
    when at least ``effective_min_px`` of its mask pixels fall there; the
    quadrant's signal is the max score among its detections. Quadrants with no
    evidence are absent from the result (signal 0).
    """
    floor = effective_min_px(min_px, width, height)
    best: dict[tuple[int, int], float] = {}
    for rle_str, score in masks:
        for key, n in rle_quadrant_pixels(rle_str, width, height, group).items():
            if n >= floor and score > best.get(key, 0.0):
                best[key] = float(score)
    return best


def scan_response_masks(response: dict) -> list[tuple[str, float]]:
    """Extract (rle, score) pairs from a completed detection response.

    Handles both response shapes: the plugin path's ``masks`` list of
    ``{"rle": ..., "score": ...}`` dicts and the raw ``rle`` + ``scores``
    parallel lists. Malformed entries are dropped (a dropped entry can only
    KEEP a tile, never skip one, so decoding stays fail-open)."""
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
    block: list[tuple[int, int, int]],
    response: dict,
    group: int,
    min_score: float,
    min_px: int,
) -> tuple[set[int], set[int]]:
    """Split one scanned block's tiles into (skip, keep) index sets.

    A tile is skipped when its quadrant's best instance evidence stays below
    ``min_score``; anything else (evidence, or any decode doubt) keeps the
    tile for the full detection pass."""
    try:
        width = int(response.get("width") or 0)
        height = int(response.get("height") or 0)
        best = quadrant_scores(
            scan_response_masks(response), width, height, group, min_px)
    except Exception:  # noqa: BLE001 - decode doubt keeps every tile
        best = None
    skip: set[int] = set()
    keep: set[int] = set()
    for idx, qr, qc in block:
        if best is not None and best.get((qr, qc), 0.0) < min_score:
            skip.add(idx)
        else:
            keep.add(idx)
    return skip, keep
