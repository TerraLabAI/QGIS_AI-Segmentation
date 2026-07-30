"""Which mask crops of one tile can be polygonized in a single call.

rasterio's ``shapes()`` carries a fixed GDAL setup cost of a few milliseconds
per call whatever it is handed, and the detection path calls it once per
returned mask: a dense tile pays it well over a hundred times. Crops whose
bounding boxes are disjoint cannot share a pixel, so they can be painted into
ONE label raster and polygonized together, and a label's polygon is its own
cells whether or not another label sits beside it.

The packing decision is plain integer work, so it lives here with no numpy and
no Qt import: it is the safety property the batching rests on, and it is
testable without a running QGIS. The painting and the polygonize itself stay in
``polygon_exporter``.
"""
from __future__ import annotations

#: Cap on a pack's raster side, in pixels of the tile grid. Batching every mask
#: of a tile into one raster would trade the per-call cost for a full-tile scan
#: and gain nothing, so a pack closes once its union box would grow past this.
#: Shipped value, and the fallback :func:`pack_max_side` returns.
PACK_MAX_SIDE = 320

#: Row band used to visit crops in roughly row-major order. Members of a pack
#: then sit near each other, which is what keeps its union box small.
_ROW_BAND = 64


def pack_max_side() -> int:
    """Cap on a pack's raster side, in pixels of the tile grid.

    Bounded on both sides: below the floor a pack holds barely one crop and the
    batching buys nothing, above the ceiling one pack scans more background
    than the calls it saves are worth. Cache-only and never raises, so it is
    safe to read from the detection worker's thread.
    """
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("detection_policy.exemplar.pack_max_side",
                                 PACK_MAX_SIDE, 64, 1024))
    except Exception:  # noqa: BLE001 -- packing is an optimisation  # nosec B110
        return PACK_MAX_SIDE


def pack_disjoint_crops(boxes: list, max_side: int | None = None) -> list:
    """Group crop bounding boxes into packs that can share one label raster.

    ``boxes`` are ``(row0, row1, col0, col1)``, inclusive, in the tile's pixel
    grid. Two crops in the same pack are guaranteed not to overlap, so painting
    them into one raster loses nothing; overlapping crops would overwrite each
    other and go to different packs.

    Returns a list of ``(indices, union_box)``. Order within a pack follows the
    visit order, not the input order, so callers must key results by the label
    they painted rather than by position.

    ``max_side`` defaults to the cap in force, read here rather than at import
    time so a configuration refresh moves it.
    """
    if max_side is None:
        max_side = pack_max_side()
    packs: list = []
    order = sorted(
        range(len(boxes)),
        key=lambda i: (boxes[i][0] // _ROW_BAND, boxes[i][2]),
    )
    for i in order:
        r0, r1, c0, c1 = boxes[i]
        for pack in packs:
            br0, br1, bc0, bc1 = pack["box"]
            nr0, nr1 = min(br0, r0), max(br1, r1)
            nc0, nc1 = min(bc0, c0), max(bc1, c1)
            if (nr1 - nr0) > max_side or (nc1 - nc0) > max_side:
                continue
            if any(not (r1 < mr0 or r0 > mr1 or c1 < mc0 or c0 > mc1)
                   for mr0, mr1, mc0, mc1 in pack["members"]):
                continue
            pack["indices"].append(i)
            pack["members"].append((r0, r1, c0, c1))
            pack["box"] = (nr0, nr1, nc0, nc1)
            break
        else:
            packs.append({"indices": [i], "members": [(r0, r1, c0, c1)],
                          "box": (r0, r1, c0, c1)})
    return [(p["indices"], p["box"]) for p in packs]
