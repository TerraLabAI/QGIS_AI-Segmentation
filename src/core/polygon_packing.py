













from __future__ import annotations





PACK_MAX_SIDE = 320



_ROW_BAND = 64


def pack_max_side() -> int:







    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("detection_policy.exemplar.pack_max_side",
                                 PACK_MAX_SIDE, 64, 1024))
    except Exception:  # noqa: BLE001  # nosec B110
        return PACK_MAX_SIDE


def pack_disjoint_crops(boxes: list, max_side: int | None = None) -> list:














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
