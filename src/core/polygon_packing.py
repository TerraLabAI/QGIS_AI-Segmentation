













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
            nr0 = br0 if br0 < r0 else r0  # noqa: FURB136
            nr1 = br1 if br1 > r1 else r1  # noqa: FURB136
            if (nr1 - nr0) > max_side:
                continue
            nc0 = bc0 if bc0 < c0 else c0  # noqa: FURB136
            nc1 = bc1 if bc1 > c1 else c1  # noqa: FURB136
            if (nc1 - nc0) > max_side:
                continue
            clash = False
            for mr0, mr1, mc0, mc1 in pack["members"]:
                if not (r1 < mr0 or r0 > mr1 or c1 < mc0 or c0 > mc1):
                    clash = True
                    break
            if clash:
                continue
            pack["indices"].append(i)
            pack["members"].append((r0, r1, c0, c1))
            pack["box"] = (nr0, nr1, nc0, nc1)
            break
        else:
            packs.append({"indices": [i], "members": [(r0, r1, c0, c1)],
                          "box": (r0, r1, c0, c1)})
    return [(p["indices"], p["box"]) for p in packs]
