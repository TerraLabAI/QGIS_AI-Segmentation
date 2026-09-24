


















from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any



_MAX_ENTRIES = 512
_DIGEST_BYTES = 16


class FootprintNeighbourCache:



    def __init__(self, max_entries: int = _MAX_ENTRIES) -> None:
        self._entries: OrderedDict = OrderedDict()
        self._max_entries = max(1, int(max_entries))

    def get(self, key: tuple) -> tuple | None:
        return self._entries.get(key)

    def put(self, key: tuple, value: tuple) -> None:
        self._entries[key] = value
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def set_max_entries(self, max_entries: int) -> None:

        self._max_entries = max(1, int(max_entries))
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


_CACHE = FootprintNeighbourCache()


def sync_neighbour_cache_limit() -> None:


    try:
        from .server_dials import dial_in_range

        _CACHE.set_max_entries(dial_in_range(
            "tuning.review.footprint_neighbour_cache_max", _MAX_ENTRIES, 64, 4096))
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _frame_key(frame: tuple[float, float]) -> tuple | None:






    kx, ky = frame
    if not kx or not ky:
        return None
    try:
        return float(f"{kx:.3g}"), float(f"{ky / kx:.4g}")
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _geometry_key(geom: Any, params: Any,
                  frame: tuple[float, float]) -> tuple | None:


    frame_key = _frame_key(frame)
    if frame_key is None:
        return None
    try:
        wkb = bytes(geom.asWkb())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    if not wkb:
        return None
    digest = hashlib.blake2b(wkb, digest_size=_DIGEST_BYTES).digest()
    return (params, frame_key, digest)


def cached_neighbour_prep(sweep: Any, index: int, params: Any,
                          frame: tuple[float, float]) -> dict | None:








    geom = sweep._rows[index][1]
    kx, ky = frame
    key = _geometry_key(geom, params, frame)
    if key is not None:
        hit = _CACHE.get(key)
        if hit is not None:
            angle, fraction, cx, cy, perimeter = hit
            return {"own": (angle, fraction),
                    "center": (cx * kx, cy * ky),
                    "perimeter": perimeter}
    sweep._prepare_one(index)
    prep = sweep._prepared[index]
    if prep is None:
        return None
    own = (float(prep["own"][0]), float(prep["own"][1]))
    center = (float(prep["center"][0]), float(prep["center"][1]))
    perimeter = float(prep["perimeter"])
    if key is not None:
        _CACHE.put(key, (own[0], own[1], center[0] / kx, center[1] / ky,
                         perimeter))
    return {"own": own, "center": center, "perimeter": perimeter}
