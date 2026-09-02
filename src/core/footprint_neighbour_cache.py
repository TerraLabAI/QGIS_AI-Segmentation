"""Session cache of the save-time neighbourhood of core.footprint_alignment.

Every Semi-Auto save aligns the new footprint against the shapes already saved
in the session, and preparing one of those neighbours means repairing it,
simplifying it and histogramming its edges. The neighbours barely change from
one save to the next, so the same work runs again on the GUI thread at every
save. This module keeps the three values the consensus actually reads (the
dominant angle, the centre and the perimeter) so a neighbour is prepared once.

A cached entry is only ever served back for the exact same geometry, prepared
with the exact same dials, in a comparable metre frame: the key carries the
geometry's WKB digest, the compiled parameters and the frame, so nothing from
one run can reach another. The centre is stored in RUN-CRS units and scaled by
the caller's own frame factors on the way out, because each save re-measures
those factors at its own centroid and two saves never produce the same pair.

The cache holds no geometry, only numbers, and it is bounded: the oldest entry
goes when it is full.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

# How many prepared neighbours the session keeps. A save reads at most a few
# dozen of them, and an entry is five numbers.
_MAX_ENTRIES = 512
_DIGEST_BYTES = 16


class FootprintNeighbourCache:
    """Bounded first-in-first-out store of prepared neighbour values, keyed by
    the caller. Values are plain tuples of numbers."""

    def __init__(self, max_entries: int = _MAX_ENTRIES) -> None:
        self._entries: OrderedDict = OrderedDict()
        self._max_entries = max(1, int(max_entries))

    def get(self, key: tuple) -> tuple | None:
        return self._entries.get(key)

    def put(self, key: tuple, value: tuple) -> None:
        self._entries[key] = value
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


_CACHE = FootprintNeighbourCache()


def clear_footprint_neighbour_cache() -> None:
    """Drop everything the session has cached. Nothing else has to be done to
    end a session: an entry can only be read back by an identical geometry in
    an identical frame."""
    _CACHE.clear()


def _frame_key(frame: tuple[float, float]) -> tuple | None:
    """The metre frame reduced to what the cached values depend on: its overall
    size and its aspect ratio, both coarse. A save re-measures the frame at its
    own centroid, so the last digits differ between two saves of one session
    while the angle and the perimeter they feed do not: a frame this close
    moves an edge azimuth by far less than one histogram bin. None when the
    frame is unusable."""
    kx, ky = frame
    if not kx or not ky:
        return None
    try:
        return float(f"{kx:.3g}"), float(f"{ky / kx:.4g}")
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _geometry_key(geom: Any, params: Any,
                  frame: tuple[float, float]) -> tuple | None:
    """The cache key of one neighbour, or None when it cannot be built (the
    caller then prepares the neighbour without caching it)."""
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
    """The prepared angle, centre and perimeter of one neighbour row of
    ``sweep``, from the session cache when that geometry was already prepared.

    On a miss the sweep prepares the row itself and the result is cached. The
    returned dict carries only the fields the consensus stage reads, so a
    neighbour costs three numbers instead of two geometries and a coordinate
    array. None when the row cannot be prepared at all.
    """
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
