



















from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple






Rect = Tuple[float, float, float, float]
Point = Tuple[float, float]

__all__ = [
    "JournalEntry",
    "CorrectionJournal",
    "RetryLinkState",
    "remap_object_ids",
    "unique_object_ids",
]



_MAX_QUERY_CELLS = 4096


@dataclass
class JournalEntry:






    kind: str
    fids: tuple[int, ...] = ()






class CorrectionJournal:









    def __init__(self) -> None:
        self._entries: list[JournalEntry] = []

    def push(self, entry: JournalEntry) -> None:

        self._entries.append(entry)

    def undo(self) -> JournalEntry | None:

        if not self._entries:
            return None
        return self._entries.pop()

    def clear(self) -> list[JournalEntry]:

        popped = list(reversed(self._entries))
        self._entries.clear()
        return popped

    @property
    def count(self) -> int:





        return len(self._entries)

    def __iter__(self):

        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)






def remap_object_ids(
    old: Sequence[tuple[bytes, Point, int]],
    new: Sequence[tuple[bytes, Point, Rect]],
) -> list[int]:












    by_wkb: dict[bytes, list[int]] = {}
    for oidx, (wkb, _c, _i) in enumerate(old):
        by_wkb.setdefault(wkb, []).append(oidx)
    out: list[int | None] = [None] * len(new)
    unmatched_new: list[int] = []
    matched_old: set[int] = set()
    for nidx, (wkb, _c, _b) in enumerate(new):
        stack = by_wkb.get(wkb)
        if stack:
            oidx = stack.pop(0)
            matched_old.add(oidx)
            out[nidx] = old[oidx][2]
        else:
            unmatched_new.append(nidx)
    remaining = [i for i in range(len(old)) if i not in matched_old]
    if unmatched_new and remaining:
        _match_by_centroid(old, new, unmatched_new, remaining, out)
    next_id = max((row[2] for row in old), default=-1) + 1
    result: list[int] = []
    for val in out:
        if val is None:
            val = next_id
            next_id += 1
        result.append(val)
    return result


def _centroid_cell_size(new: Sequence[tuple[bytes, Point, Rect]],
                        queries: Sequence[int]) -> float:



    total = 0.0
    for nidx in queries:
        bx0, by0, bx1, by1 = new[nidx][2]
        total += max(bx1 - bx0, by1 - by0)
    cell = total / len(queries)
    return cell if cell > 0 else 1.0


def _match_by_centroid(
    old: Sequence[tuple[bytes, Point, int]],
    new: Sequence[tuple[bytes, Point, Rect]],
    unmatched_new: Sequence[int],
    remaining: Sequence[int],
    out: list,
) -> None:









    cell = _centroid_cell_size(new, unmatched_new)
    buckets: dict[tuple[int, int], list[int]] = {}
    for oidx in remaining:
        ocx, ocy = old[oidx][1]
        try:
            key = (int(ocx // cell), int(ocy // cell))
        except (ValueError, OverflowError):
            continue
        buckets.setdefault(key, []).append(oidx)
    used: set[int] = set()
    for nidx in unmatched_new:
        entry = new[nidx]
        ncx, ncy = entry[1]
        bx0, by0, bx1, by1 = entry[2]
        try:
            gx0, gx1 = int(bx0 // cell), int(bx1 // cell)
            gy0, gy1 = int(by0 // cell), int(by1 // cell)
        except (ValueError, OverflowError):
            continue
        if (gx1 - gx0 + 1) * (gy1 - gy0 + 1) <= _MAX_QUERY_CELLS:
            candidates = (oidx
                          for gx in range(gx0, gx1 + 1)
                          for gy in range(gy0, gy1 + 1)
                          for oidx in buckets.get((gx, gy), ()))
        else:


            candidates = iter(remaining)
        best = None
        best_d = None
        for oidx in candidates:
            if oidx in used:
                continue
            ocx, ocy = old[oidx][1]
            if not (bx0 <= ocx <= bx1 and by0 <= ocy <= by1):
                continue
            d = (ocx - ncx) ** 2 + (ocy - ncy) ** 2
            if best_d is None or d < best_d or (d == best_d and oidx < best):
                best, best_d = oidx, d
        if best is not None:
            used.add(best)
            out[nidx] = old[best][2]


def unique_object_ids(proposed: Sequence[object], prior: Sequence[object] = ()) -> list[int]:







    valid_prior = [int(value) for value in prior
                   if isinstance(value, int) and value >= 0]
    valid_proposed = [int(value) for value in proposed
                      if isinstance(value, int) and value >= 0]
    next_id = max(valid_prior + valid_proposed, default=-1) + 1
    used: set[int] = set()
    result: list[int] = []
    for value in proposed:
        candidate = value if isinstance(value, int) and value >= 0 else None
        if candidate is None or candidate in used:
            while next_id in used:
                next_id += 1
            candidate = next_id
            next_id += 1
        used.add(candidate)
        result.append(candidate)
    return result






class RetryLinkState:









    def __init__(self) -> None:
        self._armed = False

    @property
    def armed(self) -> bool:

        return self._armed

    def activate(self) -> bool:

        if self._armed:
            self._armed = False
            return True
        self._armed = True
        return False

    def reset(self) -> None:

        self._armed = False
