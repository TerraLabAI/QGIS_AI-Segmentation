"""Pure logic behind the Automatic review's hand edits.

The post-run review lets the user reshape what the run found without spending
credits: merge several detections into one, or split one in two. Every edit is
recorded so the user can undo the last one or clear the round, and a rebuilt
object set keeps its colours.

This module is the testable heart of that: the journal, the colour-id remap and
the retry confirm, all pure Python. Geometry enters as plain tuples so the suite
runs anywhere, with no Qt or QGIS import:

  * a centroid is ``(x, y)`` floats,
  * a rect is ``(xmin, ymin, xmax, ymax)`` floats,
  * a saved object's geometry is opaque ``bytes`` (WKB), compared only for
    equality.

The UI mixins (built separately) convert ``QgsGeometry`` to these tuples before
calling in, and back afterwards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple

# Built with typing.Tuple, not tuple[...]: a type alias is an ASSIGNMENT, so it
# is evaluated at import time and `from __future__ import annotations` does not
# defer it. Subscripting the builtin needs Python 3.9 (PEP 585), and QGIS 3.22
# on Ubuntu 20.04 runs 3.8, where this raises TypeError and the whole plugin
# fails to load. Annotations elsewhere in this file stay modern and lazy.
Rect = Tuple[float, float, float, float]
Point = Tuple[float, float]

__all__ = [
    "JournalEntry",
    "CorrectionJournal",
    "RetryLinkState",
    "remap_object_ids",
    "unique_object_ids",
]

# Above this many grid cells, a bbox query reads the candidate pool straight
# through instead of walking the cells it covers.
_MAX_QUERY_CELLS = 4096


@dataclass
class JournalEntry:
    """A single recorded edit and its effect.

    ``kind`` is one of the hand-edit kinds from ``shape_edits`` (merge /
    split); ``fids`` are the objects the edit reshaped.
    """

    kind: str
    fids: tuple[int, ...] = ()


# ---------------------------------------------------------------------------
# Correction journal (ordered edit stack)
# ---------------------------------------------------------------------------

class CorrectionJournal:
    """Ordered stack of the edits made in the current review round.

    Undo is STRICTLY top-of-stack (the UI exposes only "Undo last"): each undo
    pops the most recent entry so its effect can be reversed. ``clear`` empties
    the round for a "Clear all". The journal holds no geometry itself; it
    records what each edit did so the caller can reverse it against the live
    object set.
    """

    def __init__(self) -> None:
        self._entries: list[JournalEntry] = []

    def push(self, entry: JournalEntry) -> None:
        """Record ``entry`` as the newest edit."""
        self._entries.append(entry)

    def undo(self) -> JournalEntry | None:
        """Pop and return the most recent entry, or ``None`` when empty."""
        if not self._entries:
            return None
        return self._entries.pop()

    def clear(self) -> list[JournalEntry]:
        """Empty the round and return every popped entry, newest first."""
        popped = list(reversed(self._entries))
        self._entries.clear()
        return popped

    @property
    def count(self) -> int:
        """Number of edits currently recorded this round.

        Drives the "N corrections this round" summary line; drops when an edit
        is undone or the round is cleared.
        """
        return len(self._entries)

    def __iter__(self):
        """Iterate the entries in insertion order (oldest first)."""
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Colour identity across an object-set rebuild
# ---------------------------------------------------------------------------

def remap_object_ids(
    old: Sequence[tuple[bytes, Point, int]],
    new: Sequence[tuple[bytes, Point, Rect]],
) -> list[int]:
    """Stable display ids for a rebuilt object set (colour identity).

    The Random review renderer hues on a per-object id; a rebuild (a grouping
    toggle, a hand edit) would renumber everything and reshuffle every colour on
    screen. ``old`` rows are ``(wkb, centroid, id)`` for the pre-rebuild
    objects, ``new`` rows are ``(wkb, centroid, bbox)`` for the rebuilt ones.
    An unchanged geometry (exact wkb match, multiset semantics) keeps its old
    id; a reshaped object inherits the id of the nearest unmatched old object
    whose centroid falls inside its bbox; anything else is genuinely new and
    gets a fresh id ABOVE every old one, so a hue is never reused by a
    different object.
    """
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
    """Grid cell for the centroid buckets: the mean bbox side of the rows that
    will query it, so a typical query touches a handful of cells. Falls back to
    1.0 for degenerate (zero-size) boxes."""
    total = 0.0
    for nidx in queries:
        _wkb, _c, (bx0, by0, bx1, by1) = new[nidx]
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
    """Give each reshaped new object the id of the nearest unclaimed old object
    whose centroid falls inside its bbox, writing into ``out``.

    A regroup changes nearly every geometry, so this runs over the whole set:
    comparing every pair would be quadratic and freeze the GUI on a dense run.
    The old centroids go into a uniform grid, and a query reads only the cells
    its bbox covers. Ties go to the lowest old index, so the result does not
    depend on the order the cells are walked.
    """
    cell = _centroid_cell_size(new, unmatched_new)
    buckets: dict[tuple[int, int], list[int]] = {}
    for oidx in remaining:
        ocx, ocy = old[oidx][1]
        try:
            key = (int(ocx // cell), int(ocy // cell))
        except (ValueError, OverflowError):
            continue  # a non-finite centroid falls inside no bbox
        buckets.setdefault(key, []).append(oidx)
    used: set[int] = set()
    for nidx in unmatched_new:
        _wkb, (ncx, ncy), (bx0, by0, bx1, by1) = new[nidx]
        try:
            gx0, gx1 = int(bx0 // cell), int(bx1 // cell)
            gy0, gy1 = int(by0 // cell), int(by1 // cell)
        except (ValueError, OverflowError):
            continue  # a non-finite bbox contains no centroid
        if (gx1 - gx0 + 1) * (gy1 - gy0 + 1) <= _MAX_QUERY_CELLS:
            candidates = (oidx
                          for gx in range(gx0, gx1 + 1)
                          for gy in range(gy0, gy1 + 1)
                          for oidx in buckets.get((gx, gy), ()))
        else:
            # A box covering a huge share of the grid: reading the pool straight
            # through beats walking millions of empty cells.
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
    """Keep usable proposed ids, minting new ones for missing or duplicate ids.

    Native QGIS operations preserve feature attributes when they can, so an
    unchanged feature normally comes back with its existing ``det_id``. A split
    can duplicate that attribute, though, and a newly drawn feature has none.
    Both must receive a new id rather than reusing a neighbour's colour.
    """
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


# ---------------------------------------------------------------------------
# Retry-link two-stage confirm
# ---------------------------------------------------------------------------

class RetryLinkState:
    """Two-stage inline confirm for "Adjust and run again" with a live journal.

    Discarding a reviewed round is destructive, so the link confirms inline
    instead of with a modal: the first activation ARMS it (the label swaps to a
    confirm prompt) and returns ``False``; the second activation CONFIRMS and
    returns ``True``, disarming. Any other interaction calls :meth:`reset`. No
    timers: the state changes only on explicit calls.
    """

    def __init__(self) -> None:
        self._armed = False

    @property
    def armed(self) -> bool:
        """True while the confirm prompt is showing."""
        return self._armed

    def activate(self) -> bool:
        """Arm on the first call (``False``), confirm on the second (``True``)."""
        if self._armed:
            self._armed = False
            return True
        self._armed = True
        return False

    def reset(self) -> None:
        """Disarm without confirming (any unrelated interaction)."""
        self._armed = False
