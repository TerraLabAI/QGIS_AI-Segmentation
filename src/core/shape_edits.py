"""Bookkeeping for the two hand edits of the Automatic review: merge and split.

``geometry_ops`` does the geometry. This module does the accounting around it:
which picked object keeps the merged shape, where the split pieces land in the
review's canonical object list, and how to put everything back when the user
hits Undo. It is pure Python and imports no QGIS, so it runs headless: an
"object row" is opaque here (the review passes its ``(geometry, score, area)``
tuple) and is only ever moved, never read.

Index discipline. The review addresses an object by its POSITION in the object
list, and four other structures key on that position: the removal set, the
reveal set, the stable colour ids and the refine cache. So an edit never
deletes a row. A merge overwrites the target row and leaves the absorbed rows
in place for the caller to mark removed; a split overwrites the target row and
APPENDS the extra pieces; a fold overwrites every row the user reshaped and
appends the objects they added. Undo pops the appended tail, then restores the
overwritten rows. That is exact as long as undo is strictly last-in-first-out,
which the correction journal guarantees: an older merge or remove can never
clobber a newer fold, because the fold is undone first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Journal kinds for the Correct-step edits. They share the correction journal,
# so one Undo stack covers every correction of the round. Merge joins pieces;
# Remove drops a false positive (recorded in _auto_correction_removed, no
# geometry rewrite); Split is retained only for legacy undo of pre-existing
# journals (the custom split gesture is gone, replaced by the QGIS bridge);
# Refine folds a batch of hand edits (an AI reshape session or a native-edit
# commit) into the canonical rows at once, so a single Undo restores every
# pre-fold base and pops the objects the fold appended.
KIND_MERGE = "merge"
KIND_SPLIT = "split"
KIND_REMOVE = "remove"
KIND_REFINE = "refine"

# A merge needs at least this many picked objects to mean anything.
MIN_MERGE_PICKS = 2

__all__ = [
    "KIND_MERGE",
    "KIND_REFINE",
    "KIND_REMOVE",
    "KIND_SPLIT",
    "MIN_MERGE_PICKS",
    "MergePlan",
    "PickSet",
    "ShapeEdit",
    "align_ids",
    "apply_merge",
    "apply_split",
    "merge_plan",
    "revert_shape_edit",
]


class PickSet:
    """Ordered set of picked object indices, toggled one click at a time.

    Order is first-picked-first so the status line and the merge itself always
    read the picks the way the user made them. Clicking an already-picked
    object drops it, which is the whole undo the gesture needs before commit.
    """

    def __init__(self) -> None:
        self._picks: list[int] = []

    def toggle(self, index: int) -> bool:
        """Add ``index`` if absent, drop it if present. True = now picked."""
        index = int(index)
        if index in self._picks:
            self._picks.remove(index)
            return False
        self._picks.append(index)
        return True

    def discard(self, indices: Iterable[int]) -> None:
        """Drop every index in ``indices`` (an object that stopped existing)."""
        drop = {int(i) for i in indices}
        self._picks = [i for i in self._picks if i not in drop]

    def clear(self) -> None:
        self._picks = []

    @property
    def indices(self) -> tuple[int, ...]:
        """The picks, in click order."""
        return tuple(self._picks)

    def __len__(self) -> int:
        return len(self._picks)

    def __contains__(self, index: object) -> bool:
        return index in self._picks

    def __iter__(self):
        return iter(self._picks)


@dataclass(frozen=True)
class MergePlan:
    """Which object keeps the merged shape, and which ones it swallows.

    ``target`` is the row that receives the merged geometry; ``absorbed`` are
    the rows the caller marks removed; ``score`` is the score the merged object
    carries.
    """

    target: int
    absorbed: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class ShapeEdit:
    """Everything needed to undo one merge, split or fold.

    ``restored`` are ``(index, row)`` pairs to write back, ``appended`` is how
    many rows to pop off the tail, and ``unremoved`` are the indices to drop
    from the removal set. ``exempted`` are the ``det_id`` values a fold added to
    the gate-exemption set (objects the user touched), dropped on undo so an
    object exempted twice keeps the earlier exemption. Merge and split leave it
    empty; only the batch fold uses it.
    """

    kind: str
    restored: tuple[tuple[int, object], ...]
    appended: int
    unremoved: tuple[int, ...]
    exempted: tuple[int, ...] = ()


def merge_plan(
    picks: Sequence[tuple[int, float]],
    removed: frozenset[int] | set[int] = frozenset(),
) -> MergePlan | None:
    """Turn picked ``(index, score)`` pairs into a merge plan, or None.

    None means there is nothing to merge: fewer than :data:`MIN_MERGE_PICKS`
    usable picks after dropping duplicates and anything already removed. The
    caller answers that with a status line, never an error.

    The target is the pick with the HIGHEST score, ties going to the lowest
    index, and the merged object carries that score. Taking the best score
    matters: the review filters on it, so merging a strong detection with a
    weak one must not push the result under the confidence cutoff and make the
    user's own edit vanish.

    An object the user already removed with a red box is dropped from the
    picks, so a merge can never resurrect it.
    """
    seen: set[int] = set()
    usable: list[tuple[int, float]] = []
    for index, score in picks:
        index = int(index)
        if index in seen or index in removed:
            continue
        seen.add(index)
        usable.append((index, float(score)))
    if len(usable) < MIN_MERGE_PICKS:
        return None
    target, score = min(usable, key=lambda row: (-row[1], row[0]))
    absorbed = tuple(sorted(i for i, _s in usable if i != target))
    return MergePlan(target=target, absorbed=absorbed, score=score)


def apply_merge(objects: list, plan: MergePlan, merged_row: object) -> ShapeEdit:
    """Write ``merged_row`` into the plan's target row.

    The absorbed rows stay in the list untouched (see the index discipline in
    the module docstring); the returned edit names them so the caller can mark
    them removed and put them back on undo.
    """
    restored = ((plan.target, objects[plan.target]),)
    objects[plan.target] = merged_row
    return ShapeEdit(
        kind=KIND_MERGE,
        restored=restored,
        appended=0,
        unremoved=plan.absorbed,
    )


def apply_split(
    objects: list, ids: list, target: int, rows: Sequence[object]
) -> ShapeEdit | None:
    """Put the first piece in the target row and append the rest.

    Returns None when there is nothing to apply (fewer than two pieces, or a
    target outside the list), which is the "the line cut nothing" case: the
    object is left exactly as it was.

    ``ids`` is the parallel list of stable colour ids. Each appended piece gets
    a fresh id above every existing one, so a hue is never reused by a
    different object.
    """
    if len(rows) < 2 or not 0 <= target < len(objects):
        return None
    align_ids(objects, ids)
    restored = ((target, objects[target]),)
    objects[target] = rows[0]
    next_id = max(ids, default=-1) + 1
    for offset, row in enumerate(rows[1:]):
        objects.append(row)
        ids.append(next_id + offset)
    return ShapeEdit(
        kind=KIND_SPLIT,
        restored=restored,
        appended=len(rows) - 1,
        unremoved=(),
    )


def revert_shape_edit(objects: list, ids: list, edit: ShapeEdit) -> tuple[int, ...]:
    """Undo one edit and return the indices to drop from the removal set.

    Pops the appended tail first, then restores the overwritten rows, so a
    stack of edits reverses cleanly newest first. Out-of-range indices are
    skipped rather than raising: an Undo must never be the thing that breaks a
    review.
    """
    for _ in range(max(0, edit.appended)):
        if objects:
            objects.pop()
        if ids:
            ids.pop()
    for index, row in edit.restored:
        if 0 <= index < len(objects):
            objects[index] = row
    return edit.unremoved


def align_ids(objects: Sequence[object], ids: list) -> None:
    """Pad or trim ``ids`` in place so it runs parallel to ``objects``.

    The two lists are built together, but a failed remap can leave them out of
    step, and appending to a short id list would then shift every colour. A pad
    value is just the row's own position, which is the same fallback the review
    uses when the id list is missing.
    """
    while len(ids) < len(objects):
        ids.append(len(ids))
    del ids[len(objects):]
