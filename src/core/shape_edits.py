




















from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass









KIND_MERGE = "merge"
KIND_SPLIT = "split"
KIND_REMOVE = "remove"
KIND_REFINE = "refine"


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







    def __init__(self) -> None:
        self._picks: list[int] = []

    def toggle(self, index: int) -> bool:

        index = int(index)
        if index in self._picks:
            self._picks.remove(index)
            return False
        self._picks.append(index)
        return True

    def discard(self, indices: Iterable[int]) -> None:

        drop = {int(i) for i in indices}
        self._picks = [i for i in self._picks if i not in drop]

    def clear(self) -> None:
        self._picks = []

    @property
    def indices(self) -> tuple[int, ...]:

        return tuple(self._picks)

    def __len__(self) -> int:
        return len(self._picks)

    def __contains__(self, index: object) -> bool:
        return index in self._picks

    def __iter__(self):
        return iter(self._picks)


@dataclass(frozen=True)
class MergePlan:







    target: int
    absorbed: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class ShapeEdit:










    kind: str
    restored: tuple[tuple[int, object], ...]
    appended: int
    unremoved: tuple[int, ...]
    exempted: tuple[int, ...] = ()


def merge_plan(
    picks: Sequence[tuple[int, float]],
    removed: frozenset[int] | set[int] = frozenset(),
) -> MergePlan | None:















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











    align_ids(objects, ids)
    for _ in range(max(0, edit.appended)):
        if not objects:
            break
        objects.pop()
        ids.pop()
    for index, row in edit.restored:
        if 0 <= index < len(objects):
            objects[index] = row
    return edit.unremoved


def align_ids(objects: Sequence[object], ids: list) -> None:







    del ids[len(objects):]
    if len(ids) == len(objects):
        return
    used = set(ids)
    next_id = max(used, default=-1) + 1
    while len(ids) < len(objects):
        candidate = len(ids)
        if candidate in used:
            candidate = next_id
            next_id += 1
        ids.append(candidate)
        used.add(candidate)
