



















from __future__ import annotations

from typing import Any, NamedTuple

TILE_FILTERS_VERSION = 1


STEP_SATURATION = "saturation"
STEP_SEMANTIC_RESCUE = "semantic_rescue"
STEP_WHOLE_TILE = "whole_tile"
STEP_MAP_FLOOR = "map_floor"
STEP_COMPACT = "compact"
STEP_SLIVER = "sliver_px"



_ARCHIVE_STEPS = frozenset({
    STEP_SATURATION, STEP_SEMANTIC_RESCUE, STEP_WHOLE_TILE,
    STEP_MAP_FLOOR, STEP_COMPACT, STEP_SLIVER,
})


_VERDICT_MARKS = ("drop", "trimmed", "gate")

_DROP_REASONS = ("hard_cover", "tile_span", "not_compact", "map_lowscore", "sliver")


class TileFilterVerdict(NamedTuple):



    filtered: bool
    applied: frozenset

    raw_count: int | None

    saturated: bool | None
    dropped: dict
    armed: int
    kept_map: int
    map_cover_scores: tuple


_NOT_FILTERED = TileFilterVerdict(
    filtered=False, applied=frozenset(), raw_count=None, saturated=None,
    dropped={}, armed=0, kept_map=0, map_cover_scores=())


def mask_entry_dropped(entry: Any) -> bool:

    return isinstance(entry, dict) and "drop" in entry


def _plain_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    try:
        return max(0, int(value))
    except (OverflowError, ValueError):
        return 0


def _scores(values: Any) -> tuple:
    if not isinstance(values, list):
        return ()
    out = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return tuple(out)


def stored_model_mask_count(masks: list) -> int:


    return sum(
        1 for m in masks
        if isinstance(m, dict) and not m.get("trimmed") and not m.get("rescued"))


def read_tile_filters(response: Any) -> TileFilterVerdict:







    if not isinstance(response, dict):
        return _NOT_FILTERED
    record = response.get("tile_filters")
    masks = response.get("masks")
    masks = masks if isinstance(masks, list) else []
    if isinstance(record, dict):
        version = record.get("v")
        if (isinstance(version, int) and not isinstance(version, bool)
                and version >= TILE_FILTERS_VERSION):
            applied = record.get("applied")
            steps = frozenset(
                s for s in applied if isinstance(s, str)) if isinstance(applied, list) else frozenset()
            raw = record.get("raw_count")
            saturated = record.get("saturated")
            dropped_in = record.get("dropped")
            dropped = {
                reason: _plain_int(dropped_in.get(reason))
                for reason in _DROP_REASONS
            } if isinstance(dropped_in, dict) else {}
            return TileFilterVerdict(
                filtered=True,
                applied=steps,
                raw_count=(_plain_int(raw) if isinstance(raw, (int, float))
                           and not isinstance(raw, bool) else None),
                saturated=(saturated if isinstance(saturated, bool)
                           and STEP_SATURATION in steps else None),
                dropped=dropped,
                armed=_plain_int(record.get("armed")),
                kept_map=_plain_int(record.get("kept_map")),
                map_cover_scores=_scores(record.get("map_cover_scores")),
            )
    if any(isinstance(m, dict) and any(k in m for k in _VERDICT_MARKS) for m in masks):
        return TileFilterVerdict(
            filtered=True, applied=_ARCHIVE_STEPS,
            raw_count=stored_model_mask_count(masks), saturated=None,
            dropped={}, armed=0, kept_map=0, map_cover_scores=())
    return _NOT_FILTERED


def tile_saturated(verdict: TileFilterVerdict, raw_count: int, max_masks: int) -> bool:


    if verdict.saturated is not None:
        return verdict.saturated
    return max_masks > 0 and raw_count >= max_masks


def mask_spans_tile(col0: int, col1: int, row0: int, row1: int,
                    full_w: int, full_h: int) -> bool:


    return col1 - col0 + 1 >= full_w and row1 - row0 + 1 >= full_h


def fills_oriented_box(geom, min_fill: float) -> bool:


    try:
        _obb, obb_area, _angle, _w, _h = geom.orientedMinimumBoundingBox()
        if obb_area and obb_area > 0.0:
            return geom.area() / obb_area >= min_fill
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return False
