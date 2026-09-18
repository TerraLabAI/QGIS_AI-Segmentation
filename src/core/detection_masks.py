








from __future__ import annotations

import logging
import math
import operator

try:
    from .venv_manager import ensure_venv_packages_available
except ImportError:
    pass
else:
    ensure_venv_packages_available()

import numpy as np  # noqa: E402

from .review_defaults import HOLE_NOISE_CEILING_M  # noqa: E402

__all__ = [
    "_MASK_CELL_SLACK",
    "_PINHOLE_GROUND_M",
    "_RLE_FORMAT",
    "_TILE_SIMPLIFY_MULT",
    "_apply_rle_pairs_slow",
    "_iter_mask_entries",
    "_iter_masks",
    "_opt_float",
    "_response_mask_list",
    "_rle_pairs",
    "_vector_step",
    "decode_detection_response",
    "decode_rle_to_mask",
    "detection_mask_count",
    "iter_detection_masks",
    "logger",
    "mask_cell_size",
    "mask_scale_field",
    "parse_semantic_fields",
    "pinhole_fill_limit_px",
    "should_request_semantic",
    "should_rescue_with_semantic",
    "tile_simplify_tolerance",
]









_RLE_FORMAT: str = "offset_count_row_major_one_based"

logger = logging.getLogger(__name__)


def _mask_dimensions(height, width) -> tuple[int, int]:

    try:
        if isinstance(height, bool) or isinstance(width, bool):
            raise ValueError("mask dimensions must be positive integers")
        height, width = operator.index(height), operator.index(width)
    except TypeError as exc:
        raise ValueError("mask dimensions must be positive integers") from exc
    if height <= 0 or width <= 0 or height > np.iinfo(np.intp).max // width:
        raise ValueError("mask dimensions must be positive and addressable")
    return height, width


def _mask_box(entry: dict) -> list[float]:

    raw_box = entry.get("box")
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        try:
            box = [float(v) for v in raw_box]
        except (ValueError, TypeError, OverflowError):
            pass
        else:
            if all(math.isfinite(v) for v in box):
                return box
    return [0.0, 0.0, 0.0, 0.0]


def decode_rle_to_mask(rle: str | dict, height: int, width: int,
                       strict: bool = False) -> np.ndarray:





























    height, width = _mask_dimensions(height, width)
    flat = np.zeros(height * width, dtype=bool)

    if isinstance(rle, dict):
        if strict:
            raise ValueError("mask encoding is not readable")
        logger.warning(
            "decode_rle_to_mask: received dict RLE (unsupported format); "
            "returning empty mask for tile %dx%d",
            width,
            height,
        )
        return flat.reshape((height, width))

    if not isinstance(rle, str):
        if strict:
            raise ValueError("mask encoding is not readable")
        return flat.reshape((height, width))

    if not rle.strip():


        return flat.reshape((height, width))

    tokens = rle.split()
    total = height * width

    if strict and len(tokens) % 2:
        raise ValueError("mask encoding ends on a run with no count")

    pairs = _rle_pairs(tokens)
    if pairs is None:
        if strict:
            raise ValueError("mask encoding carries a token that is not a number")


        _apply_rle_pairs_slow(flat, tokens, total)
    else:
        offsets, counts = pairs
        idx = offsets - 1
        bad = (idx < 0) | (counts <= 0)
        if bad.any():
            if strict:
                raise ValueError("mask encoding carries a malformed run")
            logger.warning(
                "decode_rle_to_mask: %d malformed run pair(s) skipped "
                "(offset below 1 or non-positive count)",
                int(bad.sum()),
            )
        ends = idx + counts
        over = ends > total
        if over.any():
            if strict:
                raise ValueError("mask encoding carries a run past the end of the mask")
            logger.warning(
                "decode_rle_to_mask: %d run(s) exceed mask size %d; clipping",
                int(over.sum()), total,
            )
            ends = np.minimum(ends, total)
        keep = ~bad
        for start, end in zip(idx[keep].tolist(), ends[keep].tolist()):
            flat[start:end] = True

    return flat.reshape((height, width))


def _rle_pairs(tokens: list[str]):



    n = (len(tokens) // 2) * 2
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    try:
        nums = np.asarray(tokens[:n], dtype=np.int64)
    except (ValueError, TypeError, OverflowError):
        return None
    safe_limit = np.iinfo(np.int64).max // 2
    if ((nums < -safe_limit) | (nums > safe_limit)).any():
        return None
    return nums[0::2], nums[1::2]


def _apply_rle_pairs_slow(flat: np.ndarray, tokens: list[str], total: int) -> None:


    for i in range(0, len(tokens) - 1, 2):
        try:
            offset = int(tokens[i])
            count = int(tokens[i + 1])
        except ValueError:
            logger.warning(
                "decode_rle_to_mask: invalid token pair at index %d; skipping", i
            )
            continue


        idx = offset - 1
        if idx < 0:
            logger.warning(
                "decode_rle_to_mask: offset %d is below 1; skipping pair", offset
            )
            continue

        if count <= 0:


            logger.warning(
                "decode_rle_to_mask: non-positive count %d at offset %d; skipping",
                count, offset,
            )
            continue

        end = idx + count
        if end > total:
            logger.warning(
                "decode_rle_to_mask: run [%d, %d) exceeds mask size %d; clipping",
                idx,
                end,
                total,
            )
            end = total

        flat[idx:end] = True















_MASK_CELL_SLACK: float = 1.000001


def mask_cell_size(
    ground_w: float, ground_h: float, mask_w: int, mask_h: int
) -> float:







    if (mask_w <= 0 or mask_h <= 0 or ground_w <= 0 or ground_h <= 0
            or not all(math.isfinite(v) for v in (ground_w, ground_h, mask_w, mask_h))):
        return 0.0
    return max(ground_w / float(mask_w), ground_h / float(mask_h))


def _vector_step(native_gsd: float, mask_cell: float) -> float:



    if mask_cell > native_gsd * _MASK_CELL_SLACK:
        return mask_cell
    return native_gsd





_TILE_SIMPLIFY_MULT: float = 0.75





_PINHOLE_GROUND_M: float = HOLE_NOISE_CEILING_M


def tile_simplify_tolerance(
    native_gsd: float, mask_cell: float = 0.0, mult: float = 0.0
) -> float:













    if native_gsd <= 0:
        return 0.0
    if mult <= 0:
        mult = _TILE_SIMPLIFY_MULT
    return mult * _vector_step(native_gsd, mask_cell)


def pinhole_fill_limit_px(
    native_gsd: float, mask_cell: float = 0.0, ground_m: float = 0.0
) -> int:













    if native_gsd <= 0:
        return 36
    if ground_m <= 0:
        ground_m = _PINHOLE_GROUND_M
    step = _vector_step(native_gsd, mask_cell)
    return max(9, int((ground_m / step) ** 2))


def should_request_semantic(
    enabled: bool, has_prompt: bool, merge_separate: bool
) -> bool:







    return bool(enabled) and bool(has_prompt) and not bool(merge_separate)


def mask_scale_field(scale: int | None) -> int | None:









    return 2 if scale == 2 else None


def _opt_float(val: object) -> float | None:


    if isinstance(val, (int, float)) and not isinstance(val, bool):
        try:
            value = float(val)
        except OverflowError:
            return None
        return value if math.isfinite(value) else None
    return None


def parse_semantic_fields(
    response: dict,
) -> tuple[str | None, float | None, float | None]:







    rle = response.get("semantic_rle")
    if not isinstance(rle, str) or not rle.strip():
        rle = None
    return (
        rle,
        _opt_float(response.get("semantic_coverage")),
        _opt_float(response.get("presence")),
    )


def should_rescue_with_semantic(
    instance_count: int,
    coverage: float | None,
    has_rle: bool,
    enabled: bool,
    coverage_floor: float,
) -> bool:








    if not enabled or instance_count != 0 or not has_rle:
        return False
    if not isinstance(coverage, (int, float)) or isinstance(coverage, bool):
        return False
    coverage = _opt_float(coverage)
    floor = _opt_float(coverage_floor)
    return coverage is not None and floor is not None and coverage >= floor


def _response_mask_list(response: dict) -> list:

    raw_masks = response.get("masks") or []
    return raw_masks if isinstance(raw_masks, list) else []


def _iter_mask_entries(raw_masks: list, score_threshold: float):






    for entry in raw_masks:
        if not isinstance(entry, dict):
            continue
        raw_score = entry.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("mask confidence is not a finite number") from exc
        if isinstance(raw_score, bool) or not math.isfinite(score):
            raise ValueError("mask confidence is not a finite number")
        if score < score_threshold:
            continue
        yield entry, score


def detection_mask_count(response: dict, score_threshold: float = 0.0) -> int:





    return sum(1 for _ in _iter_mask_entries(
        _response_mask_list(response), score_threshold))


def _iter_masks(raw_masks: list, decode_h: int, decode_w: int,
                score_threshold: float, strict: bool = False):

    for entry, score in _iter_mask_entries(raw_masks, score_threshold):
        mask = decode_rle_to_mask(
            entry.get("rle", ""), decode_h, decode_w, strict=strict)
        yield mask, score, _mask_box(entry)


def iter_detection_masks(
    response: dict,
    tile_w: int,
    tile_h: int,
    score_threshold: float = 0.0,
    strict: bool = False,
):
























    raw_masks = response.get("masks") or []
    if not isinstance(raw_masks, list):
        logger.warning("decode_detection_response: 'masks' is not a list; returning []")
        return iter(())


    srv_w = response.get("width")
    srv_h = response.get("height")
    decode_h, decode_w = _mask_dimensions(
        srv_h if srv_h is not None else tile_h,
        srv_w if srv_w is not None else tile_w)

    return _iter_masks(raw_masks, decode_h, decode_w, score_threshold, strict)


def decode_detection_response(
    response: dict,
    tile_w: int,
    tile_h: int,
    score_threshold: float = 0.0,
) -> list[tuple[np.ndarray, float, list[float]]]:
















    return list(iter_detection_masks(response, tile_w, tile_h, score_threshold))
