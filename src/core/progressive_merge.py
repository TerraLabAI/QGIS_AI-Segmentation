
















from __future__ import annotations

from typing import Any

from .server_dials import dial_in_range



_WINDOW_FRAC = 0.25
_WINDOW_MIN_PX = 32


def _window_half_size(kind: str, h: int, w: int, window_frac, window_min_px) -> int:

    if kind == "trim":
        if window_frac is None:
            window_frac = dial_in_range(
                "tuning.click.trim_window_frac", _WINDOW_FRAC, 0.02, 0.5)
        if window_min_px is None:
            window_min_px = dial_in_range(
                "tuning.click.trim_window_min_px", _WINDOW_MIN_PX, 4, 400)
    else:
        if window_frac is None:
            window_frac = dial_in_range(
                "tuning.click.merge_window_frac", _WINDOW_FRAC, 0.02, 0.5)
        if window_min_px is None:
            window_min_px = dial_in_range(
                "tuning.click.merge_window_min_px", _WINDOW_MIN_PX, 4, 400)
    return max(int(window_min_px), int(window_frac * min(h, w)))


def _row_spans(window: Any) -> tuple | None:

    import numpy as np
    h, w = window.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = window
    edges = np.diff(padded, axis=1)
    starts = np.argwhere(edges == 1)
    ends = np.argwhere(edges == -1)
    if len(starts) == 0 or len(starts) != len(ends):
        return None
    return (starts[:, 0].tolist(), starts[:, 1].tolist(), ends[:, 1].tolist())


def _region_reached_by_click(window: Any, seed_row: int, seed_col: int) -> Any:









    import bisect

    import numpy as np
    if not window[seed_row, seed_col]:
        return window
    spans = _row_spans(window)
    if spans is None:
        return window
    rows, lo, hi = spans
    h = window.shape[0]
    by_row_lo: list = [[] for _ in range(h)]
    by_row_hi: list = [[] for _ in range(h)]
    by_row_idx: list = [[] for _ in range(h)]
    for i, r in enumerate(rows):
        by_row_lo[r].append(lo[i])
        by_row_hi[r].append(hi[i])
        by_row_idx[r].append(i)
    seed = -1
    at = bisect.bisect_right(by_row_lo[seed_row], seed_col) - 1
    if at >= 0 and by_row_hi[seed_row][at] > seed_col:
        seed = by_row_idx[seed_row][at]
    if seed < 0:
        return window
    seen = bytearray(len(rows))
    seen[seed] = 1
    stack = [seed]





    visits_left = h * ((window.shape[1] + 1) // 2) + 1
    while stack and visits_left > 0:
        visits_left -= 1
        i = stack.pop()
        r = rows[i]
        for near in (r - 1, r + 1):
            if not 0 <= near < h:
                continue


            k = bisect.bisect_right(by_row_hi[near], lo[i])
            row_lo = by_row_lo[near]
            row_idx = by_row_idx[near]
            while k < len(row_idx) and row_lo[k] < hi[i]:
                j = row_idx[k]
                if not seen[j]:
                    seen[j] = 1
                    stack.append(j)
                k += 1
    reach = np.zeros_like(window)
    for i in range(len(rows)):
        if seen[i]:
            reach[rows[i], lo[i]:hi[i]] = True
    return reach


def subtract_click_region(
    prev_mask: Any,
    new_mask: Any,
    click_row: int,
    click_col: int,
    window_frac: float | None = None,
    window_min_px: int | None = None,
    keep_points: Any = None,
) -> Any:





















    if prev_mask is None or new_mask is None:
        return new_mask
    try:
        import numpy as np
        out_dtype = np.asarray(new_mask).dtype
        prev = np.asarray(prev_mask) > 0
        new = np.asarray(new_mask) > 0
        if prev.shape != new.shape or prev.ndim != 2:
            return new_mask
        h, w = new.shape
        if not (0 <= click_row < h and 0 <= click_col < w):
            return new_mask
        r = _window_half_size("trim", h, w, window_frac, window_min_px)
        r0, r1 = max(0, click_row - r), min(h, click_row + r + 1)
        c0, c1 = max(0, click_col - r), min(w, click_col + r + 1)
        sub = prev[r0:r1, c0:c1] & ~new[r0:r1, c0:c1]
        if not sub.any():
            return prev.astype(out_dtype)
        reach = _region_reached_by_click(sub, click_row - r0, click_col - c0)
        reach = _spare_keep_side(reach, prev, r0, c0, click_row, click_col,
                                 keep_points)
        merged = prev.copy()
        merged[r0:r1, c0:c1] &= ~reach
        return merged.astype(out_dtype)
    except Exception:  # noqa: BLE001  # nosec B110
        return new_mask


def _trim_collapse_share() -> float:


    return dial_in_range("tuning.click.trim_collapse_share", 0.0, 0.0, 0.9)


def _keep_click_side(shape: tuple, click_row: int, click_col: int,
                     keep_points: Any) -> Any:

    import numpy as np
    h, w = shape
    rows = np.arange(h, dtype=np.float64)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]
    to_click = (rows - click_row) ** 2 + (cols - click_col) ** 2
    to_keep = None
    for row, col in keep_points:
        d = (rows - int(row)) ** 2 + (cols - int(col)) ** 2
        to_keep = d if to_keep is None else np.minimum(to_keep, d)
    return to_keep <= to_click


def _spare_keep_side(reach: Any, prev: Any, r0: int, c0: int,
                     click_row: int, click_col: int, keep_points: Any) -> Any:









    if not keep_points:
        return reach
    h, w = reach.shape
    swallows_keep = any(
        r0 <= int(row) < r0 + h and c0 <= int(col) < c0 + w
        and reach[int(row) - r0, int(col) - c0] for row, col in keep_points)
    side = _keep_click_side(prev.shape, click_row, click_col, keep_points) & prev
    win_side = side[r0:r0 + h, c0:c0 + w]
    side_px = int(side.sum())
    left_px = side_px - int((reach & win_side).sum())
    if not swallows_keep and left_px >= _trim_collapse_share() * side_px:
        return reach
    return reach & ~win_side


def trim_answer_collapsed(prev_mask: Any, new_mask: Any,
                          keep_points: Any, click_rc: Any = None) -> bool:






    try:
        import numpy as np
        prev = np.asarray(prev_mask) > 0
        new = np.asarray(new_mask) > 0
        if prev.shape != new.shape or prev.ndim != 2:
            return False
        shape_px = int(prev.sum())
        if shape_px <= 0:
            return False
        h, w = new.shape
        for row, col in keep_points or ():
            if 0 <= int(row) < h and 0 <= int(col) < w and prev[int(row), int(col)] \
                    and not new[int(row), int(col)]:
                return True
        share = _trim_collapse_share()
        if click_rc is not None and keep_points:
            side = _keep_click_side(prev.shape, int(click_rc[0]), int(click_rc[1]),
                                    keep_points) & prev
            side_px = int(side.sum())
            if side_px > 0:
                return int((side & new).sum()) < share * side_px
        return int((prev & new).sum()) < share * shape_px
    except Exception:  # noqa: BLE001  # nosec B110
        return False


def pick_trim_candidate(prev_mask: Any, candidates: Any, click_row: int,
                        click_col: int, keep_points: Any) -> int | None:



    try:
        import numpy as np
        prev = np.asarray(prev_mask) > 0
        best, best_kept = None, -1
        for i, cand in enumerate(candidates):
            m = np.asarray(cand) > 0
            if m.shape != prev.shape:
                continue
            h, w = m.shape
            if 0 <= click_row < h and 0 <= click_col < w and m[click_row, click_col]:
                continue
            if any(0 <= int(r) < h and 0 <= int(c) < w and not m[int(r), int(c)]
                   for r, c in keep_points or ()):
                continue
            kept = int((prev & m).sum())
            if kept > best_kept:
                best, best_kept = i, kept
        return best
    except Exception:  # noqa: BLE001  # nosec B110
        return None


def progressive_merge_masks(
    prev_mask: Any,
    new_mask: Any,
    click_row: int,
    click_col: int,
    window_frac: float | None = None,
    window_min_px: int | None = None,
) -> Any:













    if prev_mask is None or new_mask is None:
        return new_mask
    try:
        import numpy as np
        prev = np.asarray(prev_mask) > 0
        new = np.asarray(new_mask) > 0
        if prev.shape != new.shape or prev.ndim != 2:
            return new_mask
        h, w = new.shape
        if not (0 <= click_row < h and 0 <= click_col < w):
            return new_mask
        diff = prev ^ new
        if not diff.any():
            return new_mask

        r = _window_half_size("merge", h, w, window_frac, window_min_px)
        r0, r1 = max(0, click_row - r), min(h, click_row + r + 1)
        c0, c1 = max(0, click_col - r), min(w, click_col + r + 1)
        sub = diff[r0:r1, c0:c1]
        sr, sc = click_row - r0, click_col - c0
        reach = _region_reached_by_click(sub, sr, sc)
        merged = prev.copy()
        win_merged = merged[r0:r1, c0:c1]
        win_new = new[r0:r1, c0:c1]
        win_merged[reach] = win_new[reach]
        merged[r0:r1, c0:c1] = win_merged


        return merged.astype(np.asarray(new_mask).dtype)
    except Exception:  # noqa: BLE001  # nosec B110
        return new_mask
