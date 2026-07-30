"""FocalClick-style Progressive Merge: bound an interactive-segmentation click
to a LOCAL region so a later click cannot rewrite a part of the mask the user
already accepted.

Reference: Chen et al., FocalClick, CVPR 2022 (Progressive Merge). Vendored idea,
not code. The key guarantee: a single click may only change pixels inside a
bounded window around where the user clicked; everything outside that window
keeps the previous mask verbatim. This holds even when the model globally
reinterprets the mask after one extra point (the failure a plain "keep the whole
changed region" merge cannot stop, because a global collapse IS one big changed
region).

Pure NumPy, no SciPy: the connected change reachable from the click is found by
geodesic dilation of the click seed inside the (small) window, so nothing native
runs on the interactive GUI click path. Best-effort throughout: every path
returns the new mask unchanged on any failure and never raises.
"""
from __future__ import annotations

from typing import Any


def _region_reached_by_click(window: Any, seed_row: int, seed_col: int) -> Any:
    """The part of a boolean ``window`` the click can reach, by geodesic
    dilation of the seed. A seed that sits on a False pixel reaches the whole
    window, so a click is never a silent no-op; the window itself is what keeps
    it local."""
    import numpy as np
    if not window[seed_row, seed_col]:
        return window
    reach = np.zeros_like(window)
    reach[seed_row, seed_col] = True
    while True:
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= window
        if np.array_equal(grown, reach):
            return reach
        reach = grown


def subtract_click_region(
    prev_mask: Any,
    new_mask: Any,
    click_row: int,
    click_col: int,
    window_frac: float = 0.15,
    window_min_px: int = 40,
) -> Any:
    """What a TRIM click is allowed to take off the shape.

    A trim click says "not this bit". It may only REMOVE ground, and only the
    one piece the user pointed at: the removal is the part of ``prev & ~new``
    reachable from the click, inside the same bounded window as above.
    Everything else keeps ``prev`` verbatim.

    Without this the click's answer replaces the shape wholesale, and the answer
    is a fresh reading of the whole object rather than an edit of it. On a long
    shape the model returns one short section, so trimming a car off a road
    deleted the road.

    Returns ``prev_mask`` when the click reached nothing to remove, ``new_mask``
    when there is no previous shape to protect or on any error.
    """
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
        removed = prev & ~new
        r = max(int(window_min_px), int(window_frac * min(h, w)))
        r0, r1 = max(0, click_row - r), min(h, click_row + r + 1)
        c0, c1 = max(0, click_col - r), min(w, click_col + r + 1)
        sub = removed[r0:r1, c0:c1]
        if not sub.any():
            return prev.astype(out_dtype)
        reach = _region_reached_by_click(sub, click_row - r0, click_col - c0)
        merged = prev.copy()
        merged[r0:r1, c0:c1] &= ~reach
        return merged.astype(out_dtype)
    except Exception:  # noqa: BLE001 -- click path is best-effort  # nosec B110
        return new_mask


def progressive_merge_masks(
    prev_mask: Any,
    new_mask: Any,
    click_row: int,
    click_col: int,
    window_frac: float = 0.15,
    window_min_px: int = 40,
) -> Any:
    """Merge new_mask into prev_mask only within a bounded window around the
    click, keeping prev_mask everywhere else.

    The window half-size is ``max(window_min_px, window_frac * min(H, W))``
    pixels. Inside the window, only the change connected to the click (geodesic
    dilation of the click seed through the symmetric difference) is applied, so
    a second, disconnected change in the same window is left alone. Outside the
    window prev_mask is preserved exactly, so a click can never rewrite the far
    side of the shape.

    Returns new_mask unchanged when prev_mask is None, shapes differ, the click
    is out of bounds, there is no change, or on any error (fail-safe: never
    worse than a normal non-local update)."""
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
        # The local window: the ONLY region this click may change.
        r = max(int(window_min_px), int(window_frac * min(h, w)))
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
        # Preserve the input dtype so downstream code (mask_to_polygons,
        # apply_mask_refinement) sees the same array kind it does today.
        return merged.astype(np.asarray(new_mask).dtype)
    except Exception:  # noqa: BLE001 -- click path is best-effort  # nosec B110
        return new_mask
