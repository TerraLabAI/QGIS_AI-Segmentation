"""Keeping the shape a refine click edits in one piece.

A keep click on an object that is open for editing says "this belongs to the
object too". The model answers with whatever it recognizes under the cursor, and
that answer can be a different thing standing clear of the object: click the
garden beside an open house and the answer is the garden alone. Adding it leaves
two islands under one name, which is not an edit of the house.

So the answer is kept only where it reaches the shape. A narrow gap still counts
as reaching, because a roof split by a shadow or a branch comes back as two
pieces of one object; that gap is welded, and only there, so a courtyard the
detection correctly left out is not quietly filled by every click.

The shape itself can never shrink here: every pixel it already held is in the
result. Pixels only, no Qt and no QGIS, so this is testable anywhere. numpy is
imported per call because the callers already hold it (the plugin reads it from
the isolated venv, not from QGIS's Python).
"""
from __future__ import annotations

# A gap no wider than this reads as one object interrupted, not two objects.
WELD_GAP_M = 0.5
WELD_RADIUS_PX_CAP = 3

# Reaching is decided by walking the answer pixel by pixel, so it is billed a
# budget of pixel visits rather than a step count: a wide answer gets fewer
# steps, a small one gets plenty. Over budget, the walk gives up and the caller
# keeps the plain union, which is the behaviour that shipped before this rule.
REACH_WORK_BUDGET = 40_000_000
REACH_MIN_STEPS = 32


def weld_radius_px(pixel_size_m: float) -> int:
    """Half the widest gap to weld, in pixels (closing bridges twice this)."""
    if not pixel_size_m or pixel_size_m <= 0:
        return 1
    radius = int(round(WELD_GAP_M / (2.0 * pixel_size_m)))
    return max(1, min(WELD_RADIUS_PX_CAP, radius))


def grow_shape_with_click(prior_mask, click_mask, pixel_size_m: float = 0.0,
                          click_answer=None):
    """``(grown, click_stood_clear)`` for one keep click on an open shape.

    ``prior_mask`` is the shape before this click, on the same pixel grid as
    ``click_mask``. ``click_stood_clear`` is True when everything the click
    proposed to add was dropped for not reaching the shape, which is the
    caller's cue to undo the click and say so; the shape then comes back exactly
    as it was.

    Pass ``click_answer`` when ``click_mask`` already has the shape folded into
    it, which is what the progressive-merge step upstream does: what the click
    proposed is then read off the answer alone, or every click would look like
    it landed on the shape.

    Falls back to the plain union (the shape plus the whole answer) whenever the
    two masks cannot be paired or the walk runs out of budget, so a click is
    never lost to this.
    """
    import numpy as np

    if click_mask is None:
        return click_mask, False
    if prior_mask is None:
        return click_mask, False
    prior = prior_mask.astype(bool, copy=False)
    answer = click_mask.astype(bool, copy=False)
    if prior.shape != answer.shape or not prior.any():
        return click_mask, False

    union = np.logical_or(prior, answer)
    if not answer.any():
        # Nothing to add, so the shape stands. "Nothing here" is judged on the
        # click's own answer by the caller, not by handing back an empty shape:
        # a click that added nothing must never be the thing that empties it.
        return prior.astype(click_mask.dtype, copy=False), False
    proposed = answer if click_answer is None else click_answer.astype(bool, copy=False)
    if proposed.shape != prior.shape:
        proposed = answer
    proposed = np.logical_and(proposed, np.logical_not(prior))

    radius = weld_radius_px(pixel_size_m)
    # Weld material is invented pixels, so it is allowed only around THIS
    # click's answer: elsewhere the union is left exactly as the shape and the
    # answer drew it.
    candidate = union
    if radius > 0:
        welded = np.logical_and(_closed(union, radius), np.logical_not(union))
        welded = np.logical_and(welded, _dilated(answer, radius + 1))
        if welded.any():
            candidate = np.logical_or(union, welded)

    reached = _reached_from(prior, candidate)
    if reached is None:
        return union.astype(click_mask.dtype, copy=False), False

    if proposed.any() and not np.logical_and(proposed, reached).any():
        # What the click proposed is a different thing standing clear of the
        # shape. The shape comes back untouched, not merely un-grown: no weld,
        # no filled notch.
        return prior.astype(click_mask.dtype, copy=False), True
    return reached.astype(click_mask.dtype, copy=False), False


def _dilated(mask, radius: int):
    """``mask`` grown by ``radius`` steps of a 3x3 square."""
    out = mask
    for _ in range(max(0, radius)):
        out = _dilated_once(out)
    return out


def _dilated_once(mask):
    import numpy as np

    rows = mask.copy()
    rows[1:, :] = np.logical_or(rows[1:, :], mask[:-1, :])
    rows[:-1, :] = np.logical_or(rows[:-1, :], mask[1:, :])
    out = rows.copy()
    out[:, 1:] = np.logical_or(out[:, 1:], rows[:, :-1])
    out[:, :-1] = np.logical_or(out[:, :-1], rows[:, 1:])
    return out


def _closed(mask, radius: int):
    """Dilate then erode: bridges gaps up to ``2 * radius`` px wide.

    Padded with the edge value so a shape touching the crop border is not eaten
    by the erosion, which would read as the shape losing ground.
    """
    import numpy as np

    if radius <= 0:
        return mask
    pad = radius + 1
    padded = np.pad(mask, pad, mode="edge")
    grown = _dilated(padded, radius)
    shrunk = np.logical_not(_dilated(np.logical_not(grown), radius))
    return shrunk[pad:-pad, pad:-pad]


def _reached_from(seed, mask):
    """The part of ``mask`` connected to ``seed``, or None if the walk gave up.

    scipy is optional in the plugin's venv, so the numpy walk is the path most
    installs take. It runs inside the bounding box of the material that is not
    already seed: every route through the seed is free, so no route it could
    take leaves that box.
    """
    import numpy as np

    inside = np.logical_and(seed, mask)
    frontier = np.logical_and(mask, np.logical_not(seed))
    if not frontier.any():
        return inside
    try:
        from scipy.ndimage import binary_propagation
        return binary_propagation(
            inside, structure=np.ones((3, 3), dtype=bool), mask=mask)
    except Exception:  # noqa: BLE001 -- scipy is optional, the walk below is not  # nosec B110
        pass

    rows = np.flatnonzero(frontier.any(axis=1))
    cols = np.flatnonzero(frontier.any(axis=0))
    r0, r1 = max(0, int(rows[0]) - 1), min(mask.shape[0], int(rows[-1]) + 2)
    c0, c1 = max(0, int(cols[0]) - 1), min(mask.shape[1], int(cols[-1]) + 2)
    window = mask[r0:r1, c0:c1]
    walked = inside[r0:r1, c0:c1].copy()
    budget = max(REACH_MIN_STEPS, int(REACH_WORK_BUDGET / max(1, walked.size)))
    for _ in range(budget):
        step = np.logical_and(_dilated_once(walked), window)
        if np.array_equal(step, walked):
            out = inside.copy()
            out[r0:r1, c0:c1] = np.logical_or(out[r0:r1, c0:c1], walked)
            return out
        walked = step
    return None
