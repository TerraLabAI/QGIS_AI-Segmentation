

















from __future__ import annotations

from .shape_policy_dials import reach_min_steps, reach_work_budget, weld_gap_m, weld_radius_px_cap


WELD_GAP_M = 0.5
WELD_RADIUS_PX_CAP = 3





REACH_WORK_BUDGET = 40_000_000
REACH_MIN_STEPS = 32


def weld_radius_px(pixel_size_m: float) -> int:

    if not pixel_size_m or pixel_size_m <= 0:
        return 1
    radius = int(round(weld_gap_m(WELD_GAP_M) / (2.0 * pixel_size_m)))
    return max(1, min(weld_radius_px_cap(WELD_RADIUS_PX_CAP), radius))


def grow_shape_with_click(prior_mask, click_mask, pixel_size_m: float = 0.0,
                          click_answer=None):

















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



        return prior.astype(click_mask.dtype, copy=False), False
    proposed = answer if click_answer is None else click_answer.astype(bool, copy=False)
    if proposed.shape != prior.shape:
        proposed = answer
    proposed = np.logical_and(proposed, np.logical_not(prior))

    radius = weld_radius_px(pixel_size_m)



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



        return prior.astype(click_mask.dtype, copy=False), True
    return reached.astype(click_mask.dtype, copy=False), False


def _dilated(mask, radius: int):

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





    import numpy as np

    if radius <= 0:
        return mask
    pad = radius + 1
    padded = np.pad(mask, pad, mode="edge")
    grown = _dilated(padded, radius)
    shrunk = np.logical_not(_dilated(np.logical_not(grown), radius))
    return shrunk[pad:-pad, pad:-pad]


def _reached_from(seed, mask):







    import numpy as np

    inside = np.logical_and(seed, mask)
    frontier = np.logical_and(mask, np.logical_not(seed))
    if not frontier.any():
        return inside
    try:
        from scipy.ndimage import binary_propagation
        return binary_propagation(
            inside, structure=np.ones((3, 3), dtype=bool), mask=mask)
    except Exception:  # noqa: BLE001  # nosec B110
        pass

    rows = np.flatnonzero(frontier.any(axis=1))
    cols = np.flatnonzero(frontier.any(axis=0))
    r0, r1 = max(0, int(rows[0]) - 1), min(mask.shape[0], int(rows[-1]) + 2)
    c0, c1 = max(0, int(cols[0]) - 1), min(mask.shape[1], int(cols[-1]) + 2)
    window = mask[r0:r1, c0:c1]
    walked = inside[r0:r1, c0:c1].copy()
    budget = max(reach_min_steps(REACH_MIN_STEPS),
                 int(reach_work_budget(REACH_WORK_BUDGET) / max(1, walked.size)))
    for _ in range(budget):
        step = np.logical_and(_dilated_once(walked), window)
        if np.array_equal(step, walked):
            out = inside.copy()
            out[r0:r1, c0:c1] = np.logical_or(out[r0:r1, c0:c1], walked)
            return out
        walked = step
    return None
